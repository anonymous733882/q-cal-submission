"""Local single-image multi-turn dataset loaders for dualsignal runners."""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from pathlib import Path
import glob
import json
import os
import pickle
import random
import tempfile
import urllib.request
import zipfile

from PIL import Image


DUALSIGNAL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = DUALSIGNAL_ROOT.parent
BEA_DIR = WORKSPACE / "qcal_support"
DATA_ROOT = Path(os.environ.get("QCAL_DATA_ROOT", str(BEA_DIR / "datasets")))
CACHE_HOME = Path(os.environ.get("DUALSIGNAL_CACHE_HOME", str(Path.home() / ".cache")))
HF_HUB_CACHE = Path(os.environ.get("TRANSFORMERS_CACHE", str(CACHE_HOME / "huggingface/hub")))
DUALSIGNAL_CACHE = Path(os.environ.get("DUALSIGNAL_PREPARED_CACHE", str(CACHE_HOME / "qcal")))


def _hf_snapshot(dataset: str, suffix: str = "") -> Path:
    for namespace in ("lmms-lab-encoder", "lmms-lab"):
        cache_base = HF_HUB_CACHE / f"datasets--{namespace}--{dataset}"
        snaps = sorted(cache_base.glob(f"snapshots/*/{suffix}" if suffix else "snapshots/*"))
        if snaps:
            return snaps[0]
    raise FileNotFoundError(f"{dataset} parquet cache not found under {HF_HUB_CACHE}")


def _rgb_from_bytes(data) -> Image.Image:
    if isinstance(data, dict) and "bytes" in data:
        data = data["bytes"]
    return Image.open(BytesIO(data)).convert("RGB")


def _sample(image: Image.Image, qas: list[tuple[str, str]], questions_per_image: int) -> dict:
    random.shuffle(qas)
    return {
        "image": image.convert("RGB"),
        "qas": [
            {"question": str(q), "answer": str(a)}
            for q, a in qas[:questions_per_image]
        ],
    }


def _sample_with_grounding(
    image: Image.Image,
    qas: list[dict],
    questions_per_image: int,
    metadata: dict | None = None,
) -> dict:
    random.shuffle(qas)
    selected = qas[:questions_per_image]
    return {
        "image": image.convert("RGB"),
        "qas": [
            {
                "question": str(qa["question"]),
                "answer": str(qa["answer"]),
                "grounding": qa.get("grounding", {}),
            }
            for qa in selected
        ],
        "metadata": metadata or {},
    }


def _box_from_gqa_object(obj: dict) -> list[float] | None:
    try:
        return [
            float(obj["x"]),
            float(obj["y"]),
            float(obj["x"]) + float(obj["w"]),
            float(obj["y"]) + float(obj["h"]),
        ]
    except Exception:
        return None


def _extract_gqa_object_ids(question: dict) -> list[str]:
    object_ids: list[str] = []
    annotations = question.get("annotations") or {}
    for key in ("question", "answer", "fullAnswer"):
        values = annotations.get(key) or {}
        if isinstance(values, dict):
            for value in values.values():
                if value is None:
                    continue
                object_ids.append(str(value))
    for step in question.get("semantic") or []:
        argument = str(step.get("argument", ""))
        start = 0
        while True:
            left = argument.find("(", start)
            right = argument.find(")", left + 1)
            if left < 0 or right < 0:
                break
            candidate = argument[left + 1:right].strip()
            if candidate.isdigit():
                object_ids.append(candidate)
            start = right + 1
    seen = set()
    unique = []
    for object_id in object_ids:
        if object_id not in seen:
            seen.add(object_id)
            unique.append(object_id)
    return unique


def load_gqa_multi_round(num_images: int, questions_per_image: int = 15) -> list[dict]:
    import pyarrow.parquet as pq

    snap = _hf_snapshot("GQA")
    q_files = sorted(glob.glob(str(snap / "testdev_balanced_instructions" / "testdev-*.parquet")))
    img_files = sorted(glob.glob(str(snap / "testdev_balanced_images" / "testdev-*.parquet")))
    if not q_files or not img_files:
        raise FileNotFoundError(f"GQA parquet files missing in {snap}")

    img_questions: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pf in q_files:
        tbl = pq.read_table(pf, columns=["imageId", "question", "answer"])
        for iid, q, a in zip(
            tbl.column("imageId").to_pylist(),
            tbl.column("question").to_pylist(),
            tbl.column("answer").to_pylist(),
        ):
            img_questions[str(iid)].append((q, a))

    eligible = [iid for iid, qas in img_questions.items() if qas]
    random.shuffle(eligible)
    if num_images > 0:
        eligible = eligible[:num_images]

    need = set(eligible)
    img_store = {}
    for pf in img_files:
        tbl = pq.read_table(pf, columns=["id", "image"])
        ids = tbl.column("id").to_pylist()
        imgs = tbl.column("image")
        for row, iid in enumerate(ids):
            iid = str(iid)
            if iid in need:
                img_store[iid] = _rgb_from_bytes(imgs[row].as_py())

    return [
        _sample(img_store[iid], img_questions[iid], questions_per_image)
        for iid in eligible
        if iid in img_store
    ]


def _load_gqa_hf_images(image_ids: list[str]) -> dict[str, Image.Image]:
    import pyarrow.parquet as pq

    snap = _hf_snapshot("GQA")
    img_files = sorted(glob.glob(str(snap / "testdev_balanced_images" / "testdev-*.parquet")))
    if not img_files:
        raise FileNotFoundError(f"GQA image parquet files missing in {snap}")
    need = set(str(x) for x in image_ids)
    img_store = {}
    for pf in img_files:
        if len(img_store) >= len(need):
            break
        tbl = pq.read_table(pf, columns=["id", "image"])
        ids = tbl.column("id").to_pylist()
        imgs = tbl.column("image")
        for row, iid in enumerate(ids):
            iid = str(iid)
            if iid in need and iid not in img_store:
                img_store[iid] = _rgb_from_bytes(imgs[row].as_py())
    return img_store


def _load_visualgenome_url_map(
    data_dir: str | os.PathLike = DATA_ROOT / "visualgenome",
) -> dict[str, str]:
    image_zip = Path(data_dir) / "image_data.json.zip"
    if not image_zip.exists():
        return {}
    with zipfile.ZipFile(image_zip) as zf:
        image_data = json.load(zf.open("image_data.json"))
    return {
        str(item["image_id"]): str(item.get("url") or "")
        for item in image_data
        if item.get("image_id") is not None
    }


def _load_gqa_grounding_images(image_ids: list[str]) -> dict[str, Image.Image]:
    img_store = _load_gqa_hf_images(image_ids)
    missing = [iid for iid in image_ids if iid not in img_store]
    if not missing:
        return img_store
    url_by_id = _load_visualgenome_url_map()
    cache_dir = DATA_ROOT / "gqa_grounding/images_cache"
    for iid in missing:
        cache_path = cache_dir / f"{iid}.jpg"
        url = url_by_id.get(str(iid))
        img = _open_or_download_image(cache_path, url)
        if img is not None:
            img_store[str(iid)] = img
    return img_store


def load_gqa_grounding_multi_round(
    num_images: int,
    questions_per_image: int = 15,
    data_dir: str | os.PathLike = DATA_ROOT / "gqa_grounding",
) -> list[dict]:
    data_dir = Path(data_dir)
    scene_zip = data_dir / "sceneGraphs.zip"
    questions_zip = data_dir / "questions1.2.zip"
    if not scene_zip.exists():
        raise FileNotFoundError(f"GQA scene graph zip not found at {scene_zip}")
    if not questions_zip.exists():
        raise FileNotFoundError(f"GQA questions zip not found at {questions_zip}")

    with zipfile.ZipFile(scene_zip) as zf:
        sg_name = "val_sceneGraphs.json" if "val_sceneGraphs.json" in zf.namelist() else zf.namelist()[0]
        scene_graphs = json.load(zf.open(sg_name))

    with zipfile.ZipFile(questions_zip) as zf:
        names = zf.namelist()
        preferred = [
            "val_balanced_questions.json",
            "testdev_balanced_questions.json",
            "val_all_questions.json",
        ]
        q_name = next((name for name in preferred if name in names), None)
        if q_name is None:
            candidates = [name for name in names if name.endswith("_questions.json")]
            if not candidates:
                raise FileNotFoundError(f"No GQA questions json found in {questions_zip}")
            q_name = sorted(candidates)[0]
        questions = json.load(zf.open(q_name))

    img_questions: dict[str, list[dict]] = defaultdict(list)
    for qid, question in questions.items():
        if not question.get("isBalanced", True):
            continue
        image_id = str(question.get("imageId", ""))
        scene_graph = scene_graphs.get(image_id)
        if not image_id or not scene_graph:
            continue
        objects = scene_graph.get("objects") or {}
        boxes = []
        object_ids = []
        for object_id in _extract_gqa_object_ids(question):
            obj = objects.get(str(object_id))
            if not obj:
                continue
            box = _box_from_gqa_object(obj)
            if box is None:
                continue
            object_ids.append(str(object_id))
            boxes.append(box)
        if not boxes:
            continue
        img_questions[image_id].append({
            "question": question.get("question", ""),
            "answer": question.get("answer", ""),
            "grounding": {
                "source": "gqa",
                "question_id": str(qid),
                "object_ids": object_ids,
                "boxes_xyxy": boxes,
                "image_size": [int(scene_graph.get("width", 0)), int(scene_graph.get("height", 0))],
                "types": question.get("types", {}),
            },
        })

    eligible = [iid for iid, qas in img_questions.items() if qas]
    random.shuffle(eligible)
    cache_dir = DATA_ROOT / "gqa_grounding/images_cache"
    cached_ids = {p.stem for p in cache_dir.glob("*.jpg") if p.stat().st_size > 0}
    cached = [iid for iid in eligible if str(iid) in cached_ids]
    uncached = [iid for iid in eligible if str(iid) not in cached_ids]
    random.shuffle(cached)
    random.shuffle(uncached)
    eligible = cached + uncached
    if num_images > 0:
        eligible = eligible[:num_images]
    img_store = _load_gqa_grounding_images(eligible)
    samples = []
    for iid in eligible:
        if iid not in img_store:
            continue
        samples.append(_sample_with_grounding(
            img_store[iid],
            img_questions[iid],
            questions_per_image,
            {"benchmark": "gqa_grounding", "image_id": iid},
        ))
    return samples


def load_pope_multi_round(num_images: int, questions_per_image: int = 15) -> list[dict]:
    import pyarrow.parquet as pq

    data_dir = _hf_snapshot("POPE", "data")
    pq_files = sorted(glob.glob(str(data_dir / "test-*.parquet")))
    if not pq_files:
        raise FileNotFoundError(f"POPE parquet files missing in {data_dir}")

    img_questions: dict[str, list[tuple[str, str]]] = defaultdict(list)
    img_row_map: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for fi, pf in enumerate(pq_files):
        tbl = pq.read_table(pf, columns=["image_source", "question", "answer"])
        for ri, (src, q, a) in enumerate(zip(
            tbl.column("image_source").to_pylist(),
            tbl.column("question").to_pylist(),
            tbl.column("answer").to_pylist(),
        )):
            src = str(src)
            img_questions[src].append((q, a))
            img_row_map[src].append((fi, ri))

    eligible = [iid for iid, qas in img_questions.items() if qas]
    random.shuffle(eligible)
    if num_images > 0:
        eligible = eligible[:num_images]

    need_from_file: dict[int, dict[int, str]] = defaultdict(dict)
    for iid in eligible:
        fi, ri = img_row_map[iid][0]
        need_from_file[fi][ri] = iid

    img_store = {}
    for fi, rows in sorted(need_from_file.items()):
        tbl = pq.read_table(pq_files[fi], columns=["image"])
        imgs = tbl.column("image")
        for ri, iid in rows.items():
            img_store[iid] = _rgb_from_bytes(imgs[ri].as_py())

    return [
        _sample(img_store[iid], img_questions[iid], questions_per_image)
        for iid in eligible
        if iid in img_store
    ]


def load_vqav2_multi_round(num_images: int, questions_per_image: int = 15) -> list[dict]:
    import pyarrow.parquet as pq

    data_dir = _hf_snapshot("VQAv2", "data")
    pq_files = sorted(glob.glob(str(data_dir / "validation-*.parquet")))
    if not pq_files:
        raise FileNotFoundError(f"VQAv2 validation parquet files missing in {data_dir}")

    snap_id = data_dir.parent.name
    cache_dir = DUALSIGNAL_CACHE / "prepared_vqav2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"validation_{snap_id}_n{num_images}_q{questions_per_image}_seed42.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        return [
            {"image": _rgb_from_bytes(img_bytes), "qas": [{"question": q, "answer": a} for q, a in qas]}
            for img_bytes, qas in cached
        ]

    img_questions = defaultdict(list)
    for fi, pf in enumerate(pq_files):
        tbl = pq.read_table(pf, columns=["image_id", "question", "multiple_choice_answer"])
        for ri in range(len(tbl)):
            iid = str(tbl["image_id"][ri].as_py())
            q = tbl["question"][ri].as_py()
            a = tbl["multiple_choice_answer"][ri].as_py() or ""
            img_questions[iid].append((q, a, fi, ri))

    eligible = [(iid, qas) for iid, qas in img_questions.items() if qas]
    random.shuffle(eligible)
    if num_images > 0:
        eligible = eligible[:num_images]

    need_from_file: dict[int, dict[int, str]] = defaultdict(dict)
    for iid, qas in eligible:
        fi, ri = qas[0][2], qas[0][3]
        need_from_file[fi][ri] = iid

    img_store = {}
    for fi, rows in sorted(need_from_file.items()):
        tbl = pq.read_table(pq_files[fi], columns=["image"])
        imgs = tbl.column("image")
        for ri, iid in rows.items():
            img_store[iid] = imgs[ri].as_py()

    samples = []
    cached = []
    for iid, qas in eligible:
        if iid not in img_store:
            continue
        img_data = img_store[iid]
        img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
        qa_pairs = [(q, a) for q, a, _, _ in qas]
        random.shuffle(qa_pairs)
        selected = qa_pairs[:questions_per_image]
        samples.append({"image": _rgb_from_bytes(img_bytes), "qas": [{"question": q, "answer": a} for q, a in selected]})
        cached.append((img_bytes, selected))

    try:
        fd, tmp = tempfile.mkstemp(prefix=cache_path.name + ".", suffix=".tmp", dir=cache_dir)
        with os.fdopen(fd, "wb") as f:
            pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)
    except Exception:
        pass
    return samples


def load_clevr_multi_round(
    num_images: int,
    questions_per_image: int = 15,
    data_dir: str | os.PathLike = DATA_ROOT / "CLEVR_v1.0",
) -> list[dict]:
    data_dir = Path(data_dir)
    q_path = data_dir / "questions/CLEVR_val_questions.json"
    if not q_path.exists():
        raise FileNotFoundError(f"CLEVR questions not found at {q_path}")
    qdata = json.loads(q_path.read_text())
    img_questions: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in qdata["questions"]:
        img_questions[item["image_filename"]].append((item["question"], str(item["answer"])))

    eligible = [(fname, qas) for fname, qas in img_questions.items() if qas]
    random.shuffle(eligible)
    if num_images > 0:
        eligible = eligible[:num_images]

    img_dir = data_dir / "images/val"
    samples = []
    for fname, qas in eligible:
        img_path = img_dir / fname
        if img_path.exists():
            samples.append(_sample(Image.open(img_path), qas, questions_per_image))
    return samples


def load_visual7w_multi_round(
    num_images: int,
    questions_per_image: int = 15,
    data_dir: str | os.PathLike = DATA_ROOT / "visual7w",
) -> list[dict]:
    data_dir = Path(data_dir)
    json_path = data_dir / "dataset_v7w_telling.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Visual7W JSON not found at {json_path}")
    v7w = json.loads(json_path.read_text())
    test_images = [img for img in v7w["images"] if img.get("split") == "test"]
    eligible = [(img["image_id"], img["qa_pairs"]) for img in test_images if img.get("qa_pairs")]
    random.shuffle(eligible)
    if num_images > 0:
        eligible = eligible[:num_images]

    cache_dir = data_dir / "images_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_ids = {p.stem for p in cache_dir.glob("*.jpg")}
    cached = [item for item in eligible if str(item[0]) in cached_ids]
    uncached = [item for item in eligible if str(item[0]) not in cached_ids]
    random.shuffle(cached)
    random.shuffle(uncached)
    eligible = cached + uncached
    bases = [
        "https://cs.stanford.edu/people/rak248/VG_100K",
        "https://cs.stanford.edu/people/rak248/VG_100K_2",
    ]
    samples = []
    for iid, qa_pairs in eligible:
        cache_path = cache_dir / f"{iid}.jpg"
        if cache_path.exists():
            img = Image.open(cache_path).convert("RGB")
        else:
            img = None
            for base in bases:
                try:
                    data = urllib.request.urlopen(f"{base}/{iid}.jpg", timeout=10).read()
                    cache_path.write_bytes(data)
                    img = Image.open(BytesIO(data)).convert("RGB")
                    break
                except Exception:
                    continue
            if img is None:
                continue
        qas = [(qa["question"], qa["answer"]) for qa in qa_pairs]
        samples.append(_sample(img, qas, questions_per_image))
    return samples


def _open_or_download_image(path: Path, url: str | None = None) -> Image.Image | None:
    if path.exists() and path.stat().st_size > 0:
        return Image.open(path).convert("RGB")
    if not url:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=20).read()
        path.write_bytes(data)
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None


def load_visualgenomeqa_multi_round(
    num_images: int,
    questions_per_image: int = 15,
    data_dir: str | os.PathLike = DATA_ROOT / "visualgenome",
) -> list[dict]:
    data_dir = Path(data_dir)
    qa_zip = data_dir / "question_answers.json.zip"
    image_zip = data_dir / "image_data.json.zip"
    if not qa_zip.exists():
        raise FileNotFoundError(f"Visual Genome QA zip not found at {qa_zip}")
    if not image_zip.exists():
        raise FileNotFoundError(f"Visual Genome image metadata zip not found at {image_zip}")

    with zipfile.ZipFile(image_zip) as zf:
        image_data = json.load(zf.open("image_data.json"))
    url_by_id = {
        int(item["image_id"]): str(item.get("url") or "")
        for item in image_data
        if item.get("image_id") is not None
    }

    with zipfile.ZipFile(qa_zip) as zf:
        qa_data = json.load(zf.open("question_answers.json"))

    eligible = []
    for item in qa_data:
        image_id = int(item.get("image_id", item.get("id")))
        qas = [
            (qa.get("question", ""), qa.get("answer", ""))
            for qa in item.get("qas", [])
            if qa.get("question") and qa.get("answer")
        ]
        if qas and url_by_id.get(image_id):
            eligible.append((image_id, qas, url_by_id[image_id]))
    random.shuffle(eligible)

    cache_dir = data_dir / "images_cache"
    samples = []
    for image_id, qas, url in eligible:
        if num_images > 0 and len(samples) >= num_images:
            break
        cache_path = cache_dir / f"{image_id}.jpg"
        img = _open_or_download_image(cache_path, url)
        if img is None:
            continue
        samples.append(_sample(img, qas, questions_per_image))
    return samples


def load_tallyqa_multi_round(
    num_images: int,
    questions_per_image: int = 15,
    data_dir: str | os.PathLike = DATA_ROOT / "tallyqa",
) -> list[dict]:
    data_dir = Path(data_dir)
    zip_path = data_dir / "tallyqa.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"TallyQA zip not found at {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        rows = json.load(zf.open("test.json"))

    img_questions: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        image_ref = str(row.get("image", ""))
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        if image_ref and question and answer:
            img_questions[image_ref].append((question, answer))

    eligible = [(image_ref, qas) for image_ref, qas in img_questions.items() if qas]
    random.shuffle(eligible)

    cache_dir = data_dir / "images_cache"
    coco_root = DATA_ROOT / "coco"
    samples = []
    for image_ref, qas in eligible:
        if num_images > 0 and len(samples) >= num_images:
            break
        image_path = Path(image_ref)
        img = None
        if image_ref.startswith("VG_100K"):
            cache_path = cache_dir / image_path.name
            url = f"https://cs.stanford.edu/people/rak248/{image_ref}"
            img = _open_or_download_image(cache_path, url)
        else:
            candidates = [
                coco_root / image_ref,
                coco_root / image_path.name,
                coco_root / "train2017" / image_path.name,
                coco_root / "val2017" / image_path.name,
                coco_root / "val2014" / image_path.name,
                coco_root / "train2014" / image_path.name,
            ]
            for candidate in candidates:
                img = _open_or_download_image(candidate)
                if img is not None:
                    break
            if img is None:
                collection = next((part for part in image_path.parts
                                   if part in ("train2014", "val2014", "train2017", "val2017")), None)
                if collection is None:
                    collection = "train2014" if "train2014" in image_path.name else "val2014"
                cache_path = coco_root / collection / image_path.name
                url = f"https://images.cocodataset.org/{collection}/{image_path.name}"
                img = _open_or_download_image(cache_path, url)
        if img is None:
            continue
        samples.append(_sample(img, qas, questions_per_image))
    return samples


def _read_invig_image(data_dir: Path, filename: str) -> Image.Image | None:
    candidates = [
        data_dir / "invig21k_imgs" / filename,
        data_dir / "images" / "invig21k_imgs" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return Image.open(candidate).convert("RGB")
    zip_path = data_dir / "invig21k_imgs.zip"
    if not zip_path.exists():
        return None
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        exact = filename if filename in names else None
        if exact is None:
            suffix = "/" + filename
            exact = next((name for name in names if name.endswith(suffix)), None)
        if exact is None:
            return None
        return Image.open(BytesIO(zf.read(exact))).convert("RGB")


def _load_invig_hf_rows(split: str) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("jxu124/invig", split="validation" if split == "valid" else split)
    rows = []
    for item in dataset:
        image_info = item["image_info"]
        width, height = int(image_info["width"]), int(image_info["height"])
        for reference_index, reference in enumerate(item["ref_list"]):
            dialog = reference["dialog"]
            if not dialog or not dialog[0][0]:
                continue
            questions = []
            answers = []
            for current, following in zip(dialog, dialog[1:]):
                if current[1] and following[0]:
                    questions.append(current[1])
                    answers.append(following[0])
            x1, y1, x2, y2 = reference["bbox"]
            rows.append({
                "id": reference.get("id", f"{image_info['id']}:{reference_index}"),
                "filename": image_info["file_name"],
                "width": width,
                "height": height,
                "label": reference["category"],
                "_image": item["image"],
                "_source": "jxu124/invig",
                "ann": {
                    "ref_exp": dialog[0][0],
                    "questions": questions,
                    "answers": answers,
                    "ref_bboxes": [[x1 * width, y1 * height, x2 * width, y2 * height]],
                },
            })
    return rows


def load_invig_multi_round(
    num_images: int,
    questions_per_image: int = 15,
    data_dir: str | os.PathLike = DATA_ROOT / "invig",
    split: str = "test",
) -> list[dict]:
    data_dir = Path(data_dir)
    ann_candidates = [
        data_dir / f"invig21k_{split}_anns.jsonl",
        data_dir / "invig21k_anns" / f"invig21k_{split}_anns.jsonl",
    ]
    ann_path = next((path for path in ann_candidates if path.exists()), None)
    if ann_path is None:
        rows = _load_invig_hf_rows(split)
    else:
        rows = []
        with ann_path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    random.shuffle(rows)

    samples = []
    for row in rows:
        if num_images > 0 and len(samples) >= num_images:
            break
        ann = row.get("ann") or {}
        img = row.get("_image")
        if img is None:
            img = _read_invig_image(data_dir, str(row.get("filename", "")))
        if img is None:
            continue
        ref_exp = str(ann.get("ref_exp", ""))
        questions = [str(q) for q in ann.get("questions", [])]
        answers = [str(a) for a in ann.get("answers", [])]
        ref_boxes = ann.get("ref_bboxes") or []
        qas = []
        if ref_exp:
            qas.append({
                "question": f"Initial instruction: {ref_exp}\nWhich object is being referred to?",
                "answer": str(row.get("label", "target object")),
                "grounding": {
                    "source": "invig",
                    "turn": "initial_ref_exp",
                    "boxes_xyxy": ref_boxes,
                    "image_size": [int(row.get("width", 0)), int(row.get("height", 0))],
                    "ref_exp": ref_exp,
                },
            })
        history = [f"Initial instruction: {ref_exp}"] if ref_exp else []
        for idx, (question, answer) in enumerate(zip(questions, answers)):
            prompt = "\n".join(history + [f"Question: {question}"])
            qas.append({
                "question": prompt,
                "answer": answer,
                "grounding": {
                    "source": "invig",
                    "turn": idx,
                    "boxes_xyxy": ref_boxes,
                    "image_size": [int(row.get("width", 0)), int(row.get("height", 0))],
                    "ref_exp": ref_exp,
                },
            })
            history.extend([f"Question: {question}", f"Answer: {answer}"])
        if not qas:
            continue
        samples.append(_sample_with_grounding(
            img,
            qas,
            questions_per_image,
            {"benchmark": "invig", "id": row.get("id"), "filename": row.get("filename"),
             "source": row.get("_source", "invig21k_jsonl")},
        ))
    return samples


def load_multiround_samples(
    benchmark: str,
    num_images: int,
    questions_per_image: int = 15,
) -> list[dict]:
    """Load samples through the same path used by stage2 dual-policy runs."""
    import sys

    strategy_dir = BEA_DIR / "strategy_test"
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))
    old_cwd = Path.cwd()
    os.chdir(BEA_DIR)
    try:
        import multi_round_benchmark as stage2_mr

        loaders = {
            "gqa": lambda: stage2_mr.load_gqa_multi_round(num_images, questions_per_image),
            "pope": lambda: stage2_mr.load_pope_multi_round(num_images, questions_per_image),
            "vqav2": lambda: stage2_mr.load_vqav2_multi_round(num_images, questions_per_image),
            "clevr": lambda: stage2_mr.load_clevr_multi_round(
                num_images,
                questions_per_image,
                data_dir=str(DATA_ROOT / "CLEVR_v1.0"),
            ),
            "visual7w": lambda: load_visual7w_multi_round(
                num_images,
                questions_per_image,
            ),
            "visualgenomeqa": lambda: load_visualgenomeqa_multi_round(
                num_images,
                questions_per_image,
            ),
            "tallyqa": lambda: load_tallyqa_multi_round(
                num_images,
                questions_per_image,
            ),
            "gqa_grounding": lambda: load_gqa_grounding_multi_round(
                num_images,
                questions_per_image,
            ),
            "invig": lambda: load_invig_multi_round(
                num_images,
                questions_per_image,
            ),
        }
        if benchmark not in loaders:
            raise KeyError(benchmark)
        random_state = random.getstate()
        random.seed(42)
        try:
            raw_samples = loaders[benchmark]()
        finally:
            random.setstate(random_state)
    finally:
        os.chdir(old_cwd)

    samples = []
    for raw in raw_samples:
        if isinstance(raw, dict):
            samples.append({
                "image": raw["image"].convert("RGB"),
                "qas": [
                    {
                        "question": str(qa["question"]),
                        "answer": str(qa["answer"]),
                        "grounding": qa.get("grounding", {}),
                    }
                    for qa in raw.get("qas", [])[:questions_per_image]
                ],
                "metadata": raw.get("metadata", {}),
            })
            continue
        image, qas = raw
        samples.append({
            "image": image.convert("RGB"),
            "qas": [
                {"question": str(q), "answer": str(a)}
                for q, a in qas[:questions_per_image]
            ],
        })
    return samples
