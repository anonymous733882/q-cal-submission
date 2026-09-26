"""Acquire the benchmark files expected by the main-evaluation loaders."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile
import urllib.request
import zipfile


HF_SOURCES = {
    "gqa": ("lmms-lab-encoder/GQA", ["testdev_balanced_instructions/*", "testdev_balanced_images/*"]),
    "pope": ("lmms-lab-encoder/POPE", ["data/test-*.parquet"]),
    "vqav2": ("lmms-lab-encoder/VQAv2", ["data/validation-*.parquet"]),
}

ARCHIVES = {
    "visualgenomeqa": {
        "visualgenome/question_answers.json.zip":
            "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/question_answers.json.zip",
        "visualgenome/image_data.json.zip":
            "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
    },
    "tallyqa": {
        "tallyqa/tallyqa.zip":
            "https://raw.githubusercontent.com/manoja328/TallyQA_dataset/master/tallyqa.zip",
    },
    "gqa_grounding": {
        "gqa_grounding/sceneGraphs.zip":
            "https://nlp.stanford.edu/data/gqa/sceneGraphs.zip",
        "gqa_grounding/questions1.2.zip":
            "https://nlp.stanford.edu/data/gqa/questions1.2.zip",
        "visualgenome/image_data.json.zip":
            "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
    },
}

VISUAL7W_URL = "https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip"


def _download(url: str, path: Path) -> None:
    if path.is_file() and zipfile.is_zipfile(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            request = urllib.request.Request(url, headers={"User-Agent": "Q-Cal/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, output)
        if not zipfile.is_zipfile(temporary):
            raise ValueError(f"Downloaded file is not a ZIP archive: {url}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _ensure_visual7w(data_root: Path) -> None:
    target = data_root / "visual7w/dataset_v7w_telling.json"
    if target.is_file():
        return
    archive = target.parent / "dataset_v7w_telling.zip"
    _download(VISUAL7W_URL, archive)
    with zipfile.ZipFile(archive) as source:
        members = [name for name in source.namelist() if name.endswith("dataset_v7w_telling.json")]
        if len(members) != 1:
            raise ValueError(f"Visual7W archive has no unique telling annotation: {archive}")
        target.write_bytes(source.read(members[0]))


def ensure_main_benchmarks(benchmarks: list[str], data_root: Path, hf_cache: Path) -> None:
    """Prepare annotation archives and HF parquet files; images load on demand."""
    for benchmark in dict.fromkeys(benchmarks):
        print(f"Preparing {benchmark} benchmark data", flush=True)
        if benchmark in HF_SOURCES:
            from huggingface_hub import snapshot_download

            repo, patterns = HF_SOURCES[benchmark]
            snapshot_download(repo_id=repo, repo_type="dataset", cache_dir=hf_cache,
                              allow_patterns=patterns)
        elif benchmark == "visual7w":
            _ensure_visual7w(data_root)
        elif benchmark in ARCHIVES:
            for relative, url in ARCHIVES[benchmark].items():
                _download(url, data_root / relative)
        elif benchmark == "invig":
            # The loader uses local InViG 21K files when present, then the
            # published test split with embedded images on Hugging Face.
            continue
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
