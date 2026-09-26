"""
BLINK Benchmark Loader
Multi-image tasks from BLINK benchmark

Stratified sampling: num_samples evenly distributed across multi-image task configs.
Images loaded from local JPEG files (pre-extracted from HuggingFace arrow).
"""

import os
import math
import random
from collections import defaultdict
from pathlib import Path
from datasets import Dataset, load_dataset
from PIL import Image

_BLINK_CACHE = os.environ.get(
    "QCAL_BLINK_CACHE",
    str(Path(os.environ.get("HF_DATASETS_CACHE", str(Path.home() / ".cache/huggingface/datasets"))) / "BLINK-Benchmark___blink"),
)
_BLINK_HASH = "a3666eb249237ba3d5eca8db21176cc47967e040"
_BLINK_IMG_DIR = str(Path(os.environ.get("QCAL_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "datasets"))) / "blink")

BLINK_MULTI_IMAGE_TASKS = [
    "Multi-view_Reasoning",
    "Visual_Correspondence",
    "Jigsaw",
]


def _load_blink_task_local(task, split="val"):
    version = Path(_BLINK_CACHE) / task / "0.0.0"
    arrow = version / _BLINK_HASH / f"blink-{split}.arrow"
    if not arrow.is_file():
        matches = sorted(version.glob(f"*/blink-{split}.arrow"))
        if not matches:
            raise FileNotFoundError(f"BLINK Arrow cache missing for {task}: {version}")
        arrow = matches[-1]
    return Dataset.from_file(str(arrow))


def load_blink(num_samples: int = 200, tasks: list = None, split: str = "val",
               seed: int = 42) -> list:
    if tasks is None:
        tasks = BLINK_MULTI_IMAGE_TASKS

    per_task = math.ceil(num_samples / len(tasks))
    rng = random.Random(seed)
    task_buckets = defaultdict(list)

    for task in tasks:
        try:
            ds = _load_blink_task_local(task, split=split)
        except FileNotFoundError:
            ds = load_dataset("BLINK-Benchmark/BLINK", task, split=split)
        try:
            img_cols = [c for c in ds.column_names if c.startswith("image_")]
            ds_text = ds.remove_columns(img_cols)
            task_dir = os.path.join(_BLINK_IMG_DIR, task)

            for idx, item in enumerate(ds_text):
                # Collect pre-extracted image paths
                img_paths = []
                for col in img_cols:
                    p = os.path.join(task_dir, f"{idx:04d}_{col}.jpg")
                    if os.path.exists(p):
                        img_paths.append(p)
                if len(img_paths) < 2:
                    Path(task_dir).mkdir(parents=True, exist_ok=True)
                    image_row = ds[idx]
                    for col in img_cols:
                        p = os.path.join(task_dir, f"{idx:04d}_{col}.jpg")
                        if not os.path.exists(p) and image_row[col] is not None:
                            image = image_row[col]
                            if isinstance(image, dict):
                                from io import BytesIO
                                image = Image.open(BytesIO(image["bytes"]))
                            image.convert("RGB").save(p, format="JPEG")
                        if os.path.exists(p) and p not in img_paths:
                            img_paths.append(p)
                if len(img_paths) < 2:
                    continue

                question = item.get("question", "")
                choices = item.get("choices", [])
                answer = item.get("answer", "")
                if choices:
                    choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
                    full_question = f"{question}\n\n{choices_text}"
                else:
                    full_question = question

                task_buckets[task].append({
                    "question": full_question,
                    "images": img_paths,
                    "answer": answer,
                    "choices": choices,
                    "task": task,
                    "source": "blink",
                })
        except Exception as e:
            raise RuntimeError(f"Could not load BLINK task '{task}'") from e

    samples = []
    for task in tasks:
        bucket = task_buckets.get(task, [])
        rng.shuffle(bucket)
        samples.extend(bucket[:per_task])
    rng.shuffle(samples)
    samples = samples[:num_samples]

    task_counts = defaultdict(int)
    for s in samples:
        task_counts[s["task"]] += 1
    print(f"BLINK: loaded {len(samples)} samples — "
          + ", ".join(f"{k}: {v}" for k, v in sorted(task_counts.items())))
    return samples


def evaluate_response(response: str, sample: dict) -> bool:
    """Check if response matches the BLINK answer.

    Handles answer formats: 'B', '(B)', 'B. text', 'text matching choice'.
    """
    raw_answer = sample.get("answer", "").strip()
    choices = sample.get("choices", [])
    response = response.strip()

    # Normalize answer: strip parentheses, e.g. '(B)' → 'B'
    answer = raw_answer.strip("()")
    # Also normalize response parentheses
    resp_clean = response.strip("()")

    # Direct letter match
    if resp_clean.upper() == answer.upper():
        return True

    # Extract leading letter from response
    resp_upper = resp_clean.upper()
    for letter in ["A", "B", "C", "D"]:
        if resp_upper.startswith(letter) and letter == answer.upper():
            return True

    # Match full answer text from choices
    if choices and answer:
        answer_upper = answer.upper()
        if answer_upper in ["A", "B", "C", "D"]:
            idx = ord(answer_upper) - ord("A")
            if idx < len(choices) and choices[idx].lower() in response.lower():
                return True

    return False


if __name__ == "__main__":
    print("Testing BLINK loader...")
    samples = load_blink(num_samples=200)
    print(f"Loaded {len(samples)} samples")
    for i, s in enumerate(samples[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Task: {s['task']}")
        print(f"  Images: {len(s['images_pil'])}")
        print(f"  Question: {s['question'][:100]}...")
        print(f"  Answer: {s['answer']}")
