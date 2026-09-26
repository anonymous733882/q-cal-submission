"""DocVQA benchmark loader.

Source: lmms-lab/DocVQA on HuggingFace (validation split, 5349 samples).
Falls back to local arrow file if HF download unavailable.
High-resolution document images with extractive QA.
"""

import os
from datasets import load_dataset


def _parse_answers(answers):
    if isinstance(answers, str):
        import ast
        try:
            answers = ast.literal_eval(answers)
        except (ValueError, SyntaxError):
            answers = [answers]
    return answers


def load_docvqa(num_samples: int = 100, data_dir: str = "datasets/docvqa_val",
                seed: int = 42) -> list:
    """Load DocVQA validation samples.

    Tries lmms-lab/DocVQA first, falls back to local arrow file.

    Returns:
        list of {image: PIL.Image, question: str, answers: list[str]}
    """
    ds = None

    # Try HuggingFace download first
    try:
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        print(f"  [DocVQA] Loaded {len(ds)} samples from HuggingFace", flush=True)
    except Exception as e:
        print(f"  [DocVQA] HF download failed ({e}), trying local arrow", flush=True)

    # Fallback to local arrow
    if ds is None:
        arrow_path = os.path.join(data_dir, "data-00000-of-00001.arrow")
        if not os.path.exists(arrow_path):
            print(f"DocVQA data not found at {arrow_path}")
            return []
        ds = load_dataset("arrow", data_files=arrow_path, split="train")

    # Select subset BEFORE loading images (avoid loading all 5349 images)
    import random as _rng
    if len(ds) > num_samples:
        indices = list(range(len(ds)))
        _rng.Random(seed).shuffle(indices)
        ds = ds.select(indices[:num_samples])

    samples = []
    for item in ds:
        img = item.get("image")
        if img is None:
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")

        answers = _parse_answers(item.get("answers", []))

        samples.append({
            "image": img,
            "question": item["question"],
            "answers": answers,
        })

    return samples
