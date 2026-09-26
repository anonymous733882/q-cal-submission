"""TextVQA benchmark loader.

Uses lmms-lab/textvqa from HuggingFace Hub.
Validation split: ~5,000 samples with OCR-rich images.
Each sample has 10 annotator answers for soft-voting evaluation.
"""

import random
from datasets import load_dataset


def load_textvqa(num_samples: int = 200, seed: int = 42) -> list:
    """Load TextVQA validation samples.

    Returns:
        list of {image: PIL.Image, question: str, answers: list[str]}
    """
    ds = load_dataset("lmms-lab/textvqa", split="validation")

    samples = []
    for item in ds:
        img = item["image"]
        if img is None:
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Parse answers (may be string repr of list)
        answers = item.get("answers", [])
        if isinstance(answers, str):
            import ast
            try:
                answers = ast.literal_eval(answers)
            except (ValueError, SyntaxError):
                answers = [answers]

        samples.append({
            "image": img,
            "question": item["question"],
            "answers": answers,
        })

        if len(samples) >= num_samples * 3:
            # Buffer enough to sample from
            break

    # Deterministic subsample
    random.seed(seed)
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)

    return samples
