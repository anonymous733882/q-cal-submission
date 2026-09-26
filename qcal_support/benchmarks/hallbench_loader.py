"""HallusionBench loader.

Uses lmms-lab/HallusionBench from HuggingFace Hub (image split, 951 samples).
Tests visual hallucination via yes/no questions (gt_answer: '0'=no, '1'=yes).
"""

import random
from datasets import load_dataset


def load_hallbench(num_samples: int = 200, seed: int = 42) -> list:
    """Load HallusionBench image-split samples.

    Returns:
        list of {image: PIL.Image, question: str, answer: str, category: str}
    """
    ds = load_dataset("lmms-lab/HallusionBench", split="image")

    samples = []
    for item in ds:
        img = item.get("image")
        if img is None:
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")

        # gt_answer is '0' (no) or '1' (yes)
        gt = item.get("gt_answer", "")
        answer = "yes" if str(gt).strip() == "1" else "no"

        samples.append({
            "image": img,
            "question": item["question"],
            "answer": answer,
            "category": item.get("category", "unknown"),
        })

    random.seed(seed)
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)

    return samples
