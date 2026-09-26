"""RealWorldQA loader.

Uses lmms-lab/RealWorldQA from HuggingFace Hub (test split, 765 samples).
Multiple-choice questions about real-world images. Answer is a letter (A/B/C/D).
"""

import random
from datasets import load_dataset


def load_realworldqa(num_samples: int = 200, seed: int = 42) -> list:
    """Load RealWorldQA test samples.

    Returns:
        list of {image: PIL.Image, question: str, answer: str}
    """
    ds = load_dataset("lmms-lab/RealWorldQA", split="test")

    samples = []
    for item in ds:
        img = item.get("image")
        if img is None:
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")

        samples.append({
            "image": img,
            "question": item["question"],
            "answer": item["answer"].strip(),
        })

    random.seed(seed)
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)

    return samples
