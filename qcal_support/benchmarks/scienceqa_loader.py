"""ScienceQA benchmark loader.

ScienceQA: Science Question Answering with multi-modal context.
Source: derek-thomas/ScienceQA on HuggingFace
Format: {image, question, choices, answer (0-based index), subject, ...}

Only loads samples WITH images (filters out text-only questions).
Metric: multiple-choice accuracy.
"""

import random
from datasets import load_dataset


def load_scienceqa(num_samples: int = 100, seed: int = 42) -> list:
    """Load ScienceQA test samples with images.

    Returns:
        list of {image: PIL.Image, question: str, answer: str (letter),
                 choices: list[str], subject: str}
    """
    ds = load_dataset("derek-thomas/ScienceQA", split="test")

    letters = "ABCDEFGH"
    all_samples = []

    for item in ds:
        # Skip text-only questions
        if item.get("image") is None:
            continue

        choices = item["choices"]
        ans_idx = item["answer"]
        if ans_idx >= len(letters):
            continue

        # Format question with choices
        q = item["question"]
        hint = item.get("hint", "")
        if hint:
            q = f"Hint: {hint}\n{q}"

        choice_text = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
        full_q = f"{q}\n{choice_text}\nPlease answer directly with the letter."

        all_samples.append({
            "image": item["image"],
            "question": full_q,
            "answer": letters[ans_idx],
            "choices": choices,
            "subject": item.get("subject", ""),
        })

        if len(all_samples) >= num_samples * 2:
            break

    random.seed(seed)
    random.shuffle(all_samples)
    return all_samples[:num_samples]
