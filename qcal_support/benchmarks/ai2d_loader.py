"""AI2D benchmark loader.

AI2D: AI2 Diagrams — multiple-choice questions about scientific diagrams.
Source: lmms-lab/ai2d on HuggingFace
Format: {question, options (list), answer (0-based index), image}

Metric: multiple-choice accuracy.
"""

import random
from datasets import load_dataset


def load_ai2d(num_samples: int = 100, seed: int = 42) -> list:
    """Load AI2D test samples.

    Returns:
        list of {image: PIL.Image, question: str, answer: str (letter),
                 options: list[str]}
    """
    ds = load_dataset("lmms-lab/ai2d", split="test")

    letters = "ABCDEFGH"
    all_samples = []

    for item in ds:
        if item.get("image") is None:
            continue

        options = item["options"]
        ans_idx = int(item["answer"])
        if ans_idx >= len(options) or ans_idx >= len(letters):
            continue

        choice_text = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
        full_q = f"{item['question']}\n{choice_text}\nPlease answer directly with the letter."

        all_samples.append({
            "image": item["image"],
            "question": full_q,
            "answer": letters[ans_idx],
            "options": options,
        })

        if len(all_samples) >= num_samples * 2:
            break

    random.seed(seed)
    random.shuffle(all_samples)
    return all_samples[:num_samples]
