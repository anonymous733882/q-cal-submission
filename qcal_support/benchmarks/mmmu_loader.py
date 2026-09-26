"""MMMU benchmark loader.

MMMU: Massive Multi-discipline Multimodal Understanding.
Source: MMMU/MMMU on HuggingFace (per-subject configs)
Format: {question, options (list), answer (letter), image_1..image_7, subfield}

Only loads single-image questions (image_1 is not None, image_2 is None).
Metric: multiple-choice accuracy.
"""

import random
from datasets import load_dataset

# Representative subjects covering different disciplines
MMMU_SUBJECTS = [
    "Accounting", "Architecture_and_Engineering", "Art",
    "Biology", "Chemistry", "Computer_Science",
    "Economics", "Electronics", "Geography",
    "History", "Math", "Mechanical_Engineering",
    "Music", "Physics", "Psychology",
]


def load_mmmu(num_samples: int = 100, seed: int = 42) -> list:
    """Load MMMU validation samples (single-image only, stratified by subject).

    Returns:
        list of {image: PIL.Image, question: str, answer: str (letter),
                 options: list[str], subfield: str}
    """
    letters = "ABCDEFGHIJKLMNOP"
    # Budget per subject: load enough to get num_samples total
    per_subject = max(10, num_samples // len(MMMU_SUBJECTS) + 5)
    all_samples = []

    for subject in MMMU_SUBJECTS:
        subject_count = 0
        for split in ("validation", "test"):
            if subject_count >= per_subject:
                break
            try:
                ds = load_dataset("MMMU/MMMU", subject, split=split)
            except Exception as e:
                continue

            for item in ds:
                if subject_count >= per_subject:
                    break
                # Only single-image questions
                img = item.get("image_1")
                if img is None:
                    continue
                if item.get("image_2") is not None:
                    continue

                options = item.get("options", [])
                if isinstance(options, str):
                    import ast
                    try:
                        options = ast.literal_eval(options)
                    except:
                        continue

                answer = item.get("answer", "")
                if answer not in letters:
                    continue

                # Replace <image 1> placeholder
                q = item["question"].replace("<image 1>", "")

                choice_text = "\n".join(
                    f"{letters[i]}. {o}" for i, o in enumerate(options))
                full_q = f"{q.strip()}\n{choice_text}\nPlease answer directly with the letter."

                all_samples.append({
                    "image": img,
                    "question": full_q,
                    "answer": answer,
                    "options": options,
                    "subfield": item.get("subfield", subject),
                })
                subject_count += 1

    random.seed(seed)
    random.shuffle(all_samples)
    return all_samples[:num_samples]
