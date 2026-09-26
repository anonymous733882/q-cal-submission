"""ChartQA benchmark loader.

Dataset: HuggingFaceM4/ChartQA, test split (2500 samples).
Fields: image (PIL), query, label (stringified list, e.g. "['14']").
Metric: relaxed accuracy — prediction matches any answer with ±5% numeric tolerance.
"""

import ast


def _parse_label(label_str):
    """Parse label field into a list of answer strings."""
    try:
        val = ast.literal_eval(label_str)
        if isinstance(val, list):
            return [str(v).strip() for v in val]
        return [str(val).strip()]
    except Exception:
        return [str(label_str).strip()]


def load_chartqa(num_samples=30):
    """Load ChartQA test samples.

    Returns list of dicts with keys: image (PIL), question, answers (list[str]).
    """
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/ChartQA", split="test")
    n = min(num_samples, len(ds))
    samples = []
    for i in range(n):
        row = ds[i]
        label = row["label"]
        # label is already a list of strings in the HF dataset
        answers = [str(v).strip() for v in label] if isinstance(label, list) else [str(label).strip()]
        samples.append({
            "image": row["image"].convert("RGB"),
            "question": row["query"],
            "answers": answers,
        })
    return samples
