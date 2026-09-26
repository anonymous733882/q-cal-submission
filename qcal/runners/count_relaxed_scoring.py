"""Relaxed scoring for counting-style short answers."""

from __future__ import annotations

import re
from typing import Any


COUNT_CUES = (
    "how many",
    "number of",
    "count",
    "total number",
    "amount of",
)

NUMBER_WORDS = {
    "zero": 0,
    "none": 0,
    "no": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def is_count_question(question: Any) -> bool:
    text = str(question or "").lower()
    return any(cue in text for cue in COUNT_CUES)


def extract_number(text: Any) -> float | None:
    text = str(text or "").strip().lower()
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except Exception:
            return None
    for word in re.findall(r"[a-z]+", text):
        if word in NUMBER_WORDS:
            return float(NUMBER_WORDS[word])
    return None


def relaxed_count_score(pred: Any, gt: Any) -> float | None:
    pred_num = extract_number(pred)
    gt_num = extract_number(gt)
    if pred_num is None or gt_num is None:
        return None
    denom = max(abs(float(gt_num)), 1.0)
    return max(0.0, 1.0 - abs(float(pred_num) - float(gt_num)) / denom)


def maybe_relaxed_count_score(pred: Any, gt: Any, question: Any = None) -> float | None:
    gt_num = extract_number(gt)
    if gt_num is None:
        return None
    if question is not None and not is_count_question(question):
        # Numeric non-count answers still appear in OCR/doc tasks. Only relax
        # them when the question is count-like; callers without question context
        # may intentionally relax all numeric GT answers.
        return None
    return relaxed_count_score(pred, gt)
