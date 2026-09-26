from __future__ import annotations

import re
from typing import Any


def _norm_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _norm_label(text: Any) -> str:
    return _norm_text(text).strip(" .,:;!?\"'()[]{}")


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def extract_yes_no(text: Any) -> str | None:
    s = _norm_text(text)
    if not s:
        return None
    if re.match(r"^(yes|no)\b", s):
        return _norm_label(s.split()[0])
    bold = re.search(r"\*\*\s*(yes|no)\s*\*\*", s)
    if bold:
        return bold.group(1)
    if re.search(r"\b(not|does not|do not|did not|is not|are not|cannot|can't|no,|not possible)\b", s):
        return "no"
    if re.search(r"\b(yes|does|do|did|is|are)\b", s):
        return "yes"
    return None


def hallbench_yesno_score(output: Any, sample: dict[str, Any]) -> float | None:
    answer = _norm_label(sample.get("answer", sample.get("label", "")))
    if answer not in ("yes", "no"):
        return None
    pred = extract_yes_no(output)
    if pred is None:
        return 0.0
    return 1.0 if pred == answer else 0.0


def docvqa_relaxed_score(output: Any, sample: dict[str, Any]) -> float | None:
    answers = sample.get("answers", [])
    if not answers:
        return None
    pred = _norm_text(output)
    if not pred:
        return 0.0
    best = 0.0
    for answer in answers:
        gt = _norm_text(answer)
        if not gt:
            continue
        if pred == gt:
            best = max(best, 1.0)
        else:
            dist = _levenshtein(pred, gt)
            similarity = 1.0 - (dist / max(len(pred), len(gt)))
            best = max(best, similarity if similarity >= 0.5 else 0.0)
    return float(best)


def diagnostic_scores(benchmark: str, output: Any, sample: dict[str, Any]) -> dict[str, float]:
    if benchmark == "hallbench":
        score = hallbench_yesno_score(output, sample)
        return {"hallbench_yesno": float(score)} if score is not None else {}
    if benchmark == "docvqa":
        score = docvqa_relaxed_score(output, sample)
        return {"docvqa_relaxed": float(score)} if score is not None else {}
    return {}
