"""Local VSR loader.

Expects prepared records at qcal_support/datasets/vsr/records.json. Records reference
local COCO image paths and use yes/no answers.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from PIL import Image


def _root() -> Path:
    return (Path(__file__).resolve().parents[2] / "qcal").resolve()


def _bea() -> Path:
    return (_root().parent / "qcal_support").resolve()


def _data() -> Path:
    return Path(os.environ.get("VSR_DATA_DIR", _bea() / "datasets" / "vsr")).resolve()


def _records_path() -> Path:
    return Path(os.environ.get("VSR_RECORDS", _data() / "records.json")).resolve()


def _load_records() -> list[dict]:
    path = _records_path()
    if not path.exists():
        raise FileNotFoundError(
            f"VSR records not found: {path}; prepare qcal_support/datasets/vsr/records.json first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _open_images(rec: dict) -> list[Image.Image]:
    out = []
    for item in rec.get("images", []):
        path = Path(item)
        if not path.is_absolute():
            path = _data() / "images" / item
        out.append(Image.open(path).convert("RGB"))
    return out


def load_vsr(num_samples: int = 200, seed: int = 42) -> list[dict]:
    records = _load_records()[:num_samples]
    samples = []
    for rec in records:
        sample = {k: v for k, v in rec.items() if k != "images"}
        sample["images"] = _open_images(rec)
        samples.append(sample)
    print(f"VSR: loaded {len(samples)} samples")
    return samples


def evaluate_response(response: str, sample: dict) -> bool:
    match = re.search(r"\b(yes|no|true|false)\b", str(response).strip().lower())
    if not match:
        return False
    pred = {"true": "yes", "false": "no"}.get(match.group(1), match.group(1))
    gold = {"true": "yes", "false": "no", "1": "yes", "0": "no"}.get(
        str(sample.get("answer", "")).strip().lower(),
        str(sample.get("answer", "")).strip().lower(),
    )
    return pred == gold
