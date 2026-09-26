"""Fixed inputs, signal selection, and scoring for the public diagnostics."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import random
import sys
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "qcal_support"))

from baselines.evaluation.run_broad_comparison import _vqa_accuracy, eval_metric, max_new_tokens_map  # noqa: E402
from model_backends import load_backend  # noqa: E402


MODEL_SPECS = {
    "qwen25vl3b": ("Qwen/Qwen2.5-VL-3B-Instruct", "qwen"),
    "qwen25vl": ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen"),
    "qwen3vl4b": ("Qwen/Qwen3-VL-4B-Instruct", "qwen"),
    "llava15_7b": ("llava-hf/llava-1.5-7b-hf", "llava"),
}
PREFERENCE_MODELS = ("qwen25vl3b", "qwen25vl", "qwen3vl4b")
PROMPT_MODELS = PREFERENCE_MODELS + ("llava15_7b",)
TASK_MANIFEST = ROOT / "data/final240/manifest.json"
PROMPT_GROUPS = ROOT / "data/prompt_invariance_100/groups.json"
VQA_BENCHMARKS = frozenset({"vqav2", "vizwiz", "okvqa"})
TASK_TYPES = ("overall", "global_like", "local_like", "no_difference")


def load_model(model_id: str):
    path, backend = MODEL_SPECS[model_id]
    return load_backend(path, backend)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prompt_groups(limit: int | None = None) -> list[dict[str, Any]]:
    payload = json.loads(PROMPT_GROUPS.read_text(encoding="utf-8"))
    groups = payload["groups"]
    if len(groups) != 100 or any(len(group["qas"]) != 5 for group in groups):
        raise ValueError("The fixed prompt-invariance input must contain 100 images and five questions each")
    return groups[:limit] if limit is not None else groups


def decode_group_image(group: dict[str, Any]) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(group["image_b64"], validate=True))).convert("RGB")


def load_tasks(limit: int | None = None) -> list[dict[str, Any]]:
    payload = json.loads(TASK_MANIFEST.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    if len(tasks) != 240:
        raise ValueError("The fixed selection-behavior input must contain 240 tasks")
    return tasks[:limit] if limit is not None else tasks


def prompt_mismatches(tasks: list[dict[str, Any]]) -> list[str]:
    return [
        row["id"] for row in tasks
        if row["benchmark"] != "grounding_iou"
        and not str(row.get("prompt") or "").strip().startswith(str(row.get("question") or "").strip())
    ]


def require_task_alignment(tasks: list[dict[str, Any]]) -> None:
    mismatches = prompt_mismatches(tasks)
    if mismatches:
        raise ValueError(
            "Fixed-task prompt/question mismatch; results would not score the stated question: "
            + ", ".join(mismatches)
        )


def task_context(task: dict[str, Any], *, protocol: str) -> tuple[Image.Image, dict[str, Any], str, str, int]:
    image_path = TASK_MANIFEST.parent / task["image"]
    if file_sha256(image_path) != task["image_sha256"]:
        raise ValueError(f"Image checksum mismatch for {task['id']}")
    image = Image.open(image_path).convert("RGB")
    benchmark = "grounding" if task["benchmark"] == "grounding_iou" else task["benchmark"]
    answers = task.get("answers") or []
    if not answers and task.get("answer") not in (None, ""):
        answers = [str(task["answer"])]
    sample = {
        "image": image,
        "images": [image],
        "question": task["question"],
        "answer": task.get("answer"),
        "answers": answers,
        "expression": task.get("expression"),
        "bbox": task.get("bbox"),
        "image_size": (image.width, image.height),
    }
    if protocol == "preference":
        max_new = max_new_tokens_map(benchmark)
    elif protocol == "depth":
        max_new = 128 if benchmark == "grounding" else {
            "mme": 16, "chartqa": 32, "ocrbench": 32, "textvqa": 32,
            "vizwiz": 32, "vqav2": 16, "okvqa": 16,
        }.get(benchmark, 32)
    else:
        raise ValueError(f"Unknown diagnostic protocol: {protocol}")
    return image, sample, task["prompt"], benchmark, max_new


def score_answer(benchmark: str, output: str, sample: dict[str, Any]) -> float:
    if benchmark in VQA_BENCHMARKS:
        value = _vqa_accuracy(output, sample["answers"])
    else:
        value = eval_metric(benchmark, output, sample)
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"Non-finite {benchmark} score")
    return score


def align_scores(scores: Any, n_vis: int) -> torch.Tensor:
    values = torch.as_tensor(scores, dtype=torch.float32).flatten().cpu()
    if values.numel() != n_vis:
        raise ValueError(f"Signal has {values.numel()} scores for {n_vis} visual tokens")
    if not torch.isfinite(values).all():
        raise ValueError("Signal contains non-finite scores")
    return values


def norm01(scores: torch.Tensor) -> torch.Tensor:
    values = scores.float().flatten().cpu()
    gap = values.max() - values.min()
    return (values - values.min()) / gap if float(gap) > 1e-9 else torch.zeros_like(values)


def topk_mask(scores: torch.Tensor, budget_percent: float) -> list[bool]:
    n_vis = scores.numel()
    n_keep = max(1, int(n_vis * budget_percent / 100.0))
    keep = set(scores.argsort(descending=True)[:n_keep].tolist())
    return [index in keep for index in range(n_vis)]


def diversity_mask(scores: torch.Tensor, visual_embeddings: torch.Tensor,
                   budget_percent: float, gamma: float = 20.0) -> list[bool]:
    n_vis = scores.numel()
    n_keep = max(1, int(n_vis * budget_percent / 100.0))
    current = norm01(scores)
    embeddings = F.normalize(torch.as_tensor(visual_embeddings, dtype=torch.float32).reshape(n_vis, -1).cpu(), dim=-1)
    distance_sq = (1.0 - embeddings @ embeddings.T).clamp(min=0.0).square()
    weights = torch.exp(-gamma * distance_sq)
    selected: list[int] = []
    available = torch.ones(n_vis, dtype=torch.bool)
    for _ in range(n_keep):
        masked = current.clone()
        masked[~available] = -float("inf")
        best = int(masked.argmax())
        selected.append(best)
        available[best] = False
        current -= weights[best] * float(current[best])
        current.clamp_(min=0.0)
    keep = set(selected)
    return [index in keep for index in range(n_vis)]


def depth_layer(total_layers: int, fraction: float) -> int:
    layer_number = max(1, min(total_layers, math.floor(total_layers * fraction + 0.5)))
    return layer_number - 1


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return ranks


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() != right.numel() or left.numel() <= 1:
        raise ValueError("Spearman requires equal-length vectors with at least two tokens")
    a, b = _rankdata(left), _rankdata(right)
    a, b = a - a.mean(), b - b.mean()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def jaccard(left: torch.Tensor, right: torch.Tensor, budget_percent: float) -> float:
    if left.numel() != right.numel() or not left.numel():
        raise ValueError("Jaccard requires nonempty equal-length vectors")
    k = max(1, int(left.numel() * budget_percent / 100.0))
    a = set(left.argsort(descending=True)[:k].tolist())
    b = set(right.argsort(descending=True)[:k].tolist())
    return len(a & b) / len(a | b)


def compare_rankings(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    return {"j10": jaccard(left, right, 10.0), "j20": jaccard(left, right, 20.0),
            "spearman": spearman(left, right)}


def summarize_task_scores(records: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    output = {}
    for task_type in TASK_TYPES:
        rows = records if task_type == "overall" else [r for r in records if r["task_type"] == task_type]
        baseline = [float(row["full_score"]) for row in rows]
        block = {"tasks": len(rows), "full_accuracy": mean(baseline) if baseline else None, "methods": {}}
        for method in methods:
            scores = [float(row["methods"][method]["score"]) for row in rows]
            block["methods"][method] = {"accuracy": mean(scores) if scores else None, "tasks": len(scores)}
        output[task_type] = block
    return output


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)
