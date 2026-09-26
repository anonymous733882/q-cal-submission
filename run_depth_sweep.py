#!/usr/bin/env python3
"""Evaluate decoder text-to-visual attention at six depths on fixed tasks."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from statistics import mean, pvariance
import sys

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "qcal/runners"))

from diagnostic_common import (  # noqa: E402
    PREFERENCE_MODELS, TASK_MANIFEST, TASK_TYPES, align_scores, depth_layer,
    file_sha256, load_model, load_tasks, prompt_mismatches, require_task_alignment,
    score_answer, set_seed, summarize_task_scores, task_context, topk_mask, write_json,
)


DEFAULT_DEPTHS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=PREFERENCE_MODELS, default=list(PREFERENCE_MODELS))
    parser.add_argument("--budgets", nargs="+", type=int, default=[10, 20, 33])
    parser.add_argument("--depths", nargs="+", type=float, default=list(DEFAULT_DEPTHS))
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--max-pixels", type=int, default=1016064)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/diagnostics/depth_sweep")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_tasks(args: argparse.Namespace) -> list[dict]:
    tasks = load_tasks()
    if args.task_ids:
        chosen = set(args.task_ids)
        tasks = [task for task in tasks if task["id"] in chosen]
        if len(tasks) != len(chosen):
            raise ValueError("Unknown or duplicate task ID")
    if args.limit_tasks is not None:
        tasks = tasks[:args.limit_tasks]
    if not tasks:
        raise ValueError("No tasks selected")
    return tasks


def method_names(args: argparse.Namespace) -> list[str]:
    return [f"depth_d{round(depth * 100):02d}_b{budget}" for depth in args.depths
            for budget in args.budgets]


@torch.no_grad()
def evaluate_one(model, task: dict, args: argparse.Namespace, layer_map: dict[str, int]) -> dict:
    image, sample, prompt, benchmark, max_new = task_context(task, protocol="depth")
    inputs = model.prepare_inputs_any([image], prompt, args.max_pixels, args.max_pixels)
    _, full_output = model.generate_ids_and_text(inputs, max_new)
    full_score = score_answer(benchmark, full_output, sample)
    visual_embeddings, grid = model.extract_visual_embeddings(inputs)
    n_vis = int(visual_embeddings.shape[0])
    scores_by_layer = model.layer_attention_scores(inputs, sorted(set(layer_map.values())))
    methods = {}
    for depth in args.depths:
        name = f"d{round(depth * 100):02d}"
        layer = layer_map[name]
        scores = align_scores(scores_by_layer[layer], n_vis)
        for budget in args.budgets:
            mask = topk_mask(scores, budget)
            output = model.run_pruned(inputs, visual_embeddings, grid, mask, max_new)
            methods[f"depth_{name}_b{budget}"] = {
                "output": output, "score": score_answer(benchmark, output, sample),
                "kept_visual_tokens": sum(mask), "layer_index": layer,
                "layer_number": layer + 1,
            }
    return {"task_id": task["id"], "task_type": task["task_type"],
            "benchmark": task["benchmark"], "question": task["question"], "prompt": prompt,
            "answer": task.get("answer"), "answers": task.get("answers"),
            "full_output": full_output, "full_score": full_score,
            "visual_tokens": n_vis, "methods": methods}


def read_checkpoint(path: Path, tasks: list[dict], methods: list[str], plan: dict) -> list[dict]:
    if not path.exists():
        return []
    meta = path.with_suffix(".plan.json")
    if not meta.exists() or json.loads(meta.read_text(encoding="utf-8")) != plan:
        raise ValueError(f"Checkpoint plan mismatch: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if [row["task_id"] for row in records] != [task["id"] for task in tasks[:len(records)]]:
        raise ValueError(f"Checkpoint task order mismatch: {path}")
    if any(set(row["methods"]) != set(methods) for row in records):
        raise ValueError(f"Checkpoint method set mismatch: {path}")
    return records


def run_model(model_id: str, tasks: list[dict], args: argparse.Namespace, plan: dict) -> dict:
    methods = method_names(args)
    record_path = args.out_dir / f"{model_id}.jsonl"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and record_path.exists():
        records = read_checkpoint(record_path, tasks, methods, plan)
    else:
        records = []
        write_json(record_path.with_suffix(".plan.json"), plan)
        record_path.write_text("", encoding="utf-8")
    if len(records) < len(tasks):
        model = load_model(model_id)
        layer_map = {f"d{round(depth * 100):02d}": depth_layer(model.total_layers, depth)
                     for depth in args.depths}
        for task in tasks[len(records):]:
            row = evaluate_one(model, task, args, layer_map)
            with record_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            records.append(row)
            print(f"{model_id}: {len(records)}/{len(tasks)} tasks", flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_task_scores(records, methods)
    write_json(args.out_dir / f"{model_id}.summary.json", {"plan": plan, "model": model_id,
                                                           "summary": summary})
    return {"model": model_id, "summary": summary}


def cross_model(results: list[dict], methods: list[str]) -> dict:
    output = {}
    for task_type in TASK_TYPES:
        blocks = [result["summary"][task_type] for result in results]
        if blocks[0]["tasks"] == 0:
            output[task_type] = {"tasks_per_model": 0, "full_accuracy_mean": None,
                                 "methods": {name: {"accuracy_mean": None, "accuracy_variance": None,
                                                    "accuracy_min_model": None, "accuracy_max_model": None}
                                             for name in methods}}
            continue
        full = [block["full_accuracy"] for block in blocks]
        output[task_type] = {
            "tasks_per_model": blocks[0]["tasks"], "full_accuracy_mean": mean(full),
            "methods": {
                name: {
                    "accuracy_mean": mean(block["methods"][name]["accuracy"] for block in blocks),
                    "accuracy_variance": pvariance(block["methods"][name]["accuracy"] for block in blocks),
                    "accuracy_min_model": min(block["methods"][name]["accuracy"] for block in blocks),
                    "accuracy_max_model": max(block["methods"][name]["accuracy"] for block in blocks),
                } for name in methods
            },
        }
    return output


def main() -> int:
    args = parse_args()
    if (args.limit_tasks is not None and args.limit_tasks < 1) or any(
            not 0 < depth <= 1 for depth in args.depths) or any(
            not 0 < budget <= 100 for budget in args.budgets):
        raise SystemExit("Invalid task limit, depth, or budget")
    if len({round(depth * 100) for depth in args.depths}) != len(args.depths) or len(set(args.budgets)) != len(args.budgets):
        raise SystemExit("Duplicate depth or budget labels")
    tasks = selected_tasks(args)
    plan = {"task_manifest_sha256": file_sha256(TASK_MANIFEST), "task_ids": [t["id"] for t in tasks],
            "models": args.models, "budgets": args.budgets, "depth_fractions": args.depths,
            "layer_rounding": "one-based floor(total_layers * fraction + 0.5), then subtract one",
            "max_pixels": args.max_pixels, "seed": args.seed,
            "generation": "source depth-sweep protocol; full baseline rerun for each fixed task"}
    if args.dry_run:
        print(json.dumps({"plan": plan, "prompt_question_mismatches": prompt_mismatches(tasks)}, indent=2))
        return 0
    require_task_alignment(tasks)
    set_seed(args.seed)
    results = [run_model(model_id, tasks, args, plan) for model_id in args.models]
    write_json(args.out_dir / "summary_cross_model.json", {
        "plan": plan, "models": args.models, "summary": cross_model(results, method_names(args))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
