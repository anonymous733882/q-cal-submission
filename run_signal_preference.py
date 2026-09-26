#!/usr/bin/env python3
"""Evaluate signal preference on the fixed 240-task diagnostic pool."""

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
    diversity_mask, file_sha256, load_model, load_tasks, norm01, prompt_mismatches,
    require_task_alignment, score_answer, set_seed, summarize_task_scores,
    task_context, topk_mask, write_json,
)


FAMILIES = ("visual", "early", "dual", "deep")
TOPK_WEIGHTS = {10: 0.1, 20: 0.1, 33: 0.2}
DIVERSITY_WEIGHTS = {10: 0.1, 20: 0.2, 33: 0.4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=PREFERENCE_MODELS, default=list(PREFERENCE_MODELS))
    parser.add_argument("--budgets", nargs="+", type=int, choices=(10, 20, 33), default=[10, 20, 33])
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--modes", nargs="+", choices=("topk", "diversity"),
                        default=["topk", "diversity"])
    parser.add_argument("--task-ids", nargs="+", help="Run named fixed tasks; useful for a small integration check")
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--max-pixels", type=int, default=1016064)
    parser.add_argument("--deep-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/diagnostics/signal_preference")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def method_names(args: argparse.Namespace) -> list[str]:
    return [f"{family}_{mode}_b{budget}" for budget in args.budgets
            for family in args.families for mode in args.modes]


def selected_tasks(args: argparse.Namespace) -> list[dict]:
    tasks = load_tasks()
    if args.task_ids:
        selected = set(args.task_ids)
        tasks = [task for task in tasks if task["id"] in selected]
        if len(tasks) != len(selected):
            raise ValueError("Unknown or duplicate fixed task ID")
    if args.limit_tasks is not None:
        tasks = tasks[:args.limit_tasks]
    if not tasks:
        raise ValueError("No tasks selected")
    return tasks


def read_checkpoint(path: Path, tasks: list[dict], methods: list[str], plan: dict) -> list[dict]:
    if not path.exists():
        return []
    meta_path = path.with_suffix(".plan.json")
    if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")) != plan:
        raise ValueError(f"Checkpoint plan mismatch: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if [row["task_id"] for row in rows] != [task["id"] for task in tasks[:len(rows)]]:
        raise ValueError(f"Checkpoint task order mismatch: {path}")
    if any(set(row["methods"]) != set(methods) for row in rows):
        raise ValueError(f"Checkpoint method set mismatch: {path}")
    return rows


@torch.no_grad()
def evaluate_one(model, task: dict, args: argparse.Namespace, deep_layer: int) -> dict:
    image, sample, prompt, benchmark, max_new = task_context(task, protocol="preference")
    inputs = model.prepare_inputs_any([image], prompt, args.max_pixels, args.max_pixels)
    _, full_output = model.generate_ids_and_text(inputs, max_new)
    full_score = score_answer(benchmark, full_output, sample)
    visual_embeddings, grid = model.extract_visual_embeddings(inputs)
    n_vis = int(visual_embeddings.shape[0])
    layers = model.layer_attention_scores(inputs, sorted({0, deep_layer}))
    early = align_scores(layers[0], n_vis)
    deep = align_scores(layers[deep_layer], n_vis)
    visual = align_scores(model.visual_received_attention_scores(inputs, n_vis), n_vis)
    signals = {"visual": visual, "early": early, "deep": deep}
    methods = {}
    for budget in args.budgets:
        for family in args.families:
            for mode in args.modes:
                if family == "dual":
                    weight = (TOPK_WEIGHTS if mode == "topk" else DIVERSITY_WEIGHTS)[budget]
                    scores = weight * norm01(early) + (1.0 - weight) * norm01(visual)
                else:
                    weight = None
                    scores = signals[family]
                mask = (topk_mask(scores, budget) if mode == "topk" else
                        diversity_mask(scores, visual_embeddings, budget))
                output = model.run_pruned(inputs, visual_embeddings, grid, mask, max_new)
                key = f"{family}_{mode}_b{budget}"
                methods[key] = {"output": output, "score": score_answer(benchmark, output, sample),
                                "kept_visual_tokens": sum(mask), "dual_early_weight": weight}
    return {"task_id": task["id"], "task_type": task["task_type"],
            "benchmark": task["benchmark"], "question": task["question"], "prompt": prompt,
            "answer": task.get("answer"), "answers": task.get("answers"),
            "full_output": full_output, "full_score": full_score,
            "visual_tokens": n_vis, "deep_layer_index": deep_layer, "methods": methods}


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
        deep_layer = depth_layer(model.total_layers, args.deep_fraction)
        for task in tasks[len(records):]:
            row = evaluate_one(model, task, args, deep_layer)
            with record_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            records.append(row)
            print(f"{model_id}: {len(records)}/{len(tasks)} tasks", flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_task_scores(records, methods)
    write_json(args.out_dir / f"{model_id}.summary.json", {"plan": plan, "model": model_id, "summary": summary})
    return {"model": model_id, "summary": summary}


def cross_model(results: list[dict], methods: list[str]) -> dict:
    output = {}
    for task_type in TASK_TYPES:
        blocks = [result["summary"][task_type] for result in results]
        if blocks[0]["tasks"] == 0:
            output[task_type] = {"tasks_per_model": 0, "full_accuracy_mean": None,
                                 "full_accuracy_variance": None,
                                 "methods": {name: {"accuracy_mean": None, "accuracy_variance": None}
                                             for name in methods}}
            continue
        values = [block["full_accuracy"] for block in blocks]
        output[task_type] = {
            "tasks_per_model": blocks[0]["tasks"],
            "full_accuracy_mean": mean(values),
            "full_accuracy_variance": pvariance(values),
            "methods": {
                method: {
                    "accuracy_mean": mean(block["methods"][method]["accuracy"] for block in blocks),
                    "accuracy_variance": pvariance(block["methods"][method]["accuracy"] for block in blocks),
                } for method in methods
            },
        }
    return output


def main() -> int:
    args = parse_args()
    if not 0 < args.deep_fraction <= 1 or (args.limit_tasks is not None and args.limit_tasks < 1):
        raise SystemExit("Invalid depth fraction or task limit")
    tasks = selected_tasks(args)
    plan = {"task_manifest_sha256": file_sha256(TASK_MANIFEST), "task_ids": [t["id"] for t in tasks],
            "models": args.models, "budgets": args.budgets, "families": args.families,
            "modes": args.modes,
            "dual_topk_early_weights": {str(b): TOPK_WEIGHTS[b] for b in args.budgets},
            "dual_diversity_early_weights": {str(b): DIVERSITY_WEIGHTS[b] for b in args.budgets},
            "deep_fraction": args.deep_fraction, "early_layer_index": 0,
            "max_pixels": args.max_pixels, "seed": args.seed,
            "generation": "source preference protocol; full baseline rerun for each fixed task"}
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
