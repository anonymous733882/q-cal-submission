"""Evaluate layer-local author-method adapters on the main task protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
import sys

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "qcal/runners"))

from model_backends import load_backend
from baselines.evaluation.run_mask import (
    COST_SCHEMA,
    _cost_record,
    _summarize_cost,
    eval_metric,
    load_samples_ext,
    max_new_tokens_map,
    resolve_model,
    sample_images,
    turn_prompts,
)


def prefill_target_for_total_cost(method: str, target_percent: float, n_vis: int,
                                  total_layers: int, full_gen_tokens: int) -> tuple[float, dict]:
    """Choose a layer-local prefill target from the requested total token-layer cost."""
    if not 0 < target_percent < 100 or n_vis < 1 or total_layers < 2 or full_gen_tokens < 0:
        raise ValueError("Invalid total-cost target or token counts")
    stages = [2] if method == "FastV" else list(range(min(4, total_layers - 1), -1, -1))
    if method not in {"FastV", "PACT"}:
        raise ValueError(f"Unsupported layer-local method: {method}")
    full_cost = n_vis * total_layers * (1 + full_gen_tokens)
    requested_cost = full_cost * target_percent / 100.0
    candidates = []
    for stage in stages:
        denominator = total_layers - stage + total_layers * full_gen_tokens
        estimated_keep = round((requested_cost - stage * n_vis) / denominator)
        for kept in {max(1, min(n_vis, estimated_keep + delta)) for delta in (-1, 0, 1)}:
            prefill_percent = 100.0 * (stage * n_vis + (total_layers - stage) * kept) / (total_layers * n_vis)
            layer_equivalents = total_layers * prefill_percent / 100.0
            actual_stage = (2 if method == "FastV" else
                            min(4, max(0, math.floor(layer_equivalents - 1 + 1e-9))))
            if actual_stage != stage or layer_equivalents <= stage:
                continue
            estimated_total_cost = stage * n_vis + denominator * kept
            candidates.append((abs(estimated_total_cost - requested_cost), -stage,
                               prefill_percent, kept, stage, estimated_total_cost))
    if not candidates:
        raise ValueError(f"No feasible {method} retention for {target_percent}% total cost")
    _, _, prefill_percent, kept, stage, estimated_total_cost = min(candidates)
    return prefill_percent, {
        "requested_total_token_layer_percent": float(target_percent),
        "estimated_total_token_layer_percent": 100.0 * estimated_total_cost / full_cost,
        "estimated_prefill_target_percent": float(prefill_percent),
        "estimated_reduction_layer": int(stage),
        "estimated_retained_visual_tokens": int(kept),
        "reference_generation_tokens": int(full_gen_tokens),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["PACT", "FastV"], default="PACT")
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--target-token-layer-percent", required=True, type=float)
    parser.add_argument("--num-images", type=int, default=650)
    parser.add_argument("--questions-per-image", type=int, default=15)
    parser.add_argument("--resize-square", type=int, default=1008)
    parser.add_argument("--max-pixels", type=int, default=1016064)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate(args):
    if not torch.cuda.is_available():
        raise RuntimeError(f"{args.method} model evaluation requires a CUDA device")
    if os.environ.get("DUALSIGNAL_QWEN_ATTN_IMPLEMENTATION") != "sdpa":
        raise RuntimeError("Layer-local reduction requires DUALSIGNAL_QWEN_ATTN_IMPLEMENTATION=sdpa")
    model_path, backend = resolve_model(args.model)
    model = load_backend(model_path, backend)
    run_method = getattr(model, f"run_{args.method.lower()}", None)
    if run_method is None:
        raise RuntimeError(f"No {args.method} layer-local implementation for backend {backend}")
    samples = load_samples_ext(args.benchmark, args.num_images, args.questions_per_image)
    baseline_rows = []
    method_rows = []
    records = []
    max_tokens = max_new_tokens_map(args.benchmark)
    for sample_idx, sample in enumerate(samples):
        images = sample_images(sample, args.resize_square)
        turns = turn_prompts(args.benchmark, sample, args.questions_per_image)
        for turn in turns:
            setup_start = time.perf_counter()
            inputs = model.prepare_inputs_any(images, turn["prompt"], args.max_pixels, args.max_pixels)
            shared_setup_sec = time.perf_counter() - setup_start
            vis_pos, _ = model.visual_and_text_positions(inputs)
            n_vis = len(vis_pos)
            start = time.perf_counter()
            _, full_text = model.generate_ids_and_text(inputs, max_tokens)
            full_elapsed = time.perf_counter() - start
            full_score = eval_metric(args.benchmark, full_text, turn["sample"])
            full_cost = _cost_record(
                n_vis=n_vis, kept_tokens=n_vis, total_layers=model.total_layers,
                n_gen_tokens=len(model.processor.tokenizer.encode(full_text, add_special_tokens=False)),
                shared_setup_sec=shared_setup_sec, shared_question_sec=0,
                baseline_wall_sec=full_elapsed)
            start = time.perf_counter()
            prefill_target, target_details = prefill_target_for_total_cost(
                args.method, args.target_token_layer_percent, n_vis, model.total_layers,
                len(model.processor.tokenizer.encode(full_text, add_special_tokens=False)))
            output, method_details = run_method(
                inputs, max_tokens,
                target_token_layer_percent=prefill_target)
            method_details.update(target_details)
            elapsed = time.perf_counter() - start
            score = eval_metric(args.benchmark, output, turn["sample"])
            kept = method_details["kept_visual_tokens"]
            generation_count = len(model.processor.tokenizer.encode(output, add_special_tokens=False))
            cost = _cost_record(
                n_vis=n_vis, kept_tokens=kept, total_layers=model.total_layers,
                n_gen_tokens=generation_count, shared_setup_sec=shared_setup_sec,
                shared_question_sec=0, method_total_wall_sec=elapsed,
                method_prefill_wall_sec=0)
            cost["prefill_tl"] = float(
                method_details["reduction_layer"] * n_vis +
                (model.total_layers - method_details["reduction_layer"]) * kept)
            cost["total_tl"] = cost["prefill_tl"] + cost["gen_tl"]
            baseline_rows.append({"baseline_cost": full_cost})
            method_rows.append({"cost": cost})
            records.append({
                "sample_idx": sample_idx, "turn_id": turn["turn_id"],
                "baseline_score": full_score, "score": score,
                "baseline_output": full_text, "output": output,
                "method_details": method_details, "n_visual_tokens": n_vis,
            })
    if not records:
        raise RuntimeError(f"{args.method} produced no question records")
    baseline_scores = [row["baseline_score"] for row in records]
    method_scores = [row["score"] for row in records]
    baseline_mean = sum(baseline_scores) / len(records)
    method_mean = sum(method_scores) / len(records)
    return {
        "model": args.model, "benchmark": args.benchmark,
        "config": {
            "runner": "baselines/evaluation/run_layer.py",
            "method": args.method,
            "algorithm": (
                "EUTI + DBDPC + hidden-state merging + proportional attention"
                if args.method == "PACT" else
                "author FastV previous-layer attention top-k at decoder layer 2"
            ),
            "target_token_layer_percent": args.target_token_layer_percent,
            "token_layer_percent_scope": "visual tokens in decoder prefill and answer generation",
            "num_images": args.num_images,
            "questions_per_image": args.questions_per_image,
            "attention_implementation": "sdpa",
            "multi_round_semantics": "independent question prefill",
        },
        "n_samples": len(records),
        "baseline_mean": baseline_mean,
        "baseline_per_sample": baseline_scores,
        "baseline_cost": _summarize_cost(baseline_rows, key="baseline_cost"),
        "cost_schema": COST_SCHEMA,
        "methods": {
            args.method: {
                "score": method_mean,
                "preservation": method_mean / baseline_mean if baseline_mean else 0,
                "per_sample": method_scores,
                "cost": _summarize_cost(method_rows),
                "actual_visual_prefill_tl_percent_per_sample": [
                    row["method_details"]["actual_visual_prefill_tl_percent"] for row in records],
            }
        },
        "records": records,
    }


def main():
    args = parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
