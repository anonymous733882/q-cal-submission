#!/usr/bin/env python3
"""Run held-out policy training and export the global policy JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path


DUALSIGNAL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = DUALSIGNAL_ROOT.parent
RUNNERS = DUALSIGNAL_ROOT / "runners"
TRAIN = RUNNERS / "run_gradient_policy_heldout_suite.py"
EXPORT = RUNNERS / "export_policy_from_heldout_suite.py"
CORE8 = ["realworldqa", "blink", "textvqa", "docvqa", "chartqa", "ai2d", "mmmu", "hallbench"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--backend_type", default="qwen")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--benchmarks", nargs="+", default=CORE8)
    p.add_argument("--budgets", nargs="+", type=float, default=[10.0, 5.0])
    p.add_argument("--calib_samples", type=int, default=32)
    p.add_argument("--calib_offset", type=int, default=0)
    p.add_argument("--eval_samples", type=int, default=0)
    p.add_argument("--eval_offset", type=int, default=900)
    p.add_argument("--num_sweep", type=int, default=256)
    p.add_argument("--calib_filter_visual_grounded", action="store_true")
    p.add_argument("--calib_candidates_per_target", type=int, default=12)
    p.add_argument("--calibration_indices_manifest", default=None)
    p.add_argument("--calib_min_baseline_score", type=float, default=1e-9)
    p.add_argument("--max_layer_frac", type=float, default=0.75)
    p.add_argument("--max_active_layers", type=int, default=5)
    p.add_argument("--active_weight_threshold", type=float, default=0.05)
    p.add_argument("--fit_device", choices=["cuda"], default="cuda")
    p.add_argument("--train_token_pool", default="full")
    p.add_argument("--stage1_frac", type=float, default=0.33)
    p.add_argument("--stage1_gamma", type=float, default=20.0)
    p.add_argument("--fit_loss", default="budget_overlap_pairwise")
    p.add_argument("--optim_steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--rank_temperature", type=float, default=0.12)
    p.add_argument("--oracle_temperature", type=float, default=0.25)
    p.add_argument("--depth_lambda", type=float, default=0.01)
    p.add_argument("--entropy_lambda", type=float, default=0.001)
    p.add_argument("--l2_lambda", type=float, default=0.001)
    p.add_argument("--effective_layers_lambda", type=float, default=0.05)
    p.add_argument("--effective_layers_target", type=float, default=2.5)
    p.add_argument("--consistency_lambda", type=float, default=0.15)
    p.add_argument("--uncertainty_lambda", type=float, default=0.05)
    p.add_argument("--consistency_mode", default="static")
    p.add_argument("--min_deep_layer_frac", type=float, default=0.5)
    p.add_argument("--min_deep_weight", type=float, default=0.45)
    p.add_argument("--min_deep_weight_lambda", type=float, default=10.0)
    p.add_argument("--oracle_clip_quantile", type=float, default=0.95)
    p.add_argument("--oracle_score_transform", default="log1p")
    p.add_argument("--oracle_mode", default="gradient")
    p.add_argument("--oracle_target", default="gt")
    p.add_argument("--gradient_oracle_score", default="directional")
    p.add_argument("--oracle_candidate_tokens", type=int, default=96)
    p.add_argument("--oracle_insertion_groups", type=int, default=8)
    p.add_argument("--oracle_insertion_weight", type=float, default=0.4)
    p.add_argument("--oracle_visual_contrast_weight", type=float, default=0.4)
    p.add_argument("--oracle_gradient_weight", type=float, default=0.2)
    p.add_argument("--oracle_interaction_weight", type=float, default=0.0)
    p.add_argument("--oracle_component_transform", default="minmax")
    p.add_argument("--oracle_attribution_mode", default="group_insertion")
    p.add_argument("--oracle_budget_attribution_fracs", default="10,5")
    p.add_argument("--oracle_budget_attribution_weights", default="0.5,0.5")
    p.add_argument("--oracle_insertion_score_mode", default="auto")
    p.add_argument("--calibration_cache_dir", default=None)
    p.add_argument("--test_cache_dir", default=None)
    p.add_argument("--force_retrain", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    suite_json = out_dir / "suite.json"
    policy_json = out_dir / "global_core8_policy.json"
    if suite_json.exists() and policy_json.exists() and not args.force_retrain:
        print(json.dumps({"skip": str(out_dir), "reason": "suite_and_policy_exist"}, indent=2), flush=True)
        return 0

    cmd = [
        sys.executable, "-u", str(TRAIN),
        "--model_path", args.model_path,
        "--backend_type", args.backend_type,
        "--out_dir", str(out_dir),
        "--benchmarks", *args.benchmarks,
        "--budgets", *[str(x) for x in args.budgets],
        "--calib_samples", str(args.calib_samples),
        "--calib_offset", str(args.calib_offset),
        "--eval_samples", str(args.eval_samples),
        "--eval_offset", str(args.eval_offset),
        "--num_sweep", str(args.num_sweep),
        "--calib_candidates_per_target", str(args.calib_candidates_per_target),
        "--calib_min_baseline_score", str(args.calib_min_baseline_score),
        "--max_layer_frac", str(args.max_layer_frac),
        "--max_active_layers", str(args.max_active_layers),
        "--active_weight_threshold", str(args.active_weight_threshold),
        "--fit_device", args.fit_device,
        "--train_token_pool", args.train_token_pool,
        "--stage1_frac", str(args.stage1_frac),
        "--stage1_gamma", str(args.stage1_gamma),
        "--fit_loss", args.fit_loss,
        "--shared_budget_policy",
        "--global_only",
        "--optim_steps", str(args.optim_steps),
        "--lr", str(args.lr),
        "--rank_temperature", str(args.rank_temperature),
        "--oracle_temperature", str(args.oracle_temperature),
        "--depth_lambda", str(args.depth_lambda),
        "--entropy_lambda", str(args.entropy_lambda),
        "--l2_lambda", str(args.l2_lambda),
        "--effective_layers_lambda", str(args.effective_layers_lambda),
        "--effective_layers_target", str(args.effective_layers_target),
        "--consistency_lambda", str(args.consistency_lambda),
        "--uncertainty_lambda", str(args.uncertainty_lambda),
        "--consistency_mode", args.consistency_mode,
        "--min_deep_layer_frac", str(args.min_deep_layer_frac),
        "--min_deep_weight", str(args.min_deep_weight),
        "--min_deep_weight_lambda", str(args.min_deep_weight_lambda),
        "--oracle_clip_quantile", str(args.oracle_clip_quantile),
        "--oracle_score_transform", args.oracle_score_transform,
        "--oracle_mode", args.oracle_mode,
        "--oracle_target", args.oracle_target,
        "--gradient_oracle_score", args.gradient_oracle_score,
        "--oracle_candidate_tokens", str(args.oracle_candidate_tokens),
        "--oracle_insertion_groups", str(args.oracle_insertion_groups),
        "--oracle_insertion_weight", str(args.oracle_insertion_weight),
        "--oracle_visual_contrast_weight", str(args.oracle_visual_contrast_weight),
        "--oracle_gradient_weight", str(args.oracle_gradient_weight),
        "--oracle_interaction_weight", str(args.oracle_interaction_weight),
        "--oracle_component_transform", args.oracle_component_transform,
        "--oracle_attribution_mode", args.oracle_attribution_mode,
        "--oracle_budget_attribution_fracs", args.oracle_budget_attribution_fracs,
        "--oracle_budget_attribution_weights", args.oracle_budget_attribution_weights,
        "--oracle_insertion_score_mode", args.oracle_insertion_score_mode,
    ]
    if args.calib_filter_visual_grounded:
        cmd.append("--calib_filter_visual_grounded")
    if args.calibration_indices_manifest:
        cmd.extend(["--calibration_indices_manifest", args.calibration_indices_manifest])
    if args.calibration_cache_dir:
        cmd.extend(["--calibration_cache_dir", args.calibration_cache_dir])
    if args.test_cache_dir:
        cmd.extend(["--test_cache_dir", args.test_cache_dir])

    print(json.dumps({"train_cmd": cmd}, indent=2), flush=True)
    subprocess.run(cmd, check=True)
    export_cmd = [
        sys.executable, "-u", str(EXPORT),
        "--suite_json", str(suite_json),
        "--policy_name", "global",
        "--out_json", str(policy_json),
    ]
    try:
        subprocess.run(export_cmd, check=True)
    except KeyError:
        export_cmd[export_cmd.index("global")] = "global_core8"
        subprocess.run(export_cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
