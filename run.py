#!/usr/bin/env python3
"""Run the Q-Cal main evaluation, diagnostics, or calibration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from qcal.runners.benchmark_sources import ensure_main_benchmarks
from run_main_experiment import BENCHMARKS, MODELS


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("QCAL_DATA_ROOT", ROOT / "qcal_support/datasets")).resolve()
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub")).resolve()
MODEL_IDS = {
    "qwen2vl2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2vl": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen25vl3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3vl4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
}
if tuple(MODEL_IDS) != MODELS:
    raise RuntimeError("Calibration and main-evaluation model lists differ")


def commands(workflow: str) -> list[list[str]]:
    python = sys.executable
    if workflow == "main":
        common = ["--data-root", str(DATA_ROOT), "--hf-hub-cache", str(HF_CACHE), "--resume"]
        return [
            [python, str(ROOT / "run_main_experiment.py"), *common],
            [python, str(ROOT / "baselines/evaluation/run_experiment.py"), *common],
            [python, str(ROOT / "summarize_experiment.py"), "--kind", "qcal",
             "--results-dir", str(ROOT / "outputs/main"), "--out", str(ROOT / "outputs/main_summary.json")],
            [python, str(ROOT / "summarize_experiment.py"), "--kind", "baseline",
             "--results-dir", str(ROOT / "outputs/baselines"),
             "--out", str(ROOT / "outputs/baseline_summary.json")],
        ]
    if workflow == "diagnostics":
        return [
            [python, str(ROOT / "run_prompt_invariance.py")],
            [python, str(ROOT / "run_signal_preference.py"), "--resume"],
            [python, str(ROOT / "run_depth_sweep.py"), "--resume"],
        ]
    if workflow == "calibrate":
        settings = [
            "--benchmarks", "realworldqa", "blink", "textvqa", "docvqa", "chartqa", "ai2d", "mmmu", "hallbench",
            "--budgets", "5.0", "10.0", "20.0", "33.0",
            "--calib_samples", "128", "--calib_filter_visual_grounded",
            "--calib_candidates_per_target", "12", "--train_token_pool", "stage1",
            "--stage1_frac", "0.33", "--fit_loss", "budget_overlap_pairwise",
            "--oracle_target", "gt", "--gradient_oracle_score", "directional",
            "--max_layer_frac", "0.75", "--max_active_layers", "0", "--optim_steps", "800",
            "--consistency_lambda", "0.25", "--uncertainty_lambda", "0.08",
            "--effective_layers_lambda", "0.08", "--effective_layers_target", "3.5",
            "--min_deep_weight", "0.55", "--min_deep_weight_lambda", "8.0",
        ]
        return [
            [python, str(ROOT / "qcal/runners/run_qwen_policy_train_export.py"),
             "--model_path", model_id,
             "--out_dir", str(ROOT / "outputs/calibration/stage1_50_33" / alias),
             *settings]
            for alias, model_id in MODEL_IDS.items()
        ]
    raise ValueError(f"Unknown workflow: {workflow}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=("main", "diagnostics", "calibrate"))
    parser.add_argument("--dry-run", action="store_true", help="Print the steps without running them")
    args = parser.parse_args()
    steps = commands(args.workflow)
    if args.dry_run:
        for step in steps:
            print(" ".join(step))
        return 0

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this workflow")
    env = os.environ.copy()
    env["QCAL_DATA_ROOT"] = str(DATA_ROOT)
    env["HF_HUB_CACHE"] = str(HF_CACHE)
    env["QCAL_HF_HUB_CACHE"] = str(HF_CACHE)
    env["DUALSIGNAL_STAGE1_WEIGHT"] = "0.5"
    if args.workflow == "main":
        ensure_main_benchmarks(list(BENCHMARKS), DATA_ROOT, HF_CACHE)
    for index, step in enumerate(steps, start=1):
        print(f"[{args.workflow} {index}/{len(steps)}] {' '.join(step)}", flush=True)
        subprocess.run(step, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
