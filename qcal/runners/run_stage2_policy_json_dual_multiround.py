#!/usr/bin/env python3
"""Run one multi-round benchmark with original attention and a JSON policy."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


DUALSIGNAL_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = DUALSIGNAL_DIR.parent
BEA_DIR = WORKSPACE / "qcal_support"
HF_HUB = Path(os.environ.get("TRANSFORMERS_CACHE", str(WORKSPACE / ".cache/huggingface/hub")))

sys.path.insert(0, str(DUALSIGNAL_DIR / "runners"))
sys.path.insert(0, str(BEA_DIR))
sys.path.insert(0, str(BEA_DIR / "strategy_test"))

import multi_round_benchmark as bea_mr  # noqa: E402
from run_stage2_policy_dual_multiround import _patch_qwen_wrapper_image_options  # noqa: E402
from run_stage2_policy_multiround import (  # noqa: E402
    _patch_per_question_prune_generate,
    _patch_stage1_diversity,
)
from stage2_policies import Stage2Policy, MODEL_TOTAL_LAYERS, original_attention_policy  # noqa: E402
from multiround_dataset_loaders import (  # noqa: E402
    load_clevr_multi_round,
)
from count_relaxed_scoring import relaxed_count_score  # noqa: E402
from main_benchmarks import MAIN_BENCHMARKS, install_main_loader  # noqa: E402


DEFAULT_JSON_POLICY_NAME = "selected_stage2_policy"

QWEN_ALIAS_CONFIG = {
    "qwen2vl2b": {"N": 28, "S": 0, "scoring": "single", "type": "qwen", "model_id": "Qwen/Qwen2-VL-2B-Instruct"},
    "qwen2vl": {"N": 28, "S": 0, "scoring": "single", "type": "qwen", "model_id": "Qwen/Qwen2-VL-7B-Instruct"},
    "qwen25vl3b": {"N": 36, "S": 0, "scoring": "single", "type": "qwen", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct"},
    "qwen25vl": {"N": 28, "S": 0, "scoring": "single", "type": "qwen", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
    "qwen3vl4b": {"N": 36, "S": 0, "scoring": "single", "type": "qwen", "model_id": "Qwen/Qwen3-VL-4B-Instruct"},
    "qwen3vl": {"N": 36, "S": 0, "scoring": "single", "type": "qwen", "model_id": "Qwen/Qwen3-VL-8B-Instruct"},
}


def _register_qwen_alias_configs() -> None:
    for key, cfg in QWEN_ALIAS_CONFIG.items():
        bea_mr.MODEL_CONFIG[key] = dict(cfg)


def _patch_relaxed_count_scorer() -> None:
    original_score_gqa = bea_mr.score_gqa

    def patched_score_gqa(pred, gt_answer):
        relaxed = relaxed_count_score(pred, gt_answer)
        if relaxed is not None:
            return float(relaxed)
        return original_score_gqa(pred, gt_answer)

    bea_mr.score_gqa = patched_score_gqa


def _json_policy_for(model_key: str, policy_json: Path, budget_label: str, output_name: str) -> Stage2Policy:
    payload = json.loads(policy_json.read_text())
    policy_root = payload.get("policy", payload)
    by_budget = policy_root.get("by_budget")
    if by_budget:
        if budget_label not in by_budget:
            raise ValueError(f"budget label {budget_label!r} not found in {policy_json}")
        entry = by_budget[budget_label]
    else:
        entry = policy_root
    raw_layers = tuple(int(x) for x in entry["active_layers"])
    raw_weights = tuple(float(x) for x in entry["active_weights"])
    if len(raw_layers) != len(raw_weights) or not raw_layers:
        raise ValueError(f"policy layers/weights mismatch or empty: {policy_json}")
    if any(not math.isfinite(weight) or weight < 0 for weight in raw_weights):
        raise ValueError(f"policy has invalid weights: {policy_json}")
    active = [(layer, weight) for layer, weight in zip(raw_layers, raw_weights) if weight > 0.05]
    if not active:
        raise ValueError(f"policy has no layer with weight > 0.05: {policy_json}")
    layers, weights = (tuple(values) for values in zip(*active))
    total_layers = int(entry.get("total_layers") or MODEL_TOTAL_LAYERS[model_key])
    if total_layers != MODEL_TOTAL_LAYERS[model_key]:
        raise ValueError(
            f"policy total_layers={total_layers} does not match {model_key} "
            f"total_layers={MODEL_TOTAL_LAYERS[model_key]}"
        )
    return Stage2Policy(
        name=f"{output_name}:{entry.get('name', policy_json.name)}",
        model_key=model_key,
        total_layers=total_layers,
        kind="weighted",
        layers=layers,
        weights=weights,
        source=str(policy_json),
        notes=(
            f"Loaded from budget label {budget_label}; active_weight_threshold=>0.05; "
            f"score_fusion={entry.get('score_fusion', 'unknown')}; "
            f"train_token_pool={entry.get('train_token_pool', 'unknown')}."
        ),
    )


def _policies_for(
    model_key: str,
    policy_json: Path,
    budget_label: str,
    json_policy_name: str,
    *,
    single_policy_only: bool,
):
    json_policy = _json_policy_for(model_key, policy_json, budget_label, json_policy_name)
    if single_policy_only:
        return {json_policy_name: json_policy}
    return {
        "original_attention": original_attention_policy(model_key),
        json_policy_name: json_policy,
    }


def _budget_label(value: float) -> str:
    return f"{float(value):.1f}"


def _budget_dir_name(value: float) -> str:
    return f"b{int(round(float(value))):02d}"


def _single_layer_policy_for(model_key: str, layer: int, output_name: str) -> Stage2Policy:
    total_layers = MODEL_TOTAL_LAYERS[model_key]
    if layer < 0 or layer >= total_layers:
        raise ValueError(f"single-layer control L{layer} is out of range for {model_key} ({total_layers} layers)")
    return Stage2Policy(
        name=f"{output_name}:single_layer_L{layer}",
        model_key=model_key,
        total_layers=total_layers,
        kind="single",
        layers=(int(layer),),
        weights=(1.0,),
        source="main-runner-equivalent single-layer Stage-2 control",
        notes="Uses the same main multi-round Stage-1, prompt, scoring, and per-question pruning path.",
    )


def _policies_for_budgets(
    model_key: str,
    policy_json: Path,
    budgets: list[float],
    json_policy_name: str,
    *,
    single_policy_only: bool,
    single_layer_controls: list[int] | None = None,
) -> tuple[dict[str, Stage2Policy], tuple[str, ...], dict[str, dict[str, object]]]:
    policies: dict[str, Stage2Policy] = {}
    metadata: dict[str, dict[str, object]] = {}
    names: list[str] = []
    original_policy = None if single_policy_only else original_attention_policy(model_key)
    single_layer_controls = single_layer_controls or []
    for budget in budgets:
        budget_label = _budget_label(budget)
        budget_dir = _budget_dir_name(budget)
        stage2_frac = float(budget) / 100.0
        if not single_policy_only:
            original_name = f"{budget_dir}/original_attention"
            policies[original_name] = original_policy
            metadata[original_name] = {
                "budget": float(budget),
                "budget_label": budget_label,
                "budget_dir": budget_dir,
                "stage2_frac": stage2_frac,
                "family": "original_attention",
            }
            names.append(original_name)
        for layer in single_layer_controls:
            layer_name = f"{budget_dir}/single_layer_L{int(layer)}"
            policies[layer_name] = _single_layer_policy_for(model_key, int(layer), layer_name)
            metadata[layer_name] = {
                "budget": float(budget),
                "budget_label": budget_label,
                "budget_dir": budget_dir,
                "stage2_frac": stage2_frac,
                "family": "single_layer_control",
                "layer": int(layer),
            }
            names.append(layer_name)
        json_name = f"{budget_dir}/{json_policy_name}"
        policies[json_name] = _json_policy_for(model_key, policy_json, budget_label, json_name)
        metadata[json_name] = {
            "budget": float(budget),
            "budget_label": budget_label,
            "budget_dir": budget_dir,
            "stage2_frac": stage2_frac,
            "family": json_policy_name,
        }
        names.append(json_name)
    return policies, tuple(names), metadata


def _apply_dual_config(
    model_key: str,
    policies: dict[str, Stage2Policy],
    policy_names: tuple[str, ...],
    *,
    stage2_frac: float | None,
    method_metadata: dict[str, dict[str, object]] | None = None,
):
    method_metadata = method_metadata or {}
    stage2_fracs = [
        float(method_metadata.get(name, {}).get("stage2_frac", stage2_frac or bea_mr.STAGE2_FRAC))
        for name in policy_names
    ]
    bea_mr.STAGE2_FRAC = max(stage2_fracs) if stage2_fracs else float(stage2_frac or bea_mr.STAGE2_FRAC)
    cfg = dict(bea_mr.MODEL_CONFIG[model_key])
    methods = []
    max_s = 0
    for name in policy_names:
        policy = policies[name]
        meta = method_metadata.get(name, {})
        method_stage2_frac = float(meta.get("stage2_frac", bea_mr.STAGE2_FRAC))
        max_s = max(max_s, int(policy.prune_start_layer))
        methods.append(
            {
                "output_name": name,
                "policy": policy,
                "S": int(policy.prune_start_layer),
                "stage2_frac": method_stage2_frac,
                "budget_label": meta.get("budget_label"),
                "budget": meta.get("budget"),
                "policy_json": policy.to_json(),
            }
        )

    cfg["S"] = max_s
    cfg["scoring"] = "policy"
    cfg["policy_only"] = True
    cfg["policy_methods"] = methods
    cfg["policy_name"] = (
        "json_policy_only" if len(policy_names) == 1 else "original_attention_plus_json_policy"
    )
    cfg["policy_kind"] = "single" if len(policy_names) == 1 else "dual"
    cfg["prune_start_layer"] = max_s
    cfg["cost"] = bea_mr.STAGE1_FRAC * max_s + bea_mr.STAGE2_FRAC * (cfg["N"] - max_s)
    cfg["Y"] = cfg["cost"] / cfg["N"]
    cfg["stage1_frac"] = bea_mr.STAGE1_FRAC
    cfg["stage2_frac"] = bea_mr.STAGE2_FRAC
    cfg["stage2_frac_by_method"] = {
        name: float(method_metadata.get(name, {}).get("stage2_frac", bea_mr.STAGE2_FRAC))
        for name in policy_names
    }
    cfg["budget_by_method"] = {
        name: method_metadata.get(name, {}).get("budget")
        for name in policy_names
    }
    cfg["stage1_shallow_weight"] = float(os.environ.get("DUALSIGNAL_STAGE1_WEIGHT", "0.5"))
    cfg["stage1_vit_weight"] = 1.0 - cfg["stage1_shallow_weight"]
    cfg["model_id"] = QWEN_ALIAS_CONFIG[model_key]["model_id"]
    bea_mr.MODEL_CONFIG[model_key] = cfg

    _patch_stage1_diversity()
    _patch_per_question_prune_generate(policies[policy_names[0]])
    return cfg


def _patch_dual_outputs(
    out_root: Path,
    benchmark: str,
    policies: dict[str, Stage2Policy],
    policy_names: tuple[str, ...],
    method_metadata: dict[str, dict[str, object]] | None = None,
) -> None:
    method_metadata = method_metadata or {}
    def dual_report(model_name, cfg, results, diversity=None, costs=None):
        baseline = float(np.mean(results["baseline"])) if results.get("baseline") else 0.0
        print("\n" + "=" * 60)
        report_kind = "JSON policy-only" if len(policy_names) == 1 else "JSON dual-policy"
        print(f"{report_kind} multi-round report: {model_name}")
        print("=" * 60)
        print(f"Baseline: {baseline:.3f}")
        for name in policy_names:
            score = float(np.mean(results[name])) if results.get(name) else 0.0
            preservation = score / baseline if baseline > 0 else 0.0
            print(f"{name}: {score:.3f} ({preservation:.1%} of baseline)")

    def _cost_summary(cost_rows: dict) -> dict:
        if not cost_rows or not cost_rows.get("total_tl"):
            return {"available": False}
        keys = [
            "prefill_tl", "gen_tl", "total_tl", "n_vis_decode", "n_gen_tokens",
            "shared_setup_sec", "shared_question_sec", "policy_prefill_wall_sec",
            "policy_decode_wall_sec", "policy_total_wall_sec",
        ]
        out = {"available": True}
        for key in keys:
            if key not in cost_rows:
                continue
            values = [float(x) for x in cost_rows[key]]
            out[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
            out[f"{key}_per_sample"] = values
        return out

    def dual_save(model_name, cfg, results, diversity, out_dir, costs=None):
        baseline_values = results.get("baseline", [])
        baseline = float(np.mean(baseline_values)) if baseline_values else 0.0
        costs = costs or {}
        for name in policy_names:
            policy_values = results.get(name, [])
            score = float(np.mean(policy_values)) if policy_values else 0.0
            data = {
                "model": model_name,
                "benchmark": benchmark,
                "config": {
                    k: v for k, v in cfg.items()
                    if k not in ("type", "policy_methods")
                },
                "n_samples": len(baseline_values),
                "baseline_mean": baseline,
                "baseline_per_sample": [float(x) for x in baseline_values],
                "baseline_cost": _cost_summary(costs.get("baseline", {})),
                "methods": {
                    name: {
                        "score": score,
                        "preservation": score / baseline if baseline > 0 else 0.0,
                        "per_sample": [float(x) for x in policy_values],
                        "source_internal_method": name,
                        "stage2_policy": policies[name].to_json(),
                        "cost": _cost_summary(costs.get(name, {})),
                    }
                },
                "dual_process_reuse": {
                    "model_loaded_once": True,
                    "dataset_loaded_once": True,
                    "baseline_generated_once": True,
                    "stage1_shared": True,
                    "budgets_evaluated_in_one_process": sorted(
                        {
                            float(meta["budget"])
                            for meta in method_metadata.values()
                            if meta.get("budget") is not None
                        }
                    ),
                    "vision_encoder_sees_full_image_tokens": True,
                    "stage1_pruning_point": "before LLM visual-token prefix",
                    "stage2_pruning_point": "at each policy's deepest active scoring layer",
                    "stage2_keep_scope": (
                        f"{100.0 * float(method_metadata.get(name, {}).get('stage2_frac', bea_mr.STAGE2_FRAC)):.1f}% "
                        "of full visual tokens for this method"
                    ),
                },
            }
            out_path = out_root / name / benchmark / f"{model_name}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(data, indent=2) + "\n")
            print(f"Saved -> {out_path}", flush=True)

    bea_mr.print_report = dual_report
    bea_mr.save_results = dual_save


def parse_args():
    _register_qwen_alias_configs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(bea_mr.MODEL_CONFIG.keys()))
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=[
            "textvqa",
            "gqa",
            "pope",
            "docvqa",
            "vqav2",
            "clevr",
            "visual7w",
            "gqa_grounding",
            "invig",
            "visualgenomeqa",
            "tallyqa",
            "scienceqa",
            "vsr",
        ],
    )
    parser.add_argument("--num_images", type=int, default=2000)
    parser.add_argument("--questions_per_image", type=int, default=15)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--policy_json", required=True)
    parser.add_argument("--policy_budget_label", default="10.0")
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=None,
        help="Evaluate multiple final-token budgets in this single process.",
    )
    parser.add_argument("--json_policy_name", default=DEFAULT_JSON_POLICY_NAME)
    parser.add_argument(
        "--single_policy_only",
        action="store_true",
        help="Run only the JSON policy. This keeps S equal to that policy's deepest active layer.",
    )
    parser.add_argument(
        "--single_layer_controls",
        nargs="*",
        type=int,
        default=None,
        help="Also evaluate single-layer Stage-2 controls in the same main-runner process.",
    )
    parser.add_argument(
        "--stage2_frac",
        type=float,
        default=None,
        help="Override BEA STAGE2_FRAC, e.g. 0.10 or 0.05 measured against full visual tokens.",
    )
    parser.add_argument("--skip_cclass", action="store_true")
    parser.add_argument("--resize_square", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("QWEN_ATTN_IMPLEMENTATION", "flash_attention_2")
    if args.resize_square is not None:
        os.environ["DUALSIGNAL_RESIZE_SQUARE"] = str(args.resize_square)
    else:
        os.environ.pop("DUALSIGNAL_RESIZE_SQUARE", None)
    _patch_qwen_wrapper_image_options(args.resize_square, args.max_pixels)
    _patch_relaxed_count_scorer()
    real_benchmark = args.benchmark
    bea_benchmark = real_benchmark
    if real_benchmark in MAIN_BENCHMARKS:
        bea_benchmark = install_main_loader(real_benchmark, bea_mr)
    elif real_benchmark == "clevr":
        bea_mr.load_clevr_multi_round = (
            lambda num_images, questions_per_image=15: [
                (sample["image"], [(qa["question"], qa["answer"]) for qa in sample["qas"]])
                for sample in load_clevr_multi_round(num_images, questions_per_image)
            ]
        )
    elif real_benchmark == "scienceqa":
        from benchmarks.scienceqa_loader import load_scienceqa
        bea_mr.load_vqav2_multi_round = (
            lambda num_images, questions_per_image=15: [
                (sample["image"], [(sample["question"], sample["answer"])])
                for sample in load_scienceqa(num_images)
            ]
        )
        bea_benchmark = "vqav2"
    elif real_benchmark == "vsr":
        from benchmarks.vsr_loader import load_vsr

        def _vsr_multi_round(num_images, questions_per_image=15):
            rows = []
            for sample in load_vsr(num_images):
                images = sample.get("images") or []
                if not images:
                    continue
                question = sample.get("question", "")
                if "answer yes or no" not in question.lower():
                    question = f"{question}\nAnswer yes or no."
                rows.append((images[0], [(question, sample.get("answer", ""))]))
            return rows

        bea_mr.load_vqav2_multi_round = _vsr_multi_round
        bea_benchmark = "vqav2"

    if args.budgets:
        policies, policy_names, method_metadata = _policies_for_budgets(
            args.model,
            Path(args.policy_json),
            args.budgets,
            args.json_policy_name,
            single_policy_only=args.single_policy_only,
            single_layer_controls=args.single_layer_controls,
        )
    else:
        policies = _policies_for(
            args.model,
            Path(args.policy_json),
            args.policy_budget_label,
            args.json_policy_name,
            single_policy_only=args.single_policy_only,
        )
        policy_names = (
            (args.json_policy_name,)
            if args.single_policy_only
            else ("original_attention", args.json_policy_name)
        )
        method_metadata = {
            name: {
                "budget_label": args.policy_budget_label,
                "budget": float(args.policy_budget_label),
                "budget_dir": _budget_dir_name(float(args.policy_budget_label)),
                "stage2_frac": float(args.stage2_frac)
                if args.stage2_frac is not None
                else float(args.policy_budget_label) / 100.0,
            }
            for name in policy_names
        }
    cfg = _apply_dual_config(
        args.model,
        policies,
        policy_names,
        stage2_frac=args.stage2_frac,
        method_metadata=method_metadata,
    )
    _patch_dual_outputs(Path(args.out_dir), args.benchmark, policies, policy_names, method_metadata)

    print("Using policies:", flush=True)
    for name in policy_names:
        print(f"  {name}: {policies[name].to_json()}", flush=True)
    print(
        f"Runtime cfg: max_S={cfg['S']} stage2_frac={bea_mr.STAGE2_FRAC:.4f} "
        f"Y={cfg['Y']:.4f}",
        flush=True,
    )

    sys.argv = [
        "multi_round_benchmark.py",
        "--model", args.model,
        "--benchmark", bea_benchmark,
        "--num_images", str(args.num_images),
        "--questions_per_image", str(args.questions_per_image),
        "--out_dir", str(args.out_dir),
    ]
    if args.skip_cclass:
        sys.argv.append("--skip_cclass")
    bea_mr.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
