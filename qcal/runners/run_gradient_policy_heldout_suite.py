#!/usr/bin/env python3
"""Held-out benchmark-specific vs global gradient-aligned policy suite.

No prior sweep, validation, strategy, or calibration artifacts are read.  The
script fits:

- one global policy from the union of all calibration benchmarks;
- one benchmark-specific policy per benchmark.

Both are evaluated only on held-out samples.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import re
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from qwen_vl_utils import process_vision_info


DUALSIGNAL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = DUALSIGNAL_ROOT.parent
BEA_DIR = WORKSPACE / "qcal_support"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_OUT = DUALSIGNAL_ROOT / "results/gradient_aligned_heldout_suite"

sys.path.insert(0, str(DUALSIGNAL_ROOT / "runners"))
sys.path.insert(0, str(BEA_DIR))
sys.path.insert(0, str(BEA_DIR / "strategy_test"))
sys.path.insert(0, str(WORKSPACE))

from model_backends import compute_text_vis_attention, load_backend, norm01_t, qwen_rotary_cross_attention_scores  # noqa: E402
from diagnostic_metrics import diagnostic_scores  # noqa: E402
from multi_round_benchmark import _qwen_vit_importance  # noqa: E402
from baselines.evaluation.run_broad_comparison import eval_metric as bea_eval_metric, load_split as bea_load_split, make_prompt as bea_make_prompt, max_new_tokens_map as bea_max_new_tokens_map  # noqa: E402
from baselines.evaluation.run_qwen_original import budget_fraction, budget_specs, maybe_resize, normalize_benchmark  # noqa: E402
from run_stage2_policy_multiround import _dual_signal_stage1_select_idselection_diversity  # noqa: E402
from fit_attention_policy_from_scratch import (  # noqa: E402
    allocate,
    fit_weights,
    fuse_policy_scores,
    fuse_policy_scores_with_consistency,
    generate_ids_and_text,
    gradient_oracle,
    layer_attention_scores,
    make_policy,
    prepare_inputs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks", nargs="+", default=["chartqa", "textvqa", "ocrbench"])
    p.add_argument("--budgets", nargs="+", type=float, default=[10.0, 20.0, 25.0])
    p.add_argument("--calib_samples", type=int, default=64)
    p.add_argument("--calib_offset", type=int, default=0)
    p.add_argument("--calib_filter_baseline_correct", action="store_true",
                   help="Use only calibration samples where the full-token baseline answer scores above the threshold.")
    p.add_argument("--calib_filter_visual_grounded", action="store_true",
                   help="Use only calibration samples where full-image baseline is positive and no-visual pruning degrades the score.")
    p.add_argument("--calib_min_baseline_score", type=float, default=1e-9)
    p.add_argument("--require_calib_samples", action="store_true",
                   help="Fail if filtering cannot collect calib_samples records for every benchmark.")
    p.add_argument("--calib_candidates_per_target", type=int, default=8,
                   help="When filtering calibration by baseline correctness, draw this many candidates per requested calibration sample.")
    p.add_argument("--calibration_indices_manifest", default=None,
                   help="JSON manifest with fixed per-benchmark calibration sample indices. When set, these samples are used directly without visual-grounded re-screening.")
    p.add_argument("--eval_samples", type=int, default=64)
    p.add_argument("--eval_offset", type=int, default=64)
    p.add_argument("--num_sweep", type=int, default=256)
    p.add_argument("--model_path", default=str(DEFAULT_MODEL))
    p.add_argument("--backend_type", choices=["auto", "qwen", "llava", "internvl"], default="auto")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--max_pixels", type=int, default=1016064)
    p.add_argument("--multi_max_pixels", type=int, default=262144)
    p.add_argument("--resize_square", type=int, default=1008)
    p.add_argument("--max_active_layers", type=int, default=5,
                   help="Maximum active layers after fitting. Use 0 to keep every layer above --active_weight_threshold.")
    p.add_argument("--active_weight_threshold", type=float, default=1e-8,
                   help="Final policy layers below this normalized weight are treated as inactive.")
    p.add_argument("--max_layer", type=int, default=None,
                   help="Explicit deepest language-model layer allowed, inclusive. Overrides --max_layer_frac.")
    p.add_argument("--max_layer_frac", type=float, default=0.75,
                   help="Default candidate depth as a fraction of total layers. 0.75 uses the first 75%% of layers.")
    p.add_argument("--train_token_pool", choices=["full", "stage1"], default="full",
                   help="Token universe used for policy training/evaluation. stage1 restricts to Stage1 kept visual tokens.")
    p.add_argument("--stage1_frac", type=float, default=0.33)
    p.add_argument("--stage1_gamma", type=float, default=20.0)
    p.add_argument("--global_sets", nargs="*", default=[],
                   help="Named global calibration sets, e.g. c=chartqa ct=chartqa,textvqa all=chartqa,textvqa,ocrbench.")
    p.add_argument("--global_only", action="store_true",
                   help="Fit/evaluate only global policies, skipping benchmark-specific policies.")
    p.add_argument("--cache_only", action="store_true",
                   help="Only collect/reuse calibration and test caches, then exit before fitting or validation.")
    p.add_argument("--rank_temperature", type=float, default=0.12)
    p.add_argument("--oracle_temperature", type=float, default=0.25)
    p.add_argument("--depth_lambda", type=float, default=0.01)
    p.add_argument("--entropy_lambda", type=float, default=0.001)
    p.add_argument("--l2_lambda", type=float, default=0.001)
    p.add_argument("--optim_steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--fit_loss", choices=["kl", "topk_pairwise", "budget_overlap", "budget_overlap_pairwise"], default="topk_pairwise")
    p.add_argument("--shared_budget_policy", action="store_true",
                   help="Fit one shared policy using all requested percent budgets instead of one policy per budget.")
    p.add_argument("--consistency_lambda", type=float, default=0.0,
                   help="Positive boost for layer-consensus token scores during fitting and validation.")
    p.add_argument("--uncertainty_lambda", type=float, default=0.0,
                   help="Positive penalty for layer-disagreement token scores during fitting and validation.")
    p.add_argument("--consistency_eps", type=float, default=1e-4)
    p.add_argument("--consistency_mode", choices=["weighted", "static"], default="weighted",
                   help="weighted uses layer-weight-dependent disagreement; static computes token disagreement across layers independent of weights.")
    p.add_argument("--effective_layers_lambda", type=float, default=0.0,
                   help="Penalty weight that discourages collapse below --effective_layers_target.")
    p.add_argument("--effective_layers_target", type=float, default=1.0,
                   help="Target effective number of layers, defined as 1/sum(w^2).")
    p.add_argument("--min_deep_layer_frac", type=float, default=0.0,
                   help="Layer-depth threshold for deep-mass regularization, as (layer+1)/total_layers.")
    p.add_argument("--min_deep_weight", type=float, default=0.0,
                   help="Minimum total simplex weight required on layers at or beyond --min_deep_layer_frac.")
    p.add_argument("--min_deep_weight_lambda", type=float, default=0.0,
                   help="Penalty weight for violating --min_deep_weight.")
    p.add_argument("--max_pairwise_negatives", type=int, default=256)
    p.add_argument("--fit_device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max_answer_tokens_for_grad", type=int, default=32)
    p.add_argument("--oracle_target", choices=["baseline", "gt"], default="baseline",
                   help="Use generated baseline answer tokens or sample ground-truth answer tokens for gradient oracle.")
    p.add_argument("--oracle_mode", choices=["gradient", "simplified_insertion"], default="gradient")
    p.add_argument("--gradient_oracle_score", choices=["sensitivity", "directional"], default="sensitivity",
                   help="sensitivity uses ||dL/dv_i||*||v_i||; directional uses max(0, -<dL/dv_i, v_i>) to score tokens that push the target answer.")
    p.add_argument("--oracle_clip_quantile", type=float, default=1.0,
                   help="Clip oracle token scores at this per-sample quantile before fitting. 1.0 disables clipping.")
    p.add_argument("--oracle_score_transform", choices=["none", "log1p", "rank"], default="none",
                   help="Per-sample oracle transform after clipping; rank maps scores to [0,1] by rank.")
    p.add_argument("--oracle_candidate_tokens", type=int, default=96)
    p.add_argument("--oracle_insertion_groups", type=int, default=8)
    p.add_argument("--oracle_insertion_weight", type=float, default=0.4)
    p.add_argument("--oracle_visual_contrast_weight", type=float, default=0.4)
    p.add_argument("--oracle_gradient_weight", type=float, default=0.2)
    p.add_argument("--oracle_interaction_weight", type=float, default=0.0,
                   help="Extra weight for insertion_gain * visual_contrast after component normalization.")
    p.add_argument("--oracle_component_transform", choices=["minmax", "rank"], default="minmax",
                   help="Per-component normalization before simplified-insertion oracle mixing.")
    p.add_argument("--oracle_attribution_mode", choices=["group_insertion", "group_leaveout", "token_leaveout", "budget_token_leaveout"],
                   default="group_insertion",
                   help="How simplified-insertion assigns insertion credit to candidate visual tokens.")
    p.add_argument("--oracle_budget_attribution_fracs", default="10,5",
                   help="Comma-separated budget percentages for budget_token_leaveout relative seed sets.")
    p.add_argument("--oracle_budget_attribution_weights", default="0.5,0.5",
                   help="Comma-separated weights for budget_token_leaveout contributions.")
    p.add_argument("--oracle_insertion_score_mode", choices=["auto", "logprob", "generation"], default="auto")
    p.add_argument("--calibration_cache_dir", default=None,
                   help="Directory for cached calibration oracle/attention records. Defaults to out_dir/calibration_cache.")
    p.add_argument("--no_reuse_calibration_cache", action="store_true",
                   help="Always recollect calibration records even when a matching cache file exists.")
    p.add_argument("--test_cache_dir", default=None,
                   help="Directory for cached held-out test baseline/attention records. Defaults to out_dir/test_cache.")
    p.add_argument("--no_reuse_test_cache", action="store_true",
                   help="Always recollect held-out test records even when a matching cache file exists.")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else WORKSPACE / p


def parse_global_sets(items: list[str], benchmarks: list[str]) -> dict[str, list[str]]:
    if not items:
        return {"global": benchmarks}
    out: dict[str, list[str]] = {}
    valid = set(benchmarks)
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected name=bench1,bench2 global set, got {item!r}")
        name, raw = item.split("=", 1)
        members = [normalize_benchmark(x.strip()) for x in raw.split(",") if x.strip()]
        unknown = [x for x in members if x not in valid]
        if unknown:
            raise ValueError(f"Global set {name!r} has benchmarks not in --benchmarks: {unknown}")
        if not members:
            raise ValueError(f"Global set {name!r} is empty")
        out[name] = members
    return out


def load_samples_ext(benchmark: str, num_samples: int) -> list[dict[str, Any]]:
    if benchmark == "blink_heldout":
        from benchmarks.blink_heldout_loader import load_blink_heldout
        return load_blink_heldout(num_samples)
    if benchmark == "hallbench":
        from benchmarks.hallbench_loader import load_hallbench
        return load_hallbench(num_samples)
    if benchmark == "dvqa":
        from benchmarks.dvqa_loader import load_dvqa
        return load_dvqa(num_samples)
    if benchmark == "infographicvqa":
        from benchmarks.infographicvqa_loader import load_infographicvqa
        return load_infographicvqa(num_samples)
    if benchmark == "hpope":
        from benchmarks.hpope_loader import load_hpope
        return load_hpope(num_samples)
    if benchmark == "blink":
        from benchmarks.blink_loader import load_blink
        return load_blink(num_samples)
    if benchmark == "muirbench":
        from benchmarks.muirbench_loader import load_muirbench
        return load_muirbench(num_samples)
    if benchmark == "mmiu":
        from benchmarks.mmiu_loader import load_mmiu
        return load_mmiu(num_samples)
    if benchmark == "mantis":
        from benchmarks.mantis_loader import load_mantis_eval
        return load_mantis_eval(num_samples)
    if benchmark == "mathvista":
        ds = load_dataset("AI4Math/MathVista", split="testmini")
        return [{
            "question": item.get("query") or item.get("question", ""),
            "image": item.get("decoded_image"),
            "answer": item.get("answer", ""),
            "choices": item.get("choices"),
            "question_type": item.get("question_type", ""),
            "answer_type": item.get("answer_type", ""),
            "metadata": item.get("metadata", {}),
        } for item in ds.select(range(min(num_samples, len(ds))))]
    if benchmark == "mmstar":
        ds = load_dataset("Lin-Chen/MMStar", split="val")
        return [{
            "question": item.get("question", ""),
            "image": item.get("image"),
            "answer": item.get("answer", ""),
            "category": item.get("category", ""),
            "l2_category": item.get("l2_category", ""),
        } for item in ds.select(range(min(num_samples, len(ds))))]
    if benchmark == "mmvet":
        ds = load_dataset("lmms-lab/MMVet", split="test")
        return [{
            "question": item.get("question", ""),
            "image": item.get("image"),
            "answer": item.get("answer", ""),
            "capability": item.get("capability", ""),
        } for item in ds.select(range(min(num_samples, len(ds))))]
    return bea_load_split(benchmark, split="eval", num_eval=num_samples, num_sweep=0)


def load_split_ext(benchmark: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fixed_indices = fixed_calibration_indices(args, benchmark)
    if fixed_indices is not None:
        max_idx = max(fixed_indices) if fixed_indices else -1
        requested = max(max_idx + 1, args.eval_offset + args.eval_samples) + args.num_sweep
        samples = load_samples_ext(benchmark, requested)
        loaded = len(samples)
        calib = []
        missing = []
        for idx in fixed_indices[: args.calib_samples]:
            if 0 <= idx < loaded:
                item = dict(samples[idx])
                item["__source_index"] = int(idx)
                calib.append(item)
            else:
                missing.append(int(idx))
        eval_start = args.eval_offset if loaded >= args.eval_offset + args.eval_samples else min(loaded, len(calib))
        eval_samples = samples[eval_start:eval_start + args.eval_samples]
        return calib, eval_samples, {
            "requested": requested,
            "loaded": loaded,
            "calib_start": "fixed_manifest",
            "calib_candidate_count": len(fixed_indices),
            "calib_target_count": args.calib_samples,
            "fixed_calibration_indices_manifest": str(args.calibration_indices_manifest),
            "fixed_calibration_indices_used": [int(x) for x in fixed_indices[: args.calib_samples]],
            "fixed_calibration_indices_missing": missing,
            "eval_start": eval_start,
            "eval_count": len(eval_samples),
        }
    calib_candidates = args.calib_samples
    if args.calib_filter_baseline_correct or args.calib_filter_visual_grounded:
        calib_candidates = args.calib_samples * max(1, args.calib_candidates_per_target)
    requested = max(args.calib_offset + calib_candidates, args.eval_offset + args.eval_samples) + args.num_sweep
    samples = load_samples_ext(benchmark, requested)
    loaded = len(samples)
    calib_start = args.calib_offset if loaded >= args.calib_offset + calib_candidates else 0
    eval_start = args.eval_offset if loaded >= args.eval_offset + args.eval_samples else min(loaded, calib_start + args.calib_samples)
    calib = samples[calib_start:calib_start + calib_candidates]
    eval_samples = samples[eval_start:eval_start + args.eval_samples]
    return calib, eval_samples, {
        "requested": requested,
        "loaded": loaded,
        "calib_start": calib_start,
        "calib_candidate_count": len(calib),
        "calib_target_count": args.calib_samples,
        "eval_start": eval_start,
        "eval_count": len(eval_samples),
    }


def fixed_calibration_indices(args: argparse.Namespace, benchmark: str) -> list[int] | None:
    manifest_path = getattr(args, "calibration_indices_manifest", None)
    if not manifest_path:
        return None
    payload = json.loads(resolve(manifest_path).read_text(encoding="utf-8"))
    indices_by_benchmark = payload.get("indices", payload)
    raw = indices_by_benchmark.get(normalize_benchmark(benchmark))
    if raw is None:
        raise KeyError(f"fixed calibration manifest has no indices for benchmark {benchmark!r}: {manifest_path}")
    return [int(x) for x in raw]


def sample_images(sample: dict[str, Any], resize_square: int | None) -> list[Image.Image]:
    if "images" in sample and sample.get("images") is not None:
        raw = sample.get("images")
        imgs = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    elif "image" in sample:
        imgs = [sample.get("image")]
    else:
        raise ValueError("sample has neither image nor images")
    out = []
    for img in imgs:
        if isinstance(img, Image.Image):
            im = img.convert("RGB")
        elif isinstance(img, str):
            im = Image.open(img).convert("RGB")
        else:
            raise ValueError(f"unsupported image object: {type(img)}")
        out.append(maybe_resize(im, resize_square))
    return out


def make_prompt(benchmark: str, sample: dict[str, Any]) -> str:
    if benchmark in ("blink", "blink_heldout"):
        return sample.get("question", "") + "\nAnswer with the option letter only."
    if benchmark in ("dvqa", "infographicvqa"):
        return sample.get("question", "") + "\nAnswer with a short answer."
    if benchmark == "hpope":
        q = sample.get("question", "")
        if "yes or no" not in q.lower():
            q = f"{q}\nAnswer yes or no."
        return q
    if benchmark in ("muirbench", "mmiu", "mantis"):
        q = sample.get("question", "")
        options = sample.get("options", "")
        if isinstance(options, list):
            options = "\n".join(f"{chr(65+i)}. {x}" for i, x in enumerate(options))
        if options and options not in q:
            q = f"{q}\n{options}"
        return q + "\nAnswer with the option letter or short answer."
    if benchmark in ("mathvista", "mmstar"):
        return sample.get("question", "") + "\nAnswer with the option letter or final short answer."
    if benchmark == "mmvet":
        return sample.get("question", "") + "\nAnswer with a short answer."
    return bea_make_prompt(benchmark, sample)


def prepare_inputs_any(model, images: list[Image.Image], prompt: str,
                       max_pixels: int | None, multi_max_pixels: int | None):
    if hasattr(model, "prepare_inputs_any"):
        return model.prepare_inputs_any(images, prompt, max_pixels, multi_max_pixels)
    content = []
    for image in images:
        entry = {"type": "image", "image": image}
        entry["max_pixels"] = max_pixels if len(images) == 1 else multi_max_pixels
        content.append(entry)
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return model.processor(text=[text], images=image_inputs, return_tensors="pt").to(model.model.device)


def vit_importance_any(model, inputs, n_vis: int) -> np.ndarray | None:
    if not hasattr(model, "model"):
        return None
    visual_model = getattr(getattr(model.model, "model", None), "visual", None)
    if visual_model is None:
        return None
    pixel_values = inputs.get("pixel_values")
    grid_thw = inputs.get("image_grid_thw")
    if pixel_values is None or grid_thw is None:
        return None
    return _qwen_vit_importance(visual_model, pixel_values, grid_thw, n_vis)


def stage1_keep_mask(model, inputs, vis_embeds, n_vis: int, stage1_frac: float, gamma: float) -> tuple[list[bool], dict[str, Any]]:
    k = max(1, min(n_vis, int(n_vis * float(stage1_frac))))
    l0 = model.layer_attention_scores(inputs, [0]).get(0, [])
    if len(l0) != n_vis:
        raise RuntimeError(f"L0 attention length mismatch: got {len(l0)}, expected {n_vis}")
    vit = None
    try:
        vit = vit_importance_any(model, inputs, n_vis)
    except Exception:
        vit = None
    if vit is not None and len(vit) == n_vis:
        selected = _dual_signal_stage1_select_idselection_diversity(
            np.asarray(l0, dtype=np.float64),
            np.asarray(vit, dtype=np.float64),
            vis_embeds,
            k,
            gamma=gamma,
        )
        source = "l0_vit_idselection_gaussian_diversity"
    else:
        selected = np.argsort(np.asarray(l0, dtype=np.float64))[::-1][:k].tolist()
        source = "l0_topk_vit_unavailable"
    keep = set(int(x) for x in selected)
    return [i in keep for i in range(n_vis)], {
        "mask_source": source,
        "stage1_frac": float(stage1_frac),
        "stage1_gamma": float(gamma),
    }


def restrict_scores_to_indices(
    layer_scores: dict[str, list[float]],
    oracle: list[float] | None,
    original_indices: list[int] | None,
) -> tuple[dict[str, list[float]], list[float] | None]:
    if original_indices is None:
        return layer_scores, oracle
    restricted_scores = {
        str(layer): [float(scores[i]) for i in original_indices]
        for layer, scores in layer_scores.items()
    }
    restricted_oracle = None
    if oracle is not None:
        restricted_oracle = [float(oracle[i]) for i in original_indices]
    return restricted_scores, restricted_oracle


def mask_scores_to_token_pool(
    layer_scores: dict[str, list[float]],
    oracle: list[float] | None,
    keep_mask: list[bool] | None,
    masked_score: float = -1.0e9,
) -> tuple[dict[str, list[float]], list[float] | None]:
    if keep_mask is None:
        return layer_scores, oracle
    masked_scores = {
        str(layer): [
            float(value) if bool(keep_mask[i]) else float(masked_score)
            for i, value in enumerate(scores)
        ]
        for layer, scores in layer_scores.items()
    }
    masked_oracle = None
    if oracle is not None:
        masked_oracle = [
            float(value) if bool(keep_mask[i]) else 0.0
            for i, value in enumerate(oracle)
        ]
    return masked_scores, masked_oracle


def full_mask_from_pool_mask(pool_mask: list[bool], original_indices: list[int] | None, n_full: int) -> list[bool]:
    if original_indices is None:
        return list(pool_mask)
    full = [False] * n_full
    for keep, idx in zip(pool_mask, original_indices):
        if keep:
            full[int(idx)] = True
    return full


def stage1_pool_budget_fraction(spec: dict[str, Any], n_full_vis: int, n_pool_vis: int) -> float:
    if n_pool_vis <= 0:
        return 0.0
    if spec["type"] == "percent":
        target = int(n_full_vis * max(0.0, float(spec["value"]) / 100.0))
    else:
        target = int(budget_fraction(spec, n_full_vis) * n_full_vis)
    target = max(1, min(n_pool_vis, target))
    return float(target) / float(n_pool_vis)


def spec_for_pool_budget(spec: dict[str, Any], n_full_vis: int, n_pool_vis: int) -> dict[str, Any]:
    if n_pool_vis <= 0:
        return dict(spec)
    adjusted = dict(spec)
    adjusted["type"] = "percent"
    adjusted["value"] = 100.0 * stage1_pool_budget_fraction(spec, n_full_vis, n_pool_vis)
    return adjusted


def pct_fracs_for_pool(specs: list[dict[str, Any]], args: argparse.Namespace,
                       only_label: str | None = None) -> list[float]:
    out = []
    denom = max(float(getattr(args, "stage1_frac", 1.0)), 1e-6)
    for spec in specs:
        if spec["type"] != "percent":
            continue
        if only_label is not None and spec["label"] != only_label:
            continue
        frac = max(0.0, min(1.0, float(spec["value"]) / 100.0))
        if str(getattr(args, "train_token_pool", "full")) == "stage1":
            frac = max(0.0, min(1.0, frac / denom))
        out.append(frac)
    return out


@torch.no_grad()
def stage1_pruned_layer_attention_scores(
    model,
    inputs,
    vis_embeds,
    grid_thw,
    stage1_mask: list[bool],
    layers: list[int],
    query_mode: str = "last_text",
) -> dict[str, list[float]]:
    if not all(hasattr(model, name) for name in ("_manipulated_inputs", "_embeds_from_manipulation", "visual_and_text_positions")):
        raise RuntimeError("stage1 token pool requires a Qwen-like backend with token manipulation helpers")
    mr = model._manipulated_inputs(inputs, vis_embeds, grid_thw, stage1_mask)
    pruned_inputs = {
        "input_ids": mr["new_input_ids"],
        "attention_mask": mr.get("new_attention_mask"),
        "position_ids": mr.get("position_ids"),
        "mm_token_type_ids": mr.get("new_mm_token_type_ids"),
        "image_grid_thw": mr.get("new_image_grid_thw", inputs.get("image_grid_thw")),
        "video_grid_thw": inputs.get("video_grid_thw"),
    }
    pruned_inputs = {k: v for k, v in pruned_inputs.items() if v is not None}
    vis_pos, text_pos = model.visual_and_text_positions(pruned_inputs)
    if not vis_pos or not text_pos:
        return {str(layer): [] for layer in layers}
    model_inputs = {
        **pruned_inputs,
        "inputs_embeds": model._embeds_from_manipulation(mr),
        "return_dict": True,
        "use_cache": False,
    }
    model_inputs.pop("input_ids", None)
    captured = {}
    handles = []

    def make_hook(layer_idx: int):
        def _pre_hook(_module, hook_args):
            if isinstance(hook_args, tuple) and hook_args:
                h = hook_args[0]
                captured[layer_idx] = h.detach() if h.dim() == 3 else h.unsqueeze(0).detach()
        return _pre_hook

    try:
        for layer_idx in layers:
            handles.append(model.lm.layers[layer_idx].register_forward_pre_hook(make_hook(layer_idx)))
        model.model(**model_inputs, output_hidden_states=False, output_attentions=False)
    finally:
        for handle in handles:
            handle.remove()
    out = {}
    for layer_idx in layers:
        h = captured.get(layer_idx)
        if h is None:
            out[str(layer_idx)] = [0.0 for _ in vis_pos]
        else:
            if "qwen3" in model.lm.layers[layer_idx].self_attn.__class__.__module__.lower():
                q_pos = [text_pos[-1]] if query_mode == "last_text" else text_pos
                position_embeddings = None
                position_ids = pruned_inputs.get("position_ids")
                if position_ids is not None and hasattr(model.lm, "rotary_emb"):
                    device = next(model.lm.parameters()).device
                    dummy = torch.zeros(
                        1,
                        position_ids.shape[-1],
                        model.lm.config.hidden_size,
                        device=device,
                        dtype=next(model.lm.parameters()).dtype,
                    )
                    cos, sin = model.lm.rotary_emb(dummy, position_ids.to(device))
                    position_embeddings = (cos, sin)
                mrope_section = getattr(model.lm.config, "rope_parameters", {}).get("mrope_section", None)
                scores = qwen_rotary_cross_attention_scores(
                    h,
                    q_pos,
                    vis_pos,
                    model.lm.layers[layer_idx].self_attn,
                    layer_module=model.lm.layers[layer_idx],
                    position_embeddings=position_embeddings,
                    mrope_section=mrope_section,
                )
            else:
                scores = compute_text_vis_attention(
                    model.lm.layers[layer_idx], h, text_pos, vis_pos, pruned_inputs,
                    model.lm, apply_rope=True, query_mode=query_mode)
            out[str(layer_idx)] = norm01_t(scores).tolist()
    return out


def stage1_pruned_gradient_oracle(
    model,
    inputs,
    vis_embeds,
    grid_thw,
    stage1_mask: list[bool],
    answer_ids: torch.Tensor,
    max_answer_tokens: int,
    score_mode: str = "sensitivity",
) -> list[float]:
    if not all(hasattr(model, name) for name in ("_manipulated_inputs", "_embeds_from_manipulation")):
        raise RuntimeError("stage1 token pool requires a Qwen-like backend with token manipulation helpers")
    answer_ids = answer_ids[:max_answer_tokens].to(inputs["input_ids"].device)
    if answer_ids.numel() == 0:
        raise ValueError("empty answer ids")
    model.model.zero_grad(set_to_none=True)
    mr = model._manipulated_inputs(inputs, vis_embeds, grid_thw, stage1_mask)
    new_ids = mr["new_input_ids"]
    visual_embeds = mr["new_visual_embeds"].detach().clone().requires_grad_(True)
    text_embeds = model.lm.get_input_embeddings()(new_ids).detach()
    prompt_text = text_embeds.clone()
    if visual_embeds.shape[0] > 0:
        image_mask, _ = model.model.model.get_placeholder_mask(
            new_ids, inputs_embeds=prompt_text, image_features=visual_embeds)
        prompt_embeds = prompt_text.masked_scatter(image_mask, visual_embeds)
    else:
        prompt_embeds = prompt_text
    answer_embeds = model.lm.get_input_embeddings()(answer_ids.unsqueeze(0)).to(prompt_embeds.dtype)
    inputs_embeds = torch.cat([prompt_embeds, answer_embeds], dim=1)
    attention_mask = mr.get("new_attention_mask")
    if attention_mask is not None:
        extra = torch.ones((attention_mask.shape[0], answer_ids.numel()),
                           dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, extra], dim=1)
    mm_token_type_ids = mr.get("new_mm_token_type_ids")
    if mm_token_type_ids is not None:
        extra_mm = torch.zeros((mm_token_type_ids.shape[0], answer_ids.numel()),
                               dtype=mm_token_type_ids.dtype, device=mm_token_type_ids.device)
        mm_token_type_ids = torch.cat([mm_token_type_ids, extra_mm], dim=1)
    model_inputs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "image_grid_thw": mr.get("new_image_grid_thw", inputs.get("image_grid_thw")),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "return_dict": True,
        "use_cache": False,
    }
    model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
    logits = model.model(**model_inputs).logits
    prompt_len = prompt_embeds.shape[1]
    answer_len = answer_ids.numel()
    pred_logits = logits[:, prompt_len - 1: prompt_len + answer_len - 1, :].float()
    target = answer_ids[:answer_len].to(pred_logits.device).unsqueeze(0)
    loss = F.cross_entropy(pred_logits.reshape(-1, pred_logits.shape[-1]),
                           target.reshape(-1), reduction="mean")
    loss.backward()
    grad = visual_embeds.grad
    if grad is None:
        raise RuntimeError("visual embedding gradient is None")
    if score_mode == "directional":
        imp = torch.clamp(-(grad.float() * visual_embeds.detach().float()).sum(dim=-1), min=0.0)
    else:
        imp = grad.float().norm(dim=-1) * visual_embeds.detach().float().norm(dim=-1)
    model.model.zero_grad(set_to_none=True)
    return imp.detach().cpu().tolist()


def max_new_tokens_map(benchmark: str) -> int:
    if benchmark in ("dvqa", "infographicvqa"):
        return 64
    if benchmark in ("blink_heldout", "hpope"):
        return 16
    if benchmark in ("mmvet", "mathvista"):
        return 64
    return bea_max_new_tokens_map(benchmark)


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def comparable_answer_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).strip().lower()).strip()


def gt_answer_text(sample: dict[str, Any]) -> str:
    answer = ""
    for key in ("answer", "answers", "gt_answer", "gt_answers", "label", "labels"):
        value = sample.get(key)
        if value:
            answer = value
            break
    if not answer and isinstance(sample.get("annotation"), dict):
        ann = sample["annotation"]
        for key in ("answer", "answers", "gt_answer", "gt_answers"):
            value = ann.get(key)
            if value:
                answer = value
                break
    if isinstance(answer, (list, tuple)):
        cleaned = [
            str(x).split("<and>")[0].split("<AND>")[0].strip()
            for x in answer
            if str(x).strip()
        ]
        answer = Counter(cleaned).most_common(1)[0][0] if cleaned else ""
    text = str(answer).split("<and>")[0].split("<AND>")[0].strip()
    return text


def tokenize_answer(model, text: str, device) -> torch.Tensor:
    tokenizer = getattr(getattr(model, "processor", None), "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("model backend has no tokenizer for GT oracle target")
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    return ids.to(device)


def norm01_list(values: list[float]) -> list[float]:
    if not values:
        return []
    vals = [0.0 if not isinstance(v, (int, float)) or math.isnan(float(v)) else float(v) for v in values]
    lo, hi = min(vals), max(vals)
    if hi - lo <= 1e-12:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def rank01_list(values: list[float]) -> list[float]:
    if not values:
        return []
    vals = [0.0 if not isinstance(v, (int, float)) or math.isnan(float(v)) else float(v) for v in values]
    order = sorted(range(len(vals)), key=lambda i: (vals[i], -i))
    ranks = [0.0] * len(vals)
    denom = max(1, len(vals) - 1)
    for rank, idx in enumerate(order):
        ranks[idx] = rank / denom
    return ranks


def normalize_oracle_component(values: list[float], mode: str) -> list[float]:
    if mode == "rank":
        return rank01_list(values)
    return norm01_list(values)


def parse_float_list_csv(text: str, default: list[float]) -> list[float]:
    values: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values or list(default)


def transform_oracle_scores(oracle: list[float], args: argparse.Namespace) -> tuple[list[float], dict[str, Any]]:
    values = [max(0.0, float(v)) for v in oracle]
    if not values:
        return values, {
            "oracle_clip_quantile": float(args.oracle_clip_quantile),
            "oracle_score_transform": str(args.oracle_score_transform),
        }
    clipped = 0
    q = float(args.oracle_clip_quantile)
    clip_value = None
    if 0.0 < q < 1.0:
        sorted_values = sorted(values)
        idx = max(0, min(len(sorted_values) - 1, int(math.ceil(q * len(sorted_values))) - 1))
        clip_value = float(sorted_values[idx])
        values = [min(v, clip_value) for v in values]
        clipped = sum(1 for v in oracle if float(v) > clip_value)
    mode = str(args.oracle_score_transform)
    if mode == "log1p":
        values = [math.log1p(v) for v in values]
    elif mode == "rank":
        order = sorted(range(len(values)), key=lambda i: (values[i], -i))
        ranks = [0.0] * len(values)
        denom = max(1, len(values) - 1)
        for rank, idx in enumerate(order):
            ranks[idx] = rank / denom
        values = ranks
    return values, {
        "oracle_clip_quantile": q,
        "oracle_clip_value": clip_value,
        "oracle_clipped_count": int(clipped),
        "oracle_score_transform": mode,
    }


def insertion_score(
    model,
    benchmark: str,
    sample: dict[str, Any],
    inputs,
    vis_embeds,
    grid_thw,
    keep_mask: list[bool],
    answer_ids: torch.Tensor,
    max_answer_tokens: int,
    max_tok: int,
    mode: str,
) -> tuple[float, str]:
    if mode in ("auto", "logprob") and hasattr(model, "pruned_answer_logprob"):
        try:
            return float(model.pruned_answer_logprob(
                inputs, vis_embeds, grid_thw, keep_mask, answer_ids, max_answer_tokens)), "logprob"
        except Exception as exc:
            if mode == "logprob":
                raise
            print(f"  insertion logprob fallback to generation: {type(exc).__name__}: {exc}", flush=True)
    pred = model.run_pruned(inputs, vis_embeds, grid_thw, keep_mask, max_tok)
    return float(eval_metric(benchmark, pred, sample)), "generation"


def simplified_insertion_oracle(
    model,
    benchmark: str,
    sample: dict[str, Any],
    inputs,
    vis_embeds,
    grid_thw,
    answer_ids: torch.Tensor,
    layer_scores: dict[str, list[float]],
    layers: list[int],
    args: argparse.Namespace,
    max_tok: int,
) -> tuple[list[float], dict[str, Any]]:
    n_vis = int(vis_embeds.shape[0])
    uniform = [1.0 / max(1, len(layers)) for _ in layers]
    contrast_scores = fuse_policy_scores_with_consistency(
        layer_scores,
        layers,
        uniform,
        n_vis,
        consistency_lambda=args.consistency_lambda,
        uncertainty_lambda=args.uncertainty_lambda,
        consistency_eps=args.consistency_eps,
    )
    candidate_count = max(1, min(n_vis, int(args.oracle_candidate_tokens)))
    candidate_order = sorted(range(n_vis), key=lambda i: float(contrast_scores[i]), reverse=True)[:candidate_count]
    group_count = max(1, min(candidate_count, int(args.oracle_insertion_groups)))
    groups = [[] for _ in range(group_count)]
    for pos, token_idx in enumerate(candidate_order):
        groups[pos % group_count].append(int(token_idx))

    insertion = [0.0] * n_vis
    group_scores = []
    token_scores = []
    attribution_mode = str(args.oracle_attribution_mode)
    empty_mask = [False] * n_vis
    base_score, score_mode = insertion_score(
        model, benchmark, sample, inputs, vis_embeds, grid_thw, empty_mask,
        answer_ids, args.max_answer_tokens_for_grad, max_tok, args.oracle_insertion_score_mode)
    if attribution_mode == "group_insertion":
        for group in groups:
            keep = [False] * n_vis
            for idx in group:
                keep[idx] = True
            score, used_mode = insertion_score(
                model, benchmark, sample, inputs, vis_embeds, grid_thw, keep,
                answer_ids, args.max_answer_tokens_for_grad, max_tok, args.oracle_insertion_score_mode)
            score_mode = used_mode if score_mode == used_mode else f"{score_mode}+{used_mode}"
            gain = max(0.0, float(score) - float(base_score))
            for idx in group:
                insertion[idx] = gain
            group_scores.append({"tokens": group, "score": score, "gain": gain})
    else:
        full_keep = [False] * n_vis
        for idx in candidate_order:
            full_keep[int(idx)] = True
        full_score, used_mode = insertion_score(
            model, benchmark, sample, inputs, vis_embeds, grid_thw, full_keep,
            answer_ids, args.max_answer_tokens_for_grad, max_tok, args.oracle_insertion_score_mode)
        score_mode = used_mode if score_mode == used_mode else f"{score_mode}+{used_mode}"
        if attribution_mode == "group_leaveout":
            for group in groups:
                keep = list(full_keep)
                for idx in group:
                    keep[idx] = False
                score, used_mode = insertion_score(
                    model, benchmark, sample, inputs, vis_embeds, grid_thw, keep,
                    answer_ids, args.max_answer_tokens_for_grad, max_tok, args.oracle_insertion_score_mode)
                score_mode = used_mode if score_mode == used_mode else f"{score_mode}+{used_mode}"
                contribution = max(0.0, float(full_score) - float(score))
                for idx in group:
                    insertion[idx] = contribution
                group_scores.append({"tokens": group, "score_without": score, "contribution": contribution})
        elif attribution_mode == "budget_token_leaveout":
            budget_fracs = parse_float_list_csv(args.oracle_budget_attribution_fracs, [10.0, 5.0])
            budget_weights = parse_float_list_csv(args.oracle_budget_attribution_weights, [0.5, 0.5])
            if len(budget_weights) < len(budget_fracs):
                budget_weights.extend([budget_weights[-1] if budget_weights else 1.0] * (len(budget_fracs) - len(budget_weights)))
            budget_weights = budget_weights[: len(budget_fracs)]
            weight_sum = sum(max(0.0, float(w)) for w in budget_weights)
            if weight_sum <= 1e-12:
                budget_weights = [1.0 / max(1, len(budget_fracs)) for _ in budget_fracs]
            else:
                budget_weights = [max(0.0, float(w)) / weight_sum for w in budget_weights]
            max_frac = max([float(x) for x in budget_fracs] + [1.0])
            budget_summaries = []
            for frac, budget_weight in zip(budget_fracs, budget_weights):
                rel = max(0.0, min(1.0, float(frac) / max_frac))
                seed_count = max(1, min(candidate_count, int(math.ceil(candidate_count * rel))))
                seed_tokens = [int(x) for x in candidate_order[:seed_count]]
                seed_keep = [False] * n_vis
                for idx in seed_tokens:
                    seed_keep[idx] = True
                seed_score, used_mode = insertion_score(
                    model, benchmark, sample, inputs, vis_embeds, grid_thw, seed_keep,
                    answer_ids, args.max_answer_tokens_for_grad, max_tok, args.oracle_insertion_score_mode)
                score_mode = used_mode if score_mode == used_mode else f"{score_mode}+{used_mode}"
                contribution_sum = 0.0
                for idx in seed_tokens:
                    keep = list(seed_keep)
                    keep[idx] = False
                    score, used_mode = insertion_score(
                        model, benchmark, sample, inputs, vis_embeds, grid_thw, keep,
                        answer_ids, args.max_answer_tokens_for_grad, max_tok, args.oracle_insertion_score_mode)
                    score_mode = used_mode if score_mode == used_mode else f"{score_mode}+{used_mode}"
                    contribution = max(0.0, float(seed_score) - float(score))
                    insertion[idx] += float(budget_weight) * contribution
                    contribution_sum += contribution
                    token_scores.append({
                        "budget_frac": float(frac),
                        "token": int(idx),
                        "score_without": score,
                        "contribution": contribution,
                    })
                budget_summaries.append({
                    "budget_frac": float(frac),
                    "weight": float(budget_weight),
                    "seed_count": int(seed_count),
                    "seed_score": float(seed_score),
                    "raw_contribution_sum": float(contribution_sum),
                })
            group_scores.append({"budget_token_leaveout": budget_summaries})
        else:
            for idx in candidate_order:
                keep = list(full_keep)
                keep[int(idx)] = False
                score, used_mode = insertion_score(
                    model, benchmark, sample, inputs, vis_embeds, grid_thw, keep,
                    answer_ids, args.max_answer_tokens_for_grad, max_tok, args.oracle_insertion_score_mode)
                score_mode = used_mode if score_mode == used_mode else f"{score_mode}+{used_mode}"
                contribution = max(0.0, float(full_score) - float(score))
                insertion[int(idx)] = contribution
                token_scores.append({"token": int(idx), "score_without": score, "contribution": contribution})

    grad = gradient_oracle(
        model, inputs, answer_ids,
        max_answer_tokens=args.max_answer_tokens_for_grad,
        score_mode=args.gradient_oracle_score,
    )
    component_transform = str(args.oracle_component_transform)
    insertion_n = normalize_oracle_component(insertion, component_transform)
    contrast_n = normalize_oracle_component(contrast_scores, component_transform)
    grad_n = normalize_oracle_component([float(x) for x in grad], component_transform)
    oracle = []
    for i in range(n_vis):
        oracle.append(
            float(args.oracle_insertion_weight) * insertion_n[i]
            + float(args.oracle_visual_contrast_weight) * contrast_n[i]
            + float(args.oracle_gradient_weight) * grad_n[i]
            + float(args.oracle_interaction_weight) * insertion_n[i] * contrast_n[i]
        )
    return oracle, {
        "mode": "simplified_insertion",
        "candidate_tokens": candidate_count,
        "insertion_groups": group_count,
        "insertion_score_mode": score_mode,
        "attribution_mode": attribution_mode,
        "component_transform": component_transform,
        "base_score": base_score,
        "full_candidate_score": full_score if attribution_mode != "group_insertion" else None,
        "group_scores": group_scores,
        "token_scores": token_scores[: min(len(token_scores), 128)],
        "weights": {
            "insertion": float(args.oracle_insertion_weight),
            "visual_contrast": float(args.oracle_visual_contrast_weight),
            "gradient": float(args.oracle_gradient_weight),
            "interaction": float(args.oracle_interaction_weight),
        },
    }


def _extract_last_number(text: str) -> float | None:
    nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", str(text).replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def generic_choice_acc(output: str, answer: str, choices: list[str] | None = None) -> float:
    o = _norm_text(output)
    answers = [str(x).strip().lower() for x in str(answer).split("<and>")]
    for a in answers:
        a = a.strip().strip("()")
        if not a:
            continue
        a_norm = _norm_text(a)
        if o == a:
            return 1.0
        mo = re.match(r"^\(?([a-d])\)?(?:[\s.:)-]|$)", o)
        ma = re.match(r"^\(?([a-d])\)?(?:[\s.:)-]|$)", a)
        if ma and mo and ma.group(1) == mo.group(1):
            return 1.0
        if choices and mo:
            idx = ord(mo.group(1)) - ord("a")
            if 0 <= idx < len(choices) and _norm_text(choices[idx]) == a_norm:
                return 1.0
        if ma and (o.startswith(ma.group(1)) or f"({ma.group(1)})" in o[:16]):
            return 1.0
        if choices and a in "abcd":
            idx = ord(a) - ord("a")
            if 0 <= idx < len(choices) and str(choices[idx]).strip().lower() in o:
                return 1.0
        if a in o[:120]:
            return 1.0
    return 0.0


def mathvista_acc(output: str, sample: dict[str, Any]) -> float:
    answer = str(sample.get("answer", ""))
    choices = sample.get("choices")
    if choices:
        return generic_choice_acc(output, answer, choices)
    pred_num = _extract_last_number(output)
    gold_num = _extract_last_number(answer)
    if pred_num is not None and gold_num is not None:
        tol = max(1e-3, abs(gold_num) * 0.01)
        return 1.0 if abs(pred_num - gold_num) <= tol else 0.0
    return 1.0 if _norm_text(answer) in _norm_text(output)[:160] else 0.0


def _mmvet_key_phrases(answer: str) -> list[str]:
    first = re.split(r"[\n.]", str(answer), maxsplit=1)[0]
    first = re.sub(r"^it is\s+", "", first, flags=re.I).strip()
    phrases = []
    if first:
        phrases.append(first)
        phrases.append(re.split(r"[,;(]", first, maxsplit=1)[0].strip())
    for pat in [
        r"\b(corn smut)\b",
        r"\b(gray mold)\b",
        r"\b(botrytis(?: cinerea| blight)?)\b",
        r"\b(early blight)\b",
        r"\b(club root)\b",
        r"\b(crown gall)\b",
        r"\b(brown rot)\b",
        r"\b(isaac newton|newton)\b",
        r"\b(confucius)\b",
    ]:
        m = re.search(pat, answer, flags=re.I)
        if m:
            phrases.append(m.group(1))
    seen = set()
    out = []
    for p in phrases:
        p = _norm_text(p).strip(" .;:")
        if len(p) >= 4 and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def mmvet_acc(output: str, sample: dict[str, Any]) -> float:
    o = _norm_text(output)
    for phrase in _mmvet_key_phrases(str(sample.get("answer", ""))):
        if phrase in o:
            return 1.0
    return 0.0


def eval_metric(benchmark: str, output: str, sample: dict[str, Any]) -> float:
    if benchmark in ("dvqa", "infographicvqa"):
        return bea_eval_metric("docvqa", output, sample)
    if benchmark == "hpope":
        return bea_eval_metric("pope", output, sample)
    if benchmark == "mathvista":
        return mathvista_acc(output, sample)
    if benchmark == "mmvet":
        return mmvet_acc(output, sample)
    if benchmark in ("blink", "blink_heldout", "muirbench", "mmiu", "mantis", "mmstar"):
        return generic_choice_acc(output, sample.get("answer", ""), sample.get("choices") or sample.get("options"))
    return bea_eval_metric(benchmark, output, sample)


def pct_fracs(specs: list[dict[str, Any]], only_label: str | None = None) -> list[float]:
    out = []
    for spec in specs:
        if spec["type"] != "percent":
            continue
        if only_label is not None and spec["label"] != only_label:
            continue
        out.append(max(0.0, min(1.0, float(spec["value"]) / 100.0)))
    return out


def oracle_top_tokens_by_budget(oracle: list[float], specs: list[dict[str, Any]]) -> dict[str, list[int]]:
    if not oracle:
        return {}
    order = sorted(range(len(oracle)), key=lambda i: float(oracle[i]), reverse=True)
    out = {}
    for spec in specs:
        if spec["type"] != "percent":
            continue
        frac = max(0.0, min(1.0, float(spec["value"]) / 100.0))
        k = max(1, min(len(order), int(math.ceil(len(order) * frac))))
        out[str(spec["label"])] = [int(i) for i in order[:k]]
    return out


def calibration_cache_dir(args: argparse.Namespace) -> Path:
    if args.calibration_cache_dir:
        return resolve(args.calibration_cache_dir)
    return resolve(args.out_dir) / "calibration_cache"


def calibration_cache_path(args: argparse.Namespace, benchmark: str) -> Path:
    return calibration_cache_dir(args) / f"{benchmark}.json.gz"


def test_cache_dir(args: argparse.Namespace) -> Path:
    if args.test_cache_dir:
        return resolve(args.test_cache_dir)
    return resolve(args.out_dir) / "test_cache"


def test_cache_path(args: argparse.Namespace, benchmark: str) -> Path:
    return test_cache_dir(args) / f"{benchmark}.json.gz"


def cache_metadata(args: argparse.Namespace, benchmark: str, layers: list[int],
                   total_layers: int, specs: list[dict[str, Any]],
                   sample_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "dualsignal_calibration_records_v1",
        "benchmark": benchmark,
        "model_path": str(resolve(args.model_path)),
        "backend_type": args.backend_type,
        "total_layers": int(total_layers),
        "layers": [int(layer) for layer in layers],
        "budget_labels": [str(spec["label"]) for spec in specs],
        "budgets": specs,
        "calib_samples": int(args.calib_samples),
        "calib_offset": int(args.calib_offset),
        "calib_filter_baseline_correct": bool(args.calib_filter_baseline_correct),
        "calib_filter_visual_grounded": bool(args.calib_filter_visual_grounded),
        "calib_min_baseline_score": float(args.calib_min_baseline_score),
        "calib_candidates_per_target": int(args.calib_candidates_per_target),
        "calibration_indices_manifest": str(args.calibration_indices_manifest) if args.calibration_indices_manifest else None,
        "resize_square": int(args.resize_square) if args.resize_square is not None else None,
        "max_pixels": int(args.max_pixels),
        "multi_max_pixels": int(args.multi_max_pixels),
        "train_token_pool": str(args.train_token_pool),
        "stage1_frac": float(args.stage1_frac),
        "stage1_shallow_weight": float(os.environ.get("DUALSIGNAL_STAGE1_WEIGHT", "0.5")),
        "stage1_gamma": float(args.stage1_gamma),
        "max_answer_tokens_for_grad": int(args.max_answer_tokens_for_grad),
        "oracle_target": str(args.oracle_target),
        "oracle_mode": str(args.oracle_mode),
        "gradient_oracle_score": str(args.gradient_oracle_score),
        "oracle_clip_quantile": float(args.oracle_clip_quantile),
        "oracle_score_transform": str(args.oracle_score_transform),
        "oracle_candidate_tokens": int(args.oracle_candidate_tokens),
        "oracle_insertion_groups": int(args.oracle_insertion_groups),
        "oracle_insertion_weight": float(args.oracle_insertion_weight),
        "oracle_visual_contrast_weight": float(args.oracle_visual_contrast_weight),
        "oracle_gradient_weight": float(args.oracle_gradient_weight),
        "oracle_interaction_weight": float(args.oracle_interaction_weight),
        "oracle_component_transform": str(args.oracle_component_transform),
        "oracle_attribution_mode": str(args.oracle_attribution_mode),
        "oracle_budget_attribution_fracs": str(args.oracle_budget_attribution_fracs),
        "oracle_budget_attribution_weights": str(args.oracle_budget_attribution_weights),
        "oracle_insertion_score_mode": str(args.oracle_insertion_score_mode),
        "sample_info": sample_info,
    }


def test_cache_metadata(args: argparse.Namespace, benchmark: str, layers: list[int],
                        total_layers: int, specs: list[dict[str, Any]],
                        sample_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "dualsignal_test_records_v1",
        "benchmark": benchmark,
        "model_path": str(resolve(args.model_path)),
        "backend_type": args.backend_type,
        "total_layers": int(total_layers),
        "layers": [int(layer) for layer in layers],
        "budget_labels": [str(spec["label"]) for spec in specs],
        "budgets": specs,
        "eval_samples": int(args.eval_samples),
        "eval_offset": int(args.eval_offset),
        "resize_square": int(args.resize_square) if args.resize_square is not None else None,
        "max_pixels": int(args.max_pixels),
        "multi_max_pixels": int(args.multi_max_pixels),
        "train_token_pool": str(args.train_token_pool),
        "stage1_frac": float(args.stage1_frac),
        "stage1_gamma": float(args.stage1_gamma),
        "sample_info": sample_info,
    }


def metadata_matches(cached: dict[str, Any], expected: dict[str, Any]) -> bool:
    keys = [
        "format",
        "benchmark",
        "model_path",
        "backend_type",
        "total_layers",
        "layers",
        "calib_samples",
        "calib_offset",
        "calib_filter_baseline_correct",
        "calib_filter_visual_grounded",
        "calib_min_baseline_score",
        "calib_candidates_per_target",
        "calibration_indices_manifest",
        "resize_square",
        "max_pixels",
        "multi_max_pixels",
        "train_token_pool",
        "stage1_frac",
        "stage1_shallow_weight",
        "stage1_gamma",
        "max_answer_tokens_for_grad",
        "oracle_target",
        "oracle_mode",
        "gradient_oracle_score",
        "oracle_clip_quantile",
        "oracle_score_transform",
        "oracle_candidate_tokens",
        "oracle_insertion_groups",
        "oracle_insertion_weight",
        "oracle_visual_contrast_weight",
        "oracle_gradient_weight",
        "oracle_interaction_weight",
        "oracle_component_transform",
        "oracle_attribution_mode",
        "oracle_budget_attribution_fracs",
        "oracle_budget_attribution_weights",
        "oracle_insertion_score_mode",
    ]
    return all(cached.get(key) == expected.get(key) for key in keys)


def test_metadata_matches(cached: dict[str, Any], expected: dict[str, Any]) -> bool:
    keys = [
        "format",
        "benchmark",
        "model_path",
        "backend_type",
        "total_layers",
        "layers",
        "eval_samples",
        "eval_offset",
        "resize_square",
        "max_pixels",
        "multi_max_pixels",
        "train_token_pool",
        "stage1_frac",
        "stage1_gamma",
    ]
    return all(cached.get(key) == expected.get(key) for key in keys)


def load_calibration_cache(path: Path, expected_meta: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"  calibration cache unreadable {path}: {exc}", flush=True)
        return None
    meta = payload.get("metadata", {})
    if not metadata_matches(meta, expected_meta):
        print(f"  calibration cache metadata mismatch, recollecting: {path}", flush=True)
        return None
    records = payload.get("records", [])
    print(f"  loaded calibration cache {path} records={len(records)}", flush=True)
    return records


def save_calibration_cache(path: Path, metadata: dict[str, Any],
                           records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {**metadata, "generated_at": time.time()},
        "records": records,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(path)
    print(f"  wrote calibration cache {path} records={len(records)}", flush=True)


def load_test_cache(path: Path, expected_meta: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"  test cache unreadable {path}: {exc}", flush=True)
        return None
    meta = payload.get("metadata", {})
    if not test_metadata_matches(meta, expected_meta):
        print(f"  test cache metadata mismatch, recollecting: {path}", flush=True)
        return None
    records = payload.get("records", [])
    print(f"  loaded test cache {path} records={len(records)}", flush=True)
    return records


def save_test_cache(path: Path, metadata: dict[str, Any],
                    records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {**metadata, "generated_at": time.time()},
        "records": records,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(path)
    print(f"  wrote test cache {path} records={len(records)}", flush=True)


def collect_calibration(model, benchmark: str, samples: list[dict[str, Any]],
                        layers: list[int], specs: list[dict[str, Any]],
                        args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    t0 = time.time()
    max_tok = max_new_tokens_map(benchmark)
    for idx, sample in enumerate(samples):
        try:
            source_idx = int(sample.get("__source_index", idx)) if isinstance(sample, dict) else idx
            images = sample_images(sample, args.resize_square)
            sample = dict(sample, images=images, image=images[0])
            inputs = prepare_inputs_any(
                model, images, make_prompt(benchmark, sample),
                args.max_pixels, args.multi_max_pixels)
            vis_embeds, grid_thw = model.extract_visual_embeddings(inputs)
            n_full_vis = int(vis_embeds.shape[0])
            if n_full_vis == 0:
                continue
            original_indices = None
            pool_mask = None
            token_pool_metadata = {"train_token_pool": str(args.train_token_pool)}
            if args.train_token_pool == "stage1":
                pool_mask, stage1_meta = stage1_keep_mask(
                    model, inputs, vis_embeds, n_full_vis, args.stage1_frac, args.stage1_gamma)
                original_indices = [i for i, keep in enumerate(pool_mask) if keep]
                token_pool_metadata.update(stage1_meta)
            n_pool_vis = len(original_indices) if original_indices is not None else n_full_vis
            answer_ids, baseline_text = generate_ids_and_text(model, inputs, max_tok)
            baseline_score = eval_metric(benchmark, baseline_text, sample)
            no_visual_text = None
            no_visual_score = None
            visual_grounded_reason = None
            fixed_calib = bool(getattr(args, "calibration_indices_manifest", None))
            if args.calib_filter_visual_grounded and not fixed_calib:
                no_visual_text = model.run_pruned(inputs, vis_embeds, grid_thw, [False] * int(vis_embeds.shape[0]), max_tok)
                no_visual_score = eval_metric(benchmark, no_visual_text, sample)
                if (
                    isinstance(baseline_score, (int, float))
                    and float(baseline_score) >= float(args.calib_min_baseline_score)
                    and isinstance(no_visual_score, (int, float))
                    and float(no_visual_score) < float(baseline_score)
                ):
                    visual_grounded_reason = "baseline_correct_no_visual_degraded"
                elif comparable_answer_text(baseline_text) != comparable_answer_text(no_visual_text):
                    visual_grounded_reason = "full_differs_from_no_visual"
            skip_reason = None
            if not fixed_calib and args.calib_filter_baseline_correct and baseline_score <= args.calib_min_baseline_score:
                skip_reason = "baseline_below_threshold"
            if not fixed_calib and args.calib_filter_visual_grounded and visual_grounded_reason != "baseline_correct_no_visual_degraded":
                skip_reason = "not_baseline_correct_no_visual_degraded"
            if skip_reason:
                avg = (time.time() - t0) / max(1, idx + 1)
                print(
                    f"  {benchmark} calib skip {idx + 1}/{len(samples)} "
                    f"baseline={baseline_score:.3f} no_visual={no_visual_score} "
                    f"reason={skip_reason} kept={len(records)}/{args.calib_samples} avg={avg:.1f}s",
                    flush=True,
                )
                continue
            oracle_answer_text = baseline_text
            oracle_answer_ids = answer_ids
            if args.oracle_target == "gt":
                oracle_answer_text = gt_answer_text(sample)
                if not oracle_answer_text:
                    avg = (time.time() - t0) / max(1, idx + 1)
                    print(
                        f"  {benchmark} calib skip {idx + 1}/{len(samples)} "
                        f"empty_gt_answer kept={len(records)}/{args.calib_samples} avg={avg:.1f}s",
                        flush=True,
                    )
                    continue
                oracle_answer_ids = tokenize_answer(model, oracle_answer_text, inputs["input_ids"].device)
            if args.train_token_pool == "stage1":
                if args.oracle_mode == "simplified_insertion":
                    raise RuntimeError("simplified_insertion oracle is not implemented for stage1-pruned LLM context")
                layer_scores = stage1_pruned_layer_attention_scores(
                    model, inputs, vis_embeds, grid_thw, pool_mask, layers)
            else:
                layer_scores = {
                    str(layer): scores
                    for layer, scores in layer_attention_scores(model, inputs, layers).items()
                }
            oracle_metadata = {"mode": "gradient"}
            if args.oracle_mode == "simplified_insertion":
                oracle, oracle_metadata = simplified_insertion_oracle(
                    model, benchmark, sample, inputs, vis_embeds, grid_thw,
                    oracle_answer_ids, layer_scores, layers, args, max_tok)
            else:
                if args.train_token_pool == "stage1":
                    oracle = stage1_pruned_gradient_oracle(
                        model, inputs, vis_embeds, grid_thw, pool_mask, oracle_answer_ids,
                        max_answer_tokens=args.max_answer_tokens_for_grad,
                        score_mode=args.gradient_oracle_score)
                else:
                    oracle = gradient_oracle(
                        model, inputs, oracle_answer_ids,
                        max_answer_tokens=args.max_answer_tokens_for_grad,
                        score_mode=args.gradient_oracle_score)
            oracle, transform_metadata = transform_oracle_scores(oracle, args)
            oracle_metadata = dict(oracle_metadata)
            oracle_metadata["oracle_transform"] = transform_metadata
            oracle_metadata["token_pool"] = token_pool_metadata
            record = {
                "benchmark": benchmark,
                "sample_index": source_idx,
                "calibration_order_index": idx,
                "num_vis": n_pool_vis,
                "num_vis_full": n_full_vis,
                "token_pool": token_pool_metadata,
                "original_token_indices": original_indices,
                "baseline": baseline_score,
                "baseline_text": baseline_text,
                "oracle_target": args.oracle_target,
                "gradient_oracle_score": args.gradient_oracle_score,
                "oracle_answer_text": oracle_answer_text,
                "oracle_metadata": oracle_metadata,
                "no_visual": {
                    "score": no_visual_score,
                    "prediction": no_visual_text,
                } if args.calib_filter_visual_grounded else None,
                "visual_grounded_reason": visual_grounded_reason,
                "oracle": oracle,
                "oracle_top_tokens_by_budget": oracle_top_tokens_by_budget(oracle, specs),
                "layer_scores": layer_scores,
            }
            records.append(record)
            avg = (time.time() - t0) / max(1, idx + 1)
            print(
                f"  {benchmark} calib keep {idx + 1}/{len(samples)} "
                f"baseline={baseline_score:.3f} kept={len(records)}/{args.calib_samples} avg={avg:.1f}s",
                flush=True,
            )
            if len(records) >= args.calib_samples:
                break
        except Exception:
            print(traceback.format_exc(), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return records


def collect_test_records(model, benchmark: str, samples: list[dict[str, Any]],
                         layers: list[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    t0 = time.time()
    max_tok = max_new_tokens_map(benchmark)
    for idx, sample in enumerate(samples):
        try:
            images = sample_images(sample, args.resize_square)
            sample = dict(sample, images=images, image=images[0])
            inputs = prepare_inputs_any(
                model, images, make_prompt(benchmark, sample),
                args.max_pixels, args.multi_max_pixels)
            vis_embeds, _ = model.extract_visual_embeddings(inputs)
            n_full_vis = int(vis_embeds.shape[0])
            if n_full_vis == 0:
                continue
            original_indices = None
            pool_mask = None
            token_pool_metadata = {"train_token_pool": str(args.train_token_pool)}
            if args.train_token_pool == "stage1":
                pool_mask, stage1_meta = stage1_keep_mask(
                    model, inputs, vis_embeds, n_full_vis, args.stage1_frac, args.stage1_gamma)
                original_indices = [i for i, keep in enumerate(pool_mask) if keep]
                token_pool_metadata.update(stage1_meta)
            n_pool_vis = len(original_indices) if original_indices is not None else n_full_vis
            if args.train_token_pool == "stage1":
                layer_scores = stage1_pruned_layer_attention_scores(
                    model, inputs, vis_embeds, grid_thw, pool_mask, layers)
            else:
                layer_scores = {
                    str(layer): scores
                    for layer, scores in layer_attention_scores(model, inputs, layers).items()
                }
            _answer_ids, baseline_text = generate_ids_and_text(model, inputs, max_tok)
            records.append({
                "benchmark": benchmark,
                "sample_index": idx,
                "num_vis": n_pool_vis,
                "num_vis_full": n_full_vis,
                "token_pool": token_pool_metadata,
                "original_token_indices": original_indices,
                "baseline": eval_metric(benchmark, baseline_text, sample),
                "baseline_text": baseline_text,
                "layer_scores": layer_scores,
            })
            avg = (time.time() - t0) / max(1, idx + 1)
            print(f"  {benchmark} test cache {idx + 1}/{len(samples)} avg={avg:.1f}s", flush=True)
        except Exception:
            print(traceback.format_exc(), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return records


def _attach_scoring(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    policy["score_fusion"] = "weighted_attention_with_layer_consistency"
    policy["consistency_lambda"] = float(args.consistency_lambda)
    policy["uncertainty_lambda"] = float(args.uncertainty_lambda)
    policy["consistency_eps"] = float(args.consistency_eps)
    policy["consistency_mode"] = str(args.consistency_mode)
    policy["effective_layers_lambda"] = float(args.effective_layers_lambda)
    policy["effective_layers_target"] = float(args.effective_layers_target)
    policy["train_token_pool"] = str(args.train_token_pool)
    policy["stage1_frac"] = float(args.stage1_frac)
    policy["stage1_gamma"] = float(args.stage1_gamma)
    return policy


def fit_budget_policies(records: list[dict[str, Any]], layers: list[int], total_layers: int,
                        specs: list[dict[str, Any]], args: argparse.Namespace,
                        prefix: str) -> dict[str, Any]:
    by_budget = {}
    diagnostics = {}
    if args.shared_budget_policy:
        labels = [str(spec["label"]) for spec in specs if spec["type"] == "percent"]
        shared_label = "shared_" + "_".join(labels)
        print(f"  fitting {prefix} shared budgets {', '.join(labels)}", flush=True)
        weights, diag = fit_weights(
            records, layers, total_layers,
            rank_temperature=args.rank_temperature,
            oracle_temperature=args.oracle_temperature,
            depth_lambda=args.depth_lambda,
            entropy_lambda=args.entropy_lambda,
            l2_lambda=args.l2_lambda,
            steps=args.optim_steps,
            lr=args.lr,
            fit_loss=args.fit_loss,
            fit_budget_fracs=pct_fracs_for_pool(specs, args),
            max_pairwise_negatives=args.max_pairwise_negatives,
            fit_device=args.fit_device,
            active_weight_threshold=args.active_weight_threshold,
            consistency_lambda=args.consistency_lambda,
            uncertainty_lambda=args.uncertainty_lambda,
            consistency_eps=args.consistency_eps,
            consistency_mode=args.consistency_mode,
            effective_layers_lambda=args.effective_layers_lambda,
            effective_layers_target=args.effective_layers_target,
            min_deep_layer_frac=args.min_deep_layer_frac,
            min_deep_weight=args.min_deep_weight,
            min_deep_weight_lambda=args.min_deep_weight_lambda,
        )
        shared_policy = _attach_scoring(make_policy(
            prefix + "_" + shared_label,
            weights,
            layers,
            total_layers,
            args.max_active_layers,
            args.active_weight_threshold,
        ), args)
        for label in labels:
            by_budget[label] = dict(shared_policy)
            by_budget[label]["budget_label"] = label
            diagnostics[label] = diag
        return {
            "name": prefix,
            "shared_budget_policy": True,
            "shared_budget_labels": labels,
            "by_budget": by_budget,
            "diagnostics": diagnostics,
            "weight_matrix": [{
                "budget": shared_label,
                "layers": list(shared_policy["all_layers"]),
                "weights": list(shared_policy["all_weights"]),
            }],
            "active_weight_threshold": float(args.active_weight_threshold),
            "score_fusion": "weighted_attention_with_layer_consistency",
            "consistency_lambda": float(args.consistency_lambda),
            "uncertainty_lambda": float(args.uncertainty_lambda),
            "consistency_eps": float(args.consistency_eps),
            "consistency_mode": str(args.consistency_mode),
            "effective_layers_lambda": float(args.effective_layers_lambda),
            "effective_layers_target": float(args.effective_layers_target),
            "min_deep_layer_frac": float(args.min_deep_layer_frac),
            "min_deep_weight": float(args.min_deep_weight),
            "min_deep_weight_lambda": float(args.min_deep_weight_lambda),
        }
    for spec in specs:
        if spec["type"] != "percent":
            continue
        label = spec["label"]
        print(f"  fitting {prefix} budget {label}", flush=True)
        weights, diag = fit_weights(
            records, layers, total_layers,
            rank_temperature=args.rank_temperature,
            oracle_temperature=args.oracle_temperature,
            depth_lambda=args.depth_lambda,
            entropy_lambda=args.entropy_lambda,
            l2_lambda=args.l2_lambda,
            steps=args.optim_steps,
            lr=args.lr,
            fit_loss=args.fit_loss,
            fit_budget_fracs=pct_fracs_for_pool(specs, args, label),
            max_pairwise_negatives=args.max_pairwise_negatives,
            fit_device=args.fit_device,
            active_weight_threshold=args.active_weight_threshold,
            consistency_lambda=args.consistency_lambda,
            uncertainty_lambda=args.uncertainty_lambda,
            consistency_eps=args.consistency_eps,
            consistency_mode=args.consistency_mode,
            effective_layers_lambda=args.effective_layers_lambda,
            effective_layers_target=args.effective_layers_target,
            min_deep_layer_frac=args.min_deep_layer_frac,
            min_deep_weight=args.min_deep_weight,
            min_deep_weight_lambda=args.min_deep_weight_lambda,
        )
        by_budget[label] = _attach_scoring(make_policy(
            prefix + "_" + label,
            weights,
            layers,
            total_layers,
            args.max_active_layers,
            args.active_weight_threshold,
        ), args)
        diagnostics[label] = diag
    weight_matrix = [
        {
            "budget": label,
            "layers": list(by_budget[label]["all_layers"]),
            "weights": list(by_budget[label]["all_weights"]),
        }
        for label in by_budget
    ]
    return {
        "name": prefix,
        "by_budget": by_budget,
        "diagnostics": diagnostics,
        "weight_matrix": weight_matrix,
        "active_weight_threshold": float(args.active_weight_threshold),
        "score_fusion": "weighted_attention_with_layer_consistency",
        "consistency_lambda": float(args.consistency_lambda),
        "uncertainty_lambda": float(args.uncertainty_lambda),
        "consistency_eps": float(args.consistency_eps),
        "consistency_mode": str(args.consistency_mode),
        "effective_layers_lambda": float(args.effective_layers_lambda),
        "effective_layers_target": float(args.effective_layers_target),
        "min_deep_layer_frac": float(args.min_deep_layer_frac),
        "min_deep_weight": float(args.min_deep_weight),
        "min_deep_weight_lambda": float(args.min_deep_weight_lambda),
    }


def validate_policy(model, benchmark: str, samples: list[dict[str, Any]],
                    layers: list[int], policy: dict[str, Any],
                    specs: list[dict[str, Any]], args: argparse.Namespace,
                    test_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    max_tok = max_new_tokens_map(benchmark)
    records = []
    t0 = time.time()
    cached_by_index = {
        int(record.get("sample_index")): record
        for record in (test_records or [])
        if isinstance(record.get("sample_index"), int)
    }
    for idx, sample in enumerate(samples):
        try:
            images = sample_images(sample, args.resize_square)
            sample = dict(sample, images=images, image=images[0])
            inputs = prepare_inputs_any(
                model, images, make_prompt(benchmark, sample),
                args.max_pixels, args.multi_max_pixels)
            vis_embeds, grid_thw = model.extract_visual_embeddings(inputs)
            n_full_vis = int(vis_embeds.shape[0])
            if n_full_vis == 0:
                continue
            cached = cached_by_index.get(idx)
            token_pool_metadata = {"train_token_pool": str(args.train_token_pool)}
            original_indices = None
            n_score_vis = n_full_vis
            if (
                cached is not None
                and int(cached.get("num_vis_full", cached.get("num_vis", -1))) == n_full_vis
                and str(cached.get("token_pool", {}).get("train_token_pool", "full")) == str(args.train_token_pool)
            ):
                baseline_text = str(cached.get("baseline_text", ""))
                baseline_score = cached.get("baseline")
                layer_scores = cached.get("layer_scores", {})
                token_pool_metadata = dict(cached.get("token_pool", token_pool_metadata))
                original_indices = cached.get("original_token_indices")
                n_score_vis = int(cached.get("num_vis", n_full_vis))
            else:
                _answer_ids, baseline_text = generate_ids_and_text(model, inputs, max_tok)
                baseline_score = eval_metric(benchmark, baseline_text, sample)
                if args.train_token_pool == "stage1":
                    pool_mask, stage1_meta = stage1_keep_mask(
                        model, inputs, vis_embeds, n_full_vis, args.stage1_frac, args.stage1_gamma)
                    original_indices = [i for i, keep in enumerate(pool_mask) if keep]
                    token_pool_metadata.update(stage1_meta)
                    n_score_vis = len(original_indices)
                    layer_scores = stage1_pruned_layer_attention_scores(
                        model, inputs, vis_embeds, grid_thw, pool_mask, layers)
                else:
                    layer_scores = {
                        str(layer): scores
                        for layer, scores in layer_attention_scores(model, inputs, layers).items()
                    }
            row = {
                "benchmark": benchmark,
                "sample_index": idx,
                "num_vis": n_score_vis,
                "num_vis_full": n_full_vis,
                "token_pool": token_pool_metadata,
                "original_token_indices": original_indices,
                "baseline": baseline_score,
                "baseline_text": baseline_text,
                "baseline_diagnostics": diagnostic_scores(benchmark, baseline_text, sample),
                "test_cache_reused": cached is not None,
                "budgets": {},
            }
            for spec in specs:
                label = spec["label"]
                budget_policy = policy["by_budget"][label]
                if (
                    float(budget_policy.get("consistency_lambda", args.consistency_lambda)) != 0.0
                    or float(budget_policy.get("uncertainty_lambda", args.uncertainty_lambda)) != 0.0
                ):
                    fused = fuse_policy_scores_with_consistency(
                        layer_scores,
                        budget_policy["active_layers"],
                        budget_policy["active_weights"],
                        n_score_vis,
                        consistency_lambda=float(budget_policy.get("consistency_lambda", args.consistency_lambda)),
                        uncertainty_lambda=float(budget_policy.get("uncertainty_lambda", args.uncertainty_lambda)),
                        consistency_eps=float(budget_policy.get("consistency_eps", args.consistency_eps)),
                        consistency_mode=str(budget_policy.get("consistency_mode", args.consistency_mode)),
                    )
                else:
                    fused = fuse_policy_scores(
                        layer_scores,
                        budget_policy["active_layers"],
                        budget_policy["active_weights"],
                        n_score_vis,
                    )
                try:
                    if args.train_token_pool == "stage1":
                        pool_frac = stage1_pool_budget_fraction(spec, n_full_vis, n_score_vis)
                        pool_keep_mask = allocate(fused, n_score_vis, pool_frac)
                        keep_mask = full_mask_from_pool_mask(pool_keep_mask, original_indices, n_full_vis)
                    else:
                        keep_mask = allocate(fused, n_score_vis, budget_fraction(spec, n_score_vis))
                    pred = model.run_pruned(inputs, vis_embeds, grid_thw, keep_mask, max_tok)
                    score = eval_metric(benchmark, pred, sample)
                    error = None
                except Exception as exc:
                    keep_mask = []
                    pred = ""
                    score = None
                    error = f"{type(exc).__name__}: {exc}"
                row["budgets"][label] = {
                    "score": score,
                    "prediction": pred,
                    "kept_tokens": sum(1 for v in keep_mask if v),
                    "error": error,
                    "diagnostics": diagnostic_scores(benchmark, pred, sample) if error is None else {},
                }
            records.append(row)
            avg = (time.time() - t0) / max(1, idx + 1)
            print(f"  {benchmark} eval {policy['name']} {idx + 1}/{len(samples)} avg={avg:.1f}s", flush=True)
        except Exception:
            print(traceback.format_exc(), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {"records": records, "summary": summarize(records, specs)}


def summarize(records: list[dict[str, Any]], specs: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"budgets": {}}
    for spec in specs:
        label = spec["label"]
        scores, fidelities, kept = [], [], []
        diagnostic_keys: set[str] = set()
        errors = 0
        for row in records:
            diagnostic_keys.update(row.get("baseline_diagnostics", {}).keys())
            cell = row.get("budgets", {}).get(label, {})
            diagnostic_keys.update(cell.get("diagnostics", {}).keys())
            score = cell.get("score")
            baseline = row.get("baseline")
            if isinstance(score, (int, float)):
                scores.append(float(score))
            if isinstance(baseline, (int, float)) and baseline > 0 and isinstance(score, (int, float)):
                fidelities.append(min(1.0, float(score) / float(baseline)))
            if isinstance(cell.get("kept_tokens"), int):
                kept.append(int(cell["kept_tokens"]))
            if cell.get("error"):
                errors += 1
        diagnostics = {}
        for key in sorted(diagnostic_keys):
            base_vals, pred_vals = [], []
            for row in records:
                base_diag = row.get("baseline_diagnostics", {}).get(key)
                pred_diag = row.get("budgets", {}).get(label, {}).get("diagnostics", {}).get(key)
                if isinstance(base_diag, (int, float)):
                    base_vals.append(float(base_diag))
                if isinstance(pred_diag, (int, float)):
                    pred_vals.append(float(pred_diag))
            base_mean = sum(base_vals) / len(base_vals) if base_vals else None
            pred_mean = sum(pred_vals) / len(pred_vals) if pred_vals else None
            diagnostics[key] = {
                "baseline": base_mean,
                "score": pred_mean,
                "ratio_fidelity": (
                    pred_mean / base_mean
                    if pred_mean is not None and base_mean not in (None, 0)
                    else None
                ),
                "valid": len(pred_vals),
            }
        out["budgets"][label] = {
            "accuracy": sum(scores) / len(scores) if scores else None,
            "valid": len(scores),
            "fidelity": sum(fidelities) / len(fidelities) if fidelities else None,
            "fidelity_valid": len(fidelities),
            "mean_kept_tokens": sum(kept) / len(kept) if kept else None,
            "errors": errors,
            "diagnostics": diagnostics,
        }
    return out


def dense(policy: dict[str, Any], total_layers: int) -> list[float]:
    v = [0.0] * total_layers
    for layer, weight in zip(policy["active_layers"], policy["active_weights"]):
        v[int(layer)] = float(weight)
    return v


def cosine(a: list[float], b: list[float]) -> float:
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def write_report(payload: dict[str, Any], path: Path) -> None:
    def fmt(value: Any) -> str:
        return f"{value:.3f}" if isinstance(value, (int, float)) else "NA"

    lines = [
        "# Held-Out Gradient-Aligned Policy Suite",
        "",
        "Calibration and validation samples are disjoint. No prior sweep, validation, strategy, or calibration artifacts were read.",
        "",
        "## Fit Settings",
        "",
        f"Fit loss: `{payload['plan'].get('fit_loss')}`",
        f"Shared budget policy: `{payload['plan'].get('shared_budget_policy')}`",
        f"Layer consistency lambda: `{payload['plan'].get('consistency_lambda')}`",
        f"Layer uncertainty lambda: `{payload['plan'].get('uncertainty_lambda')}`",
        f"Layer consistency mode: `{payload['plan'].get('consistency_mode')}`",
        f"Effective layers target/lambda: `{payload['plan'].get('effective_layers_target')}` / `{payload['plan'].get('effective_layers_lambda')}`",
        f"Min deep layer frac/weight/lambda: `{payload['plan'].get('min_deep_layer_frac')}` / `{payload['plan'].get('min_deep_weight')}` / `{payload['plan'].get('min_deep_weight_lambda')}`",
        f"Max active layers: `{payload['plan'].get('max_active_layers')}`",
        f"Active weight threshold: `{payload['plan'].get('active_weight_threshold')}`",
        f"Depth delay lambda: `{payload['plan'].get('depth_lambda')}`",
        f"Oracle clip quantile / transform: `{payload['plan'].get('oracle_clip_quantile')}` / `{payload['plan'].get('oracle_score_transform')}`",
        "",
        "## Held-Out Fidelity",
        "",
        "| Eval benchmark | Policy | Budget | Accuracy | Fidelity | Fidelity valid |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for bench, bench_block in payload["validation"].items():
        for policy_name, result in bench_block.items():
            for budget, cell in result["summary"]["budgets"].items():
                lines.append(
                    f"| {bench} | {policy_name} | {budget} | "
                    f"{fmt(cell['accuracy'])} | {fmt(cell['fidelity'])} | {cell['fidelity_valid']} |"
                )
    lines.extend(["", "## Policies", "", "| Policy | Budget | Active layers |", "|---|---:|---|"])
    for name, policy in payload["policies"].items():
        for budget, p in policy["by_budget"].items():
            active = ", ".join(
                f"L{l}:{w:.3f}" for l, w in zip(p["active_layers"], p["active_weights"])
            )
            lines.append(f"| {name} | {budget} | {active} |")
    if not payload["plan"].get("global_only"):
        lines.extend(["", "## Global vs Benchmark-Specific Policy Distance", "", "| Benchmark | Budget | Cosine | L1 distance | Top-5 overlap |", "|---|---:|---:|---:|---:|"])
        total_layers = payload["total_layers"]
        global_names = payload["global_policy_names"]
        for bench in payload["benchmarks"]:
            task_policy = payload["policies"][bench]
            for budget in payload["budget_labels"]:
                for global_name in global_names:
                    g = payload["policies"][global_name]["by_budget"][budget]
                    t = task_policy["by_budget"][budget]
                    gv, tv = dense(g, total_layers), dense(t, total_layers)
                    l1 = sum(abs(x - y) for x, y in zip(gv, tv))
                    overlap = len(set(g["active_layers"]) & set(t["active_layers"]))
                    lines.append(f"| {bench} vs {global_name} | {budget} | {cosine(gv, tv):.3f} | {l1:.3f} | {overlap} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = [normalize_benchmark(x) for x in args.benchmarks]
    global_sets = parse_global_sets(args.global_sets, benchmarks)
    specs = budget_specs(args.budgets, [])
    plan = {
        "model_path": args.model_path,
        "backend_type": args.backend_type,
        "benchmarks": benchmarks,
        "budgets": specs,
        "calib_samples": args.calib_samples,
        "calib_offset": args.calib_offset,
        "calib_filter_baseline_correct": args.calib_filter_baseline_correct,
        "calib_filter_visual_grounded": args.calib_filter_visual_grounded,
        "calib_min_baseline_score": args.calib_min_baseline_score,
        "require_calib_samples": args.require_calib_samples,
        "calib_candidates_per_target": args.calib_candidates_per_target,
        "eval_samples": args.eval_samples,
        "eval_offset": args.eval_offset,
        "fit_loss": args.fit_loss,
        "fit_per_budget": not args.shared_budget_policy,
        "shared_budget_policy": args.shared_budget_policy,
        "consistency_lambda": args.consistency_lambda,
        "uncertainty_lambda": args.uncertainty_lambda,
        "consistency_eps": args.consistency_eps,
        "consistency_mode": args.consistency_mode,
        "effective_layers_lambda": args.effective_layers_lambda,
        "effective_layers_target": args.effective_layers_target,
        "min_deep_layer_frac": args.min_deep_layer_frac,
        "min_deep_weight": args.min_deep_weight,
        "min_deep_weight_lambda": args.min_deep_weight_lambda,
        "loss_definition": (
            "budget_overlap maximizes soft retained mass on backward-oracle top-budget tokens; "
            "depth_lambda adds a differentiable deepest-active-layer delayed-pruning cost "
            "using the active weight threshold."
            if args.fit_loss in ("budget_overlap", "budget_overlap_pairwise")
            else None
        ),
        "max_layer": args.max_layer,
        "max_layer_frac": args.max_layer_frac,
        "max_active_layers": args.max_active_layers,
        "active_weight_threshold": args.active_weight_threshold,
        "train_token_pool": args.train_token_pool,
        "stage1_frac": args.stage1_frac,
        "stage1_shallow_weight": float(os.environ.get("DUALSIGNAL_STAGE1_WEIGHT", "0.5")),
        "stage1_gamma": args.stage1_gamma,
        "rank_temperature": args.rank_temperature,
        "oracle_temperature": args.oracle_temperature,
        "depth_lambda": args.depth_lambda,
        "entropy_lambda": args.entropy_lambda,
        "l2_lambda": args.l2_lambda,
        "optim_steps": args.optim_steps,
        "lr": args.lr,
        "max_pairwise_negatives": args.max_pairwise_negatives,
        "fit_device": args.fit_device,
        "oracle_target": args.oracle_target,
        "oracle_mode": args.oracle_mode,
        "gradient_oracle_score": args.gradient_oracle_score,
        "oracle_clip_quantile": args.oracle_clip_quantile,
        "oracle_score_transform": args.oracle_score_transform,
        "oracle_candidate_tokens": args.oracle_candidate_tokens,
        "oracle_insertion_groups": args.oracle_insertion_groups,
        "oracle_insertion_weight": args.oracle_insertion_weight,
        "oracle_visual_contrast_weight": args.oracle_visual_contrast_weight,
        "oracle_gradient_weight": args.oracle_gradient_weight,
        "oracle_interaction_weight": args.oracle_interaction_weight,
        "oracle_component_transform": args.oracle_component_transform,
        "oracle_insertion_score_mode": args.oracle_insertion_score_mode,
        "calibration_cache_dir": str(calibration_cache_dir(args)),
        "reuse_calibration_cache": not args.no_reuse_calibration_cache,
        "test_cache_dir": str(test_cache_dir(args)),
        "reuse_test_cache": not args.no_reuse_test_cache,
        "global_sets": global_sets,
        "global_only": args.global_only,
        "cache_only": args.cache_only,
        "reads_prior_sweep_results": False,
    }
    print(json.dumps(plan, indent=2), flush=True)

    backend_type = None if args.backend_type == "auto" else args.backend_type
    model = load_backend(args.model_path, backend_type)
    total_layers = model.total_layers
    if args.max_layer is None:
        max_layer = max(0, int(total_layers * args.max_layer_frac) - 1)
    else:
        max_layer = int(args.max_layer)
    max_layer = min(max_layer, total_layers - 1)
    layers = list(range(max_layer + 1))
    plan["resolved_max_layer"] = max_layer
    plan["fit_layer_count"] = len(layers)
    print(json.dumps({"resolved_max_layer": max_layer, "fit_layer_count": len(layers)}, indent=2), flush=True)

    calib_by_benchmark = {}
    all_calib = []
    eval_samples_by_benchmark = {}
    test_records_by_benchmark = {}
    sample_info_by_benchmark = {}
    for bench in benchmarks:
        calib_samples, eval_samples, sample_info = load_split_ext(bench, args)
        print(f"  {bench} samples {sample_info}", flush=True)
        eval_samples_by_benchmark[bench] = eval_samples
        cache_path = calibration_cache_path(args, bench)
        meta = cache_metadata(args, bench, layers, total_layers, specs, sample_info)
        calib = None
        cache_reused = False
        if not args.no_reuse_calibration_cache:
            calib = load_calibration_cache(cache_path, meta)
            cache_reused = calib is not None
        if calib is None:
            calib = collect_calibration(model, bench, calib_samples, layers, specs, args)
            save_calibration_cache(cache_path, meta, calib)
        if args.require_calib_samples and len(calib) < int(args.calib_samples):
            raise RuntimeError(
                f"{bench} collected {len(calib)} calibration records, "
                f"but --require_calib_samples needs {args.calib_samples}. "
                "Increase --calib_candidates_per_target or relax filters."
            )
        calib_by_benchmark[bench] = calib
        sample_info["calib_kept_count"] = len(calib)
        sample_info["calibration_cache_path"] = str(cache_path)
        sample_info["calibration_cache_reused"] = cache_reused

        eval_cache_path = test_cache_path(args, bench)
        eval_meta = test_cache_metadata(args, bench, layers, total_layers, specs, sample_info)
        test_records = None
        test_cache_reused = False
        if not args.no_reuse_test_cache:
            test_records = load_test_cache(eval_cache_path, eval_meta)
            test_cache_reused = test_records is not None
        if test_records is None:
            test_records = collect_test_records(model, bench, eval_samples, layers, args)
            save_test_cache(eval_cache_path, eval_meta, test_records)
        test_records_by_benchmark[bench] = test_records
        sample_info["test_cache_path"] = str(eval_cache_path)
        sample_info["test_cache_reused"] = test_cache_reused
        sample_info["test_cache_count"] = len(test_records)

        sample_info_by_benchmark[bench] = sample_info
        all_calib.extend(calib)

    if args.cache_only:
        payload = {
            "method": "heldout_gradient_aligned_policy_cache_only",
            "plan": plan,
            "benchmarks": benchmarks,
            "sample_info": sample_info_by_benchmark,
            "budget_labels": [spec["label"] for spec in specs],
            "total_layers": total_layers,
            "fit_layers": layers,
        }
        (out_dir / "cache_only.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps({"wrote_cache_only": str(out_dir), "benchmarks": benchmarks}, indent=2), flush=True)
        return 0

    policies = {}
    global_policy_names = []
    for name, members in global_sets.items():
        records = [
            record
            for bench in members
            for record in calib_by_benchmark[bench]
        ]
        if not records:
            raise RuntimeError(f"Global set {name!r} has zero baseline-correct calibration records")
        policy_name = f"global_{name}" if name != "global" else "global"
        policies[policy_name] = fit_budget_policies(records, layers, total_layers, specs, args, policy_name)
        global_policy_names.append(policy_name)
    if not args.global_only:
        for bench in benchmarks:
            policies[bench] = fit_budget_policies(calib_by_benchmark[bench], layers, total_layers, specs, args, bench)

    validation = {}
    for bench in benchmarks:
        validation[bench] = {}
        for policy_name in global_policy_names:
            validation[bench][policy_name] = validate_policy(
                model, bench, eval_samples_by_benchmark[bench], layers,
                policies[policy_name], specs, args, test_records_by_benchmark[bench])
        if not args.global_only:
            validation[bench][bench] = validate_policy(
                model, bench, eval_samples_by_benchmark[bench], layers,
                policies[bench], specs, args, test_records_by_benchmark[bench])

    payload = {
        "method": "heldout_gradient_aligned_policy_suite",
        "plan": plan,
        "benchmarks": benchmarks,
        "sample_info": sample_info_by_benchmark,
        "budget_labels": [spec["label"] for spec in specs],
        "total_layers": total_layers,
        "fit_layers": layers,
        "global_policy_names": global_policy_names,
        "policies": policies,
        "validation": validation,
    }
    (out_dir / "suite.json").write_text(json.dumps(payload, indent=2))
    write_report(payload, out_dir / "report.md")
    print(json.dumps({"wrote": str(out_dir), "benchmarks": benchmarks}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
