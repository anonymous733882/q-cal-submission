#!/usr/bin/env python3
"""Run one multi-round benchmark with both stage-2 policies in one process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image


DUALSIGNAL_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = DUALSIGNAL_DIR.parent
BEA_DIR = WORKSPACE / "qcal_support"

sys.path.insert(0, str(DUALSIGNAL_DIR / "runners"))
sys.path.insert(0, str(BEA_DIR))
sys.path.insert(0, str(BEA_DIR / "strategy_test"))

import multi_round_benchmark as bea_mr  # noqa: E402
from run_stage2_policy_multiround import (  # noqa: E402
    _patch_per_question_prune_generate,
    _patch_stage1_diversity,
)
from stage2_policies import calibrated_global_policy, original_attention_policy  # noqa: E402
from count_relaxed_scoring import relaxed_count_score  # noqa: E402


POLICY_NAMES = ("original_attention", "calibrated_global")


def _patch_relaxed_count_scorer() -> None:
    original_score_gqa = bea_mr.score_gqa

    def patched_score_gqa(pred, gt_answer):
        relaxed = relaxed_count_score(pred, gt_answer)
        if relaxed is not None:
            return float(relaxed)
        return original_score_gqa(pred, gt_answer)

    bea_mr.score_gqa = patched_score_gqa


def _policies_for(model_key: str):
    return {
        "original_attention": original_attention_policy(model_key),
        "calibrated_global": calibrated_global_policy(model_key),
    }


def _maybe_resize_image(image, resize_square: int | None):
    if resize_square is None:
        return image
    if not isinstance(image, Image.Image):
        return image
    img = image.convert("RGB")
    target = (int(resize_square), int(resize_square))
    if img.size == target:
        return img
    return img.resize(target, Image.Resampling.BICUBIC)


def _patch_qwen_wrapper_image_options(resize_square: int | None, max_pixels: int | None) -> None:
    if resize_square is None and max_pixels is None:
        return
    try:
        import torch
        import models.qwen3vl_wrapper as qwen_wrap
    except Exception as exc:
        print(f"Qwen wrapper image-option patch skipped: {exc}", flush=True)
        return

    cls = qwen_wrap.Qwen2VLWrapper
    patch_key = (resize_square, max_pixels)
    if getattr(cls, "_dualsignal_image_options_patch", None) == patch_key:
        return

    @torch.no_grad()
    def generate(self, image, prompt, max_new_tokens=256):
        image = _maybe_resize_image(image, resize_square)
        image_entry = {"type": "image", "image": image}
        if max_pixels is not None:
            image_entry["max_pixels"] = int(max_pixels)
        messages = [
            {
                "role": "user",
                "content": [
                    image_entry,
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = qwen_wrap.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    @torch.no_grad()
    def prepare_inputs(self, image, prompt):
        image = _maybe_resize_image(image, resize_square)
        image_entry = {"type": "image", "image": image}
        if max_pixels is not None:
            image_entry["max_pixels"] = int(max_pixels)
        messages = [
            {
                "role": "user",
                "content": [
                    image_entry,
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = qwen_wrap.process_vision_info(messages)
        return self.processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to(self.model.device)

    cls.generate = generate
    cls.prepare_inputs = prepare_inputs
    cls._dualsignal_image_options_patch = patch_key
    print(
        "Applied Qwen image options: "
        f"resize_square={resize_square}, max_pixels={max_pixels}",
        flush=True,
    )


def _apply_dual_config(model_key: str):
    policies = _policies_for(model_key)
    cfg = dict(bea_mr.MODEL_CONFIG[model_key])
    methods = []
    max_s = 0
    for name in POLICY_NAMES:
        policy = policies[name]
        max_s = max(max_s, int(policy.prune_start_layer))
        methods.append(
            {
                "output_name": name,
                "policy": policy,
                "S": int(policy.prune_start_layer),
                "policy_json": policy.to_json(),
            }
        )

    cfg["S"] = max_s
    cfg["scoring"] = "policy"
    cfg["policy_only"] = True
    cfg["policy_methods"] = methods
    cfg["policy_name"] = "dual_stage2_policies"
    cfg["policy_kind"] = "dual"
    cfg["prune_start_layer"] = max_s
    cfg["cost"] = bea_mr.STAGE1_FRAC * max_s + bea_mr.STAGE2_FRAC * (cfg["N"] - max_s)
    cfg["Y"] = cfg["cost"] / cfg["N"]
    bea_mr.MODEL_CONFIG[model_key] = cfg

    # The patched generator reads bea_mr._active_stage2_policy, which the BEA
    # per-question loop sets before each policy call.
    _patch_stage1_diversity()
    _patch_per_question_prune_generate(policies["original_attention"])
    return cfg, policies


def _patch_dual_outputs(out_root: Path, benchmark: str, policies: dict):
    def dual_report(model_name, cfg, results, diversity=None, costs=None):
        baseline = float(np.mean(results["baseline"])) if results.get("baseline") else 0.0
        print("\n" + "=" * 60)
        print(f"Dual-policy multi-round report: {model_name}")
        print("=" * 60)
        print(f"Baseline: {baseline:.3f}")
        for name in POLICY_NAMES:
            score = float(np.mean(results[name])) if results.get(name) else 0.0
            preservation = score / baseline if baseline > 0 else 0.0
            print(f"{name}: {score:.3f} ({preservation:.1%} of baseline)")

    def dual_save(model_name, cfg, results, diversity, out_dir, costs=None):
        baseline_values = results.get("baseline", [])
        baseline = float(np.mean(baseline_values)) if baseline_values else 0.0
        for name in POLICY_NAMES:
            policy_values = results.get(name, [])
            score = float(np.mean(policy_values)) if policy_values else 0.0
            entry = {
                "score": score,
                "preservation": score / baseline if baseline > 0 else 0.0,
                "per_sample": [float(x) for x in policy_values],
                "source_internal_method": name,
                "stage2_policy": policies[name].to_json(),
            }
            if costs and name in costs and costs[name].get("total_tl"):
                entry["cost"] = {
                    "prefill_tl_mean": float(np.mean(costs[name]["prefill_tl"])),
                    "gen_tl_mean": float(np.mean(costs[name]["gen_tl"])),
                    "total_tl_mean": float(np.mean(costs[name]["total_tl"])),
                    "n_vis_decode_mean": float(np.mean(costs[name]["n_vis_decode"])),
                    "n_gen_tokens_mean": float(np.mean(costs[name]["n_gen_tokens"])),
                    "prefill_tl_per_sample": [float(x) for x in costs[name]["prefill_tl"]],
                    "gen_tl_per_sample": [float(x) for x in costs[name]["gen_tl"]],
                    "total_tl_per_sample": [float(x) for x in costs[name]["total_tl"]],
                    "n_vis_decode_per_sample": [float(x) for x in costs[name]["n_vis_decode"]],
                    "n_gen_tokens_per_sample": [float(x) for x in costs[name]["n_gen_tokens"]],
                    "shared_setup_sec_mean": float(np.mean(costs[name]["shared_setup_sec"])),
                    "shared_question_sec_mean": float(np.mean(costs[name]["shared_question_sec"])),
                    "policy_prefill_wall_sec_mean": float(np.mean(costs[name]["policy_prefill_wall_sec"])),
                    "policy_decode_wall_sec_mean": float(np.mean(costs[name]["policy_decode_wall_sec"])),
                    "policy_total_wall_sec_mean": float(np.mean(costs[name]["policy_total_wall_sec"])),
                    "shared_setup_sec_per_sample": [float(x) for x in costs[name]["shared_setup_sec"]],
                    "shared_question_sec_per_sample": [float(x) for x in costs[name]["shared_question_sec"]],
                    "policy_prefill_wall_sec_per_sample": [float(x) for x in costs[name]["policy_prefill_wall_sec"]],
                    "policy_decode_wall_sec_per_sample": [float(x) for x in costs[name]["policy_decode_wall_sec"]],
                    "policy_total_wall_sec_per_sample": [float(x) for x in costs[name]["policy_total_wall_sec"]],
                }
            else:
                entry["cost"] = {"available": False}

            data = {
                "model": model_name,
                "config": {
                    k: v for k, v in cfg.items()
                    if k not in ("type", "policy_methods")
                },
                "n_samples": len(baseline_values),
                "baseline_mean": baseline,
                "baseline_per_sample": [float(x) for x in baseline_values],
                "methods": {name: entry},
                "dual_process_reuse": {
                    "model_loaded_once": True,
                    "dataset_loaded_once": True,
                    "baseline_generated_once": True,
                    "stage1_shared": True,
                },
                "cost_schema": {
                    "prefill_tl": "visual_tokens * transformer_layers for prefill",
                    "gen_tl": "visual_tokens * transformer_layers * generated_tokens",
                    "total_tl": "prefill_tl + gen_tl",
                    "n_vis_decode": "visual tokens kept during decode",
                    "n_gen_tokens": "generated token count",
                    "runtime": "wall-clock seconds for shared dual-policy model×benchmark job",
                    "shared_setup_sec": "shared image/model-side setup amortized per question",
                    "shared_question_sec": "baseline plus question embedding/preparation shared by both policies",
                    "policy_prefill_wall_sec": "policy-specific Stage1 prefix cache build amortized per question",
                    "policy_decode_wall_sec": "policy-specific stage2 pruning and generation time",
                    "policy_total_wall_sec": "shared_setup_sec + shared_question_sec + policy_prefill_wall_sec + policy_decode_wall_sec",
                },
            }
            if costs and "baseline" in costs and costs["baseline"].get("total_tl"):
                data["baseline_cost"] = {
                    "prefill_tl_mean": float(np.mean(costs["baseline"]["prefill_tl"])),
                    "gen_tl_mean": float(np.mean(costs["baseline"]["gen_tl"])),
                    "total_tl_mean": float(np.mean(costs["baseline"]["total_tl"])),
                    "n_vis_decode_mean": float(np.mean(costs["baseline"]["n_vis_decode"])),
                    "n_gen_tokens_mean": float(np.mean(costs["baseline"]["n_gen_tokens"])),
                    "prefill_tl_per_sample": [float(x) for x in costs["baseline"]["prefill_tl"]],
                    "gen_tl_per_sample": [float(x) for x in costs["baseline"]["gen_tl"]],
                    "total_tl_per_sample": [float(x) for x in costs["baseline"]["total_tl"]],
                    "n_vis_decode_per_sample": [float(x) for x in costs["baseline"]["n_vis_decode"]],
                    "n_gen_tokens_per_sample": [float(x) for x in costs["baseline"]["n_gen_tokens"]],
                    "shared_setup_sec_mean": float(np.mean(costs["baseline"]["shared_setup_sec"])),
                    "shared_question_sec_mean": float(np.mean(costs["baseline"]["shared_question_sec"])),
                    "policy_total_wall_sec_mean": float(np.mean(costs["baseline"]["policy_total_wall_sec"])),
                    "shared_setup_sec_per_sample": [float(x) for x in costs["baseline"]["shared_setup_sec"]],
                    "shared_question_sec_per_sample": [float(x) for x in costs["baseline"]["shared_question_sec"]],
                    "policy_total_wall_sec_per_sample": [float(x) for x in costs["baseline"]["policy_total_wall_sec"]],
                }
            else:
                data["baseline_cost"] = {"available": False}

            out_path = out_root / name / benchmark / f"{model_name}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(data, indent=2) + "\n")
            print(f"Saved -> {out_path}", flush=True)

    bea_mr.print_report = dual_report
    bea_mr.save_results = dual_save


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(bea_mr.MODEL_CONFIG.keys()))
    parser.add_argument("--benchmark", required=True,
                        choices=["textvqa", "gqa", "pope", "docvqa", "vqav2", "clevr", "visual7w"])
    parser.add_argument("--num_images", type=int, default=2000)
    parser.add_argument("--questions_per_image", type=int, default=15)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--skip_cclass", action="store_true")
    parser.add_argument("--resize_square", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.resize_square is not None:
        os.environ["DUALSIGNAL_RESIZE_SQUARE"] = str(args.resize_square)
    else:
        os.environ.pop("DUALSIGNAL_RESIZE_SQUARE", None)
    _patch_relaxed_count_scorer()
    _patch_qwen_wrapper_image_options(args.resize_square, args.max_pixels)
    cfg, policies = _apply_dual_config(args.model)
    _patch_dual_outputs(Path(args.out_dir), args.benchmark, policies)

    print("Using dual policies:", flush=True)
    for name in POLICY_NAMES:
        print(f"  {name}: {policies[name].to_json()}", flush=True)
    print(f"Runtime cfg: max_S={cfg['S']} Y={cfg['Y']:.4f}", flush=True)

    sys.argv = [
        "multi_round_benchmark.py",
        "--model", args.model,
        "--benchmark", args.benchmark,
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
