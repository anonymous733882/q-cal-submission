#!/usr/bin/env python3
"""Run paper-anchored Qwen2.5-VL reproductions without modifying BEA.

This runner imports BEA modules as read-only reference code. All method
selection, budgets, outputs and verification artifacts live under dualsignal.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info


WORKSPACE = Path(__file__).resolve().parents[2]
DUALSIGNAL_ROOT = WORKSPACE / "qcal"
BEA_DIR = WORKSPACE / "qcal_support"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_OUT = DUALSIGNAL_ROOT / "results/original_qwen"

sys.path.insert(0, str(BEA_DIR))
sys.path.insert(0, str(BEA_DIR / "strategy_test"))
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(DUALSIGNAL_ROOT / "runners"))

from models.qwen3vl_wrapper import Qwen2VLWrapper  # noqa: E402
from baselines.idselection.adapter import IDSelectionBaseline  # noqa: E402
from baselines.ptp.adapter import PTPBaseline  # noqa: E402
from baselines.svdprune.adapter import SVDPruneBaseline  # noqa: E402
from baselines.evaluation.run_broad_comparison import (  # noqa: E402
    eval_metric,
    load_split,
    make_prompt,
    max_new_tokens_map,
    run_pruned,
)
from baselines.d2pruner.paper_control import D2PrunerPaper  # noqa: E402
from baselines.hawk.paper_control import HAWKPaper  # noqa: E402
from baselines.idselection.importance_control import IDImportanceTopK  # noqa: E402
from baselines.zspaprune.paper_control import ZSPAPrunePaper  # noqa: E402
from qcal.runners.layer_attention_policy import LayerAttentionPolicyPruner  # noqa: E402


def _load_local_agilepruner():
    path = WORKSPACE / "baselines/agilepruner/paper_adapter.py"
    spec = importlib.util.spec_from_file_location("dualsignal_agilepruner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load AgilePruner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AgilePrunerBaseline


AgilePrunerBaseline = _load_local_agilepruner()


METHODS = {
    "D2Pruner",
    "ID-Selection",
    "ID-Importance",
    "ZSPAPrune",
    "HAWK",
    "AttentionPolicy-Global",
    "AttentionPolicy-Benchmark",
    "SVD-Prune",
    "PTP-Qwen-Adaptation",
    "AgilePruner",
}

PAPER_QWEN_METHODS = {"D2Pruner", "ID-Selection", "ID-Importance", "ZSPAPrune", "HAWK", "AgilePruner"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", default=sorted(PAPER_QWEN_METHODS))
    p.add_argument("--benchmarks", nargs="+", default=["pope", "textvqa"])
    p.add_argument("--budgets", nargs="+", type=float, default=None,
                   help="Percent of visual tokens to retain.")
    p.add_argument("--retain_tokens", nargs="*", type=int, default=[],
                   help="Absolute visual-token counts to retain, e.g. 512 256 128.")
    p.add_argument("--max_pixels", type=int, default=None,
                   help="Optional Qwen image max_pixels. HAWK fixed-resolution anchor uses 1008*1008=1016064.")
    p.add_argument("--resize_square", type=int, default=None,
                   help="Optionally resize every input image to SIZE x SIZE before processing.")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--sample_strategy", choices=["shuffle_eval", "paper_order", "category_balanced"], default="shuffle_eval")
    p.add_argument("--num_sweep", type=int, default=200,
                   help="Sweep split size used by --sample_strategy shuffle_eval.")
    p.add_argument("--pope_category", choices=["all", "adversarial", "popular", "random"], default="all")
    p.add_argument("--pope_prompt_yesno", action="store_true",
                   help="Append the official yes/no instruction to POPE prompts.")
    p.add_argument("--pope_metric", choices=["accuracy", "f1"], default="accuracy",
                   help="Summary metric for POPE. Per-sample scores remain accuracy.")
    p.add_argument("--model_path", default=str(DEFAULT_MODEL))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--sample_offset", type=int, default=0)
    p.add_argument("--allow_adaptation", action="store_true",
                   help="Allow Qwen runs for methods whose paper anchor is not Qwen.")
    p.add_argument("--zspa_core_ratio", type=float, default=None,
                   help="Override ZSPAPrune core-diversity ratio. If unset, paper benchmark defaults are used.")
    p.add_argument("--d2_bias_path", default=None,
                   help="Optional D2Pruner attention-bias prior generated under dualsignal.")
    p.add_argument("--hawk_weights_path", default=None,
                   help="Optional calibrated HAWK head weights JSON generated under dualsignal.")
    p.add_argument("--attention_policy_json", default=None,
                   help="Strategy JSON generated by summarize_attention_policy_strategies.py.")
    p.add_argument("--attention_policy_model", default="qwen25vl",
                   help="Model key inside --attention_policy_json.")
    p.add_argument("--agile_score_layer", type=int, default=1,
                   help="LLM layer used by AgilePruner for text-to-visual attention.")
    p.add_argument("--agile_erank_avg", type=float, default=16.0,
                   help="Calibration-set average effective rank for AgilePruner threshold normalization.")
    p.add_argument("--agile_tau_max", type=float, default=0.95,
                   help="Maximum cosine-distance threshold for AgilePruner.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def zspa_core_ratio_for_benchmark(benchmark: str, override: float | None) -> float:
    if override is not None:
        return override
    paper_defaults = {
        "mmmu": 0.4,
        "gqa": 0.1,
        "pope": 0.2,
    }
    return paper_defaults.get(benchmark, 0.6)


def build_method(name: str, model: Qwen2VLWrapper, benchmark: str,
                 args: argparse.Namespace):
    if name == "D2Pruner":
        return D2PrunerPaper(model, score_layer=2, pivot_ratio=0.7,
                             sim_threshold=0.8, bias_path=args.d2_bias_path)
    if name == "ID-Selection":
        return IDSelectionBaseline(model, llm_layer=2, gamma=20.0)
    if name == "ID-Importance":
        return IDImportanceTopK(model, llm_layer=2, gamma=20.0)
    if name == "ZSPAPrune":
        return ZSPAPrunePaper(
            core_ratio=zspa_core_ratio_for_benchmark(
                benchmark, args.zspa_core_ratio))
    if name == "HAWK":
        return HAWKPaper(model, weights_path=args.hawk_weights_path)
    if name in {"AttentionPolicy-Global", "AttentionPolicy-Benchmark"}:
        if not args.attention_policy_json:
            raise ValueError(f"{name} requires --attention_policy_json")
        policy = json.loads(Path(args.attention_policy_json).read_text())
        model_policy = policy["models"][args.attention_policy_model]
        fit = (
            model_policy["global"]
            if name == "AttentionPolicy-Global"
            else model_policy["by_benchmark"][benchmark]
        )
        active = fit["active_layers"]
        return LayerAttentionPolicyPruner(
            model,
            layers=[int(x["layer"]) for x in active],
            weights=[float(x["weight"]) for x in active],
        )
    if name == "SVD-Prune":
        return SVDPruneBaseline(epsilon=0.9)
    if name == "PTP-Qwen-Adaptation":
        return PTPBaseline(model, vit_block_idx=31, refine_layer=2, alpha=0.5)
    if name == "AgilePruner":
        return AgilePrunerBaseline(
            model,
            score_layer=args.agile_score_layer,
            erank_avg=args.agile_erank_avg,
            tau_max=args.agile_tau_max,
        )
    raise ValueError(f"Unsupported method: {name}")


def normalize_benchmark(name: str) -> str:
    aliases = {
        "scienceqa-img": "scienceqa",
        "sqa-img": "scienceqa",
        "realworldqa": "realworldqa",
        "textvqa": "textvqa",
        "chartqa": "chartqa",
        "docvqa": "docvqa",
        "ocrvqa": "ocrbench",
        "ocrbench": "ocrbench",
        "pope": "pope",
        "mme": "mme",
        "gqa": "gqa",
        "ai2d": "ai2d",
        "mmmu": "mmmu",
    }
    key = name.lower()
    return aliases.get(key, key)


def load_image(sample: dict[str, Any]) -> Image.Image:
    image = sample.get("image")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise ValueError("sample has no image")


def maybe_resize(image: Image.Image, size: int | None) -> Image.Image:
    if size is None:
        return image
    if image.size == (size, size):
        return image
    return image.resize((size, size), Image.Resampling.BICUBIC)


def normalize_yes_no(text: str) -> str:
    value = str(text).strip().lower()
    if value.startswith("yes"):
        return "yes"
    if value.startswith("no"):
        return "no"
    words = set(re.findall(r"[a-z]+", value))
    if "yes" in words and "no" not in words:
        return "yes"
    if "no" in words and "yes" not in words:
        return "no"
    return "unknown"


def pope_f1(pairs: list[tuple[str, str]]) -> dict[str, float]:
    tp = fp = tn = fn = invalid = 0
    for pred_text, gold_text in pairs:
        pred = normalize_yes_no(pred_text)
        gold = normalize_yes_no(gold_text)
        if pred not in {"yes", "no"}:
            invalid += 1
            pred = "no"
        if gold == "yes" and pred == "yes":
            tp += 1
        elif gold == "no" and pred == "yes":
            fp += 1
        elif gold == "no" and pred == "no":
            tn += 1
        elif gold == "yes" and pred == "no":
            fn += 1
    total = tp + fp + tn + fn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    acc = (tp + tn) / max(1, total)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": (tp + fp) / max(1, total),
        "invalid": invalid,
        "total": total,
    }


def load_pope_paper_samples(
    num_samples: int,
    sample_offset: int,
    category: str,
    strategy: str,
) -> list[dict[str, Any]]:
    ds = load_dataset("lmms-lab/POPE", split="test")
    rows = []
    for item in ds:
        if category != "all" and item.get("category") != category:
            continue
        image = item.get("image")
        if image is None:
            continue
        if image.mode != "RGB":
            image = image.convert("RGB")
        rows.append({
            "image": image,
            "question": item["question"],
            "answer": item["answer"],
            "category": item.get("category", "unknown"),
            "question_id": item.get("question_id"),
            "image_source": item.get("image_source"),
        })

    if strategy == "paper_order":
        return rows[sample_offset:sample_offset + num_samples]

    if strategy == "category_balanced" and category == "all":
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_cat.setdefault(row["category"], []).append(row)
        cats = [c for c in ("adversarial", "popular", "random") if c in by_cat]
        selected = []
        base = num_samples // max(1, len(cats))
        extra = num_samples % max(1, len(cats))
        for idx, cat in enumerate(cats):
            take = base + (1 if idx < extra else 0)
            start = sample_offset
            selected.extend(by_cat[cat][start:start + take])
        return selected[:num_samples]

    return rows[sample_offset:sample_offset + num_samples]


def budget_specs(percent_budgets: list[float], retain_tokens: list[int]) -> list[dict[str, Any]]:
    specs = [{"label": str(float(b)), "type": "percent", "value": float(b)}
             for b in percent_budgets]
    specs.extend({"label": f"tokens_{int(t)}", "type": "tokens", "value": int(t)}
                 for t in retain_tokens)
    return specs


def budget_fraction(spec: dict[str, Any], n_vis: int) -> float:
    if spec["type"] == "tokens":
        return max(0.0, min(1.0, float(spec["value"]) / max(1, n_vis)))
    return max(0.0, min(1.0, float(spec["value"]) / 100.0))


@torch.no_grad()
def prepare_inputs(model: Qwen2VLWrapper, image: Image.Image, prompt: str,
                   max_pixels: int | None):
    if max_pixels is None:
        return model.prepare_inputs(image, prompt)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image, "max_pixels": max_pixels},
        {"type": "text", "text": prompt},
    ]}]
    text = model.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return model.processor(
        text=[text], images=image_inputs, return_tensors="pt"
    ).to(model.model.device)


@torch.no_grad()
def generate_from_inputs(model: Qwen2VLWrapper, inputs, max_new_tokens: int) -> str:
    output_ids = model.model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return model.processor.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def eval_one(model: Qwen2VLWrapper, sample: dict[str, Any], benchmark: str,
             methods: dict[str, Any], specs: list[dict[str, Any]],
             max_pixels: int | None, resize_square: int | None,
             pope_prompt_yesno: bool = False) -> dict[str, Any] | None:
    image = maybe_resize(load_image(sample), resize_square)
    sample = dict(sample, image=image)
    prompt = make_prompt(benchmark, sample)
    if benchmark == "pope" and pope_prompt_yesno and "yes or no" not in prompt.lower():
        prompt = prompt + " Answer yes or no."
    max_tok = max_new_tokens_map(benchmark)

    inputs = prepare_inputs(model, image, prompt, max_pixels)
    vis_embeds, grid_thw = model.extract_visual_embeddings(inputs)
    n_vis = int(vis_embeds.shape[0])
    if n_vis == 0:
        return None

    baseline_text = generate_from_inputs(model, inputs, max_tok)
    baseline_score = eval_metric(benchmark, baseline_text, sample)
    row: dict[str, Any] = {
        "benchmark": benchmark,
        "num_vis": n_vis,
        "max_pixels": max_pixels,
        "resize_square": resize_square,
        "baseline": baseline_score,
        "baseline_text": baseline_text,
        "answer": sample.get("answer", sample.get("label", "")),
        "category": sample.get("category"),
        "question_id": sample.get("question_id"),
        "methods": {},
    }

    for method_name, method in methods.items():
        row["methods"][method_name] = {}
        for spec in specs:
            budget_frac = budget_fraction(spec, n_vis)
            try:
                if method_name == "HAWK":
                    scores = method.score_tokens(inputs)
                    keep_mask = method.allocate(scores, n_vis, budget_frac)
                elif method_name in {"AttentionPolicy-Global", "AttentionPolicy-Benchmark"}:
                    scores = method.score_tokens(inputs)
                    keep_mask = method.allocate(scores, n_vis, budget_frac)
                elif method_name == "SVD-Prune":
                    scores = method.score_tokens(vis_embeds)
                    keep_mask = method.allocate(scores, n_vis, budget_frac)
                else:
                    keep_mask = method.prune(
                        model, inputs, vis_embeds, grid_thw,
                        prune_ratio=1.0 - budget_frac,
                    )
                pred = run_pruned(model, inputs, vis_embeds, grid_thw,
                                  keep_mask, max_tok)
                score = eval_metric(benchmark, pred, sample)
                fidelity = (
                    min(1.0, float(score) / float(baseline_score))
                    if baseline_score and baseline_score > 0 and score is not None
                    else None
                )
                kept = sum(1 for v in keep_mask if v)
                err = None
            except Exception as exc:  # keep long sweeps alive
                pred = ""
                score = None
                fidelity = None
                kept = None
                err = f"{type(exc).__name__}: {exc}"
            row["methods"][method_name][spec["label"]] = {
                "budget": spec,
                "budget_fraction": budget_frac,
                "score": score,
                "fidelity": fidelity,
                "prediction": pred,
                "kept_tokens": kept,
                "error": err,
            }
    return row


def summarize(records: list[dict[str, Any]], methods: list[str],
              specs: list[dict[str, Any]], benchmark: str = "",
              pope_metric: str = "accuracy") -> dict[str, Any]:
    out: dict[str, Any] = {"num_samples": len(records), "methods": {}}
    if records:
        out["baseline_mean"] = sum(r["baseline"] for r in records) / len(records)
    else:
        out["baseline_mean"] = None
    for method in methods:
        out["methods"][method] = {}
        for spec in specs:
            label = spec["label"]
            vals = []
            fidelities = []
            kept = []
            errors = 0
            for r in records:
                cell = r.get("methods", {}).get(method, {}).get(label, {})
                if cell.get("score") is not None:
                    vals.append(float(cell["score"]))
                if cell.get("fidelity") is not None:
                    fidelities.append(float(cell["fidelity"]))
                if cell.get("kept_tokens") is not None:
                    kept.append(int(cell["kept_tokens"]))
                if cell.get("error"):
                    errors += 1
            out["methods"][method][label] = {
                "budget": spec,
                "mean": sum(vals) / len(vals) if vals else None,
                "valid": len(vals),
                "fidelity_on_baseline_positive": (
                    sum(fidelities) / len(fidelities) if fidelities else None
                ),
                "fidelity_valid": len(fidelities),
                "errors": errors,
                "mean_kept_tokens": sum(kept) / len(kept) if kept else None,
            }
            if benchmark == "pope":
                pairs = []
                for r in records:
                    cell = r.get("methods", {}).get(method, {}).get(label, {})
                    pairs.append((str(cell.get("prediction", "")), str(r.get("answer", ""))))
                metrics = pope_f1(pairs)
                out["methods"][method][label]["pope_metrics"] = metrics
                if pope_metric == "f1":
                    out["methods"][method][label]["mean"] = metrics["f1"]
        if benchmark == "pope":
            baseline_metrics = pope_f1([
                (str(r.get("baseline_text", "")), str(r.get("answer", "")))
                for r in records
            ])
            out["baseline_pope_metrics"] = baseline_metrics
            if pope_metric == "f1":
                out["baseline_mean"] = baseline_metrics["f1"]
    return out


def main() -> int:
    args = parse_args()
    requested = list(dict.fromkeys(args.methods))
    unknown = [m for m in requested if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Supported: {sorted(METHODS)}")
    nonpaper = [m for m in requested if m not in PAPER_QWEN_METHODS]
    if nonpaper and not args.allow_adaptation:
        raise SystemExit(
            f"{nonpaper} are not Qwen original-paper anchors. "
            "Use --allow_adaptation only for exploratory runs."
        )

    benchmarks = [normalize_benchmark(b) for b in args.benchmarks]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = WORKSPACE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    percent_budgets = args.budgets if args.budgets is not None else (
        [] if args.retain_tokens else [10.0])
    specs = budget_specs(percent_budgets, args.retain_tokens)
    plan = {
        "model_path": args.model_path,
        "methods": requested,
        "benchmarks": benchmarks,
        "budgets": specs,
        "max_pixels": args.max_pixels,
        "resize_square": args.resize_square,
        "zspa_core_ratio": args.zspa_core_ratio,
        "d2_bias_path": args.d2_bias_path,
        "hawk_weights_path": args.hawk_weights_path,
        "agile_score_layer": args.agile_score_layer,
        "agile_erank_avg": args.agile_erank_avg,
        "agile_tau_max": args.agile_tau_max,
        "num_samples": args.num_samples,
        "sample_strategy": args.sample_strategy,
        "num_sweep": args.num_sweep,
        "pope_category": args.pope_category,
        "pope_prompt_yesno": args.pope_prompt_yesno,
        "pope_metric": args.pope_metric,
        "out_dir": str(out_dir),
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    model = Qwen2VLWrapper(args.model_path)
    all_summary: dict[str, Any] = {"plan": plan, "benchmarks": {}}
    for benchmark in benchmarks:
        methods = {name: build_method(name, model, benchmark, args)
                   for name in requested}
        print(f"\n[{benchmark}] loading samples", flush=True)
        if benchmark == "pope" and args.sample_strategy != "shuffle_eval":
            samples = load_pope_paper_samples(
                args.num_samples,
                sample_offset=args.sample_offset,
                category=args.pope_category,
                strategy=args.sample_strategy,
            )
        else:
            samples = load_split(
                benchmark, split="eval",
                num_eval=args.num_samples + args.sample_offset,
                num_sweep=args.num_sweep,
            )[args.sample_offset:args.sample_offset + args.num_samples]
        records = []
        t0 = time.time()
        for idx, sample in enumerate(samples):
            try:
                row = eval_one(model, sample, benchmark, methods, specs,
                               args.max_pixels, args.resize_square,
                               pope_prompt_yesno=args.pope_prompt_yesno)
                if row is not None:
                    records.append(row)
                avg = (time.time() - t0) / max(1, idx + 1)
                print(f"  {idx + 1}/{len(samples)} avg={avg:.1f}s", flush=True)
            except Exception:
                print(traceback.format_exc(), flush=True)
        payload = {
            "plan": plan,
            "benchmark": benchmark,
            "records": records,
            "summary": summarize(records, requested, specs, benchmark, args.pope_metric),
        }
        out_file = out_dir / f"{benchmark}.json"
        out_file.write_text(json.dumps(payload, indent=2))
        all_summary["benchmarks"][benchmark] = payload["summary"]
        print(f"  wrote {out_file}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(all_summary, indent=2))
    print(f"\nwrote {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
