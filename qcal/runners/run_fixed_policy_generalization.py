#!/usr/bin/env python3
"""Evaluate a fixed attention policy across broad local benchmarks.

The policy is loaded from a from-scratch result JSON, but no prior sweep
artifacts are read.  This runner is intended for testing how a learned global
policy, e.g. ChartQA+OCRBench, generalizes across additional benchmark types.
It supports both single-image and multi-image samples.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any

from datasets import load_dataset
from PIL import Image


DUALSIGNAL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = DUALSIGNAL_ROOT.parent
BEA_DIR = WORKSPACE / "qcal_support"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_SOURCE = DUALSIGNAL_ROOT / "results/policy_suite.json"
DEFAULT_OUT = DUALSIGNAL_ROOT / "results/global_generalization"

sys.path.insert(0, str(DUALSIGNAL_ROOT / "runners"))
sys.path.insert(0, str(BEA_DIR))
sys.path.insert(0, str(BEA_DIR / "strategy_test"))
sys.path.insert(0, str(WORKSPACE))

from model_backends import load_backend  # noqa: E402
from models.qwen3vl_wrapper import Qwen2VLWrapper  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from baselines.evaluation.run_broad_comparison import (  # noqa: E402
    eval_metric as bea_eval_metric,
    load_samples as bea_load_samples,
    make_prompt as bea_make_prompt,
    max_new_tokens_map as bea_max_new_tokens_map,
    run_pruned,
)
from baselines.evaluation.run_qwen_original import budget_fraction, budget_specs, maybe_resize, normalize_benchmark  # noqa: E402
from fit_attention_policy_from_scratch import allocate, fuse_policy_scores, layer_attention_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks", nargs="+", default=[
        "realworldqa", "textvqa", "chartqa", "docvqa", "gqa", "ocrbench",
        "scienceqa", "ai2d", "mmmu", "pope", "mme", "hallbench",
        "grounding", "blink", "muirbench", "mmiu", "mantis", "vhs", "needle",
        "mathvista", "mmstar", "mmvet",
    ])
    p.add_argument("--budgets", nargs="+", type=float, default=[10.0, 20.0, 25.0])
    p.add_argument("--eval_samples", type=int, default=16)
    p.add_argument("--eval_offset", type=int, default=192)
    p.add_argument("--num_sweep", type=int, default=256)
    p.add_argument("--model_path", default=str(DEFAULT_MODEL))
    p.add_argument("--backend_type", choices=["auto", "qwen", "llava", "internvl"], default="auto")
    p.add_argument("--source_policy_json", default=str(DEFAULT_SOURCE))
    p.add_argument("--source_policy_name", default="global_co")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--max_layer", type=int, default=None,
                   help="Explicit deepest language-model layer allowed, inclusive. Overrides --max_layer_frac.")
    p.add_argument("--max_layer_frac", type=float, default=0.75,
                   help="Default candidate depth as a fraction of total layers. 0.75 uses the first 75%% of layers.")
    p.add_argument("--max_pixels", type=int, default=1016064)
    p.add_argument("--multi_max_pixels", type=int, default=262144)
    p.add_argument("--resize_square", type=int, default=1008)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else WORKSPACE / p


def load_policy(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return payload["policies"][name]


def load_samples_ext(benchmark: str, num_samples: int) -> list[dict[str, Any]]:
    if benchmark == "hallbench":
        from benchmarks.hallbench_loader import load_hallbench
        return load_hallbench(num_samples)
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
    if benchmark == "vhs":
        from benchmarks.vhs_loader import load_vhs
        return load_vhs(num_samples)
    if benchmark == "needle":
        from benchmarks.needle_loader import load_needle_coco
        return load_needle_coco(num_samples)
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
    return bea_load_samples(benchmark, num_samples)


def pick_eval_samples(benchmark: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = args.eval_offset + args.eval_samples + args.num_sweep
    samples = load_samples_ext(benchmark, requested)
    source_count = len(samples)
    start = args.eval_offset
    if source_count < start + args.eval_samples:
        start = 0
    chosen = samples[start:start + args.eval_samples]
    return chosen, {
        "requested": requested,
        "loaded": source_count,
        "start": start,
        "count": len(chosen),
    }


def sample_images(sample: dict[str, Any], resize_square: int | None) -> list[Image.Image]:
    imgs = []
    if "images" in sample and sample.get("images") is not None:
        raw = sample.get("images")
        if isinstance(raw, (list, tuple)):
            imgs = list(raw)
        else:
            imgs = [raw]
    elif "image" in sample:
        imgs = [sample.get("image")]
    elif "_image_paths" in sample:
        imgs = list(sample.get("images", []))
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
    if not out:
        raise ValueError("no usable images")
    return out


def make_prompt(benchmark: str, sample: dict[str, Any]) -> str:
    if benchmark == "blink":
        return sample.get("question", "") + "\nAnswer with the option letter only."
    if benchmark in ("muirbench", "mmiu", "mantis"):
        q = sample.get("question", "")
        options = sample.get("options", "")
        if isinstance(options, list):
            options = "\n".join(f"{chr(65+i)}. {x}" for i, x in enumerate(options))
        if options and options not in q:
            q = f"{q}\n{options}"
        return q + "\nAnswer with the option letter or short answer."
    if benchmark in ("vhs", "needle"):
        return sample.get("question", "") + "\nAnswer with a short answer."
    if benchmark in ("mathvista", "mmstar"):
        return sample.get("question", "") + "\nAnswer with the option letter or final short answer."
    if benchmark == "mmvet":
        return sample.get("question", "") + "\nAnswer with a short answer."
    return bea_make_prompt(benchmark, sample)


def prepare_inputs_any(model: Qwen2VLWrapper, images: list[Image.Image], prompt: str,
                       max_pixels: int | None, multi_max_pixels: int | None):
    if hasattr(model, "prepare_inputs_any"):
        return model.prepare_inputs_any(images, prompt, max_pixels, multi_max_pixels)
    if len(images) == 1:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": images[0], "max_pixels": max_pixels},
            {"type": "text", "text": prompt},
        ]}]
    else:
        content = []
        for image in images:
            entry = {"type": "image", "image": image}
            if multi_max_pixels is not None:
                entry["max_pixels"] = multi_max_pixels
            content.append(entry)
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
    text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return model.processor(text=[text], images=image_inputs, return_tensors="pt").to(model.model.device)


def generate_ids_and_text(model: Qwen2VLWrapper, inputs, max_new_tokens: int) -> tuple[Any, str]:
    if hasattr(model, "generate_ids_and_text"):
        return model.generate_ids_and_text(inputs, max_new_tokens)
    output_ids = model.model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    text = model.processor.decode(generated, skip_special_tokens=True).strip()
    return generated, text


def max_new_tokens_map(benchmark: str) -> int:
    if benchmark in ("mathvista", "mmvet"):
        return 64
    return bea_max_new_tokens_map(benchmark)


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


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
    for a in [x.strip().strip("()").lower() for x in str(answer).split("<AND>")]:
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
    if benchmark == "mathvista":
        return mathvista_acc(output, sample)
    if benchmark == "mmvet":
        return mmvet_acc(output, sample)
    if benchmark in ("blink", "muirbench", "mmiu", "mantis", "vhs", "needle", "mmstar"):
        return generic_choice_acc(output, sample.get("answer", ""), sample.get("choices") or sample.get("options"))
    if benchmark == "hallbench":
        return bea_eval_metric("hallbench", output, sample)
    return bea_eval_metric(benchmark, output, sample)


def summarize(records: list[dict[str, Any]], specs: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"budgets": {}}
    for spec in specs:
        label = spec["label"]
        scores, fidelities, kept = [], [], []
        errors = 0
        for row in records:
            cell = row.get("budgets", {}).get(label, {})
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
        out["budgets"][label] = {
            "accuracy": sum(scores) / len(scores) if scores else None,
            "valid": len(scores),
            "fidelity": sum(fidelities) / len(fidelities) if fidelities else None,
            "fidelity_valid": len(fidelities),
            "mean_kept_tokens": sum(kept) / len(kept) if kept else None,
            "errors": errors,
        }
    return out


def validate_benchmark(model: Qwen2VLWrapper, benchmark: str, samples: list[dict[str, Any]],
                       layers: list[int], policy: dict[str, Any],
                       specs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    max_tok = max_new_tokens_map(benchmark)
    records = []
    t0 = time.time()
    for idx, sample in enumerate(samples):
        try:
            images = sample_images(sample, args.resize_square)
            inputs = prepare_inputs_any(model, images, make_prompt(benchmark, sample),
                                        args.max_pixels, args.multi_max_pixels)
            vis_embeds, grid_thw = model.extract_visual_embeddings(inputs)
            n_vis = int(vis_embeds.shape[0])
            if n_vis == 0:
                continue
            _, baseline_text = generate_ids_and_text(model, inputs, max_tok)
            baseline_score = eval_metric(benchmark, baseline_text, sample)
            layer_scores = {
                str(layer): scores
                for layer, scores in layer_attention_scores(model, inputs, layers).items()
            }
            row = {
                "benchmark": benchmark,
                "sample_index": idx,
                "num_images": len(images),
                "num_vis": n_vis,
                "baseline": baseline_score,
                "baseline_text": baseline_text,
                "budgets": {},
            }
            for spec in specs:
                label = spec["label"]
                p = policy["by_budget"][label]
                fused = fuse_policy_scores(layer_scores, p["active_layers"], p["active_weights"], n_vis)
                try:
                    keep_mask = allocate(fused, n_vis, budget_fraction(spec, n_vis))
                    if hasattr(model, "run_pruned"):
                        pred = model.run_pruned(inputs, vis_embeds, grid_thw, keep_mask, max_tok)
                    else:
                        pred = run_pruned(model, inputs, vis_embeds, grid_thw, keep_mask, max_tok)
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
                }
            records.append(row)
            avg = (time.time() - t0) / max(1, idx + 1)
            print(f"  {benchmark} fixed {idx + 1}/{len(samples)} avg={avg:.1f}s", flush=True)
        except Exception:
            print(traceback.format_exc(), flush=True)
    return {"records": records, "summary": summarize(records, specs)}


def write_report(payload: dict[str, Any], path: Path) -> None:
    labels = payload["budget_labels"]
    lines = [
        "# Fixed Policy Broad Generalization",
        "",
        f"Policy: `{payload['plan']['source_policy_name']}` from `{payload['plan']['source_policy_json']}`.",
        "",
        "| Benchmark | Samples | Baseline acc | " + " | ".join(f"{label}% fidelity" for label in labels) + " |",
        "|---|---:|---:|" + "---:|" * len(labels),
    ]
    for bench, result in payload["validation"].items():
        summary = result["summary"]["budgets"]
        base_scores = [r["baseline"] for r in result["records"] if isinstance(r.get("baseline"), (int, float))]
        base = sum(base_scores) / len(base_scores) if base_scores else None
        vals = []
        for label in labels:
            v = summary[label]["fidelity"]
            vals.append(f"{v:.3f}" if isinstance(v, (int, float)) else "")
        lines.append(
            f"| {bench} | {len(result['records'])} | "
            f"{base:.3f} | " + " | ".join(vals) + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    benchmarks = [normalize_benchmark(x) for x in args.benchmarks]
    specs = budget_specs(args.budgets, [])
    labels = [spec["label"] for spec in specs]
    policy = load_policy(resolve(args.source_policy_json), args.source_policy_name)
    plan = {
        "model_path": args.model_path,
        "backend_type": args.backend_type,
        "benchmarks": benchmarks,
        "budgets": specs,
        "eval_samples": args.eval_samples,
        "eval_offset": args.eval_offset,
        "max_layer": args.max_layer,
        "max_layer_frac": args.max_layer_frac,
        "source_policy_json": str(resolve(args.source_policy_json)),
        "source_policy_name": args.source_policy_name,
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

    validation = {}
    sample_info = {}
    for bench in benchmarks:
        try:
            samples, info = pick_eval_samples(bench, args)
            sample_info[bench] = info
            if not samples:
                raise RuntimeError(f"no samples loaded for {bench}")
            validation[bench] = validate_benchmark(model, bench, samples, layers, policy, specs, args)
        except Exception:
            sample_info[bench] = {"error": traceback.format_exc()}
            print(traceback.format_exc(), flush=True)

    payload = {
        "method": "fixed_policy_broad_generalization",
        "plan": plan,
        "sample_info": sample_info,
        "benchmarks": benchmarks,
        "budget_labels": labels,
        "total_layers": total_layers,
        "fit_layers": layers,
        "policy": policy,
        "validation": validation,
    }
    (out_dir / "suite.json").write_text(json.dumps(payload, indent=2))
    write_report(payload, out_dir / "report.md")
    print(json.dumps({"wrote": str(out_dir), "benchmarks": list(validation)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
