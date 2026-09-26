#!/usr/bin/env python3
"""Recompute same-image/five-question ranking agreement on the fixed VQAv2 set."""

from __future__ import annotations

import argparse
import gc
from itertools import combinations
from pathlib import Path
from statistics import mean, pvariance
import sys

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "qcal/runners"))

from diagnostic_common import (  # noqa: E402
    PROMPT_GROUPS, PROMPT_MODELS, align_scores, compare_rankings,
    decode_group_image, depth_layer, file_sha256, load_model, load_prompt_groups,
    set_seed, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=PROMPT_MODELS, default=list(PROMPT_MODELS))
    parser.add_argument("--limit-images", type=int, default=100)
    parser.add_argument("--max-pixels", type=int, default=1016064)
    parser.add_argument("--deep-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/diagnostics/prompt_invariance")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def svd_scores(model, inputs, n_vis: int) -> torch.Tensor:
    visual, _ = model.extract_visual_embeddings(inputs)
    extra = getattr(inputs, "extra", None)
    if isinstance(extra, dict):
        visual = extra.get("score_visual_embeds", visual)
    features = torch.as_tensor(visual, dtype=torch.float32).reshape(-1, visual.shape[-1]).cpu()
    if features.shape[0] != n_vis:
        raise ValueError(f"SVD signal has {features.shape[0]} visual tokens; expected {n_vis}")
    if n_vis <= 1:
        return torch.ones(n_vis)
    u, singular, _ = torch.linalg.svd(features, full_matrices=False)
    energy = singular.square()
    rank = int(torch.searchsorted(torch.cumsum(energy, 0) / energy.sum().clamp_min(1e-12), 0.9)) + 1
    return u[:, :rank].square().sum(dim=1) / rank


@torch.no_grad()
def run_model(model_id: str, groups: list[dict], args: argparse.Namespace) -> dict:
    model = load_model(model_id)
    total_layers = model.total_layers
    deep = depth_layer(total_layers, args.deep_fraction)
    mid = round((total_layers - 1) * 0.5)
    layers = sorted({0, 1, mid, deep})
    records: dict[str, list[dict]] = {
        key: [] for key in (
            "early_layer0", "early_layer1", "visual_prior", "svd_prior",
            "mid_attention", "deep_attention", "deep_vs_early0", "deep_vs_early1",
            "deep_vs_visual", "deep_vs_svd", "deep_vs_early_family", "deep_vs_visual_family",
        )
    }
    for image_index, group in enumerate(groups):
        image = decode_group_image(group)
        question_scores = []
        first_inputs = None
        for qa in group["qas"]:
            prompt = qa["question"] + "\nAnswer the question using a single word or phrase."
            inputs = model.prepare_inputs_any([image], prompt, args.max_pixels, args.max_pixels)
            raw = model.layer_attention_scores(inputs, layers)
            n_vis = len(raw[0])
            if n_vis <= 1:
                raise ValueError(f"No usable visual tokens for image {group['image_id']}")
            scores = {layer: align_scores(raw[layer], n_vis) for layer in layers}
            if first_inputs is None:
                first_inputs = inputs
                visual = align_scores(model.visual_received_attention_scores(inputs, n_vis), n_vis)
                svd = align_scores(svd_scores(model, inputs, n_vis), n_vis)
            question_scores.append({"question_id": qa["question_id"], "scores": scores})
        for left, right in combinations(question_scores, 2):
            base = {"image_id": group["image_id"], "image_index": image_index,
                    "left_question_id": left["question_id"], "right_question_id": right["question_id"]}
            for name, layer in (("early_layer0", 0), ("early_layer1", 1),
                                ("mid_attention", mid), ("deep_attention", deep)):
                records[name].append({**base, **compare_rankings(left["scores"][layer], right["scores"][layer])})
            for name, prior in (("visual_prior", visual), ("svd_prior", svd)):
                records[name].append({**base, **compare_rankings(prior, prior)})
        for question in question_scores:
            scores = question["scores"]
            base = {"image_id": group["image_id"], "image_index": image_index,
                    "question_id": question["question_id"]}
            for name, left, right in (
                ("deep_vs_early0", scores[deep], scores[0]),
                ("deep_vs_early1", scores[deep], scores[1]),
                ("deep_vs_visual", scores[deep], visual),
                ("deep_vs_svd", scores[deep], svd),
            ):
                record = {**base, **compare_rankings(left, right)}
                records[name].append(record)
                if name in ("deep_vs_early0", "deep_vs_early1"):
                    records["deep_vs_early_family"].append(record)
                elif name in ("deep_vs_visual", "deep_vs_svd"):
                    records["deep_vs_visual_family"].append(record)
        print(f"{model_id}: {image_index + 1}/{len(groups)} images", flush=True)
    summary = {
        name: {"comparisons": len(rows), **{metric: mean(row[metric] for row in rows)
                                          for metric in ("j10", "j20", "spearman")}}
        for name, rows in records.items()
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"model": model_id, "total_layers": total_layers,
            "mid_layer_index": mid, "deep_layer_index": deep,
            "summary": summary, "records": records}


def aggregate(results: list[dict]) -> dict:
    names = results[0]["summary"]
    return {
        name: {
            metric: {
                "mean": mean(row["summary"][name][metric] for row in results),
                "variance_across_models": pvariance(row["summary"][name][metric] for row in results),
            }
            for metric in ("j10", "j20", "spearman")
        }
        for name in names
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.limit_images <= 100 or not 0 < args.deep_fraction <= 1:
        raise SystemExit("Use 1-100 images and a deep fraction in (0,1]")
    plan = {"models": args.models, "images": args.limit_images, "questions_per_image": 5,
            "jaccard_budgets_percent": [10, 20], "deep_fraction": args.deep_fraction,
            "dataset_sha256": file_sha256(PROMPT_GROUPS), "seed": args.seed,
            "ranking": "ordinal torch.argsort ranks, matching the original diagnostic"}
    if args.dry_run:
        print(plan)
        return 0
    set_seed(args.seed)
    groups = load_prompt_groups(args.limit_images)
    results = []
    for model_id in args.models:
        result = run_model(model_id, groups, args)
        write_json(args.out_dir / f"{model_id}.json", {"plan": plan, **result})
        results.append(result)
    write_json(args.out_dir / "summary.json", {"plan": plan, "models": args.models,
                                                "aggregate": aggregate(results)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
