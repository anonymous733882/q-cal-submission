#!/usr/bin/env python3
"""Fit a gradient-aligned attention-layer policy."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils import process_vision_info


DUALSIGNAL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = DUALSIGNAL_ROOT.parent
BEA_DIR = WORKSPACE / "qcal_support"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_OUT = DUALSIGNAL_ROOT / "results/gradient_aligned_policy"

sys.path.insert(0, str(BEA_DIR))
sys.path.insert(0, str(BEA_DIR / "strategy_test"))
sys.path.insert(0, str(WORKSPACE))

from baselines.common.attn_utils import compute_text_vis_attention  # noqa: E402
from models.qwen3vl_wrapper import Qwen2VLWrapper  # noqa: E402
from baselines.evaluation.run_broad_comparison import (  # noqa: E402
    eval_metric,
    load_split,
    make_prompt,
    max_new_tokens_map,
    run_pruned,
)
from baselines.evaluation.run_qwen_original import (  # noqa: E402
    budget_fraction,
    budget_specs,
    load_image,
    maybe_resize,
    normalize_benchmark,
)


IMAGE_TOKEN_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--budgets", nargs="+", type=float, default=[10.0])
    p.add_argument("--retain_tokens", nargs="*", type=int, default=[])
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--sample_offset", type=int, default=0)
    p.add_argument("--num_sweep", type=int, default=200)
    p.add_argument("--model_path", default=str(DEFAULT_MODEL))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--max_pixels", type=int, default=1016064)
    p.add_argument("--resize_square", type=int, default=1008)
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Layers to fit. Defaults to all language-model layers.")
    p.add_argument("--max_active_layers", type=int, default=5,
                   help="Maximum active layers after fitting. Use 0 to keep every layer above --active_weight_threshold.")
    p.add_argument("--active_weight_threshold", type=float, default=1e-8,
                   help="Final policy layers below this normalized weight are treated as inactive.")
    p.add_argument("--rank_temperature", type=float, default=0.08)
    p.add_argument("--oracle_temperature", type=float, default=0.08)
    p.add_argument("--depth_lambda", type=float, default=0.01)
    p.add_argument("--entropy_lambda", type=float, default=0.001,
                   help="Positive value penalizes high entropy, encouraging sparse layer weights.")
    p.add_argument("--l2_lambda", type=float, default=0.001)
    p.add_argument("--optim_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--fit_loss", choices=["kl", "topk_pairwise", "budget_overlap", "budget_overlap_pairwise"], default="kl",
                   help="Layer-weight objective. topk_pairwise and budget_overlap_pairwise are budget-aware.")
    p.add_argument("--max_pairwise_negatives", type=int, default=256,
                   help="Max oracle-negative tokens per sample/budget for topk_pairwise.")
    p.add_argument("--fit_device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Device for the small layer-weight optimization problem.")
    p.add_argument("--fit_per_budget", action="store_true",
                   help="Fit a separate layer-weight policy for each requested percent budget.")
    p.add_argument("--validate_real", action="store_true",
                   help="After fitting, run pruned generation with the learned policy.")
    p.add_argument("--max_answer_tokens_for_grad", type=int, default=32,
                   help="Limit teacher-forced generated tokens used for gradient oracle.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else WORKSPACE / p


def visual_and_text_positions(inputs: dict[str, torch.Tensor]) -> tuple[list[int], list[int]]:
    ids = inputs["input_ids"][0]
    vs = (ids == VISION_START_ID).nonzero(as_tuple=True)[0]
    ve = (ids == VISION_END_ID).nonzero(as_tuple=True)[0]
    if len(vs) > 0 and len(ve) > 0:
        spans = []
        for s in vs.tolist():
            later = [e for e in ve.tolist() if e > s]
            if later:
                spans.append((s, later[0]))
        vis_pos = [
            i
            for start, end in spans
            for i in range(start + 1, end)
            if ids[i] == IMAGE_TOKEN_ID
        ]
        in_vision = set()
        for start, end in spans:
            in_vision.update(range(start, end + 1))
        text_pos = [i for i in range(ids.shape[0]) if i not in in_vision]
        return vis_pos, text_pos
    return [], []


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
    return model.processor(text=[text], images=image_inputs, return_tensors="pt").to(model.model.device)


@torch.no_grad()
def generate_ids_and_text(model: Qwen2VLWrapper, inputs, max_new_tokens: int) -> tuple[torch.Tensor, str]:
    if hasattr(model, "generate_ids_and_text"):
        return model.generate_ids_and_text(inputs, max_new_tokens)
    output_ids = model.model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[0, inputs["input_ids"].shape[1]:].detach()
    text = model.processor.decode(generated, skip_special_tokens=True).strip()
    return generated, text


def norm01_t(values: torch.Tensor) -> torch.Tensor:
    values = values.float().cpu()
    if values.numel() == 0:
        return values
    lo, hi = values.min(), values.max()
    if (hi - lo).abs() <= 1e-9:
        return torch.zeros_like(values)
    return (values - lo) / (hi - lo)


def build_inputs_embeds_with_grad(model: Qwen2VLWrapper, inputs, answer_ids: torch.Tensor,
                                  max_answer_tokens: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """Build prompt+answer inputs_embeds with differentiable visual embeddings."""
    input_ids = inputs["input_ids"]
    answer_ids = answer_ids[:max_answer_tokens].to(input_ids.device)
    if answer_ids.numel() == 0:
        raise ValueError("empty generated answer")

    full_ids = torch.cat([input_ids, answer_ids.unsqueeze(0)], dim=1)
    lm = model.model.model.language_model
    text_embeds = lm.get_input_embeddings()(full_ids).detach()

    with torch.no_grad():
        visual_out = model.model.model.visual(
            inputs["pixel_values"],
            grid_thw=inputs["image_grid_thw"],
        )
        visual_embeds_base = visual_out.pooler_output.detach()
    visual_embeds = visual_embeds_base.clone().requires_grad_(True)

    prompt_text_embeds = text_embeds[:, : input_ids.shape[1], :].clone()
    image_mask, _ = model.model.model.get_placeholder_mask(
        input_ids,
        inputs_embeds=prompt_text_embeds,
        image_features=visual_embeds,
    )
    prompt_embeds = prompt_text_embeds.masked_scatter(image_mask, visual_embeds)
    answer_embeds = text_embeds[:, input_ids.shape[1]:, :]
    inputs_embeds = torch.cat([prompt_embeds, answer_embeds], dim=1)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        extra = torch.ones((attention_mask.shape[0], answer_ids.numel()),
                           dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, extra], dim=1)

    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        extra = torch.zeros((mm_token_type_ids.shape[0], answer_ids.numel()),
                            dtype=mm_token_type_ids.dtype, device=mm_token_type_ids.device)
        mm_token_type_ids = torch.cat([mm_token_type_ids, extra], dim=1)

    model_inputs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "image_grid_thw": inputs.get("image_grid_thw"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "return_dict": True,
        "use_cache": False,
    }
    model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
    return model_inputs, visual_embeds, input_ids.shape[1]


def _gradient_importance(grad: torch.Tensor, visual_embeds: torch.Tensor, score_mode: str) -> torch.Tensor:
    if score_mode == "directional":
        return torch.clamp(-(grad.float() * visual_embeds.detach().float()).sum(dim=-1), min=0.0)
    return grad.float().norm(dim=-1) * visual_embeds.detach().float().norm(dim=-1)


def gradient_oracle(model: Qwen2VLWrapper, inputs, answer_ids: torch.Tensor,
                    max_answer_tokens: int, score_mode: str = "sensitivity") -> list[float]:
    """Return ||grad log p(answer)|| * ||visual_embed|| per visual token."""
    if hasattr(model, "gradient_oracle"):
        return model.gradient_oracle(inputs, answer_ids, max_answer_tokens, score_mode=score_mode)
    model.model.zero_grad(set_to_none=True)
    model_inputs, visual_embeds, prompt_len = build_inputs_embeds_with_grad(
        model, inputs, answer_ids, max_answer_tokens)
    answer_len = min(answer_ids.numel(), max_answer_tokens)
    outputs = model.model(**model_inputs)
    logits = outputs.logits
    start = prompt_len
    # Token at full_ids[pos] is predicted by logits[pos-1].
    pred_logits = logits[:, start - 1: start + answer_len - 1, :].float()
    target = answer_ids[:answer_len].to(pred_logits.device).unsqueeze(0)
    loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]),
        target.reshape(-1),
        reduction="mean",
    )
    loss.backward()
    grad = visual_embeds.grad
    if grad is None:
        raise RuntimeError("visual embedding gradient is None")
    importance = _gradient_importance(grad, visual_embeds, score_mode)
    model.model.zero_grad(set_to_none=True)
    return importance.detach().cpu().tolist()


@torch.no_grad()
def layer_attention_scores(model: Qwen2VLWrapper, inputs, layers: list[int]) -> dict[int, list[float]]:
    if hasattr(model, "layer_attention_scores"):
        return model.layer_attention_scores(inputs, layers)
    lm = model.model.model.language_model
    vis_pos, text_pos = visual_and_text_positions(inputs)
    if not vis_pos or not text_pos:
        return {layer: [] for layer in layers}
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def _pre_hook(module, args):
            if isinstance(args, tuple) and args:
                h = args[0]
                captured[layer_idx] = h.detach() if h.dim() == 3 else h.unsqueeze(0).detach()
        return _pre_hook

    try:
        for layer_idx in layers:
            handles.append(lm.layers[layer_idx].register_forward_pre_hook(make_hook(layer_idx)))
        model.model(**inputs, output_hidden_states=False, output_attentions=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    out: dict[int, list[float]] = {}
    for layer_idx in layers:
        h = captured.get(layer_idx)
        if h is None:
            out[layer_idx] = [0.0 for _ in vis_pos]
            continue
        scores = compute_text_vis_attention(
            lm.layers[layer_idx],
            h,
            text_pos,
            vis_pos,
            inputs,
            lm,
            apply_rope=True,
            query_mode="last_text",
        )
        out[layer_idx] = norm01_t(scores).tolist()
    return out


def normalize_prob(values: torch.Tensor, temperature: float) -> torch.Tensor:
    x = values.float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.numel() == 0:
        return x
    x = x - x.min()
    scale = max(float(temperature), 1e-6)
    return torch.softmax(x / scale, dim=0)


def deterministic_negative_subset(indices: torch.Tensor, max_items: int) -> torch.Tensor:
    if max_items <= 0 or indices.numel() <= max_items:
        return indices
    positions = torch.linspace(0, indices.numel() - 1, steps=max_items).round().long()
    return indices[positions]


def deepest_active_depth_surrogate(
    w: torch.Tensor,
    depth: torch.Tensor,
    active_weight_threshold: float,
) -> torch.Tensor:
    threshold = max(float(active_weight_threshold), 1e-8)
    temperature = max(threshold * 0.25, 1e-4)
    active_p = torch.sigmoid((w - threshold) / temperature)
    no_deeper = torch.ones((), dtype=w.dtype, device=w.device)
    expected = torch.zeros((), dtype=w.dtype, device=w.device)
    for idx in range(w.numel() - 1, -1, -1):
        p_deepest = active_p[idx] * no_deeper
        expected = expected + depth[idx] * p_deepest
        no_deeper = no_deeper * (1.0 - active_p[idx])
    fallback = torch.sum(w * depth)
    any_active = 1.0 - no_deeper
    return expected + (1.0 - any_active) * fallback


def fit_weights(samples: list[dict[str, Any]], layers: list[int], total_layers: int,
                rank_temperature: float, oracle_temperature: float,
                depth_lambda: float, entropy_lambda: float, l2_lambda: float,
                steps: int, lr: float, fit_loss: str,
                fit_budget_fracs: list[float],
                max_pairwise_negatives: int,
                fit_device: str,
                active_weight_threshold: float = 1e-8,
                consistency_lambda: float = 0.0,
                uncertainty_lambda: float = 0.0,
                consistency_eps: float = 1e-4,
                consistency_mode: str = "weighted",
                effective_layers_lambda: float = 0.0,
                effective_layers_target: float = 1.0,
                min_deep_layer_frac: float = 0.0,
                min_deep_weight: float = 0.0,
                min_deep_weight_lambda: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    if fit_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(fit_device)
    prepared = []
    for sample in samples:
        oracle = torch.tensor(sample["oracle"], dtype=torch.float32, device=device)
        scores = torch.tensor(
            [[sample["layer_scores"][str(layer)][i] for layer in layers]
             for i in range(len(sample["oracle"]))],
            dtype=torch.float32,
            device=device,
        )
        if oracle.numel() == 0 or scores.numel() == 0:
            continue
        oracle_p = normalize_prob(oracle, oracle_temperature)
        static_consensus = None
        static_uncertainty = None
        if consistency_mode == "static" and (consistency_lambda != 0.0 or uncertainty_lambda != 0.0):
            layer_mean = scores.mean(dim=1)
            static_uncertainty = scores.std(dim=1, unbiased=False)
            static_consensus = torch.tanh(layer_mean / torch.sqrt(static_uncertainty.pow(2) + max(float(consistency_eps), 1e-12)))
        topk_pairs = []
        for frac in fit_budget_fracs:
            k = max(1, min(oracle.numel(), int(math.ceil(oracle.numel() * frac))))
            pos_idx = torch.topk(oracle, k=k, largest=True).indices
            pos_mask = torch.zeros(oracle.numel(), dtype=torch.bool, device=device)
            pos_mask[pos_idx] = True
            neg_idx = torch.where(~pos_mask)[0]
            neg_idx = deterministic_negative_subset(neg_idx, max_pairwise_negatives)
            if pos_idx.numel() > 0 and neg_idx.numel() > 0:
                topk_pairs.append((pos_idx, neg_idx))
        prepared.append((scores, oracle_p, topk_pairs, static_consensus, static_uncertainty))
    if not prepared:
        raise ValueError("No valid calibration samples for optimization")

    theta = torch.zeros(len(layers), dtype=torch.float32, requires_grad=True, device=device)
    opt = torch.optim.Adam([theta], lr=lr)
    depth = torch.tensor([(layer + 1) / max(1, total_layers) for layer in layers],
                         dtype=torch.float32, device=device)
    history = []
    for step in range(steps):
        opt.zero_grad()
        w = torch.softmax(theta, dim=0)
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        align = torch.tensor(0.0, dtype=torch.float32, device=device)
        for scores, oracle_p, topk_pairs, static_consensus, static_uncertainty in prepared:
            mu = scores @ w
            if consistency_lambda != 0.0 or uncertainty_lambda != 0.0:
                if consistency_mode == "static":
                    consensus = static_consensus
                    uncertainty = static_uncertainty
                else:
                    var = torch.sum(w.unsqueeze(0) * (scores - mu.unsqueeze(1)).pow(2), dim=1)
                    uncertainty = torch.sqrt(var + max(float(consistency_eps), 1e-12))
                    consensus = torch.tanh(mu / uncertainty)
                fused = mu + float(consistency_lambda) * consensus - float(uncertainty_lambda) * uncertainty
            else:
                fused = mu
            pred_p = torch.softmax(fused / max(rank_temperature, 1e-6), dim=0)
            if fit_loss == "kl":
                sample_loss = F.kl_div(torch.log(pred_p + 1e-12), oracle_p, reduction="sum")
            elif fit_loss in ("topk_pairwise", "budget_overlap_pairwise"):
                pair_losses = []
                for pos_idx, neg_idx in topk_pairs:
                    pos = fused[pos_idx]
                    neg = fused[neg_idx]
                    margin = (neg.unsqueeze(0) - pos.unsqueeze(1)) / max(rank_temperature, 1e-6)
                    pair_losses.append(F.softplus(margin).mean())
                pair_loss = torch.stack(pair_losses).mean() if pair_losses else torch.tensor(0.0)
                if fit_loss == "topk_pairwise":
                    sample_loss = pair_loss
                else:
                    overlap_losses = []
                    for pos_idx, _neg_idx in topk_pairs:
                        overlap = pred_p[pos_idx].sum().clamp_min(1e-12)
                        overlap_losses.append(-torch.log(overlap))
                    overlap_loss = (
                        torch.stack(overlap_losses).mean()
                        if overlap_losses
                        else torch.tensor(0.0, dtype=torch.float32, device=device)
                    )
                    sample_loss = overlap_loss + pair_loss
            else:
                overlap_losses = []
                for pos_idx, _neg_idx in topk_pairs:
                    overlap = pred_p[pos_idx].sum().clamp_min(1e-12)
                    overlap_losses.append(-torch.log(overlap))
                sample_loss = (
                    torch.stack(overlap_losses).mean()
                    if overlap_losses
                    else torch.tensor(0.0, dtype=torch.float32, device=device)
                )
            loss = loss + sample_loss
            align = align + torch.sum(pred_p * oracle_p)
        loss = loss / len(prepared)
        entropy = -torch.sum(w * torch.log(w + 1e-12))
        effective_layers = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        effective_layers_penalty = torch.relu(float(effective_layers_target) - effective_layers).pow(2)
        deepest_depth = deepest_active_depth_surrogate(w, depth, active_weight_threshold)
        deep_mask = depth >= float(min_deep_layer_frac)
        if bool(deep_mask.any()):
            deep_weight = torch.sum(w[deep_mask])
        else:
            deep_weight = torch.zeros((), dtype=torch.float32, device=device)
        deep_weight_penalty = torch.relu(float(min_deep_weight) - deep_weight).pow(2)
        reg = (
            depth_lambda * deepest_depth
            + entropy_lambda * entropy
            + l2_lambda * torch.sum(w * w)
            + float(effective_layers_lambda) * effective_layers_penalty
            + float(min_deep_weight_lambda) * deep_weight_penalty
        )
        total = loss + reg
        total.backward()
        opt.step()
        if step in {0, steps - 1} or (step + 1) % max(1, steps // 10) == 0:
            history.append({
                "step": step + 1,
                "loss": float(loss.detach()),
                "regularizer": float(reg.detach()),
                "deepest_active_depth_surrogate": float(deepest_depth.detach()),
                "effective_layers": float(effective_layers.detach()),
                "effective_layers_penalty": float(effective_layers_penalty.detach()),
                "deep_weight": float(deep_weight.detach()),
                "deep_weight_penalty": float(deep_weight_penalty.detach()),
                "oracle_overlap_proxy": float((align / len(prepared)).detach()),
            })
    w_np = torch.softmax(theta.detach(), dim=0).cpu().numpy()
    diagnostics = {
        "num_fit_samples": len(prepared),
        "history": history,
        "fit_loss": fit_loss,
        "fit_device": str(device),
        "fit_budget_fracs": fit_budget_fracs,
        "consistency_lambda": float(consistency_lambda),
        "uncertainty_lambda": float(uncertainty_lambda),
        "consistency_eps": float(consistency_eps),
        "consistency_mode": str(consistency_mode),
        "effective_layers_lambda": float(effective_layers_lambda),
        "effective_layers_target": float(effective_layers_target),
        "min_deep_layer_frac": float(min_deep_layer_frac),
        "min_deep_weight": float(min_deep_weight),
        "min_deep_weight_lambda": float(min_deep_weight_lambda),
        "objective": (
            "mean KL(softmax(oracle/tau_o) || softmax((A w)/tau_s)) + regularization"
            if fit_loss == "kl"
            else (
                (
                    "mean budget-aware pairwise softplus ranking loss over oracle top-k tokens + regularization"
                    if fit_loss == "topk_pairwise"
                    else (
                        "mean negative log soft overlap plus pairwise ranking over backward-oracle top-budget tokens + regularization"
                        if fit_loss == "budget_overlap_pairwise"
                        else "mean negative log soft overlap between predicted retain distribution and backward-oracle top-budget tokens + regularization"
                    )
                )
            )
        ),
        "regularization": {
            "depth_delay_cost": (
                "depth_lambda * differentiable deepest-active-layer depth; layer activation is "
                "sigmoid((layer_weight - active_weight_threshold) / temperature), matching the final threshold policy."
            ),
            "entropy": "entropy_lambda * entropy(layer_weight); positive values encourage concentrated layer weights",
            "l2": "l2_lambda * sum(layer_weight^2)",
            "effective_layers": "effective_layers_lambda * relu(effective_layers_target - 1/sum(layer_weight^2))^2",
            "min_deep_weight": "min_deep_weight_lambda * relu(min_deep_weight - sum(w[layer_depth >= min_deep_layer_frac]))^2",
        },
    }
    return w_np, diagnostics


def sparsify(weights: np.ndarray, layers: list[int], max_active: int,
             active_weight_threshold: float = 1e-8) -> tuple[list[int], list[float]]:
    weights = np.asarray(weights, dtype=np.float64)
    if max_active > 0 and np.count_nonzero(weights > 0.0) > max_active:
        keep = np.argsort(weights)[-max_active:]
        sparse = np.zeros_like(weights)
        sparse[keep] = weights[keep]
        weights = sparse
    if float(weights.sum()) <= 1e-12:
        weights[int(np.argmax(weights))] = 1.0
    weights = weights / max(float(weights.sum()), 1e-12)
    threshold = max(0.0, float(active_weight_threshold))
    active_mask = weights >= threshold
    if not bool(active_mask.any()):
        active_mask[int(np.argmax(weights))] = True
    active_raw = weights * active_mask
    active_raw = active_raw / max(float(active_raw.sum()), 1e-12)
    active_layers = [int(layer) for layer, keep in zip(layers, active_mask) if keep]
    active_weights = [float(weight) for weight, keep in zip(active_raw, active_mask) if keep]
    return active_layers, active_weights


def make_policy(name_prefix: str, weights: np.ndarray, layers: list[int],
                total_layers: int, max_active_layers: int,
                active_weight_threshold: float = 1e-8) -> dict[str, Any]:
    active_layers, active_weights = sparsify(
        weights.copy(), layers, max_active_layers, active_weight_threshold)
    return {
        "name": name_prefix + "_" + "_".join(
            f"L{layer}:{weight:.3f}"
            for layer, weight in zip(active_layers, active_weights)
        ),
        "active_layers": active_layers,
        "active_weights": active_weights,
        "all_layers": layers,
        "all_weights": [float(x) for x in weights],
        "total_layers": total_layers,
        "max_active_layers": max_active_layers,
        "active_weight_threshold": float(active_weight_threshold),
    }


def fuse_policy_scores(layer_scores: dict[str, list[float]], active_layers: list[int],
                       active_weights: list[float], n_vis: int) -> list[float]:
    fused = torch.zeros(n_vis, dtype=torch.float32)
    for layer, weight in zip(active_layers, active_weights):
        scores = torch.tensor(layer_scores.get(str(layer), []), dtype=torch.float32)
        if scores.shape[0] != n_vis:
            padded = torch.zeros(n_vis, dtype=torch.float32)
            m = min(n_vis, scores.shape[0])
            padded[:m] = scores[:m]
            scores = padded
        fused += scores * float(weight)
    return fused.tolist()


def fuse_policy_scores_with_consistency(
    layer_scores: dict[str, list[float]],
    active_layers: list[int],
    active_weights: list[float],
    n_vis: int,
    consistency_lambda: float = 0.0,
    uncertainty_lambda: float = 0.0,
    consistency_eps: float = 1e-4,
    consistency_mode: str = "weighted",
) -> list[float]:
    weights = torch.tensor(active_weights, dtype=torch.float32)
    if weights.numel() == 0:
        return [0.0 for _ in range(n_vis)]
    weights = torch.clamp(weights, min=0.0)
    weights = weights / weights.sum().clamp_min(1e-12)
    cols = []
    for layer in active_layers:
        scores = torch.tensor(layer_scores.get(str(layer), []), dtype=torch.float32)
        if scores.shape[0] != n_vis:
            padded = torch.zeros(n_vis, dtype=torch.float32)
            m = min(n_vis, scores.shape[0])
            padded[:m] = scores[:m]
            scores = padded
        cols.append(scores)
    if not cols:
        return [0.0 for _ in range(n_vis)]
    matrix = torch.stack(cols, dim=1)
    mu = matrix @ weights
    if consistency_lambda == 0.0 and uncertainty_lambda == 0.0:
        return mu.tolist()
    if consistency_mode == "static":
        layer_mean = matrix.mean(dim=1)
        uncertainty = matrix.std(dim=1, unbiased=False)
        consensus = torch.tanh(layer_mean / torch.sqrt(uncertainty.pow(2) + max(float(consistency_eps), 1e-12)))
    else:
        var = torch.sum(weights.unsqueeze(0) * (matrix - mu.unsqueeze(1)).pow(2), dim=1)
        uncertainty = torch.sqrt(var + max(float(consistency_eps), 1e-12))
        consensus = torch.tanh(mu / uncertainty)
    fused = mu + float(consistency_lambda) * consensus - float(uncertainty_lambda) * uncertainty
    return fused.tolist()


def allocate(scores: list[float], n_vis: int, budget_frac: float) -> list[bool]:
    n_keep = max(1, min(n_vis, int(n_vis * budget_frac)))
    tensor = torch.tensor(scores, dtype=torch.float32)
    keep = set(tensor.argsort(descending=True)[:n_keep].tolist())
    return [i in keep for i in range(n_vis)]


def write_report(payload: dict[str, Any], path: Path) -> None:
    policy = payload["policy"]
    if "by_budget" in policy:
        active_txt = "budget-specific"
    else:
        active_txt = ", ".join(
            f"L{layer}:{weight:.3f}"
            for layer, weight in zip(policy["active_layers"], policy["active_weights"])
        )
    lines = [
        "# Gradient-Aligned Attention Policy",
        "",
        "This policy was fitted from scratch using gradient-oracle alignment. No prior sweep, validation, strategy, or calibration JSON was read.",
        "",
        f"Benchmark: `{payload['benchmark']}`",
        f"Samples: `{payload['plan']['num_samples']}`; offset: `{payload['plan']['sample_offset']}`",
        f"Layers fit: `{len(payload['plan']['layers'])}`; max active layers: `{payload['plan']['max_active_layers']}`",
        f"Active policy: `{active_txt}`",
        "",
        "## Optimization",
        "",
        "| Step | KL loss | Regularizer | Oracle overlap proxy |",
        "|---:|---:|---:|---:|",
    ]
    for row in payload["fit_diagnostics"]["history"]:
        lines.append(
            f"| {row['step']} | {row['loss']:.4f} | "
            f"{row['regularizer']:.4f} | {row['oracle_overlap_proxy']:.4f} |"
        )
    if "by_budget" in policy:
        lines.extend(["", "## Policies By Budget", ""])
        lines.append("| Budget | Active policy |")
        lines.append("|---|---|")
        for label, item in policy["by_budget"].items():
            active = ", ".join(
                f"L{layer}:{weight:.3f}"
                for layer, weight in zip(item["active_layers"], item["active_weights"])
            )
            lines.append(f"| {label} | {active} |")
    if payload.get("validation"):
        lines.extend([
            "",
            "## Real Fidelity Validation",
            "",
            "| Budget | Accuracy | Fidelity | Fidelity valid | Mean kept |",
            "|---|---:|---:|---:|---:|",
        ])
        for label, cell in payload["validation"]["budgets"].items():
            lines.append(
                f"| {label} | {cell['accuracy']:.3f} | {cell['fidelity']:.3f} | "
                f"{cell['fidelity_valid']} | {cell['mean_kept_tokens']:.1f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def summarize_validation(records: list[dict[str, Any]], specs: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"budgets": {}}
    for spec in specs:
        label = spec["label"]
        scores = []
        fidelities = []
        kept = []
        errors = 0
        for row in records:
            cell = row.get("validation", {}).get(label, {})
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


def main() -> int:
    args = parse_args()
    benchmark = normalize_benchmark(args.benchmark)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    plan: dict[str, Any] = {
        "model_path": args.model_path,
        "benchmark": benchmark,
        "budgets": budget_specs(args.budgets, args.retain_tokens),
        "num_samples": args.num_samples,
        "sample_offset": args.sample_offset,
        "num_sweep": args.num_sweep,
        "max_pixels": args.max_pixels,
        "resize_square": args.resize_square,
        "max_active_layers": args.max_active_layers,
        "active_weight_threshold": args.active_weight_threshold,
        "rank_temperature": args.rank_temperature,
        "oracle_temperature": args.oracle_temperature,
        "depth_lambda": args.depth_lambda,
        "entropy_lambda": args.entropy_lambda,
        "l2_lambda": args.l2_lambda,
        "optim_steps": args.optim_steps,
        "lr": args.lr,
        "fit_loss": args.fit_loss,
        "max_pairwise_negatives": args.max_pairwise_negatives,
        "fit_device": args.fit_device,
        "fit_per_budget": args.fit_per_budget,
        "validate_real": args.validate_real,
        "reads_prior_sweep_results": False,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0

    model = Qwen2VLWrapper(args.model_path)
    total_layers = len(model.model.model.language_model.layers)
    layers = args.layers if args.layers is not None else list(range(total_layers))
    layers = [int(x) for x in layers if 0 <= int(x) < total_layers]
    if not layers:
        raise ValueError("No valid layers selected")
    plan["layers"] = layers
    specs = plan["budgets"]

    samples = load_split(
        benchmark,
        split="eval",
        num_eval=args.num_samples + args.sample_offset,
        num_sweep=args.num_sweep,
    )[args.sample_offset:args.sample_offset + args.num_samples]

    records: list[dict[str, Any]] = []
    t0 = time.time()
    max_tok = max_new_tokens_map(benchmark)
    for idx, sample in enumerate(samples):
        try:
            image = maybe_resize(load_image(sample), args.resize_square)
            sample = dict(sample, image=image)
            prompt = make_prompt(benchmark, sample)
            inputs = prepare_inputs(model, image, prompt, args.max_pixels)
            vis_embeds, grid_thw = model.extract_visual_embeddings(inputs)
            n_vis = int(vis_embeds.shape[0])
            if n_vis == 0:
                continue
            answer_ids, baseline_text = generate_ids_and_text(model, inputs, max_tok)
            baseline_score = eval_metric(benchmark, baseline_text, sample)
            oracle = gradient_oracle(
                model,
                inputs,
                answer_ids,
                max_answer_tokens=args.max_answer_tokens_for_grad,
            )
            attn = layer_attention_scores(model, inputs, layers)
            record = {
                "sample_index": args.sample_offset + idx,
                "num_vis": n_vis,
                "baseline": baseline_score,
                "baseline_text": baseline_text,
                "oracle": oracle,
                "layer_scores": {str(layer): attn.get(layer, []) for layer in layers},
            }
            records.append(record)
            avg = (time.time() - t0) / max(1, idx + 1)
            print(f"  calibration {idx + 1}/{len(samples)} avg={avg:.1f}s", flush=True)
        except Exception:
            print(traceback.format_exc(), flush=True)

    percent_specs = [spec for spec in specs if spec["type"] == "percent"]
    if args.fit_per_budget:
        policy_by_budget = {}
        diagnostics_by_budget = {}
        for spec in percent_specs:
            frac = max(0.0, min(1.0, float(spec["value"]) / 100.0))
            print(f"  fitting budget-specific policy for {spec['label']}", flush=True)
            weights, diagnostics = fit_weights(
                records,
                layers,
                total_layers,
                rank_temperature=args.rank_temperature,
                oracle_temperature=args.oracle_temperature,
                depth_lambda=args.depth_lambda,
                entropy_lambda=args.entropy_lambda,
                l2_lambda=args.l2_lambda,
                steps=args.optim_steps,
                lr=args.lr,
                fit_loss=args.fit_loss,
                fit_budget_fracs=[frac],
                max_pairwise_negatives=args.max_pairwise_negatives,
                fit_device=args.fit_device,
                active_weight_threshold=args.active_weight_threshold,
            )
            policy_by_budget[spec["label"]] = make_policy(
                "gradient_aligned_" + spec["label"], weights, layers,
                total_layers, args.max_active_layers, args.active_weight_threshold)
            diagnostics_by_budget[spec["label"]] = diagnostics
        first_label = percent_specs[0]["label"] if percent_specs else "default"
        policy = {
            "name": "gradient_aligned_budget_specific",
            "by_budget": policy_by_budget,
            "default_budget": first_label,
            "total_layers": total_layers,
        }
        diagnostics = {
            "fit_loss": args.fit_loss,
            "fit_per_budget": True,
            "by_budget": diagnostics_by_budget,
            "history": diagnostics_by_budget[first_label]["history"] if percent_specs else [],
        }
    else:
        weights, diagnostics = fit_weights(
            records,
            layers,
            total_layers,
            rank_temperature=args.rank_temperature,
            oracle_temperature=args.oracle_temperature,
            depth_lambda=args.depth_lambda,
            entropy_lambda=args.entropy_lambda,
            l2_lambda=args.l2_lambda,
            steps=args.optim_steps,
            lr=args.lr,
            fit_loss=args.fit_loss,
            fit_budget_fracs=[
                max(0.0, min(1.0, float(spec["value"]) / 100.0))
                for spec in percent_specs
            ],
            max_pairwise_negatives=args.max_pairwise_negatives,
            fit_device=args.fit_device,
            active_weight_threshold=args.active_weight_threshold,
        )
        policy = make_policy(
            "gradient_aligned", weights, layers, total_layers,
            args.max_active_layers, args.active_weight_threshold)

    if args.validate_real:
        print("  validating learned policy with real pruned generation", flush=True)
        for record, sample in zip(records, samples):
            try:
                image = maybe_resize(load_image(sample), args.resize_square)
                sample = dict(sample, image=image)
                prompt = make_prompt(benchmark, sample)
                inputs = prepare_inputs(model, image, prompt, args.max_pixels)
                vis_embeds, grid_thw = model.extract_visual_embeddings(inputs)
                n_vis = int(vis_embeds.shape[0])
                record["validation"] = {}
                for spec in specs:
                    try:
                        if "by_budget" in policy:
                            budget_policy = policy["by_budget"].get(
                                spec["label"],
                                policy["by_budget"][policy["default_budget"]],
                            )
                            active_layers = budget_policy["active_layers"]
                            active_weights = budget_policy["active_weights"]
                        else:
                            active_layers = policy["active_layers"]
                            active_weights = policy["active_weights"]
                        fused = fuse_policy_scores(record["layer_scores"], active_layers, active_weights, n_vis)
                        keep_mask = allocate(fused, n_vis, budget_fraction(spec, n_vis))
                        pred = run_pruned(model, inputs, vis_embeds, grid_thw, keep_mask, max_tok)
                        score = eval_metric(benchmark, pred, sample)
                        error = None
                    except Exception as exc:
                        keep_mask = []
                        pred = ""
                        score = None
                        error = f"{type(exc).__name__}: {exc}"
                    record["validation"][spec["label"]] = {
                        "score": score,
                        "prediction": pred,
                        "kept_tokens": sum(1 for v in keep_mask if v),
                        "error": error,
                    }
            except Exception:
                print(traceback.format_exc(), flush=True)

    payload = {
        "method": "gradient_aligned_layer_attention_policy",
        "benchmark": benchmark,
        "plan": plan,
        "policy": policy,
        "fit_diagnostics": diagnostics,
        "records": records,
        "validation": summarize_validation(records, specs) if args.validate_real else None,
    }
    (out_dir / "calibration.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "policy.json").write_text(json.dumps({
        "method": payload["method"],
        "benchmark": benchmark,
        "plan": plan,
        "policy": policy,
        "fit_diagnostics": diagnostics,
    }, indent=2))
    write_report(payload, out_dir / "report.md")
    print(json.dumps({
        "wrote": str(out_dir),
        "benchmark": benchmark,
        "records": len(records),
        "policy": policy["name"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
