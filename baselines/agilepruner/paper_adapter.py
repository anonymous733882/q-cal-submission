"""Paper-faithful AgilePruner implementation for dualsignal experiments.

The public AgilePruner repository has no detailed code yet. The project page
does, however, specify the adaptive rule:

    tau_i = order_i * erank_input / erank_avg * 0.01, tau_i <= tau_max

High-attention tokens are selected first. For each selected token, candidate
tokens whose cosine distance to it is below the dynamic threshold are removed.
Larger thresholds prune more aggressively and promote diversity; smaller
thresholds preserve nearby fine-grained high-attention tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


_IMAGE_TOKEN_ID = 151655
_VISION_START_ID = 151652
_VISION_END_ID = 151653


def effective_rank(features: torch.Tensor, eps: float = 1e-12) -> float:
    """Effective rank from normalized singular-value entropy."""
    x = features.float()
    if x.ndim != 2 or min(x.shape) == 0:
        return 1.0
    x = x - x.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(x)
    mass = singular / singular.sum().clamp_min(eps)
    entropy = -(mass * (mass + eps).log()).sum()
    return float(torch.exp(entropy).item())


class AgilePrunerBaseline:
    """AgilePruner paper-adapted implementation.

    Args:
        model: Qwen2.5-VL wrapper.
        score_layer: LLM attention layer for text-to-visual importance.
        prune_ratio: fraction of visual tokens to drop.
        erank_avg: calibration-set average effective rank. The paper rule is
            normalized by this value; keep it explicit in reports.
        tau_max: maximum cosine-distance threshold.
    """

    def __init__(
        self,
        model,
        score_layer: int = 1,
        prune_ratio: float = 0.5,
        erank_avg: float = 16.0,
        tau_max: float = 0.95,
    ):
        self.model = model
        self.score_layer = score_layer
        self.prune_ratio = prune_ratio
        self.erank_avg = erank_avg
        self.tau_max = tau_max

    def _get_positions(self, input_ids):
        ids = input_ids[0]
        vis_start = (ids == _VISION_START_ID).nonzero(as_tuple=True)[0]
        vis_end = (ids == _VISION_END_ID).nonzero(as_tuple=True)[0]
        if len(vis_start) == 0 or len(vis_end) == 0:
            return [], []
        vs, ve = vis_start[0].item(), vis_end[0].item()
        vis_pos = [i for i in range(vs + 1, ve) if ids[i] == _IMAGE_TOKEN_ID]
        text_pos = [i for i in range(ids.shape[0]) if i < vs or i > ve]
        return vis_pos, text_pos

    @torch.no_grad()
    def _attention_scores(self, model, inputs, vis_pos, text_pos):
        from baselines.common.attn_utils import compute_text_vis_attention

        lm = model.model.model.language_model
        layer = lm.layers[self.score_layer]
        captured = {}

        def _pre(_module, args):
            if isinstance(args, tuple) and args:
                h = args[0]
                captured["h"] = h.detach() if h.dim() == 3 else h.unsqueeze(0).detach()

        handle = layer.register_forward_pre_hook(_pre)
        try:
            model.model(**inputs, output_hidden_states=False, output_attentions=False, return_dict=True)
        finally:
            handle.remove()

        h = captured.get("h")
        if h is None or not vis_pos or not text_pos:
            return torch.zeros(len(vis_pos))
        return compute_text_vis_attention(
            layer, h, text_pos, vis_pos, inputs, lm, apply_rope=True, query_mode="all_text"
        )

    def dynamic_tau(self, order: int, erank_input: float) -> float:
        order = max(1, int(order))
        avg = max(float(self.erank_avg), 1e-6)
        tau = order * (float(erank_input) / avg) * 0.01
        return min(float(self.tau_max), max(0.0, tau))

    def _select_from_scores(self, scores: torch.Tensor, vis_embeds: torch.Tensor, n_keep: int):
        ranked = torch.argsort(scores.float(), descending=True).tolist()
        emb = F.normalize(vis_embeds.float(), dim=-1)
        erank_input = effective_rank(vis_embeds)
        available = set(ranked)
        selected: list[int] = []

        for order, idx in enumerate(ranked, start=1):
            if len(selected) >= n_keep:
                break
            if idx not in available:
                continue
            selected.append(idx)
            tau = self.dynamic_tau(order, erank_input)
            if tau <= 0:
                continue
            candidates = list(available)
            if not candidates:
                continue
            cand = torch.tensor(candidates, dtype=torch.long, device=emb.device)
            cosine_distance = 1.0 - (emb[cand] @ emb[idx]).clamp(-1.0, 1.0)
            remove = cand[cosine_distance < tau].cpu().tolist()
            for ridx in remove:
                available.discard(ridx)

        selected_set = set(selected)
        if len(selected_set) < n_keep:
            for idx in ranked:
                selected_set.add(idx)
                if len(selected_set) >= n_keep:
                    break
        return selected_set

    @torch.no_grad()
    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw, prune_ratio=None):
        ratio = prune_ratio if prune_ratio is not None else self.prune_ratio
        n_vis = vis_embeds.shape[0]
        n_keep = max(1, min(n_vis, int(n_vis * (1.0 - ratio))))

        vis_pos, text_pos = self._get_positions(inputs["input_ids"])
        if not vis_pos or not text_pos:
            scores = vis_embeds.float().norm(dim=-1)
        else:
            scores = self._attention_scores(model_wrapper, inputs, vis_pos, text_pos)
            if scores.shape[0] != n_vis:
                padded = torch.zeros(n_vis, dtype=torch.float32)
                m = min(scores.shape[0], n_vis)
                padded[:m] = scores[:m].float().cpu()
                scores = padded

        selected = self._select_from_scores(scores, vis_embeds, n_keep)
        return [i in selected for i in range(n_vis)]
