"""Qwen decoder-layer adapter for the official FastV attention-top-k rule."""

from __future__ import annotations

import importlib

import torch
import torch.nn.functional as F

from baselines.pact.manipulator import PactManipulator
from baselines.pact.reference import PactReduction, _projected_qk
from baselines.fastv.selector import select


class FastVManipulator(PactManipulator):
    def __init__(self, language_model, visual_positions, *, target_visual_tokens: int):
        super().__init__(language_model, visual_positions, layer=2,
                         target_visual_tokens=target_visual_tokens)
        self.previous_layer_attention = None

    def _before_layer(self, index, hidden, decoder_layer, position_embeddings):
        if index != self.layer - 1:
            return
        q, k = _projected_qk(hidden, decoder_layer)
        attention = decoder_layer.self_attn
        module = importlib.import_module(type(attention).__module__)
        cos, sin = position_embeddings
        if hasattr(module, "apply_multimodal_rotary_pos_emb"):
            rope = getattr(attention, "rope_scaling", None)
            if rope is None:
                rope = getattr(attention.config, "rope_parameters", {})
            q, k = module.apply_multimodal_rotary_pos_emb(
                q, k, cos, sin, rope["mrope_section"])
        else:
            q, k = module.apply_rotary_pos_emb(q, k, cos, sin)
        if q.shape[1] != k.shape[1]:
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        logits = torch.matmul(q[:, :, -1:, :], k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        self.previous_layer_attention = F.softmax(logits.float(), dim=-1)

    def _reduce(self, hidden, decoder_layer, position_embeddings):
        if self.previous_layer_attention is None or self.target_visual_tokens is None:
            raise RuntimeError("FastV previous-layer attention was not collected")
        selected = select(
            self.previous_layer_attention, self.visual_positions, self.target_visual_tokens)
        kept_visual = set(selected.tolist())
        visual = set(self.visual_positions)
        kept_positions = [position for position in range(hidden.shape[1])
                          if position not in visual or position in kept_visual]
        kept = torch.as_tensor(kept_positions, device=hidden.device, dtype=torch.long)
        return PactReduction(
            hidden_states=hidden.index_select(1, kept),
            kept_positions=kept_positions,
            log_token_sizes=torch.zeros(len(kept_positions), device=hidden.device),
            cutoff=0.0,
        )
