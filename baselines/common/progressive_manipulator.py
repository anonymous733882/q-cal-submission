"""Progressive layer-wise token pruning for baseline methods.

Provides a monkey-patch based forward that supports re-ranking at
layer boundaries — needed by PyramidDrop (re-rank at stage boundaries)
and FitPrune (re-rank at every layer).

Unlike BEA's ProgressiveTokenManipulator which uses pre-computed
allocations, this manipulator calls a user-supplied ``rank_fn`` at
each boundary to decide which tokens to drop based on the current
hidden states.

GAP-aware: preserves original RoPE position IDs (no re-indexing).
"""

from __future__ import annotations
import functools
import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Set, Tuple
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast


class BaselineProgressiveManipulator:
    """Monkey-patch based progressive pruning with re-ranking at boundaries.

    At each boundary layer, calls ``rank_fn(hidden_states, vis_positions,
    text_positions, layer_idx, attn_layer)`` which returns a list of
    visual seq positions to DROP.

    Args:
        wrapper: Qwen2VLWrapper
        boundary_spec: dict {layer_idx: num_tokens_to_drop}
            At layer_idx, drop this many visual tokens (re-ranked).
        vis_start: start of visual token range in sequence
        vis_end: end of visual token range in sequence (exclusive)
        rank_fn: callable(hidden_states, vis_positions, text_positions,
                         layer_idx, attn_layer) → List[int]
            Returns list of seq positions to DROP (len = num_to_drop).
    """

    def __init__(
        self,
        wrapper,
        boundary_spec: Dict[int, int],
        vis_start: int,
        vis_end: int,
        rank_fn: Callable,
        vis_token_positions: List[int] | None = None,
        recycle_fn: Callable | None = None,
        remap_fn: Callable | None = None,
    ):
        self.wrapper = wrapper
        self.boundary_spec = boundary_spec
        self.vis_start = vis_start
        self.vis_end = vis_end
        self.rank_fn = rank_fn
        self.recycle_fn = recycle_fn
        self.remap_fn = remap_fn
        self._original_forward = None
        # For multi-image: explicit list of visual token positions
        # (may be non-contiguous, skipping <vision_start/end> markers)
        self._vis_token_positions = vis_token_positions

    def patch(self):
        lm = self.wrapper.model.model.language_model
        self._original_forward = lm.forward
        lm.forward = functools.update_wrapper(
            functools.partial(self._progressive_forward, lm),
            self._original_forward,
        )

    def unpatch(self):
        if self._original_forward is not None:
            lm = self.wrapper.model.model.language_model
            lm.forward = self._original_forward
            self._original_forward = None

    install_hooks = patch
    remove_hooks = unpatch

    def _progressive_forward(self, text_model, *args, **kwargs):
        """Drop-in replacement for Qwen2VLTextModel.forward."""
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        position_ids = kwargs.get("position_ids", args[2] if len(args) > 2 else None)
        past_key_values = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
        inputs_embeds = kwargs.get("inputs_embeds", args[4] if len(args) > 4 else None)
        use_cache = kwargs.get("use_cache", args[5] if len(args) > 5 else None)
        extra_kwargs = {k: v for k, v in kwargs.items()
                        if k not in ("input_ids", "attention_mask", "position_ids",
                                     "past_key_values", "inputs_embeds", "use_cache")}

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=text_model.config)

        if inputs_embeds is None:
            inputs_embeds = text_model.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        # Causal mask
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
            mask_kwargs = {
                "config": text_model.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if text_model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = text_model.rotary_emb(hidden_states, position_ids)

        # Track current visual token positions
        seq_len = hidden_states.shape[1]
        is_prefill = seq_len > 1

        if is_prefill:
            if self._vis_token_positions is not None:
                # Multi-image: use explicit (possibly non-contiguous) positions
                vis_positions = list(self._vis_token_positions)
            else:
                # Single-image: contiguous range
                vis_positions = list(range(self.vis_start, self.vis_end))
            # Text positions = everything not visual
            vis_set = set(vis_positions)
            all_text = [p for p in range(seq_len) if p not in vis_set]
        else:
            vis_positions = None
            all_text = None

        for i, decoder_layer in enumerate(text_model.layers):
            # Prune at boundary
            if is_prefill and i in self.boundary_spec:
                num_to_drop = self.boundary_spec[i]
                if num_to_drop > 0 and len(vis_positions) > 1:
                    # Call rank_fn to get positions to drop
                    attn_layer = text_model.layers[i].self_attn
                    ranked = self.rank_fn(
                        hidden_states, vis_positions, all_text,
                        i, attn_layer,
                        layer_module=text_model.layers[i],
                        position_embeddings=position_embeddings,
                        mrope_section=getattr(
                            text_model.config, "rope_parameters", {}
                        ).get("mrope_section", None))
                    drop_scores = None
                    if isinstance(ranked, tuple):
                        ranked, drop_scores = ranked
                    max_drop = min(num_to_drop, len(vis_positions) - 1)
                    drop_positions = ranked
                    drop_positions = drop_positions[:max_drop]

                    if len(drop_positions) > 0:
                        drop_set = set(drop_positions)
                        if self.recycle_fn is not None:
                            keep_vis = [p for p in vis_positions if p not in drop_set]
                            hidden_states = self.recycle_fn(
                                hidden_states, keep_vis, drop_positions, drop_scores)
                        keep_seq = [p for p in range(hidden_states.shape[1])
                                    if p not in drop_set]

                        idx = torch.tensor(keep_seq, dtype=torch.long)

                        # Prune hidden_states
                        hidden_states = hidden_states[:, idx.to(hidden_states.device), :]

                        # Prune position_embeddings
                        cos, sin = position_embeddings
                        if cos.dim() == 4:
                            position_embeddings = (
                                cos[:, :, idx.to(cos.device), :],
                                sin[:, :, idx.to(sin.device), :],
                            )
                        else:
                            position_embeddings = (
                                cos[:, idx.to(cos.device), :],
                                sin[:, idx.to(sin.device), :],
                            )

                        # Prune causal masks
                        new_masks = {}
                        for key, mask in causal_mask_mapping.items():
                            if mask is None:
                                new_masks[key] = None
                            else:
                                idx_m = idx.to(mask.device)
                                new_masks[key] = mask[:, :, idx_m, :][:, :, :, idx_m]
                        causal_mask_mapping = new_masks

                        # Prune text_position_ids
                        if text_position_ids is not None:
                            text_position_ids = text_position_ids[:, idx.to(text_position_ids.device)]

                        # Update tracked positions
                        old_to_new = {}
                        for new_pos, old_pos in enumerate(keep_seq):
                            old_to_new[old_pos] = new_pos
                        vis_positions = [old_to_new[p] for p in vis_positions
                                         if p not in drop_set]
                        all_text = [old_to_new[p] for p in all_text]
                        if self.remap_fn is not None:
                            self.remap_fn(old_to_new)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[text_model.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **extra_kwargs,
            )

        hidden_states = text_model.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# ── Rank functions ───────────────────────────────────────────────────

def pyramiddrop_rank_fn(hidden_states, vis_positions, text_positions,
                        layer_idx, attn_layer, *, layer_module=None,
                        position_embeddings=None, mrope_section=None):
    """PyramidDrop ranking: last-text-token → visual attention.

    Returns visual seq positions sorted by ASCENDING importance
    (lowest first = drop first).

    Now applies input_layernorm + M-RoPE when layer_module and
    position_embeddings are provided.
    """
    dev = next(attn_layer.q_proj.parameters()).device
    h = hidden_states.to(dev)

    # S1: Apply input_layernorm if available
    if layer_module is not None:
        h = layer_module.input_layernorm(h)

    # Q from last text token
    if text_positions:
        last_text = text_positions[-1]
    else:
        last_text = 0
    h_query = h[:, last_text:last_text + 1, :]
    q = attn_layer.q_proj(h_query)

    num_heads = q.shape[-1] // attn_layer.head_dim
    q = q.view(1, 1, num_heads, attn_layer.head_dim).transpose(1, 2)

    # K from visual tokens
    vis_idx = torch.tensor(vis_positions, dtype=torch.long, device=dev)
    h_vis = h[:, vis_idx, :]
    k = attn_layer.k_proj(h_vis)
    num_kv_heads = k.shape[-1] // attn_layer.head_dim
    k = k.view(1, len(vis_positions), num_kv_heads, attn_layer.head_dim).transpose(1, 2)
    if num_heads != num_kv_heads:
        k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)

    # S2: Apply M-RoPE if position_embeddings provided
    if position_embeddings is not None and mrope_section is not None:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_multimodal_rotary_pos_emb)
        cos, sin = position_embeddings
        cos = cos.to(dev)
        sin = sin.to(dev)
        q_pos = torch.tensor([last_text], dtype=torch.long, device=dev)
        if cos.dim() == 4:
            cos_q = cos[:, :, q_pos, :]
            sin_q = sin[:, :, q_pos, :]
            cos_k = cos[:, :, vis_idx, :]
            sin_k = sin[:, :, vis_idx, :]
        else:
            cos_q = cos[:, q_pos, :]
            sin_q = sin[:, q_pos, :]
            cos_k = cos[:, vis_idx, :]
            sin_k = sin[:, vis_idx, :]
        q, _ = apply_multimodal_rotary_pos_emb(
            q, q, cos_q, sin_q, mrope_section)
        _, k = apply_multimodal_rotary_pos_emb(
            k, k, cos_k, sin_k, mrope_section)

    scale = attn_layer.head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn.float(), dim=-1)
    # Mean over heads → [num_vis]
    scores = attn[0].mean(dim=0).squeeze(0)

    # Sort ascending (lowest importance first)
    sorted_indices = scores.argsort()
    return [vis_positions[idx.item()] for idx in sorted_indices]


def fitprune_rank_fn(hidden_states, vis_positions, text_positions,
                     layer_idx, attn_layer, *, layer_module=None,
                     position_embeddings=None, mrope_section=None):
    """FitPrune ranking: self_attn × cross_attn product.

    self_attn: sum of attention FROM all visual tokens TO each visual token
    cross_attn: sum of attention FROM text tokens TO each visual token

    Returns visual seq positions sorted by ASCENDING importance.
    """
    dev = next(attn_layer.q_proj.parameters()).device
    h = hidden_states.to(dev)
    num_vis = len(vis_positions)

    vis_idx = torch.tensor(vis_positions, dtype=torch.long, device=dev)
    h_vis = h[:, vis_idx, :]

    # K for visual tokens (shared)
    k_vis = attn_layer.k_proj(h_vis)
    num_kv_heads = k_vis.shape[-1] // attn_layer.head_dim
    num_heads = attn_layer.q_proj.out_features // attn_layer.head_dim
    k_vis = k_vis.view(1, num_vis, num_kv_heads, attn_layer.head_dim).transpose(1, 2)
    if num_heads != num_kv_heads:
        k_vis = k_vis.repeat_interleave(num_heads // num_kv_heads, dim=1)

    scale = attn_layer.head_dim ** -0.5

    # Self-attention: Q_vis × K_vis^T, sum over query dim
    q_vis = attn_layer.q_proj(h_vis)
    q_vis = q_vis.view(1, num_vis, num_heads, attn_layer.head_dim).transpose(1, 2)
    self_attn_scores = torch.matmul(q_vis, k_vis.transpose(-2, -1)) * scale
    self_attn_scores = F.softmax(self_attn_scores.float(), dim=-1)
    # Max over heads (per official code), sum over query dim → [num_vis]
    self_attn = self_attn_scores[0].max(dim=0).values.sum(dim=0)

    del q_vis, self_attn_scores

    # Cross-attention: Q_text × K_vis^T, mean over text tokens
    if text_positions:
        text_idx = torch.tensor(text_positions, dtype=torch.long, device=dev)
        h_text = h[:, text_idx, :]
        q_text = attn_layer.q_proj(h_text)
        num_text = len(text_positions)
        q_text = q_text.view(1, num_text, num_heads, attn_layer.head_dim).transpose(1, 2)
        cross_attn_scores = torch.matmul(q_text, k_vis.transpose(-2, -1)) * scale
        cross_attn_scores = F.softmax(cross_attn_scores.float(), dim=-1)
        cross_attn = cross_attn_scores[0].max(dim=0).values.mean(dim=0)
        del q_text, cross_attn_scores
    else:
        cross_attn = torch.ones(num_vis, device=dev)

    del k_vis

    # Combined score: self × cross
    combined = self_attn * cross_attn

    # Sort ascending (lowest importance first)
    sorted_indices = combined.argsort()
    return [vis_positions[idx.item()] for idx in sorted_indices]


# FastV rank function: identical scoring logic to pyramiddrop_rank_fn
# (last-text-token → visual attention, ascending importance order)
fastv_rank_fn = pyramiddrop_rank_fn
