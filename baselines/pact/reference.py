"""PACT's EUTI and vendored DBDPC reduction for Qwen decoder hidden states.

The reduction occurs immediately before a decoder layer. This module only
computes the reduction; the caller must apply log(cluster_size) to attention
logits in that layer and all subsequent prefill/decode steps.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from baselines.pact.utils import DBDPC, merge_clusters


@dataclass
class PactReduction:
    hidden_states: torch.Tensor
    kept_positions: list[int]
    log_token_sizes: torch.Tensor
    cutoff: float


def _projected_qk(hidden_states: torch.Tensor, decoder_layer):
    attn = decoder_layer.self_attn
    normalized = decoder_layer.input_layernorm(hidden_states)
    q = attn.q_proj(normalized)
    k = attn.k_proj(normalized)
    head_dim = int(attn.head_dim)
    q_heads = q.shape[-1] // head_dim
    k_heads = k.shape[-1] // head_dim
    q = q.view(1, -1, q_heads, head_dim).transpose(1, 2)
    k = k.view(1, -1, k_heads, head_dim).transpose(1, 2)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)
    return q, k


def euti_scores(hidden_states: torch.Tensor, visual_positions: list[int], decoder_layer) -> torch.Tensor:
    """Official PACT scoring: mean non-text Q, non-text K, visual hidden norm."""
    if hidden_states.shape[0] != 1 or not visual_positions:
        raise ValueError("PACT expects one image-bearing sequence at a time")
    q, k = _projected_qk(hidden_states, decoder_layer)
    prefix_end = max(visual_positions) + 1
    q = q[:, :, :prefix_end].mean(dim=2, keepdim=True).float()
    k = k[:, :, :prefix_end].float()
    if q.shape[1] != k.shape[1]:
        if q.shape[1] % k.shape[1]:
            raise ValueError("Query heads must be divisible by KV heads")
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    scores = torch.matmul(q * (q.shape[-1] ** -0.5), k.transpose(-1, -2))
    scores = F.softmax(scores, dim=-1, dtype=torch.float32).mean(dim=1)[0, 0]
    vis = torch.as_tensor(visual_positions, device=scores.device)
    scores = scores[vis]
    return scores * hidden_states[0, vis].float().norm(p=2, dim=-1)


def _clusters(keys: torch.Tensor, cutoff: float, pruned_keys: torch.Tensor):
    config = SimpleNamespace(
        synchro=False,
        avoid_numerical_instability_DBDPC=True,
        coef_pruned=1.5,
        take_mean=True,
        get_mean_position_id=False,
    )
    model = DBDPC(dc=2)
    model.fit_variant(keys, cutoff, config, pruned_keys=pruned_keys)
    return model.get_clusters(), config


def reduce_at_layer(
    hidden_states: torch.Tensor,
    visual_positions: list[int],
    decoder_layer,
    *,
    position_embeddings=None,
    prune_keep_fraction: float = 0.55,
    cutoff: float = 0.21,
    target_visual_tokens: int | None = None,
) -> PactReduction:
    """Run EUTI, DBDPC, retrieval, and hidden-state merging at one layer."""
    if not 0 < prune_keep_fraction <= 1 or not 0 <= cutoff <= 2:
        raise ValueError("Invalid PACT pruning fraction or clustering cutoff")
    vis = torch.as_tensor(visual_positions, device=hidden_states.device)
    n_keep_euti = max(1, min(len(vis), int(len(vis) * prune_keep_fraction)))
    scores = euti_scores(hidden_states, visual_positions, decoder_layer)
    important_local = scores.argsort(descending=True)[:n_keep_euti].sort().values
    selected = vis[important_local]
    removed_mask = torch.ones(len(vis), device=vis.device, dtype=torch.bool)
    removed_mask[important_local] = False
    removed = vis[removed_mask]
    all_queries, all_keys = _projected_qk(hidden_states, decoder_layer)
    if position_embeddings is not None:
        attn = decoder_layer.self_attn
        module = importlib.import_module(type(attn).__module__)
        cos, sin = position_embeddings
        if hasattr(module, "apply_multimodal_rotary_pos_emb"):
            rope = getattr(attn, "rope_scaling", None)
            if rope is None:
                rope = getattr(attn.config, "rope_parameters", {})
            section = rope["mrope_section"]
            _, all_keys = module.apply_multimodal_rotary_pos_emb(
                all_queries, all_keys, cos, sin, section)
        else:
            _, all_keys = module.apply_rotary_pos_emb(all_queries, all_keys, cos, sin)
    key_vectors = all_keys.transpose(1, 2).reshape(hidden_states.shape[1], -1).float()
    if target_visual_tokens is not None:
        if not 1 <= target_visual_tokens <= n_keep_euti:
            raise ValueError("PACT target must be between one and the EUTI survivor count")
        low, high = 0.0, 2.0
        best = None
        for trial_cutoff in [cutoff, low, high]:
            candidate, config = _clusters(key_vectors[selected], trial_cutoff, key_vectors[removed])
            error = abs(len(candidate) - target_visual_tokens)
            if best is None or error < best[0]:
                best = (error, trial_cutoff, candidate, config)
        for _ in range(10):
            trial_cutoff = (low + high) / 2
            candidate, config = _clusters(key_vectors[selected], trial_cutoff, key_vectors[removed])
            count = len(candidate)
            error = abs(count - target_visual_tokens)
            if error < best[0]:
                best = (error, trial_cutoff, candidate, config)
            if count > target_visual_tokens:
                low = trial_cutoff
            else:
                high = trial_cutoff
        _, cutoff, clusters, config = best
    else:
        clusters, config = _clusters(key_vectors[selected], cutoff, key_vectors[removed])
    merged, center_mask, sizes, _ = merge_clusters(
        hidden_states[0, selected], clusters, config,
        pruned_hiddens=hidden_states[0, removed],
    )
    selected_centers = selected[center_mask[:, 0]]
    keep_set = set(selected_centers.tolist())
    keep_seq = [i for i in range(hidden_states.shape[1]) if i not in set(visual_positions) or i in keep_set]
    updated = hidden_states.clone()
    updated[0, selected] = merged
    output = updated[:, keep_seq]
    size_by_pos = torch.ones(hidden_states.shape[1], device=hidden_states.device, dtype=torch.float32)
    size_by_pos[selected] = sizes
    return PactReduction(
        hidden_states=output,
        kept_positions=keep_seq,
        log_token_sizes=size_by_pos[keep_seq].clamp_min(1).log(),
        cutoff=cutoff,
    )
