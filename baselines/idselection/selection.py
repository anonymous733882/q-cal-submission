"""Cached-signal ID-Selection with Gaussian diversity suppression."""

import torch

from baselines.common.masks import mask_from_indices


def select_cached(shared, n_keep: int):
    n_vis = shared.n_vis
    scores = shared.attention(2)
    selected: list[int] = []
    dist_sq = (1.0 - shared.similarity()).clamp(min=0.0).pow(2)
    weights = torch.exp(-20.0 * dist_sq)
    cur = scores.clone()
    available = torch.ones(n_vis, dtype=torch.bool)
    while bool(available.any()) and len(selected) < n_keep:
        masked = cur.clone()
        masked[~available] = -float("inf")
        idx = int(masked.argmax().item())
        selected.append(idx)
        available[idx] = False
        cur -= weights[idx] * float(cur[idx])
        cur.clamp_(min=0.0)
    return mask_from_indices(n_vis, selected), {
        "mask_source": "turn_attention_l2_sample_similarity",
    }
