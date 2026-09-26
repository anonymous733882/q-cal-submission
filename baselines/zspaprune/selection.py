"""Cached-signal ZSPAPrune selection."""

import torch

from baselines.common.masks import cosine_greedy_diversity, mask_from_indices


def select_cached(shared, n_keep: int):
    prompt = shared.text_embedding_mean()
    vis = shared.normalized_embeds()
    sim = torch.matmul(vis, prompt)
    n_core = max(1, min(n_keep, int(round(n_keep * 0.6))))
    seeds = sim.argsort(descending=True)[:n_core].tolist()
    selected = cosine_greedy_diversity(shared.embeds, n_keep, seeds=seeds)
    return mask_from_indices(shared.n_vis, selected), {
        "mask_source": "turn_prompt_similarity_plus_sample_diversity",
    }
