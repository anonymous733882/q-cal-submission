"""DivPrune baseline: diversity-based visual token pruning via Max-Min Distance.

Reference: "DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models"
           arXiv:2503.02175, CVPR 2025
           GitHub: vbdi/divprune

Core algorithm:
  1. Compute pairwise cosine distance between all visual token embeddings.
  2. Greedily select tokens to maximize the minimum pairwise distance among the
     selected set (Max-Min Diversity Problem / MMDP).
  3. No attention scores used — purely feature-space diversity.

  Greedy solver:
    - Start by selecting the token with highest L2 norm (most distinct).
    - Iteratively select the token with maximum minimum distance to any
      already-selected token.
    - Repeat until k tokens are selected.

Applied to Qwen2.5-VL: operates on vis_embeds (post-PatchMerger LLM-space embeddings).
One-shot, training-free, calibration-data-free.
"""

import torch
import torch.nn.functional as F


class DivPruneBaseline:
    """DivPrune: Max-Min Diversity visual token selection.

    Args:
        prune_ratio: fraction of tokens to DROP (default 0.5)
        min_tokens:  minimum tokens to retain (default 4)
    """

    def __init__(self, model=None, prune_ratio=0.5, min_tokens=4):
        self.prune_ratio = prune_ratio
        self.min_tokens = min_tokens

    @torch.no_grad()
    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw,
              prune_ratio=None):
        """Greedy Max-Min Diversity selection.

        Args:
            vis_embeds: [n_vis, hidden] visual token embeddings (post-merger)
            prune_ratio: fraction to DROP (override)

        Returns:
            keep_mask: list[bool], length = n_vis
        """
        ratio = prune_ratio if prune_ratio is not None else self.prune_ratio
        n_vis = vis_embeds.shape[0]
        n_keep = max(self.min_tokens, int(n_vis * (1.0 - ratio)))
        n_keep = min(n_keep, n_vis)

        if n_keep >= n_vis:
            return [True] * n_vis

        # Normalize embeddings for cosine distance
        emb = F.normalize(vis_embeds.float().cpu(), dim=-1)  # [n_vis, D]

        # Pairwise cosine similarity → distance = 1 - sim
        sim = torch.matmul(emb, emb.T)  # [n_vis, n_vis]

        # Greedy Max-Min selection
        # Paper Algorithm 1, Stage 1: select the token whose min pairwise
        # distance to any other token is the largest (most isolated token).
        # dist = 1 - sim (cosine distance)
        dist = 1.0 - sim
        dist.fill_diagonal_(float('inf'))  # exclude self
        min_dist_per_token = dist.min(dim=1).values  # [n_vis]
        first = min_dist_per_token.argmax().item()

        selected = [first]
        # min_dist_to_selected[i] = min cosine similarity to any selected token
        # We want to maximize min distance = minimize max similarity
        min_sim = sim[first].clone()  # [n_vis]
        min_sim[first] = float('inf')  # exclude already selected

        for _ in range(n_keep - 1):
            # Select token with minimum similarity to selected set
            # (= maximum distance to nearest selected token)
            next_idx = min_sim.argmin().item()
            selected.append(next_idx)
            # Update min similarities
            new_sim = sim[next_idx]
            min_sim = torch.min(min_sim, new_sim)
            min_sim[next_idx] = float('inf')

        keep_set = set(selected)
        return [i in keep_set for i in range(n_vis)]
