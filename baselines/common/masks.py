"""Shared visual-token budget and mask operations for baseline selectors."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def budget_to_keep(n_vis: int, budget_percent: float) -> int:
    if n_vis <= 0:
        return 0
    return max(1, min(n_vis, int(math.floor(n_vis * budget_percent / 100.0))))


def mask_from_indices(n_vis: int, indices: list[int]) -> list[bool]:
    keep = set(int(i) for i in indices[:n_vis] if 0 <= int(i) < n_vis)
    return [i in keep for i in range(n_vis)]


def topk_mask(scores: torch.Tensor, n_keep: int) -> list[bool]:
    if scores.numel() == 0:
        return []
    n_keep = max(1, min(int(n_keep), int(scores.numel())))
    keep = scores.float().cpu().argsort(descending=True)[:n_keep].tolist()
    return mask_from_indices(int(scores.numel()), keep)


def norm01(values: torch.Tensor) -> torch.Tensor:
    values = values.float().cpu()
    if values.numel() == 0:
        return values
    lo, hi = values.min(), values.max()
    if float((hi - lo).abs()) <= 1e-9:
        return torch.zeros_like(values)
    return (values - lo) / (hi - lo)


def cosine_greedy_diversity(embeds: torch.Tensor, n_keep: int,
                            seeds: list[int] | None = None) -> list[int]:
    n_vis = int(embeds.shape[0])
    n_keep = max(1, min(n_keep, n_vis))
    vis = F.normalize(embeds.float().cpu(), dim=-1)
    selected: list[int] = []
    seen = set()
    for idx in seeds or []:
        if 0 <= idx < n_vis and idx not in seen:
            selected.append(idx)
            seen.add(idx)
        if len(selected) >= n_keep:
            return selected
    if not selected:
        centroid = F.normalize(vis.mean(dim=0), dim=0)
        first = int(torch.matmul(vis, centroid).argmax().item())
        selected.append(first)
        seen.add(first)
    sim = torch.matmul(vis, vis.T)
    max_sim = torch.full((n_vis,), -float("inf"))
    for idx in selected:
        max_sim = torch.maximum(max_sim, sim[idx])
    while len(selected) < n_keep:
        cand = max_sim.clone()
        for idx in seen:
            cand[idx] = float("inf")
        nxt = int(cand.argmin().item())
        if nxt in seen:
            break
        selected.append(nxt)
        seen.add(nxt)
        max_sim = torch.maximum(max_sim, sim[nxt])
    return selected
