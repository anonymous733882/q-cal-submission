"""Cached-signal D2Pruner pivot and independent-set selection."""

import math

import torch


def select_cached(shared, n_keep: int):
    n_vis = shared.n_vis
    scores = shared.attention(2)
    debiased = scores.float().cpu() / shared.d2_bias()

    n_pivot = max(1, min(n_keep, int(round(n_keep * 0.7))))
    pivots = set(int(i) for i in debiased.argsort(descending=True)[:n_pivot].tolist())
    n_supplement = n_keep - len(pivots)
    if n_supplement <= 0:
        return [i in pivots for i in range(n_vis)], {
            "mask_source": "calibrated_d2_bias_attention_pivots",
        }

    remaining = [i for i in range(n_vis) if i not in pivots]
    emb = shared.normalized_embeds()
    rem_idx = torch.tensor(remaining, dtype=torch.long)
    rem_sim = emb[rem_idx] @ emb[rem_idx].T
    side = int(math.sqrt(n_vis))
    if side * side < n_vis:
        side += 1
    spatial_radius = 2
    sim_threshold = 0.8
    adjacency = [set() for _ in remaining]
    for i, gi in enumerate(remaining):
        ri, ci = divmod(gi, side)
        for j in range(i + 1, len(remaining)):
            gj = remaining[j]
            rj, cj = divmod(gj, side)
            spatial_close = abs(ri - rj) <= spatial_radius and abs(ci - cj) <= spatial_radius
            semantic_close = float(rem_sim[i, j]) > sim_threshold
            if spatial_close or semantic_close:
                adjacency[i].add(j)
                adjacency[j].add(i)

    rem_scores = debiased[rem_idx]
    selected: list[int] = []
    available = set(range(len(remaining)))
    for local_idx in rem_scores.argsort(descending=True).tolist():
        local_idx = int(local_idx)
        if local_idx not in available:
            continue
        selected.append(remaining[local_idx])
        available.discard(local_idx)
        available.difference_update(adjacency[local_idx])
        if len(selected) >= n_supplement:
            break
    if len(selected) < n_supplement:
        selected_set = pivots | set(selected)
        for local_idx in rem_scores.argsort(descending=True).tolist():
            global_idx = remaining[int(local_idx)]
            if global_idx not in selected_set:
                selected.append(global_idx)
                selected_set.add(global_idx)
            if len(selected) >= n_supplement:
                break
    keep = pivots | set(selected)
    return [i in keep for i in range(n_vis)], {
        "mask_source": "calibrated_d2_bias_attention_mis",
    }
