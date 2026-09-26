"""Cached-signal PTP tile-aware visual/instruction selection."""

from typing import Any

from baselines.common.masks import norm01, topk_mask


def tile_spans(model: Any, inputs: Any, n_vis: int) -> list[tuple[int, int]]:
    grid = None
    if hasattr(inputs, "raw"):
        grid = inputs.raw.get("image_grid_thw")
    else:
        grid = inputs.get("image_grid_thw")
    if grid is None or not hasattr(grid, "shape") or int(grid.shape[0]) <= 1:
        return [(0, n_vis)]
    merge = 1
    visual = getattr(getattr(getattr(model, "model", None), "model", None), "visual", None)
    if visual is not None:
        merge = int(getattr(visual, "spatial_merge_size", 1))
    group = max(1, merge * merge)
    spans = []
    offset = 0
    for row in grid:
        vals = [int(x) for x in row.tolist()]
        tile_n = max(1, vals[0] * vals[1] * vals[2] // group)
        start, end = offset, min(n_vis, offset + tile_n)
        if end > start:
            spans.append((start, end))
        offset = end
        if offset >= n_vis:
            break
    return spans or [(0, n_vis)]


def select_cached(shared, n_keep: int):
    n_vis = shared.n_vis
    bottom = shared.visual_scores()
    instr = shared.attention(2, query_mode="all_text")
    fused = 0.5 * norm01(bottom) + 0.5 * norm01(instr)
    spans = tile_spans(shared.model, shared.inputs, n_vis)
    if len(spans) == 1:
        return topk_mask(fused, n_keep), {
            "mask_source": "sample_visual_plus_turn_instruction_attention",
        }

    saliency = [float(bottom[s:e].mean()) if e > s else 0.0 for s, e in spans]
    total = sum(saliency) + 1e-9
    budgets = [max(1, int(n_keep * (s / total))) for s in saliency]
    while sum(budgets) < n_keep:
        budgets[max(range(len(saliency)), key=lambda i: saliency[i])] += 1
    while sum(budgets) > n_keep:
        worst = min((i for i, b in enumerate(budgets) if b > 1), key=lambda i: saliency[i], default=None)
        if worst is None:
            break
        budgets[worst] -= 1
    keep: set[int] = set()
    for (start, end), tile_budget in zip(spans, budgets):
        local_budget = min(tile_budget, end - start)
        local = fused[start:end].argsort(descending=True)[:local_budget].tolist()
        keep.update(start + int(i) for i in local)
    if len(keep) < n_keep:
        for idx in fused.argsort(descending=True).tolist():
            keep.add(int(idx))
            if len(keep) >= n_keep:
                break
    if len(keep) > n_keep:
        keep = set(sorted(keep, key=lambda i: float(fused[i]), reverse=True)[:n_keep])
    return [i in keep for i in range(n_vis)], {
        "mask_source": "tile_visual_plus_turn_instruction_attention",
    }
