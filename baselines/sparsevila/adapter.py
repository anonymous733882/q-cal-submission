"""SparseVILA context selection and decode-budget conversion."""

from __future__ import annotations

import torch


def decode_ratio(n_vis_total: int, prefill_kept: int, total_keep_percent: float) -> tuple[float, int]:
    if n_vis_total <= 0 or prefill_kept <= 0:
        return 0.0, 0
    target = max(1, min(int(prefill_kept), int(round(int(n_vis_total) * float(total_keep_percent) / 100.0))))
    return float(target) / float(prefill_kept), target


def context_mask(visual_scores: torch.Tensor, n_keep: int) -> list[bool]:
    n_vis = int(visual_scores.numel())
    if n_vis == 0:
        return []
    retained = visual_scores.float().cpu().argsort(descending=True)[:max(1, min(n_keep, n_vis))]
    selected = set(retained.tolist())
    return [i in selected for i in range(n_vis)]


def select_cached(shared, n_keep: int):
    return context_mask(shared.visual_scores(), n_keep), {
        "mask_source": "sparsevila_context_stage_query_agnostic_visual_salience",
        "paper_algorithm": "context sparsity + query-aware decode KV retrieval",
        "adapter_note": "SparseVILA uses its own decode-stage query-aware KV compaction path; it is not deduplicated with fixed-mask methods",
        "decode_stage_kv_retrieval_implemented": True,
    }
