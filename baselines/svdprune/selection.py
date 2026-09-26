"""Cached-signal SVD-Prune selection."""

from baselines.common.masks import topk_mask


def select_cached(shared, n_keep: int):
    return topk_mask(shared.svd_scores(), n_keep), {"mask_source": "sample_svd_leverage"}
