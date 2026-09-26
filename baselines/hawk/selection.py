"""Cached-signal HAWK selection."""

from baselines.common.masks import topk_mask


def select_cached(shared, n_keep: int):
    return topk_mask(shared.hawk_scores(), n_keep), {
        "mask_source": "calibrated_hawk_head_weighted_layer0",
    }
