"""Cached-signal DivPrune selection using the released selector."""

from baselines.common.masks import mask_from_indices
from baselines.divprune import selector


def select_cached(shared, n_keep: int):
    indices = selector.select(shared.embeds.float().cpu(), n_keep)
    return mask_from_indices(shared.n_vis, indices.tolist()), {
        "mask_source": "official_divprune_selector_qwen_visual_features",
    }
