"""Cached-signal AgilePruner selection using the released selector."""

from baselines.agilepruner import selector
from baselines.common.masks import mask_from_indices


def select_cached(shared, n_keep: int):
    indices = selector.select(
        shared.visual_scores().unsqueeze(0), shared.embeds.float().cpu(), n_keep
    )
    return mask_from_indices(shared.n_vis, indices.tolist()), {
        "mask_source": "official_agilepruner_selector_qwen_visual_attention",
    }
