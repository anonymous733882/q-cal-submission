"""Cached-signal FasterVLM selection using the released selector."""

from baselines.fastervlm import selector


def select_cached(shared, n_keep: int):
    selected = selector.select(
        shared.embeds.float().cpu().unsqueeze(0),
        shared.visual_scores().reshape(1, 1, -1), n_keep,
    )
    return selected[0].tolist(), {
        "mask_source": "official_fastervlm_selector_qwen_visual_attention",
    }
