"""Legacy cached-attention FastV selector; main FastV uses its layer hook."""

from baselines.common.masks import topk_mask


def select_cached(shared, n_keep: int):
    layer = min(2, max(0, int(getattr(shared.model, "total_layers", 1)) - 1))
    return topk_mask(shared.attention(layer), n_keep), {
        "mask_source": f"turn_attention_l{layer}",
    }
