"""FasterVLM selection from Theia-4869/FasterVLM, commit f5d5f12037e1.

Source: llava/model/llava_arch.py, encode_images. Apache-2.0 license.
The Qwen adapter supplies visual-token features and received-attention scores.
"""

import torch


def select(image_features: torch.Tensor, image_attentions: torch.Tensor, visual_token_num: int) -> torch.Tensor:
    if image_features.ndim != 3 or image_attentions.ndim != 3:
        raise ValueError("Expected batched features [B,N,C] and attention [B,H,N]")
    batch, n_tokens, _ = image_features.shape
    if not 0 < visual_token_num <= n_tokens:
        raise ValueError("Invalid visual token budget")
    image_attentions = image_attentions.mean(dim=1)
    token_indices = torch.topk(image_attentions, k=visual_token_num, dim=1)[1]
    index_masks = torch.zeros(batch, n_tokens, dtype=torch.bool, device=image_features.device)
    index_masks.scatter_(1, token_indices, True)
    return index_masks
