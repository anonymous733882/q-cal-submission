"""VisPruner selection from Theia-4869/VisPruner, commit aefa01adc7c7.

Source: llava/model/llava_arch.py, encode_images. Apache-2.0 license.
The Qwen adapter supplies visual-token features and received-attention scores.
"""

import torch


def select(image_features: torch.Tensor, image_attentions: torch.Tensor,
           visual_token_num: int, important_ratio: float = 0.8) -> torch.Tensor:
    if image_features.ndim != 3 or image_attentions.ndim != 3:
        raise ValueError("Expected batched features [B,N,C] and attention [B,H,N]")
    batch, n_tokens, _ = image_features.shape
    if not 0 < visual_token_num <= n_tokens or not 0 <= important_ratio <= 1:
        raise ValueError("Invalid selection budget or important ratio")
    important_token_num = int(visual_token_num * important_ratio)
    diverse_token_num = visual_token_num - important_token_num

    image_attentions = image_attentions.mean(dim=1)
    token_indices = image_attentions.argsort(dim=-1, descending=True)
    important_indices = token_indices[:, :important_token_num]
    residual_indices = token_indices[:, important_token_num:]

    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    while diverse_token_num > 0:
        residual_count = residual_indices.shape[1]
        remove_count = min(8, residual_count - diverse_token_num)
        if remove_count <= 0:
            break
        residual_tokens = image_normalized[
            torch.arange(batch, device=image_features.device).unsqueeze(-1).expand(-1, residual_count),
            residual_indices,
        ]
        even, odd = residual_tokens[..., ::2, :], residual_tokens[..., 1::2, :]
        scores = (even @ odd.transpose(-1, -2)).max(dim=-1).values
        distinct_indices = scores.argsort(dim=-1, descending=True)[:, remove_count:]
        residual_indices = torch.cat([
            residual_indices[..., ::2][
                torch.arange(batch, device=image_features.device).unsqueeze(-1).expand(
                    -1, distinct_indices.shape[1]), distinct_indices],
            residual_indices[..., 1::2],
        ], dim=-1)

    selected_indices = torch.cat([important_indices, residual_indices], dim=-1)
    if selected_indices.shape[1] != visual_token_num:
        raise RuntimeError("VisPruner selection did not meet the requested budget")
    index_masks = torch.zeros(batch, n_tokens, dtype=torch.bool, device=image_features.device)
    index_masks.scatter_(1, selected_indices, True)
    return index_masks
