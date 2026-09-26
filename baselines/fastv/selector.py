"""FastV ranking from pkunlp-icler/FastV, commit d1659729b5bf.

Source: src/transformers/src/transformers/models/llama/modeling_llama.py,
whose file header states Apache-2.0. The model-specific layer hook is in
baselines/fastv/manipulator.py.
"""

import torch


def select(last_layer_attention: torch.Tensor, visual_positions: list[int],
           attention_rank: int) -> torch.Tensor:
    if last_layer_attention.ndim != 4 or last_layer_attention.shape[0] != 1:
        raise ValueError("Expected attention [1,H,Q,K]")
    if not 0 < attention_rank <= len(visual_positions):
        raise ValueError("Invalid FastV retention count")
    last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
    last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
    visual = torch.as_tensor(visual_positions, device=last_layer_attention.device)
    last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[visual]
    return visual[last_layer_attention_avg_last_tok_image.topk(attention_rank).indices]
