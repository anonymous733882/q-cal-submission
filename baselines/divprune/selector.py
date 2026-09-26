"""DivPrune selection from vbdi/divprune, commit 799e2d950aa0.

Source: LLaVA/llava/model/llava_arch.py, DivPrune. CC BY-NC 4.0 license.
"""

import torch


def select(visual_feature_vectors: torch.Tensor, image_feature_length: int,
           cosine_matrix: torch.Tensor | None = None) -> torch.Tensor:
    if visual_feature_vectors.ndim != 2 or not 0 < image_feature_length <= visual_feature_vectors.shape[0]:
        raise ValueError("Invalid visual features or token budget")
    if cosine_matrix is None:
        norm_matrix = visual_feature_vectors / visual_feature_vectors.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cosine_matrix = 1.0 - torch.mm(norm_matrix, norm_matrix.t())

    selected = torch.empty(image_feature_length, dtype=torch.long, device=visual_feature_vectors.device)
    for i in range(image_feature_length):
        if i == 0:
            distances = cosine_matrix
            scores = torch.topk(distances, 2, dim=0, largest=False).values[1, :]
        else:
            distances = torch.index_select(cosine_matrix, 0, selected[:i])
            scores = torch.min(distances, dim=0).values
        selected[i] = torch.argmax(scores)
    return selected
