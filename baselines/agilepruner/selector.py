"""AgilePruner selector from cvsp-lab/AgilePruner, commit 6f63dd248174.

Source: llava/model/llava_arch.py, effective_rank and
select_diverse_tokens_by_attention_and_distance. Apache-2.0 license.
"""

import torch


ERANK_AVG_REF = 90
TAU_MAX = 0.25


def effective_rank(features: torch.Tensor) -> torch.Tensor:
    x = features.float().clone()
    x -= x.mean(dim=0, keepdim=True)
    covariance = torch.mm(x, x.T)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    singular = torch.sqrt(torch.clamp(eigenvalues, min=1e-12))
    probabilities = singular / (singular.sum() + 1e-12)
    entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum()
    return torch.exp(entropy)


def select(image_attentions: torch.Tensor, features: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if image_attentions.ndim != 2 or image_attentions.shape[0] != 1:
        raise ValueError("Expected image attention [1,N]")
    if features.ndim != 2 or features.shape[0] != image_attentions.shape[1]:
        raise ValueError("Visual features and attention do not align")
    if not 0 < max_tokens <= features.shape[0]:
        raise ValueError("Invalid visual token budget")
    erank_input = effective_rank(features).item()
    normalized = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    distance = 1.0 - torch.mm(normalized, normalized.t())
    attention_scores = image_attentions[0]
    n_tokens = attention_scores.shape[0]
    sorted_indices = torch.argsort(attention_scores, descending=True)
    rank_of = torch.empty(n_tokens, dtype=torch.long, device=attention_scores.device)
    rank_of[sorted_indices] = torch.arange(1, n_tokens + 1, device=attention_scores.device)
    alive = torch.ones(n_tokens, dtype=torch.bool, device=attention_scores.device)
    selected_indices = []
    for token_idx in sorted_indices:
        idx = token_idx.item()
        if not alive[idx]:
            continue
        if len(selected_indices) >= max_tokens:
            break
        selected_indices.append(idx)
        alive[idx] = False
        tau = min(rank_of[idx].item() * (erank_input / ERANK_AVG_REF * 0.01), TAU_MAX)
        alive[(distance[idx] < tau) & alive] = False
    if len(selected_indices) < max_tokens:
        selected_set = set(selected_indices)
        for token_idx in sorted_indices:
            if len(selected_indices) >= max_tokens:
                break
            idx = token_idx.item()
            if idx not in selected_set:
                selected_indices.append(idx)
                selected_set.add(idx)
    return torch.tensor(selected_indices, device=attention_scores.device)
