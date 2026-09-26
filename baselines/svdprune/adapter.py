"""SVD-Prune baseline: leverage-score based visual token selection.

Reference: SVD-Prune (arXiv 2604.11530, 2026)
  1. Take visual token feature matrix F ∈ R^(T×D) from vision encoder output
  2. SVD decomposition: F = U Σ V^T
  3. Select rank k where cumulative variance ≥ ε
  4. Compute leverage scores: ℓ_t = (1/k) ||U_t,[1:k]||²
  5. Keep tokens with highest leverage scores
  6. Binary keep/drop, one-shot static, encoder-side

Applied to Qwen2-VL: operates on visual_embeds (pooler_output from vision encoder).
"""

import torch


class SVDPruneBaseline:
    """SVD-Prune: leverage-score based visual token selection.

    Args:
        epsilon: variance retention threshold (default 0.9)
        min_tokens: minimum tokens to retain (default 4)
    """

    def __init__(self, epsilon=0.9, min_tokens=4):
        self.epsilon = epsilon
        self.min_tokens = min_tokens

    @torch.no_grad()
    def score_tokens(self, visual_embeds):
        """Compute per-token leverage scores via SVD.

        Args:
            visual_embeds: [num_vis, hidden_dim] tensor from vision encoder

        Returns:
            leverage_scores: list of floats, length = num_vis
        """
        F = visual_embeds.float()  # [T, D]
        T, D = F.shape

        if T <= self.min_tokens:
            return [1.0] * T

        # SVD: F = U Σ V^T
        # Use truncated SVD for efficiency (only need left singular vectors)
        U, S, _ = torch.linalg.svd(F, full_matrices=False)
        # U: [T, min(T,D)], S: [min(T,D)]

        # Find rank k where cumulative variance ≥ ε
        variance = S ** 2
        total_var = variance.sum()
        cumvar = variance.cumsum(dim=0) / total_var
        k = (cumvar >= self.epsilon).nonzero(as_tuple=True)[0]
        if len(k) == 0:
            k = len(S)
        else:
            k = k[0].item() + 1  # +1 because we want the first index where cumvar ≥ ε
        k = max(1, min(k, len(S)))

        # Leverage scores: ℓ_t = (1/k) ||U_t,[1:k]||²
        U_k = U[:, :k]  # [T, k]
        leverage = (U_k ** 2).sum(dim=1) / k  # [T]

        return leverage.cpu().tolist()

    def allocate(self, leverage_scores, num_vis, budget_frac):
        """Binary keep/drop allocation based on leverage scores.

        Args:
            leverage_scores: list of floats per visual token
            num_vis: total visual tokens
            budget_frac: fraction to KEEP (0.0-1.0)

        Returns:
            keep_mask: list of bool, True = keep
        """
        keep_count = max(self.min_tokens, int(num_vis * budget_frac))
        keep_count = min(keep_count, num_vis)
        indexed = sorted(enumerate(leverage_scores), key=lambda x: x[1], reverse=True)
        keep_set = set(idx for idx, _ in indexed[:keep_count])
        return [i in keep_set for i in range(num_vis)]
