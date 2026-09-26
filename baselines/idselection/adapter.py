"""ID-Selection baseline: Importance-Diversity iterative visual token selection.

Reference: "ID-Selection: Importance-Diversity Based Visual Token Selection
            for Efficient LVLM Inference"
           arXiv:2604.05601, 2026

Official algorithm for Qwen2.5-VL (no [CLS] token):
  1. Importance = cross-modal attention (cosine similarity between visual
     and instruction hidden states from early LLM layer).
  2. Diversity-aware iterative selection with Gaussian suppression:
     - Select token with highest current score
     - Compute cosine distance d(i,j) to remaining tokens
     - Gaussian weight: w_ij = exp(-γ · d²),  γ=20
     - Update: S_j ← S_j - w_ij · S_i   (subtractive suppression)
     - Repeat until k tokens selected.

Training-free, no learned components.
"""

import torch
import torch.nn.functional as F

_IMAGE_TOKEN_ID  = 151655
_VISION_START_ID = 151652
_VISION_END_ID   = 151653


class IDSelectionBaseline:
    """ID-Selection: importance + Gaussian diversity suppression.

    Args:
        model:      Qwen2VLWrapper
        llm_layer:  LLM layer for cross-modal hidden states (default 2)
        gamma:      Gaussian suppression decay (default 20, per paper)
    """

    def __init__(self, model,
                 llm_layer: int   = 2,
                 gamma:     float = 20.0):
        self.model     = model
        self.llm_layer = llm_layer
        self.gamma     = gamma

    # ------------------------------------------------------------------
    # Cross-modal importance: cosine sim between vis and text hidden states
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _importance_scores(self, model_wrapper, inputs, vis_pos, text_pos):
        """Cross-modal attention from last instruction token to visual tokens.

        Paper (arXiv:2604.05601) for Qwen2.5-VL:
          Importance = Softmax(Q_last · K_vis^T / √d), averaged over heads.
        """
        from baselines.common.attn_utils import compute_text_vis_attention

        lm    = model_wrapper.model.model.language_model
        layer = lm.layers[self.llm_layer]

        captured = {}
        def _pre(module, args):
            if isinstance(args, tuple) and len(args) > 0:
                h = args[0]
                captured["h"] = h.detach() if h.dim() == 3 else h.unsqueeze(0).detach()

        handle = layer.register_forward_pre_hook(_pre)
        try:
            model_wrapper.model(**inputs, output_hidden_states=False,
                                output_attentions=False, return_dict=True)
        finally:
            handle.remove()

        h = captured.get("h")
        if h is None or not vis_pos or not text_pos:
            return torch.zeros(len(vis_pos)), None

        # Also return hidden states for diversity computation in LLM space
        device = h.device
        v_idx = torch.tensor(vis_pos, dtype=torch.long, device=device)
        llm_vis_embeds = h[0, v_idx, :].cpu()  # [n_vis, hidden] for diversity

        scores = compute_text_vis_attention(
            layer, h, text_pos, vis_pos, inputs, lm,
            apply_rope=True, query_mode="last_text")

        return scores, llm_vis_embeds

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw,
              prune_ratio=None):
        """ID-Selection iterative selection. Returns keep_mask list[bool]."""
        ratio = prune_ratio if prune_ratio is not None else 0.5
        n_vis = vis_embeds.shape[0]
        n_keep = max(1, int(n_vis * (1.0 - ratio)))

        # Locate visual and text positions
        ids = inputs["input_ids"][0]
        vs = (ids == _VISION_START_ID).nonzero(as_tuple=True)[0]
        ve = (ids == _VISION_END_ID).nonzero(as_tuple=True)[0]
        if len(vs) > 0 and len(ve) > 0:
            vs_val, ve_val = vs[0].item(), ve[0].item()
            vis_pos  = [i for i in range(vs_val + 1, ve_val)
                       if ids[i] == _IMAGE_TOKEN_ID]
            text_pos = [i for i in range(ids.shape[0])
                       if i < vs_val or i > ve_val]
        else:
            vis_pos, text_pos = [], []

        # Importance scores (cross-modal attention from last instruction token)
        llm_vis_embeds = None
        if vis_pos and text_pos:
            scores, llm_vis_embeds = self._importance_scores(
                model_wrapper, inputs, vis_pos, text_pos)
        else:
            scores = torch.zeros(n_vis)

        # Align length
        if scores.shape[0] != n_vis:
            padded = torch.zeros(n_vis)
            m = min(scores.shape[0], n_vis)
            padded[:m] = scores[:m]
            scores = padded

        # Min-max normalize to [0, 1]
        mn, mx = scores.min(), scores.max()
        if (mx - mn).abs() > 1e-9:
            scores = (scores - mn) / (mx - mn)

        # --- Iterative selection with Gaussian suppression (Eq. 7-9) ---
        # Paper: diversity computed in LLM embedding space (post-projector)
        if llm_vis_embeds is not None and llm_vis_embeds.shape[0] == n_vis:
            emb = F.normalize(llm_vis_embeds.float(), dim=-1)
        else:
            emb = F.normalize(vis_embeds.float().cpu(), dim=-1)
        # Cosine distance matrix: d(i,j) = 1 - cos_sim(i,j)
        cos_sim = torch.matmul(emb, emb.T)
        dist_sq = (1.0 - cos_sim).clamp(min=0.0).pow(2)  # d²
        # Gaussian weight: w_ij = exp(-γ · d²)
        W = torch.exp(-self.gamma * dist_sq)  # [n_vis, n_vis]

        current_scores = scores.clone().float()
        selected = []
        available = torch.ones(n_vis, dtype=torch.bool)

        for _ in range(n_keep):
            masked = current_scores.clone()
            masked[~available] = -float('inf')
            best = masked.argmax().item()
            selected.append(best)
            available[best] = False

            # Subtractive suppression: S_j ← S_j - w_ij · S_best
            S_best = current_scores[best]
            current_scores -= W[best] * S_best
            # Clamp to prevent negative scores from dominating
            current_scores.clamp_(min=0.0)

        keep_set = set(selected)
        return [i in keep_set for i in range(n_vis)]
