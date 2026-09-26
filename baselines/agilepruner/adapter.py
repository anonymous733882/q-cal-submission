"""AgilePruner baseline: adaptive attention + diversity visual token pruning.

Reference: "AgilePruner: An Empirical Study of Attention and Diversity for
            Adaptive Visual Token Pruning in Large Vision-Language Models"
           arXiv:2603.01236, ICLR 2026
           GitHub: cvsp-lab/AgilePruner

Core algorithm:
  1. Compute text→visual attention at LLM Layer 1 (second transformer layer,
     0-indexed: layer_idx=1) using all text tokens as query (all_text mode).
     attention_score(vis_i) = mean_heads mean_text softmax(Q_text K_vis^T/√d)[vis_i]

  2. Compute image complexity indicators from LLM Layer 1 attention:
     - attention_entropy H = -sum_i p_i log(p_i) where p = softmax of scores
       High entropy → complex scene → attention scores less reliable
     - Normalized entropy h = H / log(N_vis)  ∈ [0, 1]

  3. Adaptive diversity threshold τ (cosine similarity deduplication):
     τ = τ_base + (τ_max - τ_base) * (1 - h)
     Simple images (low h) → low τ → aggressive dedup (remove more similar tokens)
     Complex images (high h) → high τ → lenient dedup (preserve more tokens)
     Default: τ_base=0.7, τ_max=0.95  (from official code)

  4. Two-phase selection:
     Phase A: rank vis tokens by attention score, keep top-K
     Phase B: from remaining tokens, remove those with cosine sim > τ to
              any already-kept token; add surviving diverse ones to fill budget

     Total budget K = n_vis * keep_frac.

  Note: AgilePruner officially supports Qwen2.5-VL-7B; this implementation
  follows the Qwen2.5-VL code path in cvsp-lab/AgilePruner.
"""

import torch
import torch.nn.functional as F
import math

_IMAGE_TOKEN_ID  = 151655
_VISION_START_ID = 151652
_VISION_END_ID   = 151653


class AgilePrunerBaseline:
    """AgilePruner: adaptive text→vis attention + diversity for Qwen2.5-VL.

    Args:
        model:       Qwen2VLWrapper
        score_layer: LLM layer index to extract attention from (default 1)
        prune_ratio: fraction of tokens to DROP (default 0.5)
        tau_base:    minimum cosine similarity dedup threshold (simple images)
        tau_max:     maximum cosine similarity dedup threshold (complex images)
    """

    def __init__(self, model,
                 score_layer: int   = 1,
                 prune_ratio: float = 0.5,
                 tau_base:    float = 0.7,
                 tau_max:     float = 0.95):
        self.model       = model
        self.score_layer = score_layer
        self.prune_ratio = prune_ratio
        self.tau_base    = tau_base
        self.tau_max     = tau_max

    # ------------------------------------------------------------------
    # Token position helpers
    # ------------------------------------------------------------------

    def _get_positions(self, input_ids):
        ids = input_ids[0]
        vis_start = (ids == _VISION_START_ID).nonzero(as_tuple=True)[0]
        vis_end   = (ids == _VISION_END_ID  ).nonzero(as_tuple=True)[0]
        if len(vis_start) == 0 or len(vis_end) == 0:
            return [], []
        vs, ve = vis_start[0].item(), vis_end[0].item()
        vis_pos  = [i for i in range(vs + 1, ve) if ids[i] == _IMAGE_TOKEN_ID]
        text_pos = [i for i in range(ids.shape[0])
                    if i < vs or i > ve]
        return vis_pos, text_pos

    # ------------------------------------------------------------------
    # Layer-1 attention capture
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _l1_attention_scores(self, model, inputs, vis_pos, text_pos):
        """Compute text→vis attention at LLM layer 1 (0-indexed).

        Returns:
            scores: Tensor [n_vis], higher = more important
        """
        from baselines.common.attn_utils import compute_text_vis_attention

        lm     = model.model.model.language_model
        layer  = lm.layers[self.score_layer]

        captured = {}

        def _pre(module, args):
            if isinstance(args, tuple) and len(args) > 0:
                h = args[0]
                captured["h"] = h.detach() if h.dim() == 3 else h.unsqueeze(0).detach()

        handle = layer.register_forward_pre_hook(_pre)
        try:
            model.model(**inputs, output_hidden_states=False,
                        output_attentions=False, return_dict=True)
        finally:
            handle.remove()

        h = captured.get("h")
        if h is None or not vis_pos or not text_pos:
            return torch.zeros(len(vis_pos))

        return compute_text_vis_attention(
            layer, h, text_pos, vis_pos, inputs, lm,
            apply_rope=True, query_mode="all_text")

    # ------------------------------------------------------------------
    # Adaptive diversity threshold
    # ------------------------------------------------------------------

    def _adaptive_tau(self, scores: torch.Tensor) -> float:
        """Compute adaptive cosine similarity dedup threshold from entropy."""
        probs = F.softmax(scores.float(), dim=0)
        # Normalized entropy h ∈ [0, 1]
        n     = max(probs.shape[0], 2)
        H     = -(probs * (probs + 1e-9).log()).sum().item()
        h     = H / math.log(n)
        h     = max(0.0, min(1.0, h))
        tau   = self.tau_base + (self.tau_max - self.tau_base) * (1.0 - h)
        return tau

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw,
              prune_ratio=None):
        """AgilePruner selection. Returns keep_mask list[bool].

        Args:
            vis_embeds: [n_vis, hidden] visual token embeddings (post-merger)
        """
        ratio  = prune_ratio if prune_ratio is not None else self.prune_ratio
        n_vis  = vis_embeds.shape[0]
        n_keep = max(1, int(n_vis * (1.0 - ratio)))

        vis_pos, text_pos = self._get_positions(inputs["input_ids"])

        if not vis_pos or not text_pos:
            keep_set = set(range(n_keep))
            return [i in keep_set for i in range(n_vis)]

        # Step 1: attention scores at LLM layer 1
        scores = self._l1_attention_scores(
            model_wrapper, inputs, vis_pos, text_pos)  # [n_vis]

        if scores.shape[0] != n_vis:
            # alignment safety
            padded = torch.zeros(n_vis)
            m = min(scores.shape[0], n_vis)
            padded[:m] = scores[:m]
            scores = padded

        # Step 2: adaptive tau from entropy
        tau = self._adaptive_tau(scores)

        # Step 3: Phase A — top attention tokens
        ranked   = torch.argsort(scores, descending=True).tolist()
        important = set(ranked[:n_keep])  # start with full budget

        # Step 4: Phase B — deduplication among important tokens
        # Remove tokens whose cosine similarity to a higher-ranked token > tau
        emb    = F.normalize(vis_embeds.float(), dim=-1)  # [n_vis, D]
        kept   = []
        kept_emb = []

        for idx in ranked:
            if len(kept) >= n_keep:
                break
            if not kept_emb:
                kept.append(idx)
                kept_emb.append(emb[idx])
                continue
            emb_stack = torch.stack(kept_emb, dim=0)  # [n_kept, D]
            max_sim = (emb_stack @ emb[idx]).max().item()
            if max_sim < tau:
                kept.append(idx)
                kept_emb.append(emb[idx])

        # If dedup removed too many, fill from remaining high-attention tokens
        kept_set = set(kept)
        if len(kept_set) < n_keep:
            for idx in ranked:
                if idx not in kept_set:
                    kept_set.add(idx)
                if len(kept_set) >= n_keep:
                    break

        return [i in kept_set for i in range(n_vis)]
