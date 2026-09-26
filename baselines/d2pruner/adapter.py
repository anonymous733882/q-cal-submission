"""D²Pruner baseline: Debiased Importance + Structural Diversity token pruning.

Reference: "D2Pruner: Debiased Importance and Structural Diversity for
            MLLM Token Pruning"
           arXiv:2512.19443, AAAI 2026
           GitHub: EvelynZhang-epiclab/D2Pruner

Core algorithm (two components):

  1. Debiased Importance (DI):
     - Compute text→visual attention at early LLM layer.
     - Model positional bias: attention scores suffer from recency bias
       (later tokens get higher attention). Estimate positional trend via
       exponential decay model and subtract it.
     - debiased_score[i] = raw_score[i] - bias[i]
       where bias is estimated from position-dependent mean.
     - Select top-p tokens as "pivot tokens" (core set).
       p = pivot_ratio * n_keep

  2. Structural Diversity (SD) via Maximal Independent Set (MIS):
     - For remaining non-pivot tokens, build a hybrid graph:
       - Spatial edge: connect tokens within spatial_radius in 2D grid
       - Semantic edge: connect tokens with cosine similarity > sim_threshold
     - Greedy MIS on hybrid graph: iteratively pick highest-importance
       available token, then remove all its graph neighbors.
     - Final set = pivot tokens + MIS supplementary tokens.

Applied to Qwen2.5-VL: LLM layer 2 for attention, vis_embeds for features.
One-shot, training-free.
"""

import torch
import torch.nn.functional as F
import math

_IMAGE_TOKEN_ID  = 151655
_VISION_START_ID = 151652
_VISION_END_ID   = 151653


class D2PrunerBaseline:
    """D²Pruner: debiased pivots + MIS structural diversity.

    Args:
        model:           Qwen2VLWrapper
        score_layer:     LLM layer for attention scoring (default 2)
        pivot_ratio:     fraction of budget for pivot tokens (default 0.7, per paper §Impl.)
        sim_threshold:   cosine similarity threshold for semantic edges (default 0.8, per paper §Impl.)
        spatial_radius:  spatial proximity radius for spatial edges (default 2)
        prune_ratio:     fraction of tokens to DROP (default 0.5)
    """

    def __init__(self, model,
                 score_layer:    int   = 2,
                 pivot_ratio:    float = 0.7,
                 sim_threshold:  float = 0.8,
                 spatial_radius: int   = 2,
                 prune_ratio:    float = 0.5):
        self.model           = model
        self.score_layer     = score_layer
        self.pivot_ratio     = pivot_ratio
        self.sim_threshold   = sim_threshold
        self.spatial_radius  = spatial_radius
        self.prune_ratio     = prune_ratio

    # ------------------------------------------------------------------
    # LLM text→visual attention
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _raw_attention_scores(self, model_wrapper, inputs, vis_pos, text_pos):
        """Compute last-text-token → visual attention with layernorm + RoPE.

        Paper: uses last text token as query (not all text mean).
        """
        from baselines.common.attn_utils import compute_text_vis_attention

        lm    = model_wrapper.model.model.language_model
        layer = lm.layers[self.score_layer]

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
            return torch.zeros(len(vis_pos))

        # D2Pruner paper: query = last text token only
        return compute_text_vis_attention(
            layer, h, text_pos, vis_pos, inputs, lm,
            apply_rope=True, query_mode="last_text")

    # ------------------------------------------------------------------
    # Debiasing: remove positional trend
    # ------------------------------------------------------------------

    def _debias_scores(self, raw_scores):
        """Remove positional bias using pre-computed COCO bias prior.

        Paper (arXiv:2512.19443): A_rel = A_ori / (A_bias + ε)
        where A_bias is the average attention over 1000 COCO images.

        Falls back to division-by-mean if bias file not found.
        """
        n = len(raw_scores)
        if n <= 1:
            return raw_scores.clone()

        scores = raw_scores.float()
        bias = self._load_bias(n)
        if bias is not None:
            return scores / (bias + 1e-7)

        # Fallback: divide by position-smoothed mean (better than linear OLS)
        mean_val = scores.mean()
        if mean_val.abs() < 1e-9:
            return scores
        return scores / (mean_val + 1e-7)

    def _load_bias(self, n_vis):
        """Load pre-computed COCO attention bias, resizing to match n_vis."""
        import os
        bias_path = os.path.join(os.path.dirname(__file__),
                                 "d2pruner_coco_bias.pt")
        if not os.path.exists(bias_path):
            return None
        bias = torch.load(bias_path, map_location="cpu", weights_only=True)
        if bias.shape[0] != n_vis:
            # Interpolate to match current visual token count
            bias = torch.nn.functional.interpolate(
                bias.unsqueeze(0).unsqueeze(0),
                size=n_vis, mode="linear", align_corners=False
            ).squeeze()
        return bias.float()

    # ------------------------------------------------------------------
    # 2D spatial grid for spatial edges
    # ------------------------------------------------------------------

    @staticmethod
    def _build_spatial_grid(n_vis, grid_thw):
        """Infer 2D grid positions for visual tokens."""
        if grid_thw is not None and grid_thw.shape[0] >= 1:
            # Use first tile's grid dimensions
            t, h, w = grid_thw[0].tolist()
            total = int(t * h * w)
            # After spatial merge (2x2), tokens are h/2 x w/2 grid
            # But we don't know merge size here, so approximate
            side = int(math.sqrt(n_vis))
            if side * side < n_vis:
                side += 1
        else:
            side = int(math.sqrt(n_vis))
            if side * side < n_vis:
                side += 1

        positions = []
        for i in range(n_vis):
            r = i // side
            c = i % side
            positions.append((r, c))
        return positions

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw,
              prune_ratio=None):
        """D²Pruner selection. Returns keep_mask list[bool]."""
        ratio = prune_ratio if prune_ratio is not None else self.prune_ratio
        n_vis = vis_embeds.shape[0]
        n_keep = max(1, int(n_vis * (1.0 - ratio)))

        # --- Get attention scores ---
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

        if vis_pos and text_pos:
            raw_scores = self._raw_attention_scores(
                model_wrapper, inputs, vis_pos, text_pos)
        else:
            raw_scores = torch.zeros(n_vis)

        if raw_scores.shape[0] != n_vis:
            padded = torch.zeros(n_vis)
            m = min(raw_scores.shape[0], n_vis)
            padded[:m] = raw_scores[:m]
            raw_scores = padded

        # --- Step 1: Debiased importance → pivot tokens ---
        debiased = self._debias_scores(raw_scores)
        n_pivot = max(1, int(n_keep * self.pivot_ratio))

        pivot_ranked = debiased.argsort(descending=True).tolist()
        pivots = set(pivot_ranked[:n_pivot])

        # --- Step 2: MIS on remaining tokens ---
        n_supplement = n_keep - len(pivots)
        if n_supplement <= 0:
            keep_set = pivots
        else:
            remaining = [i for i in range(n_vis) if i not in pivots]

            # Build hybrid graph (spatial + semantic edges)
            emb = F.normalize(vis_embeds.float().cpu(), dim=-1)
            positions = self._build_spatial_grid(n_vis, grid_thw)

            # For efficiency, compute semantic similarity only among remaining
            rem_idx = torch.tensor(remaining, dtype=torch.long)
            rem_emb = emb[rem_idx]
            rem_sim = torch.matmul(rem_emb, rem_emb.T)  # [n_rem, n_rem]

            # Build adjacency for remaining tokens
            n_rem = len(remaining)
            adj = [set() for _ in range(n_rem)]

            # Paper: S_fused = α·Ŝ_sem + (1-α)·S_spat, α=1.0 (understanding tasks)
            # With α=1.0, spatial component is zero → only semantic edges matter
            alpha_fuse = 1.0
            for i in range(n_rem):
                for j in range(i + 1, n_rem):
                    gi, gj = remaining[i], remaining[j]
                    # Spatial distance
                    ri, ci = positions[gi]
                    rj, cj = positions[gj]
                    spatial_close = (abs(ri - rj) <= self.spatial_radius and
                                    abs(ci - cj) <= self.spatial_radius)
                    # Semantic edge
                    semantic_close = rem_sim[i, j].item() > self.sim_threshold

                    # Weighted fusion: only connect if fused criterion met
                    sem_edge = 1.0 if semantic_close else 0.0
                    spat_edge = 1.0 if spatial_close else 0.0
                    fused = alpha_fuse * sem_edge + (1.0 - alpha_fuse) * spat_edge
                    if fused > 0.5:
                        adj[i].add(j)
                        adj[j].add(i)

            # Greedy MIS: pick highest debiased-importance, remove neighbors
            rem_scores = debiased[rem_idx]
            sorted_rem = rem_scores.argsort(descending=True).tolist()

            available = set(range(n_rem))
            mis_selected = []

            for local_idx in sorted_rem:
                if local_idx not in available:
                    continue
                mis_selected.append(remaining[local_idx])
                # Remove neighbors
                available.discard(local_idx)
                for nb in adj[local_idx]:
                    available.discard(nb)
                if len(mis_selected) >= n_supplement:
                    break

            # If not enough, fill with remaining by score
            if len(mis_selected) < n_supplement:
                all_selected = pivots | set(mis_selected)
                for local_idx in sorted_rem:
                    gi = remaining[local_idx]
                    if gi not in all_selected:
                        mis_selected.append(gi)
                    if len(mis_selected) >= n_supplement:
                        break

            keep_set = pivots | set(mis_selected)

        return [i in keep_set for i in range(n_vis)]
