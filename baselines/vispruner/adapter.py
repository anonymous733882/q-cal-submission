"""VisPruner baseline: ViT attention + diversity-based visual token pruning.

Reference: "Beyond Text-Visual Attention: Exploiting Visual Cues for Effective
            Token Pruning" arXiv:2412.01818v2, VisPruner (ICCV 2025)
           GitHub: Theia-4869/VisPruner

Core algorithm (two-stage):
  Stage 1 — Importance selection (identical to FasterVLM):
    score(patch_i) = mean_j [ softmax(QK^T/√d)[j,i] ]  (received attention)
    from ViT last full-attention block.
    Select top-α fraction as "significant tokens" S.

  Stage 2 — Diversity-based deduplication:
    From remaining tokens R = all \ S, compute cosine similarity between each
    token in R and every token in S. Remove tokens in R that are highly similar
    to any token in S (sim > θ_dup). Add surviving diverse tokens D to S.
    Final kept set = S ∪ D, up to total budget K.

    In practice (following official code):
      θ_dup = 0.9 (cosine similarity threshold for deduplication)
      α = importance_ratio (what fraction of budget to fill with important tokens)
        e.g., if budget=50%, α=0.8 → keep top 40% important + up to 10% diverse

Adaptation note (Qwen2.5-VL):
  No [CLS] token; we use mean received attention from ViT block 31 (same as
  FasterVLM). Patches are scored before PatchMerger and then averaged per
  merge group to get LLM visual token scores.

  For Stage 2 diversity, we use the LLM-space visual embeddings (post-merger)
  for cosine similarity computation, which is consistent with what the model
  actually processes.
"""

import torch
import torch.nn.functional as F

_IMAGE_TOKEN_ID = 151655


class VisPrunerBaseline:
    """VisPruner: importance + diversity two-stage visual token selection.

    Args:
        model:            Qwen2VLWrapper
        vit_block_idx:    ViT full-attention block to hook (default 31)
        prune_ratio:      fraction of tokens to DROP (default 0.5)
        importance_ratio: fraction of budget filled by importance selection
                          (rest filled by diversity). Default 0.8.
        dup_threshold:    cosine similarity threshold for deduplication (default 0.9)
    """

    def __init__(self, model,
                 vit_block_idx:    int   = 31,
                 prune_ratio:      float = 0.5,
                 importance_ratio: float = 0.8,
                 dup_threshold:    float = 0.9):
        self.model            = model
        self.block_idx        = vit_block_idx
        self.prune_ratio      = prune_ratio
        self.importance_ratio = importance_ratio
        self.dup_threshold    = dup_threshold

    # ------------------------------------------------------------------
    # ViT received attention  (identical to FasterVLM)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _vit_received_attention(self, pixel_values, image_grid_thw):
        """Returns patch-level received attention scores [N_patches]."""
        vit    = self.model.model.model.visual
        block  = vit.blocks[self.block_idx]
        attn_m = block.attn

        captured = {}

        def _pre(module, args):
            captured["h"] = args[0].detach()

        handle = block.register_forward_pre_hook(_pre)
        try:
            _ = vit(pixel_values, image_grid_thw)
        finally:
            handle.remove()

        h = captured["h"]
        N = h.shape[0]
        num_heads = attn_m.num_heads
        head_dim  = attn_m.head_dim

        qkv = attn_m.qkv(h)
        q, k, _ = qkv.chunk(3, dim=-1)
        q = q.view(N, num_heads, head_dim).permute(1, 0, 2).float()
        k = k.view(N, num_heads, head_dim).permute(1, 0, 2).float()

        attn = F.softmax(
            torch.matmul(q, k.transpose(-1, -2)) * (head_dim ** -0.5), dim=-1)
        return attn.mean(dim=1).mean(dim=0).cpu()  # [N_patches]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw,
              prune_ratio=None):
        """Two-stage VisPruner selection. Returns keep_mask list[bool].

        Args:
            vis_embeds: [N_vis, hidden] LLM-space visual token embeddings
                        (post PatchMerger). Used for Stage 2 diversity.
            prune_ratio: override if given
        """
        ratio  = prune_ratio if prune_ratio is not None else self.prune_ratio
        n_vis  = vis_embeds.shape[0]
        n_keep = max(1, int(n_vis * (1.0 - ratio)))

        # --- Stage 1: importance scores from ViT ---
        pixel_values   = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        if pixel_values is None:
            # fallback: keep top-n_keep by L2 norm
            norms = vis_embeds.float().norm(dim=-1)
            keep_set = set(norms.argsort(descending=True)[:n_keep].tolist())
            return [i in keep_set for i in range(n_vis)]

        patch_scores = self._vit_received_attention(pixel_values, image_grid_thw)
        merge = self.model.model.model.visual.spatial_merge_size
        m2    = merge * merge
        N_vis_from_patches = patch_scores.shape[0] // m2
        # average within merge groups
        token_scores = patch_scores[: N_vis_from_patches * m2] \
                           .view(N_vis_from_patches, m2).mean(dim=1)  # [n_vis]
        # align length
        if N_vis_from_patches != n_vis:
            # fallback: truncate or pad
            min_n = min(N_vis_from_patches, n_vis)
            padded = torch.zeros(n_vis)
            padded[:min_n] = token_scores[:min_n]
            token_scores = padded

        # Number of important tokens (Stage 1 budget)
        n_important = max(1, int(n_keep * self.importance_ratio))

        ranked_all   = torch.argsort(token_scores, descending=True).tolist()
        important    = set(ranked_all[:n_important])
        remaining    = [i for i in ranked_all[n_important:]]  # ordered by descending score

        # --- Stage 2: diversity from remaining ---
        n_diverse = n_keep - len(important)

        if n_diverse <= 0 or not remaining:
            keep_set = important
        else:
            # Cosine similarity between remaining tokens and important tokens
            imp_list  = sorted(important)
            emb       = F.normalize(vis_embeds.float(), dim=-1)  # [n_vis, D]
            emb_imp   = emb[imp_list]                             # [n_imp, D]
            emb_rem   = emb[remaining]                            # [n_rem, D]

            # sim[i, j] = cosine(remaining[i], important[j])
            sim     = torch.matmul(emb_rem, emb_imp.T)           # [n_rem, n_imp]
            max_sim = sim.max(dim=1).values                       # [n_rem]

            # Keep tokens whose max similarity to important tokens is below threshold
            diverse = []
            for local_idx, global_idx in enumerate(remaining):
                if len(diverse) >= n_diverse:
                    break
                if max_sim[local_idx].item() < self.dup_threshold:
                    diverse.append(global_idx)

            # If not enough diverse tokens, fill with highest-scored remaining
            if len(diverse) < n_diverse:
                already = set(diverse) | important
                for i in remaining:
                    if i not in already:
                        diverse.append(i)
                    if len(diverse) >= n_diverse:
                        break

            keep_set = important | set(diverse)

        return [i in keep_set for i in range(n_vis)]
