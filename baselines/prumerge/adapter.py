"""PruMerge (LLaVA-PruMerge) baseline: outlier-detection + KNN token merging.

Reference: "LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large
            Multimodal Models" arXiv:2403.15388, ICCV 2025
           GitHub: 42Shawn/LLaVA-PruMerge

Core algorithm (three steps, all before LLM):

  Step 1 — Importance scoring via ViT penultimate-layer attention:
    score(patch_i) = mean_j softmax(Q_j K^T/√d)[j, i]  (received attention)
    Original paper uses CLIP penultimate layer [CLS]→patch attention.
    Qwen2.5-VL adaptation: use ViT block 23 (penultimate full-attention block)
    mean received attention across all token pairs and all heads.

  Step 2 — Outlier-based adaptive threshold:
    μ = mean(scores),  σ = std(scores)
    Keep token i if score(i) > μ + k_σ * σ   (outlier detection)
    This adaptively selects the "significant" tokens.
    k_σ is the sigma multiplier; default k_σ=0 (keep above-mean tokens)
    following the official implementation heuristic.

  Step 3 — KNN-based token merging:
    For each unselected token u, find its nearest selected token s* by
    key-vector cosine similarity:
        s* = argmax_{s in S} cosine(K_u, K_s)
    Merge u into s*:
        emb[s*] = emb[s*] + weight * emb[u]
        weight = attention_score(u) / sum_{u' in cluster(s*)} attention_score(u')
    This augments the kept token with information from dropped tokens.

    KEY: PruMerge MODIFIES visual token embeddings (merge operation), unlike
    other methods that only select. The output N_kept embeddings are enriched.

  Final output: N_kept modified embeddings + keep_mask.

Adaptation note:
  ViT key vectors K used in KNN are extracted from the same penultimate block
  (block 23). The merge weights are the normalized received-attention scores.
  Since Qwen2.5-VL has no [CLS], we use mean received attention as the score.
"""

import torch
import torch.nn.functional as F

_IMAGE_TOKEN_ID = 151655


class PruMergeBaseline:
    """LLaVA-PruMerge for Qwen2.5-VL: outlier detection + KNN merge.

    Args:
        model:       Qwen2VLWrapper
        vit_block_idx: ViT full-attention block for scoring (default 23 = penultimate)
        prune_ratio: fraction of tokens to DROP before any merging (max limit).
                     PruMerge itself uses adaptive threshold — prune_ratio is
                     used as a hard cap (keep at least 1-prune_ratio tokens).
        k_sigma:     outlier threshold multiplier (default 0.0 = keep above-mean)
    """

    def __init__(self, model,
                 vit_block_idx: int   = 23,
                 prune_ratio:   float = 0.5,
                 k_sigma:       float = 0.0):
        self.model        = model
        self.block_idx    = vit_block_idx
        self.prune_ratio  = prune_ratio
        self.k_sigma      = k_sigma

    # ------------------------------------------------------------------
    # ViT block attention extraction (scores + key vectors)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_vit_scores_and_keys(self, pixel_values, image_grid_thw):
        """Run ViT and capture block `block_idx` hidden states.

        Returns:
            patch_scores: [N_patches] received attention scores
            patch_keys:   [N_patches, head_dim] mean key vectors (for KNN)
        """
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

        h = captured["h"]               # [N_patches, 1280]
        N         = h.shape[0]
        num_heads = attn_m.num_heads    # 16
        head_dim  = attn_m.head_dim     # 80

        qkv = attn_m.qkv(h)
        q, k, _ = qkv.chunk(3, dim=-1)   # each [N, H * head_dim]

        q_h = q.view(N, num_heads, head_dim).permute(1, 0, 2).float()  # [H, N, D]
        k_h = k.view(N, num_heads, head_dim).permute(1, 0, 2).float()  # [H, N, D]

        scale = head_dim ** -0.5
        attn  = F.softmax(
            torch.matmul(q_h, k_h.transpose(-1, -2)) * scale, dim=-1)  # [H, N, N]

        # Received attention: mean over query dimension → [H, N] → mean over heads → [N]
        patch_scores = attn.mean(dim=1).mean(dim=0).cpu()  # [N]

        # Key vectors for KNN: mean over heads → [N, head_dim]
        patch_keys = k_h.mean(dim=0).cpu()  # [N, head_dim]

        return patch_scores, patch_keys

    # ------------------------------------------------------------------
    # Map patch-level → LLM token-level
    # ------------------------------------------------------------------

    def _patch_to_token(self, patch_tensor, n_vis, merge):
        """Average patch-level tensor within each merge group → token-level."""
        m2     = merge * merge
        n_full = min(patch_tensor.shape[0], n_vis * m2)
        n_grp  = n_full // m2
        result = patch_tensor[:n_grp * m2].view(n_grp, m2, -1).mean(dim=1) \
                 if patch_tensor.dim() == 2 else \
                 patch_tensor[:n_grp * m2].view(n_grp, m2).mean(dim=1)
        if n_grp < n_vis:
            pad_shape = (n_vis - n_grp,) + (patch_tensor.shape[1:] if patch_tensor.dim() == 2 else ())
            result = torch.cat([result, torch.zeros(*pad_shape)], dim=0)
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prune_and_merge(self, model_wrapper, inputs, vis_embeds, grid_thw,
                        prune_ratio=None):
        """PruMerge selection + embedding merge.

        Args:
            vis_embeds: [n_vis, hidden] visual token embeddings (post-merger)

        Returns:
            keep_mask:        list[bool], length = n_vis
            merged_vis_embeds: Tensor [n_kept, hidden] — enriched embeddings
                               of kept tokens (None if no merging possible)
        """
        ratio  = prune_ratio if prune_ratio is not None else self.prune_ratio
        n_vis  = vis_embeds.shape[0]
        min_keep = max(1, int(n_vis * (1.0 - ratio)))

        pixel_values   = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        # Fallback: keep by L2-norm if no pixel_values
        if pixel_values is None:
            norms = vis_embeds.float().norm(dim=-1)
            ranked = norms.argsort(descending=True).tolist()
            keep_set = set(ranked[:min_keep])
            keep_mask = [i in keep_set for i in range(n_vis)]
            kept_embeds = vis_embeds[sorted(keep_set)]
            return keep_mask, kept_embeds

        merge = self.model.model.model.visual.spatial_merge_size
        patch_scores, patch_keys = self._extract_vit_scores_and_keys(
            pixel_values, image_grid_thw)

        # Map to token level
        token_scores = self._patch_to_token(patch_scores, n_vis, merge)  # [n_vis]
        token_keys   = self._patch_to_token(patch_keys,   n_vis, merge)  # [n_vis, head_dim]

        # Step 2: Outlier-based adaptive threshold
        mu    = token_scores.mean().item()
        sigma = token_scores.std().item()
        threshold = mu + self.k_sigma * sigma

        selected_mask = (token_scores > threshold)   # [n_vis] bool
        selected_idx  = selected_mask.nonzero(as_tuple=True)[0].tolist()

        # Hard cap: ensure at least min_keep tokens
        if len(selected_idx) < min_keep:
            ranked = token_scores.argsort(descending=True).tolist()
            selected_idx = ranked[:min_keep]
            selected_mask = torch.zeros(n_vis, dtype=torch.bool)
            selected_mask[selected_idx] = True

        # Hard cap from above: never keep more than (1 - ratio) fraction
        # (PruMerge is adaptive, so it can keep more; we respect the paper's intent
        #  and only apply a minimum floor, not a ceiling, to stay faithful)

        keep_set  = set(selected_idx)
        drop_list = [i for i in range(n_vis) if i not in keep_set]

        # Step 3: KNN merge — assign each dropped token to nearest kept token
        # Use token-level key vectors (post-merger averaged) for similarity
        keys_kept = F.normalize(token_keys[selected_idx].float(), dim=-1)  # [n_kept, D]
        emb       = vis_embeds.float()                                       # [n_vis, hidden]

        # cluster[s_idx] = list of (drop_idx, score) for tokens merged into kept[s_idx]
        cluster = {s: [] for s in range(len(selected_idx))}

        if drop_list:
            keys_drop = F.normalize(token_keys[drop_list].float(), dim=-1)  # [n_drop, D]
            sim = torch.matmul(keys_drop, keys_kept.T)  # [n_drop, n_kept]
            nearest = sim.argmax(dim=1).tolist()         # [n_drop]
            for local_drop, local_kept in enumerate(nearest):
                drop_global = drop_list[local_drop]
                cluster[local_kept].append(
                    (drop_global, token_scores[drop_global].item()))

        # Build merged embeddings
        merged_list = []
        for local_s, global_s in enumerate(selected_idx):
            base_emb = emb[global_s].clone()
            if cluster[local_s]:
                members = cluster[local_s]
                score_s = token_scores[global_s].item()
                total   = score_s + sum(sc for _, sc in members)
                if total > 1e-9:
                    w_base = score_s / total
                    base_emb = base_emb * w_base
                    for drop_g, drop_sc in members:
                        base_emb = base_emb + emb[drop_g] * (drop_sc / total)
            merged_list.append(base_emb)

        merged_embeds = torch.stack(merged_list, dim=0).to(vis_embeds.dtype)

        keep_mask = [i in keep_set for i in range(n_vis)]
        return keep_mask, merged_embeds

    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw, prune_ratio=None):
        """Convenience wrapper returning only keep_mask (ignores merge)."""
        keep_mask, _ = self.prune_and_merge(
            model_wrapper, inputs, vis_embeds, grid_thw, prune_ratio)
        return keep_mask
