"""Pyramid Token Pruning (PTP) baseline: region + token + instruction-guided pruning.

Reference: "Training-Free Pyramid Token Pruning for Efficient Large
            Vision-Language Models via Region, Token, and Instruction-Guided
            Importance"
           arXiv:2509.15704, 2025

Core algorithm (3-stage coarse-to-fine):

  Stage 1 — Region-level saliency allocation:
    For dynamic-resolution models that split images into tiles/sub-images,
    compute saliency per tile from ViT attention (average received attention
    within each tile). Allocate token budget proportionally to tile saliency.

    Qwen2.5-VL adaptation: image_grid_thw encodes tile structure.
    We compute per-tile average ViT attention and allocate budget proportionally.
    For single-tile images, this stage is a no-op.

  Stage 2 — Token-level selection (bottom-up):
    Within each tile, rank tokens by ViT received attention and keep top-k
    according to tile budget from Stage 1.

  Stage 3 — Instruction-aware fusion (top-down, paper §IV-A):
    Compute instruction-guided importance c_j = max_i attn(text_i → vis_j)
    at LLM layer 2, averaged over heads.
    Fuse with bottom-up score: final = (1-α)·b_norm + α·c, α=0.5 (Table III).
    Re-select top-k by fused score.

Applied to Qwen2.5-VL: ViT block 31 for saliency, LLM layer 2 for refinement.
One-shot, training-free.
"""

import torch
import torch.nn.functional as F

_IMAGE_TOKEN_ID  = 151655
_VISION_START_ID = 151652
_VISION_END_ID   = 151653


class PTPBaseline:
    """Pyramid Token Pruning for Qwen2.5-VL.

    Args:
        model:          Qwen2VLWrapper
        vit_block_idx:  ViT block for attention scoring (default 31)
        refine_layer:   LLM layer for instruction-aware scoring (default 2)
        alpha:          fusion weight for instruction score (default 0.5, paper Table III)
        prune_ratio:    fraction of tokens to DROP (default 0.5)
    """

    def __init__(self, model,
                 vit_block_idx: int   = 31,
                 refine_layer:  int   = 2,
                 alpha:         float = 0.5,
                 prune_ratio:   float = 0.5):
        self.model         = model
        self.block_idx     = vit_block_idx
        self.refine_layer  = refine_layer
        self.alpha         = alpha
        self.prune_ratio   = prune_ratio

    # ------------------------------------------------------------------
    # ViT received attention (patch-level)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _vit_received_attention(self, pixel_values, image_grid_thw):
        vit   = self.model.model.model.visual
        block = vit.blocks[self.block_idx]
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
        N, D = h.shape
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
    # LLM text→visual attention for refinement
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _instruction_attention(self, model_wrapper, inputs, vis_pos, text_pos):
        """Compute instruction→visual attention with layernorm, RoPE, and
        full-sequence softmax (paper §IV-A: softmax over ALL keys, then
        slice visual columns)."""
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_multimodal_rotary_pos_emb)

        lm    = model_wrapper.model.model.language_model
        layer = lm.layers[self.refine_layer]
        attn_l = layer.self_attn

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
            return [0.0] * len(vis_pos)

        n_heads    = attn_l.num_heads
        n_kv_heads = attn_l.num_key_value_heads
        head_dim   = attn_l.head_dim
        device = next(attn_l.q_proj.parameters()).device
        h = h.to(device)

        # S1: Apply input_layernorm
        h_normed = layer.input_layernorm(h)

        t_idx = torch.tensor(text_pos, dtype=torch.long, device=device)
        v_idx = torch.tensor(vis_pos,  dtype=torch.long, device=device)
        # Full sequence indices for keys (softmax over ALL positions)
        seq_len = h.shape[1]
        all_idx = torch.arange(seq_len, dtype=torch.long, device=device)

        q = attn_l.q_proj(h_normed[:, t_idx, :]).view(
            1, len(text_pos), n_heads, head_dim).transpose(1, 2)
        k_all = attn_l.k_proj(h_normed[:, all_idx, :]).view(
            1, seq_len, n_kv_heads, head_dim).transpose(1, 2)
        if n_heads != n_kv_heads:
            k_all = k_all.repeat_interleave(n_heads // n_kv_heads, dim=1)

        # S2: Apply M-RoPE
        position_ids = inputs.get("position_ids")
        pos_emb = None
        mrope_section = getattr(lm.config, "rope_parameters", {}).get(
            "mrope_section", None)
        if position_ids is not None and mrope_section is not None:
            if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                position_ids_rope = position_ids[1:]
            elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
                position_ids_rope = position_ids
            else:
                position_ids_rope = None
            if position_ids_rope is not None:
                dummy = torch.zeros(1, seq_len, 1, device=device,
                                    dtype=h.dtype)
                cos, sin = lm.rotary_emb(dummy,
                                         position_ids_rope.to(device))
                cos_q = cos[:, :, t_idx, :]
                sin_q = sin[:, :, t_idx, :]
                q, _ = apply_multimodal_rotary_pos_emb(
                    q, q, cos_q, sin_q, mrope_section)
                k_all, _ = apply_multimodal_rotary_pos_emb(
                    k_all, k_all, cos, sin, mrope_section)

        # Softmax over ALL keys, then extract visual columns
        attn = F.softmax(
            torch.matmul(q, k_all.transpose(-2, -1)) * (head_dim ** -0.5),
            dim=-1)
        # Slice only visual columns
        attn_vis = attn[:, :, :, v_idx]  # [1, n_heads, n_text, n_vis]
        # Paper: c_j = max_i attn(text_i → vis_j), averaged over heads
        scores = attn_vis[0].mean(0).max(0).values
        return scores.cpu().tolist()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw,
              prune_ratio=None):
        """PTP 3-stage token selection. Returns keep_mask list[bool]."""
        ratio = prune_ratio if prune_ratio is not None else self.prune_ratio
        n_vis = vis_embeds.shape[0]
        n_keep = max(1, int(n_vis * (1.0 - ratio)))

        pixel_values   = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        # --- Stage 1 & 2: ViT attention → per-tile budget allocation ---
        if pixel_values is None:
            norms = vis_embeds.float().norm(dim=-1)
            keep_set = set(norms.argsort(descending=True)[:n_keep].tolist())
            return [i in keep_set for i in range(n_vis)]

        patch_scores = self._vit_received_attention(pixel_values, image_grid_thw)
        merge = self.model.model.model.visual.spatial_merge_size
        m2 = merge * merge
        N_vis_from_patches = patch_scores.shape[0] // m2
        token_scores = patch_scores[:N_vis_from_patches * m2].view(
            N_vis_from_patches, m2).mean(dim=1)

        # Align
        if N_vis_from_patches != n_vis:
            padded = torch.zeros(n_vis)
            m = min(N_vis_from_patches, n_vis)
            padded[:m] = token_scores[:m]
            token_scores = padded

        # Per-tile budget allocation
        if image_grid_thw is not None and image_grid_thw.shape[0] > 1:
            # Multiple tiles: allocate budget proportionally to tile saliency
            tile_sizes = []
            offset = 0
            for t in range(image_grid_thw.shape[0]):
                t_val, h_val, w_val = image_grid_thw[t].tolist()
                tile_n = int(t_val * h_val * w_val) // m2
                tile_sizes.append((offset, offset + tile_n))
                offset += tile_n

            # Per-tile mean saliency
            tile_saliency = []
            for start, end in tile_sizes:
                if end > start and end <= n_vis:
                    tile_saliency.append(token_scores[start:end].mean().item())
                else:
                    tile_saliency.append(0.0)

            total_sal = sum(tile_saliency) + 1e-9
            tile_budgets = [max(1, int(n_keep * (s / total_sal)))
                           for s in tile_saliency]
            # Adjust to sum to n_keep
            diff = n_keep - sum(tile_budgets)
            if diff > 0:
                # Add to highest-saliency tiles
                sorted_tiles = sorted(range(len(tile_saliency)),
                                     key=lambda i: tile_saliency[i], reverse=True)
                for i in range(diff):
                    tile_budgets[sorted_tiles[i % len(sorted_tiles)]] += 1

            # Select top-k within each tile
            selected = set()
            for tile_idx, (start, end) in enumerate(tile_sizes):
                if end > n_vis:
                    end = n_vis
                tile_tokens = list(range(start, end))
                tile_sc = [(i, token_scores[i].item()) for i in tile_tokens]
                tile_sc.sort(key=lambda x: x[1], reverse=True)
                budget = min(tile_budgets[tile_idx], len(tile_sc))
                for i in range(budget):
                    selected.add(tile_sc[i][0])
        else:
            # Single tile: simple top-k
            ranked = token_scores.argsort(descending=True).tolist()
            selected = set(ranked[:n_keep])

        # --- Stage 3: Score fusion — (1-α)·b_norm + α·c (paper §IV-A) ---
        ids = inputs["input_ids"][0]
        vs = (ids == _VISION_START_ID).nonzero(as_tuple=True)[0]
        ve = (ids == _VISION_END_ID).nonzero(as_tuple=True)[0]
        if len(vs) > 0 and len(ve) > 0:
            vs_val, ve_val = vs[0].item(), ve[0].item()
            vis_pos  = [i for i in range(vs_val + 1, ve_val) if ids[i] == _IMAGE_TOKEN_ID]
            text_pos = [i for i in range(ids.shape[0]) if i < vs_val or i > ve_val]
        else:
            vis_pos, text_pos = [], []

        if vis_pos and text_pos and self.alpha > 0:
            instr_scores = self._instruction_attention(
                model_wrapper, inputs, vis_pos, text_pos)
            c = torch.tensor(instr_scores, dtype=torch.float32)

            # Normalize bottom-up scores to [0, 1]
            b = token_scores.clone().float()
            b_min, b_max = b.min(), b.max()
            if b_max - b_min > 1e-9:
                b = (b - b_min) / (b_max - b_min)
            else:
                b = torch.zeros_like(b)

            # Normalize instruction score to [0, 1] (10c fix: scale match)
            c_min, c_max = c.min(), c.max()
            if c_max - c_min > 1e-9:
                c = (c - c_min) / (c_max - c_min)
            else:
                c = torch.zeros_like(c)

            # Fused score
            fused = (1.0 - self.alpha) * b + self.alpha * c

            # Re-select top-k by fused score, respecting per-tile budgets
            if image_grid_thw is not None and image_grid_thw.shape[0] > 1:
                selected = set()
                for tile_idx, (start, end) in enumerate(tile_sizes):
                    if end > n_vis:
                        end = n_vis
                    tile_tokens = list(range(start, end))
                    tile_sc = [(i, fused[i].item()) for i in tile_tokens]
                    tile_sc.sort(key=lambda x: x[1], reverse=True)
                    budget = min(tile_budgets[tile_idx], len(tile_sc))
                    for i in range(budget):
                        selected.add(tile_sc[i][0])
            else:
                ranked_fused = fused.argsort(descending=True).tolist()
                selected = set(ranked_fused[:n_keep])

        # Ensure n_keep
        if len(selected) < n_keep:
            ranked = token_scores.argsort(descending=True).tolist()
            for idx in ranked:
                if idx not in selected:
                    selected.add(idx)
                if len(selected) >= n_keep:
                    break
        elif len(selected) > n_keep:
            sel_scores = [(i, token_scores[i].item()) for i in selected]
            sel_scores.sort(key=lambda x: x[1], reverse=True)
            selected = set(x[0] for x in sel_scores[:n_keep])

        return [i in selected for i in range(n_vis)]
