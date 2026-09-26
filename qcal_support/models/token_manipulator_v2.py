"""Token-level tile manipulation with actual token removal/compression.

Modifies the input sequence structure: removes skip tokens, pools coarse tokens,
and adjusts input_ids, attention_mask, mm_token_type_ids, and image_grid_thw.

GAP-aware: preserved tokens retain their ORIGINAL RoPE position IDs from the
unmodified sequence.  Removed tokens leave "gaps" in the position ID space,
so spatial / temporal relationships are identical to the full-sequence baseline.

Supports both single-image and multi-image inputs.
"""

import torch
import math
import itertools

_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653


def _compute_original_position_ids(input_ids, mm_token_type_ids, image_grid_thw,
                                    spatial_merge_size=2):
    """Replicate Qwen2VLModel.get_rope_index to compute original 3D position IDs.

    Returns:
        position_ids: [3, 1, seq_len]  (temporal, height, width)
        rope_deltas: scalar tensor
    """
    device = input_ids.device
    seq_len = input_ids.shape[0]  # already 1-D (no batch)
    mm_types = mm_token_type_ids

    # Group consecutive tokens by modality type
    input_type_group = []
    for key, group in itertools.groupby(enumerate(mm_types.tolist()), lambda x: x[1]):
        group = list(group)
        input_type_group.append((key, group[0][0], group[-1][0] + 1))

    grid_iter = iter(image_grid_thw) if image_grid_thw is not None else iter([])

    current_pos = 0
    llm_pos_ids_list = []
    for modality_type, start_idx, end_idx in input_type_group:
        if modality_type == 0:
            # Text tokens: all 3 dims get the same sequential position
            text_len = end_idx - start_idx
            llm_pos_ids_list.append(
                torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                + current_pos
            )
            current_pos += text_len
        else:
            # Vision tokens (image=1, video=2)
            grid_thw = next(grid_iter)
            llm_grid_t = grid_thw[0].item()
            llm_grid_h = grid_thw[1].item() // spatial_merge_size
            llm_grid_w = grid_thw[2].item() // spatial_merge_size
            image_seq_length = llm_grid_t * llm_grid_h * llm_grid_w

            pos_w = torch.arange(current_pos, current_pos + llm_grid_w,
                                 device=device).repeat(llm_grid_h * llm_grid_t)
            pos_h = torch.arange(current_pos, current_pos + llm_grid_h,
                                 device=device).repeat_interleave(llm_grid_w * llm_grid_t)
            pos_t = torch.full((image_seq_length,), current_pos,
                               device=device, dtype=torch.long)
            llm_pos_ids_list.append(torch.stack([pos_t, pos_h, pos_w], dim=0))
            current_pos += max(llm_grid_h, llm_grid_w)

    llm_positions = torch.cat(llm_pos_ids_list, dim=1)  # [3, seq_len]
    rope_deltas = llm_positions.max() + 1 - seq_len
    return llm_positions, rope_deltas  # [3, seq_len], scalar


class TokenManipulatorV2:
    """Manipulate visual tokens with actual sequence modification.

    Supports both square (grid_size) and rectangular (grid_h, grid_w) grids.
    """

    def __init__(self, grid_size=6, grid_h=None, grid_w=None):
        if grid_h is not None and grid_w is not None:
            self.grid_h = grid_h
            self.grid_w = grid_w
        else:
            self.grid_h = grid_size
            self.grid_w = grid_size
        # Deprecated: use grid_h/grid_w instead. Kept for legacy callers.
        self.grid_size = self.grid_h
        self.num_tiles = self.grid_h * self.grid_w

    def get_tile_token_indices(self, merged_h, merged_w, tile_id):
        """Map tile_id to token indices in the flat visual token sequence.

        Floor-based tiling (industry standard, matches adaptive_grid and
        attention_scorer):
          interior tile size = merged // grid  (floor)
          last row/col carries the remainder (edge tile >= interior tile)
        """
        tile_row = tile_id // self.grid_w
        tile_col = tile_id % self.grid_w

        th = merged_h // self.grid_h
        tw = merged_w // self.grid_w
        row_start = tile_row * th
        row_end   = merged_h if tile_row == self.grid_h - 1 else (tile_row + 1) * th
        col_start = tile_col * tw
        col_end   = merged_w if tile_col == self.grid_w - 1 else (tile_col + 1) * tw

        indices = []
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                indices.append(r * merged_w + c)
        return indices

    def apply(self, inputs, visual_embeds, grid_thw, allocation,
              spatial_merge_size=2, config=None, token_attn=None,
              coarse_mode="pool"):
        """Apply allocation by modifying the actual token sequence.

        Args:
            inputs: processor outputs dict (input_ids, attention_mask, mm_token_type_ids)
            visual_embeds: [num_visual_tokens, hidden_dim] from vision encoder
            grid_thw: [1, 3] tensor (T, H, W)
            allocation: {tile_id: "skip"|"coarse"|"lite"|"full"}
            spatial_merge_size: vision encoder merge size
            config: model config (for image_token_id)
            token_attn: optional per-token attention scores (list of floats,
                len = num_visual_tokens). If provided, used for smart
                downsampling: attention-weighted pooling (coarse) and
                attention-guided + spatial hybrid selection (lite).
            coarse_mode: "pool" (attention-weighted average, default),
                "topb" (select single token with highest attention score), or
                "unique" (select the token most different from the tile mean —
                preserves fine-grained content such as OCR text or object edges
                that get diluted by mean pooling).

        Returns:
            modified_inputs_embeds: [1, new_seq_len, hidden_dim]
            modified_attention_mask: [1, new_seq_len]
            modified_mm_token_type_ids: [1, new_seq_len]
            modified_image_grid_thw: [1, 3] or None if all skipped
            token_stats: dict with counts
        """
        input_ids = inputs["input_ids"][0]       # [seq_len]
        attn_mask = inputs["attention_mask"][0]   # [seq_len]
        mm_types = inputs["mm_token_type_ids"][0] # [seq_len]

        image_token_id = config.image_token_id if config else 151655
        t, h, w = grid_thw[0].tolist()
        merged_h = h // spatial_merge_size
        merged_w = w // spatial_merge_size
        num_visual = merged_h * merged_w

        # Find image token range in input_ids
        img_mask = (input_ids == image_token_id)
        img_positions = torch.where(img_mask)[0]
        assert len(img_positions) == num_visual, \
            f"Mismatch: {len(img_positions)} img tokens vs {num_visual} expected"

        img_start = img_positions[0].item()
        img_end = img_positions[-1].item() + 1

        # Classify each visual token
        # Build: which visual tokens to keep, pool, or drop
        keep_indices = []      # indices in visual_embeds to keep as-is
        pool_groups = []       # list of (tile_id, indices) to pool
        lite_groups = []       # list of (tile_id, indices) for lite
        skip_count = 0
        kept_count = 0

        for tile_id in range(self.num_tiles):
            tier = allocation.get(tile_id, "full")
            indices = self.get_tile_token_indices(merged_h, merged_w, tile_id)

            if tier == "full":
                keep_indices.extend(indices)
            elif tier == "lite":
                lite_groups.append((tile_id, indices))
            elif tier == "coarse":
                pool_groups.append((tile_id, indices))
            elif tier == "skip":
                skip_count += len(indices)

        # Build new visual token sequence
        new_visual_tokens = []
        new_visual_h_pos = []  # height position for each new token
        new_visual_w_pos = []  # width position for each new token
        kept_orig_indices = []  # original grid index for each new token

        # Full tokens: keep original (emit later, after SVD)
        sorted_keep = sorted(keep_indices)

        # Lite tokens: global SVD leverage-score downsampling — keep ~50% per tile
        # All full + lite tokens are stacked into one matrix for a single SVD,
        # yielding global leverage scores.  Full tiles keep all tokens regardless;
        # lite tiles select their top-50% tokens by leverage score.
        # Tiny tiles (≤2 tokens) bypass SVD.
        if lite_groups:
            # Separate tiny vs normal tiles
            tiny_lite   = [(tid, idxs) for tid, idxs in lite_groups if len(idxs) <= 2]
            normal_lite = [(tid, idxs) for tid, idxs in lite_groups if len(idxs) > 2]

            # --- Global SVD on all full + normal-lite tokens ---
            global_leverage = None
            global_flat_indices = []   # flat list of token indices in stacking order
            # full tokens first, then lite tiles
            full_svd_count = len(sorted_keep)
            global_flat_indices.extend(sorted_keep)
            lite_tile_slices = []      # (start, end) into global_flat_indices per lite tile
            if normal_lite:
                for tid, idxs in normal_lite:
                    s = len(global_flat_indices)
                    global_flat_indices.extend(idxs)
                    lite_tile_slices.append((s, len(global_flat_indices)))
            if global_flat_indices:
                try:
                    dev = visual_embeds.device
                    all_idx_t = torch.tensor(global_flat_indices, dtype=torch.long, device=dev)
                    global_mat = visual_embeds[all_idx_t].float()  # [N_total, dim]
                    U, S, _ = torch.linalg.svd(global_mat, full_matrices=False)
                    variance = S ** 2
                    total_var = variance.sum()
                    if total_var > 1e-12:
                        cumvar = variance.cumsum(dim=0) / total_var
                        k_idx = (cumvar >= 0.9).nonzero(as_tuple=True)[0]
                        k = (k_idx[0].item() + 1) if len(k_idx) > 0 else len(S)
                        k = max(1, min(k, len(S)))
                    else:
                        k = 1
                    U_k = U[:, :k]
                    global_leverage = (U_k ** 2).sum(dim=1) / k  # [N_total]
                    del global_mat, U, S, U_k
                except Exception:
                    global_leverage = None

            # Emit full tokens (all kept, leverage scores not used for selection)
            for idx in sorted_keep:
                new_visual_tokens.append(visual_embeds[idx])
                new_visual_h_pos.append(idx // merged_w)
                new_visual_w_pos.append(idx % merged_w)
                kept_orig_indices.append(idx)

            # Emit normal-lite tiles (top-50% by leverage score)
            for i, (tid, idxs) in enumerate(normal_lite):
                target_keep = max(1, len(idxs) // 2)
                kept = None
                if global_leverage is not None and lite_tile_slices:
                    s, e = lite_tile_slices[i]
                    tile_lev = global_leverage[s:e]
                    top_local = tile_lev.topk(target_keep).indices.tolist()
                    kept = sorted(idxs[j] for j in top_local)
                if kept is None:
                    # Fallback: checkerboard spatial downsampling
                    kept = [idx for idx in idxs
                            if (idx // merged_w + idx % merged_w) % 2 == 0]
                    if not kept:
                        kept = [idxs[0]]
                for idx in kept:
                    new_visual_tokens.append(visual_embeds[idx])
                    new_visual_h_pos.append(idx // merged_w)
                    new_visual_w_pos.append(idx % merged_w)
                    kept_orig_indices.append(idx)

            # Emit tiny-lite tiles (keep all)
            for tid, idxs in tiny_lite:
                for idx in idxs:
                    new_visual_tokens.append(visual_embeds[idx])
                    new_visual_h_pos.append(idx // merged_w)
                    new_visual_w_pos.append(idx % merged_w)
                    kept_orig_indices.append(idx)
        else:
            # No lite groups — just emit full tokens
            for idx in sorted_keep:
                new_visual_tokens.append(visual_embeds[idx])
                new_visual_h_pos.append(idx // merged_w)
                new_visual_w_pos.append(idx % merged_w)
                kept_orig_indices.append(idx)

        # Coarse tokens: select best token per tile based on coarse_mode.
        # "pool":   attention-weighted average (fallback: mean).
        # "topb":   pick the single token with the highest attention score (fallback: center).
        # "unique": pick the token most different from the tile mean — preserves fine-grained
        #           content (OCR text, object edges) that gets diluted by mean pooling.
        for tile_id, indices in pool_groups:
            tile_tokens = visual_embeds[torch.tensor(indices, dtype=torch.long, device=visual_embeds.device)]
            if coarse_mode == "unique":
                if len(indices) == 1:
                    selected = tile_tokens[0]
                    best_idx = indices[0]
                else:
                    tile_mean = tile_tokens.float().mean(dim=0)
                    diffs = (tile_tokens.float() - tile_mean).norm(dim=-1)  # [k]
                    best_local = int(diffs.argmax())
                    selected = tile_tokens[best_local].to(tile_tokens.dtype)
                    best_idx = indices[best_local]
            elif token_attn is not None and coarse_mode == "topb":
                attn_vals = [token_attn[i] for i in indices]
                best_local = int(max(range(len(attn_vals)), key=lambda k: attn_vals[k]))
                selected = tile_tokens[best_local]
                best_idx = indices[best_local]
            elif token_attn is not None:
                weights = torch.tensor([token_attn[i] for i in indices],
                                       device=visual_embeds.device, dtype=visual_embeds.dtype)
                weights = weights - weights.min()
                w_sum = weights.sum()
                if w_sum > 1e-8:
                    weights = weights / w_sum
                else:
                    weights = torch.ones_like(weights) / len(indices)
                selected = (tile_tokens * weights.unsqueeze(-1)).sum(dim=0)
                best_idx = indices[len(indices) // 2]
            else:
                selected = tile_tokens.mean(dim=0)
                best_idx = indices[len(indices) // 2]
            new_visual_tokens.append(selected)
            new_visual_h_pos.append(best_idx // merged_w)
            new_visual_w_pos.append(best_idx % merged_w)
            # Coarse: use negative sentinel (-(tile_id+1)) to distinguish from real indices
            kept_orig_indices.append(-(tile_id + 1))

        if not new_visual_tokens:
            # All skipped — no visual tokens
            new_num_visual = 0
            new_visual_block = torch.zeros(0, visual_embeds.shape[-1],
                                           device=visual_embeds.device,
                                           dtype=visual_embeds.dtype)
        else:
            new_visual_block = torch.stack(new_visual_tokens)
            new_num_visual = new_visual_block.shape[0]

        # Build new input_ids: [pre_img_tokens] + [image_token_id * new_num] + [post_img_tokens]
        pre_ids = input_ids[:img_start]
        post_ids = input_ids[img_end:]
        new_img_ids = torch.full((new_num_visual,), image_token_id,
                                 dtype=input_ids.dtype, device=input_ids.device)
        new_input_ids = torch.cat([pre_ids, new_img_ids, post_ids])

        # Build new mm_token_type_ids
        pre_mm = mm_types[:img_start]
        post_mm = mm_types[img_end:]
        new_img_mm = torch.ones(new_num_visual, dtype=mm_types.dtype, device=mm_types.device)
        new_mm_types = torch.cat([pre_mm, new_img_mm, post_mm])

        # Build new attention_mask
        pre_attn = attn_mask[:img_start]
        post_attn = attn_mask[img_end:]
        new_img_attn = torch.ones(new_num_visual, dtype=attn_mask.dtype, device=attn_mask.device)
        new_attn = torch.cat([pre_attn, new_img_attn, post_attn])

        # Build new image_grid_thw — we need a "fake" grid that matches new_num_visual
        # Use 1 x new_num_visual x 1 as a flat grid (multiplied by spatial_merge_size)
        if new_num_visual > 0:
            new_grid_thw = torch.tensor(
                [[1, new_num_visual * spatial_merge_size, 1 * spatial_merge_size]],
                dtype=grid_thw.dtype, device=grid_thw.device)
        else:
            new_grid_thw = None

        # ── GAP-aware position_ids ──────────────────────────────────────
        # Compute the ORIGINAL position_ids on the full unmodified sequence,
        # then index-select the positions for kept tokens.  Removed tokens
        # leave gaps so every surviving token retains its original RoPE.
        orig_pos, _ = _compute_original_position_ids(
            input_ids, mm_types, grid_thw, spatial_merge_size)
        # orig_pos: [3, orig_seq_len]

        # Build index map: for each position in the NEW sequence, which
        # position in the ORIGINAL sequence does it correspond to?
        # Pre-image text: 1:1 mapping
        orig_indices = list(range(img_start))
        # Visual tokens: map each new visual token to its original seq position
        for orig_grid_idx in kept_orig_indices:
            if orig_grid_idx >= 0:
                # Full / lite token — direct mapping
                orig_indices.append(img_start + orig_grid_idx)
            else:
                # Coarse (pooled) token — use center token of the tile
                tile_id = -(orig_grid_idx + 1)
                tile_token_indices = self.get_tile_token_indices(
                    merged_h, merged_w, tile_id)
                center = tile_token_indices[len(tile_token_indices) // 2]
                orig_indices.append(img_start + center)
        # Post-image text: 1:1 mapping to original positions
        for i in range(img_end, len(input_ids)):
            orig_indices.append(i)

        idx_tensor = torch.tensor(orig_indices, dtype=torch.long,
                                  device=input_ids.device)
        position_ids = orig_pos[:, idx_tensor].unsqueeze(1)  # [3, 1, new_seq_len]

        # rope_deltas: offset the model needs for autoregressive position IDs
        new_seq_len = new_input_ids.shape[0]
        rope_deltas = (position_ids.max().item() + 1 - new_seq_len)
        rope_deltas = torch.tensor([rope_deltas], device=input_ids.device).unsqueeze(1)

        # Count actual lite tokens kept (captured in kept_orig_indices above)
        lite_kept = sum(1 for idx in kept_orig_indices if idx >= 0
                        and any(idx in idxs for _, idxs in lite_groups))

        # Build inputs_embeds: embed text tokens + scatter visual tokens
        # We'll return all components and let the caller run generate
        token_stats = {
            "original_visual": num_visual,
            "new_visual": new_num_visual,
            "full": len(keep_indices),
            "lite": lite_kept,
            "lite_tiles": len(lite_groups),
            "coarse_tiles": len(pool_groups),
            "coarse_tokens": len(pool_groups),  # 1 token per coarse tile
            "skipped": skip_count,
            "compression_ratio": new_num_visual / num_visual if num_visual > 0 else 0,
            "new_seq_len": new_seq_len,
        }

        return {
            "new_input_ids": new_input_ids.unsqueeze(0),
            "new_attention_mask": new_attn.unsqueeze(0),
            "new_mm_token_type_ids": new_mm_types.unsqueeze(0),
            "new_image_grid_thw": new_grid_thw,
            "new_visual_embeds": new_visual_block,
            "position_ids": position_ids,
            "rope_deltas": rope_deltas,
            "token_stats": token_stats,
            "kept_orig_indices": kept_orig_indices,
        }

    # ── Multi-image support ──────────────────────────────────────────

    @staticmethod
    def _find_visual_segments(input_ids):
        """Find (start, end) of each image's visual tokens in input_ids.

        Returns list of (img_start, img_end) where input_ids[img_start:img_end]
        are all image_token_id tokens for one image.
        """
        segments = []
        ids = input_ids.tolist()
        i = 0
        while i < len(ids):
            if ids[i] == _VISION_START_TOKEN_ID:
                # Next token should be the first image_pad
                start = i + 1
                j = start
                while j < len(ids) and ids[j] != _VISION_END_TOKEN_ID:
                    j += 1
                # ids[start:j] are the image tokens for this image
                segments.append((start, j))
                i = j + 1
            else:
                i += 1
        return segments

    def _process_one_image(self, visual_embeds, merged_h, merged_w,
                           allocation, tile_offset=0, token_attn=None,
                           coarse_mode="pool"):
        """Process visual tokens for one image.

        Args:
            visual_embeds: [num_vis, hidden_dim] for this image
            merged_h, merged_w: spatial dims after merge
            allocation: {tile_id: tier} (tile_id uses global flat IDs)
            tile_offset: offset for tile_id in allocation
            token_attn: optional per-token attention scores for this image

        Returns:
            new_tokens: list of tensors
            h_pos, w_pos: spatial positions
            kept_orig_indices: original grid indices
            stats: {full, lite, coarse, skip} counts
        """
        new_tokens = []
        h_pos, w_pos = [], []
        kept_orig = []
        counts = {"full": 0, "lite": 0, "coarse": 0, "skip": 0}

        keep_indices = []
        pool_groups = []
        lite_groups = []

        for local_tid in range(self.num_tiles):
            global_tid = tile_offset + local_tid
            tier = allocation.get(global_tid, "full")
            indices = self.get_tile_token_indices(merged_h, merged_w, local_tid)

            if not indices:
                # Empty tile (grid finer than merged dims) — treat as skip
                counts["skip"] += 0  # nothing to count
                continue

            if tier == "full":
                keep_indices.extend(indices)
                counts["full"] += len(indices)
            elif tier == "lite":
                lite_groups.append((global_tid, indices))
            elif tier == "coarse":
                pool_groups.append((global_tid, indices))
            elif tier == "skip":
                counts["skip"] += len(indices)

        # Full (emit later, after SVD)
        sorted_keep = sorted(keep_indices)

        # Lite: global SVD leverage-score downsampling ~50% per tile
        # All full + normal-lite tokens are stacked for a single SVD;
        # full tiles keep all tokens, lite tiles select top-50%.
        # Tiny tiles bypass.
        if lite_groups:
            tiny_lite   = [(tid, idxs) for tid, idxs in lite_groups if len(idxs) <= 2]
            normal_lite = [(tid, idxs) for tid, idxs in lite_groups if len(idxs) > 2]

            global_leverage = None
            global_flat_indices = []
            # full tokens first, then lite tiles
            full_svd_count = len(sorted_keep)
            global_flat_indices.extend(sorted_keep)
            lite_tile_slices = []
            if normal_lite:
                for tid, idxs in normal_lite:
                    s = len(global_flat_indices)
                    global_flat_indices.extend(idxs)
                    lite_tile_slices.append((s, len(global_flat_indices)))
            if global_flat_indices:
                try:
                    dev = visual_embeds.device
                    all_idx_t = torch.tensor(global_flat_indices, dtype=torch.long, device=dev)
                    global_mat = visual_embeds[all_idx_t].float()
                    U, S, _ = torch.linalg.svd(global_mat, full_matrices=False)
                    variance = S ** 2
                    total_var = variance.sum()
                    if total_var > 1e-12:
                        cumvar = variance.cumsum(dim=0) / total_var
                        k_idx = (cumvar >= 0.9).nonzero(as_tuple=True)[0]
                        k = (k_idx[0].item() + 1) if len(k_idx) > 0 else len(S)
                        k = max(1, min(k, len(S)))
                    else:
                        k = 1
                    U_k = U[:, :k]
                    global_leverage = (U_k ** 2).sum(dim=1) / k
                    del global_mat, U, S, U_k
                except Exception:
                    global_leverage = None

            # Emit full tokens
            for idx in sorted_keep:
                new_tokens.append(visual_embeds[idx])
                h_pos.append(idx // merged_w)
                w_pos.append(idx % merged_w)
                kept_orig.append(idx)

            for i, (tid, idxs) in enumerate(normal_lite):
                target_keep = max(1, len(idxs) // 2)
                kept_ids = None
                if global_leverage is not None and lite_tile_slices:
                    s, e = lite_tile_slices[i]
                    tile_lev = global_leverage[s:e]
                    top_local = tile_lev.topk(target_keep).indices.tolist()
                    kept_ids = sorted(idxs[j] for j in top_local)
                if kept_ids is None:
                    kept_ids = [idx for idx in idxs
                                if (idx // merged_w + idx % merged_w) % 2 == 0]
                    if not kept_ids:
                        kept_ids = [idxs[0]]
                for idx in kept_ids:
                    new_tokens.append(visual_embeds[idx])
                    h_pos.append(idx // merged_w)
                    w_pos.append(idx % merged_w)
                    kept_orig.append(idx)
                counts["lite"] += len(kept_ids)

            for tid, idxs in tiny_lite:
                for idx in idxs:
                    new_tokens.append(visual_embeds[idx])
                    h_pos.append(idx // merged_w)
                    w_pos.append(idx % merged_w)
                    kept_orig.append(idx)
                counts["lite"] += len(idxs)
        else:
            # No lite groups — just emit full tokens
            for idx in sorted_keep:
                new_tokens.append(visual_embeds[idx])
                h_pos.append(idx // merged_w)
                w_pos.append(idx % merged_w)
                kept_orig.append(idx)

        # Coarse: "pool" → attention-weighted average; "topb" → argmax selection.
        for global_tid, indices in pool_groups:
            if not indices:
                continue
            tile_tokens = visual_embeds[torch.tensor(indices, dtype=torch.long, device=visual_embeds.device)]
            if token_attn is not None and coarse_mode == "topb":
                attn_vals = [token_attn[i] for i in indices]
                best_local = int(max(range(len(attn_vals)), key=lambda k: attn_vals[k]))
                selected = tile_tokens[best_local]
                best_idx = indices[best_local]
            elif token_attn is not None:
                weights = torch.tensor([token_attn[i] for i in indices],
                                       device=visual_embeds.device, dtype=visual_embeds.dtype)
                w_min = weights.min()
                weights = weights - w_min
                w_sum = weights.sum()
                if w_sum > 1e-8:
                    weights = weights / w_sum
                else:
                    weights = torch.ones_like(weights) / len(indices)
                selected = (tile_tokens * weights.unsqueeze(-1)).sum(dim=0)
                best_idx = indices[len(indices) // 2]
            else:
                selected = tile_tokens.mean(dim=0)
                best_idx = indices[len(indices) // 2]
            new_tokens.append(selected)
            h_pos.append(best_idx // merged_w)
            w_pos.append(best_idx % merged_w)
            kept_orig.append(-(global_tid + 1))
            counts["coarse"] += 1

        return new_tokens, h_pos, w_pos, kept_orig, counts

    def apply_multi(self, inputs, visual_embeds, grid_thw, allocation,
                    spatial_merge_size=2, config=None, token_attn=None,
                    coarse_mode="pool"):
        """Apply allocation with actual token removal for multi-image inputs.

        Handles multiple visual segments in input_ids, each corresponding to
        one image. The allocation uses flat tile IDs:
            tile_id = img_idx * num_tiles_per_image + local_tile_id

        Args:
            inputs: processor outputs (input_ids, attention_mask, mm_token_type_ids)
            visual_embeds: [total_visual_tokens, hidden_dim] concatenated
            grid_thw: [num_images, 3] tensor
            allocation: {flat_tile_id: tier}
            spatial_merge_size: vision encoder merge factor
            config: model config

        Returns:
            Same format as apply(), compatible with generate_with_token_manipulation_v2
        """
        input_ids = inputs["input_ids"][0]
        attn_mask = inputs["attention_mask"][0]
        mm_types = inputs["mm_token_type_ids"][0]
        image_token_id = config.image_token_id if config else 151655
        num_images = grid_thw.shape[0]

        # Find each image's visual segment in input_ids
        segments = self._find_visual_segments(input_ids)
        assert len(segments) == num_images, \
            f"Found {len(segments)} visual segments but grid_thw has {num_images} images"

        # Process each image's visual tokens
        vis_offset = 0  # offset into flat visual_embeds
        all_new_tokens = []
        all_h_pos, all_w_pos = [], []
        all_kept_orig = []
        total_counts = {"full": 0, "lite": 0, "coarse": 0, "skip": 0}
        per_image_new_counts = []  # how many new tokens per image
        original_total_vis = 0

        for img_idx in range(num_images):
            t, h, w = grid_thw[img_idx].tolist()
            mh = int(h) // spatial_merge_size
            mw = int(w) // spatial_merge_size
            n_vis = mh * mw
            original_total_vis += n_vis

            seg_start, seg_end = segments[img_idx]
            seg_len = seg_end - seg_start
            assert seg_len == n_vis, \
                f"Image {img_idx}: segment has {seg_len} tokens, expected {n_vis}"

            img_vis = visual_embeds[vis_offset:vis_offset + n_vis]
            tile_offset = img_idx * self.num_tiles

            # Slice token_attn for this image
            img_token_attn = None
            if token_attn is not None:
                img_token_attn = token_attn[vis_offset:vis_offset + n_vis]

            tokens, hp, wp, kept, counts = self._process_one_image(
                img_vis, mh, mw, allocation, tile_offset, token_attn=img_token_attn,
                coarse_mode=coarse_mode)

            # Adjust h/w positions with per-image offset to avoid collisions
            # Each image's spatial positions are relative to its own grid
            all_new_tokens.extend(tokens)
            all_h_pos.extend(hp)
            all_w_pos.extend(wp)
            all_kept_orig.extend(kept)
            per_image_new_counts.append(len(tokens))
            for k in total_counts:
                total_counts[k] += counts[k]

            vis_offset += n_vis

        new_num_visual = len(all_new_tokens)

        if new_num_visual > 0:
            new_visual_block = torch.stack(all_new_tokens)
        else:
            new_visual_block = torch.zeros(0, visual_embeds.shape[-1],
                                           device=visual_embeds.device,
                                           dtype=visual_embeds.dtype)

        # ── Rebuild input_ids, attn_mask, mm_types ──
        # Walk through the original sequence, replacing each visual segment
        new_ids_parts = []
        new_attn_parts = []
        new_mm_parts = []
        prev_end = 0
        img_tok_offset = 0

        for img_idx, (seg_start, seg_end) in enumerate(segments):
            # Text before this segment (including <vision_start>)
            # seg_start points to first image token (after vision_start)
            # We include the vision_start token
            vs_pos = seg_start - 1  # <vision_start> position
            new_ids_parts.append(input_ids[prev_end:vs_pos + 1])
            new_attn_parts.append(attn_mask[prev_end:vs_pos + 1])
            new_mm_parts.append(mm_types[prev_end:vs_pos + 1])

            # New image tokens for this image
            n_new = per_image_new_counts[img_idx]
            new_ids_parts.append(torch.full(
                (n_new,), image_token_id,
                dtype=input_ids.dtype, device=input_ids.device))
            new_attn_parts.append(torch.ones(
                n_new, dtype=attn_mask.dtype, device=attn_mask.device))
            new_mm_parts.append(torch.ones(
                n_new, dtype=mm_types.dtype, device=mm_types.device))

            # <vision_end> token
            ve_pos = seg_end  # <vision_end> position
            prev_end = ve_pos  # will pick up from <vision_end> onward

        # Remaining text after last segment
        new_ids_parts.append(input_ids[prev_end:])
        new_attn_parts.append(attn_mask[prev_end:])
        new_mm_parts.append(mm_types[prev_end:])

        new_input_ids = torch.cat(new_ids_parts)
        new_attn = torch.cat(new_attn_parts)
        new_mm_types = torch.cat(new_mm_parts)

        # ── GAP-aware position_ids (multi-image) ─────────────────────────
        # Compute ORIGINAL position_ids on the full unmodified sequence,
        # then index-select for kept tokens.  Same principle as single-image.
        orig_pos, _ = _compute_original_position_ids(
            input_ids, mm_types, grid_thw, spatial_merge_size)
        # orig_pos: [3, orig_seq_len]

        # Build index map: new-sequence position → original-sequence position
        orig_indices = []
        vis_token_cursor = 0  # cursor into all_kept_orig

        prev_end = 0
        for img_idx, (seg_start, seg_end) in enumerate(segments):
            # Text + <vision_start> before this segment: 1:1 mapping
            vs_pos = seg_start - 1  # <vision_start> position
            for i in range(prev_end, vs_pos + 1):
                orig_indices.append(i)

            # Visual tokens for this image
            n_new = per_image_new_counts[img_idx]
            # Original visual token range for this image
            t_i, h_i, w_i = grid_thw[img_idx].tolist()
            mh_i = int(h_i) // spatial_merge_size
            mw_i = int(w_i) // spatial_merge_size

            for j in range(n_new):
                orig_grid_idx = all_kept_orig[vis_token_cursor + j]
                if orig_grid_idx >= 0:
                    # Full / lite token — direct mapping to original seq position
                    orig_indices.append(seg_start + orig_grid_idx)
                else:
                    # Coarse (pooled) token — use center token of the tile
                    global_tid = -(orig_grid_idx + 1)
                    local_tid = global_tid - img_idx * self.num_tiles
                    tile_token_indices = self.get_tile_token_indices(
                        mh_i, mw_i, local_tid)
                    center = tile_token_indices[len(tile_token_indices) // 2]
                    orig_indices.append(seg_start + center)
            vis_token_cursor += n_new

            prev_end = seg_end  # points to <vision_end>

        # Remaining text after last segment
        for i in range(prev_end, len(input_ids)):
            orig_indices.append(i)

        idx_tensor = torch.tensor(orig_indices, dtype=torch.long,
                                  device=input_ids.device)
        position_ids = orig_pos[:, idx_tensor].unsqueeze(1)  # [3, 1, new_seq_len]

        # rope_deltas: offset the model needs for autoregressive position IDs
        new_seq_len = new_input_ids.shape[0]
        rope_deltas = (position_ids.max().item() + 1 - new_seq_len)
        rope_deltas = torch.tensor([rope_deltas], device=input_ids.device).unsqueeze(1)

        # Stats
        token_stats = {
            "original_visual": original_total_vis,
            "new_visual": new_num_visual,
            "full": total_counts["full"],
            "lite": total_counts["lite"],
            "coarse_tiles": total_counts["coarse"],
            "coarse_tokens": total_counts["coarse"],
            "skipped": total_counts["skip"],
            "compression_ratio": new_num_visual / original_total_vis if original_total_vis > 0 else 0,
            "new_seq_len": new_seq_len,
            "per_image_new_counts": per_image_new_counts,
        }

        return {
            "new_input_ids": new_input_ids.unsqueeze(0),
            "new_attention_mask": new_attn.unsqueeze(0),
            "new_mm_token_type_ids": new_mm_types.unsqueeze(0),
            "new_image_grid_thw": grid_thw,  # keep original for M-RoPE
            "new_visual_embeds": new_visual_block,
            "position_ids": position_ids,
            "rope_deltas": rope_deltas,
            "token_stats": token_stats,
            "kept_orig_indices": all_kept_orig,
        }

    # ── Token-level mask (for HAWK, SVD-Prune, etc.) ────────────────

    def apply_token_mask(self, inputs, visual_embeds, grid_thw, keep_mask,
                         spatial_merge_size=2, config=None):
        """Apply a flat token-level keep/drop mask to visual tokens.

        Works for both single-image and multi-image inputs.
        No tile grid needed — operates directly on individual tokens.
        GAP-aware: preserved tokens retain original RoPE position IDs.

        Args:
            inputs: processor outputs (input_ids, attention_mask, mm_token_type_ids)
            visual_embeds: [total_visual_tokens, hidden_dim]
            grid_thw: [num_images, 3] tensor
            keep_mask: list of bool, length = total_visual_tokens.
                       True = keep, False = drop.
            spatial_merge_size: vision encoder merge factor
            config: model config

        Returns:
            Same format as apply() / apply_multi(), compatible with
            generate_with_token_manipulation_v2.
        """
        input_ids = inputs["input_ids"][0]
        attn_mask = inputs["attention_mask"][0]
        mm_types = inputs["mm_token_type_ids"][0]
        image_token_id = config.image_token_id if config else 151655
        num_images = grid_thw.shape[0]

        segments = self._find_visual_segments(input_ids)
        assert len(segments) == num_images

        # Collect kept tokens per image
        vis_offset = 0
        all_new_tokens = []
        all_kept_orig = []
        per_image_new_counts = []
        original_total_vis = 0
        kept_total = 0
        skipped_total = 0

        for img_idx in range(num_images):
            t, h, w = grid_thw[img_idx].tolist()
            mh = int(h) // spatial_merge_size
            mw = int(w) // spatial_merge_size
            n_vis = mh * mw
            original_total_vis += n_vis

            seg_start, seg_end = segments[img_idx]
            img_vis = visual_embeds[vis_offset:vis_offset + n_vis]
            img_mask = keep_mask[vis_offset:vis_offset + n_vis]

            img_new = []
            img_kept = []
            for j in range(n_vis):
                if img_mask[j]:
                    img_new.append(img_vis[j])
                    img_kept.append(j)
                    kept_total += 1
                else:
                    skipped_total += 1

            all_new_tokens.extend(img_new)
            all_kept_orig.extend(img_kept)
            per_image_new_counts.append(len(img_new))
            vis_offset += n_vis

        new_num_visual = len(all_new_tokens)
        if new_num_visual > 0:
            new_visual_block = torch.stack(all_new_tokens)
        else:
            new_visual_block = torch.zeros(0, visual_embeds.shape[-1],
                                           device=visual_embeds.device,
                                           dtype=visual_embeds.dtype)

        # Rebuild input_ids, attn_mask, mm_types
        new_ids_parts, new_attn_parts, new_mm_parts = [], [], []
        prev_end = 0

        for img_idx, (seg_start, seg_end) in enumerate(segments):
            vs_pos = seg_start - 1  # <vision_start>
            new_ids_parts.append(input_ids[prev_end:vs_pos + 1])
            new_attn_parts.append(attn_mask[prev_end:vs_pos + 1])
            new_mm_parts.append(mm_types[prev_end:vs_pos + 1])

            n_new = per_image_new_counts[img_idx]
            new_ids_parts.append(torch.full(
                (n_new,), image_token_id,
                dtype=input_ids.dtype, device=input_ids.device))
            new_attn_parts.append(torch.ones(
                n_new, dtype=attn_mask.dtype, device=attn_mask.device))
            new_mm_parts.append(torch.ones(
                n_new, dtype=mm_types.dtype, device=mm_types.device))

            prev_end = seg_end

        new_ids_parts.append(input_ids[prev_end:])
        new_attn_parts.append(attn_mask[prev_end:])
        new_mm_parts.append(mm_types[prev_end:])

        new_input_ids = torch.cat(new_ids_parts)
        new_attn = torch.cat(new_attn_parts)
        new_mm_types = torch.cat(new_mm_parts)

        # GAP-aware position_ids
        orig_pos, _ = _compute_original_position_ids(
            input_ids, mm_types, grid_thw, spatial_merge_size)

        orig_indices = []
        vis_cursor = 0
        prev_end = 0
        for img_idx, (seg_start, seg_end) in enumerate(segments):
            vs_pos = seg_start - 1
            for i in range(prev_end, vs_pos + 1):
                orig_indices.append(i)
            n_new = per_image_new_counts[img_idx]
            for j in range(n_new):
                orig_grid_idx = all_kept_orig[vis_cursor + j]
                orig_indices.append(seg_start + orig_grid_idx)
            vis_cursor += n_new
            prev_end = seg_end
        for i in range(prev_end, len(input_ids)):
            orig_indices.append(i)

        idx_tensor = torch.tensor(orig_indices, dtype=torch.long,
                                  device=input_ids.device)
        position_ids = orig_pos[:, idx_tensor].unsqueeze(1)

        new_seq_len = new_input_ids.shape[0]
        rope_deltas = (position_ids.max().item() + 1 - new_seq_len)
        rope_deltas = torch.tensor([rope_deltas], device=input_ids.device).unsqueeze(1)

        token_stats = {
            "original_visual": original_total_vis,
            "new_visual": new_num_visual,
            "full": kept_total,
            "lite": 0, "coarse_tiles": 0, "coarse_tokens": 0,
            "skipped": skipped_total,
            "compression_ratio": new_num_visual / original_total_vis if original_total_vis > 0 else 0,
            "new_seq_len": new_seq_len,
            "per_image_new_counts": per_image_new_counts,
        }

        return {
            "new_input_ids": new_input_ids.unsqueeze(0),
            "new_attention_mask": new_attn.unsqueeze(0),
            "new_mm_token_type_ids": new_mm_types.unsqueeze(0),
            "new_image_grid_thw": grid_thw,
            "new_visual_embeds": new_visual_block,
            "position_ids": position_ids,
            "rope_deltas": rope_deltas,
            "token_stats": token_stats,
            "kept_orig_indices": all_kept_orig,
        }
