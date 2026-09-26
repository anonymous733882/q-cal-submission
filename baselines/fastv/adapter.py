"""FastV baseline: single-layer attention-based visual token pruning.

Reference: Chen et al., "An Image is Worth 1/2 Tokens After Layer 2:
Plug-and-Play Inference Acceleration for Large Vision-Language Models" (2024)

Algorithm:
  1. Run prefill through layer K-1, capture attention weights
  2. At layer K, rank visual tokens by last-text-token → visual attention
  3. Keep top (1-R) fraction, drop the rest
  4. Continue forward pass with pruned sequence

Adapted for Qwen2-VL: uses hook-based scoring (no model surgery).
Outputs tile-level scores compatible with BEA's allocation pipeline.
"""

import torch
import torch.nn.functional as F


_IMAGE_TOKEN_ID = 151655
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653


class FastVBaseline:
    """FastV-style scorer adapted for Qwen2-VL.

    Hooks layer[K-1] to get attention, uses last text token's attention
    over visual tokens as importance score. Produces tile-level scores.

    Args:
        model: Qwen2VLWrapper
        layer_idx: K in the paper (default 2, meaning hook layer 1, prune at layer 2)
        prune_ratio: R, fraction of visual tokens to DROP (default 0.5)
    """

    def __init__(self, model, layer_idx=3, prune_ratio=0.5):
        self.model = model
        self.layer_idx = layer_idx
        self.prune_ratio = prune_ratio
        lm = model.model.model.language_model
        cfg = lm.config
        self.num_heads = getattr(cfg, "num_attention_heads", 28)
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", 4)
        self.head_dim = getattr(cfg, "hidden_size", 3584) // self.num_heads

    @torch.no_grad()
    def score(self, inputs, grid_thw, grid_size=6):
        """Score visual tokens using FastV attention from layer K-1.

        Returns:
            tile_scores: list of dicts {tile_id, b, r, n} compatible with BEA
        """
        lm = self.model.model.model.language_model
        hook_layer = lm.layers[self.layer_idx - 1]
        attn_module = hook_layer.self_attn

        captured = {}

        def _hook(module, args, output):
            # For Qwen2, self_attn returns (attn_output, attn_weights, past_kv)
            # We need to get attention weights. If not available via output,
            # we compute Q*K manually from the hidden states.
            captured["hidden_states"] = args[0].detach() if isinstance(args, tuple) else args.detach()

        handle = hook_layer.register_forward_hook(_hook, with_kwargs=False)

        try:
            # Run full forward to capture hidden states at layer K-1 input
            # We actually need the hidden states ENTERING layer K-1
            # So hook the layer before that
            if self.layer_idx >= 2:
                pre_hook_layer = lm.layers[self.layer_idx - 2]
                captured2 = {}

                def _pre_hook(module, input, output):
                    captured2["output"] = output[0].detach()

                handle2 = pre_hook_layer.register_forward_hook(_pre_hook)
            else:
                handle2 = None

            _ = self.model.model(
                **inputs,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
        finally:
            handle.remove()
            if handle2:
                handle2.remove()

        # Use hidden states entering layer K-1 to compute Q*K attention
        if self.layer_idx >= 2 and "output" in captured2:
            hidden = captured2["output"]
        elif "hidden_states" in captured:
            hidden = captured["hidden_states"]
        else:
            raise RuntimeError("Hook did not fire")

        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        # Compute Q*K^T at layer K-1 — memory-efficient version
        # FastV only needs last-text-token → visual attention,
        # so we compute Q for last text token, K for visual tokens only.
        from baselines.common.attn_utils import prepare_rope, apply_rope_to_qk

        attn_dev = next(attn_module.q_proj.parameters()).device
        hidden = hidden.to(attn_dev)

        # Locate visual and text tokens first
        ids = inputs["input_ids"][0].to(attn_dev)
        vis_segments = _visual_segments(ids)
        text_mask = _text_mask(ids)
        text_positions = torch.where(text_mask)[0]

        seq_len = hidden.shape[1]
        if len(text_positions) > 0:
            last_text_pos = text_positions[-1].item()
        else:
            last_text_pos = seq_len - 1

        # Build visual mask
        vis_mask = torch.zeros(seq_len, dtype=torch.bool, device=attn_dev)
        for vs, ve in vis_segments:
            vis_mask[vs:ve] = True
        vis_positions = torch.where(vis_mask)[0]

        if len(vis_positions) == 0:
            # No visual tokens — return uniform scores
            merge = self.model.model.model.visual.spatial_merge_size
            tile_scores = []
            for img_idx, (vs, ve) in enumerate(vis_segments):
                for tid in range(grid_size * grid_size):
                    tile_scores.append({"tile_id": img_idx * grid_size * grid_size + tid,
                                        "b": 0.5, "r": 0.0, "n": 0.5})
            return tile_scores

        # S1: Apply input_layernorm
        lm = self.model.model.model.language_model
        hook_layer_module = lm.layers[self.layer_idx - 1]
        h_normed = hook_layer_module.input_layernorm(hidden)

        # Q for last text token only: [1, 1, hidden_dim] → [1, num_heads, 1, head_dim]
        h_query = h_normed[:, last_text_pos:last_text_pos+1, :]
        q = attn_module.q_proj(h_query)
        q = q.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # K for visual tokens only: [1, num_vis, hidden_dim]
        h_vis = h_normed[:, vis_positions, :]
        k = attn_module.k_proj(h_vis)
        num_vis_total = len(vis_positions)
        k = k.view(1, num_vis_total, self.num_kv_heads, self.head_dim).transpose(1, 2)
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)

        # S2: Apply RoPE
        cos, sin, mrope_section = prepare_rope(lm, inputs, attn_dev)
        if cos is not None:
            q_pos = torch.tensor([last_text_pos], dtype=torch.long, device=attn_dev)
            q, k = apply_rope_to_qk(q, k, cos, sin, mrope_section,
                                     q_pos, vis_positions)

        scale = self.head_dim ** -0.5
        # [1, num_heads, 1, num_vis_total]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores.float(), dim=-1)
        # [num_vis_total] — last text token's attention over all visual tokens
        vis_attn_all = attn_weights[0].mean(dim=0).squeeze(0)

        del q, k, attn_scores, attn_weights

        merge = self.model.model.model.visual.spatial_merge_size
        tile_scores = []
        vis_offset = 0

        for img_idx, (vis_start, vis_end) in enumerate(vis_segments):
            thw = grid_thw[img_idx]
            merged_h = thw[1].item() // merge
            merged_w = thw[2].item() // merge
            num_vis = vis_end - vis_start

            vis_attn = vis_attn_all[vis_offset:vis_offset + num_vis]

            # Aggregate to tiles
            for tid in range(grid_size * grid_size):
                indices = _get_tile_indices(merged_h, merged_w, grid_size, tid)
                idx_t = torch.tensor(indices, device=attn_dev)
                idx_t = idx_t[idx_t < num_vis]
                if len(idx_t) > 0:
                    score = vis_attn[idx_t].mean().item()
                else:
                    score = 0.0
                tile_scores.append({
                    "tile_id": img_idx * grid_size * grid_size + tid,
                    "b": score, "r": 0.0, "n": 0.5,
                })

            vis_offset += num_vis

        # Normalize b scores to [0, 1]
        b_vals = [s["b"] for s in tile_scores]
        mn, mx = min(b_vals), max(b_vals)
        if mx - mn > 1e-8:
            for s in tile_scores:
                s["b"] = (s["b"] - mn) / (mx - mn)

        return tile_scores

    @torch.no_grad()
    def score_tokens(self, inputs):
        """Score per visual token using FastV layer-2 attention.

        Returns:
            token_importance: list of floats, length = num_visual_tokens
                              (only includes tokens inside vision segments)
        """
        lm = self.model.model.model.language_model
        captured2 = {}
        if self.layer_idx >= 2:
            pre_hook_layer = lm.layers[self.layer_idx - 2]
            def _pre_hook(module, input, output):
                captured2["output"] = output[0].detach()
            handle2 = pre_hook_layer.register_forward_hook(_pre_hook)
        else:
            handle2 = None

        try:
            _ = self.model.model(
                **inputs,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
        finally:
            if handle2:
                handle2.remove()

        if "output" not in captured2:
            raise RuntimeError("FastV hook did not fire")

        hidden = captured2["output"]
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        from baselines.common.attn_utils import prepare_rope, apply_rope_to_qk

        hook_layer_module = lm.layers[self.layer_idx - 1]
        attn_module = hook_layer_module.self_attn
        attn_dev = next(attn_module.q_proj.parameters()).device
        hidden = hidden.to(attn_dev)

        ids = inputs["input_ids"][0].to(attn_dev)
        text_mask = _text_mask(ids)
        text_positions = torch.where(text_mask)[0]
        last_text_pos = text_positions[-1].item() if len(text_positions) > 0 else hidden.shape[1] - 1

        vis_mask = ~text_mask
        vis_positions = torch.where(vis_mask)[0]

        if len(vis_positions) == 0:
            return []

        # S1: Apply input_layernorm
        h_normed = hook_layer_module.input_layernorm(hidden)

        h_query = h_normed[:, last_text_pos:last_text_pos+1, :]
        q = attn_module.q_proj(h_query)
        q = q.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)

        h_vis = h_normed[:, vis_positions, :]
        k = attn_module.k_proj(h_vis)
        num_vis = len(vis_positions)
        k = k.view(1, num_vis, self.num_kv_heads, self.head_dim).transpose(1, 2)
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)

        # S2: Apply RoPE
        cos, sin, mrope_section = prepare_rope(lm, inputs, attn_dev)
        if cos is not None:
            q_pos = torch.tensor([last_text_pos], dtype=torch.long, device=attn_dev)
            q, k = apply_rope_to_qk(q, k, cos, sin, mrope_section,
                                     q_pos, vis_positions)

        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.nn.functional.softmax(attn_scores.float(), dim=-1)
        # [num_vis] — averaged over heads
        importance = attn_weights[0].mean(dim=0).squeeze(0)

        return importance.cpu().tolist()

    def allocate(self, tile_scores, total_tiles=None):
        """FastV allocation: top (1-R) → full, rest → skip.

        This is the pure FastV strategy: binary keep/drop.
        """
        if total_tiles is None:
            total_tiles = len(tile_scores)
        keep_count = max(1, int(total_tiles * (1.0 - self.prune_ratio)))

        ranked = sorted(tile_scores, key=lambda s: s["b"], reverse=True)
        alloc = {}
        for i, s in enumerate(ranked):
            alloc[s["tile_id"]] = "full" if i < keep_count else "skip"
        return alloc


def _text_mask(ids):
    return ~((ids == _IMAGE_TOKEN_ID) | (ids == _VISION_START_TOKEN_ID) | (ids == _VISION_END_TOKEN_ID))


def _visual_segments(ids):
    segments = []
    in_seg = False
    start = 0
    for i, tok in enumerate(ids.tolist()):
        if tok == _VISION_START_TOKEN_ID:
            in_seg = True
            start = i + 1
        elif tok == _VISION_END_TOKEN_ID and in_seg:
            segments.append((start, i))
            in_seg = False
    return segments


def _get_tile_indices(merged_h, merged_w, grid_size, tile_id):
    tile_row = tile_id // grid_size
    tile_col = tile_id % grid_size
    tph = merged_h / grid_size
    tpw = merged_w / grid_size
    row_s, row_e = int(tile_row * tph), int((tile_row + 1) * tph)
    col_s, col_e = int(tile_col * tpw), int((tile_col + 1) * tpw)
    indices = []
    for r in range(row_s, row_e):
        for c in range(col_s, col_e):
            indices.append(r * merged_w + c)
    return indices
