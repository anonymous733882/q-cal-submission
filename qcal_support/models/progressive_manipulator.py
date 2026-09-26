"""Progressive layer-wise token manipulation via monkey-patching.

Replaces the Qwen2VLTextModel forward loop with a version that modifies
visual tokens at group boundaries.  This avoids all hook-ordering
conflicts with accelerate's AlignDevicesHook.

Tier degradation support:
  At each group boundary, tiles can transition between tiers:
    full → lite:   checkerboard spatial downsample (keep ~50% of tokens)
    lite → coarse: pool remaining tokens into 1 representative token
    coarse → skip: remove the pooled token entirely
    full → coarse: pool all tokens into 1
    full → skip:   remove all tokens
    lite → skip:   remove all remaining tokens

  This is more fine-grained than the old binary keep/drop approach.
  The key mapping challenge is tracking which tokens belong to which
  tile and which are "real" vs "pooled" (coarse sentinels).

GAP-aware:
  Preserved tokens retain their ORIGINAL RoPE position IDs.
  Removed tokens leave gaps — spatial relationships maintained.
"""

from __future__ import annotations
import functools
import torch
from typing import Dict, List, Set, Tuple
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

TIER_RANK = {"skip": 0, "coarse": 1, "lite": 2, "full": 3}


class ProgressiveTokenManipulator:
    """Monkey-patch based progressive layer-wise token manipulation.

    Supports tier degradation: at each group boundary, tiles can be
    downgraded from full→lite→coarse→skip with appropriate token
    transformations (spatial downsample, pooling, removal).

    Usage::

        ptm = ProgressiveTokenManipulator(wrapper, schedule, vis_start, vis_end,
                                          merged_h, merged_w, grid_size,
                                          kept_orig_indices)
        ptm.install_hooks()
        output = wrapper.model.generate(...)
        ptm.remove_hooks()

    Args:
        kept_orig_indices: list from V2's output — one entry per V2 visual
            token, giving its original grid index (negative = coarse sentinel).
    """

    def __init__(
        self,
        wrapper,
        schedule: Dict[int, Dict[int, str]],
        vis_start: int,
        vis_end: int,
        merged_h: int,
        merged_w: int,
        grid_size: int = 6,
        kept_orig_indices: List[int] | None = None,
    ):
        self.wrapper = wrapper
        self.schedule = schedule
        self.vis_start = vis_start
        self.vis_end = vis_end
        self.merged_h = merged_h
        self.merged_w = merged_w
        self.grid_size = grid_size
        self._original_forward = None

        # kept_orig_indices: one per V2 visual token
        # positive = original grid index, negative = -(tile_id+1) for coarse
        self._kept_orig = kept_orig_indices or list(range(vis_end - vis_start))

        # Map each V2 token position (0-based in vis block) to its tile_id
        self._token_tile: List[int] = []
        for orig_idx in self._kept_orig:
            if orig_idx >= 0:
                r = orig_idx // merged_w
                c = orig_idx % merged_w
                tile_row = min(int(r / (merged_h / grid_size)), grid_size - 1)
                tile_col = min(int(c / (merged_w / grid_size)), grid_size - 1)
                self._token_tile.append(tile_row * grid_size + tile_col)
            else:
                # Coarse sentinel: -(tile_id+1)
                self._token_tile.append(-(orig_idx + 1))

        # Track which V2 tokens are "coarse" (pooled) vs "real"
        self._is_coarse: List[bool] = [idx < 0 for idx in self._kept_orig]

        # Track which V2 tokens are "lite" (checkerboard-kept) vs "full"
        # Lite tokens have (row+col) % 2 == 0 in their tile's local grid
        self._is_lite: List[bool] = [False] * len(self._kept_orig)
        # We don't know the initial tier from V2 here, but the schedule
        # tells us: schedule[0] has the group-0 allocation.
        if 0 in schedule:
            group0_alloc = schedule[0]
            for pos, tile_id in enumerate(self._token_tile):
                tier = group0_alloc.get(tile_id, "skip")
                if tier == "lite":
                    self._is_lite[pos] = True

        # Pre-compute boundary actions for each group transition
        sorted_boundaries = sorted(schedule.keys())
        self._boundary_actions: Dict[int, Dict[int, Tuple[str, str]]] = {}
        for b_idx in range(1, len(sorted_boundaries)):
            layer_idx = sorted_boundaries[b_idx]
            prev_alloc = schedule[sorted_boundaries[b_idx - 1]]
            new_alloc = schedule[layer_idx]
            actions = {}  # tile_id → (old_tier, new_tier)
            all_tiles = set(prev_alloc.keys()) | set(new_alloc.keys())
            for tid in all_tiles:
                old_tier = prev_alloc.get(tid, "skip")
                new_tier = new_alloc.get(tid, "skip")
                if TIER_RANK[new_tier] < TIER_RANK[old_tier]:
                    actions[tid] = (old_tier, new_tier)
            if actions:
                self._boundary_actions[layer_idx] = actions

    # ── public API ───────────────────────────────────────────────────

    def patch(self):
        """Replace TextModel.forward with our progressive version."""
        lm = self.wrapper.model.model.language_model
        self._original_forward = lm.forward
        lm.forward = functools.update_wrapper(
            functools.partial(self._progressive_forward, lm),
            self._original_forward,
        )

    def unpatch(self):
        """Restore original TextModel.forward."""
        if self._original_forward is not None:
            lm = self.wrapper.model.model.language_model
            lm.forward = self._original_forward
            self._original_forward = None

    install_hooks = patch
    remove_hooks = unpatch

    # ── patched forward ──────────────────────────────────────────────

    def _progressive_forward(self, text_model, *args, **kwargs):
        """Drop-in replacement for Qwen2VLTextModel.forward."""
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        position_ids = kwargs.get("position_ids", args[2] if len(args) > 2 else None)
        past_key_values = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
        inputs_embeds = kwargs.get("inputs_embeds", args[4] if len(args) > 4 else None)
        use_cache = kwargs.get("use_cache", args[5] if len(args) > 5 else None)
        extra_kwargs = {k: v for k, v in kwargs.items()
                        if k not in ("input_ids", "attention_mask", "position_ids",
                                     "past_key_values", "inputs_embeds", "use_cache")}

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=text_model.config)

        if inputs_embeds is None:
            inputs_embeds = text_model.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        # Causal mask
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
            mask_kwargs = {
                "config": text_model.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if text_model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = text_model.rotary_emb(hidden_states, position_ids)

        # ── Progressive state ────────────────────────────────────────
        seq_len = hidden_states.shape[1]
        is_prefill = seq_len > 1

        if is_prefill:
            num_vis = len(self._kept_orig)
            # vis_pos_indices[i] = current seq position of V2 visual token i
            vis_pos_indices = list(range(self.vis_start, self.vis_start + num_vis))
            # alive[i] = True if V2 token i is still in the sequence
            alive = [True] * num_vis
            # Current tier tracking per V2 token
            is_coarse = list(self._is_coarse)
            is_lite = list(self._is_lite)
        else:
            vis_pos_indices = None
            alive = None

        for i, decoder_layer in enumerate(text_model.layers):
            # Apply tier degradation at group boundary
            if is_prefill and i in self._boundary_actions:
                actions = self._boundary_actions[i]
                result = self._degrade_at_boundary(
                    hidden_states, position_embeddings, causal_mask_mapping,
                    text_position_ids,
                    vis_pos_indices, alive, is_coarse, is_lite, actions,
                )
                hidden_states = result["hidden_states"]
                position_embeddings = result["position_embeddings"]
                causal_mask_mapping = result["causal_mask_mapping"]
                text_position_ids = result["text_position_ids"]
                vis_pos_indices = result["vis_pos_indices"]
                alive = result["alive"]
                is_coarse = result["is_coarse"]
                is_lite = result["is_lite"]

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[text_model.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **extra_kwargs,
            )

        hidden_states = text_model.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    # ── tier degradation logic ──────────────────────────────────────

    def _degrade_at_boundary(
        self,
        hidden_states,
        position_embeddings,
        causal_mask_mapping,
        text_position_ids,
        vis_pos_indices,
        alive,
        is_coarse,
        is_lite,
        actions,  # {tile_id: (old_tier, new_tier)}
    ):
        """Apply tier degradation at a group boundary.

        Handles all transitions:
          full → lite:   remove checkerboard-odd tokens from the tile
          full → coarse: pool all tile tokens into 1
          full → skip:   remove all tile tokens
          lite → coarse: pool remaining lite tokens into 1
          lite → skip:   remove all remaining lite tokens
          coarse → skip: remove the single pooled token
        """
        seq_len = hidden_states.shape[1]
        device = hidden_states.device

        # Build tile → alive V2 token positions mapping
        tile_to_v2 = {}  # tile_id → list of (v2_idx, seq_pos)
        for v2i, seq_pos in enumerate(vis_pos_indices):
            if alive[v2i]:
                tid = self._token_tile[v2i]
                if tid not in tile_to_v2:
                    tile_to_v2[tid] = []
                tile_to_v2[tid].append((v2i, seq_pos))

        # Determine which seq positions to keep, pool, or drop
        positions_to_drop = set()
        pool_operations = []  # list of (target_seq_pos, source_seq_positions)

        new_alive = list(alive)
        new_is_coarse = list(is_coarse)
        new_is_lite = list(is_lite)

        for tile_id, (old_tier, new_tier) in actions.items():
            if tile_id not in tile_to_v2:
                continue
            v2_tokens = tile_to_v2[tile_id]  # [(v2_idx, seq_pos), ...]

            if new_tier == "skip":
                # Remove all tokens for this tile
                for v2i, seq_pos in v2_tokens:
                    positions_to_drop.add(seq_pos)
                    new_alive[v2i] = False

            elif new_tier == "coarse":
                if len(v2_tokens) <= 1:
                    # Already 1 token — just mark as coarse
                    for v2i, _ in v2_tokens:
                        new_is_coarse[v2i] = True
                        new_is_lite[v2i] = False
                else:
                    # Pool all tokens into the first one, drop the rest
                    source_positions = [sp for _, sp in v2_tokens]
                    keep_v2i, keep_pos = v2_tokens[0]
                    pool_operations.append((keep_pos, source_positions))
                    for v2i, seq_pos in v2_tokens[1:]:
                        positions_to_drop.add(seq_pos)
                        new_alive[v2i] = False
                    new_is_coarse[keep_v2i] = True
                    new_is_lite[keep_v2i] = False

            elif new_tier == "lite" and old_tier == "full":
                # Checkerboard downsample: keep tokens where (row+col) % 2 == 0
                # within the tile's local grid
                for v2i, seq_pos in v2_tokens:
                    orig_idx = self._kept_orig[v2i]
                    if orig_idx >= 0:
                        r = orig_idx // self.merged_w
                        c = orig_idx % self.merged_w
                        if (r + c) % 2 != 0:
                            positions_to_drop.add(seq_pos)
                            new_alive[v2i] = False
                        else:
                            new_is_lite[v2i] = True
                    else:
                        # Already a coarse token — keep it
                        new_is_lite[v2i] = True

        if not positions_to_drop and not pool_operations:
            return {
                "hidden_states": hidden_states,
                "position_embeddings": position_embeddings,
                "causal_mask_mapping": causal_mask_mapping,
                "text_position_ids": text_position_ids,
                "vis_pos_indices": vis_pos_indices,
                "alive": alive,
                "is_coarse": is_coarse,
                "is_lite": is_lite,
            }

        # Apply pooling operations (mean-pool source tokens into target position)
        if pool_operations:
            for target_pos, source_positions in pool_operations:
                source_idx = torch.tensor(source_positions, dtype=torch.long, device=device)
                pooled = hidden_states[:, source_idx, :].mean(dim=1, keepdim=True)
                hidden_states = hidden_states.clone()
                hidden_states[:, target_pos:target_pos + 1, :] = pooled

        # Build keep list (all positions except dropped ones)
        keep_seq = [pos for pos in range(seq_len) if pos not in positions_to_drop]

        if len(keep_seq) == seq_len:
            return {
                "hidden_states": hidden_states,
                "position_embeddings": position_embeddings,
                "causal_mask_mapping": causal_mask_mapping,
                "text_position_ids": text_position_ids,
                "vis_pos_indices": vis_pos_indices,
                "alive": new_alive,
                "is_coarse": new_is_coarse,
                "is_lite": new_is_lite,
            }

        # Remap vis_pos_indices after removal
        old_to_new = {}
        for new_pos, old_pos in enumerate(keep_seq):
            old_to_new[old_pos] = new_pos
        new_vis_pos_indices = list(vis_pos_indices)
        for v2i in range(len(new_vis_pos_indices)):
            if new_alive[v2i]:
                old_pos = vis_pos_indices[v2i]
                new_vis_pos_indices[v2i] = old_to_new[old_pos]

        idx = torch.tensor(keep_seq, dtype=torch.long)

        # Prune hidden_states
        new_hidden = hidden_states[:, idx.to(device), :]

        # Prune position_embeddings (shape: tuple of (cos, sin))
        cos, sin = position_embeddings
        new_pos_emb = (cos[:, :, idx.to(cos.device), :], sin[:, :, idx.to(sin.device), :])

        # Prune causal masks
        new_masks = {}
        for key, mask in causal_mask_mapping.items():
            if mask is None:
                new_masks[key] = None
            else:
                idx_m = idx.to(mask.device)
                new_masks[key] = mask[:, :, idx_m, :][:, :, :, idx_m]

        # Prune text_position_ids
        new_text_pos = text_position_ids
        if text_position_ids is not None:
            new_text_pos = text_position_ids[:, idx.to(text_position_ids.device)]

        return {
            "hidden_states": new_hidden,
            "position_embeddings": new_pos_emb,
            "causal_mask_mapping": new_masks,
            "text_position_ids": new_text_pos,
            "vis_pos_indices": new_vis_pos_indices,
            "alive": new_alive,
            "is_coarse": new_is_coarse,
            "is_lite": new_is_lite,
        }
