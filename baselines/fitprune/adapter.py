"""FitPrune baseline: faithful per-layer progressive visual token pruning.

Reference: Ye et al., "FitPrune: Visually Lossless Token Pruning for
Large Vision-Language Models" (2024)

Faithful reproduction:
  1. Pre-defined per-layer delete schedule (how many tokens to drop at each layer)
  2. At each layer with deletions > 0:
     - Re-compute self_attn × cross_attn score using CURRENT hidden states
     - Drop the lowest-scoring visual tokens
  3. Progressive: tokens removed early never come back
  4. Most pruning happens in early layers (0-6)

The delete schedule is adapted from the paper's calibration results.
For fair comparison, we scale the schedule to match the target budget_frac.

Uses monkey-patch forward to intercept at every layer with deletions.
GAP-aware: preserves original RoPE position IDs.
"""

import torch
from typing import Optional, Dict

_IMAGE_TOKEN_ID = 151655
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653

# FitPrune reference schedules: {reduction_pct: {layer: num_to_delete}}
# Adapted from the paper's calibration on 576 visual tokens (LLaVA-1.5).
# For Qwen2-VL (28 layers), we use layers 0-27.
# The distribution shape matters more than absolute counts — we rescale.
REFERENCE_SCHEDULES = {
    50: {0: 3, 1: 7, 2: 49, 3: 24, 4: 29, 5: 17, 6: 59, 7: 41, 8: 26,
         9: 8, 10: 10, 11: 6, 12: 2, 13: 5, 14: 7, 15: 2, 16: 30, 17: 3,
         18: 13, 19: 27, 20: 5, 21: 23, 22: 2, 23: 1, 24: 0, 25: 0, 26: 1, 27: 1},
    70: {0: 48, 1: 79, 2: 118, 3: 41, 4: 33, 5: 17, 6: 36, 7: 9, 8: 9,
         9: 2, 10: 3, 11: 2, 12: 1, 13: 3, 14: 4, 15: 8, 16: 18, 17: 7,
         18: 8, 19: 20, 20: 4, 21: 13, 22: 2, 23: 1, 24: 0, 25: 0, 26: 1, 27: 0},
    80: {0: 128, 1: 127, 2: 118, 3: 35, 4: 26, 5: 8, 6: 13, 7: 3, 8: 2,
         9: 1, 10: 0, 11: 0, 12: 0, 13: 2, 14: 1, 15: 8, 16: 7, 17: 8,
         18: 5, 19: 11, 20: 2, 21: 6, 22: 1, 23: 1, 24: 0, 25: 0, 26: 0, 27: 1},
    90: {0: 262, 1: 153, 2: 80, 3: 19, 4: 13, 5: 1, 6: 2, 7: 0, 8: 0,
         9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 1, 16: 1, 17: 2,
         18: 1, 19: 2, 20: 0, 21: 1, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 1},
}


class FitPruneBaseline:
    """FitPrune progressive per-layer pruning — faithful to the original paper.

    Args:
        model: Qwen2VLWrapper
        budget_frac: target keep fraction (0.0-1.0). Used to select and
            scale the reference schedule.
    """

    def __init__(self, model, budget_frac=0.5):
        self.model = model
        self.budget_frac = budget_frac
        cfg = model.model.config
        self.num_layers = getattr(cfg, "num_hidden_layers", 28)

    def _interpolate_reference(self, budget_frac):
        """Interpolate between two adjacent reference schedules.

        Instead of jumping between discrete schedules (which causes TL
        discontinuities), we linearly interpolate the per-layer drop
        distribution between the two nearest references.

        Returns:
            dict {layer: interpolated_drop_count} (float values)
        """
        reduction_pct = (1.0 - budget_frac) * 100
        available = sorted(REFERENCE_SCHEDULES.keys())  # [50, 70, 80, 90]

        # Clamp to range
        if reduction_pct <= available[0]:
            return REFERENCE_SCHEDULES[available[0]]
        if reduction_pct >= available[-1]:
            return REFERENCE_SCHEDULES[available[-1]]

        # Find bracketing schedules
        lo_ref, hi_ref = available[0], available[-1]
        for i in range(len(available) - 1):
            if available[i] <= reduction_pct <= available[i + 1]:
                lo_ref, hi_ref = available[i], available[i + 1]
                break

        # Interpolation weight: 0.0 = lo_ref, 1.0 = hi_ref
        span = hi_ref - lo_ref
        if span == 0:
            return REFERENCE_SCHEDULES[lo_ref]
        alpha = (reduction_pct - lo_ref) / span

        lo_sched = REFERENCE_SCHEDULES[lo_ref]
        hi_sched = REFERENCE_SCHEDULES[hi_ref]

        # Interpolate per-layer counts
        all_layers = set(lo_sched.keys()) | set(hi_sched.keys())
        interp = {}
        for layer in all_layers:
            lo_val = lo_sched.get(layer, 0)
            hi_val = hi_sched.get(layer, 0)
            interp[layer] = lo_val + alpha * (hi_val - lo_val)
        return interp

    def build_boundary_spec(self, num_visual_tokens: int, budget_frac: float):
        """Build {layer_idx: num_to_drop} scaled to actual token count.

        Uses interpolated reference schedules for smooth TL behavior.
        The distribution shape is interpolated, then rescaled to hit the
        exact target drop count.

        Returns:
            dict {layer_idx: num_to_drop}
        """
        if budget_frac >= 1.0:
            return {}

        ref_schedule = self._interpolate_reference(budget_frac)
        ref_total = sum(ref_schedule.values())
        if ref_total == 0:
            return {}

        # Target: drop (1 - budget_frac) of visual tokens
        target_drop = int(num_visual_tokens * (1.0 - budget_frac))
        if target_drop <= 0:
            return {}

        # Scale interpolated reference proportionally
        scale = target_drop / ref_total
        raw = {layer: max(0, round(count * scale))
               for layer, count in ref_schedule.items()
               if layer < self.num_layers}

        # Adjust to hit exact target (rounding errors)
        current_total = sum(raw.values())
        diff = target_drop - current_total
        if diff != 0:
            # Add/remove from layers with largest counts
            sorted_layers = sorted(raw.keys(), key=lambda l: raw[l], reverse=True)
            for l in sorted_layers:
                if diff == 0:
                    break
                if diff > 0:
                    raw[l] += 1
                    diff -= 1
                elif raw[l] > 0:
                    raw[l] -= 1
                    diff += 1

        # Ensure we never try to drop more than remaining
        spec = {}
        remaining = num_visual_tokens
        for layer in range(self.num_layers):
            if layer in raw and raw[layer] > 0:
                drop = min(raw[layer], remaining - 1)  # keep at least 1
                if drop > 0:
                    spec[layer] = drop
                    remaining -= drop

        return spec

    def generate_progressive(self, inputs, vis_start, vis_end,
                             budget_frac=None, max_new_tokens=256,
                             vis_token_positions=None):
        """Run FitPrune progressive generation.

        Args:
            inputs: prepared model inputs
            vis_start: start index of visual tokens
            vis_end: end index (exclusive)
            budget_frac: override budget (default: self.budget_frac)
            max_new_tokens: generation length
            vis_token_positions: explicit list of IMAGE_TOKEN_ID positions
                for multi-image (non-contiguous visual segments)

        Returns:
            Generated text string
        """
        from baselines.common.progressive_manipulator import (
            BaselineProgressiveManipulator, fitprune_rank_fn)

        bf = budget_frac if budget_frac is not None else self.budget_frac
        num_vis = len(vis_token_positions) if vis_token_positions else (vis_end - vis_start)
        boundary_spec = self.build_boundary_spec(num_vis, bf)

        if not boundary_spec:
            output_ids = self.model.model.generate(
                **inputs, max_new_tokens=max_new_tokens)
            return self.model.processor.decode(
                output_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True).strip()

        # Build inputs_embeds (full sequence, no pre-removal)
        input_ids = inputs["input_ids"]
        text_embeds = self.model.model.model.language_model.get_input_embeddings()(input_ids)

        vis_out = self.model.model.model.visual(
            inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
        vis_embeds = vis_out.pooler_output

        if vis_embeds.shape[0] > 0:
            mask, _ = self.model.model.model.get_placeholder_mask(
                input_ids, inputs_embeds=text_embeds, image_features=vis_embeds)
            inputs_embeds = text_embeds.masked_scatter(mask, vis_embeds)
        else:
            inputs_embeds = text_embeds

        self.model.model.model.rope_deltas = None

        # Install progressive manipulator
        manip = BaselineProgressiveManipulator(
            self.model, boundary_spec, vis_start, vis_end,
            rank_fn=fitprune_rank_fn,
            vis_token_positions=vis_token_positions)
        manip.patch()

        try:
            gen_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": inputs.get("attention_mask"),
                "position_ids": inputs.get("position_ids"),
                "max_new_tokens": max_new_tokens,
            }
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            output_ids = self.model.model.generate(**gen_kwargs)
            return self.model.processor.decode(
                output_ids[0], skip_special_tokens=True).strip()
        finally:
            manip.unpatch()

    def get_schedule_info(self, num_visual_tokens, budget_frac=None):
        """Return human-readable schedule info."""
        bf = budget_frac if budget_frac is not None else self.budget_frac
        spec = self.build_boundary_spec(num_visual_tokens, bf)
        remaining = num_visual_tokens
        info = []
        for layer in range(self.num_layers):
            drop = spec.get(layer, 0)
            remaining -= drop
            if drop > 0:
                info.append({
                    "layer": layer,
                    "drop": drop,
                    "remaining": remaining,
                    "keep_ratio": remaining / num_visual_tokens,
                })
        return info


def find_visual_range(input_ids):
    """Find start and end of visual tokens in input_ids."""
    ids = input_ids[0] if input_ids.dim() > 1 else input_ids
    img_mask = (ids == _IMAGE_TOKEN_ID)
    positions = torch.where(img_mask)[0]
    if len(positions) == 0:
        return 0, 0
    return positions[0].item(), positions[-1].item() + 1
