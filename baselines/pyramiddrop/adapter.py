"""PyramidDrop baseline: faithful multi-stage progressive visual token pruning.

Reference: Xia et al., "PyramidDrop: Accelerating Your Large Vision-Language
Models via Pyramid Visual Redundancy Reduction" (2024)

Faithful reproduction:
  1. Divide LLM layers into S equal stages (default S=4 for 28 layers)
  2. Stage 0: keep 100% visual tokens (no pruning)
  3. At each stage boundary (start of stage 1, 2, 3):
     - Re-rank visual tokens using last-text-token → visual attention
       computed from CURRENT hidden states
     - Keep top λ^s fraction (cumulative: 1.0 → λ → λ² → λ³)
  4. Default λ=0.5 → ratios [1.0, 0.5, 0.25, 0.125]  (paper default)

Uses monkey-patch forward to intercept at stage boundaries.
GAP-aware: preserves original RoPE position IDs.
"""

import torch
from typing import Optional, List

_IMAGE_TOKEN_ID = 151655
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653


class PyramidDropBaseline:
    """PyramidDrop progressive pruning — faithful to the original paper.

    Args:
        model: Qwen2VLWrapper
        num_stages: number of pyramid stages (default 4)
        drop_ratio: λ — fraction kept at each stage transition
            cumulative: stage s keeps λ^s of original tokens
        budget_override: if set, overrides drop_ratio to achieve this
            final keep fraction (for fair comparison across methods)
    """

    def __init__(self, model, num_stages=4, drop_ratio=None, budget_override=None):
        self.model = model
        self.num_stages = num_stages
        cfg = model.model.config
        self.num_layers = getattr(cfg, "num_hidden_layers", 28)

        # Stage boundaries: start layer of each stage (except stage 0)
        layers_per_stage = self.num_layers // num_stages
        self.stage_boundaries = [
            (s + 1) * layers_per_stage
            for s in range(num_stages - 1)
        ]
        # e.g., 28 layers, 4 stages → boundaries at [7, 14, 21]

        # Compute per-stage keep ratios
        if budget_override is not None:
            # Solve: λ^(S-1) = budget_override → λ = budget^(1/(S-1))
            if budget_override >= 1.0:
                self.drop_ratio = 1.0
            else:
                self.drop_ratio = budget_override ** (1.0 / (num_stages - 1))
        else:
            self.drop_ratio = drop_ratio if drop_ratio is not None else 0.5

        # Cumulative ratios: [1.0, λ, λ², λ³]
        self.cumulative_ratios = [self.drop_ratio ** s for s in range(num_stages)]

    def build_boundary_spec(self, num_visual_tokens: int):
        """Build {layer_idx: num_to_drop} for the manipulator.

        At each stage boundary, we drop tokens to reach the cumulative ratio.
        """
        spec = {}
        current_count = num_visual_tokens
        for s, layer_idx in enumerate(self.stage_boundaries):
            # Target count at stage s+1
            target = max(1, int(num_visual_tokens * self.cumulative_ratios[s + 1]))
            num_to_drop = current_count - target
            if num_to_drop > 0:
                spec[layer_idx] = num_to_drop
            current_count = target
        return spec

    def generate_progressive(self, inputs, vis_start, vis_end,
                             max_new_tokens=256, vis_token_positions=None):
        """Run PyramidDrop progressive generation.

        Args:
            inputs: prepared model inputs (from prepare_inputs)
            vis_start: start index of visual tokens in input_ids
            vis_end: end index (exclusive)
            max_new_tokens: generation length
            vis_token_positions: explicit list of IMAGE_TOKEN_ID positions
                for multi-image (non-contiguous visual segments)

        Returns:
            Generated text string
        """
        from baselines.common.progressive_manipulator import (
            BaselineProgressiveManipulator, pyramiddrop_rank_fn)

        num_vis = len(vis_token_positions) if vis_token_positions else (vis_end - vis_start)
        boundary_spec = self.build_boundary_spec(num_vis)

        if not boundary_spec:
            # No pruning needed (budget >= 1.0)
            output_ids = self.model.model.generate(
                **inputs, max_new_tokens=max_new_tokens)
            return self.model.processor.decode(
                output_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True).strip()

        # Build inputs_embeds (same as V2 path but without any token removal)
        input_ids = inputs["input_ids"]
        text_embeds = self.model.model.model.language_model.get_input_embeddings()(input_ids)

        # Get visual embeddings
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
            rank_fn=pyramiddrop_rank_fn,
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

    def get_final_keep_ratio(self):
        """Return the final cumulative keep ratio."""
        return self.cumulative_ratios[-1]

    def get_stage_info(self):
        """Return human-readable stage info."""
        info = []
        for s in range(self.num_stages):
            if s == 0:
                layers = f"0-{self.stage_boundaries[0] - 1}"
            elif s < len(self.stage_boundaries):
                layers = f"{self.stage_boundaries[s-1]}-{self.stage_boundaries[s] - 1}"
            else:
                layers = f"{self.stage_boundaries[-1]}-{self.num_layers - 1}"
            info.append({
                "stage": s,
                "layers": layers,
                "keep_ratio": self.cumulative_ratios[s],
            })
        return info


def find_visual_range(input_ids):
    """Find start and end of visual tokens in input_ids.

    Returns (vis_start, vis_end) — the range of IMAGE_TOKEN_ID positions.
    """
    ids = input_ids[0] if input_ids.dim() > 1 else input_ids
    img_mask = (ids == _IMAGE_TOKEN_ID)
    positions = torch.where(img_mask)[0]
    if len(positions) == 0:
        return 0, 0
    return positions[0].item(), positions[-1].item() + 1
