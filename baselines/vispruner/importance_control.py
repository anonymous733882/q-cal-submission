"""Importance-only VisPruner control."""

import torch

from baselines.vispruner.adapter import VisPrunerBaseline

class VisPrunerTopK(VisPrunerBaseline):
    """Importance-only control for VisPruner.

    It uses the same ViT received-attention global signal as VisPruner, but
    disables the Stage-2 diversity fill/deduplication and keeps the top-k visual
    tokens by score.
    """

    @torch.no_grad()
    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw, prune_ratio=None):
        ratio = prune_ratio if prune_ratio is not None else self.prune_ratio
        n_vis = int(vis_embeds.shape[0])
        n_keep = max(1, int(n_vis * (1.0 - ratio)))

        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is None:
            scores = vis_embeds.float().norm(dim=-1).cpu()
        else:
            patch_scores = self._vit_received_attention(pixel_values, image_grid_thw)
            merge = self.model.model.model.visual.spatial_merge_size
            m2 = merge * merge
            n_vis_from_patches = patch_scores.shape[0] // m2
            scores = patch_scores[: n_vis_from_patches * m2].view(
                n_vis_from_patches, m2).mean(dim=1)
            if n_vis_from_patches != n_vis:
                padded = torch.zeros(n_vis)
                m = min(n_vis_from_patches, n_vis)
                padded[:m] = scores[:m]
                scores = padded

        keep = set(scores.argsort(descending=True)[:n_keep].tolist())
        return [i in keep for i in range(n_vis)]
