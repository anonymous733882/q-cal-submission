"""Importance-only ID-Selection control."""

import torch

from baselines.idselection.adapter import IDSelectionBaseline

_VISION_START_ID = 151652
_VISION_END_ID = 151653

class IDImportanceTopK(IDSelectionBaseline):
    """Importance-only control for ID-Selection.

    It uses the same Qwen importance estimator as ID-Selection, but disables
    Gaussian diversity suppression and simply keeps the top-k visual tokens.
    """

    @torch.no_grad()
    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw, prune_ratio=None):
        ratio = 0.9 if prune_ratio is None else prune_ratio
        n_vis = int(vis_embeds.shape[0])
        n_keep = max(1, int(n_vis * (1.0 - ratio)))

        ids = inputs["input_ids"][0]
        vs = (ids == _VISION_START_ID).nonzero(as_tuple=True)[0]
        ve = (ids == _VISION_END_ID).nonzero(as_tuple=True)[0]
        if len(vs) > 0 and len(ve) > 0:
            start, end = vs[0].item(), ve[0].item()
            vis_pos = [i for i in range(start + 1, end) if ids[i] == 151655]
            text_pos = [i for i in range(ids.shape[0]) if i < start or i > end]
        else:
            vis_pos, text_pos = [], []

        if vis_pos and text_pos:
            scores, _ = self._importance_scores(
                model_wrapper, inputs, vis_pos, text_pos)
        else:
            scores = torch.zeros(n_vis)

        if scores.shape[0] != n_vis:
            padded = torch.zeros(n_vis)
            m = min(scores.shape[0], n_vis)
            padded[:m] = scores[:m]
            scores = padded

        keep = set(scores.argsort(descending=True)[:n_keep].tolist())
        return [i in keep for i in range(n_vis)]
