"""Prompt-aware ZSPAPrune paper control."""

import torch
import torch.nn.functional as F

_IMAGE_TOKEN_ID = 151655
_VISION_START_ID = 151652
_VISION_END_ID = 151653

class ZSPAPrunePaper:
    """ZSPAPrune prompt-aware core plus diversity fill.

    This implementation enforces the requested token budget exactly. It is kept
    local to dualsignal so BEA remains read-only.
    """

    def __init__(self, core_ratio: float = 0.6, min_tokens: int = 4):
        self.core_ratio = core_ratio
        self.min_tokens = min_tokens

    @torch.no_grad()
    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw, prune_ratio=None):
        ratio = 0.9 if prune_ratio is None else prune_ratio
        n_vis = int(vis_embeds.shape[0])
        n_keep = max(self.min_tokens, int(n_vis * (1.0 - ratio)))
        n_keep = min(n_keep, n_vis)
        if n_keep >= n_vis:
            return [True] * n_vis

        ids = inputs["input_ids"][0]
        vs = (ids == _VISION_START_ID).nonzero(as_tuple=True)[0]
        ve = (ids == _VISION_END_ID).nonzero(as_tuple=True)[0]
        if len(vs) > 0 and len(ve) > 0:
            start, end = vs[0].item(), ve[0].item()
            text_pos = [i for i in range(ids.shape[0]) if i < start or i > end]
        else:
            text_pos = list(range(ids.shape[0]))

        lm = model_wrapper.model.model.language_model
        text_ids = ids[text_pos]
        text_embeds = lm.embed_tokens(text_ids.unsqueeze(0)).squeeze(0)
        prompt = F.normalize(text_embeds.float().mean(dim=0).cpu(), dim=0)
        vis = F.normalize(vis_embeds.float().cpu(), dim=-1)

        prompt_sim = torch.matmul(vis, prompt)
        n_core = max(1, min(n_keep, int(n_keep * self.core_ratio)))
        ranked = prompt_sim.argsort(descending=True).tolist()
        selected = ranked[:n_core]
        selected_set = set(selected)

        if len(selected) < n_keep:
            sim = torch.matmul(vis, vis.T)
            # Distance to selected set. Higher distance means more diverse.
            max_sim_to_selected = torch.full((n_vis,), -float("inf"))
            for idx in selected:
                max_sim_to_selected = torch.maximum(max_sim_to_selected, sim[idx])
            for idx in selected_set:
                max_sim_to_selected[idx] = float("inf")

            while len(selected) < n_keep:
                candidate_scores = max_sim_to_selected.clone()
                for idx in selected_set:
                    candidate_scores[idx] = float("inf")
                next_idx = candidate_scores.argmin().item()
                if next_idx in selected_set:
                    break
                selected.append(next_idx)
                selected_set.add(next_idx)
                max_sim_to_selected = torch.maximum(max_sim_to_selected, sim[next_idx])

        # Deterministic fallback if numerical ties or duplicate protection left gaps.
        if len(selected_set) < n_keep:
            for idx in ranked:
                if idx not in selected_set:
                    selected_set.add(idx)
                if len(selected_set) >= n_keep:
                    break

        return [i in selected_set for i in range(n_vis)]
