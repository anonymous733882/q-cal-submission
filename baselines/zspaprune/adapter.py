"""ZSPAPrune baseline: Zero-Shot Prompt-Aware Token Pruning.

Reference: "ZSPAPrune: Zero-Shot Prompt-Aware Token Pruning for
            Vision-Language Models"
           arXiv:2510.17197, 2025

Core algorithm (2-phase):

  Phase 1 — Core set (task relevance):
    1. Aggregate text/instruction embeddings into a single prompt vector
       (mean pooling of all text token embeddings from LLM embedding layer).
    2. Compute cosine similarity between each visual token and prompt vector.
    3. Select top-r tokens as "core set" (most prompt-relevant).
    r = core_ratio * n_keep

  Phase 2 — Diversity augmentation:
    From remaining tokens, greedily select tokens that maximize minimum cosine
    distance to any already-selected token, until budget is filled.

Applied to Qwen2.5-VL: uses LLM embedding layer for text embeddings,
vis_embeds (post-merger) for visual tokens.
One-shot, training-free, zero-shot, no ViT/LLM attention needed.
"""

import torch
import torch.nn.functional as F

_IMAGE_TOKEN_ID  = 151655
_VISION_START_ID = 151652
_VISION_END_ID   = 151653


class ZSPAPruneBaseline:
    """ZSPAPrune: prompt-aware core + diversity visual token selection.

    Args:
        model:       Qwen2VLWrapper
        prune_ratio: fraction of tokens to DROP (default 0.5)
        core_ratio:  fraction of budget filled by prompt-relevant core (default 0.6)
        min_tokens:  minimum tokens to retain (default 4)
    """

    def __init__(self, model,
                 prune_ratio: float = 0.5,
                 core_ratio:  float = 0.6,
                 min_tokens:  int   = 4):
        self.model       = model
        self.prune_ratio = prune_ratio
        self.core_ratio  = core_ratio
        self.min_tokens  = min_tokens

    @torch.no_grad()
    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw,
              prune_ratio=None):
        """ZSPAPrune 2-phase selection. Returns keep_mask list[bool]."""
        ratio = prune_ratio if prune_ratio is not None else self.prune_ratio
        n_vis = vis_embeds.shape[0]
        n_keep = max(self.min_tokens, int(n_vis * (1.0 - ratio)))
        n_keep = min(n_keep, n_vis)

        if n_keep >= n_vis:
            return [True] * n_vis

        # --- Extract text embeddings from LLM embedding layer ---
        ids = inputs["input_ids"][0]
        vs = (ids == _VISION_START_ID).nonzero(as_tuple=True)[0]
        ve = (ids == _VISION_END_ID).nonzero(as_tuple=True)[0]

        if len(vs) > 0 and len(ve) > 0:
            vs_val, ve_val = vs[0].item(), ve[0].item()
            text_pos = [i for i in range(ids.shape[0])
                       if i < vs_val or i > ve_val]
        else:
            text_pos = list(range(ids.shape[0]))

        # Get LLM embedding layer
        lm = model_wrapper.model.model.language_model
        embed_layer = lm.embed_tokens
        text_ids = ids[text_pos]
        text_embeds = embed_layer(text_ids.unsqueeze(0)).squeeze(0)  # [n_text, hidden]

        # Prompt vector = mean of text embeddings
        prompt_vec = text_embeds.float().cpu().mean(dim=0)  # [hidden]
        prompt_vec = F.normalize(prompt_vec, dim=0)

        # Visual token embeddings normalized
        vis_norm = F.normalize(vis_embeds.float().cpu(), dim=-1)  # [n_vis, hidden]

        # --- Phase 1: Core set (prompt-relevant) ---
        n_core = max(1, int(n_keep * self.core_ratio))

        # Cosine similarity to prompt vector
        prompt_sim = torch.matmul(vis_norm, prompt_vec)  # [n_vis]
        core_ranked = prompt_sim.argsort(descending=True).tolist()
        selected = list(core_ranked[:n_core])
        selected_set = set(selected)

        # --- Phase 2: Diversity augmentation (greedy max-min distance) ---
        n_diverse = n_keep - len(selected)

        if n_diverse > 0:
            # Cosine similarity matrix for greedy selection
            sim_matrix = torch.matmul(vis_norm, vis_norm.T)  # [n_vis, n_vis]

            # Initialize min_sim as min similarity to any selected token
            min_sim = torch.full((n_vis,), float('inf'), device=vis_norm.device)
            for idx in selected:
                min_sim = torch.min(min_sim, sim_matrix[idx])
            # Mark selected as unavailable
            for idx in selected:
                min_sim[idx] = float('inf')

            for _ in range(n_diverse):
                # Select token with minimum similarity to selected set
                next_idx = min_sim.argmin().item()
                if min_sim[next_idx] == float('inf'):
                    break
                selected.append(next_idx)
                selected_set.add(next_idx)
                # Update min similarities
                min_sim = torch.min(min_sim, sim_matrix[next_idx])
                min_sim[next_idx] = float('inf')

        keep_set = set(selected)
        return [i in keep_set for i in range(n_vis)]
