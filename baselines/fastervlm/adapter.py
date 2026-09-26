"""FasterVLM baseline: ViT-side [CLS]-attention-based visual token pruning.

Reference: "[CLS] Attention is All You Need for Training-Free Visual Token Pruning"
           arXiv:2412.01818v1, FasterVLM (Theia-4869/FasterVLM)

Core algorithm:
  1. Hook the LAST full-attention ViT block to capture Q, K projections.
  2. Compute attention scores: for each patch token, how much is it attended to
     by all other tokens (received attention), averaged over all heads.
  3. In the original paper this is the [CLS]→patch attention; since Qwen2.5-VL
     uses NaViT (no [CLS] token), we use the standard adaptation:
       score(patch_i) = mean_{j} softmax(Q_j K^T / sqrt(d))[j, i]
     i.e. the column-mean of the attention matrix, averaged over all heads.
     This is the exact substitute used in the literature for non-CLS ViTs.
  4. Keep top-K patches by score; drop the rest. One-shot, before LLM.

Adaptation note (Qwen2.5-VL):
  Qwen2.5-VL ViT has 32 blocks with full-attention at blocks {7,15,23,31}.
  We use block 31 (last full-attention block) — closest to the original intent
  of using the final ViT layer attention.
  The qkv projection is fused: Linear(dim=1280, out=3840) → split into Q,K,V.
"""

import torch
import torch.nn.functional as F

_IMAGE_TOKEN_ID = 151655


class FasterVLMBaseline:
    """FasterVLM: ViT last-block received-attention scoring.

    Args:
        model: Qwen2VLWrapper
        vit_block_idx: which ViT full-attention block to hook (default 31 = last)
        prune_ratio: fraction of tokens to DROP (default 0.5 → keep 50%)
    """

    def __init__(self, model, vit_block_idx: int = 31, prune_ratio: float = 0.5):
        self.model        = model
        self.block_idx    = vit_block_idx
        self.prune_ratio  = prune_ratio

    # ------------------------------------------------------------------
    # ViT attention score extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _vit_received_attention(self, pixel_values, image_grid_thw):
        """Compute per-patch 'received attention' from ViT block `block_idx`.

        Returns:
            scores: Tensor [N_patches] (before PatchMerger), higher = more important
        """
        vit    = self.model.model.model.visual
        block  = vit.blocks[self.block_idx]
        attn_m = block.attn                    # Qwen2_5_VLVisionAttention

        captured = {}

        def _pre_hook(module, args):
            # args[0] is the hidden states entering this block: [N_patches, dim]
            captured["h"] = args[0].detach()

        handle = block.register_forward_pre_hook(_pre_hook)
        try:
            _ = vit(pixel_values, image_grid_thw)
        finally:
            handle.remove()

        h = captured["h"]               # [N_patches, dim=1280]
        N, D = h.shape
        num_heads = attn_m.num_heads    # 16
        head_dim  = attn_m.head_dim     # 80

        device = h.device
        dtype  = h.dtype

        # qkv is a fused Linear(1280 → 3840)
        qkv = attn_m.qkv(h)                          # [N, 3 * H * head_dim]
        q, k, _ = qkv.chunk(3, dim=-1)               # each [N, H * head_dim]

        q = q.view(N, num_heads, head_dim).permute(1, 0, 2).float()  # [H, N, D]
        k = k.view(N, num_heads, head_dim).permute(1, 0, 2).float()  # [H, N, D]

        scale = head_dim ** -0.5
        attn  = torch.matmul(q, k.transpose(-1, -2)) * scale         # [H, N, N]
        attn  = F.softmax(attn, dim=-1)                               # [H, N, N]

        # Received attention: column mean → how much each token is attended to
        # shape: [H, N] → average over heads → [N]
        received = attn.mean(dim=1)   # [H, N]: mean over query dimension
        scores   = received.mean(dim=0)  # [N]: mean over heads

        return scores.cpu()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_tokens(self, inputs):
        """Score visual tokens using FasterVLM received-attention.

        Args:
            inputs: dict from model.prepare_inputs(image, prompt)

        Returns:
            scores: list[float], length = number of visual tokens in LLM input
                    (after PatchMerger), higher = more important
        """
        # Run ViT to get patch-level scores (before merger)
        pixel_values   = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        if pixel_values is None or image_grid_thw is None:
            # No visual tokens
            n_vis = (inputs["input_ids"][0] == _IMAGE_TOKEN_ID).sum().item()
            return [0.0] * n_vis

        patch_scores = self._vit_received_attention(
            pixel_values, image_grid_thw)   # [N_patches]

        # PatchMerger groups spatial_merge_size² patches → 1 LLM visual token
        merge = self.model.model.model.visual.spatial_merge_size  # typically 2
        m2    = merge * merge  # 4

        N_patches = patch_scores.shape[0]
        N_vis     = N_patches // m2          # number of LLM visual tokens

        # Average patch scores within each merge group → token score
        token_scores = patch_scores[: N_vis * m2].view(N_vis, m2).mean(dim=1)
        return token_scores.tolist()

    @torch.no_grad()
    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw, prune_ratio=None):
        """Return keep_mask for visual tokens using FasterVLM scores.

        Args:
            prune_ratio: override instance prune_ratio if given

        Returns:
            keep_mask: list[bool], length = num_visual_tokens
        """
        ratio    = prune_ratio if prune_ratio is not None else self.prune_ratio
        scores   = self.score_tokens(inputs)
        n_vis    = len(scores)
        n_keep   = max(1, int(n_vis * (1.0 - ratio)))

        ranked   = sorted(range(n_vis), key=lambda i: scores[i], reverse=True)
        keep_set = set(ranked[:n_keep])
        return [i in keep_set for i in range(n_vis)]
