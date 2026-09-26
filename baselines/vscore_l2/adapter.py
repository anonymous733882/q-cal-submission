"""v_score_L2 baseline: text→vis attention × value norm at layer 2.

v_score_L2 = v_attn_L2 × v_vnorm_L2

  v_attn_L2:  mean text→vis attention using layer 2's Q/K projections
              (same as HAWK but at layer 2 instead of layer 0)
  v_vnorm_L2: ||W_V h_vis|| averaged over heads
              (value expressiveness of each visual token)

Cost vs HAWK:
  HAWK:       early-exit after layer 0 (1 layer)
  v_score_L2: pre-hook fires at layer 2 input → must run layers 0+1 (2 extra layers)
  Overhead:   +2 layers of full-sequence forward, then same pruned continuation
"""

import torch
import torch.nn.functional as F

_IMAGE_TOKEN_ID        = 151655
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID   = 151653


class VScoreL2Baseline:
    """v_score_L2 scoring: v_attn_L2 × value_norm_L2.

    Args:
        model:      Qwen2VLWrapper
        score_layer: which layer to hook (default 2)
    """

    def __init__(self, model, score_layer=2):
        self.model       = model
        self.score_layer = score_layer
        lm = model.model.model.language_model
        cfg = lm.config
        self.num_heads    = getattr(cfg, "num_attention_heads", 28)
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", 4)
        self.head_dim     = getattr(cfg, "hidden_size", 3584) // self.num_heads

    @torch.no_grad()
    def score_tokens(self, inputs):
        """Compute per-visual-token v_score_L{score_layer}.

        Returns:
            list of floats, length = num_visual_tokens
        """
        lm = self.model.model.model.language_model
        captured = {}

        def _pre_hook(module, args):
            if isinstance(args, tuple) and len(args) > 0:
                captured["hidden"] = args[0].detach()

        handle = lm.layers[self.score_layer].register_forward_pre_hook(_pre_hook)
        try:
            self.model.model(
                **inputs,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
        finally:
            handle.remove()

        if "hidden" not in captured:
            raise RuntimeError("Pre-hook did not fire")

        hidden = captured["hidden"]
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        attn_layer = lm.layers[self.score_layer].self_attn
        dev = next(attn_layer.q_proj.parameters()).device
        hidden = hidden.to(dev)

        ids = inputs["input_ids"][0].to(dev)
        text_mask = ~((ids == _IMAGE_TOKEN_ID) |
                      (ids == _VISION_START_TOKEN_ID) |
                      (ids == _VISION_END_TOKEN_ID))
        vis_mask  = (ids == _IMAGE_TOKEN_ID)

        text_pos = torch.where(text_mask)[0]
        vis_pos  = torch.where(vis_mask)[0]
        nv = len(vis_pos)

        if len(text_pos) == 0 or nv == 0:
            return [0.0] * nv

        h_text = hidden[:, text_pos, :]
        h_vis  = hidden[:, vis_pos,  :]

        # v_attn: text→vis attention
        q = attn_layer.q_proj(h_text).view(1, len(text_pos), self.num_heads, self.head_dim).transpose(1, 2)
        k = attn_layer.k_proj(h_vis ).view(1, nv, self.num_kv_heads, self.head_dim).transpose(1, 2)
        rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(rep, dim=1)

        scale = self.head_dim ** -0.5
        a = F.softmax(torch.matmul(q, k.transpose(-2, -1)).float() * scale, dim=-1)
        # [1, nh, n_text, nv] → mean over text → mean over heads → [nv]
        v_attn = a[0].mean(dim=1).mean(dim=0)  # [nv]

        # v_vnorm: ||W_V h_vis|| mean over heads
        v = attn_layer.v_proj(h_vis).view(1, nv, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.repeat_interleave(rep, dim=1)
        v_norm = v[0].float().norm(dim=-1).mean(dim=0)  # [nv]

        v_score = (v_attn * v_norm).cpu()
        return v_score.tolist()

    def allocate(self, token_importance, num_vis, budget_frac):
        """Binary keep/drop by v_score rank."""
        keep_count = max(1, int(num_vis * budget_frac))
        indexed = sorted(enumerate(token_importance), key=lambda x: x[1], reverse=True)
        keep_set = {idx for idx, _ in indexed[:keep_count]}
        return [i in keep_set for i in range(num_vis)]
