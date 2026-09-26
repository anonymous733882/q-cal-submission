"""HAWK baseline: text-guided attention scoring with head importance weighting.

Reference: HAWK (arXiv 2604.07812, CVPR 2026)
  1. Compute text→visual attention at first LLM layer (position-agnostic, no RoPE)
  2. Weight by per-head importance (uniform approximation — original requires offline calibration)
  3. Retain top-k visual tokens by importance score
  4. Binary keep/drop, one-shot static pruning

Adapted for Qwen2-VL: uses layer 0's Q/K projections.
"""

import torch
import torch.nn.functional as F

_IMAGE_TOKEN_ID = 151655
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653


class HAWKBaseline:
    """HAWK-style pruning: head-weighted text-guided attention at layer 0.

    Args:
        model: Qwen2VLWrapper
        prune_ratio: fraction of visual tokens to DROP (default 0.5)
    """

    def __init__(self, model, prune_ratio=0.5):
        self.model = model
        self.prune_ratio = prune_ratio
        lm = model.model.model.language_model
        cfg = lm.config
        self.num_heads = getattr(cfg, "num_attention_heads", 28)
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", 4)
        self.head_dim = getattr(cfg, "hidden_size", 3584) // self.num_heads
        # Load calibrated head weights if available, else uniform
        self.head_weights = self._load_head_weights()

    def _load_head_weights(self):
        import os, json
        weights_path = os.path.join(os.path.dirname(__file__),
                                    "hawk_head_weights_qwen25vl.json")
        if os.path.exists(weights_path):
            with open(weights_path) as f:
                data = json.load(f)
            w = data.get("weights", [])
            if len(w) == self.num_heads:
                print(f"HAWK: loaded calibrated head weights from {weights_path}",
                      flush=True)
                return torch.tensor(w, dtype=torch.float32)
        # Fallback: uniform
        return torch.ones(self.num_heads) / self.num_heads

    @torch.no_grad()
    def score_tokens(self, inputs):
        """Compute per-visual-token importance scores using HAWK method.

        Returns:
            token_importance: list of floats, length = num_visual_tokens
        """
        lm = self.model.model.model.language_model

        # Hook layer 0 to capture input hidden states
        captured = {}

        def _hook(module, input, output):
            # input[0] is the hidden states entering this layer
            if isinstance(input, tuple):
                captured["hidden"] = input[0].detach()
            else:
                captured["hidden"] = input.detach()

        handle = lm.layers[0].register_forward_hook(_hook, with_kwargs=False)

        # Actually we need the hidden states BEFORE layer 0 (i.e., embedding output)
        # Hook the layer to get its input
        captured2 = {}

        def _pre_hook(module, args):
            if isinstance(args, tuple) and len(args) > 0:
                captured2["hidden"] = args[0].detach()

        handle2 = lm.layers[0].register_forward_pre_hook(_pre_hook)

        try:
            _ = self.model.model(
                **inputs,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
        finally:
            handle.remove()
            handle2.remove()

        if "hidden" in captured2:
            hidden = captured2["hidden"]
        elif "hidden" in captured:
            hidden = captured["hidden"]
        else:
            raise RuntimeError("Hook did not fire")

        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        attn_layer = lm.layers[0].self_attn
        device = next(attn_layer.q_proj.parameters()).device
        hidden = hidden.to(device)

        ids = inputs["input_ids"][0].to(device)

        # Locate text and visual tokens
        text_mask = ~((ids == _IMAGE_TOKEN_ID) |
                      (ids == _VISION_START_TOKEN_ID) |
                      (ids == _VISION_END_TOKEN_ID))
        vis_mask = (ids == _IMAGE_TOKEN_ID)

        text_positions = torch.where(text_mask)[0]
        vis_positions = torch.where(vis_mask)[0]

        if len(text_positions) == 0 or len(vis_positions) == 0:
            return [0.0] * len(vis_positions)

        # S1: Apply input_layernorm (pre-norm architecture)
        h_normed = lm.layers[0].input_layernorm(hidden)

        # Q from text tokens, K from visual tokens
        # HAWK key: NO RoPE — position-agnostic attention
        h_text = h_normed[:, text_positions, :]
        h_vis = h_normed[:, vis_positions, :]

        q = attn_layer.q_proj(h_text)  # [1, num_text, num_heads * head_dim]
        k = attn_layer.k_proj(h_vis)   # [1, num_vis, num_kv_heads * head_dim]

        num_text = len(text_positions)
        num_vis = len(vis_positions)

        q = q.view(1, num_text, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, num_vis, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA expansion
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)

        # Attention scores (no RoPE, no softmax — raw dot product per HAWK)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        # [1, num_heads, num_text, num_vis]

        # Softmax along visual dimension
        attn = F.softmax(attn.float(), dim=-1)

        # Average across text tokens → c_k^i: [num_heads, num_vis]
        c = attn[0].mean(dim=1)  # [num_heads, num_vis]

        # Weighted sum across heads: I_k = Σ w_i * c_k^i
        w = self.head_weights.to(device).float()  # [num_heads]
        importance = (w.unsqueeze(1) * c).sum(dim=0)  # [num_vis]

        del q, k, attn, c

        return importance.cpu().tolist()

    def allocate(self, token_importance, num_vis, budget_frac):
        """Binary keep/drop allocation based on importance scores.

        Args:
            token_importance: list of floats per visual token
            num_vis: total visual tokens
            budget_frac: fraction to KEEP (0.0-1.0)

        Returns:
            keep_mask: list of bool, True = keep
        """
        keep_count = max(1, int(num_vis * budget_frac))
        indexed = sorted(enumerate(token_importance), key=lambda x: x[1], reverse=True)
        keep_set = set(idx for idx, _ in indexed[:keep_count])
        return [i in keep_set for i in range(num_vis)]
