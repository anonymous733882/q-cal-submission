"""HAWK paper control with local calibrated head weights."""

from pathlib import Path
import torch
import torch.nn.functional as F

from baselines.hawk.adapter import HAWKBaseline

DUALSIGNAL_ROOT = Path(__file__).resolve().parents[2] / "qcal"

class HAWKPaper(HAWKBaseline):
    """HAWK wrapper that reads calibrated weights from dualsignal."""

    def __init__(self, model, prune_ratio=0.5, weights_path: str | None = None):
        self.weights_path = Path(weights_path) if weights_path else (
            DUALSIGNAL_ROOT / "runners/calibration"
            / "hawk_qwen25vl_head_weights.json"
        )
        super().__init__(model, prune_ratio=prune_ratio)

    def _load_head_weights(self):
        import json

        if self.weights_path.exists():
            data = json.loads(self.weights_path.read_text())
            weights = data.get("weights", [])
            if len(weights) == self.num_heads:
                w = torch.tensor(weights, dtype=torch.float32)
                if w.sum().abs() > 1e-9:
                    return w / w.sum()
        return torch.ones(self.num_heads) / self.num_heads

    @torch.no_grad()
    def score_heads(self, inputs):
        """Return per-head text-guided visual scores, shape [heads, n_vis]."""
        lm = self.model.model.model.language_model
        captured = {}

        def _pre_hook(module, args):
            if isinstance(args, tuple) and len(args) > 0:
                captured["hidden"] = args[0].detach()

        handle = lm.layers[0].register_forward_pre_hook(_pre_hook)
        try:
            _ = self.model.model(
                **inputs,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        finally:
            handle.remove()

        hidden = captured.get("hidden")
        if hidden is None:
            raise RuntimeError("HAWK layer-0 pre-hook did not capture hidden states")
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        attn_layer = lm.layers[0].self_attn
        device = next(attn_layer.q_proj.parameters()).device
        hidden = hidden.to(device)
        ids = inputs["input_ids"][0].to(device)

        text_mask = ~((ids == _IMAGE_TOKEN_ID) |
                      (ids == _VISION_START_ID) |
                      (ids == _VISION_END_ID))
        vis_mask = ids == _IMAGE_TOKEN_ID
        text_positions = torch.where(text_mask)[0]
        vis_positions = torch.where(vis_mask)[0]
        if len(text_positions) == 0 or len(vis_positions) == 0:
            return torch.zeros(self.num_heads, len(vis_positions))

        h_normed = lm.layers[0].input_layernorm(hidden)
        h_text = h_normed[:, text_positions, :]
        h_vis = h_normed[:, vis_positions, :]

        q = attn_layer.q_proj(h_text)
        k = attn_layer.k_proj(h_vis)
        num_text = len(text_positions)
        num_vis = len(vis_positions)
        q = q.view(1, num_text, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, num_vis, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn.float(), dim=-1)
        return attn[0].mean(dim=1).cpu()

    @torch.no_grad()
    def score_tokens(self, inputs):
        heads = self.score_heads(inputs)
        if heads.numel() == 0:
            return []
        weights = self.head_weights.to(heads.device).float()
        return (weights.unsqueeze(1) * heads).sum(dim=0).cpu().tolist()
