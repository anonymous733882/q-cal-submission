"""Layer-attention policy control used by policy comparisons."""

import torch

from baselines.common.attn_utils import compute_text_vis_attention

_IMAGE_TOKEN_ID = 151655
_VISION_START_ID = 151652
_VISION_END_ID = 151653

def _norm01(values: torch.Tensor) -> torch.Tensor:
    values = values.float().cpu()
    if values.numel() == 0:
        return values
    lo, hi = values.min(), values.max()
    if (hi - lo).abs() <= 1e-9:
        return torch.zeros_like(values)
    return (values - lo) / (hi - lo)


class LayerAttentionPolicyPruner:
    """Weighted text-to-visual attention policy over selected LLM layers."""

    def __init__(self, model, layers: list[int], weights: list[float]):
        if len(layers) != len(weights) or not layers:
            raise ValueError("layers and weights must be non-empty and aligned")
        total = float(sum(max(0.0, float(w)) for w in weights))
        if total <= 1e-12:
            raise ValueError("policy weights sum to zero")
        self.model = model
        self.layers = [int(x) for x in layers]
        self.weights = [max(0.0, float(w)) / total for w in weights]

    @torch.no_grad()
    def score_tokens(self, inputs):
        lm = self.model.model.model.language_model
        ids = inputs["input_ids"][0]
        vs = (ids == _VISION_START_ID).nonzero(as_tuple=True)[0]
        ve = (ids == _VISION_END_ID).nonzero(as_tuple=True)[0]
        if len(vs) > 0 and len(ve) > 0:
            start, end = vs[0].item(), ve[0].item()
            vis_pos = [i for i in range(start + 1, end) if ids[i] == _IMAGE_TOKEN_ID]
            text_pos = [i for i in range(ids.shape[0]) if i < start or i > end]
        else:
            vis_pos, text_pos = [], []
        if not vis_pos or not text_pos:
            return []

        captured = {}
        handles = []

        def make_hook(layer_idx: int):
            def _pre_hook(module, args):
                if isinstance(args, tuple) and len(args) > 0:
                    h = args[0]
                    captured[layer_idx] = (
                        h.detach() if h.dim() == 3 else h.unsqueeze(0).detach()
                    )
            return _pre_hook

        try:
            for layer_idx in self.layers:
                handles.append(lm.layers[layer_idx].register_forward_pre_hook(
                    make_hook(layer_idx)))
            self.model.model(
                **inputs,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()

        fused = None
        for layer_idx, weight in zip(self.layers, self.weights):
            h = captured.get(layer_idx)
            if h is None:
                continue
            scores = compute_text_vis_attention(
                lm.layers[layer_idx],
                h,
                text_pos,
                vis_pos,
                inputs,
                lm,
                apply_rope=True,
                query_mode="last_text",
            )
            scores = _norm01(scores)
            fused = scores * weight if fused is None else fused + scores * weight
        if fused is None:
            return [0.0 for _ in vis_pos]
        return fused.cpu().tolist()

    @torch.no_grad()
    def allocate(self, scores, n_vis: int, budget_frac: float):
        n_keep = max(1, min(n_vis, int(n_vis * budget_frac)))
        tensor = torch.tensor(scores, dtype=torch.float32)
        if tensor.shape[0] != n_vis:
            padded = torch.zeros(n_vis)
            m = min(n_vis, tensor.shape[0])
            padded[:m] = tensor[:m]
            tensor = padded
        keep = set(tensor.argsort(descending=True)[:n_keep].tolist())
        return [i in keep for i in range(n_vis)]

    @torch.no_grad()
    def prune(self, model_wrapper, inputs, vis_embeds, grid_thw, prune_ratio=None):
        ratio = 0.9 if prune_ratio is None else prune_ratio
        budget_frac = max(0.0, min(1.0, 1.0 - ratio))
        n_vis = int(vis_embeds.shape[0])
        return self.allocate(self.score_tokens(inputs), n_vis, budget_frac)
