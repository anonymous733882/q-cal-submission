"""Shared utilities for computing attention with layernorm + M-RoPE.

Used by all baseline scoring functions that manually compute Q·K^T
to ensure correct pre-norm + positional encoding application.
"""

import torch
import torch.nn.functional as F


def _attention_shape(attn_l, lm):
    """Return (num_heads, num_kv_heads, head_dim) across HF model variants."""
    n_heads = getattr(attn_l, "num_heads", None)
    if n_heads is None:
        n_heads = getattr(attn_l, "num_attention_heads", None)
    if n_heads is None:
        n_heads = getattr(getattr(lm, "config", None), "num_attention_heads", None)

    n_kv_heads = getattr(attn_l, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = getattr(getattr(lm, "config", None), "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = n_heads

    head_dim = getattr(attn_l, "head_dim", None)
    if head_dim is None and n_heads is not None:
        head_dim = attn_l.q_proj.out_features // int(n_heads)

    if n_heads is None or n_kv_heads is None or head_dim is None:
        raise AttributeError("Cannot infer attention heads/head_dim for layer")
    return int(n_heads), int(n_kv_heads), int(head_dim)


def prepare_rope(lm, inputs, device):
    """Pre-compute M-RoPE position embeddings from inputs.

    Returns:
        (cos, sin, mrope_section) or (None, None, None) if unavailable.
        cos/sin shape: [3, batch, seq_len, head_dim]
    """
    position_ids = inputs.get("position_ids")
    mrope_section = getattr(lm.config, "rope_parameters", {}).get(
        "mrope_section", None)
    if position_ids is None or mrope_section is None:
        return None, None, None

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        position_ids_rope = position_ids[1:]  # drop text_position_ids dim
    elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
        position_ids_rope = position_ids
    else:
        return None, None, None

    seq_len = position_ids_rope.shape[2]
    dummy = torch.zeros(1, seq_len, 1, device=device,
                        dtype=next(lm.parameters()).dtype)
    cos, sin = lm.rotary_emb(dummy, position_ids_rope.to(device))
    return cos.to(device), sin.to(device), mrope_section


def apply_rope_to_qk(q, k, cos, sin, mrope_section, q_pos_idx, k_pos_idx):
    """Apply M-RoPE to Q and K tensors using position index slicing.

    Args:
        q: [batch, n_heads, n_q, head_dim]
        k: [batch, n_heads, n_k, head_dim]
        cos, sin: [3, batch, seq_len, head_dim] from prepare_rope
        mrope_section: int, head_dim section size per rope dimension
        q_pos_idx: LongTensor of sequence positions for Q tokens
        k_pos_idx: LongTensor of sequence positions for K tokens

    Returns:
        (q_rotated, k_rotated)
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb)

    cos_q = cos[:, :, q_pos_idx, :]
    sin_q = sin[:, :, q_pos_idx, :]
    cos_k = cos[:, :, k_pos_idx, :]
    sin_k = sin[:, :, k_pos_idx, :]

    q, _ = apply_multimodal_rotary_pos_emb(q, q, cos_q, sin_q, mrope_section)
    _, k = apply_multimodal_rotary_pos_emb(k, k, cos_k, sin_k, mrope_section)
    return q, k


def compute_text_vis_attention(layer, h, text_pos, vis_pos, inputs,
                               lm, apply_rope=True, query_mode="all_text"):
    """Compute text→vis attention scores with layernorm + optional RoPE.

    Args:
        layer: nn.Module — the transformer layer (lm.layers[L])
        h: [1, seq_len, hidden] — hidden states BEFORE this layer
        text_pos: list of int — text token positions
        vis_pos: list of int — visual token positions
        inputs: model inputs dict (for position_ids)
        lm: language_model module
        apply_rope: whether to apply RoPE
        query_mode: "all_text" or "last_text"

    Returns:
        scores: Tensor [n_vis]
    """
    attn_l = layer.self_attn
    n_heads, n_kv_heads, head_dim = _attention_shape(attn_l, lm)
    device = next(attn_l.q_proj.parameters()).device
    h = h.to(device)

    # S1: Apply input_layernorm
    h_normed = layer.input_layernorm(h)

    v_idx = torch.tensor(vis_pos, dtype=torch.long, device=device)

    if query_mode == "last_text":
        t_idx = torch.tensor([text_pos[-1]], dtype=torch.long, device=device)
        n_q = 1
    else:
        t_idx = torch.tensor(text_pos, dtype=torch.long, device=device)
        n_q = len(text_pos)

    q = attn_l.q_proj(h_normed[:, t_idx, :]).view(
        1, n_q, n_heads, head_dim).transpose(1, 2)
    k = attn_l.k_proj(h_normed[:, v_idx, :]).view(
        1, len(vis_pos), n_kv_heads, head_dim).transpose(1, 2)
    if n_heads != n_kv_heads:
        k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)

    # S2: Apply M-RoPE
    if apply_rope:
        cos, sin, mrope_section = prepare_rope(lm, inputs, device)
        if cos is not None:
            q, k = apply_rope_to_qk(q, k, cos, sin, mrope_section,
                                     t_idx, v_idx)

    attn = F.softmax(
        torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5), dim=-1)
    # Mean over heads, mean/squeeze over query dim → [n_vis]
    scores = attn[0].mean(0).mean(0)
    return scores.cpu()
