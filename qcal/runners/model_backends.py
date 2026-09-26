"""Model backend adapters for from-scratch attention-policy runners."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
import os
import math


DUALSIGNAL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = DUALSIGNAL_ROOT.parent
BEA_DIR = WORKSPACE / "qcal_support"
sys.path.insert(0, str(BEA_DIR))
sys.path.insert(0, str(BEA_DIR / "strategy_test"))
sys.path.insert(0, str(WORKSPACE))

from baselines.common.attn_utils import compute_text_vis_attention  # noqa: E402
from models.qwen3vl_wrapper import Qwen2VLWrapper, _load_model_cls  # noqa: E402
from models.token_manipulator_v2 import TokenManipulatorV2  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from transformers.modeling_outputs import BaseModelOutputWithPast  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
import functools


def deterministic_generation_kwargs() -> dict[str, Any]:
    return {"do_sample": False, "num_beams": 1}


@contextmanager
def deterministic_generation_config(model):
    config = getattr(model, "generation_config", None)
    keys = deterministic_generation_kwargs()
    old = {}
    if config is not None:
        for key, value in keys.items():
            if hasattr(config, key):
                old[key] = getattr(config, key)
                setattr(config, key, value)
    try:
        yield
    finally:
        if config is not None:
            for key, value in old.items():
                setattr(config, key, value)


def _compute_pos_embeddings(lm, position_ids, hidden_for_shape, device, dtype):
    if hasattr(lm, "rotary_emb") and position_ids is not None:
        dummy = torch.zeros(
            1,
            position_ids.shape[-1],
            lm.config.hidden_size,
            device=device,
            dtype=dtype,
        )
        cos, sin = lm.rotary_emb(dummy, position_ids.to(device))
        return {"position_embeddings": (cos, sin)}
    if position_ids is not None:
        return {"position_ids": position_ids.to(device)}
    return {}


def _clone_cache(src):
    try:
        from transformers.cache_utils import DynamicCache

        dst = DynamicCache()
        for layer in src.layers:
            dst.update(layer.keys.clone(), layer.values.clone(), layer_idx=len(dst.layers))
        return dst
    except Exception:
        cloned = []
        for item in src:
            if isinstance(item, tuple):
                cloned.append(tuple(x.clone() if torch.is_tensor(x) else x for x in item))
            else:
                cloned.append(item)
        return tuple(cloned)


def _cache_layers(cache):
    return cache.layers if hasattr(cache, "layers") else cache


def _cache_kv_at(cache, layer_idx: int):
    layer = _cache_layers(cache)[layer_idx]
    if hasattr(layer, "keys"):
        return layer.keys, layer.values
    return layer[0], layer[1]


def _cache_seq_len(cache) -> int:
    k, _ = _cache_kv_at(cache, 0)
    return int(k.shape[2])


def _project_query_states(lm, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
    layer = lm.layers[layer_idx]
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    if attn is None:
        raise RuntimeError(f"cannot find attention module at layer {layer_idx}")
    ln = (
        getattr(layer, "input_layernorm", None)
        or getattr(layer, "attention_norm", None)
        or getattr(layer, "ln_1", None)
    )
    h = hidden[0]
    if ln is not None:
        h = ln(h)
    cfg = getattr(attn, "config", getattr(lm, "config", None))
    if hasattr(attn, "wqkv"):
        n_heads = int(getattr(cfg, "num_attention_heads"))
        n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
        head_dim = int(getattr(cfg, "hidden_size")) // n_heads
        q_raw = attn.wqkv(h).float()[:, : n_heads * head_dim]
    else:
        n_heads = int(getattr(attn, "num_heads", None) or getattr(cfg, "num_attention_heads"))
        n_kv_heads = int(getattr(attn, "num_key_value_heads", None) or getattr(cfg, "num_key_value_heads", n_heads))
        head_dim = int(getattr(attn, "head_dim", None) or (getattr(cfg, "hidden_size") // n_heads))
        q_raw = attn.q_proj(h).float()
    q = q_raw.view(h.shape[0], n_heads, head_dim)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    return q.permute(1, 0, 2).contiguous(), n_heads, n_kv_heads, head_dim


@torch.no_grad()
def _importance_from_hidden_and_cache(lm, layer_idx, q_hidden, cache, vis_indices_in_cache):
    if not vis_indices_in_cache:
        return torch.empty(0, dtype=torch.float32)
    q, n_heads, n_kv_heads, head_dim = _project_query_states(lm, layer_idx, q_hidden)
    k_cache, _ = _cache_kv_at(cache, layer_idx)
    vis_t = torch.tensor(vis_indices_in_cache, device=k_cache.device, dtype=torch.long)
    k_vis = k_cache[0, :, vis_t, :].float()
    if k_vis.shape[0] < n_heads:
        k_vis = k_vis.repeat_interleave(n_heads // k_vis.shape[0], dim=0)
    scores = (q.to(k_vis.device) @ k_vis.transpose(-1, -2)) * (head_dim ** -0.5)
    weights = F.softmax(scores.float(), dim=-1)
    return weights.mean(dim=(0, 1)).detach().cpu()


def _prune_cache_positions(cache, keep_positions, device):
    from transformers.cache_utils import DynamicCache

    keep_t = torch.tensor(keep_positions, device=device, dtype=torch.long)
    pruned = DynamicCache()
    for layer_idx in range(len(_cache_layers(cache))):
        k, v = _cache_kv_at(cache, layer_idx)
        pruned.update(
            k[:, :, keep_t, :].contiguous(),
            v[:, :, keep_t, :].contiguous(),
            layer_idx=layer_idx,
        )
    return pruned


@torch.no_grad()
def _prefill_all_layers(lm, shared_embeds, position_ids=None):
    from transformers.cache_utils import DynamicCache

    device = shared_embeds.device
    dtype = shared_embeds.dtype
    pos_kwargs = _compute_pos_embeddings(lm, position_ids, shared_embeds, device, dtype)
    cache = DynamicCache()
    h = shared_embeds
    for layer in lm.layers:
        out = layer(
            hidden_states=h,
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
            **pos_kwargs,
        )
        h = out[0] if isinstance(out, tuple) else out
    return cache


@torch.no_grad()
def _generate_ids_sparsevila_from_shared_cache(
    lm,
    lm_head,
    shared_cache,
    vis_indices_in_cache,
    decode_keep_ratio,
    question_embeds,
    question_pos_ids,
    eos_id,
    max_new_tokens: int,
):
    device = question_embeds.device
    dtype = question_embeds.dtype
    cache = _clone_cache(shared_cache)
    n_vis = len(vis_indices_in_cache)
    n_decode_keep = max(1, min(n_vis, int(round(n_vis * float(decode_keep_ratio)))))
    agg_salience = torch.zeros(n_vis, dtype=torch.float64)
    shared_len = _cache_seq_len(cache)
    out = lm(
        inputs_embeds=question_embeds,
        attention_mask=torch.ones((1, shared_len + question_embeds.shape[1]), dtype=torch.long, device=device),
        position_ids=question_pos_ids,
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    for layer_idx, hidden in enumerate(out.hidden_states[:-1]):
        agg_salience += _importance_from_hidden_and_cache(
            lm, layer_idx, hidden, shared_cache, vis_indices_in_cache).double()
    cache = out.past_key_values
    logits = lm_head(out.last_hidden_state[:, -1:, :])
    next_id = logits.argmax(dim=-1).reshape(1, 1)
    generated = [int(next_id.item())]
    if generated[-1] == eos_id or max_new_tokens <= 1:
        return torch.tensor(generated, device=device, dtype=torch.long), {
            "decode_stage_kv_retrieval_implemented": True,
            "decode_keep_ratio": float(decode_keep_ratio),
            "decode_visual_tokens": int(n_decode_keep),
            "prefill_visual_tokens": int(n_vis),
        }

    top_local = agg_salience.argsort(descending=True)[:n_decode_keep].tolist()
    keep_vis = {int(vis_indices_in_cache[int(i)]) for i in top_local}
    vis_set = set(int(i) for i in vis_indices_in_cache)
    seq_len = _cache_seq_len(cache)
    keep_positions = [p for p in range(seq_len) if p not in vis_set or p in keep_vis]
    cache = _prune_cache_positions(cache, keep_positions, device)

    if question_pos_ids is not None:
        last_pos = question_pos_ids[..., -1:] + 1
    else:
        last_pos = torch.tensor([[question_embeds.shape[1]]], device=device)

    embed_tokens = lm.get_input_embeddings() if hasattr(lm, "get_input_embeddings") else lm.embed_tokens
    for _ in range(max_new_tokens - 1):
        token_emb = embed_tokens(next_id).to(dtype)
        out = lm(
            inputs_embeds=token_emb,
            attention_mask=torch.ones((1, _cache_seq_len(cache) + 1), dtype=torch.long, device=device),
            position_ids=last_pos,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        logits = lm_head(out.last_hidden_state[:, -1:, :])
        next_id = logits.argmax(dim=-1).reshape(1, 1)
        token = int(next_id.item())
        generated.append(token)
        if token == eos_id:
            break
        last_pos = last_pos + 1

    return torch.tensor(generated, device=device, dtype=torch.long), {
        "decode_stage_kv_retrieval_implemented": True,
        "decode_keep_ratio": float(decode_keep_ratio),
        "decode_visual_tokens": int(n_decode_keep),
        "prefill_visual_tokens": int(n_vis),
        "compacted_cache_seq_len": int(len(keep_positions)),
    }


@torch.no_grad()
def _generate_ids_from_shared_cache(
    lm,
    lm_head,
    shared_cache,
    question_embeds,
    question_pos_ids,
    eos_id,
    max_new_tokens: int,
):
    device = question_embeds.device
    dtype = question_embeds.dtype
    cache = _clone_cache(shared_cache)
    pos_kwargs = _compute_pos_embeddings(lm, question_pos_ids, question_embeds, device, dtype)
    h = question_embeds
    for layer in lm.layers:
        out = layer(
            hidden_states=h,
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
            **pos_kwargs,
        )
        h = out[0] if isinstance(out, tuple) else out

    h = lm.norm(h)
    logits = lm_head(h[:, -1:, :])
    next_id = logits.argmax(dim=-1).reshape(1, 1)
    generated = [int(next_id.item())]
    if generated[-1] == eos_id or max_new_tokens <= 1:
        return torch.tensor(generated, device=device, dtype=torch.long)

    if question_pos_ids is not None:
        last_pos = question_pos_ids[..., -1:] + 1
    else:
        last_pos = torch.tensor([[question_embeds.shape[1]]], device=device)

    embed_tokens = lm.get_input_embeddings() if hasattr(lm, "get_input_embeddings") else lm.embed_tokens
    for _ in range(max_new_tokens - 1):
        token_emb = embed_tokens(next_id).to(dtype)
        gen_pos_kwargs = _compute_pos_embeddings(lm, last_pos, token_emb, device, dtype)
        h = token_emb
        for layer in lm.layers:
            out = layer(
                hidden_states=h,
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
                **gen_pos_kwargs,
            )
            h = out[0] if isinstance(out, tuple) else out
        h = lm.norm(h)
        logits = lm_head(h[:, -1:, :])
        next_id = logits.argmax(dim=-1).reshape(1, 1)
        token = int(next_id.item())
        generated.append(token)
        if token == eos_id:
            break
        last_pos = last_pos + 1
    return torch.tensor(generated, device=device, dtype=torch.long)


def qwen3_pyramiddrop_rank_fn(hidden_states, vis_positions, text_positions,
                              layer_idx, attn_layer, *, layer_module=None,
                              position_embeddings=None, mrope_section=None):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    dev = next(attn_layer.q_proj.parameters()).device
    h = hidden_states.to(dev)
    if layer_module is not None:
        h = layer_module.input_layernorm(h)
    last_text = text_positions[-1] if text_positions else 0
    vis_idx = torch.tensor(vis_positions, dtype=torch.long, device=dev)
    h_query = h[:, last_text:last_text + 1, :]
    h_vis = h[:, vis_idx, :]

    head_dim = attn_layer.head_dim
    num_heads = attn_layer.config.num_attention_heads
    num_kv_heads = attn_layer.config.num_key_value_heads
    q = attn_layer.q_proj(h_query).view(1, 1, num_heads, head_dim)
    k = attn_layer.k_proj(h_vis).view(1, len(vis_positions), num_kv_heads, head_dim)
    if hasattr(attn_layer, "q_norm"):
        q = attn_layer.q_norm(q)
    if hasattr(attn_layer, "k_norm"):
        k = attn_layer.k_norm(k)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    if num_heads != num_kv_heads:
        k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        cos = cos.to(dev)
        sin = sin.to(dev)
        q_pos = torch.tensor([last_text], dtype=torch.long, device=dev)
        if cos.dim() == 4:
            cos_q, sin_q = cos[:, :, q_pos, :], sin[:, :, q_pos, :]
            cos_k, sin_k = cos[:, :, vis_idx, :], sin[:, :, vis_idx, :]
        else:
            cos_q, sin_q = cos[:, q_pos, :], sin[:, q_pos, :]
            cos_k, sin_k = cos[:, vis_idx, :], sin[:, vis_idx, :]
        q, _ = apply_rotary_pos_emb(q, q, cos_q, sin_q)
        _, k = apply_rotary_pos_emb(k, k, cos_k, sin_k)

    attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
    scores = F.softmax(attn.float(), dim=-1)[0].mean(dim=0).squeeze(0)
    order = scores.argsort()
    return [vis_positions[idx.item()] for idx in order]


def qwen3_fitprune_rank_fn(hidden_states, vis_positions, text_positions,
                           layer_idx, attn_layer, *, layer_module=None,
                           position_embeddings=None, mrope_section=None):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    dev = next(attn_layer.q_proj.parameters()).device
    h = hidden_states.to(dev)
    if layer_module is not None:
        h = layer_module.input_layernorm(h)
    vis_idx = torch.tensor(vis_positions, dtype=torch.long, device=dev)
    h_vis = h[:, vis_idx, :]
    n_vis = len(vis_positions)
    head_dim = attn_layer.head_dim
    num_heads = attn_layer.config.num_attention_heads
    num_kv_heads = attn_layer.config.num_key_value_heads

    q_vis = attn_layer.q_proj(h_vis).view(1, n_vis, num_heads, head_dim)
    k_vis = attn_layer.k_proj(h_vis).view(1, n_vis, num_kv_heads, head_dim)
    if hasattr(attn_layer, "q_norm"):
        q_vis = attn_layer.q_norm(q_vis)
    if hasattr(attn_layer, "k_norm"):
        k_vis = attn_layer.k_norm(k_vis)
    q_vis = q_vis.transpose(1, 2)
    k_vis = k_vis.transpose(1, 2)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        cos = cos.to(dev)
        sin = sin.to(dev)
        if cos.dim() == 4:
            cos_v, sin_v = cos[:, :, vis_idx, :], sin[:, :, vis_idx, :]
        else:
            cos_v, sin_v = cos[:, vis_idx, :], sin[:, vis_idx, :]
        q_vis, k_vis = apply_rotary_pos_emb(q_vis, k_vis, cos_v, sin_v)

    if num_heads != num_kv_heads:
        k_vis = k_vis.repeat_interleave(num_heads // num_kv_heads, dim=1)
    scale = head_dim ** -0.5
    self_scores = torch.matmul(q_vis, k_vis.transpose(-2, -1)) * scale
    self_scores = F.softmax(self_scores.float(), dim=-1)
    self_attn = self_scores[0].max(dim=0).values.sum(dim=0)

    if text_positions:
        text_idx = torch.tensor(text_positions, dtype=torch.long, device=dev)
        h_text = h[:, text_idx, :]
        n_text = len(text_positions)
        q_text = attn_layer.q_proj(h_text).view(1, n_text, num_heads, head_dim)
        if hasattr(attn_layer, "q_norm"):
            q_text = attn_layer.q_norm(q_text)
        q_text = q_text.transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            if cos.dim() == 4:
                cos_t, sin_t = cos[:, :, text_idx.to(cos.device), :], sin[:, :, text_idx.to(sin.device), :]
            else:
                cos_t, sin_t = cos[:, text_idx.to(cos.device), :], sin[:, text_idx.to(sin.device), :]
            q_text, _ = apply_rotary_pos_emb(q_text, q_text, cos_t.to(dev), sin_t.to(dev))
        k_cross = k_vis
        cross_scores = torch.matmul(q_text, k_cross.transpose(-2, -1)) * scale
        cross_scores = F.softmax(cross_scores.float(), dim=-1)
        cross_attn = cross_scores[0].max(dim=0).values.mean(dim=0)
    else:
        cross_attn = torch.ones(n_vis, device=dev)

    combined = self_attn * cross_attn
    order = combined.argsort()
    return [vis_positions[idx.item()] for idx in order]


def qwen_rotary_cross_attention_scores(
    hidden_states,
    query_positions,
    key_positions,
    attn_layer,
    *,
    layer_module=None,
    position_embeddings=None,
    mrope_section=None,
) -> torch.Tensor:
    if not query_positions or not key_positions:
        return torch.empty(0)
    dev = next(attn_layer.q_proj.parameters()).device
    h = hidden_states.to(dev)
    if layer_module is not None:
        h = layer_module.input_layernorm(h)
    q_idx = torch.tensor(query_positions, dtype=torch.long, device=dev)
    k_idx = torch.tensor(key_positions, dtype=torch.long, device=dev)
    h_q = h[:, q_idx, :]
    h_k = h[:, k_idx, :]
    cfg = getattr(attn_layer, "config", None)
    head_dim = getattr(attn_layer, "head_dim", None) or (attn_layer.q_proj.out_features // getattr(cfg, "num_attention_heads"))
    num_heads = getattr(attn_layer, "num_heads", None) or getattr(cfg, "num_attention_heads", None)
    num_kv_heads = getattr(attn_layer, "num_key_value_heads", None) or getattr(cfg, "num_key_value_heads", None)
    q = attn_layer.q_proj(h_q).view(1, len(query_positions), num_heads, head_dim)
    k = attn_layer.k_proj(h_k).view(1, len(key_positions), num_kv_heads, head_dim)
    if hasattr(attn_layer, "q_norm"):
        q = attn_layer.q_norm(q)
    if hasattr(attn_layer, "k_norm"):
        k = attn_layer.k_norm(k)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    if num_heads != num_kv_heads:
        k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        cos = cos.to(dev)
        sin = sin.to(dev)
        if cos.dim() == 4:
            cos_q, sin_q = cos[:, :, q_idx, :], sin[:, :, q_idx, :]
            cos_k, sin_k = cos[:, :, k_idx, :], sin[:, :, k_idx, :]
        else:
            cos_q, sin_q = cos[:, q_idx, :], sin[:, q_idx, :]
            cos_k, sin_k = cos[:, k_idx, :], sin[:, k_idx, :]
        is_qwen3 = "qwen3" in attn_layer.__class__.__module__.lower()
        if mrope_section is not None and not is_qwen3:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
            q, _ = apply_multimodal_rotary_pos_emb(q, q, cos_q, sin_q, mrope_section)
            _, k = apply_multimodal_rotary_pos_emb(k, k, cos_k, sin_k, mrope_section)
        else:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
            q, _ = apply_rotary_pos_emb(q, q, cos_q, sin_q)
            _, k = apply_rotary_pos_emb(k, k, cos_k, sin_k)

    attn = F.softmax((q @ k.transpose(-2, -1)).float() * (head_dim ** -0.5), dim=-1)
    return attn[0].mean(dim=0).mean(dim=0).detach().cpu()


def generic_rotary_attention_rank(
    hidden_states,
    vis_positions,
    text_positions,
    attn_layer,
    layer_module=None,
    position_embeddings=None,
    mode: str = "pyramiddrop",
):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    dev = next(attn_layer.q_proj.parameters()).device
    h = hidden_states.to(dev)
    if layer_module is not None and hasattr(layer_module, "input_layernorm"):
        h = layer_module.input_layernorm(h)
    vis_idx = torch.tensor(vis_positions, dtype=torch.long, device=dev)
    h_vis = h[:, vis_idx, :]
    n_vis = len(vis_positions)
    cfg = getattr(attn_layer, "config", None)
    head_dim = getattr(attn_layer, "head_dim", None) or (attn_layer.q_proj.out_features // getattr(cfg, "num_attention_heads"))
    num_heads = getattr(cfg, "num_attention_heads", attn_layer.q_proj.out_features // head_dim)
    num_kv_heads = getattr(cfg, "num_key_value_heads", attn_layer.k_proj.out_features // head_dim)

    def project_q(x):
        q = attn_layer.q_proj(x).view(1, x.shape[1], num_heads, head_dim)
        if hasattr(attn_layer, "q_norm"):
            q = attn_layer.q_norm(q)
        return q.transpose(1, 2)

    def project_k(x):
        k = attn_layer.k_proj(x).view(1, x.shape[1], num_kv_heads, head_dim)
        if hasattr(attn_layer, "k_norm"):
            k = attn_layer.k_norm(k)
        k = k.transpose(1, 2)
        if num_heads != num_kv_heads:
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
        return k

    def rope(q, k, q_idx, k_idx):
        if position_embeddings is None:
            return q, k
        cos, sin = position_embeddings
        cos = cos.to(dev)
        sin = sin.to(dev)
        if cos.dim() == 4:
            cos_q, sin_q = cos[:, :, q_idx, :], sin[:, :, q_idx, :]
            cos_k, sin_k = cos[:, :, k_idx, :], sin[:, :, k_idx, :]
        else:
            cos_q, sin_q = cos[:, q_idx, :], sin[:, q_idx, :]
            cos_k, sin_k = cos[:, k_idx, :], sin[:, k_idx, :]
        q, _ = apply_rotary_pos_emb(q, q, cos_q, sin_q)
        _, k = apply_rotary_pos_emb(k, k, cos_k, sin_k)
        return q, k

    def rope_q_only(q, q_idx):
        if position_embeddings is None:
            return q
        cos, sin = position_embeddings
        cos = cos.to(dev)
        sin = sin.to(dev)
        if cos.dim() == 4:
            cos_q, sin_q = cos[:, :, q_idx, :], sin[:, :, q_idx, :]
        else:
            cos_q, sin_q = cos[:, q_idx, :], sin[:, q_idx, :]
        q, _ = apply_rotary_pos_emb(q, q, cos_q, sin_q)
        return q

    scale = head_dim ** -0.5
    if mode == "pyramiddrop":
        last_text = text_positions[-1] if text_positions else 0
        q_idx = torch.tensor([last_text], dtype=torch.long, device=dev)
        q = project_q(h[:, last_text:last_text + 1, :])
        k = project_k(h_vis)
        q, k = rope(q, k, q_idx, vis_idx)
        attn = F.softmax((q @ k.transpose(-2, -1)).float() * scale, dim=-1)
        scores = attn[0].mean(dim=0).squeeze(0)
    else:
        q_vis = project_q(h_vis)
        k_vis = project_k(h_vis)
        q_vis, k_vis = rope(q_vis, k_vis, vis_idx, vis_idx)
        self_attn = F.softmax((q_vis @ k_vis.transpose(-2, -1)).float() * scale, dim=-1)
        self_score = self_attn[0].max(dim=0).values.sum(dim=0)
        if text_positions:
            text_idx = torch.tensor(text_positions, dtype=torch.long, device=dev)
            q_text = project_q(h[:, text_idx, :])
            q_text = rope_q_only(q_text, text_idx)
            k_cross = k_vis
            cross = F.softmax((q_text @ k_cross.transpose(-2, -1)).float() * scale, dim=-1)
            cross_score = cross[0].max(dim=0).values.mean(dim=0)
        else:
            cross_score = torch.ones(n_vis, device=dev)
        scores = self_score * cross_score
    return [vis_positions[idx.item()] for idx in scores.argsort()]


def generic_rotary_cross_attention_scores(
    hidden_states,
    query_positions,
    key_positions,
    attn_layer,
    layer_module=None,
    position_embeddings=None,
) -> torch.Tensor:
    """Mean-head attention from query positions to key positions, aggregated per key."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    if not query_positions or not key_positions:
        return torch.empty(0)
    dev = next(attn_layer.q_proj.parameters()).device
    h = hidden_states.to(dev)
    if layer_module is not None and hasattr(layer_module, "input_layernorm"):
        h = layer_module.input_layernorm(h)
    q_idx = torch.tensor(query_positions, dtype=torch.long, device=dev)
    k_idx = torch.tensor(key_positions, dtype=torch.long, device=dev)
    h_q = h[:, q_idx, :]
    h_k = h[:, k_idx, :]
    cfg = getattr(attn_layer, "config", None)
    head_dim = getattr(attn_layer, "head_dim", None) or (attn_layer.q_proj.out_features // getattr(cfg, "num_attention_heads"))
    num_heads = getattr(cfg, "num_attention_heads", attn_layer.q_proj.out_features // head_dim)
    num_kv_heads = getattr(cfg, "num_key_value_heads", attn_layer.k_proj.out_features // head_dim)

    q = attn_layer.q_proj(h_q).view(1, len(query_positions), num_heads, head_dim)
    k = attn_layer.k_proj(h_k).view(1, len(key_positions), num_kv_heads, head_dim)
    if hasattr(attn_layer, "q_norm"):
        q = attn_layer.q_norm(q)
    if hasattr(attn_layer, "k_norm"):
        k = attn_layer.k_norm(k)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    if num_heads != num_kv_heads:
        k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        cos = cos.to(dev)
        sin = sin.to(dev)
        if cos.dim() == 4:
            cos_q, sin_q = cos[:, :, q_idx, :], sin[:, :, q_idx, :]
            cos_k, sin_k = cos[:, :, k_idx, :], sin[:, :, k_idx, :]
        else:
            cos_q, sin_q = cos[:, q_idx, :], sin[:, q_idx, :]
            cos_k, sin_k = cos[:, k_idx, :], sin[:, k_idx, :]
        q, _ = apply_rotary_pos_emb(q, q, cos_q, sin_q)
        _, k = apply_rotary_pos_emb(k, k, cos_k, sin_k)

    attn = F.softmax((q @ k.transpose(-2, -1)).float() * (head_dim ** -0.5), dim=-1)
    return attn[0].mean(dim=0).mean(dim=0).detach().cpu()


def effective_rank(hidden: torch.Tensor) -> float:
    h = hidden.float()
    try:
        s = torch.linalg.svdvals(h)
    except Exception:
        return float(h.shape[0])
    s = s[s > 1e-9]
    if s.numel() == 0:
        return 1.0
    p = s / s.sum()
    entropy = -(p * (p + 1e-12).log()).sum().item()
    return math.exp(entropy)


def recycle_visual_hidden(
    hidden_states: torch.Tensor,
    keep_positions: list[int],
    drop_positions: list[int],
    drop_scores: torch.Tensor,
) -> torch.Tensor:
    if not drop_positions:
        return hidden_states
    dev = hidden_states.device
    keep_idx = torch.tensor(keep_positions, dtype=torch.long, device=dev)
    drop_idx = torch.tensor(drop_positions, dtype=torch.long, device=dev)
    kept = hidden_states[0, keep_idx, :].float()
    dropped = hidden_states[0, drop_idx, :].float()
    sim = F.normalize(dropped, dim=-1) @ F.normalize(kept, dim=-1).T
    nearest = sim.argmax(dim=1)
    weights = drop_scores.to(dev).float().clamp_min(0)
    updated = hidden_states.clone()
    for local_keep in nearest.unique().tolist():
        members = (nearest == local_keep).nonzero(as_tuple=True)[0]
        base_weight = torch.tensor(1.0, device=dev)
        member_weights = weights[members]
        total = base_weight + member_weights.sum()
        merged = kept[local_keep] * (base_weight / total)
        for member, weight in zip(members.tolist(), member_weights):
            merged = merged + dropped[member] * (weight / total)
        updated[0, keep_idx[local_keep], :] = merged.to(updated.dtype)
    return updated


def norm01_t(values: torch.Tensor) -> torch.Tensor:
    values = values.float().cpu()
    if values.numel() == 0:
        return values
    lo, hi = values.min(), values.max()
    if (hi - lo).abs() <= 1e-9:
        return torch.zeros_like(values)
    return (values - lo) / (hi - lo)


@dataclass
class BackendInputs:
    raw: dict[str, torch.Tensor]
    full_embeds: torch.Tensor
    visual_embeds: torch.Tensor
    vis_pos: list[int]
    text_pos: list[int]
    extra: dict[str, Any]

    def get(self, key: str, default=None):
        return self.raw.get(key, default)

    def __getitem__(self, key: str):
        return self.raw[key]


class QwenBackend:
    image_token_id = 151655
    vision_start_id = 151652
    vision_end_id = 151653

    def __init__(self, model_path: str):
        attn_impl = os.environ.get("DUALSIGNAL_QWEN_ATTN_IMPLEMENTATION", "").strip()
        device_map = os.environ.get("DUALSIGNAL_QWEN_DEVICE_MAP", "").strip()
        max_memory_raw = os.environ.get("DUALSIGNAL_QWEN_MAX_MEMORY_JSON", "").strip()
        if attn_impl or device_map or max_memory_raw:
            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": device_map or "auto",
            }
            if attn_impl:
                load_kwargs["attn_implementation"] = attn_impl
            if max_memory_raw:
                import json

                max_memory = json.loads(max_memory_raw)
                load_kwargs["max_memory"] = {
                    int(k) if str(k).isdigit() else k: v
                    for k, v in max_memory.items()
                }
            print(f"Loading model from {model_path} with load_kwargs={load_kwargs}...")
            model_cls = _load_model_cls(model_path)
            self.model = model_cls.from_pretrained(model_path, **load_kwargs)
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model.eval()

            class _Wrapper:
                pass

            self.wrapper = _Wrapper()
            self.wrapper.model = self.model
            self.wrapper.processor = self.processor
            print("Model loaded.")
        else:
            self.wrapper = Qwen2VLWrapper(model_path)
            self.model = self.wrapper.model
            self.processor = self.wrapper.processor
        self.lm = self.model.model.language_model
        self.lm_head = self.model.lm_head
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.total_layers = len(self.lm.layers)
        self.eos_id = self.processor.tokenizer.eos_token_id

    def prepare_inputs_any(self, images: list[Image.Image], prompt: str,
                           max_pixels: int | None, multi_max_pixels: int | None):
        content = []
        for image in images:
            entry = {"type": "image", "image": image}
            entry["max_pixels"] = max_pixels if len(images) == 1 else multi_max_pixels
            content.append(entry)
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        return self.processor(
            text=[text], images=image_inputs, return_tensors="pt").to(self.model.device)

    def visual_and_text_positions(self, inputs) -> tuple[list[int], list[int]]:
        ids = inputs["input_ids"][0]
        vs = (ids == self.vision_start_id).nonzero(as_tuple=True)[0]
        ve = (ids == self.vision_end_id).nonzero(as_tuple=True)[0]
        if len(vs) == 0 or len(ve) == 0:
            return [], []
        spans = []
        for s in vs.tolist():
            later = [e for e in ve.tolist() if e > s]
            if later:
                spans.append((s, later[0]))
        vis_pos = [
            i
            for start, end in spans
            for i in range(start + 1, end)
            if ids[i] == self.image_token_id
        ]
        in_vision = set()
        for start, end in spans:
            in_vision.update(range(start, end + 1))
        text_pos = [i for i in range(ids.shape[0]) if i not in in_vision]
        return vis_pos, text_pos

    @torch.no_grad()
    def extract_visual_embeddings(self, inputs):
        vis_out = self.model.model.visual(
            inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
        return vis_out.pooler_output, inputs["image_grid_thw"]

    @torch.no_grad()
    def generate_ids_and_text(self, inputs, max_new_tokens: int):
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **deterministic_generation_kwargs(),
        )
        generated = output_ids[0, inputs["input_ids"].shape[1]:].detach()
        text = self.processor.decode(generated, skip_special_tokens=True).strip()
        return generated, text

    def _build_grad_embeds(self, inputs, answer_ids: torch.Tensor, max_answer_tokens: int):
        input_ids = inputs["input_ids"]
        answer_ids = answer_ids[:max_answer_tokens].to(input_ids.device)
        if answer_ids.numel() == 0:
            raise ValueError("empty generated answer")
        full_ids = torch.cat([input_ids, answer_ids.unsqueeze(0)], dim=1)
        text_embeds = self.lm.get_input_embeddings()(full_ids).detach()
        with torch.no_grad():
            visual_out = self.model.model.visual(
                inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
            visual_base = visual_out.pooler_output.detach()
        visual_embeds = visual_base.clone().requires_grad_(True)
        prompt_text = text_embeds[:, : input_ids.shape[1], :].clone()
        image_mask, _ = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=prompt_text, image_features=visual_embeds)
        prompt_embeds = prompt_text.masked_scatter(image_mask, visual_embeds)
        answer_embeds = text_embeds[:, input_ids.shape[1]:, :]
        inputs_embeds = torch.cat([prompt_embeds, answer_embeds], dim=1)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            extra = torch.ones((attention_mask.shape[0], answer_ids.numel()),
                               dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, extra], dim=1)
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            extra = torch.zeros((mm_token_type_ids.shape[0], answer_ids.numel()),
                                dtype=mm_token_type_ids.dtype, device=mm_token_type_ids.device)
            mm_token_type_ids = torch.cat([mm_token_type_ids, extra], dim=1)
        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "image_grid_thw": inputs.get("image_grid_thw"),
            "video_grid_thw": inputs.get("video_grid_thw"),
            "return_dict": True,
            "use_cache": False,
        }
        return {k: v for k, v in model_inputs.items() if v is not None}, visual_embeds, input_ids.shape[1]

    def gradient_oracle(self, inputs, answer_ids: torch.Tensor, max_answer_tokens: int,
                        score_mode: str = "sensitivity"):
        self.model.zero_grad(set_to_none=True)
        model_inputs, visual_embeds, prompt_len = self._build_grad_embeds(
            inputs, answer_ids, max_answer_tokens)
        answer_len = min(answer_ids.numel(), max_answer_tokens)
        logits = self.model(**model_inputs).logits
        pred_logits = logits[:, prompt_len - 1: prompt_len + answer_len - 1, :].float()
        target = answer_ids[:answer_len].to(pred_logits.device).unsqueeze(0)
        loss = F.cross_entropy(pred_logits.reshape(-1, pred_logits.shape[-1]),
                               target.reshape(-1), reduction="mean")
        loss.backward()
        grad = visual_embeds.grad
        if grad is None:
            raise RuntimeError("visual embedding gradient is None")
        if score_mode == "directional":
            imp = torch.clamp(
                -(grad.float() * visual_embeds.detach().float()).sum(dim=-1),
                min=0.0,
            )
        else:
            imp = grad.float().norm(dim=-1) * visual_embeds.detach().float().norm(dim=-1)
        self.model.zero_grad(set_to_none=True)
        return imp.detach().cpu().tolist()

    @torch.no_grad()
    def layer_attention_scores(self, inputs, layers: list[int], query_mode: str = "last_text"):
        vis_pos, text_pos = self.visual_and_text_positions(inputs)
        if not vis_pos or not text_pos:
            return {layer: [] for layer in layers}
        captured = {}
        handles = []

        def make_hook(layer_idx: int):
            def _pre_hook(module, args):
                if isinstance(args, tuple) and args:
                    h = args[0]
                    captured[layer_idx] = h.detach() if h.dim() == 3 else h.unsqueeze(0).detach()
            return _pre_hook

        try:
            for layer_idx in layers:
                handles.append(self.lm.layers[layer_idx].register_forward_pre_hook(make_hook(layer_idx)))
            self.model(**inputs, output_hidden_states=False, output_attentions=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        out = {}
        for layer_idx in layers:
            h = captured.get(layer_idx)
            if h is None:
                out[layer_idx] = [0.0 for _ in vis_pos]
            else:
                scores = compute_text_vis_attention(
                    self.lm.layers[layer_idx], h, text_pos, vis_pos, inputs,
                    self.lm, apply_rope=True, query_mode=query_mode)
                out[layer_idx] = norm01_t(scores).tolist()
        return out

    @torch.no_grad()
    def layer_head_attention_scores(self, inputs, layers: list[int], query_mode: str = "last_text"):
        if query_mode != "last_text":
            return None
        vis_pos, text_pos = self.visual_and_text_positions(inputs)
        if not vis_pos or not text_pos:
            return {layer: [] for layer in layers}
        try:
            out = self.model(**inputs, output_hidden_states=False, output_attentions=True, return_dict=True)
            attentions = getattr(out, "attentions", None)
        except Exception as exc:
            print(f"head output_attentions unavailable: {type(exc).__name__}: {exc}", flush=True)
            return None
        if not attentions:
            return None
        q_pos = text_pos[-1]
        vis_t = torch.tensor(vis_pos, dtype=torch.long)
        by_layer = {}
        for layer_idx in layers:
            if layer_idx >= len(attentions) or attentions[layer_idx] is None:
                return None
            attn = attentions[layer_idx]
            scores = attn[0, :, q_pos, vis_t.to(attn.device)].float().detach().cpu()
            by_layer[layer_idx] = [norm01_t(scores[h]).tolist() for h in range(scores.shape[0])]
        return by_layer

    @torch.no_grad()
    def hawk_head_scores(self, inputs):
        vis_pos, text_pos = self.visual_and_text_positions(inputs)
        if not vis_pos or not text_pos:
            return torch.empty(0, 0)
        captured = {}

        def _pre_hook(_module, args):
            if isinstance(args, tuple) and args:
                h = args[0]
                captured["hidden"] = h.detach() if h.dim() == 3 else h.unsqueeze(0).detach()

        handle = self.lm.layers[0].register_forward_pre_hook(_pre_hook)
        try:
            self.model(**inputs, output_hidden_states=False, output_attentions=False, return_dict=True)
        finally:
            handle.remove()

        hidden = captured.get("hidden")
        if hidden is None:
            raise RuntimeError("HAWK layer-0 pre-hook did not capture hidden states")
        layer = self.lm.layers[0]
        attn = layer.self_attn
        h = layer.input_layernorm(hidden.to(next(attn.q_proj.parameters()).device))[0]
        text_idx = torch.tensor(text_pos, dtype=torch.long, device=h.device)
        vis_idx = torch.tensor(vis_pos, dtype=torch.long, device=h.device)
        num_heads = getattr(attn, "num_heads", None) or attn.config.num_attention_heads
        num_kv_heads = getattr(attn, "num_key_value_heads", None) or attn.config.num_key_value_heads
        head_dim = getattr(attn, "head_dim", None) or (attn.config.hidden_size // num_heads)
        q = attn.q_proj(h[text_idx]).view(len(text_pos), num_heads, head_dim).permute(1, 0, 2)
        k = attn.k_proj(h[vis_idx]).view(len(vis_pos), num_kv_heads, head_dim).permute(1, 0, 2)
        if num_kv_heads < num_heads:
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=0)
        scores = torch.softmax((q @ k.transpose(-1, -2)).float() * (head_dim ** -0.5), dim=-1)
        return scores.mean(dim=1).detach().cpu()

    def run_pruned(self, inputs, vis_embeds, grid_thw, keep_mask, max_new_tokens: int):
        tok = TokenManipulatorV2()
        merge = self.model.model.visual.spatial_merge_size
        mr = tok.apply_token_mask(inputs, vis_embeds, grid_thw, keep_mask=keep_mask,
                                  spatial_merge_size=merge, config=self.model.config)
        if hasattr(self.wrapper, "generate_with_token_manipulation_v2"):
            with deterministic_generation_config(self.model):
                return self.wrapper.generate_with_token_manipulation_v2(mr, max_new_tokens=max_new_tokens)

        embeds = self._embeds_from_manipulation(mr)
        self.model.model.rope_deltas = mr.get("rope_deltas", None)
        gen_kwargs = {
            "inputs_embeds": embeds,
            "attention_mask": mr["new_attention_mask"],
            "position_ids": mr["position_ids"],
            "max_new_tokens": max_new_tokens,
        }
        gen_kwargs.update(deterministic_generation_kwargs())
        output_ids = self.model.generate(**gen_kwargs)
        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def run_pact(self, inputs, max_new_tokens: int, *, target_token_layer_percent: float,
                 prune_keep_fraction: float = 0.55, cutoff: float = 0.21):
        return self._run_reference_layer_method(
            inputs, max_new_tokens, target_token_layer_percent=target_token_layer_percent,
            method="PACT", prune_keep_fraction=prune_keep_fraction, cutoff=cutoff)

    @torch.no_grad()
    def run_fastv(self, inputs, max_new_tokens: int, *, target_token_layer_percent: float):
        return self._run_reference_layer_method(
            inputs, max_new_tokens, target_token_layer_percent=target_token_layer_percent,
            method="FastV")

    def _run_reference_layer_method(self, inputs, max_new_tokens: int, *,
                                    target_token_layer_percent: float, method: str,
                                    prune_keep_fraction: float = 0.55, cutoff: float = 0.21):
        from baselines.pact.manipulator import PactManipulator
        from baselines.fastv.manipulator import FastVManipulator

        vis_pos, _ = self.visual_and_text_positions(inputs)
        if not vis_pos:
            raise ValueError(f"{method} requires visual tokens in the prompt")
        if not 0 < target_token_layer_percent < 100:
            raise ValueError("Target token-layer percentage must be between 0 and 100")
        n_vis = len(vis_pos)
        target_layer_equivalents = self.total_layers * target_token_layer_percent / 100
        if method == "PACT":
            layer_index = min(4, max(0, math.floor(target_layer_equivalents - 1)))
        elif method == "FastV":
            layer_index = 2
            if target_layer_equivalents <= layer_index:
                raise ValueError("FastV layer-2 cost floor exceeds the requested token-layer target")
        else:
            raise ValueError(f"Unknown layer-local method: {method}")
        target_kept = round(
            n_vis * (target_layer_equivalents - layer_index) / (self.total_layers - layer_index))
        target_kept = max(1, min(n_vis, target_kept))
        if method == "PACT":
            prune_keep_fraction = max(prune_keep_fraction, target_kept / n_vis)
        input_ids = inputs["input_ids"]
        text_embeds = self.lm.get_input_embeddings()(input_ids)
        visual = self.model.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
        vis_embeds = visual.pooler_output
        placeholder, _ = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=text_embeds, image_features=vis_embeds)
        inputs_embeds = text_embeds.masked_scatter(placeholder, vis_embeds)
        self.model.model.rope_deltas = None
        if method == "PACT":
            manipulator = PactManipulator(
                self.lm, vis_pos, layer=min(layer_index, self.total_layers - 1),
                prune_keep_fraction=prune_keep_fraction, cutoff=cutoff,
                target_visual_tokens=target_kept)
        else:
            manipulator = FastVManipulator(
                self.lm, vis_pos, target_visual_tokens=target_kept)
        manipulator.patch()
        try:
            with deterministic_generation_config(self.model):
                output_ids = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs.get("attention_mask"),
                    position_ids=inputs.get("position_ids"),
                    max_new_tokens=max_new_tokens,
                    **deterministic_generation_kwargs(),
                )
            generated = output_ids[0]
            text = self.processor.decode(generated, skip_special_tokens=True).strip()
            if manipulator.kept_visual_tokens is None:
                raise RuntimeError(f"{method} reduction did not execute during prefill")
            actual_percent = 100 * (
                layer_index * n_vis + (self.total_layers - layer_index) * manipulator.kept_visual_tokens
            ) / (self.total_layers * n_vis)
            metadata = {
                "kept_visual_tokens": int(manipulator.kept_visual_tokens),
                "reduction_layer": int(layer_index),
                "target_visual_tokens": int(target_kept),
                "target_token_layer_percent": float(target_token_layer_percent),
                "actual_visual_prefill_tl_percent": float(actual_percent),
            }
            if method == "PACT":
                metadata.update(cutoff=float(manipulator.chosen_cutoff),
                                prune_keep_fraction=float(prune_keep_fraction))
            else:
                metadata["source_attention_layer"] = layer_index - 1
            return text, metadata
        finally:
            manipulator.unpatch()

    @torch.no_grad()
    def pruned_answer_logprob(self, inputs, vis_embeds, grid_thw, keep_mask, answer_ids: torch.Tensor, max_answer_tokens: int) -> float:
        answer_ids = answer_ids[:max_answer_tokens].to(inputs["input_ids"].device)
        if answer_ids.numel() == 0:
            raise ValueError("empty answer ids")
        mr = self._manipulated_inputs(inputs, vis_embeds, grid_thw, keep_mask)
        prompt_embeds = self._embeds_from_manipulation(mr)
        answer_embeds = self.lm.get_input_embeddings()(answer_ids.unsqueeze(0)).to(prompt_embeds.dtype)
        inputs_embeds = torch.cat([prompt_embeds, answer_embeds], dim=1)
        attention_mask = mr["new_attention_mask"]
        extra_attn = torch.ones((attention_mask.shape[0], answer_ids.numel()), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, extra_attn], dim=1)
        mm_token_type_ids = mr.get("new_mm_token_type_ids")
        if mm_token_type_ids is not None:
            extra_mm = torch.zeros((mm_token_type_ids.shape[0], answer_ids.numel()), dtype=mm_token_type_ids.dtype, device=mm_token_type_ids.device)
            mm_token_type_ids = torch.cat([mm_token_type_ids, extra_mm], dim=1)
        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "image_grid_thw": mr.get("new_image_grid_thw", inputs.get("image_grid_thw")),
            "video_grid_thw": inputs.get("video_grid_thw"),
            "return_dict": True,
            "use_cache": False,
        }
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        logits = self.model(**model_inputs).logits.float()
        prompt_len = prompt_embeds.shape[1]
        answer_len = answer_ids.numel()
        pred_logits = logits[:, prompt_len - 1: prompt_len + answer_len - 1, :]
        logp = F.log_softmax(pred_logits, dim=-1)
        target = answer_ids.reshape(1, -1, 1)
        return float(logp.gather(-1, target).squeeze(-1).sum().detach().cpu())

    def _embeds_from_manipulation(self, mr):
        new_ids = mr["new_input_ids"]
        new_vis = mr["new_visual_embeds"]
        text_embeds = self.lm.get_input_embeddings()(new_ids)
        if new_vis.shape[0] > 0:
            mask, _ = self.model.model.get_placeholder_mask(
                new_ids, inputs_embeds=text_embeds, image_features=new_vis)
            return text_embeds.masked_scatter(mask, new_vis)
        return text_embeds

    def _manipulated_inputs(self, inputs, vis_embeds, grid_thw, keep_mask):
        tok = TokenManipulatorV2()
        merge = self.model.model.visual.spatial_merge_size
        return tok.apply_token_mask(
            inputs,
            vis_embeds,
            grid_thw,
            keep_mask=keep_mask,
            spatial_merge_size=merge,
            config=self.model.config,
        )

    @torch.no_grad()
    def build_multiround_cache(self, inputs, vis_embeds, grid_thw, keep_mask):
        mr = self._manipulated_inputs(inputs, vis_embeds, grid_thw, keep_mask)
        embeds = self._embeds_from_manipulation(mr)
        ids = mr["new_input_ids"][0]
        ve = (ids == self.vision_end_id).nonzero(as_tuple=True)[0]
        if len(ve) == 0:
            raise RuntimeError("Qwen multiround cache split failed: no vision_end token")
        shared_len = int(ve[-1].item()) + 1
        position_ids = mr["position_ids"]
        shared_pos = position_ids[:, :, :shared_len] if position_ids.dim() == 3 else position_ids[:, :shared_len]
        shared_attn = mr["new_attention_mask"][:, :shared_len]
        out = self.lm(
            inputs_embeds=embeds[:, :shared_len, :],
            attention_mask=shared_attn,
            position_ids=shared_pos,
            use_cache=True,
            return_dict=True,
        )
        shared_cache = out.past_key_values
        vis_cache_pos = [
            int(i)
            for i in range(shared_len)
            if int(ids[i].item()) == self.image_token_id
        ]
        return {
            "backend": "qwen",
            "shared_cache": shared_cache,
            "keep_mask": list(bool(x) for x in keep_mask),
            "shared_len": shared_len,
            "vis_cache_pos": vis_cache_pos,
        }

    @torch.no_grad()
    def generate_from_multiround_cache(self, cache_entry, inputs, vis_embeds, grid_thw, max_new_tokens: int):
        mr = self._manipulated_inputs(inputs, vis_embeds, grid_thw, cache_entry["keep_mask"])
        embeds = self._embeds_from_manipulation(mr)
        ids = mr["new_input_ids"][0]
        ve = (ids == self.vision_end_id).nonzero(as_tuple=True)[0]
        if len(ve) == 0:
            raise RuntimeError("Qwen multiround cache split failed: no vision_end token")
        question_start = int(ve[-1].item()) + 1
        if question_start >= embeds.shape[1]:
            raise RuntimeError("Qwen multiround cache split produced empty question suffix")
        position_ids = mr["position_ids"]
        question_pos = position_ids[:, :, question_start:] if position_ids.dim() == 3 else position_ids[:, question_start:]
        cache = _clone_cache(cache_entry["shared_cache"])
        full_attention = mr["new_attention_mask"][:, :embeds.shape[1]]
        out = self.lm(
            inputs_embeds=embeds[:, question_start:, :],
            attention_mask=full_attention,
            position_ids=question_pos,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        h = self.lm.norm(out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])
        logits = self.lm_head(h[:, -1:, :])
        next_id = logits.argmax(dim=-1).reshape(1, 1)
        generated = [int(next_id.item())]
        if generated[-1] == self.eos_id or max_new_tokens <= 1:
            return self.processor.decode(
                torch.tensor(generated, device=embeds.device, dtype=torch.long),
                skip_special_tokens=True,
            ).strip()

        last_pos = (
            question_pos[..., -1:] + 1
            if question_pos is not None
            else torch.tensor([[embeds.shape[1]]], device=embeds.device)
        )
        attention = torch.ones(
            (1, full_attention.shape[1] + 1),
            dtype=full_attention.dtype,
            device=full_attention.device,
        )
        embed_tokens = self.lm.get_input_embeddings() if hasattr(self.lm, "get_input_embeddings") else self.lm.embed_tokens
        for _ in range(max_new_tokens - 1):
            token_emb = embed_tokens(next_id).to(embeds.dtype)
            out = self.lm(
                inputs_embeds=token_emb,
                attention_mask=attention,
                position_ids=last_pos,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            h = self.lm.norm(out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])
            logits = self.lm_head(h[:, -1:, :])
            next_id = logits.argmax(dim=-1).reshape(1, 1)
            token = int(next_id.item())
            generated.append(token)
            if token == self.eos_id:
                break
            last_pos = last_pos + 1
            attention = torch.ones(
                (1, attention.shape[1] + 1),
                dtype=attention.dtype,
                device=attention.device,
            )
        return self.processor.decode(
            torch.tensor(generated, device=embeds.device, dtype=torch.long),
            skip_special_tokens=True,
        ).strip()

    @torch.no_grad()
    def generate_sparsevila_from_multiround_cache(
        self,
        cache_entry,
        inputs,
        vis_embeds,
        grid_thw,
        max_new_tokens: int,
        decode_keep_ratio: float = 0.5,
    ):
        mr = self._manipulated_inputs(inputs, vis_embeds, grid_thw, cache_entry["keep_mask"])
        embeds = self._embeds_from_manipulation(mr)
        ids = mr["new_input_ids"][0]
        ve = (ids == self.vision_end_id).nonzero(as_tuple=True)[0]
        if len(ve) == 0:
            raise RuntimeError("Qwen SparseVILA cache split failed: no vision_end token")
        question_start = int(ve[-1].item()) + 1
        if question_start >= embeds.shape[1]:
            raise RuntimeError("Qwen SparseVILA cache split produced empty question suffix")
        position_ids = mr["position_ids"]
        question_pos = position_ids[:, :, question_start:] if position_ids.dim() == 3 else position_ids[:, question_start:]
        out_ids, meta = _generate_ids_sparsevila_from_shared_cache(
            self.lm,
            self.lm_head,
            cache_entry["shared_cache"],
            cache_entry.get("vis_cache_pos", []),
            decode_keep_ratio,
            embeds[:, question_start:, :],
            question_pos,
            self.eos_id,
            max_new_tokens,
        )
        return self.processor.decode(out_ids, skip_special_tokens=True).strip(), meta

    @torch.no_grad()
    def batch_run_pruned(self, inputs, vis_embeds, grid_thw, keep_masks, max_new_tokens: int, batch_size: int = 4):
        tok = TokenManipulatorV2()
        merge = self.model.model.visual.spatial_merge_size
        outputs = []
        for start in range(0, len(keep_masks), batch_size):
            masks = keep_masks[start:start + batch_size]
            manip = [
                tok.apply_token_mask(inputs, vis_embeds, grid_thw, keep_mask=mask,
                                     spatial_merge_size=merge, config=self.model.config)
                for mask in masks
            ]
            seq_lens = {int(mr["new_input_ids"].shape[1]) for mr in manip}
            if len(seq_lens) != 1:
                with deterministic_generation_config(self.model):
                    outputs.extend([
                        self.wrapper.generate_with_token_manipulation_v2(mr, max_new_tokens=max_new_tokens)
                        for mr in manip
                    ])
                continue

            embeds = []
            attn = []
            pos = []
            rope = []
            for mr in manip:
                new_ids = mr["new_input_ids"]
                new_vis = mr["new_visual_embeds"]
                text_embeds = self.model.model.language_model.get_input_embeddings()(new_ids)
                if new_vis.shape[0] > 0:
                    mask, _ = self.model.model.get_placeholder_mask(
                        new_ids, inputs_embeds=text_embeds, image_features=new_vis)
                    input_embeds = text_embeds.masked_scatter(mask, new_vis)
                else:
                    input_embeds = text_embeds
                embeds.append(input_embeds)
                attn.append(mr["new_attention_mask"])
                pos.append(mr["position_ids"])
                rope.append(mr.get("rope_deltas"))

            old_rope = getattr(self.model.model, "rope_deltas", None)
            try:
                rope_vals = [r for r in rope if r is not None]
                if len(rope_vals) == len(rope):
                    self.model.model.rope_deltas = torch.cat(rope_vals, dim=0)
                output_ids = self.model.generate(
                    inputs_embeds=torch.cat(embeds, dim=0),
                    attention_mask=torch.cat(attn, dim=0),
                    position_ids=torch.cat(pos, dim=1),
                    max_new_tokens=max_new_tokens,
                    **deterministic_generation_kwargs(),
                )
            finally:
                self.model.model.rope_deltas = old_rope
            for row in output_ids:
                outputs.append(self.processor.decode(row, skip_special_tokens=True).strip())
        return outputs

    @torch.no_grad()
    def visual_received_attention_scores(self, inputs, n_vis: int):
        vit = self.model.model.visual
        block_idx = len(vit.blocks) - 1
        block = vit.blocks[block_idx]
        attn_m = block.attn
        captured = {}

        def _pre_hook(_module, args):
            captured["h"] = args[0].detach()

        handle = block.register_forward_pre_hook(_pre_hook)
        try:
            _ = vit(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
        finally:
            handle.remove()

        h = captured.get("h")
        if h is None:
            raise RuntimeError("failed to capture Qwen vision hidden states")
        n_patch = h.shape[0]
        qkv = attn_m.qkv(h)
        q, k, _ = qkv.chunk(3, dim=-1)
        q = q.view(n_patch, attn_m.num_heads, attn_m.head_dim).permute(1, 0, 2).float()
        k = k.view(n_patch, attn_m.num_heads, attn_m.head_dim).permute(1, 0, 2).float()
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * (attn_m.head_dim ** -0.5), dim=-1)
        patch_scores = attn.mean(dim=1).mean(dim=0).cpu()
        merge = int(getattr(vit, "spatial_merge_size", 1))
        group = max(1, merge * merge)
        n_from_patch = patch_scores.shape[0] // group
        token_scores = patch_scores[: n_from_patch * group].view(n_from_patch, group).mean(dim=1)
        return _align_score_length(token_scores, n_vis).tolist()

    @torch.no_grad()
    def run_pyramiddrop(self, inputs, budget_percent: float, max_new_tokens: int):
        from baselines.common.progressive_manipulator import pyramiddrop_rank_fn

        vis_pos, _ = self.visual_and_text_positions(inputs)
        if not vis_pos:
            return self.generate_ids_and_text(inputs, max_new_tokens)[1], 0
        n_vis = len(vis_pos)
        budget_frac = max(0.0, min(1.0, float(budget_percent) / 100.0))
        num_stages = 4
        if budget_frac >= 1.0:
            boundary_spec = {}
        else:
            drop_ratio = budget_frac ** (1.0 / (num_stages - 1))
            layers_per_stage = max(1, self.total_layers // num_stages)
            boundaries = [(s + 1) * layers_per_stage for s in range(num_stages - 1)]
            current = n_vis
            boundary_spec = {}
            for stage_idx, layer_idx in enumerate(boundaries, start=1):
                target = max(1, int(n_vis * (drop_ratio ** stage_idx)))
                drop = current - target
                if drop > 0 and layer_idx < self.total_layers:
                    boundary_spec[layer_idx] = drop
                current = target
        if not hasattr(self.lm, "has_sliding_layers"):
            self.lm.has_sliding_layers = False
        if not hasattr(self.lm.config, "layer_types"):
            self.lm.config.layer_types = ["full_attention"] * self.total_layers
        rank_fn = (
            qwen3_pyramiddrop_rank_fn
            if self.lm.__class__.__name__.lower().startswith("qwen3")
            else pyramiddrop_rank_fn
        )
        out = self.wrapper.generate_baseline_progressive(
            inputs,
            min(vis_pos),
            max(vis_pos) + 1,
            boundary_spec,
            rank_fn,
            max_new_tokens=max_new_tokens,
            vis_token_positions=vis_pos,
        )
        final_keep = n_vis - sum(boundary_spec.values())
        return out, final_keep

    @torch.no_grad()
    def run_fitprune(self, inputs, budget_percent: float, max_new_tokens: int):
        from baselines.fitprune.adapter import FitPruneBaseline
        from baselines.common.progressive_manipulator import fitprune_rank_fn

        vis_pos, _ = self.visual_and_text_positions(inputs)
        if not vis_pos:
            return self.generate_ids_and_text(inputs, max_new_tokens)[1], 0
        n_vis = len(vis_pos)
        budget_frac = max(0.0, min(1.0, float(budget_percent) / 100.0))
        fp = FitPruneBaseline(self.wrapper, budget_frac=budget_frac)
        boundary_spec = fp.build_boundary_spec(n_vis, budget_frac)
        if not hasattr(self.lm, "has_sliding_layers"):
            self.lm.has_sliding_layers = False
        if not hasattr(self.lm.config, "layer_types"):
            self.lm.config.layer_types = ["full_attention"] * self.total_layers
        rank_fn = (
            qwen3_fitprune_rank_fn
            if self.lm.__class__.__name__.lower().startswith("qwen3")
            else fitprune_rank_fn
        )
        out = self.wrapper.generate_baseline_progressive(
            inputs,
            min(vis_pos),
            max(vis_pos) + 1,
            boundary_spec,
            rank_fn,
            max_new_tokens=max_new_tokens,
            vis_token_positions=vis_pos,
        )
        final_keep = n_vis - sum(boundary_spec.values())
        return out, final_keep

    def _sparsevlm_stage_targets(self, n_vis: int, budget_percent: float):
        keep_frac = max(0.0, min(1.0, float(budget_percent) / 100.0))
        final_keep = max(1, int(n_vis * keep_frac))
        boundaries = sorted(set([
            0,
            max(1, self.total_layers // 4),
            max(1, self.total_layers // 2),
            max(1, (self.total_layers * 3) // 4),
        ]))
        boundaries = [b for b in boundaries if b < self.total_layers]
        targets = {}
        n_stages = len(boundaries)
        for stage_idx, boundary in enumerate(boundaries):
            frac = float(stage_idx + 1) / max(1, n_stages)
            target = round(n_vis - (n_vis - final_keep) * frac)
            targets[boundary] = max(final_keep, int(target))
        if boundaries:
            targets[boundaries[-1]] = final_keep
        return targets, final_keep

    @torch.no_grad()
    def run_sparsevlm(self, inputs, budget_percent: float, max_new_tokens: int):
        from baselines.common.progressive_manipulator import BaselineProgressiveManipulator

        vis_pos, text_pos = self.visual_and_text_positions(inputs)
        if not vis_pos:
            return self.generate_ids_and_text(inputs, max_new_tokens)[1], 0
        n_vis = len(vis_pos)
        stage_targets, _ = self._sparsevlm_stage_targets(n_vis, budget_percent)
        if not stage_targets:
            return self.generate_ids_and_text(inputs, max_new_tokens)[1], n_vis
        if not hasattr(self.lm, "has_sliding_layers"):
            self.lm.has_sliding_layers = False
        if not hasattr(self.lm.config, "layer_types"):
            self.lm.config.layer_types = ["full_attention"] * self.total_layers

        rater_positions = {"value": None}
        kept_count = {"value": n_vis}
        keep_frac = max(0.0, min(1.0, float(budget_percent) / 100.0))
        final_stage = max(stage_targets)

        def rank_fn(hidden_states, current_vis, current_text, layer_idx, attn_layer, *,
                    layer_module=None, position_embeddings=None, mrope_section=None):
            if not current_vis or not current_text:
                return []
            if rater_positions["value"] is None:
                n_raters = max(1, round(len(current_text) * 0.5))
                rater_scores = qwen_rotary_cross_attention_scores(
                    hidden_states, current_vis, current_text, attn_layer,
                    layer_module=layer_module,
                    position_embeddings=position_embeddings,
                    mrope_section=mrope_section,
                )
                top = rater_scores.argsort(descending=True)[:n_raters].tolist()
                rater_positions["value"] = [current_text[int(i)] for i in top]

            n_current = len(current_vis)
            erank = effective_rank(hidden_states[0, current_vis, :])
            erank_ratio = min(1.0, erank / max(1, n_current))
            adaptive = round(n_current * keep_frac * (erank_ratio ** 0.5))
            if layer_idx == final_stage:
                n_keep = stage_targets.get(layer_idx, 1)
            else:
                n_keep = max(stage_targets.get(layer_idx, 1), adaptive, 1)
            n_keep = min(n_current, int(n_keep))
            if n_current <= n_keep:
                return []
            scores = qwen_rotary_cross_attention_scores(
                hidden_states, rater_positions["value"], current_vis, attn_layer,
                layer_module=layer_module,
                position_embeddings=position_embeddings,
                mrope_section=mrope_section,
            )
            ranked = scores.argsort(descending=True).tolist()
            drop_local = [int(i) for i in ranked[n_keep:]]
            drop_positions = [current_vis[i] for i in drop_local]
            score_map = {current_vis[i]: float(scores[i]) for i in range(len(current_vis))}
            kept_count["value"] = n_current - len(drop_positions)
            return drop_positions, score_map

        def recycle_fn(hidden_states, keep_positions, drop_positions, score_map):
            drop_scores = torch.tensor(
                [score_map.get(pos, 0.0) for pos in drop_positions],
                dtype=torch.float32,
            )
            return recycle_visual_hidden(hidden_states, keep_positions, drop_positions, drop_scores)

        def remap_fn(old_to_new):
            if rater_positions["value"] is not None:
                rater_positions["value"] = [
                    old_to_new[p] for p in rater_positions["value"] if p in old_to_new
                ]

        boundary_spec = {layer: max(1, n_vis - 1) for layer in stage_targets}
        input_ids = inputs["input_ids"]
        text_embeds = self.lm.get_input_embeddings()(input_ids)
        vis_out = self.model.model.visual(
            inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
        vis_embeds = vis_out.pooler_output
        if vis_embeds.shape[0] > 0:
            mask, _ = self.model.model.get_placeholder_mask(
                input_ids, inputs_embeds=text_embeds, image_features=vis_embeds)
            inputs_embeds = text_embeds.masked_scatter(mask, vis_embeds)
        else:
            inputs_embeds = text_embeds
        self.model.model.rope_deltas = None
        manip = BaselineProgressiveManipulator(
            self.wrapper,
            boundary_spec,
            min(vis_pos),
            max(vis_pos) + 1,
            rank_fn=rank_fn,
            vis_token_positions=vis_pos,
            recycle_fn=recycle_fn,
            remap_fn=remap_fn,
        )
        manip.patch()
        try:
            gen_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": inputs.get("attention_mask"),
                "position_ids": inputs.get("position_ids"),
                "max_new_tokens": max_new_tokens,
            }
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            gen_kwargs.update(deterministic_generation_kwargs())
            output_ids = self.model.generate(**gen_kwargs)
            generated = output_ids[0, input_ids.shape[1]:]
            return self.processor.decode(generated, skip_special_tokens=True).strip(), kept_count["value"]
        finally:
            manip.unpatch()


class EmbeddingPruneBackend:
    image_token_id: int
    supports_multiround_cache = True

    @torch.no_grad()
    def _layer_attention_from_outputs(
        self,
        inputs: BackendInputs,
        layers: list[int],
        query_mode: str = "last_text",
    ):
        if query_mode != "last_text":
            return None
        out = self.lm(
            inputs_embeds=inputs.full_embeds,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
        attentions = getattr(out, "attentions", None)
        if not attentions:
            return None
        q_pos = inputs.text_pos[-1]
        scores_by_layer = {}
        for layer in layers:
            if layer >= len(attentions) or attentions[layer] is None:
                return None
            attn = attentions[layer]
            scores = attn[0, :, q_pos, inputs.vis_pos].float().mean(0).detach().cpu()
            scores_by_layer[layer] = norm01_t(scores).tolist()
        return scores_by_layer

    @torch.no_grad()
    def _layer_head_attention_from_outputs(
        self,
        inputs: BackendInputs,
        layers: list[int],
        query_mode: str = "last_text",
    ):
        if query_mode != "last_text":
            return None
        out = self.lm(
            inputs_embeds=inputs.full_embeds,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
        attentions = getattr(out, "attentions", None)
        if not attentions:
            return None
        q_pos = inputs.text_pos[-1]
        by_layer = {}
        for layer in layers:
            if layer >= len(attentions) or attentions[layer] is None:
                return None
            attn = attentions[layer]
            scores = attn[0, :, q_pos, inputs.vis_pos].float().detach().cpu()
            by_layer[layer] = [norm01_t(scores[h]).tolist() for h in range(scores.shape[0])]
        return by_layer

    def _layer_attention_from_embeds(
        self,
        inputs: BackendInputs,
        layers: list[int],
        query_mode: str = "last_text",
    ):
        if not inputs.vis_pos or not inputs.text_pos:
            return {layer: [] for layer in layers}
        try:
            actual = self._layer_attention_from_outputs(inputs, layers, query_mode=query_mode)
            if actual is not None:
                return actual
        except Exception as exc:
            print(f"output_attentions fallback: {type(exc).__name__}: {exc}", flush=True)

        captured = {}
        handles = []

        def make_hook(layer_idx: int):
            def _pre_hook(module, args):
                if isinstance(args, tuple) and args:
                    h = args[0]
                    captured[layer_idx] = h.detach() if h.dim() == 3 else h.unsqueeze(0).detach()
            return _pre_hook

        try:
            for layer_idx in layers:
                handles.append(self.lm.layers[layer_idx].register_forward_pre_hook(make_hook(layer_idx)))
            self.lm(inputs_embeds=inputs.full_embeds, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        out = {}
        for layer_idx in layers:
            h = captured.get(layer_idx)
            if h is None:
                out[layer_idx] = [0.0 for _ in inputs.vis_pos]
            else:
                scores = compute_text_vis_attention(
                    self.lm.layers[layer_idx], h, inputs.text_pos, inputs.vis_pos,
                    {}, self.lm, apply_rope=False, query_mode=query_mode)
                out[layer_idx] = norm01_t(scores).tolist()
        return out

    def extract_visual_embeddings(self, inputs: BackendInputs):
        return inputs.visual_embeds, inputs.extra.get("grid_thw")

    def layer_attention_scores(
        self,
        inputs: BackendInputs,
        layers: list[int],
        query_mode: str = "last_text",
    ):
        return self._layer_attention_from_embeds(inputs, layers, query_mode=query_mode)

    def layer_head_attention_scores(
        self,
        inputs: BackendInputs,
        layers: list[int],
        query_mode: str = "last_text",
    ):
        try:
            return self._layer_head_attention_from_outputs(inputs, layers, query_mode=query_mode)
        except Exception as exc:
            print(f"head output_attentions fallback unavailable: {type(exc).__name__}: {exc}", flush=True)
            return None

    def _pruned_embeds(self, inputs: BackendInputs, keep_mask: list[bool]):
        keep_vis_pos = {inputs.vis_pos[i] for i, keep in enumerate(keep_mask) if keep}
        positions = [
            pos for pos in range(inputs.full_embeds.shape[1])
            if pos not in set(inputs.vis_pos) or pos in keep_vis_pos
        ]
        return inputs.full_embeds[:, positions, :]

    @torch.no_grad()
    def _generate_with_model_from_embeds(self, embeds: torch.Tensor, max_new_tokens: int) -> str:
        attention_mask = torch.ones(
            embeds.shape[:2],
            dtype=torch.long,
            device=embeds.device,
        )
        out_ids = self.model.generate(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        ids = out_ids[0] if out_ids.dim() == 2 else out_ids
        return self.processor.decode(ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def _build_lm_prefix_cache(self, shared_embeds: torch.Tensor):
        attention_mask = torch.ones(
            shared_embeds.shape[:2],
            dtype=torch.long,
            device=shared_embeds.device,
        )
        position_ids = torch.arange(
            shared_embeds.shape[1],
            device=shared_embeds.device,
            dtype=torch.long,
        ).reshape(1, -1)
        out = self.lm(
            inputs_embeds=shared_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        return out.past_key_values

    @torch.no_grad()
    def _generate_ids_from_lm_cache(
        self,
        shared_cache,
        shared_len: int,
        question_embeds: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        cache = _clone_cache(shared_cache)
        q_len = int(question_embeds.shape[1])
        attention_mask = torch.ones(
            (1, shared_len + q_len),
            dtype=torch.long,
            device=question_embeds.device,
        )
        position_ids = torch.arange(
            shared_len,
            shared_len + q_len,
            device=question_embeds.device,
            dtype=torch.long,
        ).reshape(1, -1)
        out = self.lm(
            inputs_embeds=question_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        next_id = self.lm_head(out.last_hidden_state[:, -1:, :]).argmax(dim=-1).reshape(1, 1)
        generated = [int(next_id.item())]
        eos_ids = set(self.eos_id if isinstance(self.eos_id, (list, tuple, set)) else [self.eos_id])
        embed_tokens = self.lm.get_input_embeddings() if hasattr(self.lm, "get_input_embeddings") else self.lm.embed_tokens
        for _ in range(max(0, max_new_tokens - 1)):
            if generated[-1] in eos_ids:
                break
            token_embeds = embed_tokens(next_id).to(question_embeds.dtype)
            cur_len = shared_len + q_len + len(generated)
            attention_mask = torch.ones(
                (1, cur_len),
                dtype=torch.long,
                device=question_embeds.device,
            )
            position_ids = torch.tensor(
                [[cur_len - 1]],
                dtype=torch.long,
                device=question_embeds.device,
            )
            out = self.lm(
                inputs_embeds=token_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            next_id = self.lm_head(out.last_hidden_state[:, -1:, :]).argmax(dim=-1).reshape(1, 1)
            generated.append(int(next_id.item()))
        return torch.tensor(generated, device=question_embeds.device, dtype=torch.long)

    @torch.no_grad()
    def _generate_ids_sparsevila_from_lm_cache(
        self,
        shared_cache,
        shared_len: int,
        vis_indices_in_cache: list[int],
        decode_keep_ratio: float,
        question_embeds: torch.Tensor,
        max_new_tokens: int,
    ):
        device = question_embeds.device
        cache = _clone_cache(shared_cache)
        q_len = int(question_embeds.shape[1])
        attention_mask = torch.ones(
            (1, shared_len + q_len),
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.arange(
            shared_len,
            shared_len + q_len,
            device=device,
            dtype=torch.long,
        ).reshape(1, -1)
        out = self.lm(
            inputs_embeds=question_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        cache = out.past_key_values
        n_vis = len(vis_indices_in_cache)
        n_decode_keep = max(1, min(n_vis, int(round(n_vis * float(decode_keep_ratio))))) if n_vis else 0

        scores = torch.zeros(n_vis, dtype=torch.float64)
        attentions = getattr(out, "attentions", None)
        if attentions and n_vis:
            vis_t = torch.tensor(vis_indices_in_cache, dtype=torch.long)
            used_layers = 0
            for attn in attentions:
                if attn is None or attn.shape[-1] <= int(vis_t.max()):
                    continue
                layer_scores = attn[0, :, -1, vis_t.to(attn.device)].float().mean(dim=0).detach().cpu()
                scores += layer_scores.double()
                used_layers += 1
            if used_layers:
                scores /= used_layers
            else:
                scores = torch.arange(n_vis, dtype=torch.float64)
        elif n_vis:
            scores = torch.arange(n_vis, dtype=torch.float64)

        compacted_cache_seq_len = _cache_seq_len(cache)
        if n_vis and 0 < n_decode_keep < n_vis:
            top_local = scores.argsort(descending=True)[:n_decode_keep].tolist()
            keep_vis = {int(vis_indices_in_cache[int(i)]) for i in top_local}
            vis_set = set(int(i) for i in vis_indices_in_cache)
            seq_len = _cache_seq_len(cache)
            keep_positions = [p for p in range(seq_len) if p not in vis_set or p in keep_vis]
            cache = _prune_cache_positions(cache, keep_positions, device)
            compacted_cache_seq_len = len(keep_positions)

        next_id = self.lm_head(out.last_hidden_state[:, -1:, :]).argmax(dim=-1).reshape(1, 1)
        generated = [int(next_id.item())]
        eos_ids = set(self.eos_id if isinstance(self.eos_id, (list, tuple, set)) else [self.eos_id])
        embed_tokens = self.lm.get_input_embeddings() if hasattr(self.lm, "get_input_embeddings") else self.lm.embed_tokens
        absolute_next_pos = shared_len + q_len
        for _ in range(max(0, max_new_tokens - 1)):
            if generated[-1] in eos_ids:
                break
            token_embeds = embed_tokens(next_id).to(question_embeds.dtype)
            cache_len = _cache_seq_len(cache)
            attention_mask = torch.ones(
                (1, cache_len + 1),
                dtype=torch.long,
                device=device,
            )
            position_ids = torch.tensor(
                [[absolute_next_pos]],
                dtype=torch.long,
                device=device,
            )
            out = self.lm(
                inputs_embeds=token_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            next_id = self.lm_head(out.last_hidden_state[:, -1:, :]).argmax(dim=-1).reshape(1, 1)
            generated.append(int(next_id.item()))
            absolute_next_pos += 1

        return torch.tensor(generated, device=device, dtype=torch.long), {
            "decode_stage_kv_retrieval_implemented": bool(n_vis and n_decode_keep < n_vis),
            "decode_keep_ratio": float(decode_keep_ratio),
            "decode_visual_tokens": int(n_decode_keep),
            "prefill_visual_tokens": int(n_vis),
            "compacted_cache_seq_len": int(compacted_cache_seq_len),
        }

    def _visual_score_fallback(self, inputs: BackendInputs, n_vis: int, reason: str):
        emb = inputs.extra.get("score_visual_embeds")
        if emb is None:
            emb = inputs.visual_embeds
        scores = emb.float().reshape(-1, emb.shape[-1]).norm(dim=-1)
        print(
            f"{self.__class__.__name__} vision attention fallback: {reason}; "
            "using visual embedding norm salience",
            flush=True,
        )
        return _align_score_length(scores, n_vis).tolist()

    def _question_start(self, inputs: BackendInputs) -> int:
        if not inputs.vis_pos:
            return 0
        return max(inputs.vis_pos) + 1

    def _pruned_positions(self, inputs: BackendInputs, keep_mask: list[bool]):
        keep_vis_pos = {inputs.vis_pos[i] for i, keep in enumerate(keep_mask) if keep}
        return [
            pos for pos in range(inputs.full_embeds.shape[1])
            if pos not in set(inputs.vis_pos) or pos in keep_vis_pos
        ]

    @torch.no_grad()
    def build_multiround_cache(self, inputs: BackendInputs, vis_embeds, grid_thw, keep_mask):
        question_start = self._question_start(inputs)
        positions = self._pruned_positions(inputs, keep_mask)
        shared_positions = [p for p in positions if p < question_start]
        if not shared_positions:
            raise RuntimeError("multiround cache split produced empty shared prefix")
        idx = torch.tensor(shared_positions, dtype=torch.long, device=inputs.full_embeds.device)
        shared_embeds = inputs.full_embeds[:, idx, :]
        shared_pos = torch.arange(shared_embeds.shape[1], device=shared_embeds.device).reshape(1, -1)
        shared_cache = self._build_lm_prefix_cache(shared_embeds)
        vis_orig = set(inputs.vis_pos)
        vis_cache_pos = [
            cache_pos
            for cache_pos, orig_pos in enumerate(shared_positions)
            if orig_pos in vis_orig
        ]
        return {
            "backend": self.__class__.__name__,
            "shared_cache": shared_cache,
            "keep_mask": list(bool(x) for x in keep_mask),
            "shared_len": int(shared_embeds.shape[1]),
            "vis_cache_pos": vis_cache_pos,
        }

    @torch.no_grad()
    def generate_from_multiround_cache(
        self,
        cache_entry,
        inputs: BackendInputs,
        vis_embeds,
        grid_thw,
        max_new_tokens: int,
    ):
        question_start = self._question_start(inputs)
        positions = self._pruned_positions(inputs, cache_entry["keep_mask"])
        question_positions = [p for p in positions if p >= question_start]
        if not question_positions:
            raise RuntimeError("multiround cache split produced empty question suffix")
        idx = torch.tensor(question_positions, dtype=torch.long, device=inputs.full_embeds.device)
        question_embeds = inputs.full_embeds[:, idx, :]
        start = int(cache_entry["shared_len"])
        question_pos = torch.arange(
            start,
            start + question_embeds.shape[1],
            device=question_embeds.device,
        ).reshape(1, -1)
        out_ids = self._generate_ids_from_lm_cache(
            cache_entry["shared_cache"],
            int(cache_entry["shared_len"]),
            question_embeds,
            max_new_tokens,
        )
        return self.processor.decode(out_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_sparsevila_from_multiround_cache(
        self,
        cache_entry,
        inputs: BackendInputs,
        vis_embeds,
        grid_thw,
        max_new_tokens: int,
        decode_keep_ratio: float = 0.5,
    ):
        question_start = self._question_start(inputs)
        positions = self._pruned_positions(inputs, cache_entry["keep_mask"])
        question_positions = [p for p in positions if p >= question_start]
        if not question_positions:
            raise RuntimeError("SparseVILA cache split produced empty question suffix")
        idx = torch.tensor(question_positions, dtype=torch.long, device=inputs.full_embeds.device)
        question_embeds = inputs.full_embeds[:, idx, :]
        out_ids, meta = self._generate_ids_sparsevila_from_lm_cache(
            cache_entry["shared_cache"],
            int(cache_entry["shared_len"]),
            cache_entry.get("vis_cache_pos", []),
            decode_keep_ratio,
            question_embeds,
            max_new_tokens,
        )
        return self.processor.decode(out_ids, skip_special_tokens=True).strip(), meta

    def run_pruned(self, inputs: BackendInputs, vis_embeds, grid_thw, keep_mask, max_new_tokens: int):
        pruned = self._pruned_embeds(inputs, keep_mask)
        return self._generate_from_embeds(pruned, max_new_tokens)

    @torch.no_grad()
    def pruned_answer_logprob(self, inputs: BackendInputs, vis_embeds, grid_thw, keep_mask, answer_ids: torch.Tensor, max_answer_tokens: int) -> float:
        answer_ids = answer_ids[:max_answer_tokens].to(inputs.full_embeds.device)
        if answer_ids.numel() == 0:
            raise ValueError("empty answer ids")
        pruned = self._pruned_embeds(inputs, keep_mask)
        embed_tokens = self.lm.get_input_embeddings() if hasattr(self.lm, "get_input_embeddings") else self.lm.embed_tokens
        answer_embeds = embed_tokens(answer_ids.unsqueeze(0)).to(pruned.dtype)
        full = torch.cat([pruned, answer_embeds], dim=1)
        out = self.lm(inputs_embeds=full, use_cache=False, return_dict=True)
        logits = self.lm_head(out.last_hidden_state).float()
        prompt_len = pruned.shape[1]
        answer_len = answer_ids.numel()
        pred_logits = logits[:, prompt_len - 1: prompt_len + answer_len - 1, :]
        logp = F.log_softmax(pred_logits, dim=-1)
        target = answer_ids.reshape(1, -1, 1)
        return float(logp.gather(-1, target).squeeze(-1).sum().detach().cpu())

    def visual_received_attention_scores(self, inputs: BackendInputs, n_vis: int):
        raise NotImplementedError("vision received-attention hook is not implemented for this backend")

    def run_pyramiddrop(self, inputs: BackendInputs, budget_percent: float, max_new_tokens: int):
        n_vis = len(inputs.vis_pos)
        boundary_spec = self._pyramiddrop_boundary_spec(n_vis, budget_percent)
        return self._generate_progressive_from_embeds(
            inputs, boundary_spec, "pyramiddrop", max_new_tokens), n_vis - sum(boundary_spec.values())

    def run_fitprune(self, inputs: BackendInputs, budget_percent: float, max_new_tokens: int):
        n_vis = len(inputs.vis_pos)
        boundary_spec = self._fitprune_boundary_spec(n_vis, budget_percent)
        return self._generate_progressive_from_embeds(
            inputs, boundary_spec, "fitprune", max_new_tokens), n_vis - sum(boundary_spec.values())

    def run_sparsevlm(self, inputs: BackendInputs, budget_percent: float, max_new_tokens: int):
        if not inputs.vis_pos:
            return self._generate_from_embeds(inputs.full_embeds, max_new_tokens), 0
        out, kept = self._generate_sparsevlm_from_embeds(inputs, budget_percent, max_new_tokens)
        return out, kept

    def _pyramiddrop_boundary_spec(self, n_vis: int, budget_percent: float):
        if n_vis <= 0:
            return {}
        budget_frac = max(0.0, min(1.0, float(budget_percent) / 100.0))
        if budget_frac >= 1.0:
            return {}
        num_stages = 4
        drop_ratio = budget_frac ** (1.0 / (num_stages - 1))
        layers_per_stage = max(1, self.total_layers // num_stages)
        current = n_vis
        spec = {}
        for stage_idx in range(1, num_stages):
            layer_idx = stage_idx * layers_per_stage
            target = max(1, int(n_vis * (drop_ratio ** stage_idx)))
            drop = current - target
            if drop > 0 and layer_idx < self.total_layers:
                spec[layer_idx] = drop
            current = target
        return spec

    def _fitprune_boundary_spec(self, n_vis: int, budget_percent: float):
        from baselines.fitprune.adapter import FitPruneBaseline
        budget_frac = max(0.0, min(1.0, float(budget_percent) / 100.0))
        return FitPruneBaseline(self, budget_frac=budget_frac).build_boundary_spec(n_vis, budget_frac)

    def _sparsevlm_stage_targets(self, n_vis: int, budget_percent: float):
        keep_frac = max(0.0, min(1.0, float(budget_percent) / 100.0))
        final_keep = max(1, int(n_vis * keep_frac))
        boundaries = sorted(set([
            0,
            max(1, self.total_layers // 4),
            max(1, self.total_layers // 2),
            max(1, (self.total_layers * 3) // 4),
        ]))
        boundaries = [b for b in boundaries if b < self.total_layers]
        targets = {}
        n_stages = len(boundaries)
        for stage_idx, boundary in enumerate(boundaries):
            frac = float(stage_idx + 1) / max(1, n_stages)
            target = round(n_vis - (n_vis - final_keep) * frac)
            targets[boundary] = max(final_keep, int(target))
        if boundaries:
            targets[boundaries[-1]] = final_keep
        return targets, final_keep

    def _generate_progressive_from_embeds(
        self,
        inputs: BackendInputs,
        boundary_spec: dict[int, int],
        rank_mode: str,
        max_new_tokens: int,
    ):
        if not boundary_spec:
            return self._generate_from_embeds(inputs.full_embeds, max_new_tokens)
        original_forward = self.lm.forward
        backend = self

        def progressive_forward(text_model, *args, **kwargs):
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)
            past_key_values = kwargs.get("past_key_values", None)
            inputs_embeds = kwargs.get("inputs_embeds", None)
            use_cache = kwargs.get("use_cache", None)
            extra = {k: v for k, v in kwargs.items()
                     if k not in {"input_ids", "attention_mask", "position_ids",
                                  "past_key_values", "inputs_embeds", "use_cache"}}
            if inputs_embeds is None:
                inputs_embeds = text_model.embed_tokens(input_ids)
            if inputs_embeds.shape[1] <= 1 or past_key_values is not None:
                return original_forward(*args, **kwargs)

            hidden_states = inputs_embeds
            if position_ids is None:
                position_ids = torch.arange(
                    hidden_states.shape[1], device=hidden_states.device).reshape(1, -1)
            vis_positions = list(inputs.vis_pos)
            text_positions = [p for p in range(hidden_states.shape[1]) if p not in set(vis_positions)]
            position_embeddings = text_model.rotary_emb(hidden_states, position_ids)

            for layer_idx, decoder_layer in enumerate(text_model.layers):
                if layer_idx in boundary_spec:
                    drop = boundary_spec[layer_idx]
                    if drop > 0 and len(vis_positions) > drop:
                        ranked_drop = generic_rotary_attention_rank(
                            hidden_states,
                            vis_positions,
                            text_positions,
                            decoder_layer.self_attn,
                            layer_module=decoder_layer,
                            position_embeddings=position_embeddings,
                            mode=rank_mode,
                        )[:drop]
                        drop_set = set(ranked_drop)
                        keep_seq = [p for p in range(hidden_states.shape[1]) if p not in drop_set]
                        idx = torch.tensor(keep_seq, dtype=torch.long, device=hidden_states.device)
                        hidden_states = hidden_states[:, idx, :]
                        position_ids = position_ids[:, idx]
                        cos, sin = position_embeddings
                        if cos.dim() == 4:
                            position_embeddings = (cos[:, :, idx.to(cos.device), :], sin[:, :, idx.to(sin.device), :])
                        else:
                            position_embeddings = (cos[:, idx.to(cos.device), :], sin[:, idx.to(sin.device), :])
                        old_to_new = {old: new for new, old in enumerate(keep_seq)}
                        vis_positions = [old_to_new[p] for p in vis_positions if p not in drop_set]
                        text_positions = [old_to_new[p] for p in text_positions]
                        attention_mask = None

                layer_out = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **extra,
                )
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            hidden_states = text_model.norm(hidden_states)
            return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

        self.lm.forward = functools.update_wrapper(
            functools.partial(progressive_forward, self.lm), original_forward)
        try:
            return self._generate_from_embeds(inputs.full_embeds, max_new_tokens)
        finally:
            self.lm.forward = original_forward

    def _generate_sparsevlm_from_embeds(
        self,
        inputs: BackendInputs,
        budget_percent: float,
        max_new_tokens: int,
        rater_ratio: float = 0.5,
        alpha: float = 0.5,
    ):
        stage_targets, final_keep = self._sparsevlm_stage_targets(len(inputs.vis_pos), budget_percent)
        if not stage_targets:
            return self._generate_from_embeds(inputs.full_embeds, max_new_tokens), len(inputs.vis_pos)
        original_forward = self.lm.forward
        kept_count = {"value": len(inputs.vis_pos)}
        final_stage = max(stage_targets)

        def sparsevlm_forward(text_model, *args, **kwargs):
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)
            past_key_values = kwargs.get("past_key_values", None)
            inputs_embeds = kwargs.get("inputs_embeds", None)
            use_cache = kwargs.get("use_cache", None)
            extra = {k: v for k, v in kwargs.items()
                     if k not in {"input_ids", "attention_mask", "position_ids",
                                  "past_key_values", "inputs_embeds", "use_cache"}}
            if inputs_embeds is None:
                inputs_embeds = text_model.embed_tokens(input_ids)
            if inputs_embeds.shape[1] <= 1 or past_key_values is not None:
                return original_forward(*args, **kwargs)

            hidden_states = inputs_embeds
            if position_ids is None:
                position_ids = torch.arange(
                    hidden_states.shape[1], device=hidden_states.device).reshape(1, -1)
            vis_positions = list(inputs.vis_pos)
            text_positions = [p for p in range(hidden_states.shape[1]) if p not in set(vis_positions)]
            rater_positions = None
            position_embeddings = text_model.rotary_emb(hidden_states, position_ids)

            for layer_idx, decoder_layer in enumerate(text_model.layers):
                if layer_idx in stage_targets and vis_positions and text_positions:
                    if rater_positions is None:
                        n_raters = max(1, round(len(text_positions) * rater_ratio))
                        rater_scores = generic_rotary_cross_attention_scores(
                            hidden_states,
                            vis_positions,
                            text_positions,
                            decoder_layer.self_attn,
                            layer_module=decoder_layer,
                            position_embeddings=position_embeddings,
                        )
                        top = rater_scores.argsort(descending=True)[:n_raters].tolist()
                        rater_positions = [text_positions[int(i)] for i in top]

                    n_current = len(vis_positions)
                    erank = effective_rank(hidden_states[0, vis_positions, :])
                    erank_ratio = min(1.0, erank / max(1, n_current))
                    adaptive = round(n_current * (max(0.0, min(1.0, float(budget_percent) / 100.0))) * (erank_ratio ** alpha))
                    if layer_idx == final_stage:
                        n_keep = stage_targets[layer_idx]
                    else:
                        n_keep = max(stage_targets[layer_idx], adaptive, 1)
                    n_keep = min(n_current, int(n_keep))
                    if n_current > n_keep:
                        scores = generic_rotary_cross_attention_scores(
                            hidden_states,
                            rater_positions,
                            vis_positions,
                            decoder_layer.self_attn,
                            layer_module=decoder_layer,
                            position_embeddings=position_embeddings,
                        )
                        ranked = scores.argsort(descending=True).tolist()
                        keep_local = set(int(i) for i in ranked[:n_keep])
                        drop_local = [int(i) for i in ranked[n_keep:]]
                        keep_positions = [p for i, p in enumerate(vis_positions) if i in keep_local]
                        drop_positions = [vis_positions[i] for i in drop_local]
                        drop_scores = scores[drop_local] if drop_local else torch.empty(0)
                        hidden_states = recycle_visual_hidden(
                            hidden_states, keep_positions, drop_positions, drop_scores)
                        drop_set = set(drop_positions)
                        keep_seq = [p for p in range(hidden_states.shape[1]) if p not in drop_set]
                        idx = torch.tensor(keep_seq, dtype=torch.long, device=hidden_states.device)
                        hidden_states = hidden_states[:, idx, :]
                        position_ids = position_ids[:, idx]
                        cos, sin = position_embeddings
                        if cos.dim() == 4:
                            position_embeddings = (cos[:, :, idx.to(cos.device), :], sin[:, :, idx.to(sin.device), :])
                        else:
                            position_embeddings = (cos[:, idx.to(cos.device), :], sin[:, idx.to(sin.device), :])
                        old_to_new = {old: new for new, old in enumerate(keep_seq)}
                        vis_positions = [old_to_new[p] for p in vis_positions if p not in drop_set]
                        text_positions = [old_to_new[p] for p in text_positions]
                        rater_positions = [old_to_new[p] for p in rater_positions]
                        kept_count["value"] = len(vis_positions)
                        attention_mask = None

                layer_out = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **extra,
                )
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            hidden_states = text_model.norm(hidden_states)
            return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

        self.lm.forward = functools.update_wrapper(
            functools.partial(sparsevlm_forward, self.lm), original_forward)
        try:
            return self._generate_from_embeds(inputs.full_embeds, max_new_tokens), kept_count["value"]
        finally:
            self.lm.forward = original_forward

    def gradient_oracle(self, inputs: BackendInputs, answer_ids: torch.Tensor, max_answer_tokens: int,
                        score_mode: str = "sensitivity"):
        answer_ids = answer_ids[:max_answer_tokens].to(inputs.raw["input_ids"].device)
        if answer_ids.numel() == 0:
            raise ValueError("empty generated answer")
        answer_embeds = self.lm.embed_tokens(answer_ids.unsqueeze(0)).detach().to(inputs.full_embeds.dtype)
        visual = inputs.visual_embeds.detach().clone().requires_grad_(True)
        full = inputs.full_embeds.detach().clone()
        for i, pos in enumerate(inputs.vis_pos[: visual.shape[0]]):
            full[:, pos, :] = visual[i]
        full = torch.cat([full, answer_embeds], dim=1)
        self.model.zero_grad(set_to_none=True)
        out = self.lm(inputs_embeds=full, use_cache=False, return_dict=True)
        logits = self.lm_head(out.last_hidden_state)
        prompt_len = inputs.raw["input_ids"].shape[1]
        answer_len = answer_ids.numel()
        pred_logits = logits[:, prompt_len - 1: prompt_len + answer_len - 1, :].float()
        loss = F.cross_entropy(pred_logits.reshape(-1, pred_logits.shape[-1]),
                               answer_ids.unsqueeze(0).reshape(-1), reduction="mean")
        loss.backward()
        grad = visual.grad
        if grad is None:
            raise RuntimeError("visual embedding gradient is None")
        if score_mode == "directional":
            imp = torch.clamp(
                -(grad.float() * visual.detach().float()).sum(dim=-1),
                min=0.0,
            )
        else:
            imp = grad.float().norm(dim=-1) * visual.detach().float().norm(dim=-1)
        self.model.zero_grad(set_to_none=True)
        return imp.detach().cpu().tolist()


class LlavaBackend(EmbeddingPruneBackend):
    image_token_id = 32000
    supports_multiround_cache = True

    def __init__(self, model_path: str):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        print(f"Loading LLaVA model from {model_path}...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_path)
        kwargs = {"torch_dtype": torch.float16, "device_map": "cuda"}
        if os.environ.get("DUALSIGNAL_EAGER_ATTENTION", "1") == "1":
            kwargs["attn_implementation"] = "eager"
        self.model = LlavaForConditionalGeneration.from_pretrained(model_path, **kwargs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.lm = self.model.model.language_model
        self.lm_head = self.model.lm_head
        self.total_layers = len(self.lm.layers)
        self.eos_id = self.processor.tokenizer.eos_token_id
        print("Model loaded.", flush=True)

    def prepare_inputs_any(self, images: list[Image.Image], prompt: str,
                           max_pixels: int | None, multi_max_pixels: int | None):
        image_tags = "\n".join("<image>" for _ in images)
        text = f"USER: {image_tags}\n{prompt}\nASSISTANT:"
        proc_images = images[0] if len(images) == 1 else images
        raw = self.processor(text=text, images=proc_images, return_tensors="pt").to("cuda")
        if "pixel_values" in raw and raw["pixel_values"].dim() == 5:
            pv = raw["pixel_values"]
            raw["pixel_values"] = pv.reshape(-1, *pv.shape[-3:])
        input_ids = raw["input_ids"]
        with torch.no_grad():
            pixel_values = raw["pixel_values"].to(self.model.dtype)
            feature_out = self.model.model.get_image_features(
                pixel_values,
                vision_feature_layer=self.model.config.vision_feature_layer,
                vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
            )
            image_features = getattr(feature_out, "pooler_output", feature_out)
            if not hasattr(feature_out, "hidden_states"):
                score_features = image_features
            elif isinstance(self.model.config.vision_feature_layer, int):
                score_features = feature_out.hidden_states[self.model.config.vision_feature_layer]
                if self.model.config.vision_feature_select_strategy == "default":
                    score_features = score_features[:, 1:]
            else:
                score_pool = [
                    feature_out.hidden_states[layer_idx]
                    for layer_idx in self.model.config.vision_feature_layer
                ]
                if self.model.config.vision_feature_select_strategy == "default":
                    score_pool = [x[:, 1:] for x in score_pool]
                score_features = torch.cat(score_pool, dim=-1)
            if isinstance(score_features, (list, tuple)):
                score_features = torch.cat([x.reshape(-1, x.shape[-1]) for x in score_features], dim=0)
            score_features = score_features.reshape(-1, score_features.shape[-1])
            if isinstance(image_features, (list, tuple)):
                image_features = torch.cat([x.reshape(-1, x.shape[-1]) for x in image_features], dim=0)
            image_features = image_features.reshape(-1, image_features.shape[-1])
            text_emb = self.lm.embed_tokens(input_ids)
            img_positions = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
            n_vis = min(len(img_positions), image_features.shape[0])
            full = text_emb.clone()
            full[0, img_positions[:n_vis]] = image_features[:n_vis].to(text_emb.dtype)
        vis_pos = img_positions[:n_vis].tolist()
        text_pos = [p for p in range(input_ids.shape[1]) if p not in set(vis_pos) and (not vis_pos or p > max(vis_pos))]
        return BackendInputs(
            raw,
            full,
            image_features[:n_vis].detach(),
            vis_pos,
            text_pos,
            {"score_visual_embeds": score_features[:n_vis].detach()},
        )

    @torch.no_grad()
    def generate_ids_and_text(self, inputs: BackendInputs, max_new_tokens: int):
        out_ids = self.model.generate(**inputs.raw, max_new_tokens=max_new_tokens, do_sample=False)
        generated = out_ids[0, inputs.raw["input_ids"].shape[1]:].detach()
        text = self.processor.decode(generated, skip_special_tokens=True).strip()
        return generated, text

    def _generate_from_embeds(self, embeds: torch.Tensor, max_new_tokens: int):
        return self._generate_with_model_from_embeds(embeds, max_new_tokens)

    def generate_sparsevila_from_multiround_cache(
        self,
        cache_entry,
        inputs: BackendInputs,
        vis_embeds,
        grid_thw,
        max_new_tokens: int,
        decode_keep_ratio: float = 0.5,
    ):
        return super().generate_sparsevila_from_multiround_cache(
            cache_entry,
            inputs,
            vis_embeds,
            grid_thw,
            max_new_tokens,
            decode_keep_ratio=decode_keep_ratio,
        )

    @torch.no_grad()
    def visual_received_attention_scores(self, inputs: BackendInputs, n_vis: int):
        raw = inputs.raw
        pixel_values = raw.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError("missing pixel_values for LLaVA vision attention")
        vision = self.model.model.vision_tower
        try:
            out = vision(
                pixel_values.to(self.model.dtype),
                output_hidden_states=False,
                output_attentions=True,
                return_dict=True,
            )
            attentions = getattr(out, "attentions", None)
        except Exception as exc:
            return self._visual_score_fallback(inputs, n_vis, f"{type(exc).__name__}: {exc}")
        if not attentions:
            return self._visual_score_fallback(inputs, n_vis, "vision tower did not return attentions")
        layer = self.model.config.vision_feature_layer
        if isinstance(layer, (list, tuple)):
            layer = layer[-1]
        attn = attentions[int(layer)].float()
        if attn.dim() == 4 and attn.shape[-2] > 1 and self.model.config.vision_feature_select_strategy == "default":
            scores = attn[:, :, 0, 1:].mean(dim=(0, 1)).cpu()
        else:
            scores = attn.mean(dim=-2).mean(dim=(0, 1)).cpu()
        return _align_score_length(scores, n_vis).tolist()


class InternVLBackend(EmbeddingPruneBackend):
    supports_multiround_cache = True

    def __init__(self, model_path: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        print(f"Loading InternVL model from {model_path}...", flush=True)
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "cuda"}
        if os.environ.get("DUALSIGNAL_EAGER_ATTENTION", "1") == "1":
            kwargs["attn_implementation"] = "eager"
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.lm = self.model.model.language_model
        self.lm_head = self.model.lm_head
        self.total_layers = len(self.lm.layers)
        self.image_token_id = getattr(self.model.config, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.eos_id = self.processor.tokenizer.eos_token_id
        print("Model loaded.", flush=True)

    def prepare_inputs_any(self, images: list[Image.Image], prompt: str,
                           max_pixels: int | None, multi_max_pixels: int | None):
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        raw = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        input_ids = raw["input_ids"]
        pixel_values = raw.get("pixel_values")
        with torch.no_grad():
            vis_out = self.model.model.get_image_features(pixel_values.to(self.model.dtype))
            visual = getattr(vis_out, "pooler_output", vis_out)
            visual = visual.reshape(-1, visual.shape[-1])
            dtype = self.lm.layers[0].self_attn.q_proj.weight.dtype
            text_emb = self.lm.embed_tokens(input_ids).to(dtype)
            full = text_emb.clone()
            img_positions = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
            n_vis = min(len(img_positions), visual.shape[0])
            full[0, img_positions[:n_vis]] = visual[:n_vis].to(dtype)
        vis_pos = img_positions[:n_vis].tolist()
        text_pos = [p for p in range(input_ids.shape[1]) if p not in set(vis_pos) and (not vis_pos or p > max(vis_pos))]
        return BackendInputs(
            raw,
            full,
            visual[:n_vis].detach(),
            vis_pos,
            text_pos,
            {"score_visual_embeds": visual[:n_vis].detach()},
        )

    @torch.no_grad()
    def generate_ids_and_text(self, inputs: BackendInputs, max_new_tokens: int):
        out_ids = self.model.generate(**inputs.raw, max_new_tokens=max_new_tokens, do_sample=False)
        generated = out_ids[0, inputs.raw["input_ids"].shape[1]:].detach()
        text = self.processor.decode(generated, skip_special_tokens=True).strip()
        return generated, text

    def _generate_from_embeds(self, embeds: torch.Tensor, max_new_tokens: int):
        return self._generate_with_model_from_embeds(embeds, max_new_tokens)

    def generate_sparsevila_from_multiround_cache(
        self,
        cache_entry,
        inputs: BackendInputs,
        vis_embeds,
        grid_thw,
        max_new_tokens: int,
        decode_keep_ratio: float = 0.5,
    ):
        return super().generate_sparsevila_from_multiround_cache(
            cache_entry,
            inputs,
            vis_embeds,
            grid_thw,
            max_new_tokens,
            decode_keep_ratio=decode_keep_ratio,
        )

    @torch.no_grad()
    def visual_received_attention_scores(self, inputs: BackendInputs, n_vis: int):
        pixel_values = inputs.raw.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError("missing pixel_values for InternVL vision attention")
        try:
            out = self.model.model.vision_tower(
                pixel_values=pixel_values.to(self.model.dtype),
                output_attentions=True,
                return_dict=True,
            )
            attentions = getattr(out, "attentions", None)
        except Exception as exc:
            return self._visual_score_fallback(inputs, n_vis, f"{type(exc).__name__}: {exc}")
        if not attentions:
            return self._visual_score_fallback(inputs, n_vis, "vision tower did not return attentions")
        layer = getattr(self.model.config, "vision_feature_layer", -1)
        if isinstance(layer, (list, tuple)):
            layer = layer[-1]
        attn = attentions[int(layer)].float()
        if attn.dim() == 4 and attn.shape[-1] == attn.shape[-2] and attn.shape[-1] > 1:
            scores = attn[:, :, 0, 1:].mean(dim=(0, 1)).cpu()
        else:
            scores = attn.mean(dim=-2).mean(dim=(0, 1)).cpu()
        return _align_score_length(scores, n_vis).tolist()


def _align_score_length(scores: torch.Tensor, n_vis: int) -> torch.Tensor:
    scores = scores.float().cpu().reshape(-1)
    if scores.numel() == n_vis:
        return norm01_t(scores)
    if scores.numel() > n_vis and scores.numel() % max(1, n_vis) == 0:
        group = scores.numel() // max(1, n_vis)
        return norm01_t(scores[: n_vis * group].view(n_vis, group).mean(dim=1))
    if scores.numel() < n_vis and n_vis % max(1, scores.numel()) == 0:
        return norm01_t(scores.repeat_interleave(n_vis // max(1, scores.numel()))[:n_vis])
    if scores.numel() == 0:
        return torch.zeros(n_vis)
    resized = torch.nn.functional.interpolate(
        scores.reshape(1, 1, -1), size=n_vis, mode="linear", align_corners=False)
    return norm01_t(resized.reshape(-1))


def infer_backend_type(model_path: str) -> str:
    s = model_path.lower()
    if "llava" in s:
        return "llava"
    if "internvl" in s:
        return "internvl"
    return "qwen"


def load_backend(model_path: str, backend_type: str | None = None):
    kind = backend_type or infer_backend_type(model_path)
    if kind == "qwen":
        return QwenBackend(model_path)
    if kind == "llava":
        return LlavaBackend(model_path)
    if kind in {"internvl", "internvl3"}:
        return InternVLBackend(model_path)
    raise ValueError(f"Unknown backend type: {kind}")
