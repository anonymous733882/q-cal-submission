#!/usr/bin/env python
"""
多轮问答剪枝策略对比实验 — 等 token×layer 代价公平比较

方法对比:
  1. Ours (dual-stage):
     - Stage 1: L0 attention → 保留 40% (query-agnostic, 多轮共享)
     - Stage 2: L_S attention → 保留总 token 的 10% (query-dependent, 每轮独立)
     - Cost: 0.40×S + 0.10×(N−S)
  2. PACT (CVPR 2025): EUTI + DBDPC → 保留 Y% (KV cache reuse)
  3. SparseVILA (ICCV 2025): ViT attention top-k → 保留 Y% + decode salience (KV cache reuse)
  4. Attention top-k at L0 → 保留 Y% (KV cache reuse)

其中 Y = (0.40×S + 0.10×(N−S)) / N，保证 token×layer 代价相同。

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u strategy_test/multi_round_benchmark.py \\
        --model qwen25vl --num_images 30

    # 多轮 token 多样性分析 (可选):
    CUDA_VISIBLE_DEVICES=0 python -u strategy_test/multi_round_benchmark.py \\
        --model qwen25vl --num_images 10 --multi_round
"""

import argparse, os, sys, json, random, gc, string, time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(42)

STAGE1_FRAC = 0.33
STAGE2_FRAC = 0.10


def _new_cost_bucket():
    return {"prefill_tl": [], "gen_tl": [], "total_tl": [],
            "n_vis_decode": [], "n_gen_tokens": [],
            "shared_setup_sec": [], "shared_question_sec": [],
            "policy_prefill_wall_sec": [], "policy_decode_wall_sec": [],
            "policy_total_wall_sec": []}


def _decoded_token_count(tokenizer, text):
    """Count generated text tokens with the same decoded-text policy as baseline runners."""
    if tokenizer is None:
        return max(1, len(str(text).split()))
    try:
        return int(len(tokenizer.encode(str(text), add_special_tokens=False)))
    except TypeError:
        return int(len(tokenizer.encode(str(text))))
    except Exception:
        return max(1, len(str(text).split()))


def _dualsignal_resize_image(image):
    resize_square = os.environ.get("DUALSIGNAL_RESIZE_SQUARE")
    if not resize_square:
        return image
    try:
        side = int(resize_square)
    except Exception:
        return image
    if side <= 0 or not isinstance(image, Image.Image):
        return image
    img = image.convert("RGB")
    target = (side, side)
    if img.size == target:
        return img
    return img.resize(target, Image.Resampling.BICUBIC)

# ── 模型配置: N=总层数, S=Stage2剪枝层, scoring=打分策略 ──
# scoring: "single" = 只在layer S打分 (1次QK)
#          "max"    = layers 0..S-1 每层打分取max (S次QK)
MODEL_CONFIG = {
    "qwen25vl":       {"N": 28, "S": 22, "scoring": "single", "type": "qwen",      "model_id": "checkpoints/Qwen2.5-VL-7B-Instruct"},
    "qwen3vl":        {"N": 36, "S": 29, "scoring": "single", "type": "qwen",      "model_id": "Qwen/Qwen3-VL-8B-Instruct"},
    "internvl3-8b":   {"N": 28, "S": 23, "scoring": "single", "type": "internvl3", "model_id": "OpenGVLab/InternVL3-8B-hf"},
    "internvl3.5-8b": {"N": 36, "S": 23, "scoring": "single", "type": "internvl3", "model_id": "OpenGVLab/InternVL3_5-8B-hf"},
    "llava-7b":       {"N": 32, "S": 20, "scoring": "single", "type": "llava",     "model_id": "llava-hf/llava-1.5-7b-hf"},
    "llava-13b":      {"N": 40, "S": 26, "scoring": "max",    "type": "llava",     "model_id": "llava-hf/llava-1.5-13b-hf"},
}

for _k, _c in MODEL_CONFIG.items():
    _N, _S = _c["N"], _c["S"]
    _c["cost"] = STAGE1_FRAC * _S + STAGE2_FRAC * (_N - _S)
    _c["Y"] = _c["cost"] / _N


# ───────────────────────────── Scoring ─────────────────────────────

_ARTICLES = {"a", "an", "the"}
_PUNCT = set(string.punctuation)

def _normalize(text):
    text = text.lower()
    text = "".join(c if c not in _PUNCT else " " for c in text)
    return " ".join(t for t in text.split() if t not in _ARTICLES).strip()

def score_textvqa(pred, gt_answers):
    p = _normalize(pred)
    exact = sum(1 for a in gt_answers if _normalize(a) == p)
    if exact > 0:
        return min(exact / 3.0, 1.0)
    contained = sum(1 for a in gt_answers if _normalize(a) and _normalize(a) in p)
    return min(contained / 3.0, 1.0) * 0.5


def score_gqa(pred, gt_answer):
    """GQA scoring: exact match or containment after normalization."""
    p = _normalize(pred)
    g = _normalize(gt_answer)
    if not g:
        return 0.0
    if p == g:
        return 1.0
    # Check first word/phrase match (model often starts with the answer)
    p_words = p.split()
    g_words = g.split()
    if p_words[:len(g_words)] == g_words:
        return 1.0
    return 0.0


GQA_SUFFIX = " Answer with a single word or short phrase."


# ───────────────────── Importance at layer L ──────────────────────

def _get_importance_at_layer(lm, layer_idx, inputs_embeds, vis_pos, text_pos):
    """text→vis attention importance at layer L (Q@K, no RoPE)."""
    layer = lm.layers[layer_idx]
    attn = getattr(layer, 'self_attn', None) or layer.attention
    ln = getattr(layer, 'input_layernorm', None) or layer.attention_norm
    seq_len = inputs_embeds.shape[1]

    has_fused_qkv = hasattr(attn, 'wqkv')
    if has_fused_qkv:
        cfg = attn.config if hasattr(attn, 'config') else lm.config
        n_heads = cfg.num_attention_heads
        n_kv_heads = cfg.num_key_value_heads
        head_dim = cfg.hidden_size // n_heads
        with torch.no_grad():
            hidden = ln(inputs_embeds[0])
            qkv = attn.wqkv(hidden).float()
            q_size = n_heads * head_dim
            kv_size = n_kv_heads * head_dim
            q_raw = qkv[:, :q_size]
            k_raw = qkv[:, q_size:q_size + kv_size]
    else:
        n_heads = getattr(attn, 'num_heads', None) or attn.config.num_attention_heads
        n_kv_heads = getattr(attn, 'num_key_value_heads', None) or attn.config.num_key_value_heads
        head_dim = getattr(attn, 'head_dim', None) or (attn.config.hidden_size // n_heads)
        with torch.no_grad():
            hidden = ln(inputs_embeds[0])
            q_raw = attn.q_proj(hidden).float()
            k_raw = attn.k_proj(hidden).float()

    q = q_raw.view(seq_len, n_heads, head_dim).permute(1, 0, 2)
    k = k_raw.view(seq_len, n_kv_heads, head_dim).permute(1, 0, 2)
    if n_kv_heads < n_heads:
        k = k.repeat_interleave(n_heads // n_kv_heads, dim=0)

    q_text = q[:, text_pos, :]
    k_vis = k[:, vis_pos, :]
    scores = (q_text @ k_vis.transpose(-1, -2)) * (head_dim ** -0.5)
    weights = F.softmax(scores, dim=-1)
    return weights.mean(dim=(0, 1)).cpu().numpy()


# ───────── Intent-adaptive one-shot pruning ─────────

def classify_intent_v2(query: str) -> str:
    """Classify query into 4 intent categories for adaptive weight selection."""
    q = query.lower()

    GLOBAL_HINTS = [
        "is there", "are there", "how many", "how much",
        "where is", "where are",
        "describe", "what is happening", "what is going on",
        "what are the people doing", "what are they doing",
        "what should", "which action", "safest", "most likely",
        "what is this place", "what type of scene",
    ]
    if any(h in q for h in GLOBAL_HINTS):
        return "global_scan"

    DENSE_HINTS = [
        "document", "page", "form", "report", "article",
        "chart", "graph", "plot", "diagram", "figure",
        "table", "spreadsheet", "receipt", "invoice",
        "according to", "based on the", "as shown in",
        "what is the total", "what is the percentage",
        "how much does", "what is the value",
    ]
    if any(h in q for h in DENSE_HINTS):
        return "dense_content"

    FOCUS_PATTERNS = [
        "what does the sign say", "what is written on",
        "read the", "what text is", "what word",
        "what does it say", "what number is on",
        "is it true that", "is the",
        "what color is the", "what brand is",
        "what logo", "what is the name on",
    ]
    if any(h in q for h in FOCUS_PATTERNS):
        return "focused_detail"

    return "balanced"


# ViT weight lookup: intent -> budget -> vit_frac
_INTENT_VIT_WEIGHT = {
    "global_scan":    {0.05: 1.0, 0.10: 1.0, 0.20: 0.8, 0.33: 0.7},
    "dense_content":  {0.05: 0.5, 0.10: 0.5, 0.20: 0.5, 0.33: 0.5},
    "focused_detail": {0.05: 0.0, 0.10: 0.2, 0.20: 0.3, 0.33: 0.5},
    "balanced":       {0.05: 0.5, 0.10: 0.5, 0.20: 0.5, 0.33: 0.5},
}


def _get_vit_frac_for_budget(intent: str, budget: float) -> float:
    """Interpolate ViT fraction for a given intent and budget."""
    table = _INTENT_VIT_WEIGHT[intent]
    budgets = sorted(table.keys())
    if budget <= budgets[0]:
        return table[budgets[0]]
    if budget >= budgets[-1]:
        return table[budgets[-1]]
    for i in range(len(budgets) - 1):
        if budgets[i] <= budget <= budgets[i + 1]:
            t = (budget - budgets[i]) / (budgets[i + 1] - budgets[i])
            return table[budgets[i]] * (1 - t) + table[budgets[i + 1]] * t
    return 0.5


def _oneshot_adaptive_select(hawk_importance: np.ndarray,
                             vit_importance: np.ndarray,
                             vis_embeds: torch.Tensor,
                             k: int,
                             vit_frac: float) -> list[bool]:
    """One-shot adaptive selection: HAWK + ViT with intent-adaptive weights + MMR.

    Args:
        hawk_importance: [n_vis] HAWK L0 text→vis attention scores
        vit_importance:  [n_vis] ViT last-block received attention scores
        vis_embeds:      [n_vis, dim] visual token embeddings (for MMR)
        k:               number of tokens to keep
        vit_frac:        fraction of ViT weight (0.0 = pure HAWK, 1.0 = pure ViT/FPS)

    Returns:
        Boolean mask of length n_vis (True=keep).
    """
    n = len(hawk_importance)
    if k >= n:
        return [True] * n

    hawk_frac = 1.0 - vit_frac
    combined = hawk_frac * _norm01(hawk_importance) + vit_frac * _norm01(vit_importance)

    # Multiplicative MMR diversity suppression
    emb = vis_embeds.float().cpu()
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    sim_matrix = (emb @ emb.T).numpy()

    current_scores = combined.copy()
    selected = []
    available = np.ones(n, dtype=bool)

    for _ in range(k):
        masked = np.where(available, current_scores, -np.inf)
        best = int(np.argmax(masked))
        selected.append(best)
        available[best] = False
        suppression = np.clip(1.0 - sim_matrix[best], 0.0, None)
        current_scores *= suppression

    selected_set = set(selected)
    return [i in selected_set for i in range(n)]


# ───────── Dual-signal Stage 1: L0 attn + ViT attn + MMR ─────────

def _norm01(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _dual_signal_stage1_select(l0_importance: np.ndarray,
                               vit_importance: np.ndarray,
                               vis_embeds: torch.Tensor,
                               k: int) -> list[int]:
    """Dual-signal selection: LLM L0 attention + ViT last-block attention,
    equal weight, with multiplicative MMR diversity suppression.

    Args:
        l0_importance  : [n_vis] LLM layer 0 text→vis attention scores
        vit_importance : [n_vis] ViT last-block received attention scores
        vis_embeds     : [n_vis, dim] visual token embeddings (for MMR)
        k              : number of tokens to keep

    Returns:
        Sorted list of selected token indices.
    """
    n = len(l0_importance)
    if k >= n:
        return list(range(n))

    # Equal-weight combination of two signals
    combined = 0.5 * _norm01(l0_importance) + 0.5 * _norm01(vit_importance)

    # Multiplicative MMR diversity suppression
    emb = vis_embeds.float().cpu()
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    sim_matrix = (emb @ emb.T).numpy()

    current_scores = combined.copy()
    selected = []
    available = np.ones(n, dtype=bool)

    for _ in range(k):
        masked = np.where(available, current_scores, -np.inf)
        best = int(np.argmax(masked))
        selected.append(best)
        available[best] = False
        # Suppress tokens similar to the selected one
        suppression = np.clip(1.0 - sim_matrix[best], 0.0, None)
        current_scores *= suppression

    return sorted(selected)


# ───────────────── PACT selection (EUTI + DBDPC) ─────────────────

def _pact_select(lm, inputs_embeds, vis_positions, k):
    """PACT: EUTI score at L0 + density-peak clustering."""
    n = len(vis_positions)
    if k >= n:
        return list(range(n))

    attn0 = lm.layers[0].self_attn
    ln0 = getattr(lm.layers[0], 'input_layernorm', None) or lm.layers[0].attention_norm
    head_dim = getattr(attn0, 'head_dim', None)
    if head_dim is None:
        n_h = getattr(attn0, 'num_heads', None) or attn0.config.num_attention_heads
        head_dim = attn0.config.hidden_size // n_h

    with torch.no_grad():
        hidden = ln0(inputs_embeds[0])
        if hasattr(attn0, 'wqkv'):
            cfg = attn0.config if hasattr(attn0, 'config') else lm.config
            nh = cfg.num_attention_heads
            nkv = cfg.num_key_value_heads
            hd = cfg.hidden_size // nh
            qkv = attn0.wqkv(hidden).float()
            k_proj = qkv[:, nh * hd:nh * hd + nkv * hd]
        else:
            k_proj = attn0.k_proj(hidden).float()
        h_vis = hidden[vis_positions].float()

    k_vis = k_proj[vis_positions]
    global_q = k_vis.mean(dim=0, keepdim=True)
    relevance = F.softmax((k_vis * global_q).sum(dim=-1) / (head_dim ** 0.5), dim=0)
    norms = h_vis.norm(dim=-1)
    norms = norms / norms.max().clamp(min=1e-6)
    euti = (relevance * norms).cpu().numpy()

    n_cand = min(n, k * 2)
    top_idx = np.argsort(euti)[::-1][:n_cand].tolist()
    if n_cand <= k:
        return sorted(top_idx[:k])

    cand_emb = h_vis[top_idx].cpu()
    cand_emb = F.normalize(cand_emb, dim=-1).numpy()
    selected = [0]
    min_dists = 1.0 - (cand_emb @ cand_emb[0])
    for _ in range(k - 1):
        best = int(np.argmax(min_dists))
        selected.append(best)
        new_dists = 1.0 - (cand_emb @ cand_emb[best])
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[selected] = -1.0

    return sorted([top_idx[i] for i in selected])


# ──────────────── SVDPrune / DivPrune / VisPruner / FastV ────────────

def _svdprune_select(vis_embeds, k_y):
    """SVDPrune: leverage-score based visual token selection (query-agnostic)."""
    from baselines.svdprune.adapter import SVDPruneBaseline
    svd = SVDPruneBaseline(epsilon=0.9)
    scores = svd.score_tokens(vis_embeds)
    n_vis = vis_embeds.shape[0]
    mask = svd.allocate(scores, n_vis, k_y / n_vis)
    return mask  # list[bool]


def _divprune_select(vis_embeds, k_y):
    """DivPrune: max-min cosine diversity selection (query-agnostic)."""
    from baselines.divprune.adapter import DivPruneBaseline
    n_vis = vis_embeds.shape[0]
    div = DivPruneBaseline(prune_ratio=1.0 - k_y / n_vis)
    mask = div.prune(None, None, vis_embeds, None,
                     prune_ratio=1.0 - k_y / n_vis)
    return mask  # list[bool]


def _vispruner_select(vit_imp, vis_embeds, k_y,
                      importance_ratio=0.8):
    """VisPruner (ICCV 2025): ViT importance + similarity-based dedup.

    Paper: "Beyond Text-Visual Attention: Exploiting Visual Cues for
    Effective Token Pruning in VLMs" (Zhang et al. 2025)

    Stage 1: Select top (k_y * importance_ratio) tokens by ViT attention.
    Stage 2 (Algorithm 1): From remaining tokens, iteratively remove the
             most similar pairs until n_diverse tokens are left.
    """
    n_vis = len(vit_imp)
    n_important = max(1, int(k_y * importance_ratio))
    ranked = np.argsort(vit_imp)[::-1].tolist()
    important_set = set(ranked[:n_important])
    remaining_idx = [i for i in ranked[n_important:]]

    n_diverse = k_y - n_important
    if n_diverse <= 0 or not remaining_idx:
        keep_set = important_set
    else:
        # Algorithm 1: similarity-based duplication removal among remaining
        emb = F.normalize(vis_embeds.float().cpu(), dim=-1)
        rem = list(remaining_idx)
        n_to_remove = len(rem) - n_diverse
        while n_to_remove > 0 and len(rem) > n_diverse:
            rem_emb = emb[rem]  # [n_rem, D]
            # Split into pairs: even/odd indexed
            a_idx = list(range(0, len(rem), 2))
            b_idx = list(range(1, len(rem), 2))
            if not b_idx:
                break
            a_emb = rem_emb[a_idx]  # [n_a, D]
            b_emb = rem_emb[b_idx]  # [n_b, D]
            # Similarity between each a and all b
            sim = torch.matmul(a_emb, b_emb.T)  # [n_a, n_b]
            max_sim = sim.max(dim=-1).values      # [n_a]
            # Remove the most similar ones (highest sim → least diverse)
            r = min(n_to_remove, len(a_idx))
            if r <= 0:
                break
            # Keep the least similar a's (those with lowest max_sim)
            keep_a = max_sim.argsort()[:len(a_idx) - r].tolist()
            kept_a = [rem[a_idx[j]] for j in keep_a]
            kept_b = [rem[b_idx[j]] for j in range(len(b_idx))]
            rem = sorted(kept_a + kept_b)
            n_to_remove = len(rem) - n_diverse
        keep_set = important_set | set(rem[:n_diverse])

    return [i in keep_set for i in range(n_vis)]


def _run_layers(lm, hidden, start, end):
    """Run LLM layers start..end-1 on hidden states, returning updated hidden.

    Handles models that require position_embeddings by computing sequential
    rotary embeddings. Supports both Qwen2.5/3-VL (3D pos_ids) and standard
    Qwen2/InternVL3 (2D pos_ids).
    """
    for l_idx in range(start, end):
        layer = lm.layers[l_idx]
        try:
            out = layer(hidden, use_cache=False)
        except TypeError:
            # Layer requires position_embeddings — compute them
            seq_len = hidden.shape[1]
            device = hidden.device
            pos_ids_2d = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq]
            try:
                # Standard Qwen2 / InternVL3: 2D position_ids
                pos_emb = lm.rotary_emb(hidden, pos_ids_2d)
                out = layer(hidden, position_embeddings=pos_emb, use_cache=False)
            except (RuntimeError, IndexError):
                # Qwen2.5-VL / Qwen3-VL: 3D position_ids [3, batch, seq]
                pos_ids_3d = pos_ids_2d.unsqueeze(0).expand(3, 1, -1)
                pos_emb = lm.rotary_emb(hidden, pos_ids_3d)
                out = layer(hidden, position_embeddings=pos_emb, use_cache=False)
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden


def _fastv_select(lm, full_embeds, vis_pos, text_pos, k_y, layer=2):
    """FastV (ECCV 2024): received-attention based visual token pruning.

    Paper: "An Image is Worth 1/2 Tokens After Layer 2" (Chen et al. 2024)
    Scoring: "average attention-score one token received from all other
             tokens" at layer K (1-indexed).
    layer=2 (1-indexed) → score at lm.layers[1] (0-indexed),
    using hidden states after forward through layers 0..0.

    In multi-round setting, computed once with first question, then treated
    as query-agnostic for KV cache reuse.
    """
    # Paper's "layer K" is 1-indexed → 0-indexed = K-1
    score_layer_idx = min(layer - 1, len(lm.layers) - 1)

    # Forward through layers 0..(score_layer_idx-1) to get actual hidden states
    with torch.no_grad():
        hidden = _run_layers(lm, full_embeds, 0, score_layer_idx)

    # Score at score_layer_idx using actual hidden states
    score_layer = lm.layers[score_layer_idx]
    attn = getattr(score_layer, 'self_attn', None) or score_layer.attention
    ln = getattr(score_layer, 'input_layernorm', None) or score_layer.attention_norm
    seq_len = hidden.shape[1]

    has_fused_qkv = hasattr(attn, 'wqkv')
    if has_fused_qkv:
        cfg = attn.config if hasattr(attn, 'config') else lm.config
        n_heads = cfg.num_attention_heads
        n_kv_heads = cfg.num_key_value_heads
        head_dim = cfg.hidden_size // n_heads
        with torch.no_grad():
            h = ln(hidden[0])
            qkv = attn.wqkv(h).float()
            q_size = n_heads * head_dim
            kv_size = n_kv_heads * head_dim
            q_raw = qkv[:, :q_size]
            k_raw = qkv[:, q_size:q_size + kv_size]
    else:
        n_heads = getattr(attn, 'num_heads', None) or attn.config.num_attention_heads
        n_kv_heads = getattr(attn, 'num_key_value_heads', None) or attn.config.num_key_value_heads
        head_dim = getattr(attn, 'head_dim', None) or (attn.config.hidden_size // n_heads)
        with torch.no_grad():
            h = ln(hidden[0])
            q_raw = attn.q_proj(h).float()
            k_raw = attn.k_proj(h).float()

    q = q_raw.view(seq_len, n_heads, head_dim).permute(1, 0, 2)
    k = k_raw.view(seq_len, n_kv_heads, head_dim).permute(1, 0, 2)
    if n_kv_heads < n_heads:
        k = k.repeat_interleave(n_heads // n_kv_heads, dim=0)

    # FastV: received attention — "average attention-score one token
    # received from all other tokens" at layer K.
    # Compute full causal attention matrix at this layer, then sum
    # columns corresponding to visual positions.
    n_vis = len(vis_pos)
    scale = head_dim ** -0.5

    # Full Q*K^T: [n_heads, seq_len, seq_len]
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    # Apply causal mask (lower triangular)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=logits.device),
        diagonal=1)
    logits = logits + causal_mask.unsqueeze(0)
    attn_weights = F.softmax(logits, dim=-1)  # [n_heads, seq_len, seq_len]

    # Received attention for visual tokens: sum of attention weights
    # in visual-position columns, across all query positions
    vis_received = attn_weights[:, :, vis_pos].sum(dim=1)  # [n_heads, n_vis]
    importance = vis_received.mean(dim=0).cpu().numpy()  # [n_vis]

    del logits, attn_weights

    top_idx = set(np.argsort(importance)[::-1][:k_y].tolist())
    return [i in top_idx for i in range(n_vis)]


def _zspaprune_select(lm, full_embeds, vis_pos, text_pos, vis_embeds, k_y,
                      core_ratio=0.6):
    """ZSPAPrune: prompt-aware core + diversity visual token selection.

    Paper: "Zero-Shot Prompt-Aware Token Pruning for VLMs" (arXiv:2510.17197)
    Phase 1: cosine sim between vis_embeds and mean text embedding → core set.
    Phase 2: greedy max-min diversity from remaining tokens.
    No LLM forward needed — only embedding layer.
    """
    n_vis = len(vis_pos)

    # Mean text embedding as prompt vector (from input embeddings, not hidden states)
    text_emb = full_embeds[0, text_pos, :].float().cpu()  # [n_text, D]
    prompt_vec = F.normalize(text_emb.mean(dim=0, keepdim=True), dim=-1)  # [1, D]

    vis_norm = F.normalize(vis_embeds.float().cpu(), dim=-1)  # [n_vis, D]

    # Phase 1: core set (most prompt-relevant)
    n_core = max(1, int(k_y * core_ratio))
    prompt_sim = (vis_norm @ prompt_vec.T).squeeze(-1)  # [n_vis]
    core_ranked = prompt_sim.argsort(descending=True).tolist()
    selected = list(core_ranked[:n_core])
    selected_set = set(selected)

    # Phase 2: diversity augmentation (greedy max-min distance)
    n_diverse = k_y - len(selected)
    if n_diverse > 0:
        sim_matrix = vis_norm @ vis_norm.T  # [n_vis, n_vis]
        min_sim = torch.full((n_vis,), float('inf'))
        for idx in selected:
            min_sim = torch.min(min_sim, sim_matrix[idx])
        for idx in selected:
            min_sim[idx] = float('inf')
        for _ in range(n_diverse):
            next_idx = min_sim.argmin().item()
            if min_sim[next_idx] == float('inf'):
                break
            selected.append(next_idx)
            min_sim = torch.min(min_sim, sim_matrix[next_idx])
            min_sim[next_idx] = float('inf')

    keep_set = set(selected)
    return [i in keep_set for i in range(n_vis)]


def _agilepruner_select(lm, full_embeds, vis_pos, text_pos, vis_embeds, k_y,
                        score_layer=1, tau_base=0.7, tau_max=0.95):
    """AgilePruner: adaptive attention + diversity visual token pruning.

    Paper: "AgilePruner" (arXiv:2603.01236, ICLR 2026)
    Step 1: text→vis attention at LLM layer 1 (0-indexed).
    Step 2: entropy-adaptive cosine similarity dedup threshold.
    Step 3: Phase A top-K by attention, Phase B dedup among them.
    """
    import math
    n_vis = len(vis_pos)

    # Forward through layers 0..(score_layer-1) for actual hidden states
    with torch.no_grad():
        hidden = _run_layers(lm, full_embeds, 0, score_layer)

    # Compute text→vis attention at score_layer
    layer = lm.layers[score_layer]
    attn_mod = getattr(layer, 'self_attn', None) or layer.attention
    ln = getattr(layer, 'input_layernorm', None) or layer.attention_norm
    seq_len = hidden.shape[1]

    has_fused_qkv = hasattr(attn_mod, 'wqkv')
    if has_fused_qkv:
        cfg = attn_mod.config if hasattr(attn_mod, 'config') else lm.config
        n_heads = cfg.num_attention_heads
        n_kv_heads = cfg.num_key_value_heads
        head_dim = cfg.hidden_size // n_heads
        with torch.no_grad():
            h = ln(hidden[0])
            qkv = attn_mod.wqkv(h).float()
            q_size = n_heads * head_dim
            kv_size = n_kv_heads * head_dim
            q_raw = qkv[:, :q_size]
            k_raw = qkv[:, q_size:q_size + kv_size]
    else:
        n_heads = getattr(attn_mod, 'num_heads', None) or attn_mod.config.num_attention_heads
        n_kv_heads = getattr(attn_mod, 'num_key_value_heads', None) or attn_mod.config.num_key_value_heads
        head_dim = getattr(attn_mod, 'head_dim', None) or (attn_mod.config.hidden_size // n_heads)
        with torch.no_grad():
            h = ln(hidden[0])
            q_raw = attn_mod.q_proj(h).float()
            k_raw = attn_mod.k_proj(h).float()

    # Text→vis attention: Q from text, K from vis
    t_idx = text_pos
    v_idx = vis_pos
    q = q_raw[t_idx].view(len(t_idx), n_heads, head_dim).permute(1, 0, 2)
    k = k_raw[v_idx].view(len(v_idx), n_kv_heads, head_dim).permute(1, 0, 2)
    if n_kv_heads < n_heads:
        k = k.repeat_interleave(n_heads // n_kv_heads, dim=0)
    scale = head_dim ** -0.5
    attn = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
    scores = attn.mean(dim=0).mean(dim=0).cpu()  # [n_vis]

    del hidden, q_raw, k_raw, attn

    # Adaptive tau from entropy
    probs = F.softmax(scores.float(), dim=0)
    n = max(probs.shape[0], 2)
    H = -(probs * (probs + 1e-9).log()).sum().item()
    h_norm = H / math.log(n)
    h_norm = max(0.0, min(1.0, h_norm))
    tau = tau_base + (tau_max - tau_base) * (1.0 - h_norm)

    # Phase A: rank by attention, Phase B: dedup
    ranked = torch.argsort(scores, descending=True).tolist()
    emb = F.normalize(vis_embeds.float().cpu(), dim=-1)
    kept = []
    kept_emb = []
    for idx in ranked:
        if len(kept) >= k_y:
            break
        if not kept_emb:
            kept.append(idx)
            kept_emb.append(emb[idx])
            continue
        emb_stack = torch.stack(kept_emb, dim=0)
        max_sim = (emb_stack @ emb[idx]).max().item()
        if max_sim < tau:
            kept.append(idx)
            kept_emb.append(emb[idx])

    # Fill if dedup removed too many
    kept_set = set(kept)
    if len(kept_set) < k_y:
        for idx in ranked:
            if idx not in kept_set:
                kept_set.add(idx)
            if len(kept_set) >= k_y:
                break

    return [i in kept_set for i in range(n_vis)]


def _idselection_select(lm, full_embeds, vis_pos, text_pos, vis_embeds, k_y,
                        llm_layer=2, gamma=20.0):
    """ID-Selection: importance-diversity iterative selection with Gaussian suppression.

    Paper: "ID-Selection" (arXiv:2604.05601, 2026)
    Step 1: cross-modal cosine sim at LLM layer as importance.
    Step 2: iterative selection with Gaussian suppression (Eq. 7-9).
    """
    n_vis = len(vis_pos)

    # Forward through layers 0..(llm_layer-1) for hidden states
    with torch.no_grad():
        hidden = _run_layers(lm, full_embeds, 0, llm_layer)

    # Get hidden states at llm_layer input
    layer = lm.layers[llm_layer]
    ln = getattr(layer, 'input_layernorm', None) or layer.attention_norm
    with torch.no_grad():
        h = ln(hidden[0]).float()  # [seq_len, D]

    H_v = h[vis_pos]   # [n_vis, D]
    H_t = h[text_pos]  # [n_text, D]
    H_t_mean = H_t.mean(dim=0, keepdim=True)  # [1, D]

    # Importance = cosine similarity
    scores = F.cosine_similarity(H_v, H_t_mean.expand_as(H_v), dim=-1).cpu()  # [n_vis]

    del hidden, h

    # Min-max normalize to [0, 1]
    mn, mx = scores.min(), scores.max()
    if (mx - mn).abs() > 1e-9:
        scores = (scores - mn) / (mx - mn)

    # Gaussian suppression iterative selection
    emb = F.normalize(vis_embeds.float().cpu(), dim=-1)
    cos_sim = emb @ emb.T
    dist_sq = (1.0 - cos_sim).clamp(min=0.0).pow(2)
    W = torch.exp(-gamma * dist_sq)  # [n_vis, n_vis]

    current_scores = scores.clone().float()
    selected = []
    available = torch.ones(n_vis, dtype=torch.bool)

    for _ in range(k_y):
        masked = current_scores.clone()
        masked[~available] = -float('inf')
        best = masked.argmax().item()
        selected.append(best)
        available[best] = False
        S_best = current_scores[best]
        current_scores -= W[best] * S_best
        current_scores.clamp_(min=0.0)

    keep_set = set(selected)
    return [i in keep_set for i in range(n_vis)]


def _d2pruner_select(lm, full_embeds, vis_pos, text_pos, vis_embeds, k_y,
                     score_layer=2, pivot_ratio=0.7, sim_threshold=0.8,
                     spatial_radius=2, grid_thw=None):
    """D²Pruner: debiased importance + structural diversity (MIS).

    Paper: "D2Pruner" (arXiv:2512.19443, AAAI 2026)
    Step 1: debiased text→vis attention at LLM layer → pivot tokens.
       Paper uses A_rel = A_ori / (A_bias + ε) with offline-calibrated bias
       prior (1000 COCO images). Since offline calibration is infeasible here,
       we approximate by removing the linear positional trend (subtraction).
    Step 2: MIS on hybrid graph (spatial + semantic) for supplementary tokens.
    Defaults: r_pivot=0.7, θ_sim=0.8 (paper Section "Implementation Details").
    """
    import math
    n_vis = len(vis_pos)

    # Forward through layers 0..(score_layer-1)
    with torch.no_grad():
        hidden = _run_layers(lm, full_embeds, 0, score_layer)

    # Text→vis attention at score_layer
    layer = lm.layers[score_layer]
    attn_mod = getattr(layer, 'self_attn', None) or layer.attention
    ln = getattr(layer, 'input_layernorm', None) or layer.attention_norm

    has_fused_qkv = hasattr(attn_mod, 'wqkv')
    if has_fused_qkv:
        cfg_m = attn_mod.config if hasattr(attn_mod, 'config') else lm.config
        n_heads = cfg_m.num_attention_heads
        n_kv_heads = cfg_m.num_key_value_heads
        head_dim = cfg_m.hidden_size // n_heads
        with torch.no_grad():
            h = ln(hidden[0])
            qkv = attn_mod.wqkv(h).float()
            q_size = n_heads * head_dim
            kv_size = n_kv_heads * head_dim
            q_raw = qkv[:, :q_size]
            k_raw = qkv[:, q_size:q_size + kv_size]
    else:
        n_heads = getattr(attn_mod, 'num_heads', None) or attn_mod.config.num_attention_heads
        n_kv_heads = getattr(attn_mod, 'num_key_value_heads', None) or attn_mod.config.num_key_value_heads
        head_dim = getattr(attn_mod, 'head_dim', None) or (attn_mod.config.hidden_size // n_heads)
        with torch.no_grad():
            h = ln(hidden[0])
            q_raw = attn_mod.q_proj(h).float()
            k_raw = attn_mod.k_proj(h).float()

    q = q_raw[text_pos].view(len(text_pos), n_heads, head_dim).permute(1, 0, 2)
    k = k_raw[vis_pos].view(n_vis, n_kv_heads, head_dim).permute(1, 0, 2)
    if n_kv_heads < n_heads:
        k = k.repeat_interleave(n_heads // n_kv_heads, dim=0)
    scale = head_dim ** -0.5
    attn = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
    raw_scores = attn.mean(dim=0).mean(dim=0).cpu()  # [n_vis]

    del hidden, q_raw, k_raw, attn

    # Debias: remove linear positional trend
    positions = torch.arange(n_vis, dtype=torch.float32)
    scores_f = raw_scores.float()
    pos_mean = positions.mean()
    sc_mean = scores_f.mean()
    cov = ((positions - pos_mean) * (scores_f - sc_mean)).sum()
    var = ((positions - pos_mean) ** 2).sum()
    if var.abs() > 1e-9:
        a = cov / var
        b = sc_mean - a * pos_mean
        bias = a * positions + b
        debiased = scores_f - bias
    else:
        debiased = scores_f - sc_mean

    # Step 1: pivot tokens from debiased scores
    n_pivot = max(1, int(k_y * pivot_ratio))
    pivot_ranked = debiased.argsort(descending=True).tolist()
    pivots = set(pivot_ranked[:n_pivot])

    # Step 2: MIS on remaining tokens
    n_supplement = k_y - len(pivots)
    if n_supplement <= 0:
        keep_set = pivots
    else:
        remaining = [i for i in range(n_vis) if i not in pivots]
        emb = F.normalize(vis_embeds.float().cpu(), dim=-1)
        rem_idx = torch.tensor(remaining, dtype=torch.long)
        rem_emb = emb[rem_idx]
        rem_sim = rem_emb @ rem_emb.T

        # 2D spatial grid
        side = int(math.sqrt(n_vis))
        if side * side < n_vis:
            side += 1

        # Build adjacency
        n_rem = len(remaining)
        adj = [set() for _ in range(n_rem)]
        for i in range(n_rem):
            for j in range(i + 1, n_rem):
                gi, gj = remaining[i], remaining[j]
                ri, ci = gi // side, gi % side
                rj, cj = gj // side, gj % side
                spatial_close = (abs(ri - rj) <= spatial_radius and
                                abs(ci - cj) <= spatial_radius)
                semantic_close = rem_sim[i, j].item() > sim_threshold
                if spatial_close or semantic_close:
                    adj[i].add(j)
                    adj[j].add(i)

        # Greedy MIS
        rem_scores = debiased[rem_idx]
        sorted_rem = rem_scores.argsort(descending=True).tolist()
        available = set(range(n_rem))
        mis_selected = []
        for local_idx in sorted_rem:
            if local_idx not in available:
                continue
            mis_selected.append(remaining[local_idx])
            available.discard(local_idx)
            for nb in adj[local_idx]:
                available.discard(nb)
            if len(mis_selected) >= n_supplement:
                break

        # Fill if needed
        if len(mis_selected) < n_supplement:
            all_selected = pivots | set(mis_selected)
            for local_idx in sorted_rem:
                gi = remaining[local_idx]
                if gi not in all_selected:
                    mis_selected.append(gi)
                if len(mis_selected) >= n_supplement:
                    break

        keep_set = pivots | set(mis_selected)

    return [i in keep_set for i in range(n_vis)]


def _ptp_select(vit_imp, lm, full_embeds, vis_pos, text_pos, vis_embeds, k_y,
                refine_layer=2, alpha=0.5, grid_thw=None):
    """PTP: Pyramid Token Pruning — ViT saliency + instruction-aware fusion.

    Paper: "Training-Free Pyramid Token Pruning" (arXiv:2509.15704, 2025)
    Stage 1+2: ViT attention → per-tile budget allocation → within-tile top-k.
    Stage 3: Score fusion: final = (1-α)·b_norm + α·c, where b = bottom-up
             ViT saliency (min-max normalized), c = max text→vis attention
             over instruction tokens (per paper §IV-A).
    """
    import math
    n_vis = len(vis_pos)

    if vit_imp is None:
        # Fallback: L2 norm
        norms = vis_embeds.float().norm(dim=-1).cpu()
        top_idx = set(norms.argsort(descending=True)[:k_y].tolist())
        return [i in top_idx for i in range(n_vis)]

    # Stage 1+2: per-tile budget from ViT importance
    token_scores = torch.tensor(vit_imp, dtype=torch.float32)
    if grid_thw is not None and grid_thw.shape[0] > 1:
        # Multiple tiles: allocate proportionally
        merge = 2  # default spatial_merge_size
        m2 = merge * merge
        tile_sizes = []
        offset = 0
        for t in range(grid_thw.shape[0]):
            t_val, h_val, w_val = grid_thw[t].tolist()
            tile_n = int(t_val * h_val * w_val) // m2
            tile_sizes.append((offset, offset + tile_n))
            offset += tile_n

        tile_saliency = []
        for start, end in tile_sizes:
            if end > start and end <= n_vis:
                tile_saliency.append(token_scores[start:end].mean().item())
            else:
                tile_saliency.append(0.0)

        total_sal = sum(tile_saliency) + 1e-9
        tile_budgets = [max(1, int(k_y * (s / total_sal))) for s in tile_saliency]
        diff = k_y - sum(tile_budgets)
        if diff > 0:
            sorted_tiles = sorted(range(len(tile_saliency)),
                                  key=lambda i: tile_saliency[i], reverse=True)
            for i in range(diff):
                tile_budgets[sorted_tiles[i % len(sorted_tiles)]] += 1

        selected = set()
        for tile_idx, (start, end) in enumerate(tile_sizes):
            if end > n_vis:
                end = n_vis
            tile_tokens = list(range(start, end))
            tile_sc = [(i, token_scores[i].item()) for i in tile_tokens]
            tile_sc.sort(key=lambda x: x[1], reverse=True)
            budget = min(tile_budgets[tile_idx], len(tile_sc))
            for i in range(budget):
                selected.add(tile_sc[i][0])
    else:
        # Single tile: simple top-k
        ranked = token_scores.argsort(descending=True).tolist()
        selected = set(ranked[:k_y])

    # Stage 3: score fusion — (1-α)·b_norm + α·c  (paper §IV-A, Table III)
    # b = bottom-up ViT saliency, c = instruction-guided importance (max over
    # instruction tokens, averaged over heads).
    if text_pos and alpha > 0:
        with torch.no_grad():
            hidden = _run_layers(lm, full_embeds, 0, refine_layer)

        layer = lm.layers[refine_layer]
        attn_mod = getattr(layer, 'self_attn', None) or layer.attention
        ln = getattr(layer, 'input_layernorm', None) or layer.attention_norm

        has_fused_qkv = hasattr(attn_mod, 'wqkv')
        if has_fused_qkv:
            cfg_m = attn_mod.config if hasattr(attn_mod, 'config') else lm.config
            n_heads = cfg_m.num_attention_heads
            n_kv_heads = cfg_m.num_key_value_heads
            head_dim = cfg_m.hidden_size // n_heads
            with torch.no_grad():
                h = ln(hidden[0])
                qkv = attn_mod.wqkv(h).float()
                q_size = n_heads * head_dim
                kv_size = n_kv_heads * head_dim
                q_raw = qkv[:, :q_size]
                k_raw = qkv[:, q_size:q_size + kv_size]
        else:
            n_heads = getattr(attn_mod, 'num_heads', None) or attn_mod.config.num_attention_heads
            n_kv_heads = getattr(attn_mod, 'num_key_value_heads', None) or attn_mod.config.num_key_value_heads
            head_dim = getattr(attn_mod, 'head_dim', None) or (attn_mod.config.hidden_size // n_heads)
            with torch.no_grad():
                h = ln(hidden[0])
                q_raw = attn_mod.q_proj(h).float()
                k_raw = attn_mod.k_proj(h).float()

        q = q_raw[text_pos].view(len(text_pos), n_heads, head_dim).permute(1, 0, 2)
        k = k_raw[vis_pos].view(n_vis, n_kv_heads, head_dim).permute(1, 0, 2)
        if n_kv_heads < n_heads:
            k = k.repeat_interleave(n_heads // n_kv_heads, dim=0)
        scale = head_dim ** -0.5
        attn = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        # Paper: c_j = max_i attn(text_i → vis_j), averaged over heads
        instr_scores = attn.mean(dim=0).max(dim=0).values.cpu()  # [n_vis]

        del hidden, q_raw, k_raw, attn

        # Normalize bottom-up scores to [0, 1] (min-max)
        b = token_scores.clone().float()
        b_min, b_max = b.min(), b.max()
        if b_max - b_min > 1e-9:
            b = (b - b_min) / (b_max - b_min)
        else:
            b = torch.zeros_like(b)

        # c is already in [0, 1] range from softmax
        c = instr_scores.float()

        # Fused score
        fused = (1.0 - alpha) * b + alpha * c

        # Re-select top-k_y by fused score, respecting per-tile budgets
        if grid_thw is not None and grid_thw.shape[0] > 1:
            # Re-do per-tile selection with fused scores
            merge = 2
            m2 = merge * merge
            tile_sizes_r = []
            offset_r = 0
            for t in range(grid_thw.shape[0]):
                t_val, h_val, w_val = grid_thw[t].tolist()
                tile_n = int(t_val * h_val * w_val) // m2
                tile_sizes_r.append((offset_r, offset_r + tile_n))
                offset_r += tile_n

            # Reuse same tile budgets (from Stage 1 saliency allocation)
            tile_saliency_r = []
            for start, end in tile_sizes_r:
                if end > start and end <= n_vis:
                    tile_saliency_r.append(token_scores[start:end].mean().item())
                else:
                    tile_saliency_r.append(0.0)
            total_sal_r = sum(tile_saliency_r) + 1e-9
            tile_budgets_r = [max(1, int(k_y * (s / total_sal_r)))
                              for s in tile_saliency_r]
            diff_r = k_y - sum(tile_budgets_r)
            if diff_r > 0:
                sorted_t = sorted(range(len(tile_saliency_r)),
                                  key=lambda i: tile_saliency_r[i], reverse=True)
                for i in range(diff_r):
                    tile_budgets_r[sorted_t[i % len(sorted_t)]] += 1

            selected = set()
            for tile_idx, (start, end) in enumerate(tile_sizes_r):
                if end > n_vis:
                    end = n_vis
                tile_tokens = list(range(start, end))
                tile_sc = [(i, fused[i].item()) for i in tile_tokens]
                tile_sc.sort(key=lambda x: x[1], reverse=True)
                budget = min(tile_budgets_r[tile_idx], len(tile_sc))
                for i in range(budget):
                    selected.add(tile_sc[i][0])
        else:
            ranked_fused = fused.argsort(descending=True).tolist()
            selected = set(ranked_fused[:k_y])

    # Ensure exactly k_y
    if len(selected) < k_y:
        ranked = token_scores.argsort(descending=True).tolist()
        for idx in ranked:
            if idx not in selected:
                selected.add(idx)
            if len(selected) >= k_y:
                break
    elif len(selected) > k_y:
        sel_scores = [(i, token_scores[i].item()) for i in selected]
        sel_scores.sort(key=lambda x: x[1], reverse=True)
        selected = set(x[0] for x in sel_scores[:k_y])

    return [i in selected for i in range(n_vis)]


def _hawk_importance(lm, full_embeds, vis_pos, text_pos):
    """HAWK importance scores: head-weighted text→vis attention at Layer 0, NO RoPE.

    Returns: np.ndarray of shape [n_vis] with importance scores.
    """
    n_vis = len(vis_pos)

    layer = lm.layers[0]
    attn_mod = getattr(layer, 'self_attn', None) or layer.attention
    ln = getattr(layer, 'input_layernorm', None) or layer.attention_norm

    has_fused_qkv = hasattr(attn_mod, 'wqkv')
    if has_fused_qkv:
        cfg = attn_mod.config if hasattr(attn_mod, 'config') else lm.config
        n_heads = cfg.num_attention_heads
        n_kv_heads = cfg.num_key_value_heads
        head_dim = cfg.hidden_size // n_heads
        with torch.no_grad():
            h = ln(full_embeds[0])
            qkv = attn_mod.wqkv(h).float()
            q_size = n_heads * head_dim
            kv_size = n_kv_heads * head_dim
            q_raw = qkv[:, :q_size]
            k_raw = qkv[:, q_size:q_size + kv_size]
    else:
        n_heads = getattr(attn_mod, 'num_heads', None) or attn_mod.config.num_attention_heads
        n_kv_heads = getattr(attn_mod, 'num_key_value_heads', None) or attn_mod.config.num_key_value_heads
        head_dim = getattr(attn_mod, 'head_dim', None) or (attn_mod.config.hidden_size // n_heads)
        with torch.no_grad():
            h = ln(full_embeds[0])
            q_raw = attn_mod.q_proj(h).float()
            k_raw = attn_mod.k_proj(h).float()

    q = q_raw[text_pos].view(len(text_pos), n_heads, head_dim).permute(1, 0, 2)
    k = k_raw[vis_pos].view(n_vis, n_kv_heads, head_dim).permute(1, 0, 2)
    if n_kv_heads < n_heads:
        k = k.repeat_interleave(n_heads // n_kv_heads, dim=0)

    scale = head_dim ** -0.5
    attn = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
    c = attn.mean(dim=1)
    importance = c.mean(dim=0).cpu().numpy()

    del q_raw, k_raw, attn
    return importance


def _hawk_select(lm, full_embeds, vis_pos, text_pos, k_y):
    """HAWK: head-weighted text→vis attention at Layer 0, NO RoPE."""
    importance = _hawk_importance(lm, full_embeds, vis_pos, text_pos)
    top_idx = set(np.argsort(importance)[::-1][:k_y].tolist())
    return [i in top_idx for i in range(len(vis_pos))]


def _vscore_l2_select(lm, full_embeds, vis_pos, text_pos, k_y, score_layer=2):
    """VScoreL2: text→vis attention × value norm at LLM layer.

    v_score = v_attn × v_vnorm
    v_attn: text→vis attention at the score layer.
    v_vnorm: ||W_V h_vis|| averaged over heads (value expressiveness).
    """
    n_vis = len(vis_pos)

    # Forward through layers 0..(score_layer-1) for actual hidden states
    with torch.no_grad():
        hidden = _run_layers(lm, full_embeds, 0, score_layer)

    layer = lm.layers[score_layer]
    attn_mod = getattr(layer, 'self_attn', None) or layer.attention
    ln = getattr(layer, 'input_layernorm', None) or layer.attention_norm

    has_fused_qkv = hasattr(attn_mod, 'wqkv')
    if has_fused_qkv:
        cfg = attn_mod.config if hasattr(attn_mod, 'config') else lm.config
        n_heads = cfg.num_attention_heads
        n_kv_heads = cfg.num_key_value_heads
        head_dim = cfg.hidden_size // n_heads
        with torch.no_grad():
            h = ln(hidden[0])
            qkv = attn_mod.wqkv(h).float()
            q_size = n_heads * head_dim
            kv_size = n_kv_heads * head_dim
            q_raw = qkv[:, :q_size]
            k_raw = qkv[:, q_size:q_size + kv_size]
            v_raw = qkv[:, q_size + kv_size:]
    else:
        n_heads = getattr(attn_mod, 'num_heads', None) or attn_mod.config.num_attention_heads
        n_kv_heads = getattr(attn_mod, 'num_key_value_heads', None) or attn_mod.config.num_key_value_heads
        head_dim = getattr(attn_mod, 'head_dim', None) or (attn_mod.config.hidden_size // n_heads)
        with torch.no_grad():
            h = ln(hidden[0])
            q_raw = attn_mod.q_proj(h).float()
            k_raw = attn_mod.k_proj(h).float()
            v_raw = attn_mod.v_proj(h).float()

    # v_attn: text→vis attention
    q = q_raw[text_pos].view(len(text_pos), n_heads, head_dim).permute(1, 0, 2)
    k = k_raw[vis_pos].view(n_vis, n_kv_heads, head_dim).permute(1, 0, 2)
    if n_kv_heads < n_heads:
        k = k.repeat_interleave(n_heads // n_kv_heads, dim=0)

    scale = head_dim ** -0.5
    attn = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
    v_attn = attn.mean(dim=0).mean(dim=0)  # [n_vis]

    # v_vnorm: ||W_V h_vis|| per head, averaged
    v = v_raw[vis_pos].view(n_vis, n_kv_heads, head_dim).permute(1, 0, 2)
    if n_kv_heads < n_heads:
        v = v.repeat_interleave(n_heads // n_kv_heads, dim=0)
    v_norm = v.float().norm(dim=-1).mean(dim=0)  # [n_vis]

    v_score = (v_attn * v_norm).cpu().numpy()

    del hidden, q_raw, k_raw, v_raw, attn
    top_idx = set(np.argsort(v_score)[::-1][:k_y].tolist())
    return [i in top_idx for i in range(n_vis)]


# ──────────────── C-class helpers (progressive pruning) ────────────

# FitPrune reference schedules (paper calibration, 576 tokens, 28 layers)
_FITPRUNE_REF = {
    50: {0: 3, 1: 7, 2: 49, 3: 24, 4: 29, 5: 17, 6: 59, 7: 41, 8: 26,
         9: 8, 10: 10, 11: 6, 12: 2, 13: 5, 14: 7, 15: 2, 16: 30, 17: 3,
         18: 13, 19: 27, 20: 5, 21: 23, 22: 2, 23: 1, 24: 0, 25: 0, 26: 1, 27: 1},
    70: {0: 48, 1: 79, 2: 118, 3: 41, 4: 33, 5: 17, 6: 36, 7: 9, 8: 9,
         9: 2, 10: 3, 11: 2, 12: 1, 13: 3, 14: 4, 15: 8, 16: 18, 17: 7,
         18: 8, 19: 20, 20: 4, 21: 13, 22: 2, 23: 1, 24: 0, 25: 0, 26: 1, 27: 0},
    80: {0: 128, 1: 127, 2: 118, 3: 35, 4: 26, 5: 8, 6: 13, 7: 3, 8: 2,
         9: 1, 10: 0, 11: 0, 12: 0, 13: 2, 14: 1, 15: 8, 16: 7, 17: 8,
         18: 5, 19: 11, 20: 2, 21: 6, 22: 1, 23: 1, 24: 0, 25: 0, 26: 0, 27: 1},
    90: {0: 262, 1: 153, 2: 80, 3: 19, 4: 13, 5: 1, 6: 2, 7: 0, 8: 0,
         9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 1, 16: 1, 17: 2,
         18: 1, 19: 2, 20: 0, 21: 1, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 1},
}


def _fitprune_build_schedule(n_vis, k_y, n_layers):
    """Build FitPrune per-layer drop schedule from reference, scaled."""
    n_drop = n_vis - k_y
    if n_drop <= 0:
        return {}
    # Interpolate between reference schedules
    reduction_pct = n_drop / n_vis * 100
    avail = sorted(_FITPRUNE_REF.keys())
    if reduction_pct <= avail[0]:
        ref = _FITPRUNE_REF[avail[0]]
    elif reduction_pct >= avail[-1]:
        ref = _FITPRUNE_REF[avail[-1]]
    else:
        lo, hi = avail[0], avail[-1]
        for i in range(len(avail) - 1):
            if avail[i] <= reduction_pct <= avail[i + 1]:
                lo, hi = avail[i], avail[i + 1]
                break
        alpha = (reduction_pct - lo) / max(hi - lo, 1)
        ref = {}
        for layer in set(_FITPRUNE_REF[lo]) | set(_FITPRUNE_REF[hi]):
            ref[layer] = _FITPRUNE_REF[lo].get(layer, 0) + alpha * (
                _FITPRUNE_REF[hi].get(layer, 0) - _FITPRUNE_REF[lo].get(layer, 0))
    # Scale layer indices to actual n_layers (reference uses 28 layers)
    ref_n = 28
    scaled_ref = {}
    for l, c in ref.items():
        mapped_l = int(round(l * (n_layers - 1) / (ref_n - 1)))
        mapped_l = min(mapped_l, n_layers - 1)
        scaled_ref[mapped_l] = scaled_ref.get(mapped_l, 0) + c
    # Scale counts to target n_drop
    ref_total = sum(scaled_ref.values())
    if ref_total <= 0:
        return {}
    scale = n_drop / ref_total
    raw = {l: max(0, round(c * scale)) for l, c in scaled_ref.items()}
    # Adjust rounding
    diff = n_drop - sum(raw.values())
    for l in sorted(raw.keys(), key=lambda l: raw[l], reverse=True):
        if diff == 0:
            break
        if diff > 0:
            raw[l] += 1
            diff -= 1
        elif raw[l] > 0:
            raw[l] -= 1
            diff += 1
    return {l: d for l, d in raw.items() if d > 0}


def _layer_qk(layer, hidden, q_idx, k_idx):
    """Compute Q×K^T attention at a layer for given query/key indices.

    Returns (attn [n_heads, n_q, n_k], n_heads, head_dim).
    """
    attn_mod = getattr(layer, 'self_attn', None) or layer.attention
    ln = getattr(layer, 'input_layernorm', None) or getattr(layer, 'attention_norm', None)
    dev = next(attn_mod.parameters()).device
    h = hidden.to(dev)
    h_n = ln(h[0]) if ln is not None else h[0]

    has_fused = hasattr(attn_mod, 'wqkv')
    if has_fused:
        cfg_m = getattr(attn_mod, 'config', None) or getattr(attn_mod, '_config', None)
        if cfg_m is None:
            # fallback: try parent layer config
            for p in [layer, getattr(layer, 'self_attn', None)]:
                cfg_m = getattr(p, 'config', None)
                if cfg_m is not None:
                    break
        n_heads = cfg_m.num_attention_heads
        n_kv = cfg_m.num_key_value_heads
        head_dim = cfg_m.hidden_size // n_heads
        qkv = attn_mod.wqkv(h_n)
        q_size = n_heads * head_dim
        kv_size = n_kv * head_dim
        q_raw = qkv[:, :q_size]
        k_raw = qkv[:, q_size:q_size + kv_size]
    else:
        n_heads = getattr(attn_mod, 'num_heads', None) or attn_mod.config.num_attention_heads
        n_kv = getattr(attn_mod, 'num_key_value_heads', None) or attn_mod.config.num_key_value_heads
        head_dim = getattr(attn_mod, 'head_dim', None) or (attn_mod.config.hidden_size // n_heads)
        q_raw = attn_mod.q_proj(h_n)
        k_raw = attn_mod.k_proj(h_n)

    q = q_raw[q_idx].view(len(q_idx), n_heads, head_dim).permute(1, 0, 2)
    k = k_raw[k_idx].view(len(k_idx), n_kv, head_dim).permute(1, 0, 2)
    if n_kv < n_heads:
        k = k.repeat_interleave(n_heads // n_kv, dim=0)
    scale = head_dim ** -0.5
    attn = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
    return attn, n_heads, head_dim


@torch.no_grad()
def _fitprune_select(lm, full_embeds, vis_pos, text_pos, vis_embeds, k_y):
    """FitPrune: progressive per-layer pruning with self_attn × cross_attn.

    Paper: Ye et al., "FitPrune: Visually Lossless Token Pruning" (2024)
    At each scheduled layer, re-score vis tokens using self-attention
    (vis→vis) × cross-attention (text→vis) and drop the lowest.
    """
    n_vis = len(vis_pos)
    if k_y >= n_vis:
        return [True] * n_vis
    n_layers = len(lm.layers)
    schedule = _fitprune_build_schedule(n_vis, k_y, n_layers)
    if not schedule:
        top = set(range(k_y))
        return [i in top for i in range(n_vis)]

    hidden = full_embeds
    vis_pos_set = set(vis_pos)
    # alive[h_idx] = original seq position
    alive = list(range(full_embeds.shape[1]))

    for l_idx in range(n_layers):
        if l_idx in schedule and schedule[l_idx] > 0:
            vis_h = [h for h, p in enumerate(alive) if p in vis_pos_set]
            text_h = [h for h, p in enumerate(alive) if p not in vis_pos_set]
            n_drop = min(schedule[l_idx], len(vis_h) - 1)
            if n_drop > 0:
                layer = lm.layers[l_idx]
                # Self-attention: vis→vis (max over heads, per official code)
                self_attn, _, _ = _layer_qk(layer, hidden, vis_h, vis_h)
                self_sc = self_attn.max(0).values.sum(0)  # [n_cur_vis]
                # Cross-attention: text→vis (max over heads, per official code)
                if text_h:
                    cross_attn, _, _ = _layer_qk(layer, hidden, text_h, vis_h)
                    cross_sc = cross_attn.max(0).values.mean(0)  # [n_cur_vis]
                else:
                    cross_sc = torch.ones(len(vis_h), device=self_sc.device)
                combined = self_sc * cross_sc
                # Drop lowest
                drop_local = combined.argsort()[:n_drop].tolist()
                drop_h = set(vis_h[i] for i in drop_local)
                keep = [i for i in range(len(alive)) if i not in drop_h]
                hidden = hidden[:, torch.tensor(keep, device=hidden.device), :]
                dropped_pos = {alive[i] for i in drop_h}
                vis_pos_set -= dropped_pos
                alive = [alive[i] for i in keep]

        hidden = _run_layers(lm, hidden, l_idx, l_idx + 1)

    vis_pos_to_idx = {p: i for i, p in enumerate(vis_pos)}
    keep_idx = {vis_pos_to_idx[p] for p in vis_pos_set if p in vis_pos_to_idx}
    return [i in keep_idx for i in range(n_vis)]


@torch.no_grad()
def _pyramiddrop_select(lm, full_embeds, vis_pos, text_pos, vis_embeds, k_y):
    """PyramidDrop: multi-stage progressive pruning via last-text→vis attention.

    Paper: Xia et al., "PyramidDrop" (2024)
    S=4 stages, cumulative keep ratios [1, λ, λ², λ³] where λ³ = k_y/n_vis.
    At each stage boundary, rank by last-text-token → visual attention and
    keep top fraction.
    """
    n_vis = len(vis_pos)
    if k_y >= n_vis:
        return [True] * n_vis
    n_layers = len(lm.layers)
    num_stages = 4
    lps = n_layers // num_stages
    boundaries = [(s + 1) * lps for s in range(num_stages - 1)]  # e.g. [7,14,21]

    # Solve λ: λ^(S-1) = k_y/n_vis
    final_ratio = k_y / n_vis
    if final_ratio >= 1.0:
        return [True] * n_vis
    lam = final_ratio ** (1.0 / (num_stages - 1))
    # Cumulative ratios: stage 0=1.0, stage 1=λ, stage 2=λ², stage 3=λ³
    cum = [lam ** s for s in range(num_stages)]
    # Build schedule: at each boundary, drop to next cumulative ratio
    schedule = {}
    remaining = n_vis
    for s, bl in enumerate(boundaries):
        target = max(1, int(n_vis * cum[s + 1]))
        drop = remaining - target
        if drop > 0:
            schedule[bl] = drop
        remaining = target

    if not schedule:
        top = set(range(k_y))
        return [i in top for i in range(n_vis)]

    hidden = full_embeds
    vis_pos_set = set(vis_pos)
    alive = list(range(full_embeds.shape[1]))

    for l_idx in range(n_layers):
        if l_idx in schedule and schedule[l_idx] > 0:
            vis_h = [h for h, p in enumerate(alive) if p in vis_pos_set]
            text_h = [h for h, p in enumerate(alive) if p not in vis_pos_set]
            n_drop = min(schedule[l_idx], len(vis_h) - 1)
            if n_drop > 0 and text_h:
                layer = lm.layers[l_idx]
                # Last text token → vis attention
                last_text = [text_h[-1]]
                attn, _, _ = _layer_qk(layer, hidden, last_text, vis_h)
                scores = attn.mean(0).squeeze(0)  # [n_cur_vis]
                # Drop lowest
                drop_local = scores.argsort()[:n_drop].tolist()
                drop_h = set(vis_h[i] for i in drop_local)
                keep = [i for i in range(len(alive)) if i not in drop_h]
                hidden = hidden[:, torch.tensor(keep, device=hidden.device), :]
                dropped_pos = {alive[i] for i in drop_h}
                vis_pos_set -= dropped_pos
                alive = [alive[i] for i in keep]

        hidden = _run_layers(lm, hidden, l_idx, l_idx + 1)

    vis_pos_to_idx = {p: i for i, p in enumerate(vis_pos)}
    keep_idx = {vis_pos_to_idx[p] for p in vis_pos_set if p in vis_pos_to_idx}
    return [i in keep_idx for i in range(n_vis)]


@torch.no_grad()
def _sparsevlm_select(lm, full_embeds, vis_pos, text_pos, vis_embeds, k_y,
                       rater_ratio=0.5, alpha_erank=0.5):
    """SparseVLM: text-rater-guided progressive sparsification.

    Paper: "SparseVLM: Visual Token Sparsification" (arXiv:2410.04417, ICML 2025)
    1. Select top-R text raters (vis→text attention at Layer 0).
    2. At stage boundaries, score vis tokens via rater→vis attention.
    3. Adaptive sparsity using effective rank.
    No token recycling (selection-only mode for mask computation).
    """
    import math as _math
    n_vis = len(vis_pos)
    if k_y >= n_vis:
        return [True] * n_vis
    n_layers = len(lm.layers)
    num_stages = 4
    lps = n_layers // num_stages
    boundaries = [s * lps for s in range(num_stages)]  # [0, 7, 14, 21]

    ratio = 1.0 - k_y / n_vis  # total prune ratio

    # Build schedule: linear targets across stages
    stage_targets = []
    for s in range(num_stages):
        frac = (s + 1) / num_stages
        n_s = max(k_y, round(n_vis - (n_vis - k_y) * frac))
        stage_targets.append(max(k_y, n_s))
    stage_targets[-1] = k_y

    hidden = full_embeds
    vis_pos_set = set(vis_pos)
    alive = list(range(full_embeds.shape[1]))

    # Step 1: select text raters at Layer 0 (vis→text attention)
    vis_h0 = [h for h, p in enumerate(alive) if p in vis_pos_set]
    text_h0 = [h for h, p in enumerate(alive) if p not in vis_pos_set]
    n_raters = max(1, round(len(text_h0) * rater_ratio))
    if text_h0 and vis_h0:
        attn_vt, _, _ = _layer_qk(lm.layers[0], hidden, vis_h0, text_h0)
        rater_scores = attn_vt.mean(0).sum(0)  # [n_text]
        top_local = rater_scores.argsort(descending=True)[:n_raters].tolist()
        rater_h_set = {text_h0[i] for i in top_local}
    else:
        rater_h_set = set(text_h0[:n_raters]) if text_h0 else set()

    for l_idx in range(n_layers):
        if l_idx in boundaries:
            stage_idx = boundaries.index(l_idx)
            vis_h = [h for h, p in enumerate(alive) if p in vis_pos_set]
            n_current = len(vis_h)
            n_target = stage_targets[stage_idx]
            if n_current > n_target and n_current > 1:
                # Rater → vis attention scoring
                rater_h = [h for h in range(len(alive))
                           if alive[h] in {alive[r] for r in rater_h_set
                                           if r < len(alive)}
                           or h in rater_h_set]
                # Rebuild rater indices in current sequence
                rater_orig_pos = {alive[r] for r in rater_h_set if r < len(alive)}
                rater_cur = [h for h, p in enumerate(alive) if p in rater_orig_pos]
                if not rater_cur:
                    rater_cur = [h for h, p in enumerate(alive) if p not in vis_pos_set]
                    rater_cur = rater_cur[-n_raters:] if rater_cur else []

                if rater_cur:
                    attn_rv, _, _ = _layer_qk(lm.layers[l_idx], hidden,
                                              rater_cur, vis_h)
                    scores = attn_rv.mean(0).mean(0)  # [n_cur_vis]
                else:
                    scores = torch.ones(n_current)

                # Adaptive sparsity via effective rank
                vis_idx_t = torch.tensor(vis_h, dtype=torch.long,
                                         device=hidden.device)
                h_vis = hidden[0, vis_idx_t, :].float()
                try:
                    s_vals = torch.linalg.svdvals(h_vis)
                    s_vals = s_vals[s_vals > 1e-9]
                    if s_vals.numel() > 0:
                        p = s_vals / s_vals.sum()
                        erank = _math.exp(-(p * (p + 1e-12).log()).sum().item())
                    else:
                        erank = float(n_current)
                except Exception:
                    erank = float(n_current)
                erank_ratio = min(1.0, erank / max(n_current, 1))
                n_keep = max(n_target, round(n_current * (1 - ratio) *
                                             (erank_ratio ** alpha_erank)))
                n_keep = max(n_target, min(n_current, n_keep))

                n_drop = n_current - n_keep
                if n_drop > 0:
                    drop_local = scores.argsort()[:n_drop].tolist()
                    drop_h = set(vis_h[i] for i in drop_local)
                    keep = [i for i in range(len(alive)) if i not in drop_h]
                    hidden = hidden[:, torch.tensor(keep, device=hidden.device), :]
                    dropped_pos = {alive[i] for i in drop_h}
                    vis_pos_set -= dropped_pos
                    # Update rater_h_set mapping
                    old_to_new = {old_h: new_h for new_h, old_h
                                  in enumerate(keep)}
                    rater_h_set = {old_to_new[r] for r in rater_h_set
                                   if r in old_to_new}
                    alive = [alive[i] for i in keep]

        hidden = _run_layers(lm, hidden, l_idx, l_idx + 1)

    vis_pos_to_idx = {p: i for i, p in enumerate(vis_pos)}
    keep_idx = {vis_pos_to_idx[p] for p in vis_pos_set if p in vis_pos_to_idx}
    return [i in keep_idx for i in range(n_vis)]


# ──────────────── SparseVILA (ViT attention) per model ────────────

def _qwen_vit_importance(visual_model, pixel_values, image_grid_thw, n_vis):
    """ViT last-block received attention (Qwen2/3-VL)."""
    last_block = visual_model.blocks[-1]
    attn_m = last_block.attn
    captured = {}
    def _pre(module, args):
        captured["h"] = args[0].detach()
    handle = last_block.register_forward_pre_hook(_pre)
    try:
        with torch.no_grad():
            _ = visual_model(pixel_values, image_grid_thw)
    finally:
        handle.remove()

    h = captured["h"]
    N = h.shape[0]
    num_heads = attn_m.num_heads
    head_dim = attn_m.head_dim
    with torch.no_grad():
        qkv = attn_m.qkv(h)
    q, k, _ = qkv.chunk(3, dim=-1)
    q = q.view(N, num_heads, head_dim).permute(1, 0, 2).float()
    k = k.view(N, num_heads, head_dim).permute(1, 0, 2).float()

    scale = head_dim ** -0.5
    received = torch.zeros(N, device=q.device, dtype=q.dtype)
    cs = max(1, min(1024, 8 * 1024**3 // (num_heads * N * 4)))
    for s in range(0, N, cs):
        e = min(s + cs, N)
        logits = torch.matmul(q[:, s:e, :], k.transpose(-1, -2)) * scale
        attn_chunk = F.softmax(logits, dim=-1)
        received += attn_chunk.sum(dim=1).mean(dim=0)
    del q, k

    merge = visual_model.spatial_merge_size
    m2 = merge * merge
    n_from = received.shape[0] // m2
    scores = received[:n_from * m2].view(n_from, m2).mean(dim=1).cpu().numpy()
    if n_from != n_vis:
        padded = np.zeros(n_vis)
        m = min(n_from, n_vis)
        padded[:m] = scores[:m]
        scores = padded
    return scores


def _llava_vit_importance(model, pixel_values, n_vis):
    """ViT last-layer CLS→patch attention (LLaVA / CLIP-ViT).

    SparseVILA specifies: for CLIP-style encoders (with CLS token),
    use CLS-to-patch attention as importance, not full received attention.
    """
    vit = model.model.vision_tower
    encoder_layers = vit.vision_model.encoder.layers
    last_layer = encoder_layers[-1]

    captured = {}
    def _pre(module, args):
        captured["h"] = args[0].detach()
    handle = last_layer.register_forward_pre_hook(_pre)
    try:
        with torch.no_grad():
            _ = vit(pixel_values)
    finally:
        handle.remove()

    h = captured["h"][0]  # [n_patches+1, dim] (CLS at 0)
    ln1 = last_layer.layer_norm1
    h_normed = ln1(h)
    attn = last_layer.self_attn

    with torch.no_grad():
        q = attn.q_proj(h_normed).float()
        k = attn.k_proj(h_normed).float()

    num_heads = attn.num_heads
    head_dim = attn.head_dim
    Nt = h.shape[0]

    q = q.view(Nt, num_heads, head_dim).permute(1, 0, 2)
    k = k.view(Nt, num_heads, head_dim).permute(1, 0, 2)

    scores = (q @ k.transpose(-1, -2)) * (head_dim ** -0.5)
    attn_w = F.softmax(scores, dim=-1)  # [n_heads, Nt, Nt]
    # CLS at position 0 → CLS-to-patch attention (skip CLS→CLS at col 0)
    cls_to_patch = attn_w[:, 0, 1:]  # [n_heads, n_patches]
    patch_imp = cls_to_patch.mean(dim=0).cpu().numpy()

    if len(patch_imp) >= n_vis:
        return patch_imp[:n_vis]
    padded = np.zeros(n_vis)
    padded[:len(patch_imp)] = patch_imp
    return padded


def _internvl3_vit_importance(model, pixel_values, n_vis):
    """ViT last-layer received attention (InternVL3-hf / InternViT)."""
    vit = model.model.vision_tower
    # InternVLVisionEncoder uses .layer (not .layers)
    enc = vit.encoder
    layer_list = getattr(enc, 'layer', None) or getattr(enc, 'layers', None) or enc.blocks
    last_layer = layer_list[-1]

    captured = {}
    def _pre(module, args, kwargs):
        captured["h"] = args[0].detach() if len(args) > 0 else kwargs.get("hidden_states", None)
    handle = last_layer.register_forward_pre_hook(_pre, with_kwargs=True)
    try:
        with torch.no_grad():
            _ = vit(pixel_values.to(model.dtype))
    finally:
        handle.remove()

    h = captured["h"]
    if h.dim() == 3:
        h = h[0]  # [n_patches, dim]
    N = h.shape[0]

    # Find attention module and layernorm
    attn_mod = getattr(last_layer, 'attention', None) or getattr(last_layer, 'attn', None) or getattr(last_layer, 'self_attn', None)
    ln = getattr(last_layer, 'layernorm_before', None) or getattr(last_layer, 'norm1', None) or getattr(last_layer, 'layer_norm1', None)
    if ln is not None:
        h = ln(h)

    with torch.no_grad():
        if hasattr(attn_mod, 'qkv'):
            qkv = attn_mod.qkv(h).float()
            q, k, _ = qkv.chunk(3, dim=-1)
        else:
            q = attn_mod.q_proj(h).float()
            k = attn_mod.k_proj(h).float()

    num_heads = getattr(attn_mod, 'num_heads', 16)
    head_dim = q.shape[-1] // num_heads

    q = q.view(N, num_heads, head_dim).permute(1, 0, 2)
    k = k.view(N, num_heads, head_dim).permute(1, 0, 2)

    scale = head_dim ** -0.5
    received = torch.zeros(N, device=q.device, dtype=q.dtype)
    cs = max(1, min(512, 4 * 1024**3 // (num_heads * N * 4)))
    for s in range(0, N, cs):
        e = min(s + cs, N)
        logits = torch.matmul(q[:, s:e, :], k.transpose(-1, -2)) * scale
        attn_w = F.softmax(logits, dim=-1)
        received += attn_w.sum(dim=1).mean(dim=0)
    del q, k

    patch_imp = received.cpu().numpy()
    # InternVL3 pixel_shuffle merges patches (4→1)
    merge_factor = max(1, N // n_vis) if n_vis > 0 else 1
    if merge_factor > 1 and N >= n_vis * merge_factor:
        patch_imp = patch_imp[:n_vis * merge_factor].reshape(n_vis, merge_factor).mean(axis=1)
    elif len(patch_imp) > n_vis:
        patch_imp = patch_imp[:n_vis]
    elif len(patch_imp) < n_vis:
        padded = np.zeros(n_vis)
        padded[:len(patch_imp)] = patch_imp
        patch_imp = padded
    return patch_imp


# ───────────────────────── Dataset ─────────────────────────────

def load_textvqa_samples(num):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/textvqa", split="validation")
    items = [it for it in ds if it.get("image") is not None]
    random.shuffle(items)
    items = items[:num]
    samples = []
    for it in items:
        img = it["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        answers = it.get("answers", [])
        if isinstance(answers, str):
            import ast
            try: answers = ast.literal_eval(answers)
            except: answers = [answers]
        samples.append((img, it["question"], answers))
    return samples


def _main_hf_cache(dataset):
    hub = os.environ.get("QCAL_HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    for namespace in ("lmms-lab-encoder", "lmms-lab"):
        path = os.path.join(hub, f"datasets--{namespace}--{dataset}")
        if os.path.isdir(path):
            return path
    return os.path.join(hub, f"datasets--lmms-lab-encoder--{dataset}")


def load_gqa_multi_round(num_images, questions_per_image=5):
    """Load GQA dataset grouped by image for multi-round evaluation.

    Two-pass: read parquet text columns to group questions, then decode needed images.
    Returns: list of (PIL.Image, [(question, answer), ...])
    """
    import glob
    import pyarrow.parquet as pq
    from collections import defaultdict
    from io import BytesIO

    cache_base = _main_hf_cache("GQA")
    snap_dirs = glob.glob(os.path.join(cache_base, "snapshots", "*"))
    if not snap_dirs:
        raise FileNotFoundError("GQA not cached. Run: "
            "load_dataset('lmms-lab/GQA', 'testdev_balanced_instructions', split='testdev') first.")
    snap = snap_dirs[0]

    # Pass 1: read questions (text only)
    q_files = sorted(glob.glob(os.path.join(
        snap, "testdev_balanced_instructions", "testdev-*.parquet")))
    if not q_files:
        raise FileNotFoundError(f"No GQA instruction parquet in {snap}")

    print("  [GQA] Pass 1: grouping questions by image...")
    img_questions = defaultdict(list)
    for pf in q_files:
        tbl = pq.read_table(pf, columns=["imageId", "question", "answer"])
        for iid, q, a in zip(tbl.column("imageId").to_pylist(),
                             tbl.column("question").to_pylist(),
                             tbl.column("answer").to_pylist()):
            img_questions[iid].append((q, a))

    eligible = [iid for iid, qas in img_questions.items() if len(qas) >= 1]
    print(f"  [GQA] {sum(len(v) for v in img_questions.values())} QA pairs, "
          f"{len(eligible)} images")
    random.shuffle(eligible)
    if num_images > 0:
        eligible = eligible[:num_images]

    # Pass 2: decode only needed images from the images parquet
    print(f"  [GQA] Pass 2: loading {len(eligible)} images...")
    img_files = sorted(glob.glob(os.path.join(
        snap, "testdev_balanced_images", "testdev-*.parquet")))
    need_ids = set(eligible)
    img_store = {}
    for pf in img_files:
        tbl = pq.read_table(pf, columns=["id", "image"])
        ids = tbl.column("id").to_pylist()
        imgs = tbl.column("image")
        for ri, iid in enumerate(ids):
            if iid in need_ids:
                img_struct = imgs[ri].as_py()
                img = Image.open(BytesIO(img_struct["bytes"]))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_store[iid] = img

    samples = []
    for iid in eligible:
        if iid not in img_store:
            continue
        qas = img_questions[iid]
        random.shuffle(qas)
        samples.append((img_store[iid], qas[:questions_per_image]))

    print(f"  {len(samples)} images × {questions_per_image} questions/image")
    return samples


def load_pope_multi_round(num_images, questions_per_image=5):
    """Load POPE dataset grouped by image for multi-round evaluation.

    POPE has ~6 questions per COCO image (3 categories × yes/no).
    Two-pass: read parquet text columns to group, then decode needed images.
    Returns: list of (PIL.Image, [(question, answer), ...])
    """
    import glob
    import pyarrow.parquet as pq
    from collections import defaultdict
    from io import BytesIO

    cache_base = _main_hf_cache("POPE")
    snap_dirs = glob.glob(os.path.join(cache_base, "snapshots", "*", "data"))
    if not snap_dirs:
        raise FileNotFoundError("POPE not cached. Run: "
            "load_dataset('lmms-lab/POPE', split='test') first.")

    pq_files = sorted(glob.glob(os.path.join(snap_dirs[0], "test-*.parquet")))
    if not pq_files:
        raise FileNotFoundError(f"No test parquet files in {snap_dirs[0]}")

    # Pass 1: read text columns, group by image_source
    print("  [POPE] Pass 1: grouping questions by image...")
    img_questions = defaultdict(list)
    img_row_map = defaultdict(list)  # image_source -> [(file_idx, row_idx)]
    for fi, pf in enumerate(pq_files):
        tbl = pq.read_table(pf, columns=["image_source", "question", "answer"])
        sources = tbl.column("image_source").to_pylist()
        questions = tbl.column("question").to_pylist()
        answers = tbl.column("answer").to_pylist()
        for ri, (src, q, a) in enumerate(zip(sources, questions, answers)):
            img_questions[src].append((q, a))
            img_row_map[src].append((fi, ri))

    eligible = [iid for iid, qas in img_questions.items() if len(qas) >= 1]
    print(f"  [POPE] {sum(len(v) for v in img_questions.values())} QA pairs, "
          f"{len(eligible)} images")
    random.shuffle(eligible)
    if num_images > 0:
        eligible = eligible[:num_images]

    # Pass 2: decode only needed images
    print(f"  [POPE] Pass 2: loading {len(eligible)} images...")
    need_from_file = defaultdict(dict)  # file_idx -> {row_idx: image_source}
    for iid in eligible:
        fi, ri = img_row_map[iid][0]  # first occurrence
        need_from_file[fi][ri] = iid

    img_store = {}
    for fi in sorted(need_from_file):
        rows_needed = need_from_file[fi]
        tbl = pq.read_table(pq_files[fi], columns=["image"])
        img_col = tbl.column("image")
        for ri, iid in rows_needed.items():
            img_struct = img_col[ri].as_py()
            img = Image.open(BytesIO(img_struct["bytes"]))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_store[iid] = img

    samples = []
    for iid in eligible:
        if iid not in img_store:
            continue
        qas = img_questions[iid]
        random.shuffle(qas)
        samples.append((img_store[iid], qas[:questions_per_image]))

    print(f"  {len(samples)} images × {questions_per_image} questions/image")
    return samples


def load_docvqa_multi_round(num_images, questions_per_image=3):
    """Load DocVQA dataset grouped by document for multi-round evaluation.

    DocVQA has ~4-5 questions per document image.
    Two-pass: read parquet text columns to group (fast), then decode needed images.
    Returns: list of (PIL.Image, [(question, answer), ...])
    """
    import glob
    import pyarrow.parquet as pq
    from collections import defaultdict
    from io import BytesIO

    cache_base = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--lmms-lab--DocVQA")
    snap_dirs = glob.glob(os.path.join(cache_base, "snapshots", "*", "DocVQA"))
    if not snap_dirs:
        raise FileNotFoundError("DocVQA not cached. Run: "
            "load_dataset('lmms-lab/DocVQA', 'DocVQA', split='validation') first.")

    pq_files = sorted(glob.glob(os.path.join(snap_dirs[0], "validation-*.parquet")))
    if not pq_files:
        raise FileNotFoundError(f"No validation parquet files in {snap_dirs[0]}")

    # Pass 1: read text columns only, group by docId
    print(f"  [DocVQA] Pass 1: grouping questions by document...")
    doc_questions = defaultdict(list)
    doc_row_map = defaultdict(list)  # docId -> [(file_idx, row_idx)]
    for fi, pf in enumerate(pq_files):
        tbl = pq.read_table(pf, columns=["docId", "question", "answers"])
        doc_ids = tbl.column("docId").to_pylist()
        questions = tbl.column("question").to_pylist()
        answers_col = tbl.column("answers").to_pylist()
        for ri, (did, q, ans) in enumerate(zip(doc_ids, questions, answers_col)):
            answer = ans[0] if ans else ""
            doc_questions[did].append((q, answer))
            doc_row_map[did].append((fi, ri))

    eligible = [did for did, qas in doc_questions.items() if len(qas) >= 1]
    print(f"  [DocVQA] {sum(len(v) for v in doc_questions.values())} QA pairs, "
          f"{len(eligible)} documents")

    random.shuffle(eligible)
    if num_images > 0:
        eligible = eligible[:num_images]

    # Pass 2: decode only images we need
    print(f"  [DocVQA] Pass 2: loading {len(eligible)} document images...")
    need_from_file = defaultdict(dict)  # file_idx -> {row_idx: docId}
    for did in eligible:
        fi, ri = doc_row_map[did][0]  # first occurrence
        need_from_file[fi][ri] = did

    img_store = {}
    for fi in sorted(need_from_file):
        rows_needed = need_from_file[fi]
        tbl = pq.read_table(pq_files[fi], columns=["image"])
        img_col = tbl.column("image")
        for ri, did in rows_needed.items():
            img_struct = img_col[ri].as_py()
            img = Image.open(BytesIO(img_struct["bytes"]))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_store[did] = img

    samples = []
    for did in eligible:
        qas = doc_questions[did]
        random.shuffle(qas)
        samples.append((img_store[did], qas[:questions_per_image]))

    print(f"  {len(samples)} images × {questions_per_image} questions/image")
    return samples


def load_vqav2_multi_round(num_images, questions_per_image=5):
    """Load VQAv2 dataset grouped by image for multi-round evaluation.

    VQAv2 has ~5.4 questions per COCO image on average.
    Two-pass: read parquet text columns to group (fast), then decode needed images.
    Returns: list of (PIL.Image, [(question, answer), ...])
    """
    import glob
    import pickle
    import tempfile
    import pyarrow.parquet as pq
    from collections import defaultdict
    from io import BytesIO

    # Locate cached parquet files
    cache_base = _main_hf_cache("VQAv2")
    snap_dirs = glob.glob(os.path.join(cache_base, "snapshots", "*", "data"))
    if not snap_dirs:
        raise FileNotFoundError("VQAv2 not cached. Run: "
            "load_dataset('lmms-lab/VQAv2', split='validation') first.")
    data_dir = snap_dirs[0]
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "validation-*.parquet")))
    snap_id = os.path.basename(os.path.dirname(data_dir))
    cache_dir = os.path.join(
        os.environ.get("DUALSIGNAL_PREPARED_CACHE", os.path.join(os.path.expanduser("~/.cache"), "dualsignal")),
        "prepared_vqav2")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f"validation_{snap_id}_n{num_images}_q{questions_per_image}_deterministic_seed42_v2.pkl",
    )

    if os.path.exists(cache_path):
        print(f"  [VQAv2] Loading prepared cache: {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        samples = []
        for img_bytes, qas in cached:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            samples.append((img, qas))
        print(f"  {len(samples)} images × {questions_per_image} questions/image")
        return samples

    # Pass 1: read text columns only from parquet (skip image bytes)
    print("  [VQAv2] Pass 1: grouping questions by image...", flush=True)
    img_questions = defaultdict(list)
    # Store (file_idx, row_idx) for image loading
    for fi, fpath in enumerate(parquet_files):
        table = pq.read_table(fpath,
                              columns=["image_id", "question", "multiple_choice_answer"])
        for ri in range(len(table)):
            iid = table["image_id"][ri].as_py()
            q = table["question"][ri].as_py()
            a = table["multiple_choice_answer"][ri].as_py() or ""
            img_questions[iid].append((q, a, fi, ri))

    print(f"  [VQAv2] {sum(len(v) for v in img_questions.values())} QA pairs, "
          f"{len(img_questions)} images", flush=True)

    eligible = [(iid, qas) for iid, qas in img_questions.items()
                if len(qas) >= 1]
    random.shuffle(eligible)
    if num_images > 0:
        eligible = eligible[:num_images]

    # Pass 2: decode only needed images from parquet.  VQAv2 image bytes are
    # embedded in parquet, so group requested rows by file and read each image
    # column once instead of re-reading the parquet file for every selected row.
    print(f"  [VQAv2] Pass 2: loading {len(eligible)} images...", flush=True)
    need_from_file = defaultdict(dict)  # file_idx -> {row_idx: image_id}
    for iid, qas in eligible:
        fi, ri = qas[0][2], qas[0][3]
        need_from_file[fi][ri] = iid

    img_store = {}
    for fi in sorted(need_from_file):
        rows_needed = need_from_file[fi]
        table = pq.read_table(parquet_files[fi], columns=["image"])
        img_col = table.column("image")
        for ri, iid in rows_needed.items():
            img_store[iid] = img_col[ri].as_py()

    samples = []
    cache_samples = []
    for iid, qas in eligible:
        if iid not in img_store:
            continue
        img_data = img_store[iid]
        if isinstance(img_data, dict) and "bytes" in img_data:
            img_bytes = img_data["bytes"]
        else:
            img_bytes = img_data
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        qa_pairs = [(q, a) for q, a, _, _ in qas]
        random.shuffle(qa_pairs)
        selected_qas = qa_pairs[:questions_per_image]
        samples.append((img, selected_qas))
        cache_samples.append((img_bytes, selected_qas))

    print(f"  {len(samples)} images × {questions_per_image} questions/image")
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(cache_path) + ".",
            suffix=".tmp",
            dir=cache_dir,
        )
        with os.fdopen(fd, "wb") as f:
            pickle.dump(cache_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        print(f"  [VQAv2] Prepared cache saved: {cache_path}", flush=True)
    except Exception as exc:
        print(f"  [VQAv2] Prepared cache save skipped: {exc}", flush=True)
    return samples


def load_clevr_multi_round(num_images, questions_per_image=5,
                           data_dir="datasets/CLEVR_v1.0"):
    """Load CLEVR dataset grouped by image for multi-round evaluation.

    CLEVR has ~10 questions per synthetic scene image.
    Expects extracted CLEVR_v1.0 directory at data_dir.
    Returns: list of (PIL.Image, [(question, answer), ...])
    """
    from collections import defaultdict

    # Load questions JSON
    q_path = os.path.join(data_dir, "questions", "CLEVR_val_questions.json")
    if not os.path.exists(q_path):
        raise FileNotFoundError(f"CLEVR questions not found at {q_path}")

    import json as _json
    with open(q_path) as f:
        qdata = _json.load(f)

    # Group questions by image
    img_questions = defaultdict(list)
    for item in qdata["questions"]:
        fname = item["image_filename"]
        img_questions[fname].append((item["question"], str(item["answer"])))

    # Filter images with enough questions
    eligible = [(fname, qas) for fname, qas in img_questions.items()
                if len(qas) >= 1]
    random.shuffle(eligible)
    if num_images > 0:
        eligible = eligible[:num_images]

    img_dir = os.path.join(data_dir, "images", "val")
    samples = []
    for fname, qas in eligible:
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        random.shuffle(qas)
        samples.append((img, qas[:questions_per_image]))

    print(f"  {len(samples)} images × {questions_per_image} questions/image")
    return samples


def load_visual7w_multi_round(num_images, questions_per_image=4,
                              data_dir="datasets/visual7w"):
    """Load Visual7W dataset grouped by image for multi-round evaluation.

    Visual7W has ~4.9 questions per image (7W types: what/where/when/who/why/how/which).
    Images are loaded from Visual Genome URLs on-the-fly.
    Returns: list of (PIL.Image, [(question, answer), ...])
    """
    import json as _json
    import urllib.request
    from io import BytesIO

    json_path = os.path.join(data_dir, "dataset_v7w_telling.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Visual7W JSON not found at {json_path}")

    with open(json_path) as f:
        v7w = _json.load(f)

    # Use test split for evaluation
    test_images = [img for img in v7w["images"] if img["split"] == "test"]

    # Filter images with enough questions
    eligible = [(img["image_id"], img["qa_pairs"])
                for img in test_images
                if len(img["qa_pairs"]) >= 1]
    random.shuffle(eligible)
    if num_images > 0:
        eligible = eligible[:num_images]

    # Load images from Visual Genome URLs
    VG_BASES = [
        "https://cs.stanford.edu/people/rak248/VG_100K",
        "https://cs.stanford.edu/people/rak248/VG_100K_2",
    ]

    # Cache dir for downloaded images
    cache_dir = os.path.join(data_dir, "images_cache")
    os.makedirs(cache_dir, exist_ok=True)

    samples = []
    for iid, qa_pairs in eligible:
        # Try loading from cache first
        cache_path = os.path.join(cache_dir, f"{iid}.jpg")
        if os.path.exists(cache_path):
            img = Image.open(cache_path).convert("RGB")
        else:
            img = None
            for base in VG_BASES:
                url = f"{base}/{iid}.jpg"
                try:
                    data = urllib.request.urlopen(url, timeout=10).read()
                    img = Image.open(BytesIO(data)).convert("RGB")
                    # Cache locally
                    with open(cache_path, "wb") as f:
                        f.write(data)
                    break
                except Exception:
                    continue
            if img is None:
                continue

        qas = [(qa["question"], qa["answer"]) for qa in qa_pairs]
        random.shuffle(qas)
        samples.append((img, qas[:questions_per_image]))

        if len(samples) % 10 == 0:
            print(f"  loaded {len(samples)} images...", flush=True)

    print(f"  {len(samples)} images × {questions_per_image} questions/image")
    return samples


# ──────────────── Dual-stage helper (generic) ─────────────────

def dual_stage_select(lm, full_embeds, vis_pos, text_pos, n_vis, S,
                      vit_importance=None, vis_embeds=None, scoring="single"):
    """Our dual-stage: Stage1 (dual signal + MMR → 33%) → Stage2 (L_S → 10% total).

    Args:
        vit_importance: [n_vis] ViT last-block received attention (for Stage 1 dual signal)
        vis_embeds:     [n_vis, dim] visual token embeddings (for MMR diversity)
        scoring:        "single" = importance at layer S only (1 QK)
                        "max"    = max importance over layers 0..S-1 (S QK computations)

    Returns:
        final_mask: list[bool] of length n_vis (True=keep)
        stage1_idx: set of original vis indices kept in Stage 1
        stage2_idx: set of original vis indices kept in Stage 2
    """
    k1 = max(1, int(n_vis * STAGE1_FRAC))
    k2 = max(1, int(n_vis * STAGE2_FRAC))

    # Stage 1: dual signal (L0 attn + ViT attn) + MMR diversity
    imp0 = _get_importance_at_layer(lm, 0, full_embeds, vis_pos, text_pos)
    if vit_importance is not None and vis_embeds is not None:
        stage1_idx = _dual_signal_stage1_select(imp0, vit_importance, vis_embeds, k1)
    else:
        # Fallback: L0 attention only (no ViT available)
        stage1_idx = np.argsort(imp0)[::-1][:k1].tolist()
    stage1_set = set(stage1_idx)

    # Stage 2: importance scoring on reduced sequence
    surviving_vis_pos = [vis_pos[i] for i in sorted(stage1_set)]

    if scoring == "max":
        # Max over layers 0..S-1 (S QK computations)
        imp_layers = [_get_importance_at_layer(lm, L, full_embeds, surviving_vis_pos, text_pos)
                      for L in range(S)]
        impS = np.maximum.reduce(imp_layers)
    else:
        # Single layer S (1 QK computation)
        impS = _get_importance_at_layer(lm, S, full_embeds, surviving_vis_pos, text_pos)

    # Select top-k2 among survivors
    stage2_local = np.argsort(impS)[::-1][:k2]
    surviving_orig = sorted(stage1_set)
    stage2_set = set(surviving_orig[j] for j in stage2_local)

    final_mask = [i in stage2_set for i in range(n_vis)]
    return final_mask, stage1_set, stage2_set


# ──────────── Actual dual-stage generation (layer-by-layer) ──────────
#
# Architecture:
#   Layers 0..S-1 : flash-attention with Stage 1 tokens (33%)
#   Layer S       : manual Q@K to compute importance → prune to Stage 2 (10%)
#   Layers S+1..N : flash-attention with Stage 2 tokens (10%)
#   Autoregressive: greedy decode through all N layers
#
# Position encoding:
#   Qwen2.5-VL / Qwen3-VL  : MROPE, position_ids [3, 1, seq], needs position_embeddings
#   InternVL3 (Qwen2)       : standard RoPE, position_ids [1, seq]
#   LLaVA (Llama)            : standard RoPE, position_ids [1, seq]


def _compute_pos_embeddings(lm, position_ids, hidden_for_shape, device, dtype):
    """Compute (cos, sin) position embeddings if model uses explicit rotary_emb.

    Returns:
        dict of extra kwargs for layer.forward():
            {"position_embeddings": (cos, sin)} for Qwen-style MROPE
            {"position_ids": pos}               for Llama-style
    """
    if hasattr(lm, 'rotary_emb') and position_ids is not None:
        # Qwen-style: rotary_emb expects (x, position_ids) → (cos, sin)
        dummy = torch.zeros(1, position_ids.shape[-1], lm.config.hidden_size,
                            device=device, dtype=dtype)
        cos, sin = lm.rotary_emb(dummy, position_ids.to(device))
        return {"position_embeddings": (cos, sin)}
    else:
        # Llama-style: pass position_ids to each layer
        if position_ids is not None:
            return {"position_ids": position_ids.to(device)}
        return {}


def _importance_from_hidden(lm, layer_idx, hidden_states, vis_pos, text_pos):
    """Manual Q@K at layer_idx (no flash-attention) to compute text→vis importance.

    Takes hidden_states output from layer (layer_idx - 1).
    """
    layer = lm.layers[layer_idx]
    attn = getattr(layer, 'self_attn', None) or layer.attention
    ln = getattr(layer, 'input_layernorm', None) or layer.attention_norm
    seq_len = hidden_states.shape[1]

    with torch.no_grad():
        h = ln(hidden_states[0])
        if hasattr(attn, 'wqkv'):
            cfg = attn.config if hasattr(attn, 'config') else lm.config
            n_heads = cfg.num_attention_heads
            n_kv_heads = cfg.num_key_value_heads
            head_dim = cfg.hidden_size // n_heads
            qkv = attn.wqkv(h).float()
            q_raw = qkv[:, :n_heads * head_dim]
            k_raw = qkv[:, n_heads * head_dim:n_heads * head_dim + n_kv_heads * head_dim]
        else:
            n_heads = getattr(attn, 'num_heads', None) or attn.config.num_attention_heads
            n_kv_heads = getattr(attn, 'num_key_value_heads', None) or attn.config.num_key_value_heads
            head_dim = getattr(attn, 'head_dim', None) or (attn.config.hidden_size // n_heads)
            q_raw = attn.q_proj(h).float()
            k_raw = attn.k_proj(h).float()

    q = q_raw.view(seq_len, n_heads, head_dim).permute(1, 0, 2)
    k = k_raw.view(seq_len, n_kv_heads, head_dim).permute(1, 0, 2)
    if n_kv_heads < n_heads:
        k = k.repeat_interleave(n_heads // n_kv_heads, dim=0)

    q_text = q[:, text_pos, :]
    k_vis = k[:, vis_pos, :]
    scores = (q_text @ k_vis.transpose(-1, -2)) * (head_dim ** -0.5)
    weights = F.softmax(scores, dim=-1)
    return weights.mean(dim=(0, 1)).cpu().numpy()


def _clone_cache(src):
    """Deep-clone a DynamicCache (tensors are cloned, not shared)."""
    from transformers.cache_utils import DynamicCache
    dst = DynamicCache()
    for layer in src.layers:
        dst.update(layer.keys.clone(), layer.values.clone(),
                   layer_idx=len(dst.layers))
    return dst


def _prefill_shared_layers(lm, shared_embeds, S, position_ids=None):
    """Prefill shared prefix (sys + Stage1 vis + vision_end) through layers 0..S-1.

    Called ONCE per image. The returned KV cache is cloned per question.

    Args:
        lm             : language model (.layers, .embed_tokens, optionally .rotary_emb)
        shared_embeds  : [1, n_prefix, hidden]
        S              : split layer index
        position_ids   : [3, 1, n_prefix] for MROPE or [1, n_prefix] for standard.
                         If None, auto-created as arange.

    Returns:
        h_shared      : [1, n_prefix, hidden] — hidden states at layer S input
        shared_cache  : DynamicCache with S entries (layers 0..S-1)
    """
    from transformers.cache_utils import DynamicCache

    device = shared_embeds.device
    dtype = shared_embeds.dtype

    if position_ids is None:
        position_ids = torch.arange(shared_embeds.shape[1], device=device).unsqueeze(0)
    pos_kwargs = _compute_pos_embeddings(lm, position_ids, shared_embeds, device, dtype)

    cache = DynamicCache()
    h = shared_embeds
    with torch.no_grad():
        for i in range(S):
            h = lm.layers[i](
                hidden_states=h, attention_mask=None,
                past_key_values=cache, use_cache=True,
                **pos_kwargs)

    return h, cache


def _per_question_prune_generate(lm, lm_head, h_shared, shared_cache,
                                  question_embeds, vis_pos_in_shared,
                                  n_vis_stage1, S, stage2_frac,
                                  eos_id, max_new_tokens=32,
                                  shared_pos_ids=None, question_pos_ids=None,
                                  scoring="single"):
    """Per-question: clone shared cache → forward question through L0..S-1 →
    manual Q@K at layer S → prune → forward S..N-1 → generate.

    This is the actual multi-round deployment path:
      - Layers 0..S-1 KV for shared prefix are REUSED (cloned)
      - Only question tokens go through layers 0..S-1 (cheap)
      - At layer S: concatenate h_shared + h_question, manual attention → importance
      - Prune to 10% vis tokens → forward S..N-1 → generate

    Args:
        lm                 : language model
        lm_head            : output projection (Linear)
        h_shared           : [1, n_shared, hidden] — shared prefix hidden at layer S input
        shared_cache       : DynamicCache with S entries from _prefill_shared_layers
        question_embeds    : [1, n_question, hidden] — question-only token embeddings
        vis_pos_in_shared  : list[int] — visual token positions within h_shared
        n_vis_stage1       : int — count of Stage 1 visual tokens
        S                  : split layer index
        stage2_frac        : float — fraction of original vis to keep (0.10)
        eos_id             : int
        max_new_tokens     : int
        shared_pos_ids     : position_ids for shared prefix ([3,1,n_shared] or [1,n_shared])
        question_pos_ids   : position_ids for question tokens ([3,1,n_q] or [1,n_q])
        scoring            : "single" = Q@K at layer S only;
                             "max"    = max Q@K over layers 0..S-1 (collected during shallow fwd)

    Returns:
        list[int] — generated token ids
    """
    from transformers.cache_utils import DynamicCache

    N = len(lm.layers)
    device = h_shared.device
    dtype = h_shared.dtype
    n_shared = h_shared.shape[1]
    n_question = question_embeds.shape[1]

    # ── Auto-create position_ids for standard RoPE models ──
    if shared_pos_ids is None:
        shared_pos_ids = torch.arange(n_shared, device=device).unsqueeze(0)
    if question_pos_ids is None:
        question_pos_ids = torch.arange(n_shared, n_shared + n_question,
                                         device=device).unsqueeze(0)

    # ══════════════════════════════════════════════════════════
    # Step 1: Clone shared cache, forward question through L0..S-1
    # ══════════════════════════════════════════════════════════
    q_cache = _clone_cache(shared_cache)
    q_pos_kwargs = _compute_pos_embeddings(lm, question_pos_ids, question_embeds, device, dtype)

    # Causal mask for incremental forward (q_len tokens attending to kv_len cached+new):
    #   - FA2 (flash_attention_2): causal mask 右下角对齐, attention_mask=None 即可
    #   - SDPA: is_causal=True 左上角对齐 → query[0] 只看 key[0] → 必须传显式 4D mask
    attn_impl = lm.config._attn_implementation
    if attn_impl in ("flash_attention_2", "flash_attention_3", "flash_attention_4"):
        step1_mask = None  # FA2 causal 对齐方式天然正确
    else:
        kv_len = n_shared + n_question
        step1_mask = torch.full((n_question, kv_len), torch.finfo(dtype).min,
                                device=device, dtype=dtype)
        for qi in range(n_question):
            step1_mask[qi, :n_shared + qi + 1] = 0.0
        step1_mask = step1_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]

    h_q = question_embeds
    imp_layers = []  # for max scoring: collect per-layer importance
    with torch.no_grad():
        for i in range(S):
            if scoring == "max":
                # Collect importance at each layer using Q from question, K from shared cache
                imp_i = _importance_from_hidden_and_cache(
                    lm, i, h_q, shared_cache, vis_pos_in_shared)
                imp_layers.append(imp_i)
            h_q = lm.layers[i](
                hidden_states=h_q, attention_mask=step1_mask,
                past_key_values=q_cache, use_cache=True,
                **q_pos_kwargs)
    # q_cache layers 0..S-1 now have [shared + question] KV

    # ══════════════════════════════════════════════════════════
    # Step 2: Importance scoring → select 10%
    # ══════════════════════════════════════════════════════════
    h_full = torch.cat([h_shared, h_q], dim=1)  # [1, n_shared + n_question, hidden]
    vis_pos_set = set(vis_pos_in_shared)

    if scoring == "max":
        # Max over all layers collected during shallow forward
        impS = np.maximum.reduce(imp_layers)
    else:
        # Single layer S: manual Q@K using concatenated hidden states
        text_pos_full = [p for p in range(n_shared + n_question) if p not in vis_pos_set]
        impS = _importance_from_hidden(lm, S, h_full, vis_pos_in_shared, text_pos_full)

    k2 = max(1, int(n_vis_stage1 * (stage2_frac / STAGE1_FRAC)))
    k2 = min(k2, n_vis_stage1)
    stage2_local = sorted(np.argsort(impS)[::-1][:k2].tolist())

    # ══════════════════════════════════════════════════════════
    # Step 3: Prune hidden states + KV cache
    # ══════════════════════════════════════════════════════════
    vis_keep = [vis_pos_in_shared[j] for j in stage2_local]
    non_vis = [p for p in range(n_shared + n_question) if p not in vis_pos_set]
    keep_positions = sorted(set(vis_keep) | set(non_vis))
    keep_t = torch.tensor(keep_positions, device=device, dtype=torch.long)

    h_pruned = h_full[:, keep_t, :]

    # Prune KV cache (layers 0..S-1)
    pruned_cache = DynamicCache()
    for layer_kv in q_cache.layers:
        pruned_cache.update(
            layer_kv.keys[:, :, keep_t, :],
            layer_kv.values[:, :, keep_t, :],
            layer_idx=len(pruned_cache.layers))
    del q_cache, h_full

    # ══════════════════════════════════════════════════════════
    # Step 4: Position embeddings for pruned sequence
    # ══════════════════════════════════════════════════════════
    if shared_pos_ids.dim() == 3:
        full_pos = torch.cat([shared_pos_ids.to(device), question_pos_ids.to(device)], dim=2)
        pruned_pos = full_pos[:, :, keep_t]
    else:
        full_pos = torch.cat([shared_pos_ids.to(device), question_pos_ids.to(device)], dim=1)
        pruned_pos = full_pos[:, keep_t]
    pos_kwargs_pruned = _compute_pos_embeddings(lm, pruned_pos, h_pruned, device, dtype)

    # ══════════════════════════════════════════════════════════
    # Step 5: Forward through layers S..N-1  (flash-attention)
    # ══════════════════════════════════════════════════════════
    with torch.no_grad():
        for i in range(S, N):
            h_pruned = lm.layers[i](
                hidden_states=h_pruned, attention_mask=None,
                past_key_values=pruned_cache, use_cache=True,
                **pos_kwargs_pruned)

    # ══════════════════════════════════════════════════════════
    # Step 6: Autoregressive generation
    # ══════════════════════════════════════════════════════════
    h_final = lm.norm(h_pruned)
    logits = lm_head(h_final[:, -1:, :])
    next_id = logits.argmax(dim=-1).squeeze().item()
    generated = [next_id]
    # Debug: print cache shapes after forward
    if os.environ.get("DEBUG_PRUNE"):
        print(f"  [DEBUG] n_shared={n_shared} n_q={n_question} "
              f"n_pruned={h_pruned.shape[1]} n_vis_kept={len(stage2_local)} "
              f"cache_layers={len(pruned_cache)} "
              f"cache0_seq={pruned_cache.layers[0].keys.shape[2]} "
              f"cacheN_seq={pruned_cache.layers[-1].keys.shape[2]} "
              f"first_tok={next_id} "
              f"h_norm_mean={h_final[:,-1,:].abs().mean().item():.4f}", flush=True)

    if pruned_pos.dim() == 3:
        last_pos = pruned_pos[:, :, -1:] + 1
    else:
        last_pos = pruned_pos[:, -1:] + 1

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if next_id == eos_id:
                break
            tok_emb = lm.embed_tokens(
                torch.tensor([[next_id]], device=device)).to(dtype)
            gen_pos_kwargs = _compute_pos_embeddings(lm, last_pos, tok_emb, device, dtype)

            for i in range(N):
                tok_emb = lm.layers[i](
                    hidden_states=tok_emb, attention_mask=None,
                    past_key_values=pruned_cache, use_cache=True,
                    **gen_pos_kwargs)

            h_tok = lm.norm(tok_emb)
            logits = lm_head(h_tok[:, -1:, :])
            next_id = logits.argmax(dim=-1).squeeze().item()
            generated.append(next_id)
            last_pos = last_pos + 1

    return generated


# ──────── SparseVILA full: prefill pruning + decode-stage KV sparsity ────────
#
# Paper: "we set a constant prefill sparsity before the LLM and a uniform
#         decoding sparsity across all layers"
#       "Before decoding begins, SparseVILA estimates the relevance of each
#        visual token to the current query using attention-based salience"
#       "selected visual KV entries are compactly packed into a contiguous
#        memory region"
#
# Key: ONE global token set, same for ALL layers (not per-layer different sets).
# Salience = aggregate text→vis attention across all layers during prefill.

SPARSEVILA_DECODE_RATIO = 0.5  # keep 50% of prefill-retained vis tokens

def _sparsevila_full_generate(lm, lm_head, pruned_embeds, vis_pos_in_pruned,
                               decode_ratio, eos_id, max_new_tokens=32,
                               position_ids=None):
    """Full SparseVILA (ICCV 2025): prefill + uniform decode-stage KV sparsity.

    1. Forward pruned_embeds (after ViT-attention prefill pruning) through all N
       layers with full causal attention.
    2. At each layer, compute text→vis salience and accumulate across layers.
    3. Select ONE global set of top-k visual tokens (uniform across all layers).
    4. Prune KV cache: keep selected visual + all text tokens (same set, all layers).
    5. Generate autoregressively with pruned cache.

    Args:
        lm              : language model (.layers, .embed_tokens, .norm, .rotary_emb)
        lm_head         : output projection
        pruned_embeds   : [1, seq_len, hidden] — embeddings after prefill pruning
        vis_pos_in_pruned : list[int] — visual token positions within pruned_embeds
        decode_ratio    : float — fraction of visual tokens to keep during decode
        eos_id          : int
        max_new_tokens  : int
        position_ids    : [1, seq_len] or [3, 1, seq_len]. If None, auto-created.

    Returns:
        list[int] — generated token IDs
    """
    from transformers.cache_utils import DynamicCache

    N = len(lm.layers)
    device = pruned_embeds.device
    dtype = pruned_embeds.dtype
    seq_len = pruned_embeds.shape[1]

    vis_set = set(vis_pos_in_pruned)
    text_pos = [p for p in range(seq_len) if p not in vis_set]
    n_vis = len(vis_pos_in_pruned)
    n_decode_keep = max(1, int(n_vis * decode_ratio))

    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    pos_kwargs = _compute_pos_embeddings(lm, position_ids, pruned_embeds, device, dtype)

    # ── Step 1: Prefill all layers, aggregate text→vis salience ──
    cache = DynamicCache()
    h = pruned_embeds
    agg_salience = np.zeros(n_vis, dtype=np.float64)

    with torch.no_grad():
        for i in range(N):
            imp = _importance_from_hidden(lm, i, h, vis_pos_in_pruned, text_pos)
            agg_salience += imp  # accumulate across layers

            h = lm.layers[i](
                hidden_states=h, attention_mask=None,
                past_key_values=cache, use_cache=True,
                **pos_kwargs)

    # ── Step 2: First token logits (from full-attention prefill) ──
    h_final = lm.norm(h)
    logits = lm_head(h_final[:, -1:, :])
    next_id = logits.argmax(dim=-1).squeeze().item()
    generated = [next_id]

    if next_id == eos_id or max_new_tokens <= 1:
        return generated

    # ── Step 3: Select ONE global token set, prune KV cache uniformly ──
    top_local = sorted(np.argsort(agg_salience)[::-1][:n_decode_keep].tolist())
    keep_vis = [vis_pos_in_pruned[j] for j in top_local]
    keep_positions = sorted(set(keep_vis) | set(text_pos))
    keep_t = torch.tensor(keep_positions, device=device, dtype=torch.long)

    pruned_cache = DynamicCache()
    for i in range(N):
        pruned_cache.update(
            cache.layers[i].keys[:, :, keep_t, :],
            cache.layers[i].values[:, :, keep_t, :],
            layer_idx=i)
    del cache

    # ── Step 4: Autoregressive generation with uniformly pruned cache ──
    if position_ids.dim() == 3:
        last_pos = position_ids[:, :, -1:] + 1
    else:
        last_pos = position_ids[:, -1:] + 1

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if next_id == eos_id:
                break
            tok_emb = lm.embed_tokens(
                torch.tensor([[next_id]], device=device)).to(dtype)
            gen_pos_kwargs = _compute_pos_embeddings(lm, last_pos, tok_emb, device, dtype)

            for i in range(N):
                tok_emb = lm.layers[i](
                    hidden_states=tok_emb, attention_mask=None,
                    past_key_values=pruned_cache, use_cache=True,
                    **gen_pos_kwargs)

            h_tok = lm.norm(tok_emb)
            logits = lm_head(h_tok[:, -1:, :])
            next_id = logits.argmax(dim=-1).squeeze().item()
            generated.append(next_id)
            last_pos = last_pos + 1

    return generated


# ──────── Multi-round KV cache reuse for ALL baselines ────────
#
# All baselines (PACT, SparseVILA, Attn top-k) are query-agnostic:
# they select a fixed token set per image. In multi-round, we prefill
# the shared prefix (sys + selected_vis + vision_end) through ALL N
# layers ONCE, cache the KV, then per-question only forward the
# question tokens attending to the cached KV.


def _prefill_all_layers(lm, shared_embeds, position_ids=None):
    """Prefill shared prefix through ALL N layers. Returns KV cache.

    Called ONCE per image per baseline method.

    Args:
        lm             : language model
        shared_embeds  : [1, n_prefix, hidden]
        position_ids   : position IDs for MROPE or standard

    Returns:
        shared_cache  : DynamicCache with N entries (all layers)
    """
    from transformers.cache_utils import DynamicCache

    N = len(lm.layers)
    device = shared_embeds.device
    dtype = shared_embeds.dtype

    if position_ids is None:
        position_ids = torch.arange(shared_embeds.shape[1], device=device).unsqueeze(0)
    pos_kwargs = _compute_pos_embeddings(lm, position_ids, shared_embeds, device, dtype)

    cache = DynamicCache()
    h = shared_embeds
    with torch.no_grad():
        for i in range(N):
            h = lm.layers[i](
                hidden_states=h, attention_mask=None,
                past_key_values=cache, use_cache=True,
                **pos_kwargs)

    return cache, h


@torch.no_grad()
def _progressive_prefill(lm, full_embeds, vis_pos_set, drop_fn,
                          position_ids=None):
    """Progressive prefill: single N-layer forward with token pruning at boundaries.

    Unlike _prefill_all_layers (uniform token count across layers), this builds
    a KV cache where each layer may have a different number of tokens — early
    layers keep more visual tokens, later layers fewer.

    Args:
        lm           : language model
        full_embeds  : [1, seq_len, D]  (all tokens including ALL visual)
        vis_pos_set  : set of original visual token positions
        drop_fn      : callable(hidden, alive, vis_pos_set, l_idx, lm)
                       → set of alive-indices to DROP (empty = no pruning)
        position_ids : optional position IDs (MROPE or standard)

    Returns:
        cache      : DynamicCache (variable sizes per layer)
        hidden     : [1, remaining_seq, D] final hidden states
        alive      : list of original positions still alive
        total_tl   : sum of token counts across all layers (for cost)
    """
    from transformers.cache_utils import DynamicCache

    device = full_embeds.device
    dtype = full_embeds.dtype
    N = len(lm.layers)

    if position_ids is None:
        position_ids = torch.arange(full_embeds.shape[1], device=device).unsqueeze(0)
    pos_kwargs = _compute_pos_embeddings(lm, position_ids, full_embeds, device, dtype)

    cache = DynamicCache()
    hidden = full_embeds
    alive = list(range(full_embeds.shape[1]))
    total_tl = 0

    for l_idx in range(N):
        # Ask drop_fn which tokens to drop BEFORE this layer
        to_drop = drop_fn(hidden, alive, vis_pos_set, l_idx, lm)

        if to_drop:
            keep = [i for i in range(len(alive)) if i not in to_drop]
            idx = torch.tensor(keep, dtype=torch.long, device=device)

            hidden = hidden[:, idx, :]

            # Prune position embeddings
            if "position_embeddings" in pos_kwargs:
                cos, sin = pos_kwargs["position_embeddings"]
                if cos.ndim == 4:  # Qwen MROPE: [3, 1, seq, dim]
                    pos_kwargs = {"position_embeddings": (
                        cos[:, :, idx, :], sin[:, :, idx, :])}
                else:  # Standard: [1, seq, dim]
                    pos_kwargs = {"position_embeddings": (
                        cos[:, idx, :], sin[:, idx, :])}
            elif "position_ids" in pos_kwargs:
                pos_kwargs = {"position_ids":
                              pos_kwargs["position_ids"][:, idx]}

            # Update tracking
            dropped_pos = {alive[i] for i in to_drop}
            vis_pos_set = vis_pos_set - dropped_pos
            alive = [alive[i] for i in keep]

        total_tl += hidden.shape[1]

        # Forward through this layer, accumulating KV cache
        hidden = lm.layers[l_idx](
            hidden_states=hidden, attention_mask=None,
            past_key_values=cache, use_cache=True,
            **pos_kwargs)

    return cache, hidden, alive, total_tl


def _gen_from_progressive_cache(lm, lm_head, prog_cache, max_pos,
                                 question_embeds, question_pos_ids,
                                 eos_id, max_new_tokens=32):
    """Generate from a variable-size KV cache (from _progressive_prefill).

    Same as _generate_from_shared_cache but uses attention_mask=None
    (works with both flash-attn and SDPA when cache sizes differ per layer).

    Args:
        max_pos: maximum position in the original prefix (for position continuity)
    """
    from transformers.cache_utils import DynamicCache

    N = len(lm.layers)
    device = question_embeds.device
    dtype = question_embeds.dtype

    q_cache = _clone_cache(prog_cache)
    q_pos_kwargs = _compute_pos_embeddings(lm, question_pos_ids, question_embeds,
                                            device, dtype)

    # Forward question tokens — attention_mask=None works for variable caches
    h_q = question_embeds
    for i in range(N):
        h_q = lm.layers[i](
            hidden_states=h_q, attention_mask=None,
            past_key_values=q_cache, use_cache=True,
            **q_pos_kwargs)

    # First generated token
    h_final = lm.norm(h_q)
    logits = lm_head(h_final[:, -1:, :])
    next_id = logits.argmax(dim=-1).squeeze().item()
    generated = [next_id]

    if next_id == eos_id or max_new_tokens <= 1:
        return generated

    # Autoregressive
    if question_pos_ids.dim() == 3:
        last_pos = question_pos_ids[:, :, -1:] + 1
    else:
        last_pos = question_pos_ids[:, -1:] + 1

    for _ in range(max_new_tokens - 1):
        if next_id == eos_id:
            break
        tok_emb = lm.embed_tokens(
            torch.tensor([[next_id]], device=device)).to(dtype)
        gen_pos_kwargs = _compute_pos_embeddings(lm, last_pos, tok_emb, device, dtype)

        for i in range(N):
            tok_emb = lm.layers[i](
                hidden_states=tok_emb, attention_mask=None,
                past_key_values=q_cache, use_cache=True,
                **gen_pos_kwargs)

        h_tok = lm.norm(tok_emb)
        logits = lm_head(h_tok[:, -1:, :])
        next_id = logits.argmax(dim=-1).squeeze().item()
        generated.append(next_id)
        last_pos = last_pos + 1

    return generated


# ── Drop functions for progressive methods ──────────────────────────

def _make_fitprune_drop_fn(schedule):
    """Create drop function for FitPrune progressive prefill."""
    def drop_fn(hidden, alive, vis_pos_set, l_idx, lm):
        if l_idx not in schedule or schedule[l_idx] <= 0:
            return set()
        vis_h = [h for h, p in enumerate(alive) if p in vis_pos_set]
        text_h = [h for h, p in enumerate(alive) if p not in vis_pos_set]
        n_drop = min(schedule[l_idx], len(vis_h) - 1)
        if n_drop <= 0:
            return set()
        layer = lm.layers[l_idx]
        self_attn, _, _ = _layer_qk(layer, hidden, vis_h, vis_h)
        self_sc = self_attn.max(0).values.sum(0)
        if text_h:
            cross_attn, _, _ = _layer_qk(layer, hidden, text_h, vis_h)
            cross_sc = cross_attn.max(0).values.mean(0)
        else:
            cross_sc = torch.ones(len(vis_h), device=self_sc.device)
        combined = self_sc * cross_sc
        drop_local = combined.argsort()[:n_drop].tolist()
        return {vis_h[i] for i in drop_local}
    return drop_fn


def _make_pyramiddrop_drop_fn(n_vis, k_y, n_layers):
    """Create drop function for PyramidDrop progressive prefill."""
    num_stages = 4
    lps = n_layers // num_stages
    boundaries = [(s + 1) * lps for s in range(num_stages - 1)]
    final_ratio = k_y / n_vis
    if final_ratio >= 1.0:
        return lambda *a: set()
    lam = final_ratio ** (1.0 / (num_stages - 1))
    cum = [lam ** s for s in range(num_stages)]
    schedule = {}
    remaining = n_vis
    for s, bl in enumerate(boundaries):
        target = max(1, int(n_vis * cum[s + 1]))
        drop = remaining - target
        if drop > 0:
            schedule[bl] = drop
        remaining = target

    def drop_fn(hidden, alive, vis_pos_set, l_idx, lm):
        if l_idx not in schedule or schedule[l_idx] <= 0:
            return set()
        vis_h = [h for h, p in enumerate(alive) if p in vis_pos_set]
        text_h = [h for h, p in enumerate(alive) if p not in vis_pos_set]
        n_drop = min(schedule[l_idx], len(vis_h) - 1)
        if n_drop <= 0 or not text_h:
            return set()
        layer = lm.layers[l_idx]
        last_text = [text_h[-1]]
        attn, _, _ = _layer_qk(layer, hidden, last_text, vis_h)
        scores = attn.mean(0).squeeze(0)
        drop_local = scores.argsort()[:n_drop].tolist()
        return {vis_h[i] for i in drop_local}
    return drop_fn


def _make_sparsevlm_drop_fn(n_vis, k_y, n_layers, rater_ratio=0.5, alpha_erank=0.5):
    """Create drop function for SparseVLM progressive prefill."""
    import math as _math
    num_stages = 4
    lps = n_layers // num_stages
    boundaries = [s * lps for s in range(num_stages)]
    ratio = 1.0 - k_y / n_vis
    stage_targets = []
    for s in range(num_stages):
        frac = (s + 1) / num_stages
        n_s = max(k_y, round(n_vis - (n_vis - k_y) * frac))
        stage_targets.append(max(k_y, n_s))
    stage_targets[-1] = k_y

    state = {"rater_orig_pos": None, "n_raters": None, "init_done": False}

    def drop_fn(hidden, alive, vis_pos_set, l_idx, lm):
        # Initialize raters at layer 0
        if not state["init_done"]:
            vis_h0 = [h for h, p in enumerate(alive) if p in vis_pos_set]
            text_h0 = [h for h, p in enumerate(alive) if p not in vis_pos_set]
            n_raters = max(1, round(len(text_h0) * rater_ratio))
            state["n_raters"] = n_raters
            if text_h0 and vis_h0:
                attn_vt, _, _ = _layer_qk(lm.layers[0], hidden, vis_h0, text_h0)
                rater_scores = attn_vt.mean(0).sum(0)
                top_local = rater_scores.argsort(descending=True)[:n_raters].tolist()
                state["rater_orig_pos"] = {alive[text_h0[i]] for i in top_local}
            else:
                state["rater_orig_pos"] = {alive[t] for t in text_h0[:n_raters]}
            state["init_done"] = True

        if l_idx not in boundaries:
            return set()
        stage_idx = boundaries.index(l_idx)
        vis_h = [h for h, p in enumerate(alive) if p in vis_pos_set]
        n_current = len(vis_h)
        n_target = stage_targets[stage_idx]
        if n_current <= n_target or n_current <= 1:
            return set()

        # Rater → vis scoring
        rater_cur = [h for h, p in enumerate(alive) if p in state["rater_orig_pos"]]
        if not rater_cur:
            text_cur = [h for h, p in enumerate(alive) if p not in vis_pos_set]
            rater_cur = text_cur[-state["n_raters"]:] if text_cur else []

        if rater_cur:
            attn_rv, _, _ = _layer_qk(lm.layers[l_idx], hidden, rater_cur, vis_h)
            scores = attn_rv.mean(0).mean(0)
        else:
            scores = torch.ones(n_current)

        # Adaptive sparsity via effective rank
        vis_idx_t = torch.tensor(vis_h, dtype=torch.long, device=hidden.device)
        h_vis = hidden[0, vis_idx_t, :].float()
        try:
            s_vals = torch.linalg.svdvals(h_vis)
            s_vals = s_vals[s_vals > 1e-9]
            if s_vals.numel() > 0:
                p = s_vals / s_vals.sum()
                erank = _math.exp(-(p * (p + 1e-12).log()).sum().item())
            else:
                erank = float(n_current)
        except Exception:
            erank = float(n_current)
        erank_ratio = min(1.0, erank / max(n_current, 1))
        n_keep = max(n_target, round(n_current * (1 - ratio) *
                                     (erank_ratio ** alpha_erank)))
        n_keep = max(n_target, min(n_current, n_keep))

        n_drop = n_current - n_keep
        if n_drop <= 0:
            return set()
        drop_local = scores.argsort()[:n_drop].tolist()
        return {vis_h[i] for i in drop_local}
    return drop_fn


def _importance_from_hidden_and_cache(lm, layer_idx, q_hidden, cache,
                                       vis_indices_in_cache):
    """Compute query→vis importance using Q from question hidden, K from cache.

    Q is projected from question hidden states (no rotary).
    K is taken directly from the KV cache (with rotary — negligible impact on ranking).

    Args:
        lm                   : language model
        layer_idx            : which layer
        q_hidden             : [1, n_q, hidden] — question hidden states at layer input
        cache                : DynamicCache — the shared KV cache
        vis_indices_in_cache : list[int] — positions of visual tokens in the cache

    Returns:
        importance : np.ndarray[n_vis]
    """
    layer = lm.layers[layer_idx]
    attn = getattr(layer, 'self_attn', None) or layer.attention
    ln = getattr(layer, 'input_layernorm', None) or layer.attention_norm

    with torch.no_grad():
        h_q = ln(q_hidden[0])  # [n_q, hidden]

        if hasattr(attn, 'wqkv'):
            cfg = attn.config if hasattr(attn, 'config') else lm.config
            n_heads = cfg.num_attention_heads
            n_kv_heads = cfg.num_key_value_heads
            head_dim = cfg.hidden_size // n_heads
            q_raw = attn.wqkv(h_q).float()[:, :n_heads * head_dim]
        else:
            n_heads = getattr(attn, 'num_heads', None) or attn.config.num_attention_heads
            n_kv_heads = getattr(attn, 'num_key_value_heads', None) or attn.config.num_key_value_heads
            head_dim = getattr(attn, 'head_dim', None) or (attn.config.hidden_size // n_heads)
            q_raw = attn.q_proj(h_q).float()

    n_q = q_raw.shape[0]
    q = q_raw.view(n_q, n_heads, head_dim).permute(1, 0, 2)  # [n_heads, n_q, head_dim]

    # K from cache at visual positions (already includes rotary)
    vis_t = torch.tensor(vis_indices_in_cache, device=cache.layers[layer_idx].keys.device,
                         dtype=torch.long)
    k_vis = cache.layers[layer_idx].keys[0, :, vis_t, :].float()  # [n_kv_heads, n_vis, head_dim]
    if n_kv_heads < n_heads:
        k_vis = k_vis.repeat_interleave(n_heads // n_kv_heads, dim=0)

    scores = (q @ k_vis.transpose(-1, -2)) * (head_dim ** -0.5)
    weights = F.softmax(scores, dim=-1)
    return weights.mean(dim=(0, 1)).cpu().numpy()



def _prune_cache_by_salience(cache, vis_pos, text_pos_all, agg_salience,
                              n_decode_keep, device):
    """Prune KV cache: keep top-k visual tokens + all text tokens.

    Returns pruned cache and the kept position indices.
    """
    from transformers.cache_utils import DynamicCache

    n_vis = len(vis_pos)
    top_local = sorted(np.argsort(agg_salience)[::-1][:n_decode_keep].tolist())
    keep_vis = [vis_pos[j] for j in top_local]
    keep_positions = sorted(set(keep_vis) | set(text_pos_all))
    keep_t = torch.tensor(keep_positions, device=device, dtype=torch.long)

    pruned_cache = DynamicCache()
    N = len(cache.layers)
    for i in range(N):
        pruned_cache.update(
            cache.layers[i].keys[:, :, keep_t, :],
            cache.layers[i].values[:, :, keep_t, :],
            layer_idx=i)

    return pruned_cache, keep_positions


def _generate_from_shared_cache(lm, lm_head, shared_cache, shared_pos_ids,
                                 question_embeds, question_pos_ids,
                                 eos_id, max_new_tokens=32):
    """Per-question generation reusing shared KV cache.

    Clones shared_cache, forwards question tokens through all N layers
    attending to the cached KV, then generates autoregressively.

    Args:
        lm              : language model
        lm_head         : output projection
        shared_cache    : DynamicCache with N entries (from _prefill_all_layers)
        shared_pos_ids  : position IDs of shared prefix
        question_embeds : [1, n_q, hidden] — question-only embeddings
        question_pos_ids: position IDs for question tokens
        eos_id          : end-of-sequence token id
        max_new_tokens  : max tokens to generate

    Returns:
        list[int] — generated token IDs
    """
    from transformers.cache_utils import DynamicCache

    N = len(lm.layers)
    device = question_embeds.device
    dtype = question_embeds.dtype
    n_shared = shared_cache.layers[0].keys.shape[2]
    n_question = question_embeds.shape[1]

    # Clone cache so shared state is not mutated
    q_cache = _clone_cache(shared_cache)

    q_pos_kwargs = _compute_pos_embeddings(lm, question_pos_ids, question_embeds,
                                            device, dtype)

    # Attention mask for incremental forward
    attn_impl = lm.config._attn_implementation
    if attn_impl in ("flash_attention_2", "flash_attention_3", "flash_attention_4"):
        q_mask = None
    else:
        kv_len = n_shared + n_question
        q_mask = torch.full((n_question, kv_len), torch.finfo(dtype).min,
                            device=device, dtype=dtype)
        for qi in range(n_question):
            q_mask[qi, :n_shared + qi + 1] = 0.0
        q_mask = q_mask.unsqueeze(0).unsqueeze(0)

    # Forward question through all N layers
    h_q = question_embeds
    with torch.no_grad():
        for i in range(N):
            h_q = lm.layers[i](
                hidden_states=h_q, attention_mask=q_mask,
                past_key_values=q_cache, use_cache=True,
                **q_pos_kwargs)

    # First token
    h_final = lm.norm(h_q)
    logits = lm_head(h_final[:, -1:, :])
    next_id = logits.argmax(dim=-1).squeeze().item()
    generated = [next_id]

    if next_id == eos_id or max_new_tokens <= 1:
        return generated

    # Autoregressive generation
    if question_pos_ids.dim() == 3:
        last_pos = question_pos_ids[:, :, -1:] + 1
    else:
        last_pos = question_pos_ids[:, -1:] + 1

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if next_id == eos_id:
                break
            tok_emb = lm.embed_tokens(
                torch.tensor([[next_id]], device=device)).to(dtype)
            gen_pos_kwargs = _compute_pos_embeddings(lm, last_pos, tok_emb, device, dtype)

            for i in range(N):
                tok_emb = lm.layers[i](
                    hidden_states=tok_emb, attention_mask=None,
                    past_key_values=q_cache, use_cache=True,
                    **gen_pos_kwargs)

            h_tok = lm.norm(tok_emb)
            logits = lm_head(h_tok[:, -1:, :])
            next_id = logits.argmax(dim=-1).squeeze().item()
            generated.append(next_id)
            last_pos = last_pos + 1

    return generated


def _sparsevila_per_question_generate(lm, lm_head, shared_cache,
                                       vis_pos, n_vis, decode_ratio,
                                       question_embeds, question_pos_ids,
                                       eos_id, max_new_tokens=32):
    """SparseVILA per-question: clone shared cache → forward question → salience → prune → generate.

    The shared KV cache is from prefill-pruned visual tokens (query-independent).
    Decode pruning is query-conditioned: salience = question_Q @ visual_K at each layer.
    K is read directly from the KV cache (with rotary — negligible impact on ranking).

    Args:
        lm              : language model
        lm_head         : output projection
        shared_cache    : DynamicCache with N entries (from _prefill_all_layers)
        vis_pos         : list[int] — visual token positions in the prefix
        n_vis           : number of visual tokens in prefix
        decode_ratio    : fraction of visual tokens to keep for decode
        question_embeds : [1, n_q, hidden]
        question_pos_ids: position IDs for question
        eos_id          : EOS token id
        max_new_tokens  : max generate length

    Returns:
        list[int] — generated token IDs
    """
    from transformers.cache_utils import DynamicCache

    N = len(lm.layers)
    device = question_embeds.device
    dtype = question_embeds.dtype
    n_shared = shared_cache.layers[0].keys.shape[2]
    n_question = question_embeds.shape[1]
    n_decode_keep = max(1, int(n_vis * decode_ratio))

    # ── Step 1: Clone cache, forward question through all layers ──
    q_cache = _clone_cache(shared_cache)
    q_pos_kwargs = _compute_pos_embeddings(lm, question_pos_ids, question_embeds,
                                            device, dtype)

    attn_impl = lm.config._attn_implementation
    if attn_impl in ("flash_attention_2", "flash_attention_3", "flash_attention_4"):
        q_mask = None
    else:
        kv_len = n_shared + n_question
        q_mask = torch.full((n_question, kv_len), torch.finfo(dtype).min,
                            device=device, dtype=dtype)
        for qi in range(n_question):
            q_mask[qi, :n_shared + qi + 1] = 0.0
        q_mask = q_mask.unsqueeze(0).unsqueeze(0)

    # Forward question while accumulating salience
    # K is taken from the shared cache (vis positions), Q is projected from question hidden
    agg_salience = np.zeros(n_vis, dtype=np.float64)
    h_q = question_embeds
    with torch.no_grad():
        for i in range(N):
            # Query-conditioned salience: question_Q @ visual_K (K from cache)
            imp = _importance_from_hidden_and_cache(lm, i, h_q, shared_cache, vis_pos)
            agg_salience += imp

            h_q = lm.layers[i](
                hidden_states=h_q, attention_mask=q_mask,
                past_key_values=q_cache, use_cache=True,
                **q_pos_kwargs)

    # ── Step 2: First token logits ──
    h_final = lm.norm(h_q)
    logits = lm_head(h_final[:, -1:, :])
    next_id = logits.argmax(dim=-1).squeeze().item()
    generated = [next_id]

    if next_id == eos_id or max_new_tokens <= 1:
        return generated

    # ── Step 3: Prune KV cache based on query-conditioned salience ──
    vis_pos_set = set(vis_pos)
    text_pos_all = [p for p in range(n_shared + n_question) if p not in vis_pos_set]
    top_local = sorted(np.argsort(agg_salience)[::-1][:n_decode_keep].tolist())
    keep_vis = [vis_pos[j] for j in top_local]
    keep_positions = sorted(set(keep_vis) | set(text_pos_all))
    keep_t = torch.tensor(keep_positions, device=device, dtype=torch.long)

    pruned_cache = DynamicCache()
    for i in range(N):
        pruned_cache.update(
            q_cache.layers[i].keys[:, :, keep_t, :],
            q_cache.layers[i].values[:, :, keep_t, :],
            layer_idx=i)
    del q_cache

    # ── Step 4: Autoregressive generation with pruned cache ──
    if question_pos_ids.dim() == 3:
        last_pos = question_pos_ids[:, :, -1:] + 1
    else:
        last_pos = question_pos_ids[:, -1:] + 1

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if next_id == eos_id:
                break
            tok_emb = lm.embed_tokens(
                torch.tensor([[next_id]], device=device)).to(dtype)
            gen_pos_kwargs = _compute_pos_embeddings(lm, last_pos, tok_emb, device, dtype)

            for i in range(N):
                tok_emb = lm.layers[i](
                    hidden_states=tok_emb, attention_mask=None,
                    past_key_values=pruned_cache, use_cache=True,
                    **gen_pos_kwargs)

            h_tok = lm.norm(tok_emb)
            logits = lm_head(h_tok[:, -1:, :])
            next_id = logits.argmax(dim=-1).squeeze().item()
            generated.append(next_id)
            last_pos = last_pos + 1

    return generated


# ══════════════════════════════════════════════════════════════
# Per-model runners
# ══════════════════════════════════════════════════════════════

def run_qwen(model_name, images_with_qa, cfg, out_dir):
    """Benchmark for Qwen2.5-VL / Qwen3-VL."""
    from models.qwen3vl_wrapper import Qwen2VLWrapper
    from models.token_manipulator_v2 import TokenManipulatorV2
    from baselines.evaluation.run_broad_comparison import run_pruned

    wrapper = Qwen2VLWrapper(cfg["model_id"], "cuda")
    model = wrapper.model
    lm = model.model.language_model
    merge = model.model.visual.spatial_merge_size

    _VS = 151652; _VE = 151653
    S, Y = cfg["S"], cfg["Y"]

    results = {"baseline": [], "ours": [], "pact": [], "sparsevila": [], "attn_topk": []}

    for idx, (img, question, answers) in enumerate(images_with_qa):
        # ── Baseline ──
        bl_pred = wrapper.generate(img, question, max_new_tokens=32)
        bl_score = score_textvqa(bl_pred, answers)
        results["baseline"].append(bl_score)

        # ── Build full embeddings ──
        inputs = wrapper.prepare_inputs(img, question)
        vis_embeds, grid_thw = wrapper.extract_visual_embeddings(inputs)
        n_vis = vis_embeds.shape[0]

        tok = TokenManipulatorV2()
        keep_all = [True] * n_vis
        mr_full = tok.apply_token_mask(inputs, vis_embeds, grid_thw,
                                       keep_mask=keep_all,
                                       spatial_merge_size=merge,
                                       config=model.model.config)
        device = vis_embeds.device
        dtype = lm.layers[0].self_attn.q_proj.weight.dtype
        new_ids = mr_full["new_input_ids"].to(device)
        text_emb = lm.embed_tokens(new_ids).to(dtype)
        new_vis = mr_full["new_visual_embeds"].to(device).to(dtype)
        if new_vis.shape[0] > 0:
            img_mask, _ = model.model.get_placeholder_mask(
                new_ids, inputs_embeds=text_emb, image_features=new_vis)
            full_embeds = text_emb.masked_scatter(img_mask, new_vis)
        else:
            full_embeds = text_emb

        ids = mr_full["new_input_ids"][0]
        vs = (ids == _VS).nonzero(as_tuple=True)[0][0].item() + 1
        ve = (ids == _VE).nonzero(as_tuple=True)[0][0].item()
        ts = ve + 1; sl = ids.shape[0]
        vis_pos = list(range(vs, ve))
        text_pos = list(range(ts, sl))

        k_y = max(1, int(n_vis * Y))

        # ── ViT importance (for Stage 1 dual signal + SparseVILA) ──
        try:
            vit_imp = _qwen_vit_importance(
                model.model.visual,
                inputs.get("pixel_values"), inputs.get("image_grid_thw"), n_vis)
            torch.cuda.empty_cache()
        except Exception:
            vit_imp = None

        # ── Our dual-stage ──
        try:
            final_mask, s1_set, s2_set = dual_stage_select(
                lm, full_embeds, vis_pos, text_pos, n_vis, S,
                vit_importance=vit_imp, vis_embeds=vis_embeds,
                scoring=cfg.get("scoring", "single"))
            pred = run_pruned(wrapper, inputs, vis_embeds, grid_thw, final_mask, 32)
            results["ours"].append(score_textvqa(pred, answers))
            if idx == 0:
                print(f"    [debug] n_vis={n_vis} stage1={len(s1_set)} stage2={sum(final_mask)} k_y={k_y}", flush=True)
        except Exception as e:
            print(f"    [ours ERR] {e}", flush=True)
            results["ours"].append(0.0)

        # ── PACT ──
        try:
            pact_idx = _pact_select(lm, full_embeds, vis_pos, k_y)
            pact_mask = [i in set(pact_idx) for i in range(n_vis)]
            pred = run_pruned(wrapper, inputs, vis_embeds, grid_thw, pact_mask, 32)
            results["pact"].append(score_textvqa(pred, answers))
        except Exception as e:
            print(f"    [pact ERR] {e}", flush=True)
            results["pact"].append(0.0)

        # ── SparseVILA ──
        try:
            if vit_imp is None:
                raise ValueError("No ViT importance available")
            sv_idx = np.argsort(vit_imp)[::-1][:k_y]
            sv_mask = [i in set(sv_idx.tolist()) for i in range(n_vis)]
            pred = run_pruned(wrapper, inputs, vis_embeds, grid_thw, sv_mask, 32)
            results["sparsevila"].append(score_textvqa(pred, answers))
        except Exception as e:
            print(f"    [sv ERR] {e}", flush=True)
            results["sparsevila"].append(0.0)

        # ── Attention top-k at L0 (single-shot baseline) ──
        try:
            imp0 = _get_importance_at_layer(lm, 0, full_embeds, vis_pos, text_pos)
            topk_idx = np.argsort(imp0)[::-1][:k_y]
            attn_mask = [i in set(topk_idx.tolist()) for i in range(n_vis)]
            pred = run_pruned(wrapper, inputs, vis_embeds, grid_thw, attn_mask, 32)
            results["attn_topk"].append(score_textvqa(pred, answers))
        except Exception as e:
            print(f"    [attn ERR] {e}", flush=True)
            results["attn_topk"].append(0.0)

        del full_embeds
        torch.cuda.empty_cache()

        print(f"  [{idx+1}/{len(images_with_qa)}] bl={bl_score:.2f} "
              f"ours={np.mean(results['ours']):.3f} "
              f"pact={np.mean(results['pact']):.3f} "
              f"sv={np.mean(results['sparsevila']):.3f} "
              f"attn={np.mean(results['attn_topk']):.3f}", flush=True)

    del wrapper, model
    gc.collect(); torch.cuda.empty_cache()
    return results


def run_llava(model_name, images_with_qa, cfg, out_dir):
    """Benchmark for LLaVA-1.5-7B / 13B."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_id = cfg["model_id"]
    print(f"[LLaVA] Loading {model_id}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    lm = model.model.language_model
    IMAGE_TOKEN_ID = 32000
    S, Y = cfg["S"], cfg["Y"]
    N = int(cfg.get("N", len(lm.layers)))
    N = cfg["N"]

    results = {"baseline": [], "ours": [], "pact": [], "sparsevila": [], "attn_topk": []}

    def _generate_with_mask(full_embeds, input_ids, vis_pos_set, kept_vis_pos_set):
        """Generate with pruned embeddings."""
        all_pos = []
        for p in range(input_ids.shape[1]):
            if p in vis_pos_set:
                if p in kept_vis_pos_set:
                    all_pos.append(p)
            else:
                all_pos.append(p)
        pruned_embeds = full_embeds[:, all_pos, :]
        pruned_attn = torch.ones(1, len(all_pos), device="cuda", dtype=torch.long)
        with torch.no_grad():
            out_ids = model.generate(
                inputs_embeds=pruned_embeds,
                attention_mask=pruned_attn,
                max_new_tokens=32, do_sample=False)
        return processor.decode(out_ids[0], skip_special_tokens=True).strip()

    for idx, (img, question, answers) in enumerate(images_with_qa):
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = processor(text=prompt, images=img, return_tensors="pt").to("cuda")

        # ── Baseline ──
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        bl_pred = processor.decode(out_ids[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
        bl_score = score_textvqa(bl_pred, answers)
        results["baseline"].append(bl_score)

        # ── Build full embeddings ──
        with torch.no_grad():
            pixel_values = inputs["pixel_values"].to(model.dtype)
            image_features = model.model.multi_modal_projector(
                model.model.vision_tower(pixel_values).last_hidden_state)
            input_ids = inputs["input_ids"]
            text_emb = lm.embed_tokens(input_ids)
            img_positions = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
            n_vis = len(img_positions)
            full_embeds = text_emb.clone()
            full_embeds[0, img_positions[:n_vis]] = image_features[0, :n_vis].to(text_emb.dtype)

        vis_pos = img_positions.tolist()
        vis_pos_set = set(vis_pos)
        text_pos = [p for p in range(input_ids.shape[1])
                    if p not in vis_pos_set and p > max(vis_pos)]

        if n_vis < 10 or len(text_pos) < 2:
            for k in ["ours", "pact", "sparsevila", "attn_topk"]:
                results[k].append(0.0)
            continue

        k_y = max(1, int(n_vis * Y))

        # ── ViT importance (for Stage 1 + SparseVILA) ──
        try:
            vit_imp = _llava_vit_importance(model, pixel_values, n_vis)
        except Exception:
            vit_imp = None

        # ── Our dual-stage ──
        try:
            vis_embeds_mmr = image_features[0, :n_vis]
            final_mask, _, _ = dual_stage_select(
                lm, full_embeds, vis_pos, text_pos, n_vis, S,
                vit_importance=vit_imp, vis_embeds=vis_embeds_mmr,
                scoring=cfg.get("scoring", "single"))
            kept = {vis_pos[i] for i, m in enumerate(final_mask) if m}
            pred = _generate_with_mask(full_embeds, input_ids, vis_pos_set, kept)
            results["ours"].append(score_textvqa(pred, answers))
        except Exception as e:
            print(f"    [ours ERR] {e}", flush=True)
            results["ours"].append(0.0)

        # ── PACT ──
        try:
            pact_idx = _pact_select(lm, full_embeds, vis_pos, k_y)
            kept = {vis_pos[i] for i in pact_idx}
            pred = _generate_with_mask(full_embeds, input_ids, vis_pos_set, kept)
            results["pact"].append(score_textvqa(pred, answers))
        except Exception as e:
            print(f"    [pact ERR] {e}", flush=True)
            results["pact"].append(0.0)

        # ── SparseVILA ──
        try:
            if vit_imp is None:
                raise ValueError("No ViT importance available")
            sv_idx = np.argsort(vit_imp)[::-1][:k_y]
            kept = {vis_pos[i] for i in sv_idx}
            pred = _generate_with_mask(full_embeds, input_ids, vis_pos_set, kept)
            results["sparsevila"].append(score_textvqa(pred, answers))
        except Exception as e:
            print(f"    [sv ERR] {e}", flush=True)
            results["sparsevila"].append(0.0)

        # ── Attention top-k at L0 ──
        try:
            imp0 = _get_importance_at_layer(lm, 0, full_embeds, vis_pos, text_pos)
            topk_idx = np.argsort(imp0)[::-1][:k_y]
            kept = {vis_pos[i] for i in topk_idx}
            pred = _generate_with_mask(full_embeds, input_ids, vis_pos_set, kept)
            results["attn_topk"].append(score_textvqa(pred, answers))
        except Exception as e:
            print(f"    [attn ERR] {e}", flush=True)
            results["attn_topk"].append(0.0)

        del full_embeds, inputs
        torch.cuda.empty_cache()

        print(f"  [{idx+1}/{len(images_with_qa)}] bl={bl_score:.2f} "
              f"ours={np.mean(results['ours']):.3f} "
              f"pact={np.mean(results['pact']):.3f} "
              f"sv={np.mean(results['sparsevila']):.3f} "
              f"attn={np.mean(results['attn_topk']):.3f}", flush=True)

    del model, processor
    gc.collect(); torch.cuda.empty_cache()
    return results


def run_internvl3(model_name, images_with_qa, cfg, out_dir):
    """Benchmark for InternVL3-8B-hf / InternVL3.5-8B-hf."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = cfg["model_id"]
    print(f"[InternVL3-hf] Loading {model_id}...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    proc = AutoProcessor.from_pretrained(model_id)

    lm = model.model.language_model
    img_token_id = getattr(model.config, 'image_token_id', None)
    if img_token_id is None:
        img_token_id = proc.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    S, Y = cfg["S"], cfg["Y"]
    eos_id = proc.tokenizer.eos_token_id

    results = {"baseline": [], "ours": [], "pact": [], "sparsevila": [], "attn_topk": []}

    def _greedy_decode(pruned_embeds, max_new_tokens=32):
        generated = []
        cur_embeds = pruned_embeds
        past_kv = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = lm(inputs_embeds=cur_embeds, past_key_values=past_kv, use_cache=True)
                logits = model.lm_head(out.last_hidden_state[:, -1:, :])
                next_id = logits.argmax(dim=-1).squeeze()
                generated.append(next_id.item())
                if next_id.item() == eos_id:
                    break
                past_kv = out.past_key_values
                cur_embeds = lm.embed_tokens(
                    next_id.unsqueeze(0).unsqueeze(0)).to(pruned_embeds.dtype)
        return proc.decode(generated, skip_special_tokens=True).strip()

    def _generate_with_vis_mask(full_embeds, input_ids, vis_pos, vis_pos_set, kept_idx_set):
        """Generate with selected visual tokens."""
        kept_vis_pos = {vis_pos[i] for i in kept_idx_set}
        all_pos = []
        for p in range(input_ids.shape[1]):
            if p in vis_pos_set:
                if p in kept_vis_pos:
                    all_pos.append(p)
            else:
                all_pos.append(p)
        return _greedy_decode(full_embeds[:, all_pos, :])

    for idx, (img, question, answers) in enumerate(images_with_qa):
        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question}]}]
            text_prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
            proc_inputs = proc(text=[text_prompt], images=[img],
                               return_tensors="pt").to(model.device)

            # ── Baseline ──
            with torch.no_grad():
                out_ids = model.generate(**proc_inputs, max_new_tokens=32, do_sample=False)
            bl_pred = proc.decode(out_ids[0][proc_inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
            bl_score = score_textvqa(bl_pred, answers)

            # ── Build full embeddings ──
            input_ids = proc_inputs["input_ids"]
            pixel_values = proc_inputs.get("pixel_values")
            with torch.no_grad():
                if pixel_values is not None:
                    vis_out = model.model.get_image_features(pixel_values.to(model.dtype))
                    vis_features = vis_out.pooler_output

                dtype = lm.layers[0].self_attn.q_proj.weight.dtype
                text_emb = lm.embed_tokens(input_ids).to(dtype)
                full_embeds = text_emb.clone()

                img_positions = (input_ids[0] == img_token_id).nonzero(as_tuple=True)[0]
                n_vis = len(img_positions)
                if n_vis > 0 and pixel_values is not None:
                    vis_flat = vis_features.reshape(-1, vis_features.shape[-1]).to(dtype)
                    n_insert = min(n_vis, vis_flat.shape[0])
                    full_embeds[0, img_positions[:n_insert]] = vis_flat[:n_insert]

            vis_pos = img_positions.tolist()
            vis_pos_set = set(vis_pos)
            text_pos = [p for p in range(input_ids.shape[1])
                        if p not in vis_pos_set and p > max(vis_pos)] if vis_pos else []
        except Exception as e:
            print(f"  [{idx+1}] BL ERR: {e}", flush=True)
            results["baseline"].append(0.0)
            for k in ["ours", "pact", "sparsevila", "attn_topk"]:
                results[k].append(0.0)
            continue

        results["baseline"].append(bl_score)

        if n_vis < 10 or len(text_pos) < 2:
            for k in ["ours", "pact", "sparsevila", "attn_topk"]:
                results[k].append(0.0)
            print(f"  [{idx+1}/{len(images_with_qa)}] bl={bl_score:.2f} SKIP (n_vis={n_vis})",
                  flush=True)
            continue

        k_y = max(1, int(n_vis * Y))

        # ── ViT importance (for Stage 1 + SparseVILA) ──
        try:
            vit_imp = _internvl3_vit_importance(model, pixel_values, n_vis)
            torch.cuda.empty_cache()
        except Exception:
            vit_imp = None

        # ── Our dual-stage ──
        try:
            vis_embeds_mmr = vis_flat[:n_vis] if n_vis <= vis_flat.shape[0] else vis_flat
            final_mask, _, _ = dual_stage_select(
                lm, full_embeds, vis_pos, text_pos, n_vis, S,
                vit_importance=vit_imp, vis_embeds=vis_embeds_mmr,
                scoring=cfg.get("scoring", "single"))
            kept = {i for i, m in enumerate(final_mask) if m}
            pred = _generate_with_vis_mask(full_embeds, input_ids, vis_pos, vis_pos_set, kept)
            results["ours"].append(score_textvqa(pred, answers))
        except Exception as e:
            print(f"    [ours ERR] {e}", flush=True)
            results["ours"].append(0.0)

        # ── PACT ──
        try:
            pact_idx = _pact_select(lm, full_embeds, vis_pos, k_y)
            pred = _generate_with_vis_mask(full_embeds, input_ids, vis_pos, vis_pos_set, set(pact_idx))
            results["pact"].append(score_textvqa(pred, answers))
        except Exception as e:
            print(f"    [pact ERR] {e}", flush=True)
            results["pact"].append(0.0)

        # ── SparseVILA ──
        try:
            if vit_imp is None:
                raise ValueError("No ViT importance available")
            sv_idx = np.argsort(vit_imp)[::-1][:k_y]
            pred = _generate_with_vis_mask(
                full_embeds, input_ids, vis_pos, vis_pos_set, set(sv_idx.tolist()))
            results["sparsevila"].append(score_textvqa(pred, answers))
        except Exception as e:
            print(f"    [sv ERR] {e}", flush=True)
            results["sparsevila"].append(0.0)

        # ── Attention top-k at L0 ──
        try:
            imp0 = _get_importance_at_layer(lm, 0, full_embeds, vis_pos, text_pos)
            topk_idx = np.argsort(imp0)[::-1][:k_y]
            pred = _generate_with_vis_mask(
                full_embeds, input_ids, vis_pos, vis_pos_set, set(topk_idx.tolist()))
            results["attn_topk"].append(score_textvqa(pred, answers))
        except Exception as e:
            print(f"    [attn ERR] {e}", flush=True)
            results["attn_topk"].append(0.0)

        del full_embeds
        torch.cuda.empty_cache()

        print(f"  [{idx+1}/{len(images_with_qa)}] bl={bl_score:.2f} "
              f"ours={np.mean(results['ours']):.3f} "
              f"pact={np.mean(results['pact']):.3f} "
              f"sv={np.mean(results['sparsevila']):.3f} "
              f"attn={np.mean(results['attn_topk']):.3f}", flush=True)

    del model, proc
    gc.collect(); torch.cuda.empty_cache()
    return results


# ══════════════════════════════════════════════════════════════
# Multi-round token diversity analysis
# ══════════════════════════════════════════════════════════════

MULTI_ROUND_QUESTIONS = [
    "What text can you see in this image?",
    "Describe the main objects in this image.",
    "What colors are prominent in this image?",
]

def run_multi_round_analysis(model_name, images, cfg):
    """For each image, show Stage 2 selects DIFFERENT tokens for different questions.

    Returns per-image IoU between Stage 2 token sets across question pairs.
    """
    print(f"\n{'='*60}")
    print(f"Multi-round token diversity analysis — {model_name}")
    print(f"{'='*60}", flush=True)

    # Only Qwen models have full pipeline support for easy multi-question
    if cfg["type"] != "qwen":
        print("  (multi-round analysis only for Qwen models currently)", flush=True)
        return None

    from models.qwen3vl_wrapper import Qwen2VLWrapper
    from models.token_manipulator_v2 import TokenManipulatorV2

    wrapper = Qwen2VLWrapper(cfg["model_id"], "cuda")
    model = wrapper.model
    lm = model.model.language_model
    merge = model.model.visual.spatial_merge_size
    _VS = 151652; _VE = 151653
    S = cfg["S"]

    diversity_results = []

    for img_idx, img in enumerate(images):
        # Shared Stage 1 (query-agnostic)
        inputs0 = wrapper.prepare_inputs(img, MULTI_ROUND_QUESTIONS[0])
        vis_embeds, grid_thw = wrapper.extract_visual_embeddings(inputs0)
        n_vis = vis_embeds.shape[0]
        k1 = max(1, int(n_vis * STAGE1_FRAC))
        k2 = max(1, int(n_vis * STAGE2_FRAC))

        tok = TokenManipulatorV2()

        # Build full embeds with first question (for Stage 1 L0 importance)
        keep_all = [True] * n_vis
        mr_full = tok.apply_token_mask(inputs0, vis_embeds, grid_thw,
                                       keep_mask=keep_all,
                                       spatial_merge_size=merge,
                                       config=model.model.config)
        device = vis_embeds.device
        dtype = lm.layers[0].self_attn.q_proj.weight.dtype
        new_ids = mr_full["new_input_ids"].to(device)
        text_emb = lm.embed_tokens(new_ids).to(dtype)
        new_vis = mr_full["new_visual_embeds"].to(device).to(dtype)
        if new_vis.shape[0] > 0:
            img_mask, _ = model.model.get_placeholder_mask(
                new_ids, inputs_embeds=text_emb, image_features=new_vis)
            full_embeds0 = text_emb.masked_scatter(img_mask, new_vis)
        else:
            full_embeds0 = text_emb
        ids0 = mr_full["new_input_ids"][0]
        vs = (ids0 == _VS).nonzero(as_tuple=True)[0][0].item() + 1
        ve = (ids0 == _VE).nonzero(as_tuple=True)[0][0].item()
        vis_pos0 = list(range(vs, ve))
        text_pos0 = list(range(ve + 1, ids0.shape[0]))

        # Stage 1 (shared, query-agnostic)
        imp0 = _get_importance_at_layer(lm, 0, full_embeds0, vis_pos0, text_pos0)
        stage1_idx = set(np.argsort(imp0)[::-1][:k1].tolist())
        surviving = sorted(stage1_idx)

        # Stage 2 per question (query-dependent)
        stage2_sets = []
        for q in MULTI_ROUND_QUESTIONS:
            inputs_q = wrapper.prepare_inputs(img, q)
            mr_q = tok.apply_token_mask(inputs_q, vis_embeds, grid_thw,
                                        keep_mask=keep_all,
                                        spatial_merge_size=merge,
                                        config=model.model.config)
            new_ids_q = mr_q["new_input_ids"].to(device)
            text_emb_q = lm.embed_tokens(new_ids_q).to(dtype)
            new_vis_q = mr_q["new_visual_embeds"].to(device).to(dtype)
            if new_vis_q.shape[0] > 0:
                img_mask_q, _ = model.model.get_placeholder_mask(
                    new_ids_q, inputs_embeds=text_emb_q, image_features=new_vis_q)
                full_embeds_q = text_emb_q.masked_scatter(img_mask_q, new_vis_q)
            else:
                full_embeds_q = text_emb_q

            ids_q = mr_q["new_input_ids"][0]
            vs_q = (ids_q == _VS).nonzero(as_tuple=True)[0][0].item() + 1
            ve_q = (ids_q == _VE).nonzero(as_tuple=True)[0][0].item()
            surviving_vis_pos = [list(range(vs_q, ve_q))[i] for i in range(len(surviving))
                                 if i < ve_q - vs_q]
            text_pos_q = list(range(ve_q + 1, ids_q.shape[0]))

            if len(surviving_vis_pos) > 0 and len(text_pos_q) > 0:
                _scoring = cfg.get("scoring", "single")
                if _scoring == "max":
                    imp_layers = [_get_importance_at_layer(
                        lm, L, full_embeds_q, surviving_vis_pos, text_pos_q)
                        for L in range(S)]
                    impS = np.maximum.reduce(imp_layers)
                else:
                    impS = _get_importance_at_layer(
                        lm, S, full_embeds_q, surviving_vis_pos, text_pos_q)
                s2_local = np.argsort(impS)[::-1][:k2]
                s2_set = set(surviving[j] for j in s2_local if j < len(surviving))
            else:
                s2_set = set()
            stage2_sets.append(s2_set)

            del full_embeds_q

        # Compute pairwise IoU
        ious = []
        for i in range(len(stage2_sets)):
            for j in range(i + 1, len(stage2_sets)):
                a, b = stage2_sets[i], stage2_sets[j]
                if len(a | b) > 0:
                    ious.append(len(a & b) / len(a | b))
                else:
                    ious.append(1.0)
        avg_iou = np.mean(ious) if ious else 1.0

        diversity_results.append({
            "n_vis": n_vis,
            "k1": k1, "k2": k2,
            "avg_iou": float(avg_iou),
            "set_sizes": [len(s) for s in stage2_sets],
        })
        print(f"  Image {img_idx+1}: n_vis={n_vis} k2={k2} avg_IoU={avg_iou:.3f} "
              f"sizes={[len(s) for s in stage2_sets]}", flush=True)

        del full_embeds0
        torch.cuda.empty_cache()

    del wrapper, model
    gc.collect(); torch.cuda.empty_cache()
    return diversity_results


# ══════════════════════════════════════════════════════════════
# Multi-round GQA evaluation  (一张图 × 多问题)
# ══════════════════════════════════════════════════════════════

def _dual_stage_select_query_dependent(lm, full_embeds, vis_pos, text_pos,
                                        n_vis, S, stage1_set, scoring="single"):
    """Stage 2 only: given shared Stage 1, do query-dependent pruning at S.

    Args:
        stage1_set: set of original vis token indices from Stage 1 (shared)
        scoring:    "single" = Q@K at layer S; "max" = max Q@K over layers 0..S-1
    Returns:
        final_mask: list[bool] of length n_vis
    """
    k2 = max(1, int(n_vis * STAGE2_FRAC))
    surviving_vis_pos = [vis_pos[i] for i in sorted(stage1_set)]

    if scoring == "max":
        imp_layers = [_get_importance_at_layer(lm, L, full_embeds, surviving_vis_pos, text_pos)
                      for L in range(S)]
        impS = np.maximum.reduce(imp_layers)
    else:
        impS = _get_importance_at_layer(lm, S, full_embeds, surviving_vis_pos, text_pos)

    stage2_local = np.argsort(impS)[::-1][:k2]
    surviving_orig = sorted(stage1_set)
    stage2_set = set(surviving_orig[j] for j in stage2_local)
    return [i in stage2_set for i in range(n_vis)]


def run_gqa_qwen(model_name, gqa_samples, cfg, out_dir):
    """Multi-round GQA benchmark for Qwen models.

    For each image with K questions:
    - Baseline: full tokens, each question independently
    - Ours: Stage 1 shared (query-agnostic 40%) + Stage 2 per-question (10%)
    - PACT/SparseVILA/Attn: KV cache reuse (prefill once, per-question attend)
    """
    from models.qwen3vl_wrapper import Qwen2VLWrapper
    from models.token_manipulator_v2 import TokenManipulatorV2

    wrapper = Qwen2VLWrapper(cfg["model_id"], "cuda")
    model = wrapper.model
    lm = model.model.language_model
    merge = model.model.visual.spatial_merge_size

    _VS = 151652; _VE = 151653
    S, Y = cfg["S"], cfg["Y"]

    ALL_METHODS = ["baseline", "ours", "pact", "sparsevila", "attn_topk",
                   "svdprune", "divprune", "vispruner", "fastv",
                   "zspaprune", "agilepruner", "idselection", "d2pruner", "ptp",
                   "hawk", "vscore_l2",
                   "fitprune", "pyramiddrop", "sparsevlm"]
    results = {m: [] for m in ALL_METHODS}
    policy_methods = cfg.get("policy_methods") or []
    for method in policy_methods:
        results.setdefault(method["output_name"], [])
    N = cfg["N"]
    # Cost tracking: per-question token×layer (prefill + generation)
    costs = {m: _new_cost_bucket() for m in ALL_METHODS}
    for method in policy_methods:
        costs.setdefault(method["output_name"], _new_cost_bucket())

    for img_idx, (img, qa_pairs) in enumerate(gqa_samples):
        img = _dualsignal_resize_image(img)
        image_shared_start = time.perf_counter()
        # ── Extract visual features once (shared across questions) ──
        inputs0 = wrapper.prepare_inputs(img, qa_pairs[0][0] + GQA_SUFFIX)
        vis_embeds, grid_thw = wrapper.extract_visual_embeddings(inputs0)
        n_vis = vis_embeds.shape[0]
        k_y = max(1, int(n_vis * Y))

        tok = TokenManipulatorV2()
        device = vis_embeds.device
        dtype = lm.layers[0].self_attn.q_proj.weight.dtype

        # ── Compute SHARED Stage 1 (query-agnostic, L0 attention) ──
        # Use first question to build full embeddings for Stage 1
        keep_all = [True] * n_vis
        mr_full = tok.apply_token_mask(inputs0, vis_embeds, grid_thw,
                                       keep_mask=keep_all,
                                       spatial_merge_size=merge,
                                       config=model.model.config)
        new_ids = mr_full["new_input_ids"].to(device)
        text_emb = lm.embed_tokens(new_ids).to(dtype)
        new_vis = mr_full["new_visual_embeds"].to(device).to(dtype)
        if new_vis.shape[0] > 0:
            img_mask, _ = model.model.get_placeholder_mask(
                new_ids, inputs_embeds=text_emb, image_features=new_vis)
            full_embeds0 = text_emb.masked_scatter(img_mask, new_vis)
        else:
            full_embeds0 = text_emb

        ids0 = mr_full["new_input_ids"][0]
        vs = (ids0 == _VS).nonzero(as_tuple=True)[0][0].item() + 1
        ve = (ids0 == _VE).nonzero(as_tuple=True)[0][0].item()
        vis_pos0 = list(range(vs, ve))
        text_pos0 = list(range(ve + 1, ids0.shape[0]))

        k1 = max(1, int(n_vis * STAGE1_FRAC))
        imp0 = _get_importance_at_layer(lm, 0, full_embeds0, vis_pos0, text_pos0)

        # ── Compute ViT importance (used by both our Stage 1 and SparseVILA) ──
        try:
            vit_imp = _qwen_vit_importance(
                model.model.visual,
                inputs0.get("pixel_values"), inputs0.get("image_grid_thw"), n_vis)
            torch.cuda.empty_cache()
        except Exception:
            vit_imp = None

        # ── Our Stage 1: dual signal + MMR diversity ──
        if vit_imp is not None:
            stage1_idx = _dual_signal_stage1_select(imp0, vit_imp, vis_embeds, k1)
        else:
            stage1_idx = np.argsort(imp0)[::-1][:k1].tolist()
        stage1_set = set(stage1_idx)

        # ── Compute SHARED baseline token sets (query-agnostic, fixed) ──
        pact_idx = _pact_select(lm, full_embeds0, vis_pos0, k_y)
        pact_mask = [i in set(pact_idx) for i in range(n_vis)]

        if vit_imp is not None:
            sv_idx = np.argsort(vit_imp)[::-1][:k_y]
            sv_mask = [i in set(sv_idx.tolist()) for i in range(n_vis)]
        else:
            sv_mask = None

        attn_idx = np.argsort(imp0)[::-1][:k_y]
        attn_mask = [i in set(attn_idx.tolist()) for i in range(n_vis)]

        # ── NEW: SVDPrune mask (query-agnostic) ──
        try:
            svd_mask = _svdprune_select(vis_embeds, k_y)
        except Exception:
            svd_mask = None

        # ── NEW: DivPrune mask (query-agnostic) ──
        try:
            div_mask = _divprune_select(vis_embeds, k_y)
        except Exception:
            div_mask = None

        # ── NEW: VisPruner mask (query-agnostic, ViT + diversity) ──
        if vit_imp is not None:
            try:
                visp_mask = _vispruner_select(vit_imp, vis_embeds, k_y)
            except Exception:
                visp_mask = None
        else:
            visp_mask = None

        # ── NEW: FastV mask (L2 last-text attention, computed once) ──
        try:
            fastv_mask = _fastv_select(lm, full_embeds0, vis_pos0, text_pos0, k_y, layer=2)
        except Exception:
            fastv_mask = None

        # ── B-class: ZSPAPrune (prompt-aware core + diversity) ──
        try:
            zspap_mask = _zspaprune_select(lm, full_embeds0, vis_pos0, text_pos0,
                                           vis_embeds, k_y)
        except Exception:
            zspap_mask = None

        # ── B-class: AgilePruner (L1 attention + entropy-adaptive dedup) ──
        try:
            agile_mask = _agilepruner_select(lm, full_embeds0, vis_pos0, text_pos0,
                                              vis_embeds, k_y)
        except Exception:
            agile_mask = None

        # ── B-class: IDSelection (L2 cross-modal sim + Gaussian suppression) ──
        try:
            idsel_mask = _idselection_select(lm, full_embeds0, vis_pos0, text_pos0,
                                              vis_embeds, k_y)
        except Exception:
            idsel_mask = None

        # ── B-class: D²Pruner (L2 debiased attention + MIS) ──
        try:
            d2p_mask = _d2pruner_select(lm, full_embeds0, vis_pos0, text_pos0,
                                         vis_embeds, k_y, grid_thw=grid_thw)
        except Exception:
            d2p_mask = None

        # ── B-class: PTP (ViT saliency + L2 instruction refinement) ──
        try:
            ptp_mask = _ptp_select(vit_imp, lm, full_embeds0, vis_pos0, text_pos0,
                                    vis_embeds, k_y, grid_thw=grid_thw)
        except Exception:
            ptp_mask = None

        # ── B-class: HAWK (L0 no-RoPE text→vis attention) ──
        try:
            hawk_mask = _hawk_select(lm, full_embeds0, vis_pos0, text_pos0, k_y)
        except Exception:
            hawk_mask = None

        # ── B-class: VScoreL2 (L2 attn × value norm) ──
        try:
            vsl2_mask = _vscore_l2_select(lm, full_embeds0, vis_pos0, text_pos0, k_y)
        except Exception:
            vsl2_mask = None

        del full_embeds0

        # ── C-class progressive prefill (selection + KV cache in one pass) ──
        _skip_cc = cfg.get("skip_cclass", False)
        _prog_pfx_ok = False
        if not _skip_cc:
            try:
                _full_pfx_emb, _full_pfx_pos, _full_vis_in_pfx, _, _ = \
                    _build_prefix([True] * n_vis)
                _full_vis_set = set(_full_vis_in_pfx)
                _n_full_pfx = _full_pfx_emb.shape[1]
                _prog_pfx_ok = True
            except Exception:
                pass

        # ── C-class: FitPrune (progressive per-layer self×cross) ──
        fitprune_ok = False
        if _prog_pfx_ok:
            try:
                _fp_schedule = _fitprune_build_schedule(n_vis, k_y, N)
                _fp_drop_fn = _make_fitprune_drop_fn(_fp_schedule)
                fitprune_cache, _, fitprune_alive, fitprune_tl = \
                    _progressive_prefill(lm, _full_pfx_emb, set(_full_vis_set),
                                         _fp_drop_fn, _full_pfx_pos)
                fitprune_ok = True
            except Exception:
                pass

        # ── C-class: PyramidDrop (multi-stage last-text→vis) ──
        pyrdrop_ok = False
        if _prog_pfx_ok:
            try:
                _pd_drop_fn = _make_pyramiddrop_drop_fn(n_vis, k_y, N)
                pyrdrop_cache, _, pyrdrop_alive, pyrdrop_tl = \
                    _progressive_prefill(lm, _full_pfx_emb, set(_full_vis_set),
                                         _pd_drop_fn, _full_pfx_pos)
                pyrdrop_ok = True
            except Exception:
                pass

        # ── C-class: SparseVLM (text-rater guided progressive) ──
        spvlm_ok = False
        if _prog_pfx_ok:
            try:
                _sv_drop_fn = _make_sparsevlm_drop_fn(n_vis, k_y, N)
                spvlm_cache, _, spvlm_alive, spvlm_tl = \
                    _progressive_prefill(lm, _full_pfx_emb, set(_full_vis_set),
                                         _sv_drop_fn, _full_pfx_pos)
                spvlm_ok = True
            except Exception:
                pass

        if _prog_pfx_ok:
            del _full_pfx_emb

        # ── Helper: build shared prefix embeddings from a keep_mask ──
        def _build_prefix(mask):
            mr = tok.apply_token_mask(inputs0, vis_embeds, grid_thw,
                                      keep_mask=mask,
                                      spatial_merge_size=merge,
                                      config=model.model.config)
            ids_t = mr["new_input_ids"].to(device)
            t_emb = lm.embed_tokens(ids_t).to(dtype)
            v_emb = mr["new_visual_embeds"].to(device).to(dtype)
            if v_emb.shape[0] > 0:
                m, _ = model.model.get_placeholder_mask(
                    ids_t, inputs_embeds=t_emb, image_features=v_emb)
                emb = t_emb.masked_scatter(m, v_emb)
            else:
                emb = t_emb
            id_seq = mr["new_input_ids"][0]
            v_s = (id_seq == _VS).nonzero(as_tuple=True)[0][0].item() + 1
            v_e = (id_seq == _VE).nonzero(as_tuple=True)[0][0].item()
            n_pfx = v_e + 1  # inclusive of vision_end
            prefix_emb = emb[:, :n_pfx, :]
            prefix_pos = mr["position_ids"][:, :, :n_pfx].to(device)
            vis_in_pfx = list(range(v_s, v_e))
            text_in_pfx = list(range(v_e, n_pfx))  # vision_end token
            return prefix_emb, prefix_pos, vis_in_pfx, text_in_pfx, mr

        # ── Prefill shared prefix through layers 0..S-1 for OUR method (ONCE) ──
        stage1_mask = [i in stage1_set for i in range(n_vis)]
        _lm_head = model.lm_head
        _eos_id = wrapper.processor.tokenizer.eos_token_id
        policy_shared = {}
        image_shared_sec = time.perf_counter() - image_shared_start
        try:
            ours_pfx_emb, ours_pfx_pos, vis_pos_in_shared, _, mr_s1_shared = \
                _build_prefix(stage1_mask)
            n_prefix = ours_pfx_emb.shape[1]
            n_vis_stage1 = len(vis_pos_in_shared)
            shared_pos = ours_pfx_pos
            if policy_methods:
                for method in policy_methods:
                    method_S = int(method["S"])
                    prefill_start = time.perf_counter()
                    h_m, cache_m = _prefill_shared_layers(
                        lm, ours_pfx_emb, method_S, ours_pfx_pos)
                    policy_shared[method["output_name"]] = {
                        "h": h_m,
                        "cache": cache_m,
                        "S": method_S,
                        "prefill_wall_sec": time.perf_counter() - prefill_start,
                    }
                shared_ok = bool(policy_shared)
            else:
                prefill_start = time.perf_counter()
                h_shared, shared_cache = _prefill_shared_layers(
                    lm, ours_pfx_emb, S, ours_pfx_pos)
                policy_prefill_wall_sec = time.perf_counter() - prefill_start
                shared_ok = True
            del ours_pfx_emb
        except Exception as e:
            import traceback; traceback.print_exc()
            shared_ok = False

        # ── Prefill ALL N layers for PACT (ONCE per image) ──
        try:
            pact_pfx_emb, pact_pfx_pos, _, _, _ = _build_prefix(pact_mask)
            pact_cache, _ = _prefill_all_layers(lm, pact_pfx_emb, pact_pfx_pos)
            pact_ok = True
            del pact_pfx_emb
        except Exception:
            pact_ok = False

        # ── Prefill ALL N layers for SparseVILA (ONCE per image) ──
        # Prefill is query-independent (ViT saliency); decode pruning is
        # query-conditioned (per-question salience from question_Q @ visual_K_cached).
        sv_ok = False
        if sv_mask is not None:
            try:
                sv_pfx_emb, sv_pfx_pos, sv_vis_in_pfx, _, _ = _build_prefix(sv_mask)
                n_sv_vis = len(sv_vis_in_pfx)
                n_sv_decode_keep = max(1, int(n_sv_vis * SPARSEVILA_DECODE_RATIO))
                sv_cache, _ = _prefill_all_layers(lm, sv_pfx_emb, sv_pfx_pos)
                del sv_pfx_emb
                sv_ok = True
            except Exception:
                import traceback; traceback.print_exc()

        # ── Prefill ALL N layers for Attn top-k (ONCE per image) ──
        try:
            attn_pfx_emb, attn_pfx_pos, _, _, _ = _build_prefix(attn_mask)
            attn_cache, _ = _prefill_all_layers(lm, attn_pfx_emb, attn_pfx_pos)
            attn_ok = True
            del attn_pfx_emb
        except Exception:
            attn_ok = False

        # ── Prefill ALL N layers for SVDPrune (ONCE per image) ──
        svd_ok = False
        if svd_mask is not None:
            try:
                svd_pfx_emb, svd_pfx_pos, _, _, _ = _build_prefix(svd_mask)
                svd_cache, _ = _prefill_all_layers(lm, svd_pfx_emb, svd_pfx_pos)
                svd_ok = True
                del svd_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for DivPrune (ONCE per image) ──
        div_ok = False
        if div_mask is not None:
            try:
                div_pfx_emb, div_pfx_pos, _, _, _ = _build_prefix(div_mask)
                div_cache, _ = _prefill_all_layers(lm, div_pfx_emb, div_pfx_pos)
                div_ok = True
                del div_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for VisPruner (ONCE per image) ──
        visp_ok = False
        if visp_mask is not None:
            try:
                visp_pfx_emb, visp_pfx_pos, _, _, _ = _build_prefix(visp_mask)
                visp_cache, _ = _prefill_all_layers(lm, visp_pfx_emb, visp_pfx_pos)
                visp_ok = True
                del visp_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for FastV (ONCE per image) ──
        fastv_ok = False
        if fastv_mask is not None:
            try:
                fastv_pfx_emb, fastv_pfx_pos, _, _, _ = _build_prefix(fastv_mask)
                fastv_cache, _ = _prefill_all_layers(lm, fastv_pfx_emb, fastv_pfx_pos)
                fastv_ok = True
                del fastv_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for ZSPAPrune (ONCE per image) ──
        zspap_ok = False
        if zspap_mask is not None:
            try:
                zspap_pfx_emb, zspap_pfx_pos, _, _, _ = _build_prefix(zspap_mask)
                zspap_cache, _ = _prefill_all_layers(lm, zspap_pfx_emb, zspap_pfx_pos)
                zspap_ok = True
                del zspap_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for AgilePruner (ONCE per image) ──
        agile_ok = False
        if agile_mask is not None:
            try:
                agile_pfx_emb, agile_pfx_pos, _, _, _ = _build_prefix(agile_mask)
                agile_cache, _ = _prefill_all_layers(lm, agile_pfx_emb, agile_pfx_pos)
                agile_ok = True
                del agile_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for IDSelection (ONCE per image) ──
        idsel_ok = False
        if idsel_mask is not None:
            try:
                idsel_pfx_emb, idsel_pfx_pos, _, _, _ = _build_prefix(idsel_mask)
                idsel_cache, _ = _prefill_all_layers(lm, idsel_pfx_emb, idsel_pfx_pos)
                idsel_ok = True
                del idsel_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for D²Pruner (ONCE per image) ──
        d2p_ok = False
        if d2p_mask is not None:
            try:
                d2p_pfx_emb, d2p_pfx_pos, _, _, _ = _build_prefix(d2p_mask)
                d2p_cache, _ = _prefill_all_layers(lm, d2p_pfx_emb, d2p_pfx_pos)
                d2p_ok = True
                del d2p_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for PTP (ONCE per image) ──
        ptp_ok = False
        if ptp_mask is not None:
            try:
                ptp_pfx_emb, ptp_pfx_pos, _, _, _ = _build_prefix(ptp_mask)
                ptp_cache, _ = _prefill_all_layers(lm, ptp_pfx_emb, ptp_pfx_pos)
                ptp_ok = True
                del ptp_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for HAWK (ONCE per image) ──
        hawk_ok = False
        if hawk_mask is not None:
            try:
                hawk_pfx_emb, hawk_pfx_pos, _, _, _ = _build_prefix(hawk_mask)
                hawk_cache, _ = _prefill_all_layers(lm, hawk_pfx_emb, hawk_pfx_pos)
                hawk_ok = True
                del hawk_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for VScoreL2 (ONCE per image) ──
        vsl2_ok = False
        if vsl2_mask is not None:
            try:
                vsl2_pfx_emb, vsl2_pfx_pos, _, _, _ = _build_prefix(vsl2_mask)
                vsl2_cache, _ = _prefill_all_layers(lm, vsl2_pfx_emb, vsl2_pfx_pos)
                vsl2_ok = True
                del vsl2_pfx_emb
            except Exception:
                pass

        # (C-class prefill already done above via _progressive_prefill)

        torch.cuda.empty_cache()

        # ── Per-image cost parameters ──
        # Ours: Stage2 keeps this many vis tokens (computed after first question)
        n_vis_stage2 = max(1, int(n_vis_stage1 * (STAGE2_FRAC / STAGE1_FRAC))) \
            if shared_ok else 0

        # ── Per-question evaluation ──
        shared_setup_per_q = image_shared_sec / max(1, len(qa_pairs))
        for q_idx, (question_raw, gt_answer) in enumerate(qa_pairs):
            shared_question_start = time.perf_counter()
            question = question_raw + GQA_SUFFIX
            # Baseline
            baseline_scores = cfg.get("baseline_scores")
            baseline_idx = len(results["baseline"])
            if baseline_scores is not None and baseline_idx < len(baseline_scores):
                bl_pred = ""
                bl_score = float(baseline_scores[baseline_idx])
                bl_gen_len = 0
            else:
                bl_pred = wrapper.generate(img, question, max_new_tokens=32)
                bl_score = score_gqa(bl_pred, gt_answer)
                bl_gen_len = _decoded_token_count(wrapper.processor.tokenizer, bl_pred)
            results["baseline"].append(bl_score)
            # Baseline cost: full visual tokens, all layers, prefill + gen
            costs["baseline"]["prefill_tl"].append(n_vis * N)
            costs["baseline"]["gen_tl"].append(n_vis * N * bl_gen_len)
            costs["baseline"]["total_tl"].append(n_vis * N * (1 + bl_gen_len))
            costs["baseline"]["n_vis_decode"].append(n_vis)
            costs["baseline"]["n_gen_tokens"].append(bl_gen_len)
            costs["baseline"]["shared_setup_sec"].append(shared_setup_per_q)
            costs["baseline"]["shared_question_sec"].append(time.perf_counter() - shared_question_start)
            costs["baseline"]["policy_prefill_wall_sec"].append(0.0)
            costs["baseline"]["policy_decode_wall_sec"].append(0.0)
            costs["baseline"]["policy_total_wall_sec"].append(
                costs["baseline"]["shared_setup_sec"][-1] + costs["baseline"]["shared_question_sec"][-1])

            # ── Stage2 policies: shared Stage1 prefix + per-question pruning ──
            if shared_ok:
                try:
                    inputs_q = wrapper.prepare_inputs(img, question)
                    mr_s1_q = tok.apply_token_mask(inputs_q, vis_embeds, grid_thw,
                                                    keep_mask=stage1_mask,
                                                    spatial_merge_size=merge,
                                                    config=model.model.config)
                    s1q_ids = mr_s1_q["new_input_ids"].to(device)
                    s1q_text_emb = lm.embed_tokens(s1q_ids).to(dtype)
                    s1q_vis = mr_s1_q["new_visual_embeds"].to(device).to(dtype)
                    if s1q_vis.shape[0] > 0:
                        s1q_mask, _ = model.model.get_placeholder_mask(
                            s1q_ids, inputs_embeds=s1q_text_emb, image_features=s1q_vis)
                        s1q_full = s1q_text_emb.masked_scatter(s1q_mask, s1q_vis)
                    else:
                        s1q_full = s1q_text_emb

                    question_embeds = s1q_full[:, n_prefix:, :]
                    question_pos = mr_s1_q["position_ids"][:, :, n_prefix:].to(device)
                    shared_question_sec = time.perf_counter() - shared_question_start

                    if policy_methods:
                        globals()["_stage2_score_cache"] = {}
                        for method in policy_methods:
                            method_name = method["output_name"]
                            info = policy_shared[method_name]
                            globals()["_active_stage2_policy"] = method["policy"]
                            method_stage2_frac = float(method.get("stage2_frac", STAGE2_FRAC))
                            globals()["_active_stage2_frac"] = method_stage2_frac
                            policy_start = time.perf_counter()
                            gen_ids = _per_question_prune_generate(
                                lm, _lm_head, info["h"], info["cache"],
                                question_embeds, vis_pos_in_shared, n_vis_stage1,
                                info["S"], method_stage2_frac, _eos_id, max_new_tokens=32,
                                shared_pos_ids=shared_pos, question_pos_ids=question_pos,
                                scoring="policy")
                            pred = wrapper.processor.tokenizer.decode(
                                gen_ids, skip_special_tokens=True).strip()
                            policy_decode_sec = time.perf_counter() - policy_start
                            results[method_name].append(score_gqa(pred, gt_answer))
                            n_gen = _decoded_token_count(wrapper.processor.tokenizer, pred)
                            n_vis_stage2_m = max(1, int(n_vis_stage1 * (method_stage2_frac / STAGE1_FRAC)))
                            pfx_tl = n_vis_stage1 * info["S"] + n_vis_stage2_m * (N - info["S"])
                            gen_tl = n_vis_stage2_m * N * n_gen
                            costs[method_name]["prefill_tl"].append(pfx_tl)
                            costs[method_name]["gen_tl"].append(gen_tl)
                            costs[method_name]["total_tl"].append(pfx_tl + gen_tl)
                            costs[method_name]["n_vis_decode"].append(n_vis_stage2_m)
                            costs[method_name]["n_gen_tokens"].append(n_gen)
                            prefill_wall_per_q = info["prefill_wall_sec"] / max(1, len(qa_pairs))
                            costs[method_name]["shared_setup_sec"].append(shared_setup_per_q)
                            costs[method_name]["shared_question_sec"].append(shared_question_sec)
                            costs[method_name]["policy_prefill_wall_sec"].append(prefill_wall_per_q)
                            costs[method_name]["policy_decode_wall_sec"].append(policy_decode_sec)
                            costs[method_name]["policy_total_wall_sec"].append(
                                shared_setup_per_q + shared_question_sec + prefill_wall_per_q + policy_decode_sec)
                    else:
                        policy_start = time.perf_counter()
                        gen_ids = _per_question_prune_generate(
                            lm, _lm_head, h_shared, shared_cache,
                            question_embeds, vis_pos_in_shared, n_vis_stage1,
                            S, STAGE2_FRAC, _eos_id, max_new_tokens=32,
                            shared_pos_ids=shared_pos, question_pos_ids=question_pos,
                            scoring=cfg.get("scoring", "single"))
                        pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                        policy_decode_sec = time.perf_counter() - policy_start
                        results["ours"].append(score_gqa(pred, gt_answer))
                        n_gen = _decoded_token_count(wrapper.processor.tokenizer, pred)
                        pfx_tl = n_vis_stage1 * S + n_vis_stage2 * (N - S)
                        gen_tl = n_vis_stage2 * N * n_gen
                        costs["ours"]["prefill_tl"].append(pfx_tl)
                        costs["ours"]["gen_tl"].append(gen_tl)
                        costs["ours"]["total_tl"].append(pfx_tl + gen_tl)
                        costs["ours"]["n_vis_decode"].append(n_vis_stage2)
                        costs["ours"]["n_gen_tokens"].append(n_gen)
                        prefill_wall_per_q = policy_prefill_wall_sec / max(1, len(qa_pairs))
                        costs["ours"]["shared_setup_sec"].append(shared_setup_per_q)
                        costs["ours"]["shared_question_sec"].append(shared_question_sec)
                        costs["ours"]["policy_prefill_wall_sec"].append(prefill_wall_per_q)
                        costs["ours"]["policy_decode_wall_sec"].append(policy_decode_sec)
                        costs["ours"]["policy_total_wall_sec"].append(
                            shared_setup_per_q + shared_question_sec + prefill_wall_per_q + policy_decode_sec)
                    del s1q_full
                except Exception as e:
                    if img_idx == 0 and q_idx == 0:
                        import traceback; traceback.print_exc()
                    target_methods = [m["output_name"] for m in policy_methods] if policy_methods else ["ours"]
                    for method_name in target_methods:
                        results[method_name].append(0.0)
                        costs[method_name]["prefill_tl"].append(0)
                        costs[method_name]["gen_tl"].append(0)
                        costs[method_name]["total_tl"].append(0)
                        costs[method_name]["n_vis_decode"].append(0)
                        costs[method_name]["n_gen_tokens"].append(0)
            else:
                target_methods = [m["output_name"] for m in policy_methods] if policy_methods else ["ours"]
                for method_name in target_methods:
                    results[method_name].append(0.0)
                    for ck in costs[method_name]:
                        costs[method_name][ck].append(0)

            if cfg.get("policy_only", False):
                continue

            # ── Helper: extract question-only embeddings for a given mask ──
            def _get_question_embeds(mask):
                iq = wrapper.prepare_inputs(img, question)
                mr = tok.apply_token_mask(iq, vis_embeds, grid_thw,
                                          keep_mask=mask,
                                          spatial_merge_size=merge,
                                          config=model.model.config)
                ids_t = mr["new_input_ids"].to(device)
                t_emb = lm.embed_tokens(ids_t).to(dtype)
                v_emb = mr["new_visual_embeds"].to(device).to(dtype)
                if v_emb.shape[0] > 0:
                    m, _ = model.model.get_placeholder_mask(
                        ids_t, inputs_embeds=t_emb, image_features=v_emb)
                    emb = t_emb.masked_scatter(m, v_emb)
                else:
                    emb = t_emb
                id_seq = mr["new_input_ids"][0]
                v_e = (id_seq == _VE).nonzero(as_tuple=True)[0][0].item()
                n_pfx = v_e + 1
                q_emb = emb[:, n_pfx:, :]
                q_pos = mr["position_ids"][:, :, n_pfx:].to(device)
                return q_emb, q_pos

            # ── Helper: record cost for a static baseline ──
            def _record_cost(method, n_vis_m, pred):
                n_gen = _decoded_token_count(wrapper.processor.tokenizer, pred)
                pfx_tl = n_vis_m * N
                gen_tl = n_vis_m * N * n_gen
                costs[method]["prefill_tl"].append(pfx_tl)
                costs[method]["gen_tl"].append(gen_tl)
                costs[method]["total_tl"].append(pfx_tl + gen_tl)
                costs[method]["n_vis_decode"].append(n_vis_m)
                costs[method]["n_gen_tokens"].append(n_gen)

            def _record_zero(method):
                for ck in costs[method]:
                    costs[method][ck].append(0)

            # ── PACT (KV cache reuse) ──
            if pact_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(pact_mask)
                    gen_ids = _generate_from_shared_cache(
                        lm, _lm_head, pact_cache, pact_pfx_pos,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["pact"].append(score_gqa(pred, gt_answer))
                    _record_cost("pact", k_y, pred)
                except Exception:
                    results["pact"].append(0.0)
                    _record_zero("pact")
            else:
                results["pact"].append(0.0)
                _record_zero("pact")

            # ── SparseVILA (KV cache reuse + per-question decode pruning) ──
            if sv_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(sv_mask)
                    gen_ids = _sparsevila_per_question_generate(
                        lm, _lm_head, sv_cache,
                        sv_vis_in_pfx, n_sv_vis, SPARSEVILA_DECODE_RATIO,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["sparsevila"].append(score_gqa(pred, gt_answer))
                    # Cost: prefill with k_y vis, decode with n_sv_decode_keep
                    n_gen = _decoded_token_count(wrapper.processor.tokenizer, pred)
                    pfx_tl = n_sv_vis * N
                    gen_tl = n_sv_decode_keep * N * n_gen
                    costs["sparsevila"]["prefill_tl"].append(pfx_tl)
                    costs["sparsevila"]["gen_tl"].append(gen_tl)
                    costs["sparsevila"]["total_tl"].append(pfx_tl + gen_tl)
                    costs["sparsevila"]["n_vis_decode"].append(n_sv_decode_keep)
                    costs["sparsevila"]["n_gen_tokens"].append(n_gen)
                except Exception:
                    results["sparsevila"].append(0.0)
                    _record_zero("sparsevila")
            else:
                results["sparsevila"].append(0.0)
                _record_zero("sparsevila")

            # ── Attn top-k L0 (KV cache reuse) ──
            if attn_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(attn_mask)
                    gen_ids = _generate_from_shared_cache(
                        lm, _lm_head, attn_cache, attn_pfx_pos,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["attn_topk"].append(score_gqa(pred, gt_answer))
                    _record_cost("attn_topk", k_y, pred)
                except Exception:
                    results["attn_topk"].append(0.0)
                    _record_zero("attn_topk")
            else:
                results["attn_topk"].append(0.0)
                _record_zero("attn_topk")

            # ── SVDPrune (KV cache reuse) ──
            if svd_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(svd_mask)
                    gen_ids = _generate_from_shared_cache(
                        lm, _lm_head, svd_cache, svd_pfx_pos,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["svdprune"].append(score_gqa(pred, gt_answer))
                    _record_cost("svdprune", k_y, pred)
                except Exception:
                    results["svdprune"].append(0.0)
                    _record_zero("svdprune")
            else:
                results["svdprune"].append(0.0)
                _record_zero("svdprune")

            # ── DivPrune (KV cache reuse) ──
            if div_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(div_mask)
                    gen_ids = _generate_from_shared_cache(
                        lm, _lm_head, div_cache, div_pfx_pos,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["divprune"].append(score_gqa(pred, gt_answer))
                    _record_cost("divprune", k_y, pred)
                except Exception:
                    results["divprune"].append(0.0)
                    _record_zero("divprune")
            else:
                results["divprune"].append(0.0)
                _record_zero("divprune")

            # ── VisPruner (KV cache reuse) ──
            if visp_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(visp_mask)
                    gen_ids = _generate_from_shared_cache(
                        lm, _lm_head, visp_cache, visp_pfx_pos,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["vispruner"].append(score_gqa(pred, gt_answer))
                    _record_cost("vispruner", k_y, pred)
                except Exception:
                    results["vispruner"].append(0.0)
                    _record_zero("vispruner")
            else:
                results["vispruner"].append(0.0)
                _record_zero("vispruner")

            # ── FastV (KV cache reuse, L2 last-text scoring) ──
            if fastv_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(fastv_mask)
                    gen_ids = _generate_from_shared_cache(
                        lm, _lm_head, fastv_cache, fastv_pfx_pos,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["fastv"].append(score_gqa(pred, gt_answer))
                    _record_cost("fastv", k_y, pred)
                except Exception:
                    results["fastv"].append(0.0)
                    _record_zero("fastv")
            else:
                results["fastv"].append(0.0)
                _record_zero("fastv")

            # ── ZSPAPrune (KV cache reuse) ──
            if zspap_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(zspap_mask)
                    gen_ids = _generate_from_shared_cache(
                        lm, _lm_head, zspap_cache, zspap_pfx_pos,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["zspaprune"].append(score_gqa(pred, gt_answer))
                    _record_cost("zspaprune", k_y, pred)
                except Exception:
                    results["zspaprune"].append(0.0)
                    _record_zero("zspaprune")
            else:
                results["zspaprune"].append(0.0)
                _record_zero("zspaprune")

            # ── AgilePruner (KV cache reuse) ──
            if agile_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(agile_mask)
                    gen_ids = _generate_from_shared_cache(
                        lm, _lm_head, agile_cache, agile_pfx_pos,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["agilepruner"].append(score_gqa(pred, gt_answer))
                    _record_cost("agilepruner", k_y, pred)
                except Exception:
                    results["agilepruner"].append(0.0)
                    _record_zero("agilepruner")
            else:
                results["agilepruner"].append(0.0)
                _record_zero("agilepruner")

            # ── IDSelection (KV cache reuse) ──
            if idsel_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(idsel_mask)
                    gen_ids = _generate_from_shared_cache(
                        lm, _lm_head, idsel_cache, idsel_pfx_pos,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["idselection"].append(score_gqa(pred, gt_answer))
                    _record_cost("idselection", k_y, pred)
                except Exception:
                    results["idselection"].append(0.0)
                    _record_zero("idselection")
            else:
                results["idselection"].append(0.0)
                _record_zero("idselection")

            # ── D²Pruner (KV cache reuse) ──
            if d2p_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(d2p_mask)
                    gen_ids = _generate_from_shared_cache(
                        lm, _lm_head, d2p_cache, d2p_pfx_pos,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["d2pruner"].append(score_gqa(pred, gt_answer))
                    _record_cost("d2pruner", k_y, pred)
                except Exception:
                    results["d2pruner"].append(0.0)
                    _record_zero("d2pruner")
            else:
                results["d2pruner"].append(0.0)
                _record_zero("d2pruner")

            # ── PTP (KV cache reuse) ──
            if ptp_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(ptp_mask)
                    gen_ids = _generate_from_shared_cache(
                        lm, _lm_head, ptp_cache, ptp_pfx_pos,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["ptp"].append(score_gqa(pred, gt_answer))
                    _record_cost("ptp", k_y, pred)
                except Exception:
                    results["ptp"].append(0.0)
                    _record_zero("ptp")
            else:
                results["ptp"].append(0.0)
                _record_zero("ptp")

            # ── HAWK (KV cache reuse) ──
            if hawk_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(hawk_mask)
                    gen_ids = _generate_from_shared_cache(
                        lm, _lm_head, hawk_cache, hawk_pfx_pos,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["hawk"].append(score_gqa(pred, gt_answer))
                    _record_cost("hawk", k_y, pred)
                except Exception:
                    results["hawk"].append(0.0)
                    _record_zero("hawk")
            else:
                results["hawk"].append(0.0)
                _record_zero("hawk")

            # ── VScoreL2 (KV cache reuse) ──
            if vsl2_ok:
                try:
                    q_emb, q_pos = _get_question_embeds(vsl2_mask)
                    gen_ids = _generate_from_shared_cache(
                        lm, _lm_head, vsl2_cache, vsl2_pfx_pos,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["vscore_l2"].append(score_gqa(pred, gt_answer))
                    _record_cost("vscore_l2", k_y, pred)
                except Exception:
                    results["vscore_l2"].append(0.0)
                    _record_zero("vscore_l2")
            else:
                results["vscore_l2"].append(0.0)
                _record_zero("vscore_l2")

            # ── FitPrune (progressive cache) ──
            if fitprune_ok:
                try:
                    q_emb, q_pos = _get_question_embeds([True] * n_vis)
                    gen_ids = _gen_from_progressive_cache(
                        lm, _lm_head, fitprune_cache, _n_full_pfx,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["fitprune"].append(score_gqa(pred, gt_answer))
                    n_gen = _decoded_token_count(wrapper.processor.tokenizer, pred)
                    costs["fitprune"]["prefill_tl"].append(fitprune_tl)
                    costs["fitprune"]["gen_tl"].append(k_y * N * n_gen)
                    costs["fitprune"]["total_tl"].append(fitprune_tl + k_y * N * n_gen)
                    costs["fitprune"]["n_vis_decode"].append(k_y)
                    costs["fitprune"]["n_gen_tokens"].append(n_gen)
                except Exception:
                    results["fitprune"].append(0.0)
                    _record_zero("fitprune")
            else:
                results["fitprune"].append(0.0)
                _record_zero("fitprune")

            # ── PyramidDrop (progressive cache) ──
            if pyrdrop_ok:
                try:
                    q_emb, q_pos = _get_question_embeds([True] * n_vis)
                    gen_ids = _gen_from_progressive_cache(
                        lm, _lm_head, pyrdrop_cache, _n_full_pfx,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["pyramiddrop"].append(score_gqa(pred, gt_answer))
                    n_gen = _decoded_token_count(wrapper.processor.tokenizer, pred)
                    costs["pyramiddrop"]["prefill_tl"].append(pyrdrop_tl)
                    costs["pyramiddrop"]["gen_tl"].append(k_y * N * n_gen)
                    costs["pyramiddrop"]["total_tl"].append(pyrdrop_tl + k_y * N * n_gen)
                    costs["pyramiddrop"]["n_vis_decode"].append(k_y)
                    costs["pyramiddrop"]["n_gen_tokens"].append(n_gen)
                except Exception:
                    results["pyramiddrop"].append(0.0)
                    _record_zero("pyramiddrop")
            else:
                results["pyramiddrop"].append(0.0)
                _record_zero("pyramiddrop")

            # ── SparseVLM (progressive cache) ──
            if spvlm_ok:
                try:
                    q_emb, q_pos = _get_question_embeds([True] * n_vis)
                    gen_ids = _gen_from_progressive_cache(
                        lm, _lm_head, spvlm_cache, _n_full_pfx,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = wrapper.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results["sparsevlm"].append(score_gqa(pred, gt_answer))
                    n_gen = _decoded_token_count(wrapper.processor.tokenizer, pred)
                    costs["sparsevlm"]["prefill_tl"].append(spvlm_tl)
                    costs["sparsevlm"]["gen_tl"].append(k_y * N * n_gen)
                    costs["sparsevlm"]["total_tl"].append(spvlm_tl + k_y * N * n_gen)
                    costs["sparsevlm"]["n_vis_decode"].append(k_y)
                    costs["sparsevlm"]["n_gen_tokens"].append(n_gen)
                except Exception:
                    results["sparsevlm"].append(0.0)
                    _record_zero("sparsevlm")
            else:
                results["sparsevlm"].append(0.0)
                _record_zero("sparsevlm")

        if shared_ok:
            if policy_methods:
                for info in policy_shared.values():
                    del info["h"], info["cache"]
            else:
                del h_shared, shared_cache
        if pact_ok:
            del pact_cache
        if sv_ok:
            del sv_cache
        if attn_ok:
            del attn_cache
        if svd_ok:
            del svd_cache
        if div_ok:
            del div_cache
        if visp_ok:
            del visp_cache
        if fastv_ok:
            del fastv_cache
        if zspap_ok:
            del zspap_cache
        if agile_ok:
            del agile_cache
        if idsel_ok:
            del idsel_cache
        if d2p_ok:
            del d2p_cache
        if ptp_ok:
            del ptp_cache
        if hawk_ok:
            del hawk_cache
        if vsl2_ok:
            del vsl2_cache
        if fitprune_ok:
            del fitprune_cache
        if pyrdrop_ok:
            del pyrdrop_cache
        if spvlm_ok:
            del spvlm_cache
        torch.cuda.empty_cache()

        n_q = len(qa_pairs)
        if cfg.get("policy_only", False):
            policy_labels = [m["output_name"] for m in cfg.get("policy_methods", [])]
            if not policy_labels:
                policy_labels = [cfg.get("policy_name", "policy")]
            policy_summary = " ".join(
                f"{label}={np.mean(results[label][-n_q:]):.3f}"
                for label in policy_labels)
            cumul_summary = " ".join(
                f"{label}={np.mean(results[label]):.3f}"
                for label in policy_labels)
            print(f"  [img {img_idx+1}/{len(gqa_samples)}] {n_q}Q "
                  f"bl={np.mean(results['baseline'][-n_q:]):.3f} "
                  f"{policy_summary} | cumul: {cumul_summary}", flush=True)
        else:
            print(f"  [img {img_idx+1}/{len(gqa_samples)}] {n_q}Q "
                  f"bl={np.mean(results['baseline'][-n_q:]):.3f} "
                  f"ours={np.mean(results['ours'][-n_q:]):.3f} "
                  f"pact={np.mean(results['pact'][-n_q:]):.3f} "
                  f"svd={np.mean(results['svdprune'][-n_q:]):.3f} "
                  f"div={np.mean(results['divprune'][-n_q:]):.3f} "
                  f"| cumul: ours={np.mean(results['ours']):.3f}", flush=True)

    del wrapper, model
    gc.collect(); torch.cuda.empty_cache()
    return results, costs


def run_gqa_llava(model_name, gqa_samples, cfg, out_dir):
    """Multi-round GQA benchmark for LLaVA models."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_id = cfg["model_id"]
    print(f"[LLaVA] Loading {model_id}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    lm = model.model.language_model
    IMAGE_TOKEN_ID = 32000
    S, Y = cfg["S"], cfg["Y"]
    N = int(cfg.get("N", len(lm.layers)))

    ALL_METHODS = ["baseline", "ours", "pact", "sparsevila", "attn_topk",
                   "svdprune", "divprune", "vispruner", "fastv",
                   "zspaprune", "agilepruner", "idselection", "d2pruner", "ptp",
                   "hawk", "vscore_l2",
                   "fitprune", "pyramiddrop", "sparsevlm"]
    results = {m: [] for m in ALL_METHODS}
    policy_methods = cfg.get("policy_methods") or []
    for method in policy_methods:
        results.setdefault(method["output_name"], [])
    costs = {m: _new_cost_bucket()
             for m in ["baseline"] + [method["output_name"] for method in policy_methods]}

    def _gen_with_mask(full_embeds, input_ids, vis_pos_set, kept_vis_pos_set):
        all_pos = [p for p in range(input_ids.shape[1])
                   if p not in vis_pos_set or p in kept_vis_pos_set]
        pruned_embeds = full_embeds[:, all_pos, :]
        pruned_attn = torch.ones(1, len(all_pos), device="cuda", dtype=torch.long)
        with torch.no_grad():
            out_ids = model.generate(
                inputs_embeds=pruned_embeds, attention_mask=pruned_attn,
                max_new_tokens=32, do_sample=False)
        return processor.decode(out_ids[0], skip_special_tokens=True).strip()

    for img_idx, (img, qa_pairs) in enumerate(gqa_samples):
        img = _dualsignal_resize_image(img)
        image_shared_start = time.perf_counter()
        # ── Build image features once ──
        first_prompt = f"USER: <image>\n{qa_pairs[0][0] + GQA_SUFFIX}\nASSISTANT:"
        first_inputs = processor(text=first_prompt, images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            pixel_values = first_inputs["pixel_values"].to(model.dtype)
            image_features = model.model.multi_modal_projector(
                model.model.vision_tower(pixel_values).last_hidden_state)

        # ── Compute shared baseline token sets ──
        input_ids_0 = first_inputs["input_ids"]
        text_emb_0 = lm.embed_tokens(input_ids_0)
        img_positions_0 = (input_ids_0[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        n_vis = len(img_positions_0)
        full_embeds_0 = text_emb_0.clone()
        full_embeds_0[0, img_positions_0[:n_vis]] = image_features[0, :n_vis].to(text_emb_0.dtype)

        vis_pos_0 = img_positions_0.tolist()
        vis_pos_set_0 = set(vis_pos_0)
        text_pos_0 = [p for p in range(input_ids_0.shape[1])
                      if p not in vis_pos_set_0 and p > max(vis_pos_0)]

        k_y = max(1, int(n_vis * Y))
        k1 = max(1, int(n_vis * STAGE1_FRAC))
        vis_embeds_for_mmr = image_features[0, :n_vis]
        vis_embeds = vis_embeds_for_mmr

        # ViT importance (used by both Stage 1 and SparseVILA)
        try:
            vit_imp = _llava_vit_importance(model, pixel_values, n_vis)
        except Exception:
            vit_imp = None

        # Stage 1 shared: dual signal + MMR
        imp0_shared = _get_importance_at_layer(lm, 0, full_embeds_0, vis_pos_0, text_pos_0)
        if vit_imp is not None:
            stage1_idx = _dual_signal_stage1_select(imp0_shared, vit_imp, vis_embeds_for_mmr, k1)
        else:
            stage1_idx = np.argsort(imp0_shared)[::-1][:k1].tolist()
        stage1_set = set(stage1_idx)

        # PACT shared
        pact_idx = _pact_select(lm, full_embeds_0, vis_pos_0, k_y)
        pact_kept = {vis_pos_0[i] for i in pact_idx}

        # SparseVILA shared
        try:
            sv_idx = np.argsort(vit_imp)[::-1][:k_y]
            sv_kept = {vis_pos_0[i] for i in sv_idx}
        except Exception:
            sv_kept = None

        # Attn top-k shared
        attn_idx = np.argsort(imp0_shared)[::-1][:k_y]
        attn_kept = {vis_pos_0[i] for i in attn_idx}

        # ── NEW: SVDPrune (query-agnostic) ──
        try:
            svd_mask_l = _svdprune_select(vis_embeds_for_mmr, k_y)
            svd_kept = {vis_pos_0[i] for i, k in enumerate(svd_mask_l) if k}
        except Exception:
            svd_kept = None

        # ── NEW: DivPrune (query-agnostic) ──
        try:
            div_mask_l = _divprune_select(vis_embeds_for_mmr, k_y)
            div_kept = {vis_pos_0[i] for i, k in enumerate(div_mask_l) if k}
        except Exception:
            div_kept = None

        # ── NEW: VisPruner (query-agnostic, ViT + diversity) ──
        if vit_imp is not None:
            try:
                visp_mask_l = _vispruner_select(vit_imp, vis_embeds_for_mmr, k_y)
                visp_kept = {vis_pos_0[i] for i, k in enumerate(visp_mask_l) if k}
            except Exception:
                visp_kept = None
        else:
            visp_kept = None

        # ── NEW: FastV (L2 last-text attention, computed once) ──
        try:
            fastv_mask_l = _fastv_select(lm, full_embeds_0, vis_pos_0, text_pos_0, k_y, layer=2)
            fastv_kept = {vis_pos_0[i] for i, k in enumerate(fastv_mask_l) if k}
        except Exception:
            fastv_kept = None

        # ── B-class: ZSPAPrune ──
        try:
            zspap_mask_l = _zspaprune_select(lm, full_embeds_0, vis_pos_0, text_pos_0,
                                              vis_embeds, k_y)
            zspap_kept = {vis_pos_0[i] for i, k in enumerate(zspap_mask_l) if k}
        except Exception:
            zspap_kept = None

        # ── B-class: AgilePruner ──
        try:
            agile_mask_l = _agilepruner_select(lm, full_embeds_0, vis_pos_0, text_pos_0,
                                                vis_embeds, k_y)
            agile_kept = {vis_pos_0[i] for i, k in enumerate(agile_mask_l) if k}
        except Exception:
            agile_kept = None

        # ── B-class: IDSelection ──
        try:
            idsel_mask_l = _idselection_select(lm, full_embeds_0, vis_pos_0, text_pos_0,
                                                vis_embeds, k_y)
            idsel_kept = {vis_pos_0[i] for i, k in enumerate(idsel_mask_l) if k}
        except Exception:
            idsel_kept = None

        # ── B-class: D²Pruner ──
        try:
            d2p_mask_l = _d2pruner_select(lm, full_embeds_0, vis_pos_0, text_pos_0,
                                           vis_embeds, k_y)
            d2p_kept = {vis_pos_0[i] for i, k in enumerate(d2p_mask_l) if k}
        except Exception:
            d2p_kept = None

        # ── B-class: PTP ──
        try:
            ptp_mask_l = _ptp_select(vit_imp, lm, full_embeds_0, vis_pos_0, text_pos_0,
                                      vis_embeds, k_y)
            ptp_kept = {vis_pos_0[i] for i, k in enumerate(ptp_mask_l) if k}
        except Exception:
            ptp_kept = None

        # ── B-class: HAWK ──
        try:
            hawk_mask_l = _hawk_select(lm, full_embeds_0, vis_pos_0, text_pos_0, k_y)
            hawk_kept = {vis_pos_0[i] for i, k in enumerate(hawk_mask_l) if k}
        except Exception:
            hawk_kept = None

        # ── B-class: VScoreL2 ──
        try:
            vsl2_mask_l = _vscore_l2_select(lm, full_embeds_0, vis_pos_0, text_pos_0, k_y)
            vsl2_kept = {vis_pos_0[i] for i, k in enumerate(vsl2_mask_l) if k}
        except Exception:
            vsl2_kept = None

        del full_embeds_0

        # ── Helper: build prefix embeddings from a kept-vis-pos set ──
        last_vis_orig = max(vis_pos_0)
        _eos_id = processor.tokenizer.eos_token_id

        def _build_llava_prefix(kept_vis_set):
            """Build shared prefix [pre-image text + selected vis tokens]."""
            pfx_pos = [p for p in range(input_ids_0.shape[1])
                       if (p not in vis_pos_set_0 or p in kept_vis_set) and p <= last_vis_orig]
            text_emb_tmp = lm.embed_tokens(input_ids_0)
            full_tmp = text_emb_tmp.clone()
            full_tmp[0, img_positions_0[:n_vis]] = image_features[0, :n_vis].to(text_emb_tmp.dtype)
            pfx_emb = full_tmp[:, pfx_pos, :]
            del text_emb_tmp, full_tmp
            vis_in_pfx = [j for j, p in enumerate(pfx_pos) if p in kept_vis_set]
            return pfx_emb, pfx_pos, vis_in_pfx

        def _get_llava_question_embeds(full_embeds, input_ids, kept_vis_set, n_pfx):
            """Extract question-only embeddings (post-image text)."""
            vis_pos_set = set((input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0].tolist())
            all_kept = [p for p in range(input_ids.shape[1])
                        if p not in vis_pos_set or p in kept_vis_set]
            q_emb = full_embeds[:, all_kept[n_pfx:], :]
            return q_emb

        # ── Prefill shared prefix through L0..S-1 for OUR method (ONCE) ──
        policy_shared = {}
        image_shared_sec = time.perf_counter() - image_shared_start
        try:
            stage1_vis_pos_set_0 = {vis_pos_0[i] for i in stage1_set}
            ours_pfx_emb, ours_pfx_pos, vis_pos_in_s1 = \
                _build_llava_prefix(stage1_vis_pos_set_0)
            n_prefix = len(ours_pfx_pos)
            n_vis_stage1 = len(vis_pos_in_s1)
            if policy_methods:
                for method in policy_methods:
                    method_S = int(method["S"])
                    prefill_start = time.perf_counter()
                    h_m, cache_m = _prefill_shared_layers(lm, ours_pfx_emb, method_S)
                    policy_shared[method["output_name"]] = {
                        "h": h_m,
                        "cache": cache_m,
                        "S": method_S,
                        "prefill_wall_sec": time.perf_counter() - prefill_start,
                    }
                shared_ok = bool(policy_shared)
            else:
                prefill_start = time.perf_counter()
                h_shared, shared_cache = _prefill_shared_layers(lm, ours_pfx_emb, S)
                policy_prefill_wall_sec = time.perf_counter() - prefill_start
                shared_ok = True
            del ours_pfx_emb
        except Exception as e:
            import traceback; traceback.print_exc()
            shared_ok = False

        # ── Prefill ALL N layers for PACT (ONCE per image) ──
        try:
            pact_pfx_emb, pact_pfx_pos, _ = _build_llava_prefix(pact_kept)
            n_pact_pfx = len(pact_pfx_pos)
            pact_cache, _ = _prefill_all_layers(lm, pact_pfx_emb)
            pact_ok = True
            del pact_pfx_emb
        except Exception:
            pact_ok = False

        # ── Prefill ALL N layers for SparseVILA (ONCE per image) ──
        # Prefill is query-independent; decode pruning is per-question
        # using query_Q @ cached_K salience.
        sv_ok = False
        if sv_kept is not None:
            try:
                sv_pfx_emb, sv_pfx_pos, sv_vis_in_pfx = _build_llava_prefix(sv_kept)
                n_sv_pfx = len(sv_pfx_pos)
                n_sv_vis = len(sv_vis_in_pfx)
                n_decode_keep = max(1, int(n_sv_vis * SPARSEVILA_DECODE_RATIO))
                sv_cache, _ = _prefill_all_layers(lm, sv_pfx_emb)
                del sv_pfx_emb
                sv_ok = True
            except Exception:
                import traceback; traceback.print_exc()

        # ── Prefill ALL N layers for Attn top-k (ONCE per image) ──
        try:
            attn_pfx_emb, attn_pfx_pos, _ = _build_llava_prefix(attn_kept)
            n_attn_pfx = len(attn_pfx_pos)
            attn_cache, _ = _prefill_all_layers(lm, attn_pfx_emb)
            attn_ok = True
            del attn_pfx_emb
        except Exception:
            attn_ok = False

        # ── Prefill ALL N layers for SVDPrune (ONCE per image) ──
        svd_ok = False
        if svd_kept is not None:
            try:
                svd_pfx_emb, svd_pfx_pos, _ = _build_llava_prefix(svd_kept)
                n_svd_pfx = len(svd_pfx_pos)
                svd_cache, _ = _prefill_all_layers(lm, svd_pfx_emb)
                svd_ok = True
                del svd_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for DivPrune (ONCE per image) ──
        div_ok = False
        if div_kept is not None:
            try:
                div_pfx_emb, div_pfx_pos, _ = _build_llava_prefix(div_kept)
                n_div_pfx = len(div_pfx_pos)
                div_cache, _ = _prefill_all_layers(lm, div_pfx_emb)
                div_ok = True
                del div_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for VisPruner (ONCE per image) ──
        visp_ok = False
        if visp_kept is not None:
            try:
                visp_pfx_emb, visp_pfx_pos, _ = _build_llava_prefix(visp_kept)
                n_visp_pfx = len(visp_pfx_pos)
                visp_cache, _ = _prefill_all_layers(lm, visp_pfx_emb)
                visp_ok = True
                del visp_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for FastV (ONCE per image) ──
        fastv_ok = False
        if fastv_kept is not None:
            try:
                fastv_pfx_emb, fastv_pfx_pos, _ = _build_llava_prefix(fastv_kept)
                n_fastv_pfx = len(fastv_pfx_pos)
                fastv_cache, _ = _prefill_all_layers(lm, fastv_pfx_emb)
                fastv_ok = True
                del fastv_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for ZSPAPrune (ONCE per image) ──
        zspap_ok = False
        if zspap_kept is not None:
            try:
                zspap_pfx_emb, zspap_pfx_pos, _ = _build_llava_prefix(zspap_kept)
                n_zspap_pfx = len(zspap_pfx_pos)
                zspap_cache, _ = _prefill_all_layers(lm, zspap_pfx_emb)
                zspap_ok = True
                del zspap_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for AgilePruner (ONCE per image) ──
        agile_ok = False
        if agile_kept is not None:
            try:
                agile_pfx_emb, agile_pfx_pos, _ = _build_llava_prefix(agile_kept)
                n_agile_pfx = len(agile_pfx_pos)
                agile_cache, _ = _prefill_all_layers(lm, agile_pfx_emb)
                agile_ok = True
                del agile_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for IDSelection (ONCE per image) ──
        idsel_ok = False
        if idsel_kept is not None:
            try:
                idsel_pfx_emb, idsel_pfx_pos, _ = _build_llava_prefix(idsel_kept)
                n_idsel_pfx = len(idsel_pfx_pos)
                idsel_cache, _ = _prefill_all_layers(lm, idsel_pfx_emb)
                idsel_ok = True
                del idsel_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for D²Pruner (ONCE per image) ──
        d2p_ok = False
        if d2p_kept is not None:
            try:
                d2p_pfx_emb, d2p_pfx_pos, _ = _build_llava_prefix(d2p_kept)
                n_d2p_pfx = len(d2p_pfx_pos)
                d2p_cache, _ = _prefill_all_layers(lm, d2p_pfx_emb)
                d2p_ok = True
                del d2p_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for PTP (ONCE per image) ──
        ptp_ok = False
        if ptp_kept is not None:
            try:
                ptp_pfx_emb, ptp_pfx_pos, _ = _build_llava_prefix(ptp_kept)
                n_ptp_pfx = len(ptp_pfx_pos)
                ptp_cache, _ = _prefill_all_layers(lm, ptp_pfx_emb)
                ptp_ok = True
                del ptp_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for HAWK (ONCE per image) ──
        hawk_ok = False
        if hawk_kept is not None:
            try:
                hawk_pfx_emb, hawk_pfx_pos, _ = _build_llava_prefix(hawk_kept)
                n_hawk_pfx = len(hawk_pfx_pos)
                hawk_cache, _ = _prefill_all_layers(lm, hawk_pfx_emb)
                hawk_ok = True
                del hawk_pfx_emb
            except Exception:
                pass

        # ── Prefill ALL N layers for VScoreL2 (ONCE per image) ──
        vsl2_ok = False
        if vsl2_kept is not None:
            try:
                vsl2_pfx_emb, vsl2_pfx_pos, _ = _build_llava_prefix(vsl2_kept)
                n_vsl2_pfx = len(vsl2_pfx_pos)
                vsl2_cache, _ = _prefill_all_layers(lm, vsl2_pfx_emb)
                vsl2_ok = True
                del vsl2_pfx_emb
            except Exception:
                pass

        # ── C-class progressive prefill (selection + KV cache in one pass) ──
        _skip_cc = cfg.get("skip_cclass", False)
        fitprune_ok = pyrdrop_ok = spvlm_ok = False
        if not _skip_cc:
            N = len(lm.layers)
            _full_pfx_emb_l, _full_pfx_pos_l, _full_vis_in_pfx_l = \
                _build_llava_prefix(vis_pos_set_0)
            _full_vis_set_l = set(_full_vis_in_pfx_l)
            _n_full_pfx_l = len(_full_pfx_pos_l)

            try:
                _fp_schedule = _fitprune_build_schedule(n_vis, k_y, N)
                _fp_drop_fn = _make_fitprune_drop_fn(_fp_schedule)
                fitprune_cache, _, fitprune_alive, fitprune_tl = \
                    _progressive_prefill(lm, _full_pfx_emb_l, set(_full_vis_set_l),
                                         _fp_drop_fn)
                fitprune_ok = True
            except Exception:
                import traceback; traceback.print_exc()

            try:
                _pd_drop_fn = _make_pyramiddrop_drop_fn(n_vis, k_y, N)
                pyrdrop_cache, _, pyrdrop_alive, pyrdrop_tl = \
                    _progressive_prefill(lm, _full_pfx_emb_l, set(_full_vis_set_l),
                                         _pd_drop_fn)
                pyrdrop_ok = True
            except Exception:
                import traceback; traceback.print_exc()

            try:
                _sv_drop_fn = _make_sparsevlm_drop_fn(n_vis, k_y, N)
                spvlm_cache, _, spvlm_alive, spvlm_tl = \
                    _progressive_prefill(lm, _full_pfx_emb_l, set(_full_vis_set_l),
                                         _sv_drop_fn)
                spvlm_ok = True
            except Exception:
                import traceback; traceback.print_exc()

            del _full_pfx_emb_l

        torch.cuda.empty_cache()

        # ── Per-question ──
        shared_setup_per_q = image_shared_sec / max(1, len(qa_pairs))
        for q_idx, (question_raw, gt_answer) in enumerate(qa_pairs):
            shared_question_start = time.perf_counter()
            question = question_raw + GQA_SUFFIX
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
            input_ids = inputs["input_ids"]
            text_emb = lm.embed_tokens(input_ids)
            img_positions = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
            full_embeds = text_emb.clone()
            full_embeds[0, img_positions[:n_vis]] = image_features[0, :n_vis].to(text_emb.dtype)

            vis_pos = img_positions.tolist()
            vis_pos_set = set(vis_pos)

            # Baseline
            baseline_scores = cfg.get("baseline_scores")
            baseline_idx = len(results["baseline"])
            if baseline_scores is not None and baseline_idx < len(baseline_scores):
                results["baseline"].append(float(baseline_scores[baseline_idx]))
                bl_gen_len = 0
            else:
                with torch.no_grad():
                    out_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
                bl_pred = processor.decode(out_ids[0][input_ids.shape[1]:],
                                           skip_special_tokens=True).strip()
                results["baseline"].append(score_gqa(bl_pred, gt_answer))
                bl_gen_len = _decoded_token_count(processor.tokenizer, bl_pred)
            costs["baseline"]["prefill_tl"].append(n_vis * N)
            costs["baseline"]["gen_tl"].append(n_vis * N * bl_gen_len)
            costs["baseline"]["total_tl"].append(n_vis * N * (1 + bl_gen_len))
            costs["baseline"]["n_vis_decode"].append(n_vis)
            costs["baseline"]["n_gen_tokens"].append(bl_gen_len)
            costs["baseline"]["shared_setup_sec"].append(shared_setup_per_q)
            costs["baseline"]["shared_question_sec"].append(time.perf_counter() - shared_question_start)
            costs["baseline"]["policy_prefill_wall_sec"].append(0.0)
            costs["baseline"]["policy_decode_wall_sec"].append(0.0)
            costs["baseline"]["policy_total_wall_sec"].append(
                costs["baseline"]["shared_setup_sec"][-1] + costs["baseline"]["shared_question_sec"][-1])

            # ── Stage2 policies: shared Stage1 prefix + per-question pruning ──
            if shared_ok:
                try:
                    stage1_vis_pos_set_q = {vis_pos[i] for i in stage1_set}
                    s1_all_pos_q = [p for p in range(input_ids.shape[1])
                                    if p not in vis_pos_set or p in stage1_vis_pos_set_q]
                    s1_embeds_q = full_embeds[:, s1_all_pos_q, :]
                    question_embeds = s1_embeds_q[:, n_prefix:, :]
                    shared_question_sec = time.perf_counter() - shared_question_start

                    if policy_methods:
                        globals()["_stage2_score_cache"] = {}
                        for method in policy_methods:
                            method_name = method["output_name"]
                            info = policy_shared[method_name]
                            globals()["_active_stage2_policy"] = method["policy"]
                            method_stage2_frac = float(method.get("stage2_frac", STAGE2_FRAC))
                            globals()["_active_stage2_frac"] = method_stage2_frac
                            policy_start = time.perf_counter()
                            gen_ids = _per_question_prune_generate(
                                lm, model.lm_head, info["h"], info["cache"],
                                question_embeds, vis_pos_in_s1, n_vis_stage1,
                                info["S"], method_stage2_frac, _eos_id, max_new_tokens=32,
                                scoring="policy")
                            pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                            policy_decode_sec = time.perf_counter() - policy_start
                            results[method_name].append(score_gqa(pred, gt_answer))
                            n_gen = _decoded_token_count(processor.tokenizer, pred)
                            n_vis_stage2_m = max(1, int(n_vis_stage1 * (method_stage2_frac / STAGE1_FRAC)))
                            pfx_tl = n_vis_stage1 * info["S"] + n_vis_stage2_m * (N - info["S"])
                            gen_tl = n_vis_stage2_m * N * n_gen
                            costs[method_name]["prefill_tl"].append(pfx_tl)
                            costs[method_name]["gen_tl"].append(gen_tl)
                            costs[method_name]["total_tl"].append(pfx_tl + gen_tl)
                            costs[method_name]["n_vis_decode"].append(n_vis_stage2_m)
                            costs[method_name]["n_gen_tokens"].append(n_gen)
                            prefill_wall_per_q = info["prefill_wall_sec"] / max(1, len(qa_pairs))
                            costs[method_name]["shared_setup_sec"].append(shared_setup_per_q)
                            costs[method_name]["shared_question_sec"].append(shared_question_sec)
                            costs[method_name]["policy_prefill_wall_sec"].append(prefill_wall_per_q)
                            costs[method_name]["policy_decode_wall_sec"].append(policy_decode_sec)
                            costs[method_name]["policy_total_wall_sec"].append(
                                shared_setup_per_q + shared_question_sec + prefill_wall_per_q + policy_decode_sec)
                    else:
                        policy_start = time.perf_counter()
                        gen_ids = _per_question_prune_generate(
                            lm, model.lm_head, h_shared, shared_cache,
                            question_embeds, vis_pos_in_s1, n_vis_stage1,
                            S, STAGE2_FRAC, _eos_id, max_new_tokens=32,
                            scoring=cfg.get("scoring", "single"))
                        pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                        policy_decode_sec = time.perf_counter() - policy_start
                        results["ours"].append(score_gqa(pred, gt_answer))
                except Exception as e:
                    import traceback; traceback.print_exc()
                    target_methods = [m["output_name"] for m in policy_methods] if policy_methods else ["ours"]
                    for method_name in target_methods:
                        results[method_name].append(0.0)
                        if method_name in costs:
                            for ck in costs[method_name]:
                                costs[method_name][ck].append(0)
            else:
                target_methods = [m["output_name"] for m in policy_methods] if policy_methods else ["ours"]
                for method_name in target_methods:
                    results[method_name].append(0.0)
                    if method_name in costs:
                        for ck in costs[method_name]:
                            costs[method_name][ck].append(0)

            if cfg.get("policy_only", False):
                del full_embeds
                continue

            # ── PACT (KV cache reuse) ──
            if pact_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       pact_kept, n_pact_pfx)
                    # Position IDs: prefix used 0..n_pact_pfx-1, question starts at n_pact_pfx
                    q_pos = torch.arange(n_pact_pfx, n_pact_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _generate_from_shared_cache(
                        lm, model.lm_head, pact_cache, None,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["pact"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["pact"].append(0.0)
            else:
                results["pact"].append(0.0)

            # ── SparseVILA (KV cache reuse + per-question decode pruning) ──
            if sv_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       sv_kept, n_sv_pfx)
                    q_pos = torch.arange(n_sv_pfx, n_sv_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _sparsevila_per_question_generate(
                        lm, model.lm_head, sv_cache,
                        sv_vis_in_pfx, n_sv_vis, SPARSEVILA_DECODE_RATIO,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["sparsevila"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["sparsevila"].append(0.0)
            else:
                results["sparsevila"].append(0.0)

            # ── Attn top-k (KV cache reuse) ──
            if attn_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       attn_kept, n_attn_pfx)
                    q_pos = torch.arange(n_attn_pfx, n_attn_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _generate_from_shared_cache(
                        lm, model.lm_head, attn_cache, None,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["attn_topk"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["attn_topk"].append(0.0)
            else:
                results["attn_topk"].append(0.0)

            # ── SVDPrune (KV cache reuse) ──
            if svd_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       svd_kept, n_svd_pfx)
                    q_pos = torch.arange(n_svd_pfx, n_svd_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _generate_from_shared_cache(
                        lm, model.lm_head, svd_cache, None,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["svdprune"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["svdprune"].append(0.0)
            else:
                results["svdprune"].append(0.0)

            # ── DivPrune (KV cache reuse) ──
            if div_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       div_kept, n_div_pfx)
                    q_pos = torch.arange(n_div_pfx, n_div_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _generate_from_shared_cache(
                        lm, model.lm_head, div_cache, None,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["divprune"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["divprune"].append(0.0)
            else:
                results["divprune"].append(0.0)

            # ── VisPruner (KV cache reuse) ──
            if visp_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       visp_kept, n_visp_pfx)
                    q_pos = torch.arange(n_visp_pfx, n_visp_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _generate_from_shared_cache(
                        lm, model.lm_head, visp_cache, None,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["vispruner"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["vispruner"].append(0.0)
            else:
                results["vispruner"].append(0.0)

            # ── FastV (KV cache reuse, L2 last-text scoring) ──
            if fastv_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       fastv_kept, n_fastv_pfx)
                    q_pos = torch.arange(n_fastv_pfx, n_fastv_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _generate_from_shared_cache(
                        lm, model.lm_head, fastv_cache, None,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["fastv"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["fastv"].append(0.0)
            else:
                results["fastv"].append(0.0)

            # ── ZSPAPrune (KV cache reuse) ──
            if zspap_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       zspap_kept, n_zspap_pfx)
                    q_pos = torch.arange(n_zspap_pfx, n_zspap_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _generate_from_shared_cache(
                        lm, model.lm_head, zspap_cache, None,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["zspaprune"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["zspaprune"].append(0.0)
            else:
                results["zspaprune"].append(0.0)

            # ── AgilePruner (KV cache reuse) ──
            if agile_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       agile_kept, n_agile_pfx)
                    q_pos = torch.arange(n_agile_pfx, n_agile_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _generate_from_shared_cache(
                        lm, model.lm_head, agile_cache, None,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["agilepruner"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["agilepruner"].append(0.0)
            else:
                results["agilepruner"].append(0.0)

            # ── IDSelection (KV cache reuse) ──
            if idsel_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       idsel_kept, n_idsel_pfx)
                    q_pos = torch.arange(n_idsel_pfx, n_idsel_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _generate_from_shared_cache(
                        lm, model.lm_head, idsel_cache, None,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["idselection"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["idselection"].append(0.0)
            else:
                results["idselection"].append(0.0)

            # ── D²Pruner (KV cache reuse) ──
            if d2p_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       d2p_kept, n_d2p_pfx)
                    q_pos = torch.arange(n_d2p_pfx, n_d2p_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _generate_from_shared_cache(
                        lm, model.lm_head, d2p_cache, None,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["d2pruner"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["d2pruner"].append(0.0)
            else:
                results["d2pruner"].append(0.0)

            # ── PTP (KV cache reuse) ──
            if ptp_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       ptp_kept, n_ptp_pfx)
                    q_pos = torch.arange(n_ptp_pfx, n_ptp_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _generate_from_shared_cache(
                        lm, model.lm_head, ptp_cache, None,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["ptp"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["ptp"].append(0.0)
            else:
                results["ptp"].append(0.0)

            # ── HAWK (KV cache reuse) ──
            if hawk_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       hawk_kept, n_hawk_pfx)
                    q_pos = torch.arange(n_hawk_pfx, n_hawk_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _generate_from_shared_cache(
                        lm, model.lm_head, hawk_cache, None,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["hawk"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["hawk"].append(0.0)
            else:
                results["hawk"].append(0.0)

            # ── VScoreL2 (KV cache reuse) ──
            if vsl2_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       vsl2_kept, n_vsl2_pfx)
                    q_pos = torch.arange(n_vsl2_pfx, n_vsl2_pfx + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _generate_from_shared_cache(
                        lm, model.lm_head, vsl2_cache, None,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["vscore_l2"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["vscore_l2"].append(0.0)
            else:
                results["vscore_l2"].append(0.0)

            # ── FitPrune (progressive cache) ──
            if fitprune_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       vis_pos_set_0, _n_full_pfx_l)
                    q_pos = torch.arange(_n_full_pfx_l, _n_full_pfx_l + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _gen_from_progressive_cache(
                        lm, model.lm_head, fitprune_cache, _n_full_pfx_l,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["fitprune"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["fitprune"].append(0.0)
            else:
                results["fitprune"].append(0.0)

            # ── PyramidDrop (progressive cache) ──
            if pyrdrop_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       vis_pos_set_0, _n_full_pfx_l)
                    q_pos = torch.arange(_n_full_pfx_l, _n_full_pfx_l + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _gen_from_progressive_cache(
                        lm, model.lm_head, pyrdrop_cache, _n_full_pfx_l,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["pyramiddrop"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["pyramiddrop"].append(0.0)
            else:
                results["pyramiddrop"].append(0.0)

            # ── SparseVLM (progressive cache) ──
            if spvlm_ok:
                try:
                    q_emb = _get_llava_question_embeds(full_embeds, input_ids,
                                                       vis_pos_set_0, _n_full_pfx_l)
                    q_pos = torch.arange(_n_full_pfx_l, _n_full_pfx_l + q_emb.shape[1],
                                         device=q_emb.device).unsqueeze(0)
                    gen_ids = _gen_from_progressive_cache(
                        lm, model.lm_head, spvlm_cache, _n_full_pfx_l,
                        q_emb, q_pos, _eos_id, max_new_tokens=32)
                    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
                    results["sparsevlm"].append(score_gqa(pred, gt_answer))
                except Exception:
                    results["sparsevlm"].append(0.0)
            else:
                results["sparsevlm"].append(0.0)

            del full_embeds

        if shared_ok:
            if policy_methods:
                for info in policy_shared.values():
                    del info["h"], info["cache"]
            else:
                del h_shared, shared_cache
        if pact_ok:
            del pact_cache
        if sv_ok:
            del sv_cache
        if attn_ok:
            del attn_cache
        if svd_ok:
            del svd_cache
        if div_ok:
            del div_cache
        if visp_ok:
            del visp_cache
        if fastv_ok:
            del fastv_cache
        if zspap_ok:
            del zspap_cache
        if agile_ok:
            del agile_cache
        if idsel_ok:
            del idsel_cache
        if d2p_ok:
            del d2p_cache
        if ptp_ok:
            del ptp_cache
        if hawk_ok:
            del hawk_cache
        if vsl2_ok:
            del vsl2_cache
        if fitprune_ok:
            del fitprune_cache
        if pyrdrop_ok:
            del pyrdrop_cache
        if spvlm_ok:
            del spvlm_cache
        torch.cuda.empty_cache()
        n_q = len(qa_pairs)
        if cfg.get("policy_only", False):
            policy_labels = [m["output_name"] for m in cfg.get("policy_methods", [])]
            if not policy_labels:
                policy_labels = [cfg.get("policy_name", "policy")]
            policy_summary = " ".join(
                f"{label}={np.mean(results[label][-n_q:]):.3f}"
                for label in policy_labels)
            cumul_summary = " ".join(
                f"{label}={np.mean(results[label]):.3f}"
                for label in policy_labels)
            print(f"  [img {img_idx+1}/{len(gqa_samples)}] {n_q}Q "
                  f"bl={np.mean(results['baseline'][-n_q:]):.3f} "
                  f"{policy_summary} | cumul: {cumul_summary}", flush=True)
        else:
            print(f"  [img {img_idx+1}/{len(gqa_samples)}] {n_q}Q "
                  f"bl={np.mean(results['baseline'][-n_q:]):.3f} "
                  f"ours={np.mean(results['ours'][-n_q:]):.3f} "
                  f"svd={np.mean(results['svdprune'][-n_q:]):.3f} "
                  f"div={np.mean(results['divprune'][-n_q:]):.3f} "
                  f"| cumul: ours={np.mean(results['ours']):.3f}", flush=True)

    del model, processor
    gc.collect(); torch.cuda.empty_cache()
    return results, costs


def run_gqa_internvl3(model_name, gqa_samples, cfg, out_dir):
    """Multi-round GQA benchmark for InternVL3-hf models."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = cfg["model_id"]
    print(f"[InternVL3-hf] Loading {model_id}...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    proc = AutoProcessor.from_pretrained(model_id)

    lm = model.model.language_model
    img_token_id = getattr(model.config, 'image_token_id', None)
    if img_token_id is None:
        img_token_id = proc.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    eos_id = proc.tokenizer.eos_token_id
    S, Y = cfg["S"], cfg["Y"]
    N = cfg["N"]

    ALL_METHODS = ["baseline", "ours", "pact", "sparsevila", "attn_topk",
                   "svdprune", "divprune", "vispruner", "fastv",
                   "zspaprune", "agilepruner", "idselection", "d2pruner", "ptp",
                   "hawk", "vscore_l2",
                   "fitprune", "pyramiddrop", "sparsevlm"]
    results = {m: [] for m in ALL_METHODS}
    policy_methods = cfg.get("policy_methods") or []
    for method in policy_methods:
        results.setdefault(method["output_name"], [])
    costs = {m: _new_cost_bucket()
             for m in ["baseline"] + [method["output_name"] for method in policy_methods]}

    def _greedy_decode(pruned_embeds, max_new_tokens=32):
        generated = []
        cur_embeds = pruned_embeds
        past_kv = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = lm(inputs_embeds=cur_embeds, past_key_values=past_kv, use_cache=True)
                logits = model.lm_head(out.last_hidden_state[:, -1:, :])
                next_id = logits.argmax(dim=-1).squeeze()
                generated.append(next_id.item())
                if next_id.item() == eos_id:
                    break
                past_kv = out.past_key_values
                cur_embeds = lm.embed_tokens(
                    next_id.unsqueeze(0).unsqueeze(0)).to(pruned_embeds.dtype)
        return proc.decode(generated, skip_special_tokens=True).strip()

    def _gen_with_mask(full_embeds, input_ids, vis_pos, vis_pos_set, kept_idx_set):
        kept_vis_pos = {vis_pos[i] for i in kept_idx_set}
        all_pos = [p for p in range(input_ids.shape[1])
                   if p not in vis_pos_set or p in kept_vis_pos]
        return _greedy_decode(full_embeds[:, all_pos, :])

    for img_idx, (img, qa_pairs) in enumerate(gqa_samples):
        img = _dualsignal_resize_image(img)
        image_shared_start = time.perf_counter()
        # ── Build vision features and shared token sets ──
        try:
            messages0 = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": qa_pairs[0][0] + GQA_SUFFIX}]}]
            text_prompt0 = proc.apply_chat_template(messages0, add_generation_prompt=True)
            proc_inputs0 = proc(text=[text_prompt0], images=[img],
                                return_tensors="pt").to(model.device)

            input_ids_0 = proc_inputs0["input_ids"]
            pixel_values = proc_inputs0.get("pixel_values")
            with torch.no_grad():
                vis_out = model.model.get_image_features(pixel_values.to(model.dtype))
                vis_features = vis_out.pooler_output
                dtype = lm.layers[0].self_attn.q_proj.weight.dtype
                text_emb_0 = lm.embed_tokens(input_ids_0).to(dtype)
                full_embeds_0 = text_emb_0.clone()
                img_positions_0 = (input_ids_0[0] == img_token_id).nonzero(as_tuple=True)[0]
                n_vis = len(img_positions_0)
                vis_flat = vis_features.reshape(-1, vis_features.shape[-1]).to(dtype)
                n_insert = min(n_vis, vis_flat.shape[0])
                full_embeds_0[0, img_positions_0[:n_insert]] = vis_flat[:n_insert]
                vis_embeds = vis_flat[:n_vis]

            vis_pos_0 = img_positions_0.tolist()
            vis_pos_set_0 = set(vis_pos_0)
            text_pos_0 = [p for p in range(input_ids_0.shape[1])
                          if p not in vis_pos_set_0 and p > max(vis_pos_0)]

            k_y = max(1, int(n_vis * Y))
            k1 = max(1, int(n_vis * STAGE1_FRAC))

            # ViT importance (used by both Stage 1 and SparseVILA)
            try:
                vit_imp = _internvl3_vit_importance(model, pixel_values, n_vis)
                torch.cuda.empty_cache()
            except Exception:
                vit_imp = None

            # Stage 1 shared: dual signal + MMR
            imp0_shared = _get_importance_at_layer(lm, 0, full_embeds_0, vis_pos_0, text_pos_0)
            if vit_imp is not None:
                stage1_idx = _dual_signal_stage1_select(
                    imp0_shared, vit_imp, vis_flat[:n_vis], k1)
            else:
                stage1_idx = np.argsort(imp0_shared)[::-1][:k1].tolist()
            stage1_set = set(stage1_idx)

            # PACT shared
            pact_idx_set = set(_pact_select(lm, full_embeds_0, vis_pos_0, k_y))

            # SparseVILA shared
            if vit_imp is not None:
                sv_idx_set = set(np.argsort(vit_imp)[::-1][:k_y].tolist())
            else:
                sv_idx_set = None

            # Attn top-k shared
            attn_idx_set = set(np.argsort(imp0_shared)[::-1][:k_y].tolist())

            # ── NEW: SVDPrune (query-agnostic) ──
            try:
                svd_mask_l = _svdprune_select(vis_flat[:n_vis], k_y)
                svd_idx_set = set(i for i, k in enumerate(svd_mask_l) if k)
            except Exception:
                svd_idx_set = None

            # ── NEW: DivPrune (query-agnostic) ──
            try:
                div_mask_l = _divprune_select(vis_flat[:n_vis], k_y)
                div_idx_set = set(i for i, k in enumerate(div_mask_l) if k)
            except Exception:
                div_idx_set = None

            # ── NEW: VisPruner (query-agnostic, ViT + diversity) ──
            if vit_imp is not None:
                try:
                    visp_mask_l = _vispruner_select(vit_imp, vis_flat[:n_vis], k_y)
                    visp_idx_set = set(i for i, k in enumerate(visp_mask_l) if k)
                except Exception:
                    visp_idx_set = None
            else:
                visp_idx_set = None

            # ── NEW: FastV (L2 last-text attention, computed once) ──
            try:
                fastv_mask_l = _fastv_select(lm, full_embeds_0, vis_pos_0, text_pos_0, k_y, layer=2)
                fastv_idx_set = set(i for i, k in enumerate(fastv_mask_l) if k)
            except Exception:
                fastv_idx_set = None

            # ── B-class: ZSPAPrune ──
            try:
                zspap_mask_l = _zspaprune_select(lm, full_embeds_0, vis_pos_0, text_pos_0,
                                                  vis_embeds, k_y)
                zspap_idx_set = set(i for i, k in enumerate(zspap_mask_l) if k)
            except Exception:
                zspap_idx_set = None

            # ── B-class: AgilePruner ──
            try:
                agile_mask_l = _agilepruner_select(lm, full_embeds_0, vis_pos_0, text_pos_0,
                                                    vis_embeds, k_y)
                agile_idx_set = set(i for i, k in enumerate(agile_mask_l) if k)
            except Exception:
                agile_idx_set = None

            # ── B-class: IDSelection ──
            try:
                idsel_mask_l = _idselection_select(lm, full_embeds_0, vis_pos_0, text_pos_0,
                                                    vis_embeds, k_y)
                idsel_idx_set = set(i for i, k in enumerate(idsel_mask_l) if k)
            except Exception:
                idsel_idx_set = None

            # ── B-class: D²Pruner ──
            try:
                d2p_mask_l = _d2pruner_select(lm, full_embeds_0, vis_pos_0, text_pos_0,
                                               vis_embeds, k_y)
                d2p_idx_set = set(i for i, k in enumerate(d2p_mask_l) if k)
            except Exception:
                d2p_idx_set = None

            # ── B-class: PTP ──
            try:
                ptp_mask_l = _ptp_select(vit_imp, lm, full_embeds_0, vis_pos_0, text_pos_0,
                                          vis_embeds, k_y)
                ptp_idx_set = set(i for i, k in enumerate(ptp_mask_l) if k)
            except Exception:
                ptp_idx_set = None

            # ── B-class: HAWK ──
            try:
                hawk_mask_l = _hawk_select(lm, full_embeds_0, vis_pos_0, text_pos_0, k_y)
                hawk_idx_set = set(i for i, k in enumerate(hawk_mask_l) if k)
            except Exception:
                hawk_idx_set = None

            # ── B-class: VScoreL2 ──
            try:
                vsl2_mask_l = _vscore_l2_select(lm, full_embeds_0, vis_pos_0, text_pos_0, k_y)
                vsl2_idx_set = set(i for i, k in enumerate(vsl2_mask_l) if k)
            except Exception:
                vsl2_idx_set = None

            del full_embeds_0

            # ── Helper: build prefix embeddings from a kept-idx set ──
            last_vis_orig = max(vis_pos_0)

            def _build_iv3_prefix(kept_idx_set):
                kept_abs = {vis_pos_0[i] for i in kept_idx_set}
                pfx_pos = [p for p in range(input_ids_0.shape[1])
                           if (p not in vis_pos_set_0 or p in kept_abs) and p <= last_vis_orig]
                text_emb_tmp = lm.embed_tokens(input_ids_0).to(dtype)
                full_tmp = text_emb_tmp.clone()
                full_tmp[0, img_positions_0[:n_insert]] = vis_flat[:n_insert]
                pfx_emb = full_tmp[:, pfx_pos, :]
                del text_emb_tmp, full_tmp
                vis_in_pfx = [j for j, p in enumerate(pfx_pos) if p in kept_abs]
                return pfx_emb, pfx_pos, vis_in_pfx

            def _get_iv3_question_embeds(full_embeds, input_ids, kept_idx_set, n_pfx):
                vis_pos_q = (input_ids[0] == img_token_id).nonzero(as_tuple=True)[0].tolist()
                vis_pos_set_q = set(vis_pos_q)
                kept_abs = {vis_pos_q[i] for i in kept_idx_set}
                all_kept = [p for p in range(input_ids.shape[1])
                            if p not in vis_pos_set_q or p in kept_abs]
                q_emb = full_embeds[:, all_kept[n_pfx:], :]
                return q_emb

            # ── Prefill shared prefix L0..S-1 for OUR method (ONCE) ──
            policy_shared = {}
            image_shared_sec = time.perf_counter() - image_shared_start
            stage1_vis_pos_set_0 = {vis_pos_0[i] for i in stage1_set}
            ours_pfx_emb, ours_pfx_pos, vis_pos_in_s1 = \
                _build_iv3_prefix(stage1_set)
            n_prefix = len(ours_pfx_pos)
            n_vis_stage1 = len(vis_pos_in_s1)
            if policy_methods:
                for method in policy_methods:
                    method_S = int(method["S"])
                    prefill_start = time.perf_counter()
                    h_m, cache_m = _prefill_shared_layers(lm, ours_pfx_emb, method_S)
                    policy_shared[method["output_name"]] = {
                        "h": h_m,
                        "cache": cache_m,
                        "S": method_S,
                        "prefill_wall_sec": time.perf_counter() - prefill_start,
                    }
                shared_ok = bool(policy_shared)
            else:
                prefill_start = time.perf_counter()
                h_shared, shared_cache = _prefill_shared_layers(lm, ours_pfx_emb, S)
                policy_prefill_wall_sec = time.perf_counter() - prefill_start
                shared_ok = True
            del ours_pfx_emb

            # ── Prefill ALL N layers for PACT (ONCE) ──
            try:
                pact_pfx_emb, pact_pfx_pos, _ = _build_iv3_prefix(pact_idx_set)
                n_pact_pfx = len(pact_pfx_pos)
                pact_cache, _ = _prefill_all_layers(lm, pact_pfx_emb)
                pact_ok = True
                del pact_pfx_emb
            except Exception:
                pact_ok = False

            # ── Prefill ALL N layers for SparseVILA (ONCE per image) ──
            # Prefill is query-independent; decode pruning is per-question
            # using query_Q @ cached_K salience.
            sv_ok = False
            if sv_idx_set is not None:
                try:
                    sv_pfx_emb, sv_pfx_pos, sv_vis_in_pfx = _build_iv3_prefix(sv_idx_set)
                    n_sv_pfx = len(sv_pfx_pos)
                    n_sv_vis = len(sv_vis_in_pfx)
                    n_decode_keep = max(1, int(n_sv_vis * SPARSEVILA_DECODE_RATIO))
                    sv_cache, _ = _prefill_all_layers(lm, sv_pfx_emb)
                    del sv_pfx_emb
                    sv_ok = True
                except Exception:
                    import traceback; traceback.print_exc()

            # ── Prefill ALL N layers for Attn top-k (ONCE) ──
            try:
                attn_pfx_emb, attn_pfx_pos, _ = _build_iv3_prefix(attn_idx_set)
                n_attn_pfx = len(attn_pfx_pos)
                attn_cache, _ = _prefill_all_layers(lm, attn_pfx_emb)
                attn_ok = True
                del attn_pfx_emb
            except Exception:
                attn_ok = False

            # ── Prefill ALL N layers for SVDPrune (ONCE) ──
            svd_ok = False
            if svd_idx_set is not None:
                try:
                    svd_pfx_emb, svd_pfx_pos, _ = _build_iv3_prefix(svd_idx_set)
                    n_svd_pfx = len(svd_pfx_pos)
                    svd_cache, _ = _prefill_all_layers(lm, svd_pfx_emb)
                    svd_ok = True
                    del svd_pfx_emb
                except Exception:
                    pass

            # ── Prefill ALL N layers for DivPrune (ONCE) ──
            div_ok = False
            if div_idx_set is not None:
                try:
                    div_pfx_emb, div_pfx_pos, _ = _build_iv3_prefix(div_idx_set)
                    n_div_pfx = len(div_pfx_pos)
                    div_cache, _ = _prefill_all_layers(lm, div_pfx_emb)
                    div_ok = True
                    del div_pfx_emb
                except Exception:
                    pass

            # ── Prefill ALL N layers for VisPruner (ONCE) ──
            visp_ok = False
            if visp_idx_set is not None:
                try:
                    visp_pfx_emb, visp_pfx_pos, _ = _build_iv3_prefix(visp_idx_set)
                    n_visp_pfx = len(visp_pfx_pos)
                    visp_cache, _ = _prefill_all_layers(lm, visp_pfx_emb)
                    visp_ok = True
                    del visp_pfx_emb
                except Exception:
                    pass

            # ── Prefill ALL N layers for FastV (ONCE) ──
            fastv_ok = False
            if fastv_idx_set is not None:
                try:
                    fastv_pfx_emb, fastv_pfx_pos, _ = _build_iv3_prefix(fastv_idx_set)
                    n_fastv_pfx = len(fastv_pfx_pos)
                    fastv_cache, _ = _prefill_all_layers(lm, fastv_pfx_emb)
                    fastv_ok = True
                    del fastv_pfx_emb
                except Exception:
                    pass

            # ── Prefill ALL N layers for ZSPAPrune (ONCE) ──
            zspap_ok = False
            if zspap_idx_set is not None:
                try:
                    zspap_pfx_emb, zspap_pfx_pos, _ = _build_iv3_prefix(zspap_idx_set)
                    n_zspap_pfx = len(zspap_pfx_pos)
                    zspap_cache, _ = _prefill_all_layers(lm, zspap_pfx_emb)
                    zspap_ok = True
                    del zspap_pfx_emb
                except Exception:
                    pass

            # ── Prefill ALL N layers for AgilePruner (ONCE) ──
            agile_ok = False
            if agile_idx_set is not None:
                try:
                    agile_pfx_emb, agile_pfx_pos, _ = _build_iv3_prefix(agile_idx_set)
                    n_agile_pfx = len(agile_pfx_pos)
                    agile_cache, _ = _prefill_all_layers(lm, agile_pfx_emb)
                    agile_ok = True
                    del agile_pfx_emb
                except Exception:
                    pass

            # ── Prefill ALL N layers for IDSelection (ONCE) ──
            idsel_ok = False
            if idsel_idx_set is not None:
                try:
                    idsel_pfx_emb, idsel_pfx_pos, _ = _build_iv3_prefix(idsel_idx_set)
                    n_idsel_pfx = len(idsel_pfx_pos)
                    idsel_cache, _ = _prefill_all_layers(lm, idsel_pfx_emb)
                    idsel_ok = True
                    del idsel_pfx_emb
                except Exception:
                    pass

            # ── Prefill ALL N layers for D²Pruner (ONCE) ──
            d2p_ok = False
            if d2p_idx_set is not None:
                try:
                    d2p_pfx_emb, d2p_pfx_pos, _ = _build_iv3_prefix(d2p_idx_set)
                    n_d2p_pfx = len(d2p_pfx_pos)
                    d2p_cache, _ = _prefill_all_layers(lm, d2p_pfx_emb)
                    d2p_ok = True
                    del d2p_pfx_emb
                except Exception:
                    pass

            # ── Prefill ALL N layers for PTP (ONCE) ──
            ptp_ok = False
            if ptp_idx_set is not None:
                try:
                    ptp_pfx_emb, ptp_pfx_pos, _ = _build_iv3_prefix(ptp_idx_set)
                    n_ptp_pfx = len(ptp_pfx_pos)
                    ptp_cache, _ = _prefill_all_layers(lm, ptp_pfx_emb)
                    ptp_ok = True
                    del ptp_pfx_emb
                except Exception:
                    pass

            # ── Prefill ALL N layers for HAWK (ONCE) ──
            hawk_ok = False
            if hawk_idx_set is not None:
                try:
                    hawk_pfx_emb, hawk_pfx_pos, _ = _build_iv3_prefix(hawk_idx_set)
                    n_hawk_pfx = len(hawk_pfx_pos)
                    hawk_cache, _ = _prefill_all_layers(lm, hawk_pfx_emb)
                    hawk_ok = True
                    del hawk_pfx_emb
                except Exception:
                    pass

            # ── Prefill ALL N layers for VScoreL2 (ONCE) ──
            vsl2_ok = False
            if vsl2_idx_set is not None:
                try:
                    vsl2_pfx_emb, vsl2_pfx_pos, _ = _build_iv3_prefix(vsl2_idx_set)
                    n_vsl2_pfx = len(vsl2_pfx_pos)
                    vsl2_cache, _ = _prefill_all_layers(lm, vsl2_pfx_emb)
                    vsl2_ok = True
                    del vsl2_pfx_emb
                except Exception:
                    pass

            # ── C-class progressive prefill (selection + KV cache in one pass) ──
            _skip_cc = cfg.get("skip_cclass", False)
            fitprune_ok = pyrdrop_ok = spvlm_ok = False
            if not _skip_cc:
                N = len(lm.layers)
                _all_idx_set = set(range(n_vis))
                _full_pfx_emb_iv, _full_pfx_pos_iv, _full_vis_in_pfx_iv = \
                    _build_iv3_prefix(_all_idx_set)
                _full_vis_set_iv = set(_full_vis_in_pfx_iv)
                _n_full_pfx_iv = len(_full_pfx_pos_iv)

                try:
                    _fp_schedule = _fitprune_build_schedule(n_vis, k_y, N)
                    _fp_drop_fn = _make_fitprune_drop_fn(_fp_schedule)
                    fitprune_cache, _, fitprune_alive, fitprune_tl = \
                        _progressive_prefill(lm, _full_pfx_emb_iv, set(_full_vis_set_iv),
                                             _fp_drop_fn)
                    fitprune_ok = True
                except Exception:
                    import traceback; traceback.print_exc()

                try:
                    _pd_drop_fn = _make_pyramiddrop_drop_fn(n_vis, k_y, N)
                    pyrdrop_cache, _, pyrdrop_alive, pyrdrop_tl = \
                        _progressive_prefill(lm, _full_pfx_emb_iv, set(_full_vis_set_iv),
                                             _pd_drop_fn)
                    pyrdrop_ok = True
                except Exception:
                    import traceback; traceback.print_exc()

                try:
                    _sv_drop_fn = _make_sparsevlm_drop_fn(n_vis, k_y, N)
                    spvlm_cache, _, spvlm_alive, spvlm_tl = \
                        _progressive_prefill(lm, _full_pfx_emb_iv, set(_full_vis_set_iv),
                                             _sv_drop_fn)
                    spvlm_ok = True
                except Exception:
                    import traceback; traceback.print_exc()

                del _full_pfx_emb_iv

            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [img {img_idx+1}] setup ERR: {e}", flush=True)
            import traceback; traceback.print_exc()
            for _ in qa_pairs:
                for k in results:
                    results[k].append(0.0)
            continue

        # ── Per-question ──
        shared_setup_per_q = image_shared_sec / max(1, len(qa_pairs))
        for q_idx, (question_raw, gt_answer) in enumerate(qa_pairs):
            shared_question_start = time.perf_counter()
            question = question_raw + GQA_SUFFIX
            try:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question}]}]
                text_prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
                proc_inputs = proc(text=[text_prompt], images=[img],
                                   return_tensors="pt").to(model.device)
                input_ids = proc_inputs["input_ids"]

                baseline_scores = cfg.get("baseline_scores")
                baseline_idx = len(results["baseline"])
                with torch.no_grad():
                    text_emb = lm.embed_tokens(input_ids).to(dtype)
                    full_embeds = text_emb.clone()
                    img_positions = (input_ids[0] == img_token_id).nonzero(as_tuple=True)[0]
                    full_embeds[0, img_positions[:n_insert]] = vis_flat[:n_insert]

                vis_pos = img_positions.tolist()
                vis_pos_set = set(vis_pos)

                if baseline_scores is not None and baseline_idx < len(baseline_scores):
                    results["baseline"].append(float(baseline_scores[baseline_idx]))
                    bl_gen_len = 0
                else:
                    with torch.no_grad():
                        out_ids = model.generate(**proc_inputs, max_new_tokens=32, do_sample=False)
                    bl_pred = proc.decode(out_ids[0][input_ids.shape[1]:],
                                          skip_special_tokens=True).strip()
                    results["baseline"].append(score_gqa(bl_pred, gt_answer))
                    bl_gen_len = _decoded_token_count(proc.tokenizer, bl_pred)
                costs["baseline"]["prefill_tl"].append(n_vis * N)
                costs["baseline"]["gen_tl"].append(n_vis * N * bl_gen_len)
                costs["baseline"]["total_tl"].append(n_vis * N * (1 + bl_gen_len))
                costs["baseline"]["n_vis_decode"].append(n_vis)
                costs["baseline"]["n_gen_tokens"].append(bl_gen_len)
                costs["baseline"]["shared_setup_sec"].append(shared_setup_per_q)
                costs["baseline"]["shared_question_sec"].append(time.perf_counter() - shared_question_start)
                costs["baseline"]["policy_prefill_wall_sec"].append(0.0)
                costs["baseline"]["policy_decode_wall_sec"].append(0.0)
                costs["baseline"]["policy_total_wall_sec"].append(
                    costs["baseline"]["shared_setup_sec"][-1] + costs["baseline"]["shared_question_sec"][-1])

                # ── Stage2 policies: shared Stage1 prefix + per-question pruning ──
                if shared_ok:
                    try:
                        stage1_vis_pos_set_q = {vis_pos[i] for i in stage1_set}
                        s1_all_pos_q = [p for p in range(input_ids.shape[1])
                                        if p not in vis_pos_set or p in stage1_vis_pos_set_q]
                        s1_embeds_q = full_embeds[:, s1_all_pos_q, :]
                        question_embeds = s1_embeds_q[:, n_prefix:, :]
                        shared_question_sec = time.perf_counter() - shared_question_start

                        if policy_methods:
                            globals()["_stage2_score_cache"] = {}
                            for method in policy_methods:
                                method_name = method["output_name"]
                                info = policy_shared[method_name]
                                globals()["_active_stage2_policy"] = method["policy"]
                                method_stage2_frac = float(method.get("stage2_frac", STAGE2_FRAC))
                                globals()["_active_stage2_frac"] = method_stage2_frac
                                policy_start = time.perf_counter()
                                gen_ids = _per_question_prune_generate(
                                    lm, model.lm_head, info["h"], info["cache"],
                                    question_embeds, vis_pos_in_s1, n_vis_stage1,
                                    info["S"], method_stage2_frac, eos_id, max_new_tokens=32,
                                scoring="policy")
                                pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                                policy_decode_sec = time.perf_counter() - policy_start
                                results[method_name].append(score_gqa(pred, gt_answer))
                                n_gen = _decoded_token_count(proc.tokenizer, pred)
                                n_vis_stage2_m = max(1, int(n_vis_stage1 * (method_stage2_frac / STAGE1_FRAC)))
                                pfx_tl = n_vis_stage1 * info["S"] + n_vis_stage2_m * (N - info["S"])
                                gen_tl = n_vis_stage2_m * N * n_gen
                                costs[method_name]["prefill_tl"].append(pfx_tl)
                                costs[method_name]["gen_tl"].append(gen_tl)
                                costs[method_name]["total_tl"].append(pfx_tl + gen_tl)
                                costs[method_name]["n_vis_decode"].append(n_vis_stage2_m)
                                costs[method_name]["n_gen_tokens"].append(n_gen)
                                prefill_wall_per_q = info["prefill_wall_sec"] / max(1, len(qa_pairs))
                                costs[method_name]["shared_setup_sec"].append(shared_setup_per_q)
                                costs[method_name]["shared_question_sec"].append(shared_question_sec)
                                costs[method_name]["policy_prefill_wall_sec"].append(prefill_wall_per_q)
                                costs[method_name]["policy_decode_wall_sec"].append(policy_decode_sec)
                                costs[method_name]["policy_total_wall_sec"].append(
                                    shared_setup_per_q + shared_question_sec + prefill_wall_per_q + policy_decode_sec)
                        else:
                            policy_start = time.perf_counter()
                            gen_ids = _per_question_prune_generate(
                                lm, model.lm_head, h_shared, shared_cache,
                                question_embeds, vis_pos_in_s1, n_vis_stage1,
                                S, STAGE2_FRAC, eos_id, max_new_tokens=32,
                                scoring=cfg.get("scoring", "single"))
                            pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                            policy_decode_sec = time.perf_counter() - policy_start
                            results["ours"].append(score_gqa(pred, gt_answer))
                    except Exception as e:
                        if img_idx == 0 and q_idx == 0:
                            import traceback; traceback.print_exc()
                        target_methods = [m["output_name"] for m in policy_methods] if policy_methods else ["ours"]
                        for method_name in target_methods:
                            results[method_name].append(0.0)
                            if method_name in costs:
                                for ck in costs[method_name]:
                                    costs[method_name][ck].append(0)
                else:
                    target_methods = [m["output_name"] for m in policy_methods] if policy_methods else ["ours"]
                    for method_name in target_methods:
                        results[method_name].append(0.0)
                        if method_name in costs:
                            for ck in costs[method_name]:
                                costs[method_name][ck].append(0)

                if cfg.get("policy_only", False):
                    del full_embeds
                    continue

                # ── PACT (KV cache reuse) ──
                if pact_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          pact_idx_set, n_pact_pfx)
                        q_pos = torch.arange(n_pact_pfx, n_pact_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _generate_from_shared_cache(
                            lm, model.lm_head, pact_cache, None,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["pact"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["pact"].append(0.0)
                else:
                    results["pact"].append(0.0)

                # ── SparseVILA (KV cache reuse + per-question decode pruning) ──
                if sv_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          sv_idx_set, n_sv_pfx)
                        q_pos = torch.arange(n_sv_pfx, n_sv_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _sparsevila_per_question_generate(
                            lm, model.lm_head, sv_cache,
                            sv_vis_in_pfx, n_sv_vis, SPARSEVILA_DECODE_RATIO,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["sparsevila"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["sparsevila"].append(0.0)
                else:
                    results["sparsevila"].append(0.0)

                # ── Attn top-k (KV cache reuse) ──
                if attn_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          attn_idx_set, n_attn_pfx)
                        q_pos = torch.arange(n_attn_pfx, n_attn_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _generate_from_shared_cache(
                            lm, model.lm_head, attn_cache, None,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["attn_topk"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["attn_topk"].append(0.0)
                else:
                    results["attn_topk"].append(0.0)

                # ── SVDPrune (KV cache reuse) ──
                if svd_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          svd_idx_set, n_svd_pfx)
                        q_pos = torch.arange(n_svd_pfx, n_svd_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _generate_from_shared_cache(
                            lm, model.lm_head, svd_cache, None,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["svdprune"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["svdprune"].append(0.0)
                else:
                    results["svdprune"].append(0.0)

                # ── DivPrune (KV cache reuse) ──
                if div_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          div_idx_set, n_div_pfx)
                        q_pos = torch.arange(n_div_pfx, n_div_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _generate_from_shared_cache(
                            lm, model.lm_head, div_cache, None,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["divprune"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["divprune"].append(0.0)
                else:
                    results["divprune"].append(0.0)

                # ── VisPruner (KV cache reuse) ──
                if visp_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          visp_idx_set, n_visp_pfx)
                        q_pos = torch.arange(n_visp_pfx, n_visp_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _generate_from_shared_cache(
                            lm, model.lm_head, visp_cache, None,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["vispruner"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["vispruner"].append(0.0)
                else:
                    results["vispruner"].append(0.0)

                # ── FastV (KV cache reuse, L2 last-text scoring) ──
                if fastv_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          fastv_idx_set, n_fastv_pfx)
                        q_pos = torch.arange(n_fastv_pfx, n_fastv_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _generate_from_shared_cache(
                            lm, model.lm_head, fastv_cache, None,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["fastv"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["fastv"].append(0.0)
                else:
                    results["fastv"].append(0.0)

                # ── ZSPAPrune (KV cache reuse) ──
                if zspap_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          zspap_idx_set, n_zspap_pfx)
                        q_pos = torch.arange(n_zspap_pfx, n_zspap_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _generate_from_shared_cache(
                            lm, model.lm_head, zspap_cache, None,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["zspaprune"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["zspaprune"].append(0.0)
                else:
                    results["zspaprune"].append(0.0)

                # ── AgilePruner (KV cache reuse) ──
                if agile_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          agile_idx_set, n_agile_pfx)
                        q_pos = torch.arange(n_agile_pfx, n_agile_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _generate_from_shared_cache(
                            lm, model.lm_head, agile_cache, None,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["agilepruner"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["agilepruner"].append(0.0)
                else:
                    results["agilepruner"].append(0.0)

                # ── IDSelection (KV cache reuse) ──
                if idsel_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          idsel_idx_set, n_idsel_pfx)
                        q_pos = torch.arange(n_idsel_pfx, n_idsel_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _generate_from_shared_cache(
                            lm, model.lm_head, idsel_cache, None,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["idselection"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["idselection"].append(0.0)
                else:
                    results["idselection"].append(0.0)

                # ── D²Pruner (KV cache reuse) ──
                if d2p_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          d2p_idx_set, n_d2p_pfx)
                        q_pos = torch.arange(n_d2p_pfx, n_d2p_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _generate_from_shared_cache(
                            lm, model.lm_head, d2p_cache, None,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["d2pruner"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["d2pruner"].append(0.0)
                else:
                    results["d2pruner"].append(0.0)

                # ── PTP (KV cache reuse) ──
                if ptp_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          ptp_idx_set, n_ptp_pfx)
                        q_pos = torch.arange(n_ptp_pfx, n_ptp_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _generate_from_shared_cache(
                            lm, model.lm_head, ptp_cache, None,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["ptp"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["ptp"].append(0.0)
                else:
                    results["ptp"].append(0.0)

                # ── HAWK (KV cache reuse) ──
                if hawk_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          hawk_idx_set, n_hawk_pfx)
                        q_pos = torch.arange(n_hawk_pfx, n_hawk_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _generate_from_shared_cache(
                            lm, model.lm_head, hawk_cache, None,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["hawk"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["hawk"].append(0.0)
                else:
                    results["hawk"].append(0.0)

                # ── VScoreL2 (KV cache reuse) ──
                if vsl2_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          vsl2_idx_set, n_vsl2_pfx)
                        q_pos = torch.arange(n_vsl2_pfx, n_vsl2_pfx + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _generate_from_shared_cache(
                            lm, model.lm_head, vsl2_cache, None,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["vscore_l2"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["vscore_l2"].append(0.0)
                else:
                    results["vscore_l2"].append(0.0)

                # ── FitPrune (progressive cache) ──
                if fitprune_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          _all_idx_set, _n_full_pfx_iv)
                        q_pos = torch.arange(_n_full_pfx_iv, _n_full_pfx_iv + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _gen_from_progressive_cache(
                            lm, model.lm_head, fitprune_cache, _n_full_pfx_iv,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["fitprune"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        import traceback; traceback.print_exc()
                        results["fitprune"].append(0.0)
                else:
                    results["fitprune"].append(0.0)

                # ── PyramidDrop (progressive cache) ──
                if pyrdrop_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          _all_idx_set, _n_full_pfx_iv)
                        q_pos = torch.arange(_n_full_pfx_iv, _n_full_pfx_iv + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _gen_from_progressive_cache(
                            lm, model.lm_head, pyrdrop_cache, _n_full_pfx_iv,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["pyramiddrop"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["pyramiddrop"].append(0.0)
                else:
                    results["pyramiddrop"].append(0.0)

                # ── SparseVLM (progressive cache) ──
                if spvlm_ok:
                    try:
                        q_emb = _get_iv3_question_embeds(full_embeds, input_ids,
                                                          _all_idx_set, _n_full_pfx_iv)
                        q_pos = torch.arange(_n_full_pfx_iv, _n_full_pfx_iv + q_emb.shape[1],
                                             device=q_emb.device).unsqueeze(0)
                        gen_ids = _gen_from_progressive_cache(
                            lm, model.lm_head, spvlm_cache, _n_full_pfx_iv,
                            q_emb, q_pos, eos_id, max_new_tokens=32)
                        pred = proc.decode(gen_ids, skip_special_tokens=True).strip()
                        results["sparsevlm"].append(score_gqa(pred, gt_answer))
                    except Exception:
                        results["sparsevlm"].append(0.0)
                else:
                    results["sparsevlm"].append(0.0)

                del full_embeds
            except Exception as e:
                for k in results:
                    if len(results[k]) < len(results["baseline"]):
                        results[k].append(0.0)

        if shared_ok:
            if policy_methods:
                for info in policy_shared.values():
                    del info["h"], info["cache"]
            else:
                del h_shared, shared_cache
        if pact_ok:
            del pact_cache
        if sv_ok:
            del sv_cache
        if attn_ok:
            del attn_cache
        if svd_ok:
            del svd_cache
        if div_ok:
            del div_cache
        if visp_ok:
            del visp_cache
        if fastv_ok:
            del fastv_cache
        if zspap_ok:
            del zspap_cache
        if agile_ok:
            del agile_cache
        if idsel_ok:
            del idsel_cache
        if d2p_ok:
            del d2p_cache
        if ptp_ok:
            del ptp_cache
        if hawk_ok:
            del hawk_cache
        if vsl2_ok:
            del vsl2_cache
        if fitprune_ok:
            del fitprune_cache
        if pyrdrop_ok:
            del pyrdrop_cache
        if spvlm_ok:
            del spvlm_cache
        torch.cuda.empty_cache()
        n_q = len(qa_pairs)
        if cfg.get("policy_only", False):
            policy_labels = [m["output_name"] for m in cfg.get("policy_methods", [])]
            if not policy_labels:
                policy_labels = [cfg.get("policy_name", "policy")]
            policy_summary = " ".join(
                f"{label}={np.mean(results[label][-n_q:]):.3f}"
                for label in policy_labels)
            cumul_summary = " ".join(
                f"{label}={np.mean(results[label]):.3f}"
                for label in policy_labels)
            print(f"  [img {img_idx+1}/{len(gqa_samples)}] {n_q}Q "
                  f"bl={np.mean(results['baseline'][-n_q:]):.3f} "
                  f"{policy_summary} | cumul: {cumul_summary}", flush=True)
        else:
            print(f"  [img {img_idx+1}/{len(gqa_samples)}] {n_q}Q "
                  f"bl={np.mean(results['baseline'][-n_q:]):.3f} "
                  f"ours={np.mean(results['ours'][-n_q:]):.3f} "
                  f"svd={np.mean(results['svdprune'][-n_q:]):.3f} "
                  f"div={np.mean(results['divprune'][-n_q:]):.3f} "
                  f"| cumul: ours={np.mean(results['ours']):.3f}", flush=True)

    del model, proc
    gc.collect(); torch.cuda.empty_cache()
    return results, costs


GQA_RUNNERS = {
    "qwen":      run_gqa_qwen,
    "llava":     run_gqa_llava,
    "internvl3": run_gqa_internvl3,
}


# ══════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════

def print_report(model_name, cfg, results, diversity=None, costs=None):
    bl_mean = np.mean(results["baseline"])
    N, S, Y = cfg["N"], cfg["S"], cfg["Y"]

    scoring = cfg.get("scoring", "single")
    methods = ["ours", "pact", "sparsevila", "attn_topk",
               "svdprune", "divprune", "vispruner", "fastv",
               "zspaprune", "agilepruner", "idselection", "d2pruner", "ptp",
               "hawk", "vscore_l2",
               "fitprune", "pyramiddrop", "sparsevlm"]
    # Only include methods that have results
    methods = [m for m in methods if m in results and len(results[m]) > 0]
    if scoring == "max":
        ours_label = f"Dual-stage (33%→10%@max_0_{S})"
    else:
        ours_label = f"Dual-stage (33%→10%@L{S})"
    method_names = {
        "ours": ours_label,
        "pact": f"PACT (Y={Y:.1%})",
        "sparsevila": f"SparseVILA (Y={Y:.1%})",
        "attn_topk": f"Attn-TopK-L0 (Y={Y:.1%})",
        "svdprune": f"SVDPrune (Y={Y:.1%})",
        "divprune": f"DivPrune (Y={Y:.1%})",
        "vispruner": f"VisPruner (Y={Y:.1%})",
        "fastv": f"FastV-L2 (Y={Y:.1%})",
        "zspaprune": f"ZSPAPrune (Y={Y:.1%})",
        "agilepruner": f"AgilePruner (Y={Y:.1%})",
        "idselection": f"IDSelection (Y={Y:.1%})",
        "d2pruner": f"D²Pruner (Y={Y:.1%})",
        "ptp": f"PTP (Y={Y:.1%})",
        "hawk": f"HAWK (Y={Y:.1%})",
        "vscore_l2": f"VScoreL2 (Y={Y:.1%})",
        "fitprune": f"FitPrune (Y={Y:.1%})",
        "pyramiddrop": f"PyramidDrop (Y={Y:.1%})",
        "sparsevlm": f"SparseVLM (Y={Y:.1%})",
    }

    print(f"\n{'='*70}")
    print(f"Results — {model_name} (N={N}, S={S}, scoring={scoring}, cost={cfg['cost']:.1f}, Y={Y:.1%})")
    print(f"  Baseline: {bl_mean:.4f}")
    print(f"{'='*70}")

    if costs:
        bl_total = np.mean(costs["baseline"]["total_tl"])
        print(f"{'Method':<35} {'Score':>8} {'Preserv':>8} {'PrefillTL':>10} "
              f"{'GenTL':>10} {'TotalTL':>10} {'Savings':>8}")
        print("-" * 95)
        for m in methods:
            sc = np.mean(results[m])
            pres = sc / bl_mean * 100 if bl_mean > 0 else 0
            pfx = np.mean(costs[m]["prefill_tl"])
            gen = np.mean(costs[m]["gen_tl"])
            tot = np.mean(costs[m]["total_tl"])
            sav = (1 - tot / bl_total) * 100 if bl_total > 0 else 0
            marker = " ★" if m == "ours" else ""
            print(f"  {method_names[m]:<33} {sc:>8.4f} {pres:>7.1f}% "
                  f"{pfx:>10.0f} {gen:>10.0f} {tot:>10.0f} {sav:>7.1f}%{marker}")

        # Cost detail summary
        print(f"\n  Cost breakdown (mean per question):")
        print(f"  {'Method':<25} {'VisDecod':>8} {'GenToks':>8} "
              f"{'GenTL/PfxTL':>12}")
        print(f"  {'-'*55}")
        for m in ["baseline"] + methods:
            nv = np.mean(costs[m]["n_vis_decode"])
            ng = np.mean(costs[m]["n_gen_tokens"])
            pfx = np.mean(costs[m]["prefill_tl"])
            gen = np.mean(costs[m]["gen_tl"])
            ratio = gen / pfx if pfx > 0 else 0
            print(f"  {m:<25} {nv:>8.1f} {ng:>8.1f} {ratio:>11.1f}×")
    else:
        print(f"{'Method':<35} {'Score':>8} {'Preserv':>10}")
        print("-" * 55)
        for m in methods:
            sc = np.mean(results[m])
            pres = sc / bl_mean * 100 if bl_mean > 0 else 0
            marker = " ★" if m == "ours" else ""
            print(f"  {method_names[m]:<33} {sc:>8.4f} {pres:>9.1f}%{marker}")
    print()

    if diversity:
        avg_iou = np.mean([d["avg_iou"] for d in diversity])
        print(f"Multi-round token diversity:")
        print(f"  Average IoU between Stage-2 selections = {avg_iou:.3f}")
        print(f"  (Lower IoU = more question-specific token selection)\n")


def save_results(model_name, cfg, results, diversity, out_dir, costs=None):
    os.makedirs(out_dir, exist_ok=True)
    bl_mean = np.mean(results["baseline"])
    data = {
        "model": model_name,
        "config": {k: v for k, v in cfg.items() if k != "type"},
        "n_samples": len(results["baseline"]),
        "baseline_mean": float(bl_mean),
        "methods": {},
    }
    all_methods = ["ours", "pact", "sparsevila", "attn_topk",
                   "svdprune", "divprune", "vispruner", "fastv",
                   "zspaprune", "agilepruner", "idselection", "d2pruner", "ptp",
                   "hawk", "vscore_l2",
                   "fitprune", "pyramiddrop", "sparsevlm"]
    for m in all_methods:
        if m not in results or len(results[m]) == 0:
            continue
        sc = float(np.mean(results[m]))
        entry = {
            "score": sc,
            "preservation": sc / bl_mean if bl_mean > 0 else 0,
            "per_sample": [float(x) for x in results[m]],
        }
        if costs and m in costs:
            entry["cost"] = {
                "prefill_tl_mean": float(np.mean(costs[m]["prefill_tl"])),
                "gen_tl_mean": float(np.mean(costs[m]["gen_tl"])),
                "total_tl_mean": float(np.mean(costs[m]["total_tl"])),
                "n_vis_decode_mean": float(np.mean(costs[m]["n_vis_decode"])),
                "n_gen_tokens_mean": float(np.mean(costs[m]["n_gen_tokens"])),
            }
        data["methods"][m] = entry
    if costs and "baseline" in costs:
        data["baseline_cost"] = {
            "total_tl_mean": float(np.mean(costs["baseline"]["total_tl"])),
            "n_vis_decode_mean": float(np.mean(costs["baseline"]["n_vis_decode"])),
            "n_gen_tokens_mean": float(np.mean(costs["baseline"]["n_gen_tokens"])),
        }
    if diversity:
        data["multi_round_diversity"] = diversity

    out_path = os.path.join(out_dir, f"{model_name}.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved → {out_path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

RUNNERS = {
    "qwen":      run_qwen,
    "llava":     run_llava,
    "internvl3": run_internvl3,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--num_images", type=int, default=30)
    MULTI_ROUND_BENCHMARKS = ["gqa", "pope", "docvqa", "vqav2", "clevr", "visual7w"]
    parser.add_argument("--benchmark", default="textvqa",
                        choices=["textvqa"] + MULTI_ROUND_BENCHMARKS,
                        help="textvqa=单问题, gqa/pope/docvqa/vqav2/clevr/visual7w=多轮多问题")
    parser.add_argument("--questions_per_image", type=int, default=5,
                        help="GQA: 每张图的问题数")
    parser.add_argument("--multi_round", action="store_true",
                        help="(textvqa only) 额外的多轮 token 多样性分析")
    parser.add_argument("--out_dir", default="strategy_test/results/multi_round_benchmark")
    parser.add_argument("--skip_cclass", action="store_true",
                        help="Skip C-class methods (FitPrune, PyramidDrop, SparseVLM)")
    args = parser.parse_args()

    cfg = dict(MODEL_CONFIG[args.model])
    cfg["skip_cclass"] = args.skip_cclass
    scoring = cfg.get("scoring", "single")
    scoring_label = f"max_0_{cfg['S']}" if scoring == "max" else f"single_L{cfg['S']}"
    print(f"Model: {args.model}")
    print(f"  N={cfg['N']} layers, scoring={scoring_label}, cost={cfg['cost']:.1f}, "
          f"Y={cfg['Y']:.1%}")
    print(f"  Stage1: {STAGE1_FRAC:.0%}, Stage2: {STAGE2_FRAC:.0%} of total")
    print(f"  Benchmark: {args.benchmark}")
    print(flush=True)

    MULTI_ROUND_LOADERS = {
        "gqa":      lambda: load_gqa_multi_round(args.num_images, args.questions_per_image),
        "pope":     lambda: load_pope_multi_round(args.num_images, args.questions_per_image),
        "docvqa":   lambda: load_docvqa_multi_round(args.num_images, args.questions_per_image),
        "vqav2":    lambda: load_vqav2_multi_round(args.num_images, args.questions_per_image),
        "clevr":    lambda: load_clevr_multi_round(args.num_images, args.questions_per_image),
        "visual7w": lambda: load_visual7w_multi_round(args.num_images, args.questions_per_image),
    }

    if args.benchmark in MULTI_ROUND_LOADERS:
        # ── Multi-round evaluation ──
        print(f"Loading {args.benchmark}...", flush=True)
        mr_samples = MULTI_ROUND_LOADERS[args.benchmark]()

        gqa_runner = GQA_RUNNERS[cfg["type"]]
        ret = gqa_runner(args.model, mr_samples, cfg, args.out_dir)
        if isinstance(ret, tuple):
            results, costs = ret
        else:
            results, costs = ret, None

        print_report(args.model, cfg, results, diversity=None, costs=costs)
        save_results(args.model, cfg, results, diversity=None,
                     out_dir=os.path.join(args.out_dir, args.benchmark), costs=costs)
    else:
        # ── TextVQA single-question evaluation ──
        print("Loading TextVQA...", flush=True)
        samples = load_textvqa_samples(args.num_images)
        print(f"  {len(samples)} samples\n", flush=True)

        runner = RUNNERS[cfg["type"]]
        results = runner(args.model, samples, cfg, args.out_dir)

        diversity = None
        if args.multi_round:
            images = [s[0] for s in samples[:min(10, len(samples))]]
            diversity = run_multi_round_analysis(args.model, images, cfg)

        print_report(args.model, cfg, results, diversity)
        save_results(args.model, cfg, results, diversity, args.out_dir)


if __name__ == "__main__":
    main()
