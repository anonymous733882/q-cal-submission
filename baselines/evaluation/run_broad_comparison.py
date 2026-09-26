"""Broad benchmark comparison: L20 vs 13 baselines + BEA.

Benchmarks (12 total):
  VQA:      RealWorldQA, TextVQA, ChartQA, DocVQA, GQA, OCRBench
  MC:       ScienceQA, AI2D, MMMU
  Yes/No:   POPE, MME
  Grounding: Grounding (COCO RefExp)
Methods (16): HAWK, FastV, L20, FasterVLM, VisPruner, DivPrune, SVD-Prune,
              PTP, ZSPAPrune, ID-Selection, D²Pruner, AgilePruner,
              PyramidDrop, FitPrune, SparseVLM, BEA
Budget levels: 5%, 10%, 20%, 33%

Optimizations:
  - Single LLM forward captures layers {0, 1, 2, 20} hidden states
  - Single ViT forward captures block 31
  - Score-only methods reuse cached scores across budgets
  - Multi-GPU sharding via --shard_idx / --num_shards

Output: per-sample JSON with intent labels for spatial/non-spatial analysis.
"""

import argparse, json, os, re, sys, time, random as _random, traceback
import torch
import torch.nn.functional as F
from PIL import Image

_random.seed(42)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "qcal_support"))

from models.qwen3vl_wrapper import Qwen2VLWrapper
from models.token_manipulator_v2 import TokenManipulatorV2

_IMAGE_TOKEN_ID  = 151655
_VISION_START_ID = 151652
_VISION_END_ID   = 151653

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
BUDGET_LEVELS = [5.0, 10.0, 20.0, 33.0]

# ── Intent classifier (spatial vs non-spatial) ──

COVERAGE_HINT = [
    "is there", "are there", "how many", "how much",
    "where is", "where are", "describe",
    "what is happening", "what is going on",
    "what are the people doing", "what are they doing",
    "locate", "bounding box", "bbox", "position of",
]

OCR_INTENT = {
    "read", "reads", "reading", "written", "write", "writes", "writing",
    "says", "say", "stated", "printed",
    "spell", "spells", "spelled", "spelling", "translate", "translation",
    "text", "texts", "word", "words", "letter", "letters",
    "character", "characters", "digit", "digits", "numeral", "numerals",
    "symbol", "symbols", "sentence", "paragraph", "inscription", "slogan",
    "sign", "signs", "label", "labels", "caption", "captions",
    "title", "titles", "heading", "subtitle",
    "banner", "banners", "poster", "posters", "billboard", "billboards",
    "menu", "menus", "plate", "plates", "jersey",
    "brand", "brands", "logo", "logos", "trademark",
    "price", "prices", "cost", "score", "scores",
    "date", "dates", "address", "equation", "formula",
    "percentage", "percent", "phone", "telephone",
    "url", "website", "email", "number", "numbers",
}

_TEMPLATE_STRIP = re.compile(
    r'(?:please\s+answer\s+directly.*|answer\s+the\s+question\s+using.*)',
    re.IGNORECASE)
_OPTION_STRIP = re.compile(r'\n\s*[A-D]\.\s+.*', re.DOTALL)

def _clean_question(q):
    q = _OPTION_STRIP.sub('', q)
    q = _TEMPLATE_STRIP.sub('', q)
    return q.strip()

def query_intent(question):
    """Classify question as spatial or non-spatial (ocr/precision)."""
    p = _clean_question(question).lower()
    words = set(p.split())
    for hint in COVERAGE_HINT:
        if hint in p:
            return "spatial"
    if words & OCR_INTENT:
        return "ocr"
    return "precision"

def intent_group(intent):
    """Binary grouping: spatial vs non-spatial."""
    return "spatial" if intent == "spatial" else "non-spatial"


# ── Prompt / eval ──

def make_prompt(benchmark, sample):
    q = sample.get("question", sample.get("query", ""))
    if benchmark in ("textvqa", "chartqa", "docvqa", "gqa", "ocrbench"):
        return q + "\nAnswer the question using a single word or phrase."
    if benchmark == "grounding":
        expr = sample.get("expression", "")
        return f"Please locate \"{expr}\" in the image and output its bounding box coordinates."
    return q

def max_new_tokens_map(benchmark):
    return {"textvqa": 32, "docvqa": 64, "chartqa": 32, "gqa": 16,
            "grounding": 64, "mme": 16, "hallbench": 16,
            "ocrbench": 64, "scienceqa": 16, "ai2d": 16, "mmmu": 16,
            }.get(benchmark, 16)

def _generic_acc(output, label):
    o = output.strip().lower()
    l = label.strip().lower()
    if not l:
        return 0.0
    # Exact match
    if o == l:
        return 1.0
    # Multiple-choice: extract leading option letter
    pred_letter = re.match(r'^([a-d])\b', o)
    label_letter = re.match(r'^([a-d])\b', l)
    if pred_letter and label_letter:
        return 1.0 if pred_letter.group(1) == label_letter.group(1) else 0.0
    if label_letter:
        # label is option letter, check if pred starts with it
        return 1.0 if o.startswith(label_letter.group(1)) else 0.0
    # yes/no type (POPE, MME etc)
    if l in ("yes", "no"):
        return 1.0 if o.startswith(l) else 0.0
    return 0.0

def _vqa_accuracy(prediction, answers):
    if not answers:
        return 0.0
    pred = prediction.strip().lower()
    matches = sum(1 for a in answers if a.strip().lower() == pred)
    return min(1.0, matches / 3.0)

def _iou_from_text(pred_text, gt_bbox, img_size):
    """Parse bbox from model output and compute IoU with ground truth.

    Qwen2.5-VL outputs bbox_2d in pixel coordinates as JSON.
    GT bbox from COCO is also in pixel coordinates.
    Both are normalized to [0, 1] for IoU computation.
    """
    import re as _re
    w, h = img_size
    # Try bbox_2d JSON format first: {"bbox_2d": [x1, y1, x2, y2]}
    m = _re.search(r'bbox_2d["\s:]*\[([^\]]+)\]', pred_text)
    if m:
        nums = _re.findall(r'[\d.]+', m.group(1))
    else:
        # Fallback: extract all numbers, skip leading non-coord numbers
        all_nums = _re.findall(r'[\d.]+', pred_text)
        # Filter for plausible coordinate values (> 1)
        nums = [n for n in all_nums if float(n) > 1]
        if len(nums) < 4:
            nums = all_nums
    if len(nums) < 4:
        return 0.0
    try:
        px1, py1, px2, py2 = [float(x) for x in nums[:4]]
    except ValueError:
        return 0.0
    # Normalize predicted coords to [0, 1]
    if max(px1, py1, px2, py2) > 1.0:
        px1, py1, px2, py2 = px1/w, py1/h, px2/w, py2/h
    # Normalize GT coords to [0, 1]
    gx1, gy1, gx2, gy2 = gt_bbox
    if max(gx1, gy1, gx2, gy2) > 1.0:
        gx1, gy1, gx2, gy2 = gx1/w, gy1/h, gx2/w, gy2/h
    # IoU
    ix1 = max(px1, gx1); iy1 = max(py1, gy1)
    ix2 = min(px2, gx2); iy2 = min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_p = max(0, px2 - px1) * max(0, py2 - py1)
    area_g = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    union = area_p + area_g - inter
    return inter / union if union > 0 else 0.0

def _ocr_accuracy(prediction, answers):
    """OCRBench: containment match (official metric)."""
    pred = prediction.strip().lower()
    for a in answers:
        if str(a).strip().lower() in pred:
            return 1.0
    return 0.0

def eval_metric(benchmark, output, sample):
    if benchmark == "chartqa":
        answers = sample.get("answers", [])
        pred = output.strip().lower()
        for a in answers:
            a_clean = a.strip().lower()
            if a_clean == pred:
                return 1.0
            # Relaxed accuracy: ±5% numerical tolerance (paper standard)
            try:
                pred_num = float(pred.replace(",", "").replace("%", ""))
                a_num = float(a_clean.replace(",", "").replace("%", ""))
                if a_num != 0 and abs(pred_num - a_num) / abs(a_num) <= 0.05:
                    return 1.0
            except ValueError:
                pass
        return 0.0
    if benchmark in ("textvqa", "docvqa"):
        return _vqa_accuracy(output, sample.get("answers", []))
    if benchmark in ("pope", "mme", "hallbench", "realworldqa"):
        return _generic_acc(output, sample.get("answer", sample.get("label", "")))
    if benchmark in ("gqa",):
        return _generic_acc(output, sample.get("answer", ""))
    if benchmark in ("scienceqa", "ai2d", "mmmu"):
        return _generic_acc(output, sample.get("answer", ""))
    if benchmark == "ocrbench":
        return _ocr_accuracy(output, sample.get("answers", []))
    if benchmark == "grounding":
        return _iou_from_text(output, sample.get("bbox", [0,0,0,0]),
                              sample.get("image_size", (1, 1)))
    return 0.0


# ── Token position helpers ──

def get_vis_positions(input_ids):
    return torch.where(input_ids[0] == _IMAGE_TOKEN_ID)[0].tolist()

def get_text_positions(input_ids):
    ids = input_ids[0]
    vs = (ids == _VISION_START_ID).nonzero(as_tuple=True)[0]
    ve = (ids == _VISION_END_ID).nonzero(as_tuple=True)[0]
    if len(vs) == 0 or len(ve) == 0:
        return []
    vs, ve = vs[0].item(), ve[0].item()
    return [i for i in range(ids.shape[0]) if i < vs or i > ve]


# ── Shared attention score extraction (multi-layer, one forward) ──

@torch.no_grad()
def extract_multi_layer_scores(model, inputs, vis_pos, text_pos,
                                layers=(0, 1, 2, 20),
                                apply_rope=True):
    """Single LLM forward capturing hidden states at multiple layers.

    Returns dict: {layer_idx: scores_list} for text→vis attention at each layer.
    Also returns last-text query scores for FastV (layer 2, last_text mode).

    Args:
        apply_rope: if True, apply input_layernorm + M-RoPE before Q·K^T.
                    Set False for HAWK (position-agnostic by design).
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb)

    lm = model.model.model.language_model
    captured = {}
    hooks = []

    for L in layers:
        def make_hook(layer_idx):
            def _pre(module, args):
                if isinstance(args, tuple) and len(args) > 0:
                    h = args[0]
                    captured[layer_idx] = (h.detach() if h.dim() == 3
                                           else h.unsqueeze(0).detach())
            return _pre
        h = lm.layers[L].register_forward_pre_hook(make_hook(L))
        hooks.append(h)

    try:
        model.model(**inputs, output_hidden_states=False,
                    output_attentions=False, return_dict=True)
    finally:
        for h in hooks:
            h.remove()

    # Pre-compute position embeddings for RoPE if needed
    pos_emb = None
    if apply_rope:
        position_ids = inputs.get("position_ids")
        if position_ids is not None:
            # position_ids: [4, batch, seq] or [3, batch, seq]
            if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                position_ids_rope = position_ids[1:]  # drop text_position_ids
            elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
                position_ids_rope = position_ids
            else:
                position_ids_rope = None
            if position_ids_rope is not None:
                # Use a dummy tensor to get device/dtype right
                dummy = next(lm.parameters())
                dummy_x = torch.zeros(1, position_ids_rope.shape[2],
                                      dummy.shape[-1] if dummy.ndim > 1 else 1,
                                      device=dummy.device, dtype=dummy.dtype)
                pos_emb = lm.rotary_emb(dummy_x,
                                        position_ids_rope.to(dummy.device))
        mrope_section = getattr(lm.config, "rope_parameters", {}).get(
            "mrope_section", None)
        if mrope_section is None:
            # Fallback: try to get from config
            mrope_section = getattr(lm.config, "mrope_section", None)

    results = {}
    for L in layers:
        h = captured.get(L)
        if h is None or not vis_pos or not text_pos:
            results[L] = {"all_text": [0.0] * len(vis_pos),
                          "last_text": [0.0] * len(vis_pos)}
            continue

        attn_l = lm.layers[L].self_attn
        n_heads    = attn_l.num_heads
        n_kv_heads = attn_l.num_key_value_heads
        head_dim   = attn_l.head_dim
        device = next(attn_l.q_proj.parameters()).device
        h = h.to(device)

        # S1: Apply input_layernorm (pre-norm architecture)
        h_normed = lm.layers[L].input_layernorm(h)

        v_idx = torch.tensor(vis_pos, dtype=torch.long, device=device)

        # All-text query
        t_idx_all = torch.tensor(text_pos, dtype=torch.long, device=device)
        q_all = attn_l.q_proj(h_normed[:, t_idx_all, :]).view(
            1, len(text_pos), n_heads, head_dim).transpose(1, 2)
        k = attn_l.k_proj(h_normed[:, v_idx, :]).view(
            1, len(vis_pos), n_kv_heads, head_dim).transpose(1, 2)
        if n_heads != n_kv_heads:
            k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)

        # S2: Apply M-RoPE
        if apply_rope and pos_emb is not None and mrope_section is not None:
            cos, sin = pos_emb
            cos = cos.to(device)
            sin = sin.to(device)
            # Slice position embeddings for query (text) and key (vis) positions
            cos_q = cos[:, :, t_idx_all, :]
            sin_q = sin[:, :, t_idx_all, :]
            cos_k = cos[:, :, v_idx, :]
            sin_k = sin[:, :, v_idx, :]
            q_all, _ = apply_multimodal_rotary_pos_emb(
                q_all, q_all, cos_q, sin_q, mrope_section)
            _, k = apply_multimodal_rotary_pos_emb(
                k, k, cos_k, sin_k, mrope_section)

        attn_all = F.softmax(
            torch.matmul(q_all, k.transpose(-2, -1)) * (head_dim ** -0.5),
            dim=-1)
        scores_all = attn_all[0].mean(0).mean(0).cpu().tolist()

        # Last-text query (for FastV)
        t_idx_last = torch.tensor([text_pos[-1]], dtype=torch.long, device=device)
        q_last = attn_l.q_proj(h_normed[:, t_idx_last, :]).view(
            1, 1, n_heads, head_dim).transpose(1, 2)

        if apply_rope and pos_emb is not None and mrope_section is not None:
            cos_ql = cos[:, :, t_idx_last, :]
            sin_ql = sin[:, :, t_idx_last, :]
            q_last, _ = apply_multimodal_rotary_pos_emb(
                q_last, q_last, cos_ql, sin_ql, mrope_section)

        attn_last = F.softmax(
            torch.matmul(q_last, k.transpose(-2, -1)) * (head_dim ** -0.5),
            dim=-1)
        scores_last = attn_last[0].mean(0).mean(0).cpu().tolist()

        results[L] = {"all_text": scores_all, "last_text": scores_last}

    return results


# ── Score → keep mask (top-k) ──

def topk_mask(scores, k):
    n = len(scores)
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    keep = set(ranked[:k])
    return [i in keep for i in range(n)]


# ── IntentAdaptive optimal config (from sweep) ──
_IA_WEIGHTS = {
    "spatial": {
        "5.0": {"w": (0.0, 0.0, 1.0, 0.0), "cr": 1.0},
        "10.0": {"w": (0.0, 0.0, 1.0, 0.0), "cr": 0.8},
        "20.0": {"w": (0.0, 0.0, 1.0, 0.0), "cr": 1.0},
        "33.0": {"w": (0.0, 0.0, 1.0, 0.0), "cr": 1.0},
    },
    "ocr": {
        "5.0": {"w": (1.0, 0.0, 0.0, 0.0), "cr": 1.0},
        "10.0": {"w": (1.0, 0.0, 0.0, 0.0), "cr": 1.0},
        "20.0": {"w": (1.0, 0.0, 0.0, 0.0), "cr": 1.0},
        "33.0": {"w": (1.0, 0.0, 0.0, 0.0), "cr": 0.8},
    },
    "precision": {
        "5.0": {"w": (0.5, 0.0, 0.5, 0.0), "cr": 0.8},
        "10.0": {"w": (1.0, 0.0, 0.0, 0.0), "cr": 1.0},
        "20.0": {"w": (0.3, 0.0, 0.7, 0.0), "cr": 1.0},
        "33.0": {"w": (1.0, 0.0, 0.0, 0.0), "cr": 1.0},
    },
}

def ia_keep_mask(hawk_sc, fastv_sc, vit_sc, svd_sc, vis_embeds, k, intent, budget):
    """IntentAdaptive: weighted signal combination + diversity fill."""
    import torch
    cfg = _IA_WEIGHTS.get(intent, _IA_WEIGHTS["precision"])
    b_cfg = cfg.get(str(budget), cfg.get("10.0"))
    w_h, w_f, w_v, w_s = b_cfg["w"]
    core_ratio = b_cfg["cr"]

    combined = (w_h * norm01(hawk_sc) + w_f * norm01(fastv_sc)
                + w_v * norm01(vit_sc) + w_s * norm01(svd_sc))
    n = len(combined)
    k = min(k, n)
    n_core = max(1, int(k * core_ratio))
    ranked = sorted(range(n), key=lambda i: combined[i], reverse=True)
    core = ranked[:n_core]

    if n_core >= k:
        keep = set(core[:k])
    else:
        # Diversity fill
        import torch.nn.functional as F
        emb = F.normalize(vis_embeds.float().cpu(), dim=-1)
        sim = torch.matmul(emb, emb.T)
        min_sim = torch.full((n,), float('inf'))
        for idx in core:
            min_sim = torch.min(min_sim, sim[idx])
        for idx in core:
            min_sim[idx] = float('inf')
        selected = list(core)
        for _ in range(k - len(core)):
            next_idx = min_sim.argmin().item()
            if min_sim[next_idx] == float('inf'):
                break
            selected.append(next_idx)
            min_sim = torch.min(min_sim, sim[next_idx])
            min_sim[next_idx] = float('inf')
        keep = set(selected)

    return [i in keep for i in range(n)]


def norm01(s):
    import torch
    if isinstance(s, list):
        s = torch.tensor(s, dtype=torch.float32)
    s = s.float()
    mn, mx = s.min(), s.max()
    if (mx - mn).abs() < 1e-9:
        return torch.zeros_like(s)
    return (s - mn) / (mx - mn)


# ── Run pruned generation ──

def run_pruned(model, inputs, vis_embeds, grid_thw, keep_mask, max_tok,
               modified_embeds=None):
    tok   = TokenManipulatorV2()
    merge = model.model.model.visual.spatial_merge_size
    if modified_embeds is not None:
        kept_idx = [i for i, k in enumerate(keep_mask) if k]
        assert modified_embeds.shape[0] == len(kept_idx)
        new_vis = vis_embeds.clone()
        for new_i, old_i in enumerate(kept_idx):
            new_vis[old_i] = modified_embeds[new_i]
        mr = tok.apply_token_mask(inputs, new_vis, grid_thw, keep_mask=keep_mask,
                                  spatial_merge_size=merge, config=model.model.config)
    else:
        mr = tok.apply_token_mask(inputs, vis_embeds, grid_thw, keep_mask=keep_mask,
                                  spatial_merge_size=merge, config=model.model.config)
    return model.generate_with_token_manipulation_v2(mr, max_new_tokens=max_tok)


# ── Benchmark loaders ──

def load_samples(benchmark, num_samples):
    if benchmark == "realworldqa":
        from benchmarks.realworldqa_loader import load_realworldqa
        return load_realworldqa(num_samples)
    elif benchmark == "textvqa":
        from benchmarks.textvqa_loader import load_textvqa
        return load_textvqa(num_samples)
    elif benchmark == "chartqa":
        from benchmarks.chartqa_loader import load_chartqa
        return load_chartqa(num_samples)
    elif benchmark == "pope":
        from benchmarks.pope_loader import load_pope
        return load_pope(num_samples)
    elif benchmark == "docvqa":
        from benchmarks.docvqa_loader import load_docvqa
        return load_docvqa(num_samples)
    elif benchmark == "mme":
        from benchmarks.mme_loader import load_mme
        return load_mme(num_samples)
    elif benchmark == "gqa":
        from benchmarks.gqa_loader import load_gqa
        return load_gqa(num_samples)
    elif benchmark == "scienceqa":
        from benchmarks.scienceqa_loader import load_scienceqa
        return load_scienceqa(num_samples)
    elif benchmark == "ai2d":
        from benchmarks.ai2d_loader import load_ai2d
        return load_ai2d(num_samples)
    elif benchmark == "ocrbench":
        from benchmarks.ocrbench_loader import load_ocrbench
        return load_ocrbench(num_samples)
    elif benchmark == "mmmu":
        from benchmarks.mmmu_loader import load_mmmu
        return load_mmmu(num_samples)
    elif benchmark == "grounding":
        from benchmarks.grounding_loader import load_grounding
        return load_grounding(num_samples)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def load_split(benchmark, split="eval", num_eval=250, num_sweep=200, seed=42):
    """Load samples with deterministic random split, no overlap.

    Shuffles all available samples with a fixed seed, then:
      - split="eval":  returns samples[0:num_eval]
      - split="sweep": returns samples[num_eval:num_eval+num_sweep]

    Both splits come from the same shuffle, guaranteeing no overlap.
    """
    import random as _rng
    total = num_eval + num_sweep
    all_samples = load_samples(benchmark, total)
    indices = list(range(len(all_samples)))
    _rng.Random(seed).shuffle(indices)
    if split == "eval":
        chosen = indices[:num_eval]
    elif split == "sweep":
        chosen = indices[num_eval:num_eval + num_sweep]
    else:
        raise ValueError(f"Unknown split: {split}")
    return [all_samples[i] for i in chosen]


# ── Per-sample evaluation ──

@torch.no_grad()
def eval_sample(model, sample, benchmark, baselines):
    img = sample.get("image")
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
        sample = dict(sample, image=img)

    prompt  = make_prompt(benchmark, sample)
    max_tok = max_new_tokens_map(benchmark)
    question = sample.get("question", sample.get("query",
                          sample.get("expression", "")))
    intent  = "spatial" if benchmark == "grounding" else query_intent(question)

    inputs               = model.prepare_inputs(img, prompt)
    vis_embeds, grid_thw = model.extract_visual_embeddings(inputs)
    vis_pos              = get_vis_positions(inputs["input_ids"])
    text_pos             = get_text_positions(inputs["input_ids"])
    n_vis                = len(vis_pos)

    if n_vis == 0:
        return None

    # ── Baseline (unpruned) ──
    bl_out    = model.generate(img, prompt, max_new_tokens=max_tok)
    bl_metric = eval_metric(benchmark, bl_out, sample)

    result = {
        "baseline": bl_metric,
        "num_vis":  n_vis,
        "intent":   intent,
        "intent_group": intent_group(intent),
        "methods":  {},
    }
    # MME: preserve pair info for standard aggregate scoring
    if benchmark == "mme":
        result["question_id"] = sample.get("question_id", "")
        result["category"]    = sample.get("category", "")
        result["pair_idx"]    = sample.get("pair_idx", 0)

    # ── Shared LLM forward: capture layers {0, 1, 3, 20} ──
    # Two passes: one with RoPE (FastV/L20) and one without (HAWK, position-agnostic)
    layer_scores = extract_multi_layer_scores(
        model, inputs, vis_pos, text_pos, layers=(0, 1, 3, 20),
        apply_rope=True)
    hawk_scores_norope = extract_multi_layer_scores(
        model, inputs, vis_pos, text_pos, layers=(0,),
        apply_rope=False)

    hawk_sc = hawk_scores_norope[0]["all_text"]  # HAWK (L0, no RoPE)
    fastv_sc = layer_scores[3]["last_text"]      # FastV (L3, last_text) — fastv_k=3
    l20_sc  = layer_scores[20]["all_text"]       # L20 (all_text)

    # ── ViT scores (shared ViT forward) ──
    vit_sc = baselines["fastervlm"].score_tokens(inputs)

    # ── SVD-Prune scores ──
    svd_sc = baselines["svdprune"].score_tokens(vis_embeds)

    # ── Helper: eval a method with score-based top-k ──
    def eval_topk(name, scores):
        result["methods"][name] = {}
        for budget in BUDGET_LEVELS:
            k = max(1, int(n_vis * budget / 100.0))
            try:
                km = topk_mask(scores, k)
                pred = run_pruned(model, inputs, vis_embeds, grid_thw, km, max_tok)
                metric = eval_metric(benchmark, pred, sample)
            except Exception as e:
                print(f"    [WARN] {name} b={budget}%: {e}", flush=True)
                metric = None
            result["methods"][name][str(budget)] = metric

    # ── Helper: eval a method using its own prune() ──
    def eval_prune(name, baseline_obj, is_merge=False):
        result["methods"][name] = {}
        for budget in BUDGET_LEVELS:
            bf = budget / 100.0
            try:
                if is_merge:
                    km, mod_emb = baseline_obj.prune_and_merge(
                        model, inputs, vis_embeds, grid_thw, prune_ratio=1.0 - bf)
                else:
                    km = baseline_obj.prune(
                        model, inputs, vis_embeds, grid_thw, prune_ratio=1.0 - bf)
                    mod_emb = None
                pred = run_pruned(model, inputs, vis_embeds, grid_thw,
                                  km, max_tok, mod_emb)
                metric = eval_metric(benchmark, pred, sample)
            except Exception as e:
                print(f"    [WARN] {name} b={budget}%: {e}", flush=True)
                metric = None
            result["methods"][name][str(budget)] = metric

    # ── Helper: eval progressive method ──
    def eval_progressive(name, gen_fn):
        result["methods"][name] = {}
        for budget in BUDGET_LEVELS:
            bf = budget / 100.0
            try:
                pred = gen_fn(bf)
                metric = eval_metric(benchmark, pred, sample)
            except Exception as e:
                print(f"    [WARN] {name} b={budget}%: {e}", flush=True)
                metric = None
            result["methods"][name][str(budget)] = metric

    # ── Score-based methods (reuse cached scores) ──
    eval_topk("HAWK", hawk_sc)
    eval_topk("FastV", fastv_sc)
    eval_topk("L20", l20_sc)
    eval_topk("FasterVLM", vit_sc)

    # ── SVD-Prune (score + allocate) ──
    result["methods"]["SVD-Prune"] = {}
    for budget in BUDGET_LEVELS:
        bf = budget / 100.0
        try:
            km = baselines["svdprune"].allocate(svd_sc, n_vis, bf)
            pred = run_pruned(model, inputs, vis_embeds, grid_thw, km, max_tok)
            metric = eval_metric(benchmark, pred, sample)
        except Exception as e:
            print(f"    [WARN] SVD-Prune b={budget}%: {e}", flush=True)
            metric = None
        result["methods"]["SVD-Prune"][str(budget)] = metric

    # ── Methods with their own prune() ──
    eval_prune("VisPruner", baselines["vispruner"])
    eval_prune("DivPrune", baselines["divprune"])
    eval_prune("PTP", baselines["ptp"])
    eval_prune("ZSPAPrune", baselines["zspaprune"])
    eval_prune("ID-Selection", baselines["idselection"])
    eval_prune("D2Pruner", baselines["d2pruner"])
    eval_prune("AgilePruner", baselines["agilepruner"])

    # ── Progressive methods ──
    from baselines.pyramiddrop.adapter import find_visual_range as _fvr
    _vis_start, _vis_end = _fvr(inputs["input_ids"])

    def _pd_gen(bf):
        from baselines.pyramiddrop.adapter import PyramidDropBaseline
        pd = PyramidDropBaseline(model, budget_override=bf)
        return pd.generate_progressive(
            inputs, _vis_start, _vis_end,
            max_new_tokens=max_tok,
            vis_token_positions=vis_pos if vis_pos else None)
    eval_progressive("PyramidDrop", _pd_gen)

    def _fit_gen(bf):
        from baselines.fitprune.adapter import FitPruneBaseline
        fp = FitPruneBaseline(model, budget_frac=bf)
        return fp.generate_progressive(
            inputs, _vis_start, _vis_end,
            budget_frac=bf,
            max_new_tokens=max_tok,
            vis_token_positions=vis_pos if vis_pos else None)
    eval_progressive("FitPrune", _fit_gen)

    eval_prune("SparseVLM", baselines["sparsevlm"])

    # ── IntentAdaptive (ours): optimal signal combo + diversity ──
    result["methods"]["IntentAdaptive"] = {}
    for budget in BUDGET_LEVELS:
        bf = budget / 100.0
        k = max(1, int(n_vis * bf))
        try:
            km = ia_keep_mask(hawk_sc, fastv_sc, vit_sc, svd_sc,
                              vis_embeds, k, intent, budget)
            pred = run_pruned(model, inputs, vis_embeds, grid_thw, km, max_tok)
            metric = eval_metric(benchmark, pred, sample)
        except Exception as e:
            print(f"    [WARN] IntentAdaptive b={budget}%: {e}", flush=True)
            metric = None
        result["methods"]["IntentAdaptive"][str(budget)] = metric

    # ── Random baseline ──
    result["methods"]["random"] = {}
    for budget in BUDGET_LEVELS:
        k = max(1, int(n_vis * budget / 100.0))
        try:
            sel = set(_random.sample(range(n_vis), k))
            km = [i in sel for i in range(n_vis)]
            pred = run_pruned(model, inputs, vis_embeds, grid_thw, km, max_tok)
            metric = eval_metric(benchmark, pred, sample)
        except Exception as e:
            print(f"    [WARN] random b={budget}%: {e}", flush=True)
            metric = None
        result["methods"]["random"][str(budget)] = metric

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+",
                        default=["realworldqa", "textvqa", "chartqa",
                                 "pope", "docvqa", "mme", "gqa",
                                 "scienceqa", "ai2d", "ocrbench",
                                 "mmmu", "grounding"])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--out_dir", default="strategy_test/results/broad_comparison")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--sample_offset", type=int, default=0,
                        help="Skip first N samples to avoid overlap")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model...", flush=True)
    model = Qwen2VLWrapper(MODEL_PATH)
    model.model.eval()
    print("Model loaded.", flush=True)

    # ── Load all baselines ──
    from baselines.fastervlm.adapter import FasterVLMBaseline
    from baselines.vispruner.adapter import VisPrunerBaseline
    from baselines.svdprune.adapter import SVDPruneBaseline
    from baselines.divprune.adapter import DivPruneBaseline
    from baselines.ptp.adapter import PTPBaseline
    from baselines.zspaprune.adapter import ZSPAPruneBaseline
    from baselines.idselection.adapter import IDSelectionBaseline
    from baselines.d2pruner.adapter import D2PrunerBaseline
    from baselines.agilepruner.adapter import AgilePrunerBaseline
    from baselines.sparsevlm.adapter import SparseVLMBaseline

    baselines = {
        "fastervlm":   FasterVLMBaseline(model),
        "vispruner":   VisPrunerBaseline(model),
        "svdprune":    SVDPruneBaseline(),
        "divprune":    DivPruneBaseline(),
        "ptp":         PTPBaseline(model),
        "zspaprune":   ZSPAPruneBaseline(model),
        "idselection": IDSelectionBaseline(model),
        "d2pruner":    D2PrunerBaseline(model),
        "agilepruner": AgilePrunerBaseline(model),
        "sparsevlm":   SparseVLMBaseline(model),
    }
    print(f"Loaded {len(baselines)} baseline instances.", flush=True)

    for bm in args.benchmarks:
        print(f"\n{'='*60}", flush=True)
        print(f"[{bm}] Loading eval split (random, no overlap with sweep)",
              flush=True)
        samples = load_split(bm, split="eval",
                             num_eval=args.num_samples, num_sweep=200)

        if args.num_shards > 1:
            samples = [s for i, s in enumerate(samples)
                       if i % args.num_shards == args.shard_idx]

        from collections import Counter
        def _q(s):
            return s.get("question", s.get("query", s.get("expression", "")))
        def _intent(s):
            return "spatial" if bm == "grounding" else query_intent(_q(s))
        intents = Counter(_intent(s) for s in samples)
        groups = Counter(intent_group(_intent(s)) for s in samples)
        print(f"  {len(samples)} samples | intents={dict(intents)} | groups={dict(groups)}",
              flush=True)

        results = []
        t0 = time.time()
        for i, sample in enumerate(samples):
            try:
                r = eval_sample(model, sample, bm, baselines)
                if r is not None:
                    results.append(r)
                    elapsed = time.time() - t0
                    avg_t = elapsed / (i + 1)
                    eta = avg_t * (len(samples) - i - 1)
                    print(f"  [{bm}] {i+1}/{len(samples)} "
                          f"bl={r['baseline']:.2f} intent={r['intent']} "
                          f"avg={avg_t:.1f}s ETA={eta:.0f}s", flush=True)
            except Exception:
                print(f"  [{bm}] sample {i} error:\n{traceback.format_exc()}",
                      flush=True)

        suffix = f"_shard{args.shard_idx}" if args.num_shards > 1 else ""
        out_f = os.path.join(args.out_dir, f"{bm}{suffix}.json")
        with open(out_f, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {out_f}  ({len(results)} samples)", flush=True)

        # MME: compute standard aggregate scores (Perception + Cognition)
        if bm == "mme" and results:
            from benchmarks.mme_loader import mme_aggregate_scores
            # Baseline aggregate
            bl_records = [{"question_id": r["question_id"],
                           "category": r["category"],
                           "pair_idx": r["pair_idx"],
                           "score": r["baseline"]} for r in results
                          if "question_id" in r]
            bl_agg = mme_aggregate_scores(bl_records)
            print(f"  [MME] Baseline: P={bl_agg['perception']:.0f} "
                  f"C={bl_agg['cognition']:.0f} "
                  f"T={bl_agg['total']:.0f}", flush=True)
            # Per-method aggregate
            method_names = list(results[0].get("methods", {}).keys())
            for mn in method_names:
                for budget in BUDGET_LEVELS:
                    bk = str(budget)
                    records = []
                    for r in results:
                        if "question_id" not in r:
                            continue
                        sc = r.get("methods", {}).get(mn, {}).get(bk)
                        if sc is not None:
                            records.append({
                                "question_id": r["question_id"],
                                "category": r["category"],
                                "pair_idx": r["pair_idx"],
                                "score": sc})
                    if records:
                        agg = mme_aggregate_scores(records)
                        print(f"  [MME] {mn} b={budget}%: "
                              f"P={agg['perception']:.0f} "
                              f"C={agg['cognition']:.0f} "
                              f"T={agg['total']:.0f}", flush=True)
            # Save aggregate
            agg_f = os.path.join(args.out_dir, f"mme_aggregate{suffix}.json")
            with open(agg_f, "w") as f:
                json.dump({"baseline": bl_agg}, f, indent=2)
            print(f"  Saved MME aggregate → {agg_f}", flush=True)


if __name__ == "__main__":
    main()
