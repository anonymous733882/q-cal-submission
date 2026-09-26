#!/usr/bin/env python3
"""Six-model adaptation runner for non-HAWK baseline methods.

This file is intentionally separate from BEA.  It adapts only the parts of a
baseline that can be expressed with the current backend contract.  Methods that
need original per-layer/per-step/vision-attention hooks fail explicitly instead
of silently becoming one-shot variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


WORKSPACE = Path(__file__).resolve().parents[2]
DUALSIGNAL = WORKSPACE / "qcal"
BEA_DIR = WORKSPACE / "qcal_support"
DEFAULT_OUT = DUALSIGNAL / "results/nonhawk_six_model_adaptation"
CALIBRATION_DIR = DUALSIGNAL / "runners" / "calibration"
D2_QWEN25_BIAS_PATH = CALIBRATION_DIR / "d2_qwen25vl_attention_bias_n1000.pt"
HAWK_QWEN25_WEIGHTS_PATH = CALIBRATION_DIR / "hawk_qwen25vl_head_weights_ablation_pope_textvqa_chartqa_n16.json"

sys.path.insert(0, str(BEA_DIR))
sys.path.insert(0, str(BEA_DIR / "strategy_test"))
sys.path.insert(0, str(DUALSIGNAL / "runners"))
sys.path.insert(0, str(WORKSPACE))

from baselines.sparsevila.adapter import decode_ratio as sparsevila_decode_ratio_for_backend  # noqa: E402
from baselines.common.masks import budget_to_keep, norm01  # noqa: E402
from baselines.agilepruner.selection import select_cached as select_agilepruner  # noqa: E402
from baselines.d2pruner.selection import select_cached as select_d2pruner  # noqa: E402
from baselines.divprune.selection import select_cached as select_divprune  # noqa: E402
from baselines.fastv.selection import select_cached as select_fastv  # noqa: E402
from baselines.fastervlm.selection import select_cached as select_fastervlm  # noqa: E402
from baselines.hawk.selection import select_cached as select_hawk  # noqa: E402
from baselines.idselection.selection import select_cached as select_idselection  # noqa: E402
from baselines.ptp.selection import select_cached as select_ptp  # noqa: E402
from baselines.sparsevila.adapter import select_cached as select_sparsevila  # noqa: E402
from baselines.svdprune.selection import select_cached as select_svdprune  # noqa: E402
from baselines.vispruner.selection import select_cached as select_vispruner  # noqa: E402
from baselines.zspaprune.selection import select_cached as select_zspaprune  # noqa: E402

from model_backends import load_backend  # noqa: E402
from run_fixed_policy_generalization import (  # noqa: E402
    eval_metric as base_eval_metric,
    load_samples_ext as base_load_samples_ext,
    make_prompt as base_make_prompt,
    max_new_tokens_map as base_max_new_tokens_map,
    sample_images,
)
from multiround_dataset_loaders import load_multiround_samples  # noqa: E402
from count_relaxed_scoring import maybe_relaxed_count_score  # noqa: E402


COST_FIELDS = (
    "prefill_tl",
    "gen_tl",
    "total_tl",
    "n_vis_prefill",
    "n_vis_first_generation",
    "n_vis_decode",
    "n_gen_tokens",
    "shared_setup_sec",
    "shared_question_sec",
    "baseline_wall_sec",
    "method_prefill_wall_sec",
    "method_decode_wall_sec",
    "method_total_wall_sec",
)


COST_SCHEMA = {
    "unit": "per_question_turn",
    "fields": list(COST_FIELDS),
    "visual_token_layer_accounting": (
        "The first answer token is predicted before SparseVILA compacts its visual KV cache; "
        "later answer tokens use the compacted cache. n_gen_tokens counts decoded answer tokens."
    ),
    "time_accounting": (
        "shared_* fields record method-independent input/image setup only. "
        "baseline_wall_sec records reference full-attention generation and is not added to method costs. "
        "Each method records private pruning/prefix-prefill/decode wall time in method_* fields. "
        "For end-to-end method comparisons, sum shared_setup_sec + shared_question_sec + method_total_wall_sec."
    ),
}


MODEL_TOTAL_LAYERS = {
    "qwen2vl2b": 28,
    "qwen2vl": 28,
    "qwen25vl": 28,
    "qwen25vl3b": 36,
    "qwen3vl": 36,
    "qwen3vl4b": 36,
    "internvl3-8b": 28,
    "internvl3.5-8b": 36,
    "llava-7b": 32,
    "llava-13b": 40,
}

BEA_MULTIROUND_TL_BUDGET = {
    "source": "embedded token-layer equivalent-budget formula",
    "formula": "Y = (stage1_frac*S + stage2_frac*(N-S)) / N",
    "stage1_frac": 0.33,
    "stage2_frac": 0.10,
    "stage_layer": {
        "qwen2vl2b": 20,
        "qwen2vl": 20,
        "qwen25vl3b": 24,
        "qwen25vl": 22,
        "qwen3vl4b": 26,
        "qwen3vl": 30,
        "internvl3-8b": 4,
        "internvl3.5-8b": 8,
        "llava-7b": 10,
        "llava-13b": 6,
    },
}


MODEL_IDS = {
    "qwen2vl2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2vl": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen25vl3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3vl4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
}


NONHAWK_METHODS = [
    "SVD-Prune",
    "PTP",
    "AgilePruner",
    "DivPrune",
    "ZSPAPrune",
    "ID-Selection",
    "FastV",
    "FasterVLM",
    "VisPruner",
    "SparseVILA",
    "D2Pruner",
    "HAWK",
]

CALIBRATED_METHOD_MODELS = {
    "D2Pruner": {"qwen25vl"},
    "HAWK": {"qwen25vl"},
}


MULTIROUND_BENCHMARKS = {
    "gqa",
    "pope",
    "vqav2",
    "clevr",
    "visual7w",
    "visualgenomeqa",
    "tallyqa",
    "gqa_grounding",
    "invig",
}
MULTIROUND_SUFFIX = " Answer with a single word or short phrase."


def _norm_answer(text: str) -> str:
    import string

    punct = set(string.punctuation)
    text = str(text).lower()
    text = "".join(c if c not in punct else " " for c in text)
    words = [w for w in text.split() if w not in {"a", "an", "the"}]
    return " ".join(words).strip()


def _short_answer_score(output: str, answer: Any, question: Any = None) -> float:
    relaxed = maybe_relaxed_count_score(output, answer, question)
    if relaxed is not None:
        return float(relaxed)
    pred = _norm_answer(output)
    answers = answer if isinstance(answer, (list, tuple)) else [answer]
    for ans in answers:
        gold = _norm_answer(ans)
        if not gold:
            continue
        if pred == gold:
            return 1.0
        pred_words = pred.split()
        gold_words = gold.split()
        if pred_words[:len(gold_words)] == gold_words:
            return 1.0
    return 0.0


def load_samples_ext(benchmark: str, num_samples: int, questions_per_image: int | None = None) -> list[dict[str, Any]]:
    if benchmark in MULTIROUND_BENCHMARKS:
        return load_multiround_samples(benchmark, num_samples, questions_per_image or 15)
    return base_load_samples_ext(benchmark, num_samples)


def make_prompt(benchmark: str, sample: dict[str, Any]) -> str:
    if benchmark in MULTIROUND_BENCHMARKS:
        question = str(sample.get("question", "")).strip()
        if benchmark == "pope":
            return question + " Answer yes or no."
        return question + MULTIROUND_SUFFIX
    return base_make_prompt(benchmark, sample)


def eval_metric(benchmark: str, output: str, sample: dict[str, Any]) -> float:
    if benchmark in MULTIROUND_BENCHMARKS:
        return _short_answer_score(output, sample.get("answer", ""), sample.get("question", ""))
    return base_eval_metric(benchmark, output, sample)


def max_new_tokens_map(benchmark: str) -> int:
    if benchmark in MULTIROUND_BENCHMARKS:
        return 32
    return base_max_new_tokens_map(benchmark)


class OriginalHookRequired(RuntimeError):
    pass


@dataclass(frozen=True)
class MethodInfo:
    name: str
    status: str
    original_requirement: str
    adapter_scope: str


METHOD_INFO = {
    "SVD-Prune": MethodInfo(
        "SVD-Prune",
        "executable",
        "visual embedding low-rank/leverage scoring",
        "backend visual embeddings",
    ),
    "DivPrune": MethodInfo(
        "DivPrune",
        "executable",
        "diversity selection in visual embedding space",
        "backend visual embeddings",
    ),
    "ZSPAPrune": MethodInfo(
        "ZSPAPrune",
        "executable",
        "prompt-aware similarity core plus diversity fill",
        "backend visual/text embeddings",
    ),
    "ID-Selection": MethodInfo(
        "ID-Selection",
        "executable",
        "LLM attention importance with Gaussian diversity suppression",
        "backend text-to-visual layer attention",
    ),
    "FastV": MethodInfo(
        "FastV",
        "executable",
        "early LLM text-to-visual attention ranking",
        "backend text-to-visual layer attention",
    ),
    "AgilePruner": MethodInfo(
        "AgilePruner",
        "executable",
        "attention seed plus adaptive diversity expansion",
        "backend all-text text-to-visual attention and visual embeddings",
    ),
    "PTP": MethodInfo(
        "PTP",
        "executable",
        "region/instruction-guided pruning and refinement pipeline",
        "backend vision received-attention plus instruction attention",
    ),
    "FasterVLM": MethodInfo(
        "FasterVLM",
        "executable",
        "visual-encoder received-attention token scoring",
        "backend vision received-attention exposure",
    ),
    "VisPruner": MethodInfo(
        "VisPruner",
        "executable",
        "ViT received-attention scoring plus diversity filtering",
        "backend vision received-attention exposure plus visual embeddings",
    ),
    "SparseVILA": MethodInfo(
        "SparseVILA",
        "executable",
        "query-agnostic context sparsity plus query-aware decode-stage visual KV retrieval",
        "six-model continuous multi-round adaptation: first-turn context sparsity with prefix-KV reuse and per-question query-aware decode KV compaction",
    ),
    "D2Pruner": MethodInfo(
        "D2Pruner",
        "executable",
        "debiased text-to-visual attention pivots plus structural diversity fill",
        "backend text-to-visual attention and visual embeddings",
    ),
    "HAWK": MethodInfo(
        "HAWK",
        "executable",
        "calibrated layer-0 head-weighted text-to-visual attention",
        "qwen25vl calibrated HAWK head weights",
    ),
    "PyramidDrop": MethodInfo(
        "PyramidDrop",
        "executable",
        "progressive layer-wise visual-token dropping",
        "backend progressive generation hook",
    ),
    "FitPrune": MethodInfo(
        "FitPrune",
        "executable",
        "progressive schedule fitted to layer/token budget",
        "backend progressive generation hook",
    ),
    "SparseVLM": MethodInfo(
        "SparseVLM",
        "executable",
        "progressive recycling/sparsification during generation",
        "backend progressive generation hook with token recycling",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(MODEL_IDS))
    p.add_argument("--methods", nargs="+", default=NONHAWK_METHODS)
    p.add_argument("--benchmarks", nargs="+", default=["pope", "gqa"])
    p.add_argument("--eval-samples", type=int, default=0,
                   help="0 writes only manifests; >0 runs executable adapters.")
    p.add_argument("--num-images", type=int, default=None,
                   help="Alias for --eval-samples for single-image multi-turn full runs.")
    p.add_argument("--questions-per-image", type=int, default=None,
                   help="Alias for --max-turns-per-sample for multi-turn full runs.")
    p.add_argument("--budgets", nargs="+", type=float, default=[20.0],
                   help="Visual-token retain percentages used when --budget-mode=explicit.")
    p.add_argument("--budget-mode", choices=["bea_multiround_tl", "explicit"], default="bea_multiround_tl",
                   help="How to set baseline retain percentages. bea_multiround_tl matches BEA v4 token*layer cost.")
    p.add_argument("--sparsevila-decode-keep-percent", type=float, default=10.0,
                   help="SparseVILA decode-stage visual KV retain percentage, measured against original visual-token count.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--resize-square", type=int, default=1008)
    p.add_argument("--max-pixels", type=int, default=1016064)
    p.add_argument("--multi-max-pixels", type=int, default=262144)
    p.add_argument("--sample-offset", type=int, default=0)
    p.add_argument("--max-turns-per-sample", type=int, default=8)
    p.add_argument("--allow-hook-required", action="store_true",
                   help="Include hook-required methods as recorded skipped rows.")
    p.add_argument("--full-json", action="store_true",
                   help="Also write stage2-compatible per model/benchmark JSON summaries.")
    p.add_argument("--list-methods", action="store_true")
    return p.parse_args()


def _latest_snapshot(path: Path) -> Path | None:
    if path.is_dir() and path.name == "snapshots":
        snaps = sorted([p for p in path.iterdir() if p.is_dir()])
        return snaps[-1] if snaps else None
    return path if path.exists() else None


def resolve_model(model_key: str) -> tuple[str, str]:
    if model_key not in MODEL_IDS:
        raise ValueError(f"no packaged model ID for {model_key}")
    return MODEL_IDS[model_key], "qwen"


def bea_multiround_budget_percent(model_key: str) -> float:
    n_layers = MODEL_TOTAL_LAYERS[model_key]
    stage = BEA_MULTIROUND_TL_BUDGET["stage_layer"][model_key]
    stage1 = float(BEA_MULTIROUND_TL_BUDGET["stage1_frac"])
    stage2 = float(BEA_MULTIROUND_TL_BUDGET["stage2_frac"])
    return 100.0 * ((stage1 * stage) + (stage2 * (n_layers - stage))) / n_layers


def budget_metadata_for_model(model_key: str) -> dict[str, Any]:
    n_layers = MODEL_TOTAL_LAYERS[model_key]
    stage = BEA_MULTIROUND_TL_BUDGET["stage_layer"][model_key]
    budget = bea_multiround_budget_percent(model_key)
    return {
        "mode": "bea_multiround_tl",
        "source": BEA_MULTIROUND_TL_BUDGET["source"],
        "formula": BEA_MULTIROUND_TL_BUDGET["formula"],
        "total_layers_N": n_layers,
        "stage_layer_S": stage,
        "stage1_frac": BEA_MULTIROUND_TL_BUDGET["stage1_frac"],
        "stage2_frac": BEA_MULTIROUND_TL_BUDGET["stage2_frac"],
        "equivalent_static_budget_percent": budget,
        "rounded_display_percent": round(budget, 1),
    }


def effective_budgets_for_model(args: argparse.Namespace, model_key: str) -> list[float]:
    if args.budget_mode == "bea_multiround_tl":
        return [bea_multiround_budget_percent(model_key)]
    return [float(x) for x in args.budgets]


def write_manifest(out_dir: Path, methods: list[str], models: list[str], args: argparse.Namespace) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "non-HAWK six-model baseline adaptation",
        "budgeting": {
            "mode": args.budget_mode,
            "explicit_budgets_percent": args.budgets,
            "bea_multiround_tl": {
                "source": BEA_MULTIROUND_TL_BUDGET["source"],
                "formula": BEA_MULTIROUND_TL_BUDGET["formula"],
                "stage1_frac": BEA_MULTIROUND_TL_BUDGET["stage1_frac"],
                "stage2_frac": BEA_MULTIROUND_TL_BUDGET["stage2_frac"],
            },
            "effective_budgets_by_model": {},
        },
        "models": {},
        "methods": {},
    }
    for model_key in models:
        manifest["budgeting"]["effective_budgets_by_model"][model_key] = effective_budgets_for_model(args, model_key)
        try:
            path, backend = resolve_model(model_key)
            manifest["models"][model_key] = {
                "status": "available",
                "path": str(path),
                "backend": backend,
                "total_layers": MODEL_TOTAL_LAYERS[model_key],
                "budget": (
                    budget_metadata_for_model(model_key)
                    if args.budget_mode == "bea_multiround_tl"
                    else {"mode": "explicit", "budgets_percent": args.budgets}
                ),
            }
        except Exception as exc:
            manifest["models"][model_key] = {"status": "missing", "error": str(exc)}
    for name in methods:
        info = METHOD_INFO[name]
        manifest["methods"][name] = {
            "status": info.status,
            "original_requirement": info.original_requirement,
            "adapter_scope": info.adapter_scope,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adaptation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def attention_scores(
    model: Any,
    inputs: Any,
    layer: int,
    n_vis: int,
    query_mode: str = "last_text",
) -> torch.Tensor:
    layer = max(0, min(int(layer), int(getattr(model, "total_layers", layer + 1)) - 1))
    by_layer = model.layer_attention_scores(inputs, [layer], query_mode=query_mode)
    scores = torch.tensor(by_layer.get(layer, []), dtype=torch.float32)
    if scores.shape[0] != n_vis:
        padded = torch.zeros(n_vis, dtype=torch.float32)
        m = min(n_vis, int(scores.shape[0]))
        padded[:m] = scores[:m]
        scores = padded
    return norm01(scores)


class SharedPruneCache:
    def __init__(
        self,
        model: Any,
        inputs: Any,
        embeds: torch.Tensor,
        grid_thw: Any,
        sample_cache: dict[str, Any],
    ):
        self.model = model
        self.inputs = inputs
        self.embeds = embeds
        self.grid_thw = grid_thw
        self.n_vis = int(embeds.shape[0])
        self.sample_cache = sample_cache
        self.turn_cache: dict[Any, Any] = {}

    def n_keep(self, budget: float) -> int:
        return budget_to_keep(self.n_vis, budget)

    def visual_scores(self) -> torch.Tensor:
        key = ("visual_scores", self.n_vis)
        if key not in self.sample_cache:
            scores = torch.tensor(
                self.model.visual_received_attention_scores(self.inputs, self.n_vis),
                dtype=torch.float32,
            )
            self.sample_cache[key] = norm01(scores)
        return self.sample_cache[key]

    def attention(self, layer: int, query_mode: str = "last_text") -> torch.Tensor:
        key = ("attn", int(layer), query_mode, self.n_vis)
        if key not in self.turn_cache:
            self.turn_cache[key] = attention_scores(
                self.model, self.inputs, layer, self.n_vis, query_mode=query_mode)
        return self.turn_cache[key]

    def normalized_embeds(self) -> torch.Tensor:
        key = ("norm_embeds", self.n_vis)
        if key not in self.sample_cache:
            self.sample_cache[key] = F.normalize(self.embeds.float().cpu(), dim=-1)
        return self.sample_cache[key]

    def similarity(self) -> torch.Tensor:
        key = ("cosine_similarity", self.n_vis)
        if key not in self.sample_cache:
            emb = self.normalized_embeds()
            self.sample_cache[key] = torch.matmul(emb, emb.T)
        return self.sample_cache[key]

    def svd_scores(self) -> torch.Tensor:
        key = ("svd_leverage", self.n_vis)
        if key not in self.sample_cache:
            x = self.embeds.float().cpu()
            if x.shape[0] <= 1:
                self.sample_cache[key] = torch.ones(int(x.shape[0]))
            else:
                u, s, _ = torch.linalg.svd(x, full_matrices=False)
                energy = s.square()
                cutoff = int(torch.searchsorted(
                    torch.cumsum(energy, 0) / energy.sum(), 0.9).item()) + 1
                cutoff = max(1, min(cutoff, u.shape[1]))
                self.sample_cache[key] = u[:, :cutoff].square().sum(dim=1) / float(cutoff)
        return self.sample_cache[key]

    def text_embedding_mean(self) -> torch.Tensor:
        key = ("text_mean", self.n_vis)
        if key not in self.turn_cache:
            if hasattr(self.inputs, "text_pos") and hasattr(self.inputs, "full_embeds") and self.inputs.text_pos:
                text = self.inputs.full_embeds[0, self.inputs.text_pos, :].float().cpu()
            else:
                ids = self.inputs["input_ids"][0]
                text = self.model.lm.get_input_embeddings()(ids.unsqueeze(0)).squeeze(0).float().cpu()
            self.turn_cache[key] = F.normalize(text.mean(dim=0), dim=0)
        return self.turn_cache[key]

    def d2_bias(self) -> torch.Tensor:
        key = ("d2_bias", self.n_vis)
        if key not in self.sample_cache:
            if not D2_QWEN25_BIAS_PATH.exists():
                raise RuntimeError(f"D2Pruner calibration bias not found: {D2_QWEN25_BIAS_PATH}")
            payload = torch.load(D2_QWEN25_BIAS_PATH, map_location="cpu", weights_only=True)
            bias = payload.get("bias") if isinstance(payload, dict) else payload
            if bias is None:
                raise RuntimeError(f"D2Pruner calibration bias payload is invalid: {D2_QWEN25_BIAS_PATH}")
            bias = bias.float().flatten()
            if int(bias.shape[0]) != self.n_vis:
                bias = F.interpolate(
                    bias.reshape(1, 1, -1),
                    size=self.n_vis,
                    mode="linear",
                    align_corners=False,
                ).reshape(-1)
            self.sample_cache[key] = bias.clamp_min(1e-6)
        return self.sample_cache[key]

    def hawk_scores(self) -> torch.Tensor:
        key = ("hawk_scores", self.n_vis)
        if key not in self.sample_cache:
            if not HAWK_QWEN25_WEIGHTS_PATH.exists():
                raise RuntimeError(f"HAWK calibration weights not found: {HAWK_QWEN25_WEIGHTS_PATH}")
            if not hasattr(self.model, "hawk_head_scores"):
                raise RuntimeError("HAWK calibrated head scoring is implemented only for calibrated Qwen backend")
            data = json.loads(HAWK_QWEN25_WEIGHTS_PATH.read_text())
            weights = torch.tensor(data.get("weights", []), dtype=torch.float32)
            head_scores = self.model.hawk_head_scores(self.inputs).float()
            if head_scores.numel() == 0:
                self.sample_cache[key] = torch.zeros(self.n_vis)
            else:
                if weights.numel() != head_scores.shape[0]:
                    weights = torch.ones(head_scores.shape[0], dtype=torch.float32)
                weights = weights.clamp_min(0)
                if float(weights.sum()) <= 1e-9:
                    weights = torch.ones(head_scores.shape[0], dtype=torch.float32)
                weights = weights / weights.sum()
                scores = (weights.reshape(-1, 1) * head_scores.cpu()).sum(dim=0)
                self.sample_cache[key] = norm01(scores)
        return self.sample_cache[key]


CACHED_SELECTORS = {
    "AgilePruner": select_agilepruner,
    "D2Pruner": select_d2pruner,
    "DivPrune": select_divprune,
    "FastV": select_fastv,
    "FasterVLM": select_fastervlm,
    "HAWK": select_hawk,
    "ID-Selection": select_idselection,
    "PTP": select_ptp,
    "SVD-Prune": select_svdprune,
    "SparseVILA": select_sparsevila,
    "VisPruner": select_vispruner,
    "ZSPAPrune": select_zspaprune,
}


def cached_keep_mask(method: str, budget: float, shared: SharedPruneCache) -> tuple[list[bool], dict[str, Any]]:
    n_keep = shared.n_keep(budget)
    selector = CACHED_SELECTORS.get(method)
    if selector is not None:
        mask, details = selector(shared, n_keep)
        return mask, {"shared_cache": True, **details}

    return ADAPTERS[method](shared.model, shared.inputs, shared.embeds, budget), {"shared_cache": False}


def hook_required(name: str) -> Callable[[Any, Any, torch.Tensor, float], list[bool]]:
    def _raise(model: Any, inputs: Any, embeds: torch.Tensor, budget: float) -> list[bool]:
        info = METHOD_INFO[name]
        raise OriginalHookRequired(
            f"{name} requires original hook: {info.original_requirement}; "
            f"adapter scope: {info.adapter_scope}"
        )
    return _raise


ADAPTERS: dict[str, Callable[[Any, Any, torch.Tensor, float], list[bool]]] = {
    "FitPrune": hook_required("FitPrune"),
    "SparseVLM": hook_required("SparseVLM"),
}


PROGRESSIVE_ADAPTERS = {"PyramidDrop", "FitPrune", "SparseVLM"}


def turn_prompts(benchmark: str, sample: dict[str, Any], max_turns: int) -> list[dict[str, Any]]:
    candidates = []
    for key in ("turns", "questions", "qas", "conversation", "conversations"):
        value = sample.get(key)
        if isinstance(value, list) and value:
            candidates = value
            break
    if not candidates:
        return [{"turn_id": 0, "prompt": make_prompt(benchmark, sample), "sample": sample}]

    turns = []
    for idx, item in enumerate(candidates[:max_turns]):
        if isinstance(item, str):
            prompt = item
            answer = sample.get("answer", "")
        elif isinstance(item, dict):
            prompt = item.get("question") or item.get("query") or item.get("prompt") or item.get("value") or ""
            answer = item.get("answer", sample.get("answer", ""))
        else:
            continue
        if not prompt:
            continue
        turn_sample = dict(sample)
        turn_sample["question"] = prompt
        turn_sample["answer"] = answer
        if isinstance(item, dict) and "grounding" in item:
            turn_sample["grounding"] = item.get("grounding") or {}
        turns.append({"turn_id": idx, "prompt": make_prompt(benchmark, turn_sample), "sample": turn_sample})
    if not turns:
        return [{"turn_id": 0, "prompt": make_prompt(benchmark, sample), "sample": sample}]
    return turns


def _token_count(model: Any, text: str) -> int:
    tokenizer = getattr(getattr(model, "processor", None), "tokenizer", None)
    if tokenizer is None:
        return max(1, len(str(text).split()))
    try:
        return int(len(tokenizer.encode(str(text), add_special_tokens=False)))
    except TypeError:
        return int(len(tokenizer.encode(str(text))))
    except Exception:
        return max(1, len(str(text).split()))


def _grid_hw_from_grid_thw(grid_thw: Any, n_vis: int) -> tuple[int, int]:
    try:
        grid = grid_thw.detach().cpu().tolist() if hasattr(grid_thw, "detach") else grid_thw
        if isinstance(grid, list) and grid and isinstance(grid[0], list) and len(grid[0]) >= 3:
            h = int(grid[0][1])
            w = int(grid[0][2])
            if h > 0 and w > 0:
                if h * w == n_vis:
                    return h, w
                merge = math.sqrt((h * w) / max(1, n_vis))
                if merge > 1.0:
                    mh = max(1, int(round(h / merge)))
                    mw = max(1, int(round(w / merge)))
                    if mh * mw == n_vis:
                        return mh, mw
    except Exception:
        pass
    h = max(1, int(round(math.sqrt(max(1, n_vis)))))
    while h > 1 and n_vis % h != 0:
        h -= 1
    return h, max(1, int(math.ceil(n_vis / h)))


def _box_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def grounding_metrics(sample: dict[str, Any], keep_mask: list[bool], grid_thw: Any, n_vis: int) -> dict[str, Any] | None:
    grounding = sample.get("grounding") or {}
    boxes = grounding.get("boxes_xyxy") or []
    if not boxes:
        return None
    image_size = grounding.get("image_size") or []
    try:
        image_w = float(image_size[0])
        image_h = float(image_size[1])
    except Exception:
        image_w = image_h = 0.0
    if image_w <= 0 or image_h <= 0:
        return None
    gh, gw = _grid_hw_from_grid_thw(grid_thw, n_vis)
    kept = [i for i, keep in enumerate(keep_mask[:n_vis]) if keep]
    if not kept:
        return {
            "available": True,
            "source": grounding.get("source"),
            "n_boxes": len(boxes),
            "box_recall_any": 0.0,
            "box_center_recall": 0.0,
            "kept_tokens": 0,
            "grid_hw": [gh, gw],
        }
    cell_w = image_w / max(1, gw)
    cell_h = image_h / max(1, gh)
    token_cells = []
    for idx in kept:
        row = idx // gw
        col = idx % gw
        if row >= gh:
            continue
        token_cells.append((
            float(col * cell_w),
            float(row * cell_h),
            float((col + 1) * cell_w),
            float((row + 1) * cell_h),
        ))
    any_hits = 0
    center_hits = 0
    valid_boxes = 0
    for raw_box in boxes:
        try:
            x1, y1, x2, y2 = [float(x) for x in raw_box[:4]]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        valid_boxes += 1
        box = (x1, y1, x2, y2)
        if any(_box_intersects(cell, box) for cell in token_cells):
            any_hits += 1
        for cell in token_cells:
            cx = 0.5 * (cell[0] + cell[2])
            cy = 0.5 * (cell[1] + cell[3])
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                center_hits += 1
                break
    if valid_boxes <= 0:
        return None
    return {
        "available": True,
        "source": grounding.get("source"),
        "n_boxes": valid_boxes,
        "box_recall_any": any_hits / valid_boxes,
        "box_center_recall": center_hits / valid_boxes,
        "kept_tokens": len(kept),
        "grid_hw": [gh, gw],
    }


def _mask_fingerprint(mask: list[bool]) -> str:
    packed = "".join("1" if bool(x) else "0" for x in mask)
    return hashlib.sha1(packed.encode("ascii")).hexdigest()[:12]


def _cost_record(
    *,
    n_vis: int,
    kept_tokens: int,
    prefill_tokens: int | None = None,
    first_generation_tokens: int | None = None,
    total_layers: int,
    n_gen_tokens: int,
    shared_setup_sec: float,
    shared_question_sec: float,
    baseline_wall_sec: float = 0.0,
    method_prefill_wall_sec: float = 0.0,
    method_decode_wall_sec: float = 0.0,
    method_total_wall_sec: float = 0.0,
) -> dict[str, float]:
    decode_kept = max(0, int(kept_tokens))
    prefill_kept = max(0, int(prefill_tokens if prefill_tokens is not None else kept_tokens))
    first_gen_kept = max(0, int(first_generation_tokens if first_generation_tokens is not None else kept_tokens))
    layers = max(0, int(total_layers))
    gen = max(0, int(n_gen_tokens))
    prefill_tl = float(prefill_kept * layers)
    gen_tl = float(layers * (first_gen_kept * min(gen, 1) + decode_kept * max(gen - 1, 0)))
    return {
        "prefill_tl": prefill_tl,
        "gen_tl": gen_tl,
        "total_tl": prefill_tl + gen_tl,
        "n_vis_prefill": float(prefill_kept),
        "n_vis_first_generation": float(first_gen_kept),
        "n_vis_decode": float(decode_kept),
        "n_gen_tokens": float(gen),
        "shared_setup_sec": float(shared_setup_sec),
        "shared_question_sec": float(shared_question_sec),
        "baseline_wall_sec": float(baseline_wall_sec),
        "method_prefill_wall_sec": float(method_prefill_wall_sec),
        "method_decode_wall_sec": float(method_decode_wall_sec),
        "method_total_wall_sec": float(method_total_wall_sec),
    }


def _summarize_cost(rows: list[dict[str, Any]], key: str = "cost") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in COST_FIELDS:
        values = [float((row.get(key) or {}).get(field, 0.0)) for row in rows]
        out[f"{field}_mean"] = sum(values) / len(values) if values else 0.0
        out[f"{field}_per_sample"] = values
    return out


def _budget_tag_from_percent(value: float) -> str:
    pct = int(round(float(value)))
    return f"b{pct:02d}"


def _budget_tag_for_args(args: argparse.Namespace, value: float) -> str:
    if args.budget_mode == "explicit" and len(args.budgets) == 2:
        if abs(float(value) - float(args.budgets[0])) < 1e-4:
            return "b10"
        if abs(float(value) - float(args.budgets[1])) < 1e-4:
            return "b05"
    return _budget_tag_from_percent(value)


def write_full_json_outputs(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    out_dir = Path(args.out_dir)
    by_pair: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in records:
        by_pair.setdefault(
            (row["model"], row["benchmark"], float(row["budget_percent"])),
            [],
        ).append(row)

    multi_budget = len({float(r["budget_percent"]) for r in records}) > 1
    for (model_key, benchmark, budget_percent), rows in sorted(by_pair.items()):
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        baseline_by_turn: dict[tuple[int, int], dict[str, Any]] = {}
        for row in rows:
            turn_key = (int(row["sample_idx"]), int(row["turn_id"]))
            if turn_key not in baseline_by_turn:
                baseline_by_turn[turn_key] = row
        baseline_rows = [baseline_by_turn[k] for k in sorted(baseline_by_turn)]
        baseline_scores = [float(r.get("baseline_score", 0.0)) for r in baseline_rows]
        baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0

        data: dict[str, Any] = {
            "model": model_key,
            "benchmark": benchmark,
                "config": {
                    "runner": "baselines/evaluation/run_mask.py",
                    "scope": "single_image_multi_turn_baseline_six_model",
                    "multi_round_semantics": (
                        "continuous first-turn pruning: each method computes its visual-token "
                        "keep mask once from the first question of an image, then reuses the "
                        "same mask for all later questions of that image."
                    ),
                    "num_images": args.eval_samples,
                    "questions_per_image": args.max_turns_per_sample,
                    "budget_mode": args.budget_mode,
                    "budgets": [budget_percent],
                    "explicit_budgets": args.budgets,
                    "sparsevila_decode_keep_percent_of_total_visual_tokens": args.sparsevila_decode_keep_percent,
                    "bea_multiround_tl_budget": (
                        budget_metadata_for_model(model_key)
                        if args.budget_mode == "bea_multiround_tl" else None
                    ),
                    "methods": args.methods,
                    "shared_compute": {
                    "sample_level": [
                        "visual embeddings derived masks",
                        "SVD leverage",
                        "DivPrune diversity order",
                        "vision received-attention",
                        "visual cosine similarity",
                    ],
                        "turn_level": [
                            "duplicate pruned-generation outputs for identical masks",
                        ],
                        "method_image_level": [
                            "first-turn pruning mask",
                            "pruned shared-prefix KV cache when backend supports build_multiround_cache",
                        ],
                    },
                },
            "n_samples": len(baseline_rows),
            "baseline_mean": baseline_mean,
            "baseline_per_sample": baseline_scores,
            "cost_schema": COST_SCHEMA,
            "baseline_cost": _summarize_cost(baseline_rows, key="baseline_cost"),
            "runtime": {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "record_count": len(rows),
                "ok_record_count": len(ok_rows),
            },
            "methods": {},
        }
        methods = sorted({r["method"] for r in rows})
        for method in methods:
            method_rows = [
                r for r in rows
                if r["method"] == method and r.get("status") == "ok"
            ]
            if not method_rows:
                continue
            scores = [float(r.get("score", 0.0)) for r in method_rows]
            score = sum(scores) / len(scores) if scores else 0.0
            data["methods"][method] = {
                "score": score,
                "preservation": score / baseline_mean if baseline_mean > 0 else 0.0,
                "per_sample": scores,
                "cost": _summarize_cost(method_rows, key="cost"),
            }
        if multi_budget:
            path = out_dir / "full_json" / _budget_tag_for_args(args, budget_percent) / benchmark / f"{model_key}.json"
        else:
            path = out_dir / "full_json" / benchmark / f"{model_key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(path)


def method_allowed_for_model(method: str, model_key: str) -> bool:
    allowed = CALIBRATED_METHOD_MODELS.get(method)
    return allowed is None or model_key in allowed


def run_eval(args: argparse.Namespace, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    out_dir = Path(args.out_dir)
    requested_methods = [
        m for m in args.methods
        if METHOD_INFO[m].status == "executable"
        or args.allow_hook_required
    ]
    for model_key in args.models:
        model_meta = manifest["models"].get(model_key, {})
        if model_meta.get("status") != "available":
            continue
        model_budgets = effective_budgets_for_model(args, model_key)
        methods = [m for m in requested_methods if method_allowed_for_model(m, model_key)]
        skipped_calibrated = [
            m for m in requested_methods
            if m not in methods and m in CALIBRATED_METHOD_MODELS
        ]
        model = load_backend(model_meta["path"], model_meta["backend"])
        for benchmark in args.benchmarks:
            samples = load_samples_ext(
                benchmark,
                args.sample_offset + args.eval_samples,
                args.max_turns_per_sample,
            )
            samples = samples[args.sample_offset: args.sample_offset + args.eval_samples]
            max_tok = max_new_tokens_map(benchmark)
            for sample_idx, sample in enumerate(samples):
                images = sample_images(sample, args.resize_square)
                turns = turn_prompts(benchmark, sample, args.max_turns_per_sample)
                if not turns:
                    continue
                sample_cache: dict[str, Any] = {}
                source_turn = turns[0]
                source_inputs = model.prepare_inputs_any(
                    images, source_turn["prompt"], args.max_pixels, args.multi_max_pixels)
                vis_embeds, grid_thw = model.extract_visual_embeddings(source_inputs)
                sample_cache["visual_embeds_grid"] = (vis_embeds, grid_thw)
                n_vis = int(vis_embeds.shape[0])
                first_shared = SharedPruneCache(model, source_inputs, vis_embeds, grid_thw, sample_cache)
                first_turn_masks: dict[tuple[str, float], dict[str, Any]] = {}
                for method in methods:
                    if method in PROGRESSIVE_ADAPTERS:
                        continue
                    for budget in model_budgets:
                        mask_start = time.perf_counter()
                        try:
                            keep_mask, cache_reuse = cached_keep_mask(method, budget, first_shared)
                            mask_elapsed = time.perf_counter() - mask_start
                            kv_cache_entry = None
                            kv_cache_build_sec = 0.0
                            disable_prefix_kv_cache = os.environ.get(
                                "DUALSIGNAL_DISABLE_PREFIX_KV_CACHE", ""
                            ).lower() in {"1", "true", "yes", "on"}
                            supports_kv_cache = (
                                (not disable_prefix_kv_cache or method == "SparseVILA")
                                and getattr(
                                    model,
                                    "supports_multiround_cache",
                                    hasattr(model, "build_multiround_cache"),
                                )
                            )
                            if supports_kv_cache and hasattr(model, "build_multiround_cache"):
                                kv_start = time.perf_counter()
                                kv_cache_entry = model.build_multiround_cache(
                                    source_inputs, vis_embeds, grid_thw, keep_mask)
                                kv_cache_build_sec = time.perf_counter() - kv_start
                            first_turn_masks[(method, float(budget))] = {
                                "keep_mask": keep_mask,
                                "mask_key": tuple(bool(x) for x in keep_mask),
                                "kept": int(sum(bool(x) for x in keep_mask)),
                                "cache_reuse": cache_reuse,
                                "mask_elapsed_sec": mask_elapsed,
                                "kv_cache_entry": kv_cache_entry,
                                "kv_cache_build_sec": kv_cache_build_sec,
                                "mask_fingerprint": _mask_fingerprint(keep_mask),
                            }
                        except Exception as exc:
                            first_turn_masks[(method, float(budget))] = {
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(limit=8),
                            }

                for turn_idx, turn in enumerate(turns):
                    setup_start = time.perf_counter()
                    if turn_idx == 0:
                        inputs = source_inputs
                    else:
                        inputs = model.prepare_inputs_any(
                            images, turn["prompt"], args.max_pixels, args.multi_max_pixels)
                    pruned_output_cache: dict[tuple[bool, ...], dict[str, Any]] = {}
                    shared_setup_sec = time.perf_counter() - setup_start
                    baseline_start = time.perf_counter()
                    _, baseline_text = model.generate_ids_and_text(inputs, max_tok)
                    baseline_wall_sec = time.perf_counter() - baseline_start
                    baseline_gen_tokens = _token_count(model, baseline_text)
                    baseline_score = eval_metric(benchmark, baseline_text, turn["sample"])
                    baseline_cost = _cost_record(
                        n_vis=n_vis,
                        kept_tokens=n_vis,
                        total_layers=getattr(model, "total_layers", 0),
                        n_gen_tokens=baseline_gen_tokens,
                        shared_setup_sec=shared_setup_sec,
                        shared_question_sec=0.0,
                        baseline_wall_sec=baseline_wall_sec,
                    )
                    def finish_row(row: dict[str, Any]) -> None:
                        records.append(row)
                        with (out_dir / "records.jsonl").open("a") as handle:
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

                    pending_pruned: list[dict[str, Any]] = []
                    for method in methods:
                        for budget in model_budgets:
                            row = {
                                "model": model_key,
                                "benchmark": benchmark,
                                "sample_idx": sample_idx,
                                "turn_id": turn["turn_id"],
                                "method": method,
                                "budget_percent": budget,
                                "budget_mode": args.budget_mode,
                                "budget_metadata": (
                                    budget_metadata_for_model(model_key)
                                    if args.budget_mode == "bea_multiround_tl" else None
                                ),
                                "n_vis": n_vis,
                                "baseline_output": baseline_text,
                                "baseline_score": baseline_score,
                                "baseline_cost": baseline_cost,
                            }
                            if skipped_calibrated:
                                row["calibrated_methods_skipped_for_model"] = skipped_calibrated
                            try:
                                method_start = time.perf_counter()
                                cache_reuse: dict[str, Any] = {"shared_cache": False}
                                if method in PROGRESSIVE_ADAPTERS:
                                    raise OriginalHookRequired(
                                        f"{method} is progressive and cannot satisfy the continuous "
                                        "first-turn pruning/cache-reuse setting in this native runner."
                                    )
                                else:
                                    mask_entry = first_turn_masks.get((method, float(budget)))
                                    if not mask_entry:
                                        raise RuntimeError("first-turn mask was not initialized")
                                    if "error" in mask_entry:
                                        raise RuntimeError(mask_entry["error"])
                                    keep_mask = mask_entry["keep_mask"]
                                    kept = int(mask_entry["kept"])
                                    mask_key = mask_entry["mask_key"]
                                    generation_key = (method, *mask_key) if method == "SparseVILA" else mask_key
                                    mask_fp = str(mask_entry["mask_fingerprint"])
                                    prefill_cost_sec = (
                                        float(mask_entry["mask_elapsed_sec"])
                                        + float(mask_entry.get("kv_cache_build_sec", 0.0))
                                        if turn_idx == 0 else 0.0
                                    )
                                    kv_cache_entry = mask_entry.get("kv_cache_entry")
                                    kv_enabled = kv_cache_entry is not None and hasattr(model, "generate_from_multiround_cache")
                                    cache_reuse = {
                                        **mask_entry["cache_reuse"],
                                        "multi_round_pruning": "first_turn_fixed_mask",
                                        "prune_source_turn": int(source_turn["turn_id"]),
                                        "current_turn": int(turn["turn_id"]),
                                        "first_turn_mask_reuse": turn_idx > 0,
                                        "mask_fingerprint": mask_fp,
                                        "kv_cache_reuse_implemented": bool(kv_enabled),
                                        "prefix_kv_cache_reuse": bool(kv_enabled),
                                        "prefix_kv_cache_hit": bool(kv_enabled and turn_idx > 0),
                                    }
                                    cached_output = pruned_output_cache.get(generation_key)
                                    if cached_output is not None:
                                        output = cached_output["output"]
                                        decode_kept = int(cached_output.get("decode_visual_tokens", kept))
                                        cache_reuse = {
                                            **cache_reuse,
                                            "generation_cache_hit": True,
                                            "generation_cache_source": cached_output["method"],
                                        }
                                        lookup_sec = time.perf_counter() - method_start
                                        method_total_sec = prefill_cost_sec + lookup_sec
                                        method_gen_tokens = _token_count(model, output)
                                        row.update({
                                            "status": "ok",
                                            "kept_tokens": kept,
                                            "decode_kept_tokens": decode_kept,
                                            "multi_round_pruning": "first_turn_fixed_mask",
                                            "prune_source_turn": int(source_turn["turn_id"]),
                                            "mask_fingerprint": mask_fp,
                                            "output": output,
                                            "score": eval_metric(benchmark, output, turn["sample"]),
                                            "grounding": grounding_metrics(
                                                turn["sample"], keep_mask, grid_thw, n_vis),
                                            "cache_reuse": cache_reuse,
                                            "cost": _cost_record(
                                                n_vis=n_vis,
                                                kept_tokens=decode_kept,
                                                prefill_tokens=kept,
                                                first_generation_tokens=kept if method == "SparseVILA" else None,
                                                total_layers=getattr(model, "total_layers", 0),
                                                n_gen_tokens=method_gen_tokens,
                                                shared_setup_sec=shared_setup_sec,
                                                shared_question_sec=0.0,
                                                method_prefill_wall_sec=prefill_cost_sec,
                                                method_decode_wall_sec=lookup_sec,
                                                method_total_wall_sec=method_total_sec,
                                            ),
                                        })
                                        finish_row(row)
                                        continue
                                    else:
                                        pending_pruned.append({
                                            "row": row,
                                            "method": method,
                                            "budget": budget,
                                            "keep_mask": keep_mask,
                                            "mask_key": mask_key,
                                            "kept": kept,
                                            "mask_fingerprint": mask_fp,
                                            "prune_source_turn": int(source_turn["turn_id"]),
                                            "cache_reuse": {**cache_reuse, "generation_cache_hit": False},
                                            "mask_elapsed_sec": prefill_cost_sec,
                                            "kv_cache_entry": kv_cache_entry,
                                            "generation_key": generation_key,
                                        })
                                        continue
                                method_total_sec = time.perf_counter() - method_start
                                method_gen_tokens = _token_count(model, output)
                                row.update({
                                    "status": "ok",
                                    "kept_tokens": kept,
                                    "output": output,
                                    "score": eval_metric(benchmark, output, turn["sample"]),
                                    "cache_reuse": cache_reuse,
                                    "cost": _cost_record(
                                        n_vis=n_vis,
                                        kept_tokens=kept,
                                        total_layers=getattr(model, "total_layers", 0),
                                        n_gen_tokens=method_gen_tokens,
                                        shared_setup_sec=shared_setup_sec,
                                        shared_question_sec=0.0,
                                        method_decode_wall_sec=method_total_sec,
                                        method_total_wall_sec=method_total_sec,
                                    ),
                                })
                                finish_row(row)
                            except OriginalHookRequired as exc:
                                row.update({"status": "original_hook_required", "error": str(exc)})
                                finish_row(row)
                            except Exception as exc:
                                row.update({
                                    "status": "error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc(limit=8),
                                })
                                finish_row(row)

                    unique_pending: dict[tuple[bool, ...], dict[str, Any]] = {}
                    for item in pending_pruned:
                        unique_pending.setdefault(item.get("generation_key", item["mask_key"]), item)
                    unique_items = list(unique_pending.values())
                    generated_by_mask: dict[Any, str | None] = {}
                    generation_errors: dict[Any, str] = {}
                    sparsevila_meta_by_key: dict[Any, dict[str, Any]] = {}
                    try:
                        if unique_items and any(item.get("kv_cache_entry") is not None for item in unique_items):
                            for item in unique_items:
                                gen_start = time.perf_counter()
                                gen_key = item.get("generation_key", item["mask_key"])
                                if item["method"] == "SparseVILA" and hasattr(model, "generate_sparsevila_from_multiround_cache"):
                                    decode_ratio, decode_target = sparsevila_decode_ratio_for_backend(
                                        n_vis,
                                        int(item["kept"]),
                                        float(args.sparsevila_decode_keep_percent),
                                    )
                                    output, sv_meta = model.generate_sparsevila_from_multiround_cache(
                                        item["kv_cache_entry"],
                                        inputs,
                                        vis_embeds,
                                        grid_thw,
                                        max_tok,
                                        decode_keep_ratio=decode_ratio,
                                    )
                                    sv_meta = {
                                        **sv_meta,
                                        "decode_keep_percent_scope": "original_visual_tokens",
                                        "decode_keep_percent_of_total_visual_tokens": float(args.sparsevila_decode_keep_percent),
                                        "decode_target_visual_tokens": int(decode_target),
                                        "backend_decode_keep_ratio_over_prefill": float(decode_ratio),
                                    }
                                    generated_by_mask[gen_key] = output
                                    sparsevila_meta_by_key[gen_key] = sv_meta
                                else:
                                    generated_by_mask[gen_key] = model.generate_from_multiround_cache(
                                        item["kv_cache_entry"],
                                        inputs,
                                        vis_embeds,
                                        grid_thw,
                                        max_tok,
                                    )
                                item["batch_gen_sec"] = time.perf_counter() - gen_start
                        elif unique_items and hasattr(model, "batch_run_pruned"):
                            batch_start = time.perf_counter()
                            outputs = model.batch_run_pruned(
                                inputs,
                                vis_embeds,
                                grid_thw,
                                [item["keep_mask"] for item in unique_items],
                                max_tok,
                            )
                            batch_elapsed = time.perf_counter() - batch_start
                            per_unique_gen_sec = batch_elapsed / max(1, len(unique_items))
                            for item, output in zip(unique_items, outputs):
                                generated_by_mask[item["mask_key"]] = output
                                item["batch_gen_sec"] = per_unique_gen_sec
                        else:
                            for item in unique_items:
                                gen_start = time.perf_counter()
                                generated_by_mask[item.get("generation_key", item["mask_key"])] = model.run_pruned(
                                    inputs, vis_embeds, grid_thw, item["keep_mask"], max_tok)
                                item["batch_gen_sec"] = time.perf_counter() - gen_start
                    except Exception:
                        for item in unique_items:
                            try:
                                gen_start = time.perf_counter()
                                if item.get("kv_cache_entry") is not None:
                                    gen_key = item.get("generation_key", item["mask_key"])
                                    if item["method"] == "SparseVILA" and hasattr(model, "generate_sparsevila_from_multiround_cache"):
                                        decode_ratio, decode_target = sparsevila_decode_ratio_for_backend(
                                            n_vis,
                                            int(item["kept"]),
                                            float(args.sparsevila_decode_keep_percent),
                                        )
                                        output, sv_meta = model.generate_sparsevila_from_multiround_cache(
                                            item["kv_cache_entry"],
                                            inputs,
                                            vis_embeds,
                                            grid_thw,
                                            max_tok,
                                            decode_keep_ratio=decode_ratio,
                                        )
                                        sv_meta = {
                                            **sv_meta,
                                            "decode_keep_percent_scope": "original_visual_tokens",
                                            "decode_keep_percent_of_total_visual_tokens": float(args.sparsevila_decode_keep_percent),
                                            "decode_target_visual_tokens": int(decode_target),
                                            "backend_decode_keep_ratio_over_prefill": float(decode_ratio),
                                        }
                                        generated_by_mask[gen_key] = output
                                        sparsevila_meta_by_key[gen_key] = sv_meta
                                    else:
                                        generated_by_mask[gen_key] = model.generate_from_multiround_cache(
                                            item["kv_cache_entry"],
                                            inputs,
                                            vis_embeds,
                                            grid_thw,
                                            max_tok,
                                        )
                                else:
                                    generated_by_mask[item.get("generation_key", item["mask_key"])] = model.run_pruned(
                                        inputs, vis_embeds, grid_thw, item["keep_mask"], max_tok)
                                item["batch_gen_sec"] = time.perf_counter() - gen_start
                            except Exception as exc:
                                gen_key = item.get("generation_key", item["mask_key"])
                                generated_by_mask[gen_key] = None
                                generation_errors[gen_key] = f"{type(exc).__name__}: {exc}"

                    timing_by_mask = {
                        item.get("generation_key", item["mask_key"]): float(item.get("batch_gen_sec", 0.0))
                        for item in unique_items
                    }
                    source_by_mask: dict[Any, str] = {}
                    for item in pending_pruned:
                        row = item["row"]
                        gen_key = item.get("generation_key", item["mask_key"])
                        output = generated_by_mask.get(gen_key)
                        if output is None:
                            row.update({
                                "status": "error",
                                "error": generation_errors.get(
                                    gen_key, "RuntimeError: batched and fallback pruned generation both failed"
                                ),
                            })
                            finish_row(row)
                            continue
                        duplicate = gen_key in source_by_mask
                        if duplicate:
                            item["cache_reuse"] = {
                                **item["cache_reuse"],
                                **sparsevila_meta_by_key.get(gen_key, {}),
                                "generation_cache_hit": True,
                                "generation_cache_source": source_by_mask[gen_key],
                            }
                            if item.get("kv_cache_entry") is not None:
                                item["cache_reuse"]["generated_from_prefix_kv_cache"] = True
                            gen_sec = 0.0
                        else:
                            source_by_mask[gen_key] = item["method"]
                            pruned_output_cache[gen_key] = {
                                "output": output,
                                "method": item["method"],
                                "decode_visual_tokens": int(
                                    sparsevila_meta_by_key.get(gen_key, {}).get("decode_visual_tokens", item["kept"])
                                ),
                            }
                            gen_sec = timing_by_mask.get(gen_key, 0.0)
                            if item.get("kv_cache_entry") is not None:
                                item["cache_reuse"] = {
                                    **item["cache_reuse"],
                                    "generated_from_prefix_kv_cache": True,
                                }
                                if item["method"] == "SparseVILA":
                                    item["cache_reuse"] = {
                                        **item["cache_reuse"],
                                        **sparsevila_meta_by_key.get(gen_key, {}),
                                        "sparsevila_decode_keep_percent": float(args.sparsevila_decode_keep_percent),
                                        "sparsevila_decode_keep_percent_scope": "original_visual_tokens",
                                    }
                            elif hasattr(model, "batch_run_pruned"):
                                item["cache_reuse"] = {
                                    **item["cache_reuse"],
                                    "batched_pruned_decode": True,
                                }
                        method_total_sec = float(item["mask_elapsed_sec"]) + float(gen_sec)
                        method_gen_tokens = _token_count(model, output)
                        decode_kept = int(item["cache_reuse"].get("decode_visual_tokens", item["kept"]))
                        row.update({
                            "status": "ok",
                            "kept_tokens": item["kept"],
                            "decode_kept_tokens": decode_kept,
                            "multi_round_pruning": "first_turn_fixed_mask",
                            "prune_source_turn": item["prune_source_turn"],
                            "mask_fingerprint": item["mask_fingerprint"],
                            "output": output,
                            "score": eval_metric(benchmark, output, turn["sample"]),
                            "grounding": grounding_metrics(
                                turn["sample"], item["keep_mask"], grid_thw, n_vis),
                            "cache_reuse": item["cache_reuse"],
                            "cost": _cost_record(
                                n_vis=n_vis,
                                kept_tokens=decode_kept,
                                prefill_tokens=item["kept"],
                                first_generation_tokens=item["kept"] if item["method"] == "SparseVILA" else None,
                                total_layers=getattr(model, "total_layers", 0),
                                n_gen_tokens=method_gen_tokens,
                                shared_setup_sec=shared_setup_sec,
                                shared_question_sec=0.0,
                                method_prefill_wall_sec=float(item["mask_elapsed_sec"]),
                                method_decode_wall_sec=float(gen_sec),
                                method_total_wall_sec=method_total_sec,
                            ),
                        })
                        finish_row(row)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[str, str, str, float], list[dict[str, Any]]] = {}
    for row in records:
        key = (row["model"], row["benchmark"], row["method"], float(row["budget_percent"]))
        cells.setdefault(key, []).append(row)
    out = []
    for (model, benchmark, method, budget), rows in sorted(cells.items()):
        ok = [r for r in rows if r.get("status") == "ok"]
        grounding_ok = [
            r.get("grounding") for r in ok
            if isinstance(r.get("grounding"), dict) and r["grounding"].get("available")
        ]
        out.append({
            "model": model,
            "benchmark": benchmark,
            "method": method,
            "budget_percent": budget,
            "valid": len(ok),
            "total": len(rows),
            "accuracy": sum(float(r["score"]) for r in ok) / len(ok) if ok else None,
            "baseline_accuracy": sum(float(r["baseline_score"]) for r in ok) / len(ok) if ok else None,
            "mean_kept_tokens": sum(int(r["kept_tokens"]) for r in ok) / len(ok) if ok else None,
            "grounding_valid": len(grounding_ok),
            "grounding_box_recall_any": (
                sum(float(g["box_recall_any"]) for g in grounding_ok) / len(grounding_ok)
                if grounding_ok else None
            ),
            "grounding_box_center_recall": (
                sum(float(g["box_center_recall"]) for g in grounding_ok) / len(grounding_ok)
                if grounding_ok else None
            ),
            "hook_required": sum(1 for r in rows if r.get("status") == "original_hook_required"),
            "errors": sum(1 for r in rows if r.get("status") == "error"),
        })
    return {"cells": out}


def main() -> int:
    args = parse_args()
    if args.num_images is not None:
        args.eval_samples = args.num_images
    if args.questions_per_image is not None:
        args.max_turns_per_sample = args.questions_per_image
    bad_models = [m for m in args.models if m not in MODEL_TOTAL_LAYERS]
    bad_methods = [m for m in args.methods if m not in METHOD_INFO]
    if bad_models:
        raise ValueError(f"unknown models: {bad_models}")
    if bad_methods:
        raise ValueError(f"unknown methods: {bad_methods}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(out_dir, args.methods, args.models, args)
    if args.list_methods:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    if args.eval_samples <= 0:
        print(f"Wrote adaptation manifest to {out_dir / 'adaptation_manifest.json'}")
        return 0
    (out_dir / "records.jsonl").write_text("", encoding="utf-8")
    records = run_eval(args, manifest)
    summary = summarize(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if args.full_json:
        write_full_json_outputs(args, records)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
