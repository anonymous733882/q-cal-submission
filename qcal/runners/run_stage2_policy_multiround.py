#!/usr/bin/env python3
"""Run BEA multi-round benchmark with dualsignal stage-2 policies.

BEA is imported read-only. This launcher patches the runtime configuration so
the old `ours` method can be evaluated with either:

- original_attention: model-specific old attention policy
- calibrated_global: calibrated weighted attention policy
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


DUALSIGNAL_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = DUALSIGNAL_DIR.parent
BEA_DIR = WORKSPACE / "qcal_support"

sys.path.insert(0, str(DUALSIGNAL_DIR / "runners"))
sys.path.insert(0, str(BEA_DIR))
sys.path.insert(0, str(BEA_DIR / "strategy_test"))

import multi_round_benchmark as bea_mr  # noqa: E402
from stage2_policies import calibrated_global_policy, original_attention_policy  # noqa: E402
from count_relaxed_scoring import relaxed_count_score  # noqa: E402

STAGE1_DUAL_WEIGHT = float(os.environ.get("DUALSIGNAL_STAGE1_WEIGHT", "0.5"))


def _patch_relaxed_count_scorer() -> None:
    original_score_gqa = bea_mr.score_gqa

    def patched_score_gqa(pred, gt_answer):
        relaxed = relaxed_count_score(pred, gt_answer)
        if relaxed is not None:
            return float(relaxed)
        return original_score_gqa(pred, gt_answer)

    bea_mr.score_gqa = patched_score_gqa


def _combine_policy_scores(policy, layer_scores: dict[int, np.ndarray]) -> np.ndarray:
    if policy.kind == "single":
        return layer_scores[policy.layers[0]]
    if policy.kind == "max":
        return np.maximum.reduce([layer_scores[layer] for layer in policy.layers])
    if policy.kind == "weighted":
        scores = None
        for layer, weight in zip(policy.layers, policy.weights):
            weighted = float(weight) * layer_scores[layer]
            scores = weighted if scores is None else scores + weighted
        if scores is None:
            raise ValueError("weighted policy has no layers")
        return scores
    raise ValueError(f"unsupported policy kind: {policy.kind}")


def _policy_importance(lm, full_embeds, vis_pos, text_pos, policy):
    layer_scores = {
        layer: bea_mr._get_importance_at_layer(lm, layer, full_embeds, vis_pos, text_pos)
        for layer in policy.layers
    }
    return _combine_policy_scores(policy, layer_scores)


def _causal_mask(seq_len: int, device, dtype):
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def _dual_signal_stage1_select_idselection_diversity(
    l0_importance: np.ndarray,
    vit_importance: np.ndarray,
    vis_embeds: torch.Tensor,
    k: int,
    gamma: float = 20.0,
) -> list[int]:
    """Stage-1 dual signal with ID-Selection-style Gaussian suppression.

    Keeps the validation-selected Stage-1 mixture of shallow text-to-visual
    attention and ViT received attention, but replaces multiplicative MMR with
    ID-Selection's subtractive Gaussian diversity update.
    """
    n = len(l0_importance)
    if k >= n:
        return list(range(n))

    w = STAGE1_DUAL_WEIGHT
    combined = w * bea_mr._norm01(l0_importance) + (1.0 - w) * bea_mr._norm01(vit_importance)
    scores = torch.tensor(combined, dtype=torch.float32)

    emb = F.normalize(vis_embeds.float().cpu(), dim=-1)
    cos_sim = emb @ emb.T
    dist_sq = (1.0 - cos_sim).clamp(min=0.0).pow(2)
    weights = torch.exp(-float(gamma) * dist_sq)

    current_scores = scores.clone()
    selected: list[int] = []
    available = torch.ones(n, dtype=torch.bool)
    for _ in range(k):
        masked = current_scores.clone()
        masked[~available] = -float("inf")
        best = int(masked.argmax().item())
        selected.append(best)
        available[best] = False
        selected_score = float(current_scores[best])
        current_scores -= weights[best] * selected_score
        current_scores.clamp_(min=0.0)

    return sorted(selected)


def _patch_stage1_diversity() -> None:
    bea_mr._dual_signal_stage1_select = _dual_signal_stage1_select_idselection_diversity


def _patch_dual_stage_select(policy):
    original = bea_mr.dual_stage_select

    def patched(lm, full_embeds, vis_pos, text_pos, n_vis, S,
                vit_importance=None, vis_embeds=None, scoring="single"):
        if scoring != "policy":
            return original(
                lm, full_embeds, vis_pos, text_pos, n_vis, S,
                vit_importance=vit_importance, vis_embeds=vis_embeds, scoring=scoring)

        k1 = max(1, int(n_vis * bea_mr.STAGE1_FRAC))
        k2 = max(1, int(n_vis * bea_mr.STAGE2_FRAC))

        imp0 = bea_mr._get_importance_at_layer(lm, 0, full_embeds, vis_pos, text_pos)
        if vit_importance is not None and vis_embeds is not None:
            stage1_idx = _dual_signal_stage1_select_idselection_diversity(
                imp0, vit_importance, vis_embeds, k1)
        else:
            stage1_idx = np.argsort(imp0)[::-1][:k1].tolist()
        stage1_set = set(stage1_idx)

        surviving_vis_pos = [vis_pos[i] for i in sorted(stage1_set)]
        imp = _policy_importance(lm, full_embeds, surviving_vis_pos, text_pos, policy)
        stage2_local = np.argsort(imp)[::-1][:k2]
        surviving_orig = sorted(stage1_set)
        stage2_set = set(surviving_orig[j] for j in stage2_local)
        return [i in stage2_set for i in range(n_vis)], stage1_set, stage2_set

    bea_mr.dual_stage_select = patched


def _patch_per_question_prune_generate(policy):
    original = bea_mr._per_question_prune_generate

    def patched_prefill_shared_layers(lm, shared_embeds, S, position_ids=None):
        from transformers.cache_utils import DynamicCache

        device = shared_embeds.device
        dtype = shared_embeds.dtype
        if position_ids is None:
            position_ids = torch.arange(shared_embeds.shape[1], device=device).unsqueeze(0)
        pos_kwargs = bea_mr._compute_pos_embeddings(lm, position_ids, shared_embeds, device, dtype)
        attn_mask = None
        if getattr(lm.config, "_attn_implementation", None) not in (
            "flash_attention_2",
            "flash_attention_3",
            "flash_attention_4",
        ):
            attn_mask = _causal_mask(shared_embeds.shape[1], device, dtype)

        cache = DynamicCache()
        h = shared_embeds
        with torch.no_grad():
            for i in range(S):
                h = lm.layers[i](
                    hidden_states=h,
                    attention_mask=attn_mask,
                    past_key_values=cache,
                    use_cache=True,
                    **pos_kwargs)
        return h, cache

    def patched(lm, lm_head, h_shared, shared_cache,
                question_embeds, vis_pos_in_shared,
                n_vis_stage1, S, stage2_frac,
                eos_id, max_new_tokens=32,
                shared_pos_ids=None, question_pos_ids=None,
                scoring="single"):
        if scoring != "policy":
            return original(
                lm, lm_head, h_shared, shared_cache,
                question_embeds, vis_pos_in_shared,
                n_vis_stage1, S, stage2_frac,
                eos_id, max_new_tokens=max_new_tokens,
                shared_pos_ids=shared_pos_ids, question_pos_ids=question_pos_ids,
                scoring=scoring)

        from transformers.cache_utils import DynamicCache

        active_policy = getattr(bea_mr, "_active_stage2_policy", None) or policy
        stage2_frac = float(getattr(bea_mr, "_active_stage2_frac", stage2_frac))

        N = len(lm.layers)
        device = h_shared.device
        dtype = h_shared.dtype
        n_shared = h_shared.shape[1]
        n_question = question_embeds.shape[1]

        if shared_pos_ids is None:
            shared_pos_ids = torch.arange(n_shared, device=device).unsqueeze(0)
        if question_pos_ids is None:
            question_pos_ids = torch.arange(n_shared, n_shared + n_question,
                                            device=device).unsqueeze(0)

        q_cache = bea_mr._clone_cache(shared_cache)
        q_pos_kwargs = bea_mr._compute_pos_embeddings(
            lm, question_pos_ids, question_embeds, device, dtype)

        attn_impl = lm.config._attn_implementation
        if attn_impl in ("flash_attention_2", "flash_attention_3", "flash_attention_4"):
            step1_mask = None
        else:
            kv_len = n_shared + n_question
            step1_mask = torch.full(
                (n_question, kv_len), torch.finfo(dtype).min,
                device=device, dtype=dtype)
            for qi in range(n_question):
                step1_mask[qi, : n_shared + qi + 1] = 0.0
            step1_mask = step1_mask.unsqueeze(0).unsqueeze(0)

        h_q = question_embeds
        policy_layers = set(int(layer) for layer in active_policy.layers)
        score_cache = getattr(bea_mr, "_stage2_score_cache", None)
        score_key = (
            active_policy.kind,
            tuple(int(x) for x in active_policy.layers),
            tuple(round(float(x), 12) for x in active_policy.weights),
            int(S),
            int(n_vis_stage1),
        )
        cached_imp = score_cache.get(score_key) if isinstance(score_cache, dict) else None
        layer_scores: dict[int, np.ndarray] = {}
        with torch.no_grad():
            for i in range(S):
                if cached_imp is None and i in policy_layers:
                    layer_scores[i] = bea_mr._importance_from_hidden_and_cache(
                        lm, i, h_q, shared_cache, vis_pos_in_shared)
                h_q = lm.layers[i](
                    hidden_states=h_q,
                    attention_mask=step1_mask,
                    past_key_values=q_cache,
                    use_cache=True,
                    **q_pos_kwargs)

        if cached_imp is None:
            if S in policy_layers:
                h_for_score = torch.cat([h_shared, h_q], dim=1)
                vis_pos_set_for_score = set(vis_pos_in_shared)
                text_pos_full = [
                    p for p in range(n_shared + n_question)
                    if p not in vis_pos_set_for_score
                ]
                layer_scores[S] = bea_mr._importance_from_hidden(
                    lm, S, h_for_score, vis_pos_in_shared, text_pos_full)
                del h_for_score
            missing = [layer for layer in active_policy.layers if layer not in layer_scores]
            if missing:
                raise ValueError(
                    f"policy layer(s) {missing} unavailable before prune_start L{S}")
            imp = _combine_policy_scores(active_policy, layer_scores)
            if isinstance(score_cache, dict):
                score_cache[score_key] = imp
        else:
            imp = cached_imp

        k2 = max(1, int(n_vis_stage1 * (stage2_frac / bea_mr.STAGE1_FRAC)))
        k2 = min(k2, n_vis_stage1)
        stage2_local = sorted(np.argsort(imp)[::-1][:k2].tolist())

        vis_pos_set = set(vis_pos_in_shared)
        vis_keep = [vis_pos_in_shared[j] for j in stage2_local]
        non_vis = [p for p in range(n_shared + n_question) if p not in vis_pos_set]
        keep_positions = sorted(set(vis_keep) | set(non_vis))
        keep_t = torch.tensor(keep_positions, device=device, dtype=torch.long)

        h_full = torch.cat([h_shared, h_q], dim=1)
        h_pruned = h_full[:, keep_t, :]

        pruned_cache = DynamicCache()
        for layer_kv in q_cache.layers:
            pruned_cache.update(
                layer_kv.keys[:, :, keep_t, :],
                layer_kv.values[:, :, keep_t, :],
                layer_idx=len(pruned_cache.layers))
        del q_cache, h_full

        if shared_pos_ids.dim() == 3:
            full_pos = torch.cat([shared_pos_ids.to(device), question_pos_ids.to(device)], dim=2)
            pruned_pos = full_pos[:, :, keep_t]
        else:
            full_pos = torch.cat([shared_pos_ids.to(device), question_pos_ids.to(device)], dim=1)
            pruned_pos = full_pos[:, keep_t]
        pos_kwargs_pruned = bea_mr._compute_pos_embeddings(lm, pruned_pos, h_pruned, device, dtype)
        pruned_mask = None
        if getattr(lm.config, "_attn_implementation", None) not in (
            "flash_attention_2",
            "flash_attention_3",
            "flash_attention_4",
        ):
            pruned_mask = _causal_mask(h_pruned.shape[1], device, dtype)

        with torch.no_grad():
            for i in range(S, N):
                h_pruned = lm.layers[i](
                    hidden_states=h_pruned,
                    attention_mask=pruned_mask,
                    past_key_values=pruned_cache,
                    use_cache=True,
                    **pos_kwargs_pruned)

        h_final = lm.norm(h_pruned)
        logits = lm_head(h_final[:, -1:, :])
        next_id = logits.argmax(dim=-1).squeeze().item()
        generated = [next_id]

        if next_id == eos_id:
            return generated

        cur_id = torch.tensor([[next_id]], device=device)
        cur_pos = pruned_pos[:, :, -1:] + 1 if pruned_pos.dim() == 3 else pruned_pos[:, -1:] + 1
        for _ in range(max_new_tokens - 1):
            cur_emb = lm.embed_tokens(cur_id)
            pos_kwargs = bea_mr._compute_pos_embeddings(lm, cur_pos, cur_emb, device, dtype)
            h = cur_emb
            with torch.no_grad():
                for i in range(N):
                    h = lm.layers[i](
                        hidden_states=h,
                        attention_mask=None,
                        past_key_values=pruned_cache,
                        use_cache=True,
                        **pos_kwargs)
                h = lm.norm(h)
                logits = lm_head(h[:, -1:, :])
                next_id = logits.argmax(dim=-1).squeeze().item()
            generated.append(next_id)
            if next_id == eos_id:
                break
            cur_id = torch.tensor([[next_id]], device=device)
            cur_pos = cur_pos + 1
        return generated

    bea_mr._prefill_shared_layers = patched_prefill_shared_layers
    bea_mr._per_question_prune_generate = patched


def _policy_for(name: str, model_key: str):
    if name == "original_attention":
        return original_attention_policy(model_key)
    if name == "calibrated_global":
        return calibrated_global_policy(model_key)
    raise ValueError(f"unknown policy: {name}")


def _apply_policy_to_config(model_key: str, policy_name: str):
    policy = _policy_for(policy_name, model_key)
    cfg = dict(bea_mr.MODEL_CONFIG[model_key])
    cfg["S"] = policy.prune_start_layer
    cfg["scoring"] = "policy"
    _patch_stage1_diversity()
    _patch_dual_stage_select(policy)
    _patch_per_question_prune_generate(policy)

    N = cfg["N"]
    S = cfg["S"]
    cfg["cost"] = bea_mr.STAGE1_FRAC * S + bea_mr.STAGE2_FRAC * (N - S)
    cfg["Y"] = cfg["cost"] / N
    cfg["policy_name"] = policy.name
    cfg["policy_kind"] = policy.kind
    cfg["policy_layers"] = list(policy.layers)
    cfg["policy_weights"] = list(policy.weights)
    cfg["prune_start_layer"] = policy.prune_start_layer
    cfg["policy_only"] = True
    bea_mr.MODEL_CONFIG[model_key] = cfg
    return policy, cfg


def _load_baseline_scores(out_dir: str, policy_name: str, benchmark: str, model_key: str):
    """Reuse the first policy's full-baseline scores for later policies."""
    if policy_name == "original_attention":
        return None
    baseline_path = Path(out_dir) / "original_attention" / benchmark / f"{model_key}.json"
    if not baseline_path.exists():
        return None
    try:
        data = json.loads(baseline_path.read_text())
    except Exception:
        return None
    scores = data.get("baseline_per_sample")
    if not isinstance(scores, list) or not scores:
        return None
    return [float(x) for x in scores]


def _patch_policy_only_outputs(policy_name: str, policy):
    """Keep BEA's current-policy path, but suppress unrelated baselines.

    BEA names the active dual-stage method `ours` internally.  For this
    dualsignal experiment, that method is the requested stage-2 policy
    (`original_attention` or `calibrated_global`), so outputs are renamed.
    """

    def _no_indices(*args, **kwargs):
        return []

    def _disabled(*args, **kwargs):
        raise RuntimeError("disabled in dualsignal policy-only run")

    def _disabled_prefill(*args, **kwargs):
        return None, None

    # These selectors/prefills feed unrelated comparison methods.  Returning
    # empty PACT indices keeps setup code alive until prefill is skipped.
    bea_mr._pact_select = _no_indices
    for name in (
        "_svdprune_select",
        "_divprune_select",
        "_vispruner_select",
        "_fastv_select",
        "_zspaprune_select",
        "_agilepruner_select",
        "_idselection_select",
        "_d2pruner_select",
        "_ptp_select",
        "_hawk_select",
        "_vscore_l2_select",
        "_sparsevila_per_question_generate",
        "_generate_from_shared_cache",
        "_progressive_prefill",
    ):
        if hasattr(bea_mr, name):
            setattr(bea_mr, name, _disabled)
    bea_mr._prefill_all_layers = _disabled_prefill

    def policy_only_report(model_name, cfg, results, diversity=None, costs=None):
        baseline = float(np.mean(results["baseline"])) if results.get("baseline") else 0.0
        score = float(np.mean(results["ours"])) if results.get("ours") else 0.0
        preservation = score / baseline if baseline > 0 else 0.0
        print("\n" + "=" * 60)
        print(f"Policy-only multi-round report: {model_name}")
        print("=" * 60)
        print(f"Baseline: {baseline:.3f}")
        print(f"{policy_name}: {score:.3f} ({preservation:.1%} of baseline)")

    def policy_only_save(model_name, cfg, results, diversity, out_dir, costs=None):
        os.makedirs(out_dir, exist_ok=True)
        baseline_values = results.get("baseline", [])
        policy_values = results.get("ours", [])
        baseline = float(np.mean(baseline_values)) if baseline_values else 0.0
        score = float(np.mean(policy_values)) if policy_values else 0.0
        entry = {
            "score": score,
            "preservation": score / baseline if baseline > 0 else 0.0,
            "per_sample": [float(x) for x in policy_values],
            "source_internal_method": "ours",
            "stage2_policy": policy.to_json(),
        }
        if costs and "ours" in costs:
            entry["cost"] = {
                "prefill_tl_mean": float(np.mean(costs["ours"]["prefill_tl"])),
                "gen_tl_mean": float(np.mean(costs["ours"]["gen_tl"])),
                "total_tl_mean": float(np.mean(costs["ours"]["total_tl"])),
                "n_vis_decode_mean": float(np.mean(costs["ours"]["n_vis_decode"])),
                "n_gen_tokens_mean": float(np.mean(costs["ours"]["n_gen_tokens"])),
            }

        data = {
            "model": model_name,
            "config": {k: v for k, v in cfg.items() if k != "type"},
            "n_samples": len(baseline_values),
            "baseline_mean": baseline,
            "baseline_per_sample": [float(x) for x in baseline_values],
            "methods": {policy_name: entry},
        }
        if costs and "baseline" in costs:
            data["baseline_cost"] = {
                "total_tl_mean": float(np.mean(costs["baseline"]["total_tl"])),
                "n_vis_decode_mean": float(np.mean(costs["baseline"]["n_vis_decode"])),
                "n_gen_tokens_mean": float(np.mean(costs["baseline"]["n_gen_tokens"])),
            }

        out_path = Path(out_dir) / f"{model_name}.json"
        out_path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Saved -> {out_path}", flush=True)

    bea_mr.print_report = policy_only_report
    bea_mr.save_results = policy_only_save


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=["original_attention", "calibrated_global"])
    parser.add_argument("--model", required=True, choices=list(bea_mr.MODEL_CONFIG.keys()))
    parser.add_argument("--benchmark", default="gqa",
                        choices=["textvqa", "gqa", "pope", "docvqa", "vqav2", "clevr", "visual7w"])
    parser.add_argument("--num_images", type=int, default=2000)
    parser.add_argument("--questions_per_image", type=int, default=15)
    parser.add_argument("--out_dir", default="outputs/stage2_policy_multiround")
    parser.add_argument("--skip_cclass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _patch_relaxed_count_scorer()
    policy, cfg = _apply_policy_to_config(args.model, args.policy)
    baseline_scores = _load_baseline_scores(args.out_dir, args.policy, args.benchmark, args.model)
    if baseline_scores is not None:
        cfg["baseline_scores"] = baseline_scores
        print(
            f"Reusing {len(baseline_scores)} baseline scores from original_attention.",
            flush=True,
        )
    _patch_policy_only_outputs(args.policy, policy)
    out_dir = Path(args.out_dir) / args.policy
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using policy: {policy.to_json()}", flush=True)
    print(f"Runtime cfg: S={cfg['S']} scoring={cfg['scoring']} Y={cfg['Y']:.4f}", flush=True)

    sys.argv = [
        "multi_round_benchmark.py",
        "--model", args.model,
        "--benchmark", args.benchmark,
        "--num_images", str(args.num_images),
        "--questions_per_image", str(args.questions_per_image),
        "--out_dir", str(out_dir),
    ]
    if args.skip_cclass:
        sys.argv.append("--skip_cclass")
    bea_mr.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
