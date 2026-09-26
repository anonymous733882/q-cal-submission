"""Stage-2 attention policy registry for dualsignal multi-round experiments.

The registry is deliberately dualsignal-local.  BEA artifacts are treated as
read-only provenance for the old original-attention policies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import os
from pathlib import Path
from typing import Iterable


DUALSIGNAL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = DUALSIGNAL_ROOT.parent


@dataclass(frozen=True)
class Stage2Policy:
    name: str
    model_key: str
    total_layers: int
    kind: str
    layers: tuple[int, ...]
    weights: tuple[float, ...]
    source: str
    notes: str = ""

    @property
    def deepest_layer(self) -> int:
        return max(self.layers) if self.layers else -1

    @property
    def prune_start_layer(self) -> int:
        """Stage-2 pruning starts at the deepest scoring layer."""
        return self.deepest_layer

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        data["layers"] = list(self.layers)
        data["weights"] = list(self.weights)
        data["deepest_layer"] = self.deepest_layer
        data["prune_start_layer"] = self.prune_start_layer
        return data


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


# Old model-specific scoring policies used for original_attention.
# For max_0_26, layers are Python half-open range(0, 26), i.e. L0..L25.
ORIGINAL_ATTENTION_STRATEGIES = {
    "qwen2vl2b": {
        "strategy": "single_L20",
        "source": "BEA multi_round_benchmark_v4 stage layer recorded in baselines/evaluation/run_mask.py: S=L20",
        "notes": "Six-Qwen compatibility entry for original_attention baseline.",
    },
    "qwen2vl": {
        "strategy": "single_L20",
        "source": "BEA multi_round_benchmark_v4 stage layer recorded in baselines/evaluation/run_mask.py: S=L20",
        "notes": "Six-Qwen compatibility entry for original_attention baseline.",
    },
    "qwen25vl3b": {
        "strategy": "single_L24",
        "source": "BEA multi_round_benchmark_v4 stage layer recorded in baselines/evaluation/run_mask.py: S=L24",
        "notes": "Six-Qwen compatibility entry for original_attention baseline.",
    },
    "qwen25vl": {
        "strategy": "single_L22",
        "source": "BEA multi_round_benchmark_v4 config: S=L22",
        "notes": "Qwen2.5-VL was absent from scoring_sweep_v3 final table; use v4 model-specific old S.",
    },
    "qwen3vl4b": {
        "strategy": "single_L26",
        "source": "BEA multi_round_benchmark_v4 stage layer recorded in baselines/evaluation/run_mask.py: S=L26",
        "notes": "Six-Qwen compatibility entry for original_attention baseline.",
    },
    "qwen3vl": {
        "strategy": "single_L29",
        "source": "BEA scoring_sweep_v3 final selected strategy",
        "notes": "v4 multi-round config had S=L30; final scoring sweep selected single_L29.",
    },
    "internvl3-8b": {
        "strategy": "single_L23",
        "source": "BEA scoring_sweep_v3 final selected strategy",
        "notes": "",
    },
    "internvl3.5-8b": {
        "strategy": "single_L23",
        "source": "BEA scoring_sweep_v3 final selected strategy",
        "notes": "",
    },
    "llava-7b": {
        "strategy": "single_L10",
        "source": "BEA multi_round_benchmark_v4 config: S=L10",
        "notes": "LLaVA-1.5-7B was absent from scoring_sweep_v3 final table; use v4 model-specific old S.",
    },
    "llava-13b": {
        "strategy": "max_0_26",
        "source": "BEA scoring_sweep_v3 final selected strategy",
        "notes": "This is the one old-policy model using multi-layer max attention.",
    },
}


# Model-specific calibrated global scoring rules.  Do not silently transfer the
# Qwen2.5-VL fit to other architectures: each model must have its own calibrated
# layers and weights here before calibrated_global experiments are valid.
CALIBRATED_GLOBAL_POLICIES = {
    "qwen25vl": {
        "name": "calibrated_global:qwen25vl_core8_l20_n8_b10_baseline_correct",
        "layers": (1, 15, 18, 19, 20),
        "weights": (
            0.2706040143966675,
            0.044611040502786636,
            0.4215523600578308,
            0.05510523542761803,
            0.20812734961509705,
        ),
        "source": (
            "dualsignal/results/capability_global_core8_l20_n8_baseline_correct/"
            "suite.json global_core8 by_budget 10.0"
        ),
        "notes": (
            "Qwen2.5-VL b10 core8 calibration over eight capability benchmarks; "
            "baseline-correct calibration subset, 8 samples per benchmark."
        ),
    },
    # Non-Qwen2.5 model-specific b10 policies are not present in local results.
    # Existing crossmodel_*_d75_n32_b20 suites are 20% calibration and must not
    # be used for 10% visual-token retention experiments.
}


def _load_calibrated_global_overrides() -> dict[str, dict[str, object]]:
    path = WORKSPACE / "qcal/runners/calibration/stage2_calibrated_global_overrides.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to load calibrated_global overrides from {path}: {exc}") from exc
    policies = data.get("calibrated_global_policies", data)
    if not isinstance(policies, dict):
        raise RuntimeError(f"Invalid calibrated_global override payload in {path}")
    return policies


CALIBRATED_GLOBAL_POLICIES.update(_load_calibrated_global_overrides())


def _parse_strategy(strategy: str, total_layers: int) -> tuple[str, tuple[int, ...], tuple[float, ...]]:
    if strategy.startswith("single_L"):
        layer = int(strategy.split("L", 1)[1])
        return "single", (layer,), (1.0,)
    if strategy.startswith("max_0_"):
        end = int(strategy.rsplit("_", 1)[1])
        return "max", tuple(range(end)), tuple(1.0 for _ in range(end))
    raise ValueError(f"Unsupported original_attention strategy: {strategy}")


def _check_layers(layers: Iterable[int], total_layers: int) -> None:
    bad = [layer for layer in layers if layer < 0 or layer >= total_layers]
    if bad:
        raise ValueError(f"Layer(s) out of range for {total_layers} layers: {bad}")


def original_attention_policy(model_key: str) -> Stage2Policy:
    total_layers = MODEL_TOTAL_LAYERS[model_key]
    entry = ORIGINAL_ATTENTION_STRATEGIES[model_key]
    strategy = entry["strategy"]
    kind, layers, weights = _parse_strategy(strategy, total_layers)
    _check_layers(layers, total_layers)
    return Stage2Policy(
        name=f"original_attention:{strategy}",
        model_key=model_key,
        total_layers=total_layers,
        kind=kind,
        layers=layers,
        weights=weights,
        source=entry["source"],
        notes=entry.get("notes", ""),
    )


def calibrated_global_policy(model_key: str) -> Stage2Policy:
    total_layers = MODEL_TOTAL_LAYERS[model_key]
    if model_key not in CALIBRATED_GLOBAL_POLICIES:
        raise ValueError(
            f"Missing model-specific calibrated_global policy for {model_key}. "
            "Do not reuse Qwen2.5-VL calibrated weights by relative-depth mapping."
        )
    entry = CALIBRATED_GLOBAL_POLICIES[model_key]
    merged: dict[int, float] = {}
    for layer, weight in zip(entry["layers"], entry["weights"]):
        merged[layer] = merged.get(layer, 0.0) + weight
    layers = tuple(sorted(merged))
    weights = tuple(merged[layer] for layer in layers)
    _check_layers(layers, total_layers)
    return Stage2Policy(
        name=entry["name"],
        model_key=model_key,
        total_layers=total_layers,
        kind="weighted",
        layers=layers,
        weights=weights,
        source=entry["source"],
        notes=entry.get("notes", ""),
    )


def all_policies() -> list[Stage2Policy]:
    policies: list[Stage2Policy] = []
    for model_key in MODEL_TOTAL_LAYERS:
        policies.append(original_attention_policy(model_key))
        policies.append(calibrated_global_policy(model_key))
    return policies


def markdown_report(policies: list[Stage2Policy]) -> str:
    lines = [
        "# Stage-2 Policy Registry",
        "",
        "BEA is read-only provenance. New experiments should import this dualsignal-local registry.",
        "",
        "| Model | Policy | Kind | Scoring layers | Weights | Prune starts | Source |",
        "|---|---|---|---|---|---:|---|",
    ]
    for policy in policies:
        layer_text = ", ".join(f"L{x}" for x in policy.layers)
        weight_text = ", ".join(f"{w:.3f}" for w in policy.weights)
        lines.append(
            f"| {policy.model_key} | {policy.name} | {policy.kind} | "
            f"{layer_text} | {weight_text} | L{policy.prune_start_layer} | "
            f"{policy.source} |"
        )
    lines.append("")
    lines.append(
        "For `max_0_26`, the scoring layers are L0 through L25 and stage-2 pruning starts at L25."
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown.")
    parser.add_argument("--out", default=None, help="Optional output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policies = all_policies()
    if args.json:
        text = json.dumps([p.to_json() for p in policies], indent=2)
    else:
        text = markdown_report(policies)
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = WORKSPACE / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"Wrote {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
