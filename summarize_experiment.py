#!/usr/bin/env python3
"""Validate and aggregate main-evaluation accuracy, fidelity, and cost."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from baselines.evaluation.run_experiment import METHODS
from run_main_experiment import BENCHMARKS, MODELS, ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['qcal', 'baseline'], required=True)
    parser.add_argument('--results-dir', type=Path, default=None)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--benchmarks', nargs='+', choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument('--budgets', nargs='+', type=int, choices=[5, 10], default=[5, 10])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--out', type=Path, default=None)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError('Cannot aggregate an empty sample set')
    return float(statistics.fmean(values))


def _time_seconds(cost: dict, kind: str, *, is_baseline: bool) -> float:
    if kind == 'qcal':
        return float(cost['policy_total_wall_sec_mean'])
    shared = float(cost['shared_setup_sec_mean']) + float(cost['shared_question_sec_mean'])
    private = 'baseline_wall_sec_mean' if is_baseline else 'method_total_wall_sec_mean'
    return shared + float(cost[private])


def summarize(args: argparse.Namespace) -> dict:
    results_dir = args.results_dir or ROOT / ('outputs/main' if args.kind == 'qcal' else 'outputs/baselines')
    expected_methods = ['qcal'] if args.kind == 'qcal' else list(args.methods)
    cells = []
    for model in args.models:
        for benchmark in args.benchmarks:
            for budget in args.budgets:
                tag = f'b{budget:02d}'
                path = (results_dir / tag / 'qcal' / benchmark / f'{model}.json'
                        if args.kind == 'qcal' else results_dir / tag / benchmark / f'{model}.json')
                data = json.loads(path.read_text(encoding='utf-8'))
                if data['model'] != model or data['benchmark'] != benchmark:
                    raise ValueError(f'Model/benchmark mismatch: {path}')
                n_samples = int(data['n_samples'])
                baseline_scores = [float(x) for x in data['baseline_per_sample']]
                if n_samples < 1 or len(baseline_scores) != n_samples:
                    raise ValueError(f'Incomplete full-token records: {path}')
                baseline_acc = _mean(baseline_scores)
                if abs(baseline_acc - float(data['baseline_mean'])) > 1e-6:
                    raise ValueError(f'Incorrect baseline mean: {path}')
                for method in expected_methods:
                    key = f'{tag}/qcal' if args.kind == 'qcal' else method
                    entry = data['methods'][key]
                    reference_cost = entry.get('baseline_cost', data['baseline_cost'])
                    baseline_tl = float(reference_cost['total_tl_mean'])
                    baseline_time = _time_seconds(reference_cost, args.kind, is_baseline=True)
                    if baseline_tl <= 0 or baseline_time <= 0:
                        raise ValueError(f'Missing baseline cost for {method}: {path}')
                    scores = [float(x) for x in entry['per_sample']]
                    if len(scores) != n_samples:
                        raise ValueError(f'Incomplete {method} records: {path}')
                    acc = _mean(scores)
                    if abs(acc - float(entry['score'])) > 1e-6:
                        raise ValueError(f'Incorrect {method} mean: {path}')
                    cost = entry['cost']
                    method_tl = float(cost['total_tl_mean'])
                    method_time = _time_seconds(cost, args.kind, is_baseline=False)
                    if method_tl <= 0 or method_time <= 0:
                        raise ValueError(f'Missing {method} cost: {path}')
                    cells.append({
                        'model': model, 'benchmark': benchmark, 'budget': tag,
                        'method': method, 'n_questions': n_samples,
                        'accuracy': acc, 'full_token_accuracy': baseline_acc,
                        'fidelity_percent': 100.0 * acc / baseline_acc if baseline_acc > 0 else None,
                        'relative_token_layer_cost_percent': 100.0 * method_tl / baseline_tl,
                        'within_run_speedup': baseline_time / method_time,
                    })

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for cell in cells:
        groups[(cell['model'], cell['budget'], cell['method'])].append(cell)
    model_summary = []
    for (model, budget, method), rows in sorted(groups.items()):
        if len(rows) != len(args.benchmarks):
            raise ValueError(f'Missing benchmark for {model}/{budget}/{method}')
        fidelity = [row['fidelity_percent'] for row in rows]
        model_summary.append({
            'model': model, 'budget': budget, 'method': method,
            'benchmark_count': len(rows),
            'mean_accuracy': _mean([row['accuracy'] for row in rows]),
            'mean_fidelity_percent': _mean(fidelity) if all(x is not None for x in fidelity) else None,
            'mean_relative_token_layer_cost_percent': _mean(
                [row['relative_token_layer_cost_percent'] for row in rows]),
            'mean_within_run_speedup': _mean([row['within_run_speedup'] for row in rows]),
        })
    return {
        'kind': args.kind,
        'aggregation': 'equal weight per benchmark in each model/budget/method group',
        'cells': cells,
        'model_summary': model_summary,
    }


def main() -> int:
    args = parse_args()
    result = summarize(args)
    payload = json.dumps(result, indent=2) + '\n'
    if args.out is None:
        print(payload, end='')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding='utf-8')
        print(f'Wrote {args.out}: {len(result["cells"])} complete cells')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
