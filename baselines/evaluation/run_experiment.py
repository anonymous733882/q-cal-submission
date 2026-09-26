#!/usr/bin/env python3
"""Run all baseline methods and record each implementation's provenance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_main_experiment import BENCHMARKS, MODELS  # noqa: E402


METHODS = (
    'AgilePruner', 'DivPrune', 'FastV', 'FasterVLM', 'ID-Selection',
    'PACT', 'PTP', 'SVD-Prune', 'SparseVILA', 'VisPruner', 'ZSPAPrune',
)
OFFICIAL_SELECTOR_METHODS = frozenset({'AgilePruner', 'DivPrune', 'FasterVLM', 'VisPruner'})
LAYER_METHODS = frozenset({'FastV', 'PACT'})
RUNNER = ROOT / 'baselines/evaluation/run_mask.py'
LAYER_RUNNER = ROOT / 'baselines/evaluation/run_layer.py'
DEFAULT_TARGETS = ROOT / 'data/main_baseline_budget_targets.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--benchmarks', nargs='+', choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--budgets', nargs='+', type=int, choices=[5, 10], default=[5, 10])
    parser.add_argument('--budget-targets', type=Path, default=DEFAULT_TARGETS)
    parser.add_argument('--num-images', type=int, default=650)
    parser.add_argument('--questions-per-image', type=int, default=15)
    parser.add_argument('--resize-square', type=int, default=1008)
    parser.add_argument('--max-pixels', type=int, default=1016064)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'qcal_support/datasets')
    parser.add_argument('--hf-hub-cache', type=Path, default=Path.home() / '.cache/huggingface/hub')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/baselines')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list[dict]:
    if args.num_images < 1 or not 1 <= args.questions_per_image <= 15:
        raise ValueError('Use at least one image and between 1 and 15 questions per image')
    if len(set(args.methods)) != len(args.methods) or len(set(args.budgets)) != len(args.budgets):
        raise ValueError('Duplicate methods or budgets are not allowed')
    targets = json.loads(args.budget_targets.read_text(encoding='utf-8'))['targets']
    jobs = []
    for model in args.models:
        for benchmark in args.benchmarks:
            for budget in args.budgets:
                tag = f'b{budget:02d}'
                target = float(targets[model][benchmark][tag])
                if not 0 < target < 100:
                    raise ValueError(f'Invalid retention target for {model}/{benchmark}/{tag}: {target}')
                job_dir = args.out_dir / '_work' / tag / model / benchmark
                result = args.out_dir / tag / benchmark / f'{model}.json'
                mask_methods = [method for method in args.methods if method not in LAYER_METHODS]
                command = [
                    sys.executable, str(RUNNER),
                    '--models', model, '--benchmarks', benchmark,
                    '--methods', *mask_methods,
                    '--num-images', str(args.num_images),
                    '--questions-per-image', str(args.questions_per_image),
                    '--budget-mode', 'explicit', '--budgets', str(target),
                    '--out-dir', str(job_dir), '--full-json',
                    '--resize-square', str(args.resize_square),
                    '--max-pixels', str(args.max_pixels),
                ] if mask_methods else None
                pact_output = job_dir / 'pact.json'
                pact_command = [
                    sys.executable, str(LAYER_RUNNER),
                    '--model', model, '--benchmark', benchmark,
                    '--target-token-layer-percent', str(target),
                    '--num-images', str(args.num_images),
                    '--questions-per-image', str(args.questions_per_image),
                    '--resize-square', str(args.resize_square),
                    '--max-pixels', str(args.max_pixels),
                    '--output', str(pact_output),
                ] if 'PACT' in args.methods else None
                fastv_output = job_dir / 'fastv.json'
                fastv_command = [
                    sys.executable, str(LAYER_RUNNER),
                    '--method', 'FastV',
                    '--model', model, '--benchmark', benchmark,
                    '--target-token-layer-percent', str(target),
                    '--num-images', str(args.num_images),
                    '--questions-per-image', str(args.questions_per_image),
                    '--resize-square', str(args.resize_square),
                    '--max-pixels', str(args.max_pixels),
                    '--output', str(fastv_output),
                ] if 'FastV' in args.methods else None
                jobs.append({
                    'model': model, 'benchmark': benchmark, 'budget': tag,
                    'equivalent_retention_percent': target,
                    'methods': list(args.methods), 'command': command,
                    'pact_command': pact_command, 'pact_output': str(pact_output),
                    'fastv_command': fastv_command, 'fastv_output': str(fastv_output),
                    'work_dir': str(job_dir), 'result': str(result),
                })
    return jobs


def environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update({
        'DUALSIGNAL_ROOT': str(ROOT / 'qcal'),
        'BEA_DIR': str(ROOT / 'qcal_support'),
        'QCAL_DATA_ROOT': str(args.data_root.resolve()),
        'TRANSFORMERS_CACHE': str(args.hf_hub_cache.resolve()),
        'QCAL_HF_HUB_CACHE': str(args.hf_hub_cache.resolve()),
        'DUALSIGNAL_PREPARED_CACHE': str((ROOT / 'outputs/prepared_cache').resolve()),
        'DUALSIGNAL_DISABLE_PREFIX_KV_CACHE': '1',
        'DUALSIGNAL_QWEN_ATTN_IMPLEMENTATION': (
            'sdpa' if any(method in LAYER_METHODS for method in args.methods) else
            os.environ.get('DUALSIGNAL_QWEN_ATTN_IMPLEMENTATION', 'flash_attention_2')),
        'TOKENIZERS_PARALLELISM': 'false',
    })
    return env


def complete(
    path: Path,
    methods: list[str],
    provenance: dict[str, str] | None = None,
    *,
    expected_budget_percent: float | None = None,
    num_images: int | None = None,
    questions_per_image: int | None = None,
) -> bool:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        n_samples = int(data['n_samples'])
        config = data.get('config', {})
        return (
            n_samples > 0
            and len(data['baseline_per_sample']) == n_samples
            and (expected_budget_percent is None or (
                len(config.get('budgets', [])) == 1
                and abs(float(config['budgets'][0]) - expected_budget_percent) < 1e-8
            ))
            and (num_images is None or int(config.get('num_images', -1)) == num_images)
            and (questions_per_image is None or
                 int(config.get('questions_per_image', -1)) == questions_per_image)
            and set(methods) <= set(data['methods'])
            and all(len(data['methods'][name]['per_sample']) == n_samples for name in methods)
            and (provenance is None or all(
                data['methods'][name].get('implementation_provenance') == provenance[name]
                for name in methods
            ))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _verify_question_alignment(job: dict, source: dict, other: dict) -> None:
    if 'records' in source:
        source_rows = source['records']
    else:
        records_path = Path(job['work_dir']) / 'records.jsonl'
        source_rows = [json.loads(line) for line in records_path.read_text(encoding='utf-8').splitlines()]
    by_turn = {}
    for row in source_rows:
        key = (int(row['sample_idx']), int(row['turn_id']))
        if key in by_turn and by_turn[key]['baseline_output'] != row['baseline_output']:
            raise RuntimeError(f'Inconsistent full-token output at {key}')
        by_turn[key] = row
    other_rows = other.get('records', [])
    if len(by_turn) != len(other_rows):
        raise RuntimeError('Baseline runners have different question counts')
    for source_key, other_row in zip(sorted(by_turn), other_rows):
        other_key = (int(other_row['sample_idx']), int(other_row['turn_id']))
        if source_key != other_key or by_turn[source_key]['baseline_output'] != other_row['baseline_output']:
            raise RuntimeError(f'Baseline question or full-token output differs at {source_key} vs {other_key}')


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    jobs = build_jobs(args)
    plan = {
        'protocol': ('single-image multi-round baseline; PACT and FastV use layer-local '
                     'reduction per question; pre-decoder methods reuse a first-turn mask'),
        'budget_targets': str(args.budget_targets.resolve()),
        'num_images_cap': args.num_images,
        'questions_per_image': args.questions_per_image,
        'attention_implementation': environment(args)['DUALSIGNAL_QWEN_ATTN_IMPLEMENTATION'],
        'implementation_provenance': {
            method: ('pact_reference_adaptation_total_cost_v2' if method == 'PACT' else
                     'vendored_author_layer_method_qwen_backend_total_cost_v2' if method == 'FastV' else
                     'vendored_author_selector_qwen_backend' if method in OFFICIAL_SELECTOR_METHODS else
                     'historical_local_adapter_sparsevila_cache_v3_first_generation_cost' if method == 'SparseVILA' else
                     'historical_local_adapter')
            for method in args.methods
        },
        'jobs': jobs,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    env = environment(args)
    for number, job in enumerate(jobs, start=1):
        result = Path(job['result'])
        if args.resume and complete(
            result, job['methods'], plan['implementation_provenance'],
            expected_budget_percent=job['equivalent_retention_percent'],
            num_images=args.num_images, questions_per_image=args.questions_per_image,
        ):
            print(f'[{number}/{len(jobs)}] skip {job["model"]} {job["benchmark"]} {job["budget"]}', flush=True)
            continue
        print(f'[{number}/{len(jobs)}] run {job["model"]} {job["benchmark"]} {job["budget"]}', flush=True)
        source = None
        if job['command'] is not None:
            subprocess.run(job['command'], cwd=ROOT, env=env, check=True)
            source_files = list(Path(job['work_dir']).glob('full_json/**/*.json'))
            if len(source_files) != 1 or not complete(source_files[0],
                                                      [m for m in job['methods'] if m not in LAYER_METHODS]):
                raise RuntimeError(f'Incomplete mask baseline output for {job["model"]}/{job["benchmark"]}/{job["budget"]}')
            source = json.loads(source_files[0].read_text(encoding='utf-8'))
        for method in ('FastV', 'PACT'):
            command = job[f'{method.lower()}_command']
            if command is None:
                continue
            output_path = Path(job[f'{method.lower()}_output'])
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            layer_result = json.loads(output_path.read_text(encoding='utf-8'))
            if not complete(output_path, [method]):
                raise RuntimeError(f'Incomplete {method} output for {job["model"]}/{job["benchmark"]}/{job["budget"]}')
            if source is None:
                source = layer_result
            else:
                if source['n_samples'] != layer_result['n_samples'] or any(
                    abs(float(a) - float(b)) > 1e-6 for a, b in zip(
                        source['baseline_per_sample'], layer_result['baseline_per_sample'])):
                    raise RuntimeError(f'{method} and existing baseline question/full-token records do not align')
                _verify_question_alignment(job, source, layer_result)
                source['methods'][method] = layer_result['methods'][method]
                source['methods'][method]['baseline_cost'] = layer_result['baseline_cost']
                source['config'][f'{method.lower()}_runner'] = layer_result['config']
        if source is None:
            raise RuntimeError('No baseline method was selected')
        source.setdefault('config', {})['implementation_provenance'] = plan['implementation_provenance']
        for method, provenance in plan['implementation_provenance'].items():
            source['methods'][method]['implementation_provenance'] = provenance
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps(source, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
