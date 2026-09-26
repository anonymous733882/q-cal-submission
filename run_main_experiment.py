#!/usr/bin/env python3
"""Run the six-model, eight-benchmark Q-Cal multi-round main experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
RUNNERS = ROOT / 'qcal/runners'
MODELS = (
    'qwen2vl2b', 'qwen2vl', 'qwen25vl3b',
    'qwen25vl', 'qwen3vl4b', 'qwen3vl',
)
BENCHMARKS = (
    'gqa', 'pope', 'vqav2', 'visual7w',
    'visualgenomeqa', 'tallyqa', 'gqa_grounding', 'invig',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--benchmarks', nargs='+', choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument('--budgets', nargs='+', type=float, default=[5.0, 10.0])
    parser.add_argument('--num-images', type=int, default=650)
    parser.add_argument('--questions-per-image', type=int, default=15)
    parser.add_argument('--resize-square', type=int, default=1008)
    parser.add_argument('--max-pixels', type=int, default=1016064)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'qcal_support/datasets')
    parser.add_argument('--hf-hub-cache', type=Path, default=Path.home() / '.cache/huggingface/hub')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/main')
    parser.add_argument('--check-data', action='store_true', help='Load a small candidate set per selected benchmark and validate its rounds.')
    parser.add_argument('--check-data-images', type=int, default=16, help='Candidate images inspected by --check-data.')
    parser.add_argument('--dry-run', action='store_true', help='Print the complete job plan without loading data or models.')
    parser.add_argument('--resume', action='store_true', help='Skip model-benchmark jobs with valid output files at every budget.')
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list[dict]:
    if not 1 <= args.questions_per_image <= 15 or args.num_images < 1:
        raise ValueError('Use at least one image and between 1 and 15 questions per image')
    if not args.budgets or any(not 0 < budget < 33 for budget in args.budgets):
        raise ValueError('Final budgets must be positive and below the 33% Stage-1 retention')
    if len(set(args.budgets)) != len(args.budgets):
        raise ValueError('Duplicate final budgets are not allowed')
    jobs = []
    for model in args.models:
        policy = ROOT / 'policies' / f'{model}.json'
        payload = json.loads(policy.read_text(encoding='utf-8'))
        available = payload['policy']['by_budget']
        if payload['model'] != model:
            raise ValueError(f'Policy model mismatch: {policy}')
        for budget in args.budgets:
            if f'{budget:.1f}' not in available:
                raise ValueError(f'No policy for {model} at {budget}%')
        for benchmark in args.benchmarks:
            command = [
                sys.executable, str(RUNNERS / 'run_stage2_policy_json_dual_multiround.py'),
                '--model', model, '--benchmark', benchmark,
                '--num_images', str(args.num_images),
                '--questions_per_image', str(args.questions_per_image),
                '--out_dir', str(args.out_dir),
                '--policy_json', str(policy), '--json_policy_name', 'qcal',
                '--budgets', *(str(budget) for budget in args.budgets),
                '--single_policy_only',
                '--resize_square', str(args.resize_square),
                '--max_pixels', str(args.max_pixels),
            ]
            expected = [
                args.out_dir / f'b{int(round(budget)):02d}' / 'qcal' / benchmark / f'{model}.json'
                for budget in args.budgets
            ]
            jobs.append({
                'model': model,
                'benchmark': benchmark,
                'command': command,
                'expected_outputs': [str(path) for path in expected],
            })
    return jobs


def environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update({
        'DUALSIGNAL_ROOT': str(ROOT / 'qcal'),
        'BEA_DIR': str(ROOT / 'qcal_support'),
        'QCAL_DATA_ROOT': str(args.data_root.resolve()),
        'QCAL_HF_HUB_CACHE': str(args.hf_hub_cache.resolve()),
        'TRANSFORMERS_CACHE': str(args.hf_hub_cache.resolve()),
        'DUALSIGNAL_PREPARED_CACHE': str((ROOT / 'outputs/prepared_cache').resolve()),
        'DUALSIGNAL_STAGE1_WEIGHT': '0.5',
        'QWEN_ATTN_IMPLEMENTATION': os.environ.get('QWEN_ATTN_IMPLEMENTATION', 'flash_attention_2'),
        'TOKENIZERS_PARALLELISM': 'false',
    })
    return env


def complete(job: dict, questions_per_image: int) -> bool:
    for path_text in job['expected_outputs']:
        try:
            data = json.loads(Path(path_text).read_text(encoding='utf-8'))
            if data['model'] != job['model'] or data['benchmark'] != job['benchmark']:
                return False
            config = data.get('config', {})
            if (abs(float(config.get('stage1_frac', -1)) - 0.33) > 1e-9
                    or abs(float(config.get('stage1_shallow_weight', -1)) - 0.5) > 1e-9):
                return False
            n_samples = int(data['n_samples'])
            if n_samples < 1 or len(data['baseline_per_sample']) != n_samples:
                return False
            tag = Path(path_text).parts[-4]
            method = data['methods'][f'{tag}/qcal']
            if len(method['per_sample']) != n_samples:
                return False
            if n_samples > int(job['command'][job['command'].index('--num_images') + 1]) * questions_per_image:
                return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
    return True


def check_data(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(RUNNERS))
    sys.path.insert(0, str(ROOT / 'qcal_support'))
    sys.path.insert(0, str(ROOT / 'qcal_support/strategy_test'))
    import multi_round_benchmark as bea_mr
    from main_benchmarks import load_main_samples

    for benchmark in args.benchmarks:
        samples = load_main_samples(benchmark, args.check_data_images, args.questions_per_image, bea_mr)
        if not samples:
            raise RuntimeError(f'{benchmark}: no usable images among {args.check_data_images} candidates')
        print(f'{benchmark}: {len(samples)} images, first has {len(samples[0][1])} question rounds')


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    jobs = build_jobs(args)
    if args.dry_run:
        print(json.dumps({
            'protocol': 'one image, up to 15 questions, shared Stage-1, per-question true Stage-2',
            'models': args.models,
            'benchmarks': args.benchmarks,
            'budgets': args.budgets,
            'stage1_retention': 0.33,
            'stage1_shallow_weight': 0.5,
            'active_weight_threshold': '>0.05',
            'attention_implementation': environment(args)['QWEN_ATTN_IMPLEMENTATION'],
            'jobs': jobs,
        }, indent=2))
        return 0

    os.environ.update(environment(args))
    if args.check_data:
        check_data(args)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        'protocol': 'single-image multi-round, shared Stage-1 and per-question true Stage-2',
        'models': args.models,
        'benchmarks': args.benchmarks,
        'budgets': args.budgets,
        'num_images_cap': args.num_images,
        'questions_per_image': args.questions_per_image,
        'resize_square': args.resize_square,
        'max_pixels': args.max_pixels,
        'active_weight_threshold': '>0.05',
        'attention_implementation': environment(args)['QWEN_ATTN_IMPLEMENTATION'],
        'jobs': jobs,
    }
    (args.out_dir / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    env = environment(args)
    for number, job in enumerate(jobs, start=1):
        if args.resume and complete(job, args.questions_per_image):
            print(f'[{number}/{len(jobs)}] skip {job["model"]} {job["benchmark"]}', flush=True)
            continue
        print(f'[{number}/{len(jobs)}] run {job["model"]} {job["benchmark"]}', flush=True)
        subprocess.run(job['command'], cwd=ROOT, env=env, check=True)
        if not complete(job, args.questions_per_image):
            raise RuntimeError(f'Incomplete outputs for {job["model"]}/{job["benchmark"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
