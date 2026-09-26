#!/usr/bin/env python3
"""Check the immutable diagnostic package and exported Q-Cal policies."""

from __future__ import annotations

from collections import Counter
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
MODELS = {
    'qwen2vl2b': 28, 'qwen2vl': 28,
    'qwen25vl3b': 36, 'qwen25vl': 28,
    'qwen3vl4b': 36, 'qwen3vl': 36,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    for required in (
        'qcal/runners/model_backends.py',
        'baselines/evaluation/run_layer.py',
        'baselines/evaluation/run_mask.py',
        'qcal_support/strategy_test/multi_round_benchmark.py',
        'baselines/pact/utils.py',
        'baselines/pact/LICENSE',
        'data/prompt_invariance_100/groups.json',
    ):
        assert (ROOT / required).is_file(), required
    for directory in ('qcal', 'qcal_support', 'baselines'):
        for path in (ROOT / directory).rglob('*'):
            assert not path.is_symlink(), f'external source link is forbidden: {path}'
    manifest = json.loads((ROOT / 'data/final240/manifest.json').read_text(encoding='utf-8'))
    tasks = manifest['tasks']
    assert len(tasks) == 240, len(tasks)
    assert [row['id'] for row in tasks] == [f'sa240-{i:04d}' for i in range(1, 241)]
    assert Counter(row['task_type'] for row in tasks) == {
        'global_like': 80, 'local_like': 80, 'no_difference': 80,
    }
    assert all(
        row['benchmark'] == 'grounding_iou'
        or row['prompt'].strip().startswith(row['question'].strip())
        for row in tasks
    ), 'fixed task question/prompt mismatch'
    for row in tasks:
        image_rel = Path(row['image'])
        assert not image_rel.is_absolute() and image_rel.parts[0] == 'images'
        image = ROOT / 'data/final240' / image_rel
        assert image.is_file(), image
        assert digest(image) == row['image_sha256'], row['id']
        with Image.open(image) as loaded:
            assert loaded.size == (row['image_width'], row['image_height']), row['id']
    prompt_groups = json.loads((ROOT / 'data/prompt_invariance_100/groups.json').read_text(encoding='utf-8'))['groups']
    assert len(prompt_groups) == 100
    assert len({group['image_id'] for group in prompt_groups}) == 100
    for group in prompt_groups:
        assert len(group['qas']) == 5
        assert len({qa['question_id'] for qa in group['qas']}) == 5
        with Image.open(BytesIO(base64.b64decode(group['image_b64'], validate=True))) as loaded:
            loaded.verify()
    for name, total_layers in MODELS.items():
        policy = json.loads((ROOT / 'policies' / f'{name}.json').read_text(encoding='utf-8'))
        assert policy['model'] == name
        for budget in ('5.0', '10.0', '20.0', '33.0'):
            entry = policy['policy']['by_budget'][budget]
            layers, weights = entry['active_layers'], entry['active_weights']
            assert entry['total_layers'] == total_layers
            assert layers and len(layers) == len(weights)
            assert min(layers) >= 0 and max(layers) < total_layers
            assert all(weight >= 0 for weight in weights)
            assert abs(sum(weights) - 1.0) < 1e-3
    print('OK: 240 fixed tasks, 100 five-question images, all images, six policies')


if __name__ == '__main__':
    main()
