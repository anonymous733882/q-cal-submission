"""Single-image, multi-round tasks for the eight main-evaluation benchmarks."""

from __future__ import annotations

import random

from PIL import Image

from multiround_dataset_loaders import (
    load_gqa_grounding_multi_round,
    load_invig_multi_round,
    load_tallyqa_multi_round,
    load_visual7w_multi_round,
    load_visualgenomeqa_multi_round,
)


MAIN_BENCHMARKS = (
    "gqa",
    "pope",
    "vqav2",
    "visual7w",
    "visualgenomeqa",
    "tallyqa",
    "gqa_grounding",
    "invig",
)

CUSTOM_LOADERS = {
    "visual7w": load_visual7w_multi_round,
    "visualgenomeqa": load_visualgenomeqa_multi_round,
    "tallyqa": load_tallyqa_multi_round,
    "gqa_grounding": load_gqa_grounding_multi_round,
    "invig": load_invig_multi_round,
}


def _load_with_fixed_seed(loader, num_images, questions_per_image):
    state = random.getstate()
    random.seed(42)
    try:
        return loader(num_images, questions_per_image)
    finally:
        random.setstate(state)


def load_main_samples(benchmark, num_images, questions_per_image, bea_mr):
    """Return one image with up to 15 ordered question-answer rounds per task."""
    if benchmark not in MAIN_BENCHMARKS:
        raise ValueError(f"Not a main-evaluation benchmark: {benchmark}")
    if num_images < 1 or not 1 <= questions_per_image <= 15:
        raise ValueError("Use at least one image and between 1 and 15 questions per image")

    if benchmark in CUSTOM_LOADERS:
        raw = _load_with_fixed_seed(CUSTOM_LOADERS[benchmark], num_images, questions_per_image)
        samples = [
            (
                item["image"],
                [(qa["question"], qa["answer"]) for qa in item["qas"]],
            )
            for item in raw
        ]
    else:
        loader = getattr(bea_mr, f"load_{benchmark}_multi_round")
        samples = _load_with_fixed_seed(loader, num_images, questions_per_image)

    for index, (image, rounds) in enumerate(samples):
        if not isinstance(image, Image.Image):
            raise TypeError(f"{benchmark} task {index} has no PIL image")
        if not 1 <= len(rounds) <= questions_per_image:
            raise ValueError(f"{benchmark} task {index} has {len(rounds)} rounds")
        if any(not str(question).strip() for question, _ in rounds):
            raise ValueError(f"{benchmark} task {index} has an empty question")
    return samples


def install_main_loader(benchmark, bea_mr):
    """Bind the common task contract to the existing true-progressive runner."""
    if benchmark not in MAIN_BENCHMARKS:
        raise ValueError(f"Not a main-evaluation benchmark: {benchmark}")
    if benchmark not in CUSTOM_LOADERS:
        loader_name = f"load_{benchmark}_multi_round"
        original = getattr(bea_mr, loader_name)
        setattr(
            bea_mr, loader_name,
            lambda num_images, questions_per_image=15: _load_with_fixed_seed(
                original, num_images, questions_per_image
            ),
        )
        return benchmark
    proxy = "visual7w" if benchmark == "visual7w" else "vqav2"
    setattr(
        bea_mr,
        f"load_{proxy}_multi_round",
        lambda num_images, questions_per_image=15: load_main_samples(
            benchmark, num_images, questions_per_image, bea_mr
        ),
    )
    return proxy
