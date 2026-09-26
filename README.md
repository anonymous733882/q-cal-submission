# Q-Cal: Query-Calibrated Visual Token Pruning

This repository implements Q-Cal calibration, true two-stage visual-token
pruning, and the evaluation protocols used in the paper. The default workflows
use six Qwen vision-language models. The main evaluation covers eight visual
question-answering benchmarks.

## Setup

Use Python 3.11 with CUDA and a compatible PyTorch installation. Install the
remaining dependencies and FlashAttention-2:

```bash
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

Run the commands below from the repository root. A GPU, sufficient disk space,
and network access are needed to download model checkpoints and source benchmark
data on the first run. Downloads are cached. Dataset acquisition and loading are
handled by the runners; no benchmark paths need to be passed on the command line.
The source datasets and model checkpoints remain subject to their own licenses.

## Workflows

```bash
python run.py main
python run.py diagnostics
python run.py calibrate
```

`main` runs Q-Cal and the comparison methods on the same eight-benchmark,
multi-round protocol. It then produces accuracy, fidelity, token-layer cost, and
speedup summaries. It resumes completed model-benchmark jobs. Results are in
`outputs/main/`, `outputs/baselines/`, and `outputs/*_summary.json`.

`diagnostics` runs same-image prompt invariance, the fixed 240-task
selection-behavior evaluation, and the depth-wise attention sweep. The 240-task
manifest and images are bundled in `data/final240/`; the 100-image,
five-question prompt set is in `data/prompt_invariance_100/`. Results are in
`outputs/diagnostics/`.

`calibrate` fits model-specific policy weights from ground-truth teacher-forced
gradient-oracle token sets. It uses the eight calibration capability sources and
optimizes the budget-aware token-set overlap objective with the paper's
layer-cost and regularization terms. This optional refit uses the 0.5/0.5,
33% Stage-1 configuration and writes its artifacts to
`outputs/calibration/stage1_50_33/`. At inference,
only layers with a policy weight strictly greater than 0.05 compute the
Stage-2 attention signal.

Baseline implementations are organized by method under `baselines/`; each
method directory contains its adapter, any released selector code used by the
adapter, and the relevant license. Shared evaluation orchestration is under
`baselines/evaluation/`. Methods without released code are implemented from
their published specifications; provenance is recorded in baseline outputs.
