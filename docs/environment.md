# Experiment environment

The project uses Python 3.13, `uv`, and a repository-local `.venv`. Exact
resolved package versions are recorded in `uv.lock`.

The Hugging Face stack stays on its mutually compatible major-version line
(`transformers` 4.x, `datasets` 4.x, and `trl` 0.x); `uv.lock` pins the exact
versions used by every run.

## Bootstrap

```bash
uv sync --all-groups
uv run python scripts/check_environment.py
uv run pytest
```

Run commands through `uv run ...`, or activate the environment directly:

```bash
source .venv/bin/activate
```

## Included stack

- PyTorch, Transformers, Accelerate, Datasets, Evaluate, PEFT, and TRL for
  training and evaluation;
- NumPy, pandas, Arrow, SciPy, scikit-learn, matplotlib, and seaborn for data
  processing and analysis;
- ClearML for experiment tracking and artifact storage, plus TensorBoard and
  Loguru for local metrics and logs;
- Hydra and Pydantic Settings for reproducible configuration, and Optuna for
  hyperparameter search;
- JupyterLab, pytest, Ruff, mypy, and pre-commit for interactive work and
  development.

ClearML is optional at runtime. Copy `.env.example` to `.env` and configure
credentials, or run experiments in offline mode when no server is available:

```bash
CLEARML_OFFLINE_MODE=1 uv run python your_experiment.py
```

## Experiment configuration

The SFT smoke experiment loads a typed YAML configuration. Its default is
`configs/sft_smoke.yaml`; another file and individual typed overrides can be
selected from the command line:

```bash
uv run python scripts/sft_smoke.py
uv run python scripts/sft_smoke.py --config configs/sft_smoke.yaml \
  training.batch_size=4 runtime.device=mps
```

By default this is a bounded smoke-training run: it samples
`training.batch_size` records without replacement using `training.seed`, then
repeats that fixed batch for `training.smoke_steps` optimizer steps (10 by
default). The same seed selects the same batch; change it to test another one.
After every step the script generates answers for that batch, reports
exact-match accuracy, and prints the configured number of predictions.
`runtime.prediction_log_limit=null` prints the entire batch; set it to `1` for
compact output. To run the complete dataset by epochs instead, disable the
bound explicitly:

```bash
uv run python scripts/sft_smoke.py training.smoke_steps=null training.epochs=1
```

Complete epoch runs evaluate generated-answer exact match on the full train
and test files before optimization (`epoch=0`) and after every epoch. Evaluation
is batched independently from training; configure it with
`evaluation.batch_size`:

```bash
uv run python scripts/sft_smoke.py \
  training.smoke_steps=null \
  training.epochs=5 \
  evaluation.batch_size=64
```

Smoke runs retain their fixed-batch per-step evaluation and do not scan the
full train/test datasets.

Set `runtime.output_dir` to persist a resolved config, JSONL metrics,
environment metadata, final predictions, and (by default) a reloadable model.
For disposable smoke runs, `runtime.save_model=false` keeps the small diagnostic
artifacts without writing a full checkpoint.

Unknown keys, incompatible types, and invalid numeric values fail before model
loading. Enable `tracking.enabled` to create a distinct ClearML task and connect
the resolved configuration as editable task parameters. For a first local run,
offline mode avoids requiring a ClearML server:

```bash
CLEARML_OFFLINE_MODE=1 uv run python scripts/sft_smoke.py tracking.enabled=true
```

The default PyPI PyTorch build supports Apple Silicon through MPS. CUDA-only
tools such as bitsandbytes and DeepSpeed are intentionally not installed on
macOS; add them in a Linux/GPU-specific dependency group when that environment
is introduced.

## Updating

```bash
uv lock --upgrade
uv sync --all-groups
```

Commit both `pyproject.toml` and `uv.lock` after a successful update.
