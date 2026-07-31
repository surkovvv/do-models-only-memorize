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
