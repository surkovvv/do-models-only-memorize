#!/usr/bin/env python3
"""Smoke-check the dependencies used by the experiment stack."""

from __future__ import annotations

import importlib
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

PACKAGES = {
    "accelerate": "accelerate",
    "clearml": "clearml",
    "datasets": "datasets",
    "einops": "einops",
    "evaluate": "evaluate",
    "huggingface-hub": "huggingface_hub",
    "hydra-core": "hydra",
    "loguru": "loguru",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "optuna": "optuna",
    "pandas": "pandas",
    "peft": "peft",
    "pyarrow": "pyarrow",
    "pydantic-settings": "pydantic_settings",
    "pyyaml": "yaml",
    "rich": "rich",
    "safetensors": "safetensors",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "sentencepiece": "sentencepiece",
    "tensorboard": "tensorboard",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "trl": "trl",
}


def main() -> None:
    failures: list[str] = []
    for distribution, module in PACKAGES.items():
        print(f"Checking {distribution}...", flush=True)
        try:
            importlib.import_module(module)
            installed_version = version(distribution)
        except (ImportError, PackageNotFoundError) as error:
            failures.append(f"{distribution}: {error}")
            continue
        print(f"  {installed_version}", flush=True)

    import torch

    print()
    print(
        "PyTorch device support: "
        f"CUDA={torch.cuda.is_available()}, "
        f"MPS-built={torch.backends.mps.is_built()}, "
        f"MPS-available={torch.backends.mps.is_available()}"
    )

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise SystemExit(f"\nEnvironment check failed:\n{details}")

    print("Environment check passed.")


if __name__ == "__main__":
    main()
