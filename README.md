# JEPA LLM Fine-Tuning Toolkit

This repository packages the training script for the original project into a reusable
Python module with a configuration-driven command line interface.  It supports
fine-tuning causal language models with an additional JEPA-style representation
regularisation loss as well as a plain Hugging Face trainer.

## Getting started

1. Install the package in an environment that already contains the CUDA-enabled
   PyTorch build required for your hardware:

   ```bash
   pip install -e .
   ```

2. Copy the sample configuration and adjust the paths and hyperparameters to
   match your setup:

   ```bash
   cp configs/example.yaml my-run.yaml
   ```

3. Launch training with the CLI, pointing it to your configuration file:

   ```bash
   jepa-llm --config my-run.yaml
   ```

The CLI prints the selected options on the primary process, then orchestrates
model initialisation, dataset preparation, trainer selection, and checkpointing.

## Configuration reference

The CLI accepts a single `--config` argument pointing at a YAML file.  The
configuration is split into four sections:

- `general` — debugging verbosity and the random seed used for data shuffling
  and training.
- `model` — model identifier, optional LoRA settings, whether to pretrain from
  scratch, and flags forwarded to the Hugging Face loaders (`cache_dir`,
  `trust_remote_code`).
- `dataset` — data sources and preprocessing options, including the optional
  `remove_thinking` toggle that strips hidden reasoning tags and the maximum
  sequence length used during tokenisation.
- `training` — optimisation hyperparameters, JEPA loss weights, and trainer
  behaviour toggles such as FLOP tracking, additive masking, or the
  memory-efficient path.

Refer to [`configs/example.yaml`](configs/example.yaml) for a complete list of
available fields and their defaults.

## Development

Compile-time checks can be run without GPU access:

```bash
PYTHONPATH=src python -m compileall src/jepa_llm
```

The repository uses `pyproject.toml` for packaging metadata and exposes the
`jepa-llm` entry point for convenience.
