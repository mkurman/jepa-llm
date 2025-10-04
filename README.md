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
available fields and their defaults. The sections expand to the following
parameters:

### `general`

- `debug` — increases logging verbosity when set to a truthy value.
- `finetune_seed` — random seed forwarded to dataset shuffling and training.

### `model`

- `name` — Hugging Face model identifier or local path to load.
- `use_lora` — enables LoRA adapters instead of full fine-tuning when `true`.
- `lora_rank` — rank of the LoRA decomposition; higher increases adapter
  capacity.
- `pretrain` — starts from randomly initialised weights instead of a pretrained
  checkpoint.
- `cache_dir` — optional directory passed to the Hugging Face loaders for
  storing downloaded weights.
- `trust_remote_code` — allows loading custom model code when required.

### `dataset`

- `train_file` — path to the JSONL file containing training samples.
- `eval_file` — path to the JSONL file containing evaluation samples.
- `data_file` — alternative single JSONL file used when train and eval are
  split dynamically.
- `eval_split` — fraction of examples reserved for evaluation when using a
  combined dataset file.
- `split_seed` — seed controlling the deterministic split of combined datasets.
- `max_items` — limits the number of examples loaded for quick experiments.
- `max_length` — truncation length applied during tokenisation.
- `predictors` — number of asynchronous dataloader worker processes.
- `train_all` — toggles training on the full dataset without a validation split.
- `plain` — forces the plain Hugging Face trainer without the JEPA loss.
- `remove_thinking` — strips hidden reasoning tags from the dataset when `true`.
- `cache_dir` — local cache directory for processed dataset artefacts.

### `training`

- `output_dir` — directory where checkpoints, logs, and metrics are stored.
- `batch_size` — number of examples processed per device step.
- `grad_accum` — gradient accumulation steps to simulate larger batches.
- `learning_rate` — base learning rate for the optimiser.
- `num_epochs` — maximum number of epochs to train.
- `eval_steps` — interval (in optimiser steps) between evaluation runs.
- `lbd` — JEPA representation loss weight.
- `gamma` — JEPA predictor loss weight.
- `last_token` — token index used when computing the JEPA objective (`-1`
  selects the final token).
- `regular` — disables the JEPA loss and runs a standard language modelling
  objective when `true`.
- `track_flop` — enables FLOP counting during training for benchmarking.
- `additive_mask` — switches the attention mask computation to the additive
  form; incompatible with memory-efficient mode.
- `memory_efficient` — enables chunked attention to reduce peak VRAM usage.

When training, we recommend enabling the memory-efficient path to reduce peak
VRAM usage and maintain consistent results. The trade-offs are:

⚖️ Trade-offs:
Advantages:

- 2-3x reduction in peak VRAM usage
- Enables larger batch sizes and longer sequences
- Identical training results
- Easy to enable with single flag

Disadvantages:

- Slightly slower (3 forward passes vs 1 concatenated)
- Cannot be used with `--additive_mask` (incompatible)

## Development

Compile-time checks can be run without GPU access:

```bash
PYTHONPATH=src python -m compileall src/jepa_llm
```

The repository uses `pyproject.toml` for packaging metadata and exposes the
`jepa-llm` entry point for convenience.
