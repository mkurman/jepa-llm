# JEPA LLM Fine-Tuning Toolkit

![jepa_banner](jepa.png)

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

- `train_file` — dataset identifier or path used for the training split when
  supplied directly.
- `eval_file` — optional dataset identifier/path for the evaluation split.
- `data_file` — single dataset to be split on the fly when dedicated train/eval
  splits are not available.
- `eval_split` — fraction reserved for evaluation when splitting `data_file`.
- `split_seed` — seed applied to the deterministic split of `data_file`.
- `max_items` — optional cap on training examples (handy for smoke tests).
- `max_eval_items` — optional cap applied only to the evaluation dataset.
- `config_name` — optional Hugging Face dataset configuration name.
- `train_split` — optional split name when pulling remote datasets.
- `max_length` — maximum sequence length for tokenisation.
- `predictors` — number of predictor tokens appended to the user prompt.
- `train_all` — compute supervised loss over every token instead of assistant-only.
- `plain` — bypass the chat template and feed raw message text to the tokenizer.
- `remove_thinking` — strip hidden reasoning tags like `<think>` before tokenising.
- `cache_dir` — location where dataset artefacts should be cached.

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
- `dtype` — computation precision requested from the trainer (`bf16`, `fp16`,
  or `float32`).
- `optimizer` — optimiser identifier passed to the Hugging Face trainer.

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

# Credits
To view the original version, please visit this repository: [https://github.com/rbalestr-lab/llm-jepa](https://github.com/rbalestr-lab/llm-jepa).
