"""Command-line interface for fine-tuning JEPA-style language models."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from .callbacks.profiler_flop_callback import ProfilerFLOPCallback
from .config import Config, load_config
from .dataset import load_and_prepare_dataset
from .model_setup import setup_model_and_tokenizer
from .trainers.representation_trainer import RepresentationTrainer
from .utils import is_primary_process


logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser that only accepts the configuration file."""

    parser = argparse.ArgumentParser(description="Fine-tune JEPA-enhanced LLMs")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML configuration file containing all run settings.",
    )
    return parser


def _looks_like_local_path(path: str) -> bool:
    candidate = Path(path)
    return candidate.exists() or candidate.suffix != ""


def _resolve_dataset_split(
    path: str, requested_split: Optional[str], default: str
) -> str:
    if requested_split:
        return requested_split
    if _looks_like_local_path(path):
        return "train"
    return default


def _validate_config(config: Config) -> None:
    dataset_cfg = config.dataset
    training_cfg = config.training

    has_train_split = bool(dataset_cfg.train_file)
    has_combined_file = bool(dataset_cfg.data_file)

    if has_train_split == has_combined_file:
        raise ValueError(
            "Configuration must provide either 'train_file' or 'data_file'."
        )

    if training_cfg.memory_efficient and training_cfg.additive_mask:
        raise ValueError(
            "'memory_efficient' and 'additive_mask' modes are mutually exclusive."
        )


def _log_startup(config: Config) -> None:
    if not is_primary_process():
        return

    dataset_cfg = config.dataset
    model_cfg = config.model
    training_cfg = config.training

    logger.info("=== Fine-tuning Script ===")

    if dataset_cfg.train_file:
        logger.info("Train file: %s", dataset_cfg.train_file)
        if dataset_cfg.eval_file:
            logger.info("Eval file: %s", dataset_cfg.eval_file)
        else:
            logger.info("No eval file provided - training without evaluation")
    else:
        logger.info(
            "Data file: %s (will split %.1f%% for eval)",
            dataset_cfg.data_file,
            dataset_cfg.eval_split * 100,
        )

    logger.info("Model: %s", model_cfg.name)
    logger.info("Output: %s", training_cfg.output_dir)
    logger.info("Using LoRA: %s", model_cfg.use_lora)
    if model_cfg.use_lora:
        logger.info("LoRA rank: %s", model_cfg.lora_rank)
    logger.info("Memory efficient mode: %s", training_cfg.memory_efficient)
    if training_cfg.memory_efficient:
        logger.info(
            "  → Will process sequences separately to reduce VRAM usage by 2-3x"
        )


def _init_distributed_if_needed() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1 and torch.cuda.is_available():
        if is_primary_process():
            logger.info(
                "Running with torchrun: world_size=%s, local_rank=%s",
                world_size,
                local_rank,
            )
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)


def _prepare_datasets(config: Config, tokenizer) -> Tuple[object, object]:
    dataset_cfg = config.dataset
    model_cfg = config.model
    general_cfg = config.general
    training_cfg = config.training

    if dataset_cfg.train_file:
        if is_primary_process():
            logger.info("Loading training data from %s", dataset_cfg.train_file)
        train_dataset = load_and_prepare_dataset(
            dataset_cfg.train_file,
            tokenizer,
            model_cfg.name,
            dataset_cfg.max_length,
            debug=general_cfg.debug,
            predictors=dataset_cfg.predictors,
            regular=training_cfg.regular,
            train_all=dataset_cfg.train_all,
            plain=dataset_cfg.plain,
            max_items=dataset_cfg.max_items,
            seed=general_cfg.finetune_seed,
            remove_thinking=dataset_cfg.remove_thinking,
            cache_dir=dataset_cfg.cache_dir,
            dataset_split=_resolve_dataset_split(
                dataset_cfg.train_file, dataset_cfg.train_split, default="train"
            ),
            config_name=dataset_cfg.config_name,
            preprocess_num_proc=dataset_cfg.preprocess_num_proc,
            tokenize_num_proc=dataset_cfg.tokenize_num_proc,
            tokenize_batch_size=dataset_cfg.tokenize_batch_size,
        )

        eval_limit = (
            dataset_cfg.max_eval_items
            if dataset_cfg.max_eval_items is not None
            else dataset_cfg.max_items
        )

        if dataset_cfg.eval_file:
            if is_primary_process():
                logger.info("Loading evaluation data from %s", dataset_cfg.eval_file)
            eval_dataset = load_and_prepare_dataset(
                dataset_cfg.eval_file,
                tokenizer,
                model_cfg.name,
                dataset_cfg.max_length,
                debug=general_cfg.debug,
                predictors=dataset_cfg.predictors,
                regular=training_cfg.regular,
                train_all=dataset_cfg.train_all,
                plain=dataset_cfg.plain,
                max_items=eval_limit,
                seed=general_cfg.finetune_seed,
                remove_thinking=dataset_cfg.remove_thinking,
                cache_dir=dataset_cfg.cache_dir,
                dataset_split=_resolve_dataset_split(
                    dataset_cfg.eval_file, None, default="test"
                ),
                config_name=dataset_cfg.config_name,
                preprocess_num_proc=dataset_cfg.preprocess_num_proc,
                tokenize_num_proc=dataset_cfg.tokenize_num_proc,
                tokenize_batch_size=dataset_cfg.tokenize_batch_size,
            )
        else:
            eval_dataset = None
            if is_primary_process():
                logger.info("No evaluation file provided")
    else:
        if dataset_cfg.data_file is None:
            raise ValueError(
                "'data_file' must be provided when 'train_file' is omitted."
            )
        if is_primary_process():
            logger.info("Loading data from %s and splitting...", dataset_cfg.data_file)
        full_dataset = load_and_prepare_dataset(
            dataset_cfg.data_file,
            tokenizer,
            model_cfg.name,
            dataset_cfg.max_length,
            debug=general_cfg.debug,
            predictors=dataset_cfg.predictors,
            regular=training_cfg.regular,
            train_all=dataset_cfg.train_all,
            plain=dataset_cfg.plain,
            max_items=dataset_cfg.max_items,
            seed=general_cfg.finetune_seed,
            remove_thinking=dataset_cfg.remove_thinking,
            cache_dir=dataset_cfg.cache_dir,
            dataset_split=_resolve_dataset_split(
                dataset_cfg.data_file, dataset_cfg.train_split, default="train"
            ),
            config_name=dataset_cfg.config_name,
            preprocess_num_proc=dataset_cfg.preprocess_num_proc,
            tokenize_num_proc=dataset_cfg.tokenize_num_proc,
            tokenize_batch_size=dataset_cfg.tokenize_batch_size,
        )

        if dataset_cfg.eval_split > 0:
            split_dataset = full_dataset.train_test_split(
                test_size=dataset_cfg.eval_split,
                seed=dataset_cfg.split_seed,
                shuffle=True,
            )
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
            if dataset_cfg.max_eval_items is not None:
                eval_dataset = eval_dataset.select(
                    range(
                        min(len(eval_dataset), dataset_cfg.max_eval_items)
                    )
                )
        else:
            train_dataset = full_dataset
            eval_dataset = None

    if is_primary_process():
        logger.info("Train samples: %s", len(train_dataset))
        if eval_dataset:
            logger.info("Eval samples: %s", len(eval_dataset))
        else:
            logger.info("No evaluation dataset")

    return train_dataset, eval_dataset


def _create_training_arguments(
    config: Config, world_size: int, has_eval: bool
) -> TrainingArguments:
    training_cfg = config.training
    model_cfg = config.model
    general_cfg = config.general

    eval_steps = training_cfg.eval_steps
    output_dir = os.path.abspath(training_cfg.output_dir)
    evaluation_strategy = "steps" if has_eval else "no"
    ddp_backend = "nccl" if world_size > 1 else None

    dtype = training_cfg.dtype.lower()
    use_fp16 = dtype == "fp16"
    use_bf16 = dtype == "bf16"
    bf16_full_eval = use_bf16
    use_tf32 = dtype == "float32"

    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=training_cfg.batch_size,
        per_device_eval_batch_size=training_cfg.batch_size,
        gradient_accumulation_steps=training_cfg.grad_accum,
        learning_rate=training_cfg.learning_rate,
        num_train_epochs=training_cfg.num_epochs,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=training_cfg.save_total_limit,
        logging_dir=f"{training_cfg.output_dir}/logs",
        logging_steps=training_cfg.logging_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        bf16_full_eval=bf16_full_eval,
        gradient_checkpointing=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        ddp_backend=ddp_backend,
        fsdp="",
        fsdp_config={},
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=has_eval,
        tf32=use_tf32,
        optim=training_cfg.optimizer,
        seed=general_cfg.finetune_seed,
        data_seed=general_cfg.finetune_seed,
    )


def _save_model(
    trainer: Trainer,
    model,
    tokenizer,
    output_dir: Path,
    use_lora: bool,
) -> None:
    if use_lora:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        trainer.save_model()
        trainer.save_state()
        tokenizer.save_pretrained(output_dir)


def main(argv: List[str] | None = None) -> None:
    parser = build_argument_parser()
    cli_args = parser.parse_args(argv)
    config = load_config(cli_args.config)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _validate_config(config)
    _log_startup(config)

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        _init_distributed_if_needed()

    if is_primary_process():
        logger.info("")
        logger.info("1. Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(
        config.model.name,
        use_lora=config.model.use_lora,
        lora_rank=config.model.lora_rank,
        pretrain=config.model.pretrain,
        debug=config.general.debug,
        seed=config.general.finetune_seed,
        cache_dir=config.model.cache_dir,
        trust_remote_code=config.model.trust_remote_code,
    )

    if is_primary_process():
        logger.info("")
        logger.info("2. Loading and preparing dataset...")

    train_dataset, eval_dataset = _prepare_datasets(config, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None,
    )

    training_args = _create_training_arguments(
        config, world_size, eval_dataset is not None
    )

    callbacks = [ProfilerFLOPCallback()] if config.training.track_flop else []

    if config.training.regular:
        if is_primary_process():
            logger.info("")
            logger.info("3. Initialising regular trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
    else:
        if is_primary_process():
            logger.info("")
            logger.info("3. Initialising representation trainer...")
        trainer = RepresentationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            lbd=config.training.lbd,
            gamma=config.training.gamma,
            last_token=config.training.last_token,
            debug=config.general.debug,
            additive_mask=config.training.additive_mask,
            memory_efficient=config.training.memory_efficient,
        )

    if is_primary_process() and config.model.use_lora:
        logger.info("=== PEFT Model Check ===")
        model.print_trainable_parameters()
        trainable_params = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        logger.info("Trainable parameters: %s", len(trainable_params))
        if not trainable_params:
            logger.error("ERROR: No parameters require gradients!")
        else:
            logger.info("First few trainable params: %s", trainable_params[:5])

    if is_primary_process():
        logger.info("")
        logger.info("4. Starting training...")
    try:
        trainer.train()
    except Exception as error:
        if is_primary_process():
            logger.error("Training failed with error: %s", error)
            logger.warning(
                "This might be due to FSDP/sharding issues. Try running with LoRA enabled for a lighter configuration."
            )
        raise

    if is_primary_process():
        logger.info("")
        logger.info("5. Saving final model...")

    output_dir = Path(os.path.abspath(config.training.output_dir))
    retry_attempts = 3
    while retry_attempts > 0:
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            _save_model(trainer, model, tokenizer, output_dir, config.model.use_lora)
            break
        except Exception as error:
            if is_primary_process():
                logger.warning("Saving model encountered an error: %s", error)
            retry_attempts -= 1
            if retry_attempts <= 0:
                raise
            time.sleep(10)

    if is_primary_process():
        logger.info("")
        logger.info(
            "✅ Training completed! Model saved to %s", config.training.output_dir
        )
        logger.info("")
        logger.info("🎉 Fine-tuning finished successfully!")


if __name__ == "__main__":
    main()
