"""Command-line interface for fine-tuning JEPA-style language models."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from .callbacks.profiler_flop_callback import ProfilerFLOPCallback
from .config import Config, load_config
from .dataset import load_and_prepare_dataset
from .model_setup import setup_model_and_tokenizer
from .trainers.representation_trainer import RepresentationTrainer
from .utils import is_primary_process


def build_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser that only accepts the configuration file."""

    parser = argparse.ArgumentParser(description="Fine-tune JEPA-enhanced LLMs")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML configuration file containing all run settings.",
    )
    return parser


def _validate_config(config: Config) -> None:
    dataset_cfg = config.dataset
    training_cfg = config.training

    has_train_split = bool(dataset_cfg.train_file)
    has_combined_file = bool(dataset_cfg.data_file)

    if has_train_split == has_combined_file:
        raise ValueError("Configuration must provide either 'train_file' or 'data_file'.")

    if training_cfg.memory_efficient and training_cfg.additive_mask:
        raise ValueError("'memory_efficient' and 'additive_mask' modes are mutually exclusive.")


def _log_startup(config: Config) -> None:
    if not is_primary_process():
        return

    dataset_cfg = config.dataset
    model_cfg = config.model
    training_cfg = config.training

    print("=== Fine-tuning Script ===")

    if dataset_cfg.train_file:
        print(f"Train file: {dataset_cfg.train_file}")
        if dataset_cfg.eval_file:
            print(f"Eval file: {dataset_cfg.eval_file}")
        else:
            print("No eval file provided - training without evaluation")
    else:
        print(
            f"Data file: {dataset_cfg.data_file} (will split {dataset_cfg.eval_split:.1%} for eval)"
        )

    print(f"Model: {model_cfg.name}")
    print(f"Output: {training_cfg.output_dir}")
    print(f"Using LoRA: {model_cfg.use_lora}")
    if model_cfg.use_lora:
        print(f"LoRA rank: {model_cfg.lora_rank}")
    print(f"Memory efficient mode: {training_cfg.memory_efficient}")
    if training_cfg.memory_efficient:
        print("  → Will process sequences separately to reduce VRAM usage by 2-3x")


def _init_distributed_if_needed() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1 and torch.cuda.is_available():
        if is_primary_process():
            print(
                f"Running with torchrun: world_size={world_size}, local_rank={local_rank}"
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
            print(f"Loading training data from {dataset_cfg.train_file}")
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
        )

        if dataset_cfg.eval_file:
            if is_primary_process():
                print(f"Loading evaluation data from {dataset_cfg.eval_file}")
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
                max_items=dataset_cfg.max_items,
                seed=general_cfg.finetune_seed,
                remove_thinking=dataset_cfg.remove_thinking,
                cache_dir=dataset_cfg.cache_dir,
            )
        else:
            eval_dataset = None
            if is_primary_process():
                print("No evaluation file provided")
    else:
        if dataset_cfg.data_file is None:
            raise ValueError("'data_file' must be provided when 'train_file' is omitted.")
        if is_primary_process():
            print(f"Loading data from {dataset_cfg.data_file} and splitting...")
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
        )

        if dataset_cfg.eval_split > 0:
            split_dataset = full_dataset.train_test_split(
                test_size=dataset_cfg.eval_split,
                seed=dataset_cfg.split_seed,
                shuffle=True,
            )
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            train_dataset = full_dataset
            eval_dataset = None

    if is_primary_process():
        print(f"Train samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Eval samples: {len(eval_dataset)}")
        else:
            print("No evaluation dataset")

    return train_dataset, eval_dataset


def _create_training_arguments(
    config: Config, world_size: int, has_eval: bool
) -> TrainingArguments:
    training_cfg = config.training
    model_cfg = config.model
    general_cfg = config.general

    eval_steps = (
        training_cfg.eval_steps
        if not model_cfg.pretrain
        else training_cfg.eval_steps * 20
    )
    output_dir = os.path.abspath(training_cfg.output_dir)
    evaluation_strategy = "steps" if has_eval else "no"
    ddp_backend = "nccl" if world_size > 1 else None

    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=training_cfg.batch_size,
        per_device_eval_batch_size=training_cfg.batch_size,
        gradient_accumulation_steps=training_cfg.grad_accum,
        learning_rate=training_cfg.learning_rate,
        num_train_epochs=training_cfg.num_epochs,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        logging_dir=f"{training_cfg.output_dir}/logs",
        logging_steps=1,
        fp16=False,
        bf16=True,
        bf16_full_eval=True,
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
        tf32=False,
        optim="adamw_8bit",
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

    _validate_config(config)
    _log_startup(config)

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        _init_distributed_if_needed()

    if is_primary_process():
        print("\n1. Loading model and tokenizer...")
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
        print("\n2. Loading and preparing dataset...")

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
            print("\n3. Initialising regular trainer...")
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
            print("\n3. Initialising representation trainer...")
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
        print("=== PEFT Model Check ===")
        model.print_trainable_parameters()
        trainable_params = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        print(f"Trainable parameters: {len(trainable_params)}")
        if not trainable_params:
            print("ERROR: No parameters require gradients!")
        else:
            print("First few trainable params:", trainable_params[:5])

    if is_primary_process():
        print("\n4. Starting training...")
    try:
        trainer.train()
    except Exception as error:
        if is_primary_process():
            print(f"Training failed with error: {error}")
            print(
                "This might be due to FSDP/sharding issues. Try running with LoRA enabled for a lighter configuration."
            )
        raise

    if is_primary_process():
        print("\n5. Saving final model...")

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
                print(f"Success Rate: Saving model encounter error: {error}")
            retry_attempts -= 1
            if retry_attempts <= 0:
                raise
            time.sleep(10)

    if is_primary_process():
        print(f"\n✅ Training completed! Model saved to {config.training.output_dir}")
        print("\n🎉 Fine-tuning finished successfully!")


if __name__ == "__main__":
    main()
