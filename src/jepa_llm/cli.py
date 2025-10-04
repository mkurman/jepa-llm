"""Command-line interface for fine-tuning JEPA-style language models."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .callbacks.profiler_flop_callback import ProfilerFLOPCallback
from .dataset import load_and_prepare_dataset
from .model_setup import setup_model_and_tokenizer
from .trainers.representation_trainer import RepresentationTrainer
from .utils import is_primary_process


def _load_config_file(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping of options")
    return data


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3-1B")
    parser.add_argument("--config", type=str, default=None, help="YAML configuration file")
    parser.add_argument("--train_file", type=str, help="Path to training JSONL file")
    parser.add_argument("--eval_file", type=str, help="Path to evaluation JSONL file")
    parser.add_argument(
        "--data_file", type=str, help="Path to single JSONL file (will be split)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name/path",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./llama3-1b-fted", help="Output directory"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Per device batch size"
    )
    parser.add_argument(
        "--max_items", type=int, default=None, help="Number of items from dataset"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--eval_steps", type=int, default=10, help="Evaluation steps")
    parser.add_argument(
        "--lora", action="store_true", help="Enable LoRA (default: full fine-tuning)"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16, help="LoRA rank. Default: 16."
    )
    parser.add_argument(
        "--eval_split",
        type=float,
        default=0.2,
        help="Evaluation split ratio (if using single data file)",
    )
    parser.add_argument(
        "--split_seed", type=int, default=42, help="Random seed for train/eval split"
    )
    parser.add_argument(
        "--finetune_seed", type=int, default=42, help="Random seed for fine-tuning"
    )
    parser.add_argument(
        "--predictors", type=int, default=0, help="Number of predictor tokens"
    )
    parser.add_argument(
        "--lbd", type=float, default=0.1, help="Lambda for similarity loss"
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma for LLM loss")
    parser.add_argument(
        "--last_token",
        type=int,
        default=-1,
        help="Index of last token, -1 is '<|eot|>'",
    )
    parser.add_argument(
        "--debug", type=int, default=0, help="Debug level. 0 means no debug"
    )
    parser.add_argument(
        "--regular", action="store_true", help="Use regular transformer."
    )
    parser.add_argument(
        "--track_flop", action="store_true", help="Whether to track FLOPs."
    )
    parser.add_argument(
        "--pretrain", action="store_true", help="Whether to pretrain from scratch."
    )
    parser.add_argument(
        "--train_all",
        action="store_true",
        help="Whether to compute loss from all tokens.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="When set, do not apply chat format.",
    )
    parser.add_argument(
        "--additive_mask",
        action="store_true",
        help="Use an additive mask to compute both user and assistant in 1 forward pass.",
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help=(
            "Process sequences separately instead of concatenating them, reducing VRAM usage."
        ),
    )
    return parser


def parse_arguments(
    argv: List[str] | None = None,
) -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = build_argument_parser()
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = base_parser.parse_known_args(argv)

    config_defaults: Dict[str, Any] = {}

    if config_args.config:
        config_defaults = _load_config_file(config_args.config)
        config_defaults.setdefault("config", config_args.config)

    if config_defaults:
        valid_options = {action.dest for action in parser._actions if action.dest}
        filtered_defaults = {
            key: value for key, value in config_defaults.items() if key in valid_options
        }
        parser.set_defaults(**filtered_defaults)

    args = parser.parse_args(remaining)
    return args, parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.train_file and not args.data_file:
        parser.error("Must provide either --train_file or --data_file")

    if args.train_file and args.data_file:
        parser.error("Cannot use both --train_file and --data_file. Choose one.")

    if args.memory_efficient and args.additive_mask:
        parser.error("Cannot use both --memory_efficient and --additive_mask. Choose one.")


def _log_startup(args: argparse.Namespace) -> None:
    if not is_primary_process():
        return

    print("=== Fine-tuning Script ===")
    if args.train_file:
        print(f"Train file: {args.train_file}")
        if args.eval_file:
            print(f"Eval file: {args.eval_file}")
        else:
            print("No eval file provided - training without evaluation")
    else:
        print(f"Data file: {args.data_file} (will split {args.eval_split:.1%} for eval)")

    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Using LoRA: {args.lora}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Memory efficient mode: {args.memory_efficient}")
    if args.memory_efficient:
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


def _prepare_datasets(args: argparse.Namespace, tokenizer) -> tuple:
    if args.train_file:
        if is_primary_process():
            print(f"Loading training data from {args.train_file}")
        train_dataset = load_and_prepare_dataset(
            args.train_file,
            tokenizer,
            args.model_name,
            args.max_length,
            predictors=args.predictors,
            regular=args.regular,
            debug=args.debug,
            train_all=args.train_all,
            plain=args.plain,
            max_items=args.max_items,
            seed=args.finetune_seed,
        )

        if args.eval_file:
            if is_primary_process():
                print(f"Loading evaluation data from {args.eval_file}")
            eval_dataset = load_and_prepare_dataset(
                args.eval_file,
                tokenizer,
                args.model_name,
                args.max_length,
                regular=args.regular,
                debug=args.debug,
                train_all=args.train_all,
                plain=args.plain,
                max_items=args.max_items,
                seed=args.finetune_seed,
            )
        else:
            eval_dataset = None
            if is_primary_process():
                print("No evaluation file provided")
    else:
        if is_primary_process():
            print(f"Loading data from {args.data_file} and splitting...")
        full_dataset = load_and_prepare_dataset(
            args.data_file,
            tokenizer,
            args.model_name,
            args.max_length,
            predictors=args.predictors,
            regular=args.regular,
            debug=args.debug,
            train_all=args.train_all,
            plain=args.plain,
            max_items=args.max_items,
            seed=args.finetune_seed,
        )

        if args.eval_split > 0:
            split_dataset = full_dataset.train_test_split(
                test_size=args.eval_split, seed=args.split_seed, shuffle=True
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
    args: argparse.Namespace, world_size: int, has_eval: bool
) -> TrainingArguments:
    eval_steps = args.eval_steps if not args.pretrain else args.eval_steps * 20
    output_dir = os.path.abspath(args.output_dir)
    evaluation_strategy = "steps" if has_eval else "no"
    ddp_backend = "nccl" if world_size > 1 else None

    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        logging_dir=f"{args.output_dir}/logs",
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
        seed=args.finetune_seed,
        data_seed=args.finetune_seed,
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
    args, parser = parse_arguments(argv)
    _validate_args(args, parser)

    _log_startup(args)

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        _init_distributed_if_needed()

    if is_primary_process():
        print("\n1. Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        use_lora=args.lora,
        lora_rank=args.lora_rank,
        pretrain=args.pretrain,
        debug=args.debug,
        seed=args.finetune_seed,
    )

    if is_primary_process():
        print("\n2. Loading and preparing dataset...")

    train_dataset, eval_dataset = _prepare_datasets(args, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None,
    )

    training_args = _create_training_arguments(args, world_size, eval_dataset is not None)

    callbacks = [ProfilerFLOPCallback()] if args.track_flop else []

    if args.regular:
        if is_primary_process():
            print("\n3. Initializing regular trainer...")
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
            print("\n3. Initializing representation trainer...")
        trainer = RepresentationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            lbd=args.lbd,
            gamma=args.gamma,
            last_token=args.last_token,
            debug=args.debug,
            additive_mask=args.additive_mask,
            memory_efficient=args.memory_efficient,
        )

    if is_primary_process() and args.lora:
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
                "This might be due to FSDP/sharding issues. Try running with --lora flag for LoRA fine-tuning."
            )
        raise

    if is_primary_process():
        print("\n5. Saving final model...")

    output_dir = Path(os.path.abspath(args.output_dir))
    retry_attempts = 3
    while retry_attempts > 0:
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            _save_model(trainer, model, tokenizer, output_dir, args.lora)
            break
        except Exception as error:
            if is_primary_process():
                print(f"Success Rate: Saving model encounter error: {error}")
            retry_attempts -= 1
            if retry_attempts <= 0:
                raise
            time.sleep(10)

    if is_primary_process():
        print(f"\n✅ Training completed! Model saved to {args.output_dir}")
        print("\n🎉 Fine-tuning finished successfully!")


if __name__ == "__main__":
    main()
