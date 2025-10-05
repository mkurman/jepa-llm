"""Dataset loading and preparation utilities."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Set

import logging

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from .messages import get_assistant_messages, get_messages, get_user_messages
from .text_cleanup import remove_thinking_content
from .utils import is_primary_process

Message = Dict[str, str]


logger = logging.getLogger(__name__)


INPUT_COLUMN_CANDIDATES = [
    "question",
    "input",
    "instruction",
    "prompt",
]

OUTPUT_COLUMN_CANDIDATES = [
    "response",
    "output",
    "completion",
    "answer",
]

SYSTEM_COLUMN_CANDIDATES = [
    "system_prompt",
    "system",
    "system_message",
]


def create_labels_for_all(input_ids: Sequence[int], attention_mask: Sequence[int]) -> List[int]:
    """Create labels for all tokens except those masked out by padding."""
    labels: List[int] = []
    for token_id, mask in zip(input_ids, attention_mask):
        labels.append(token_id if mask != 0 else -100)
    return labels


def create_masked_labels(
    messages: Sequence[Message],
    tokenizer: PreTrainedTokenizerBase,
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    debug: int = 0,
) -> List[int]:
    """Mask out user tokens leaving only assistant content for loss computation."""
    labels = [-100] * len(input_ids)

    for index, mask in enumerate(attention_mask):
        if mask == 0:
            labels[index] = -100

    for message in messages:
        if message["role"] != "assistant":
            continue

        assistant_content = message["content"]
        assistant_tokens = tokenizer.encode(
            assistant_content, add_special_tokens=False
        )

        decoded_assistant = [tokenizer.decode(token) for token in assistant_tokens]
        decoded_input = [tokenizer.decode(token) for token in input_ids]

        for start_index in range(len(input_ids) - len(assistant_tokens) + 1):
            window = decoded_input[start_index : start_index + len(assistant_tokens)]
            if window == decoded_assistant:
                for offset in range(len(assistant_tokens)):
                    current_index = start_index + offset
                    if attention_mask[current_index] == 1:
                        labels[current_index] = input_ids[current_index]
                break

        if debug == 4 and is_primary_process():
            logger.debug("assistant_tokens: %s", assistant_tokens)
            logger.debug("decoded assistant: %s", decoded_assistant)
            logger.debug("decoded input: %s", decoded_input)
            torch.cuda.synchronize()
            raise SystemExit(0)

    return labels


def _clean_messages(record: Dict[str, Any], *, remove_thinking: bool) -> Dict[str, Any]:
    cleaned_messages = []
    for message in record["messages"]:
        content = message["content"]
        if remove_thinking:
            content = remove_thinking_content(content)
        cleaned_messages.append({"role": message["role"], "content": content.strip()})
    return {"messages": cleaned_messages}


def _tokenize_conversations(
    examples: Dict[str, Any],
    *,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    max_length: int,
    debug: int,
    predictors: int,
    regular: bool,
    train_all: bool,
    plain: bool,
) -> Dict[str, Any]:
    input_ids_list: List[List[int]] = []
    labels_list: List[List[int]] = []
    attention_mask_list: List[List[int]] = []
    user_input_ids_list: List[List[int]] = []
    user_labels_list: List[List[int]] = []
    user_attention_mask_list: List[List[int]] = []
    assistant_input_ids_list: List[List[int]] = []
    assistant_labels_list: List[List[int]] = []
    assistant_attention_mask_list: List[List[int]] = []

    for messages in examples["messages"]:
        full_messages = get_messages(model_name, messages)
        if plain:
            if train_all:
                formatted_chat = messages[1]["content"] + "<|eot_id|>"
            else:
                formatted_chat = (
                    messages[1]["content"]
                    + "<|perception|>"
                    + messages[2]["content"]
                    + "<|eot_id|>"
                )
        else:
            formatted_chat = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        tokenized = tokenizer(
            formatted_chat,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        if train_all:
            labels = create_labels_for_all(input_ids, attention_mask)
        else:
            labels = create_masked_labels(
                messages, tokenizer, input_ids, attention_mask, debug=debug
            )

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

        user_messages = get_user_messages(model_name, messages)
        predictors_to_add = predictors
        while predictors_to_add > 0:
            user_messages[0]["content"] += f"<|predictor_{predictors_to_add}|>"
            predictors_to_add -= 1

        if plain:
            formatted_chat_user = user_messages[0]["content"]
        else:
            formatted_chat_user = tokenizer.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=False
            )

        tokenized_user = tokenizer(
            formatted_chat_user,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        user_input_ids_list.append(tokenized_user["input_ids"])
        user_labels_list.append([-100] * len(tokenized_user["input_ids"]))
        user_attention_mask_list.append(tokenized_user["attention_mask"])

        assistant_messages = get_assistant_messages(model_name, messages)
        if plain:
            formatted_chat_assistant = assistant_messages[0]["content"]
        else:
            formatted_chat_assistant = tokenizer.apply_chat_template(
                assistant_messages, tokenize=False, add_generation_prompt=False
            )
        tokenized_assistant = tokenizer(
            formatted_chat_assistant,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        assistant_input_ids_list.append(tokenized_assistant["input_ids"])
        assistant_labels_list.append([-100] * len(tokenized_assistant["input_ids"]))
        assistant_attention_mask_list.append(tokenized_assistant["attention_mask"])

        if debug == 3 and is_primary_process():
            logger.debug("messages: %s", messages)
            logger.debug("input_ids_list: %s", input_ids_list)
            logger.debug("decoded first input_ids: %s", tokenizer.decode(input_ids_list[0]))
            logger.debug("labels_list: %s", labels_list)
            logger.debug(
                "decoded labels: %s",
                tokenizer.decode([item for item in labels_list[0] if item != -100]),
            )
            logger.debug("attention_mask_list: %s", attention_mask_list)
            logger.debug("user Token IDs: %s", tokenized_user["input_ids"])
            logger.debug(
                "user Decoded: %s", tokenizer.decode(tokenized_user["input_ids"])
            )
            logger.debug("assistant Token IDs: %s", tokenized_assistant["input_ids"])
            logger.debug(
                "assistant Decoded: %s",
                tokenizer.decode(tokenized_assistant["input_ids"]),
            )
            raise SystemExit(0)

    if regular:
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list,
        }

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
        "input_ids_user": user_input_ids_list,
        "labels_user": user_labels_list,
        "attention_mask_user": user_attention_mask_list,
        "input_ids_assistant": assistant_input_ids_list,
        "labels_assistant": assistant_labels_list,
        "attention_mask_assistant": assistant_attention_mask_list,
    }


def load_and_prepare_dataset(
    data_file: str,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    max_length: int = 2048,
    debug: int = 0,
    predictors: int = 0,
    regular: bool = False,
    train_all: bool = False,
    plain: bool = False,
    max_items: int | None = None,
    seed: int = 42,
    remove_thinking: bool = True,
    cache_dir: Optional[str] = None,
    dataset_split: Optional[str] = None,
    config_name: Optional[str] = None,
    preprocess_num_proc: int = 24,
    tokenize_num_proc: int = 12,
    tokenize_batch_size: int = 1000,
) -> Dataset:
    """Load the dataset file and return tokenized conversations."""

    resolved_split = dataset_split or "train"

    load_kwargs = {"data_files": data_file, "split": resolved_split}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    if data_file.endswith(".jsonl"):
        dataset = load_dataset("json", **load_kwargs)
    elif data_file.endswith(".parquet"):
        dataset = load_dataset("parquet", **load_kwargs)
    else:
        extra_kwargs: Dict[str, Any] = {"split": resolved_split}
        if cache_dir:
            extra_kwargs["cache_dir"] = cache_dir
        if config_name:
            extra_kwargs["name"] = config_name
        dataset = load_dataset(data_file, **extra_kwargs)

    if is_primary_process():
        logger.info("Loaded %d examples from %s", len(dataset), data_file)

    dataset = _ensure_messages_column(dataset)

    clean_fn = partial(_clean_messages, remove_thinking=remove_thinking)

    dataset = dataset.map(
        clean_fn,
        num_proc=preprocess_num_proc,
    ).filter(
        lambda record: record["messages"] is not None
        and all(message["content"] for message in record["messages"]),
        num_proc=preprocess_num_proc,
    )

    dataset = dataset.shuffle(seed=seed)

    if max_items is not None:
        dataset = dataset.select(range(min(len(dataset), max_items)))

    map_fn = partial(
        _tokenize_conversations,
        tokenizer=tokenizer,
        model_name=model_name,
        max_length=max_length,
        debug=debug,
        predictors=predictors,
        regular=regular,
        train_all=train_all,
        plain=plain,
    )

    tokenized_dataset = dataset.map(
        map_fn,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=dataset.column_names,
        num_proc=tokenize_num_proc,
    )

    return tokenized_dataset


def _find_first_column(columns: Set[str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _ensure_messages_column(dataset: Dataset) -> Dataset:
    columns: Set[str] = set(dataset.column_names)

    if "messages" in columns:
        return dataset

    if "conversations" in columns and "messages" not in columns:
        dataset = dataset.rename_column("conversations", "messages")
        columns = set(dataset.column_names)
        if "messages" in columns:
            return dataset

    input_column = _find_first_column(columns, INPUT_COLUMN_CANDIDATES)
    output_column = _find_first_column(columns, OUTPUT_COLUMN_CANDIDATES)

    if input_column and output_column:
        system_column = _find_first_column(columns, SYSTEM_COLUMN_CANDIDATES)

        def map_record(
            record: Dict[str, Any],
            *,
            system_column: Optional[str] = system_column,
            input_column: str = input_column,
            output_column: str = output_column,
        ) -> Dict[str, Any]:
            messages: List[Message] = []

            if system_column:
                system_content = record.get(system_column)
                if system_content:
                    messages.append({"role": "system", "content": system_content})

            user_content = record.get(input_column)
            if user_content:
                messages.append({"role": "user", "content": user_content})

            assistant_content = record.get(output_column)
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            return {"messages": messages}

        dataset = dataset.map(map_record)

    return dataset


__all__ = [
    "create_labels_for_all",
    "create_masked_labels",
    "load_and_prepare_dataset",
]
