"""Model and tokenizer setup helpers."""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from .utils import is_primary_process


logger = logging.getLogger(__name__)


SPECIAL_TOKENS = [
    "<|predictor_1|>",
    "<|predictor_2|>",
    "<|predictor_3|>",
    "<|predictor_4|>",
    "<|predictor_5|>",
    "<|predictor_6|>",
    "<|predictor_7|>",
    "<|predictor_8|>",
    "<|predictor_9|>",
    "<|predictor_10|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|perception|>",
]


def setup_model_and_tokenizer(
    model_name: str,
    use_lora: bool = True,
    lora_rank: int = 16,
    pretrain: bool = False,
    debug: int = 0,
    seed: int | None = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load tokenizer and model, applying optional LoRA fine-tuning adapters."""

    tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
    if cache_dir:
        tokenizer_kwargs["cache_dir"] = cache_dir

    if "apple/OpenELM" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", **tokenizer_kwargs
        )
        tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'system' %}
{{ '<|system|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>\n'  + message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ '<|assistant|>' }}
{% endif %}
{% endfor %}"""
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    if "microsoft/phi" in model_name:
        tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
        if is_primary_process():
            logger.info("Added <|startoftext|> token")

    new_tokens = [token for token in SPECIAL_TOKENS if token not in tokenizer.vocab]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        if is_primary_process():
            logger.info("Added %d new special tokens", len(new_tokens))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = None
    if torch.cuda.is_available():
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size == 1:
            device_map = "auto"

    if pretrain:
        if seed is not None:
            torch.manual_seed(seed)
        config_kwargs = {"trust_remote_code": trust_remote_code}
        if cache_dir:
            config_kwargs["cache_dir"] = cache_dir
        config = AutoConfig.from_pretrained(model_name, **config_kwargs)
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
        )
        dist_available = torch.distributed.is_available()
        dist_initialized = dist_available and torch.distributed.is_initialized()

        if torch.cuda.is_available():
            if dist_initialized:
                rank = torch.distributed.get_rank()
                device = torch.device(f"cuda:{rank}")
            else:
                device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        model.to(device)

        if dist_initialized and torch.distributed.get_world_size() > 1:
            for parameter in model.parameters():
                torch.distributed.broadcast(parameter.data, src=0)
            for buffer in model.buffers():
                torch.distributed.broadcast(buffer.data, src=0)
        if debug == 6:
            for name, parameter in model.named_parameters():
                logger.debug(
                    "Parameter name: %s, Shape: %s", name, parameter.shape
                )
                logger.debug("%s", parameter)
                raise SystemExit(0)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            use_cache=False,
            cache_dir=cache_dir,
        )

    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        if is_primary_process():
            model.print_trainable_parameters()

    return model, tokenizer


__all__ = ["setup_model_and_tokenizer"]
