"""Custom trainer implementing JEPA-style representation regularization."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from transformers import Trainer

from ..utils import is_primary_process


class RepresentationTrainer(Trainer):
    """Optimized trainer for representation regularization with JEPA-style loss."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.lbd = kwargs.pop("lbd", 1.0)
        self.gamma = kwargs.pop("gamma", 1.0)
        self.last_token = kwargs.pop("last_token", -2)
        self.debug = kwargs.pop("debug", 0)
        self.additive_mask = kwargs.pop("additive_mask", False)
        self.memory_efficient = kwargs.pop("memory_efficient", False)
        super().__init__(*args, **kwargs)
        self.gradient_accumulation_steps = getattr(
            self.args, "gradient_accumulation_steps", 1
        )

    def _last_token_index_vectorized(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        range_tensor = torch.arange(seq_len, device=attention_mask.device).expand(
            batch_size, -1
        )
        masked_positions = torch.where(attention_mask == 1, range_tensor, -1)
        last_valid_positions = masked_positions.max(dim=1)[0]
        result_indices = last_valid_positions + self.last_token
        result_indices = torch.clamp(result_indices, 0, seq_len - 1)

        if self.debug == 1 and is_primary_process():
            print(f"Last valid positions: {last_valid_positions}")
            print(f"Result indices: {result_indices}")

        return result_indices

    def _build_additive_mask_vectorized(
        self, lengths: torch.Tensor | list[int], max_seq_len: int, device: torch.device
    ) -> torch.Tensor:
        if isinstance(lengths, torch.Tensor):
            lengths_list = lengths.tolist()
        else:
            lengths_list = lengths

        base_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, device=device), diagonal=1
        )
        base_mask = torch.where(base_mask == 1, -torch.inf, 0.0)
        masks = base_mask.unsqueeze(0).expand(len(lengths_list), -1, -1).clone()

        for idx, length in enumerate(lengths_list):
            if length < max_seq_len:
                masks[idx, length:, :] = -torch.inf
                masks[idx, :, length:] = -torch.inf

        return masks

    def build_with_additive_mask_optimized(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = inputs["input_ids"].shape[0]
        seq_length = inputs["input_ids"].shape[-1]
        device = inputs["input_ids"].device

        last_token = self._last_token_index_vectorized(
            inputs["input_ids"], inputs["attention_mask"]
        )
        last_token_user = self._last_token_index_vectorized(
            inputs["input_ids_user"], inputs["attention_mask_user"]
        )
        last_token_assistant = self._last_token_index_vectorized(
            inputs["input_ids_assistant"], inputs["attention_mask_assistant"]
        )

        total_batch_size = batch_size * 2
        combined_input_ids = torch.empty(
            (total_batch_size, seq_length),
            dtype=inputs["input_ids"].dtype,
            device=device,
        )
        combined_labels = torch.empty(
            (total_batch_size, seq_length),
            dtype=inputs["labels"].dtype,
            device=device,
        )

        combined_input_ids[:batch_size] = inputs["input_ids"]
        combined_labels[:batch_size] = inputs["labels"]

        user_sequences = inputs["input_ids_user"].clone()
        user_labels = inputs["labels_user"].clone()

        for index in range(batch_size):
            length_user = last_token_user[index] + 1
            length_assistant = last_token_assistant[index] + 1
            end_pos = min(length_user + length_assistant, seq_length)
            assistant_len = min(length_assistant, seq_length - length_user)

            if length_user < seq_length and assistant_len > 0:
                user_sequences[index, length_user:end_pos] = inputs[
                    "input_ids_assistant"
                ][index, :assistant_len]
                user_labels[index, length_user:end_pos] = inputs["labels_assistant"][
                    index, :assistant_len
                ]

        combined_input_ids[batch_size:] = user_sequences
        combined_labels[batch_size:] = user_labels

        main_lengths = (last_token + 1).tolist()
        user_lengths = (last_token_user + last_token_assistant + 2).tolist()

        attention_mask = torch.full(
            (total_batch_size, 1, seq_length, seq_length), -torch.inf, device=device
        )

        main_masks = self._build_additive_mask_vectorized(main_lengths, seq_length, device)
        user_masks = self._build_additive_mask_vectorized(user_lengths, seq_length, device)

        attention_mask[:batch_size, 0] = main_masks
        attention_mask[batch_size:, 0] = user_masks

        self._last_token_user = last_token_user
        self._last_token_assistant = last_token_assistant + last_token_user + 1

        return {
            "input_ids": combined_input_ids,
            "labels": combined_labels,
            "attention_mask": attention_mask,
        }

    def forward_memory_efficient(
        self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        main_inputs = {
            "input_ids": inputs["input_ids"],
            "labels": inputs["labels"],
            "attention_mask": inputs["attention_mask"],
        }
        main_outputs = model(**main_inputs, output_hidden_states=True)

        user_inputs = {
            "input_ids": inputs["input_ids_user"],
            "labels": inputs["labels_user"],
            "attention_mask": inputs["attention_mask_user"],
        }
        user_outputs = model(**user_inputs, output_hidden_states=True)

        assistant_inputs = {
            "input_ids": inputs["input_ids_assistant"],
            "labels": inputs["labels_assistant"],
            "attention_mask": inputs["attention_mask_assistant"],
        }
        assistant_outputs = model(**assistant_inputs, output_hidden_states=True)

        user_hidden_states = user_outputs.hidden_states[-1]
        assistant_hidden_states = assistant_outputs.hidden_states[-1]

        if self.debug == 2 and is_primary_process():
            print(f"=====main_outputs.loss.shape:{main_outputs.loss.shape}=====")
            print(f"=====user_outputs.loss.shape:{user_outputs.loss.shape}=====")
            print(f"=====assistant_outputs.loss.shape:{assistant_outputs.loss.shape}=====")
            print(f"=====user_hidden_states.shape:{user_hidden_states.shape}=====")
            print(
                f"=====assistant_hidden_states.shape:{assistant_hidden_states.shape}====="
            )

        return {
            "main_outputs": main_outputs,
            "user_outputs": user_outputs,
            "assistant_outputs": assistant_outputs,
            "user_hidden_states": user_hidden_states,
            "assistant_hidden_states": assistant_hidden_states,
        }

    def forward(
        self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        if self.additive_mask:
            llm_inputs = self.build_with_additive_mask_optimized(inputs)
        else:
            batch_size = inputs["input_ids"].shape[0]
            seq_length = inputs["input_ids"].shape[-1]
            device = inputs["input_ids"].device
            total_batch_size = batch_size * 3

            combined_input_ids = torch.empty(
                (total_batch_size, seq_length),
                dtype=inputs["input_ids"].dtype,
                device=device,
            )
            combined_labels = torch.empty(
                (total_batch_size, seq_length),
                dtype=inputs["labels"].dtype,
                device=device,
            )
            combined_attention_mask = torch.empty(
                (total_batch_size, seq_length),
                dtype=inputs["attention_mask"].dtype,
                device=device,
            )

            combined_input_ids[:batch_size] = inputs["input_ids"]
            combined_input_ids[batch_size : 2 * batch_size] = inputs["input_ids_user"]
            combined_input_ids[2 * batch_size :] = inputs["input_ids_assistant"]

            combined_labels[:batch_size] = inputs["labels"]
            combined_labels[batch_size : 2 * batch_size] = inputs["labels_user"]
            combined_labels[2 * batch_size :] = inputs["labels_assistant"]

            combined_attention_mask[:batch_size] = inputs["attention_mask"]
            combined_attention_mask[batch_size : 2 * batch_size] = inputs[
                "attention_mask_user"
            ]
            combined_attention_mask[2 * batch_size :] = inputs[
                "attention_mask_assistant"
            ]

            llm_inputs = {
                "input_ids": combined_input_ids,
                "labels": combined_labels,
                "attention_mask": combined_attention_mask,
            }

        if self.debug >= 2 and is_primary_process():
            if self.debug == 7:
                torch.set_printoptions(threshold=float("inf"), linewidth=360)
                print(">>>input_ids<<<")
                print(llm_inputs["input_ids"])
                print(">>>labels<<<")
                print(llm_inputs["labels"])
                print(">>>attention_mask<<<")
                print(llm_inputs["attention_mask"])
                if self.additive_mask:
                    print(">>>last_token_user<<<")
                    print(self._last_token_user)
                    print(">>>last_token_assistant<<<")
                    print(self._last_token_assistant)
                raise SystemExit(0)
            elif self.debug == 2:
                print("=====before:outputs=====")
                print(f"input_ids shape: {llm_inputs['input_ids'].shape}")
                print(f"labels shape: {llm_inputs['labels'].shape}")
                print(f"attention_mask shape: {llm_inputs['attention_mask'].shape}")

        outputs = model(**llm_inputs, output_hidden_states=True)

        if self.debug == 2 and is_primary_process():
            print(f"=====outputs.loss.shape:{outputs.loss.shape}=====")
            print(
                f"=====outputs.hidden_states[-1].shape:{outputs.hidden_states[-1].shape}====="
            )

        hidden_states = outputs.hidden_states[-1]
        if self.additive_mask:
            batch_size = llm_inputs["input_ids"].shape[0] // 2
            user_hidden_states = hidden_states[batch_size:]
            assistant_hidden_states = user_hidden_states
        else:
            batch_size = llm_inputs["input_ids"].shape[0] // 3
            user_hidden_states = hidden_states[batch_size : 2 * batch_size]
            assistant_hidden_states = hidden_states[2 * batch_size :]

        if self.debug == 2 and is_primary_process():
            print(f"====={user_hidden_states.shape}=====")
            print(f"====={assistant_hidden_states.shape}=====")

        return {
            "main_outputs": outputs,
            "user_hidden_states": user_hidden_states,
            "assistant_hidden_states": assistant_hidden_states,
        }

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]] | torch.Tensor:
        if self.memory_efficient:
            return self.compute_loss_memory_efficient(
                model, inputs, return_outputs, num_items_in_batch
            )

        batch_size = inputs["input_ids_user"].shape[0]

        if not self.additive_mask:
            index_user = self._last_token_index_vectorized(
                inputs["input_ids_user"], inputs["attention_mask_user"]
            )
            index_assistant = self._last_token_index_vectorized(
                inputs["input_ids_assistant"], inputs["attention_mask_assistant"]
            )

        if self.debug == 1 and is_primary_process():
            print("=====last tokens=====")
            batch_indices = torch.arange(batch_size)
            print(inputs["input_ids_user"][batch_indices, index_user])
            print(inputs["input_ids_user"][batch_indices, index_user - 1])
            print(inputs["input_ids_assistant"][batch_indices, index_assistant])
            print(inputs["input_ids_assistant"][batch_indices, index_assistant - 1])

        forward_results = self.forward(model, inputs)

        if self.additive_mask:
            index_user = self._last_token_user
            index_assistant = self._last_token_assistant

        main_outputs = forward_results["main_outputs"]
        lm_loss = main_outputs.loss
        user_hidden_states = forward_results["user_hidden_states"]
        assistant_hidden_states = forward_results["assistant_hidden_states"]

        batch_indices = torch.arange(batch_size, device=user_hidden_states.device)
        user_embedding = user_hidden_states[batch_indices, index_user, :]
        assistant_embedding = assistant_hidden_states[batch_indices, index_assistant, :]

        cosine_similarity = F.cosine_similarity(
            user_embedding, assistant_embedding, dim=-1
        )

        if self.debug == 1 and is_primary_process():
            print(f"User embedding shape: {user_embedding.shape}")
            print(f"Assistant embedding shape: {assistant_embedding.shape}")
            print(f"Cosine similarity shape: {cosine_similarity.shape}")

        jepa_loss = 1.0 - cosine_similarity.mean()
        total_loss = self.gamma * lm_loss + self.lbd * jepa_loss

        if self.model.training:
            total_loss = total_loss / self.gradient_accumulation_steps

        if self.debug >= 1 and is_primary_process():
            if self.debug in [1, 2]:
                print(
                    f"LM loss: {lm_loss.item():.4f}, JEPA loss: {jepa_loss.item():.4f}"
                )
                raise SystemExit(0)
            elif self.debug == 5:
                print(
                    f"llm_loss: {lm_loss.float():.4f}, jepa_loss: {jepa_loss.float():.4f}"
                )

        return (total_loss, main_outputs) if return_outputs else total_loss

    def compute_loss_memory_efficient(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]] | torch.Tensor:
        batch_size = inputs["input_ids_user"].shape[0]

        index_user = self._last_token_index_vectorized(
            inputs["input_ids_user"], inputs["attention_mask_user"]
        )
        index_assistant = self._last_token_index_vectorized(
            inputs["input_ids_assistant"], inputs["attention_mask_assistant"]
        )

        if self.debug == 1 and is_primary_process():
            print("=====last tokens (memory efficient)=====")
            batch_indices = torch.arange(batch_size)
            print(inputs["input_ids_user"][batch_indices, index_user])
            print(inputs["input_ids_user"][batch_indices, index_user - 1])
            print(inputs["input_ids_assistant"][batch_indices, index_assistant])
            print(inputs["input_ids_assistant"][batch_indices, index_assistant - 1])

        forward_results = self.forward_memory_efficient(model, inputs)

        main_outputs = forward_results["main_outputs"]
        user_hidden_states = forward_results["user_hidden_states"]
        assistant_hidden_states = forward_results["assistant_hidden_states"]

        lm_loss = main_outputs.loss

        batch_indices = torch.arange(batch_size, device=user_hidden_states.device)
        user_embedding = user_hidden_states[batch_indices, index_user, :]
        assistant_embedding = assistant_hidden_states[batch_indices, index_assistant, :]

        cosine_similarity = F.cosine_similarity(
            user_embedding, assistant_embedding, dim=-1
        )

        if self.debug == 1 and is_primary_process():
            print(f"User embedding shape: {user_embedding.shape}")
            print(f"Assistant embedding shape: {assistant_embedding.shape}")
            print(f"Cosine similarity shape: {cosine_similarity.shape}")

        jepa_loss = 1.0 - cosine_similarity.mean()
        total_loss = self.gamma * lm_loss + self.lbd * jepa_loss

        if self.model.training:
            total_loss = total_loss / self.gradient_accumulation_steps

        if self.debug >= 1 and is_primary_process():
            if self.debug in [1, 2]:
                print(
                    f"LM loss (avg): {lm_loss.item():.4f}, JEPA loss: {jepa_loss.item():.4f}"
                )
                print(f"Main loss: {main_outputs.loss.item():.4f}")
                print(f"User loss: {forward_results['user_outputs'].loss.item():.4f}")
                print(
                    f"Assistant loss: {forward_results['assistant_outputs'].loss.item():.4f}"
                )
                raise SystemExit(0)
            elif self.debug == 5:
                print(
                    f"llm_loss: {lm_loss.float():.4f}, jepa_loss: {jepa_loss.float():.4f}"
                )

        return (total_loss, main_outputs) if return_outputs else total_loss


__all__ = ["RepresentationTrainer"]
