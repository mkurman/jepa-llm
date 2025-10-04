"""Utilities for manipulating conversational message sequences."""

from __future__ import annotations

import copy
from typing import List, Dict, Sequence

Message = Dict[str, str]


def get_messages(model_name: str, messages: Sequence[Message]) -> List[Message]:
    """Return messages formatted according to the target model's expectations."""
    if "google/gemma" in model_name:
        full_messages = copy.deepcopy(messages)[1:3]
        full_messages[0]["content"] = (
            messages[0]["content"] + "\n\n" + full_messages[0]["content"]
        )
        return full_messages
    return list(messages)


def get_user_messages(model_name: str, messages: Sequence[Message]) -> List[Message]:
    """Return messages representing the user turns for the configured model."""
    return copy.deepcopy(messages)[1:2]


def get_assistant_messages(
    model_name: str, dataset: str, messages: Sequence[Message]
) -> List[Message]:
    """Return assistant messages formatted for the specified model and dataset."""
    del dataset  # Dataset is unused but kept for signature compatibility.

    if "google/gemma" in model_name:
        assistant_messages = copy.deepcopy(messages)[2:3]
        assistant_messages[0]["role"] = "user"
        return assistant_messages
    return [message for message in messages if message["role"] == "assistant"]


__all__ = [
    "get_messages",
    "get_user_messages",
    "get_assistant_messages",
]
