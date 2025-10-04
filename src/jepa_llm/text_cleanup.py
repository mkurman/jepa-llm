"""Utility helpers for cleaning assistant outputs."""

from __future__ import annotations

import re


THINKING_PATTERNS = [
    (r"<think>.*?</think>", ""),
    (r"<think>", ""),
    (r"</think>", ""),
    (r"<Thought>.*?</Thought>", ""),
    (r"<Thought>", ""),
    (r"</Thought>", ""),
    (r"<Output>", ""),
    (r"</Output>", ""),
    (r"<Answer>", ""),
    (r"</Answer>", ""),
]


def remove_thinking_content(text: str) -> str:
    """Remove hidden reasoning or meta content enclosed in known tags."""
    cleaned_text = text
    for pattern, replacement in THINKING_PATTERNS:
        cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.DOTALL)
    return cleaned_text


__all__ = ["remove_thinking_content"]
