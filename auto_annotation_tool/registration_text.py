"""Shared registration-number normalization for Android/Desktop interoperability."""

from __future__ import annotations

from typing import Any


NORMALIZATION_POLICY = "uppercase_alphanumeric.v1"


def normalize_registration(value: Any) -> str:
    """
    Canonical registration key used for comparison and grouping.

    Raw predictions are preserved elsewhere.

    Rules:
    - ignore letter case,
    - remove whitespace and non-alphanumeric separators,
    - do not substitute O <-> 0 or I <-> 1.
    """
    return "".join(
        char
        for char in str(value or "").upper()
        if char.isalnum()
    )


def registration_key(value: Any) -> str:
    """Alias for code that deals with grouping rather than comparison."""
    return normalize_registration(value)