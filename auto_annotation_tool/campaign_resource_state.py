"""Normalized resource snapshots for campaign transitions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .campaign_resource_catalog import (
    campaign_resource_counted,
    campaign_resource_description,
    campaign_resource_label,
    campaign_resource_short_label,
    normalize_campaign_resource_key,
)


@dataclass(frozen=True)
class CampaignResourceSnapshot:
    """UI-neutral state of one campaign resource.

    This object is deliberately small. It can be built from the current E1
    widgets today, and from a pure campaign state store later.
    """

    key: str
    canonical_key: str
    code: str
    label: str
    requirement: str = "optional"
    source: str = "Nie wskazano"
    validation: str = "Brak"
    tone: str = "muted"
    counter_text: str = ""
    description: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def counted(self) -> bool:
        return campaign_resource_counted(self.canonical_key)

    @property
    def display_label(self) -> str:
        if self.counter_text and self.counted:
            return f"{self.label} [{self.counter_text}]"
        return self.label

    @property
    def has_source(self) -> bool:
        value = str(self.source or "").strip().lower()
        return bool(value and value not in {"nie wskazano", "brak", "-"})

    @property
    def counter_value(self) -> int:
        match = re.search(r"\d+", str(self.counter_text or ""))
        if not match:
            return 0
        try:
            return int(match.group(0))
        except Exception:
            return 0

    @property
    def is_required(self) -> bool:
        return str(self.requirement or "").strip().lower() in {
            "required",
            "route_required",
            "route_required_plate",
            "route_required_char",
        }

    @property
    def is_enabled(self) -> bool:
        return str(self.requirement or "").strip().lower() != "disabled"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "canonical_key": self.canonical_key,
            "code": self.code,
            "label": self.label,
            "display_label": self.display_label,
            "requirement": self.requirement,
            "source": self.source,
            "validation": self.validation,
            "tone": self.tone,
            "counter": self.counter_text,
            "counter_value": str(self.counter_value),
            "description": self.description,
            "meta": dict(self.meta or {}),
        }


def _safe_var_get(value: Any, fallback: str = "") -> str:
    try:
        getter = getattr(value, "get", None)
        if callable(getter):
            return str(getter() or fallback)
    except Exception:
        pass
    return str(fallback or "")


def _safe_widget_text(widget: Any, fallback: str = "") -> str:
    try:
        cget = getattr(widget, "cget", None)
        if callable(cget):
            return str(cget("text") or fallback)
    except Exception:
        pass
    return str(fallback or "")


def build_campaign_resource_snapshot(
    row_key: str,
    row: Mapping[str, Any] | None = None,
    *,
    fallback_label: str = "",
    fallback_requirement: str = "optional",
    fallback_validation: str = "Brak",
    fallback_tone: str = "muted",
) -> CampaignResourceSnapshot:
    """Build a normalized snapshot from an E1 asset row dictionary."""

    key = str(row_key or "").strip()
    canonical_key = normalize_campaign_resource_key(key)
    row_data = row if isinstance(row, Mapping) else {}

    label = campaign_resource_label(
        canonical_key,
        str(row_data.get("base_label") or fallback_label or key),
    )
    source = str(row_data.get("source_full_text") or "").strip()
    if not source:
        source = _safe_var_get(row_data.get("source_var"), "Nie wskazano") or "Nie wskazano"
    validation = str(row_data.get("validation_full_text") or "").strip()
    if not validation:
        validation = _safe_widget_text(row_data.get("validation_lbl"), fallback_validation) or fallback_validation
    requirement = str(row_data.get("requirement") or fallback_requirement or "optional").strip().lower()
    tone = str(row_data.get("tone") or fallback_tone or "muted").strip().lower() or "muted"
    counter_text = str(row_data.get("counter_text") or "").strip()
    meta = row_data.get("meta")
    if not isinstance(meta, Mapping):
        meta = row_data.get("resource_meta")
    if not isinstance(meta, Mapping):
        meta = {}
    return CampaignResourceSnapshot(
        key=key,
        canonical_key=canonical_key,
        code=campaign_resource_short_label(canonical_key, canonical_key.upper()),
        label=label,
        requirement=requirement,
        source=source,
        validation=validation,
        tone=tone,
        counter_text=counter_text,
        description=campaign_resource_description(canonical_key),
        meta=dict(meta),
    )
