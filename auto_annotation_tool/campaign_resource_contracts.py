"""Authoritative readiness helpers for campaign resources.

Resource snapshots carry UI text, but gate decisions need a stricter contract.
This module keeps that contract in one place so the graph, resource modals and
transition reports do not drift into slightly different interpretations.
"""

from __future__ import annotations

from typing import Any, Mapping

from .campaign_resource_catalog import normalize_campaign_resource_key
from .campaign_resource_state import CampaignResourceSnapshot


def build_resource_contract_meta(
    contract_kind: str,
    *,
    contract_ready: bool = False,
    contract_enforced: bool = False,
    requires_rematch: bool = False,
    stale: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Return a normalized contract payload stored on resource snapshots."""

    meta = dict(extra)
    meta.update(
        contract_kind=str(contract_kind or "").strip(),
        contract_ready=bool(contract_ready),
        contract_enforced=bool(contract_enforced),
        requires_rematch=bool(requires_rematch),
    )
    if stale:
        meta["stale"] = True
    return meta


def resource_snapshot_meta(snapshot: CampaignResourceSnapshot | None) -> dict[str, Any]:
    if snapshot is None:
        return {}
    meta = getattr(snapshot, "meta", {}) or {}
    return dict(meta) if isinstance(meta, Mapping) else {}


def resource_contract_message(snapshot: CampaignResourceSnapshot | None, fallback: str = "") -> str:
    meta = resource_snapshot_meta(snapshot)
    for key in ("contract_message", "validation_message", "message"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    if snapshot is not None:
        value = str(getattr(snapshot, "validation", "") or "").strip()
        if value:
            return value
    return str(fallback or "").strip()


def resource_has_physical_source(snapshot: CampaignResourceSnapshot | None) -> bool:
    if snapshot is None:
        return False
    try:
        if bool(snapshot.has_source):
            return True
    except Exception:
        pass
    try:
        return int(snapshot.counter_value or 0) > 0
    except Exception:
        return False


def resource_snapshot_contract_managed(snapshot: CampaignResourceSnapshot | None) -> bool:
    """Return whether the new contract is allowed to drive gate decisions.

    The metadata can be useful for diagnostics and UI, but it must not become
    authoritative just because it exists. Only explicit enforcement or hard
    invalidation flags should bypass the legacy fallback path.
    """

    meta = resource_snapshot_meta(snapshot)
    return bool(
        meta.get("contract_enforced")
        or meta.get("requires_rematch")
        or meta.get("stale")
    )


def resource_contract_ready(
    snapshot: CampaignResourceSnapshot | None,
    *,
    required: bool = False,
) -> bool:
    """Return whether a snapshot can satisfy a gate/resource contract."""

    if snapshot is None:
        return False

    meta = resource_snapshot_meta(snapshot)
    if meta.get("requires_rematch") or meta.get("stale"):
        return False if required else resource_has_physical_source(snapshot)
    if bool(meta.get("contract_ready")):
        return True
    if meta.get("contract_enforced") and "contract_ready" in meta:
        return False if required else resource_has_physical_source(snapshot)

    try:
        tone = str(getattr(snapshot, "tone", "") or "").strip().lower()
    except Exception:
        tone = ""
    if tone == "success":
        return True
    if tone in {"error", "danger"}:
        return False

    canonical = normalize_campaign_resource_key(getattr(snapshot, "canonical_key", "") or getattr(snapshot, "key", ""))
    source_mode = str(meta.get("source_mode") or "").strip().lower()
    if required and canonical in {"plate_run", "char_run"}:
        if source_mode == "draft":
            return False
        return False
    if required and tone in {"warning", "muted", ""}:
        return False

    return resource_has_physical_source(snapshot)
