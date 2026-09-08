"""Readiness evaluation for campaign graph transitions.

The visual graph should not own transition rules. This module keeps the first
small slice of gate logic close to the declarative transition specs so the old
wizard UI can be retired step by step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .campaign_resource_catalog import normalize_campaign_resource_key
from .campaign_resource_contracts import resource_contract_ready, resource_snapshot_contract_managed
from .campaign_resource_state import CampaignResourceSnapshot
from .campaign_transition_graph import campaign_stage_status_key, campaign_stage_step
from .campaign_transition_specs import CampaignTransitionSpec


@dataclass(frozen=True)
class CampaignTransitionEvalContext:
    selected_path: str = ""
    explicit_selected_path: str = ""
    current_step: int = 1
    current_iteration: int = 1
    stage_status: Mapping[str, str] | None = None
    image_count: int = 0
    min_images: int = 10
    plate_material_count: int = 0
    min_plates: int = 10
    material_ready: bool = False
    image_potential: bool = False
    resource_snapshots: Mapping[str, CampaignResourceSnapshot] | None = None

    def resource_snapshot(self, key: str | None) -> CampaignResourceSnapshot | None:
        snapshots = self.resource_snapshots or {}
        raw_key = str(key or "").strip()
        canonical = normalize_campaign_resource_key(key)
        return snapshots.get(raw_key) or snapshots.get(canonical)

    def resource_counter(self, key: str | None) -> int:
        snapshot = self.resource_snapshot(key)
        if snapshot is None:
            return 0
        return int(snapshot.counter_value or 0)

    def resource_has_source(self, key: str | None) -> bool:
        snapshot = self.resource_snapshot(key)
        return bool(snapshot and snapshot.has_source)


def stage_number(stage_key: str | None) -> int:
    return campaign_stage_step(stage_key)


def is_transition_path_active(spec: CampaignTransitionSpec | None, selected_path: str | None) -> bool:
    if spec is None:
        return True
    path = str(selected_path or "").strip()
    if spec.path_key:
        return path == spec.path_key
    if spec.edge_key == "e1_to_e2":
        return path in {"plate_training", "char_from_images"}
    if spec.source == "E3":
        return path in {"char_from_images", "char_from_ready_plates"}
    return True


def is_transition_completed(spec: CampaignTransitionSpec | None, ctx: CampaignTransitionEvalContext) -> bool:
    if spec is None:
        return False
    if not is_transition_path_active(spec, ctx.selected_path):
        return False
    source_num = stage_number(spec.source)
    stage_status = ctx.stage_status or {}
    source_status = str(
        stage_status.get(campaign_stage_status_key(spec.source), "")
        or stage_status.get(spec.source, "")
        or ""
    ).strip().lower()
    return bool(source_status == "approved" or int(ctx.current_step or 1) > source_num)


def is_transition_ready(spec: CampaignTransitionSpec | None, ctx: CampaignTransitionEvalContext) -> bool:
    if spec is None:
        return False
    if not is_transition_path_active(spec, ctx.selected_path):
        return False
    if is_transition_completed(spec, ctx):
        return True
    if int(ctx.current_step or 1) != stage_number(spec.source):
        return False

    if spec.key == "e1_to_e2_prepare_plate_annotations":
        explicit_path = str(ctx.explicit_selected_path or "").strip()
        if explicit_path not in {"plate_training", "char_from_images"}:
            return False
        image_snapshot = ctx.resource_snapshot("images")
        image_count = max(int(ctx.image_count or 0), ctx.resource_counter("images"))
        images_ready = bool(image_count > 0 and resource_contract_ready(image_snapshot, required=True))
        if explicit_path == "plate_training":
            return images_ready

        plate_snapshot = ctx.resource_snapshot("plate_run") or ctx.resource_snapshot("approved_plates")
        plate_snapshot_count = 0
        plate_snapshot_ready = False
        if plate_snapshot is not None:
            plate_snapshot_count = int(plate_snapshot.counter_value or 0)
            plate_snapshot_ready = resource_contract_ready(plate_snapshot, required=True)
        plate_count = max(int(ctx.plate_material_count or 0), plate_snapshot_count)
        if plate_snapshot is not None and resource_snapshot_contract_managed(plate_snapshot):
            plate_ready = bool(plate_snapshot_ready)
        else:
            plate_ready = bool(ctx.material_ready or plate_snapshot_ready or plate_count >= int(ctx.min_plates or 10))
        return bool(images_ready or plate_ready)
    if spec.key == "e1_to_e3_char_from_ready_plates":
        plate_snapshot = ctx.resource_snapshot("plate_run")
        plate_snapshot_count = 0
        plate_snapshot_ready = False
        if plate_snapshot is not None:
            plate_snapshot_ready = resource_contract_ready(plate_snapshot, required=True)
            if plate_snapshot_ready:
                plate_snapshot_count = int(plate_snapshot.counter_value or 0)
        plate_count = max(int(ctx.plate_material_count or 0), plate_snapshot_count)
        if plate_snapshot is not None and resource_snapshot_contract_managed(plate_snapshot):
            return bool(plate_snapshot_ready)
        return bool(ctx.material_ready or plate_snapshot_ready or plate_count >= int(ctx.min_plates or 10))

    required_resources = [
        resource
        for resource in getattr(spec, "resources", ()) or ()
        if str(getattr(resource, "requirement", "") or "").strip().lower()
        in {"required", "route_required", "route_required_plate", "route_required_char"}
    ]
    if required_resources:
        for resource in required_resources:
            resource_key = str(getattr(resource, "key", "") or "").strip()
            if resource_key == "approved_plates":
                snapshot = (ctx.resource_snapshots or {}).get("approved_plates")
            else:
                snapshot = ctx.resource_snapshot(getattr(resource, "key", ""))
            if snapshot is None:
                return False
            if resource_key == "approved_plates":
                if not (
                    int(snapshot.counter_value or 0) >= int(ctx.min_plates or 10)
                    or str(snapshot.tone or "").strip().lower() == "success"
                ):
                    return False
                continue
            if not resource_contract_ready(snapshot, required=True):
                return False
        return True

    return False
