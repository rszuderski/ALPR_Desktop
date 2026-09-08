#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reader window for lightweight campaign project history."""

from __future__ import annotations

import json
import re
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Any

from ..campaign_manager import CAMPAIGN
from ..campaign_history_resources import HISTORY_RESOURCE_SCHEMA, HistoryResourceReader
from ..campaign_iteration_paths import STEP1_ITERATION_PATHS, iteration_path_target
from ..campaign_transition_graph import campaign_stage_step
from ..campaign_transition_specs import EDGE_KEY_ALIASES, TRANSITION_SPECS
from ..utils import safe_load_yaml
from .z2_shared_ui import campaign_visible_gate_id
from .web_slim_scrollbar import blend_hex_colors


_TRANSITION_SPEC_BY_BADGE = {
    str(getattr(spec, "badge_id", "") or "").strip().upper(): spec
    for spec in TRANSITION_SPECS
    if str(getattr(spec, "badge_id", "") or "").strip()
}
if "T01" in _TRANSITION_SPEC_BY_BADGE:
    _TRANSITION_SPEC_BY_BADGE.setdefault("T02", _TRANSITION_SPEC_BY_BADGE["T01"])
_TRANSITION_SPEC_BY_KEY = {}
for _spec in TRANSITION_SPECS:
    for _key in (getattr(_spec, "key", ""), getattr(_spec, "edge_key", "")):
        _normalized_key = str(_key or "").strip()
        if _normalized_key and _normalized_key not in _TRANSITION_SPEC_BY_KEY:
            _TRANSITION_SPEC_BY_KEY[_normalized_key] = _spec
for _alias_key, _canonical_keys in EDGE_KEY_ALIASES.items():
    _alias_text = str(_alias_key or "").strip()
    if not _alias_text or _alias_text in _TRANSITION_SPEC_BY_KEY:
        continue
    for _canonical_key in tuple(_canonical_keys or ()):
        _alias_spec = _TRANSITION_SPEC_BY_KEY.get(str(_canonical_key or "").strip())
        if _alias_spec is not None:
            _TRANSITION_SPEC_BY_KEY[_alias_text] = _alias_spec
            break

_PATH_ACTIONS = {
    "set_iteration_path",
    "approve_step1",
    "approve_step1_ready_plates",
    "approve_step2",
    "approve_step3",
    "approve_step4",
    "approve_step4_without_training",
    "start_next_iteration",
}


def _short_text(value: Any, limit: int = 96) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _format_history_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    return text.replace("T", " ")[:19]


def _event_title(entry: dict) -> str:
    title = str(entry.get("title") or "").strip()
    if title:
        return title
    action = str(entry.get("action") or "").strip()
    if action:
        return action
    return str(entry.get("event_type") or "Zdarzenie").strip()


def _event_gate(entry: dict) -> str:
    spec = _event_transition_spec(entry)
    if spec is not None:
        badge = str(getattr(spec, "badge_id", "") or "").strip().upper()
        if badge:
            return campaign_visible_gate_id(badge) or badge
    gate = str(entry.get("gate_id") or "").strip().upper()
    if gate.startswith("T"):
        return campaign_visible_gate_id(gate) or gate
    return str(entry.get("transition_id") or "").strip() or "-"


def _event_status(entry: dict) -> str:
    status = str(entry.get("status") or "ok").strip().lower()
    if status == "ok":
        return "OK"
    if status == "error":
        return "Błąd"
    return status or "-"


def _event_stage_label(entry: dict) -> str:
    try:
        step = int(entry.get("step", 0) or 0)
    except Exception:
        step = 0
    if step == 4:
        details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
        target = str(details.get("target") or "").strip().lower()
        if target not in {"plate", "char"}:
            target = iteration_path_target(str(details.get("path") or "").strip())
        if target == "plate":
            return "E4T"
        if target == "char":
            return "E4Z"
    return f"E{step}" if step > 0 else "-"


def _metric_percent(value: Any) -> str:
    try:
        number = float(value or 0.0)
    except Exception:
        return "-"
    return f"{number * 100:.1f}%"


def _short_path_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    try:
        return Path(text).name or text
    except Exception:
        return text


def _training_run_kind(run: dict) -> str:
    blob = " ".join(
        str(run.get(key) or "")
        for key in ("name", "dataset_path", "base_model", "best_weights")
    ).lower()
    if "plate" in blob or "pose" in blob:
        return "model tablic"
    if "char" in blob or "detect" in blob:
        return "model znaków"
    return "model"


def _model_identity_from_sidecar(model_path: Any) -> str:
    path_text = str(model_path or "").strip()
    if not path_text:
        return ""
    try:
        model_path_obj = Path(path_text)
    except Exception:
        return ""
    candidates = [
        model_path_obj.with_suffix(f"{model_path_obj.suffix}.metadata.json"),
        model_path_obj.with_suffix(".metadata.json"),
        model_path_obj.with_name("model_metadata.json"),
    ]
    for meta_path in candidates:
        try:
            if not meta_path.exists():
                continue
            payload = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            info = dict((payload.get("model") or {}).get("info") or {})
            label = str(
                info.get("architecture_label")
                or info.get("source_architecture_label")
                or info.get("yolo_variant")
                or ""
            ).strip()
            if label:
                return label
        except Exception:
            continue
    return ""


def _load_project_training_runs(project_name: str, *, limit: int | None = 3) -> list[dict]:
    try:
        project_root = Path(CAMPAIGN.get_project_root_dir(project_name))
    except Exception:
        return []
    history_path = project_root / "5_training_runs" / "training_history.json"
    if not history_path.exists():
        return []
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    runs = []
    for run_id, raw_run in dict(payload.get("runs") or {}).items():
        if not isinstance(raw_run, dict):
            continue
        run = dict(raw_run)
        run.setdefault("id", str(run_id))
        runs.append(run)
    runs.sort(
        key=lambda item: str(item.get("finished_at") or item.get("started_at") or item.get("created_at") or ""),
        reverse=True,
    )
    if limit is None:
        return runs
    try:
        max_items = max(1, int(limit or 3))
    except Exception:
        max_items = 3
    return runs[:max_items]


def _event_badge_id(entry: dict) -> str:
    spec = _event_transition_spec(entry)
    if spec is not None:
        badge = str(getattr(spec, "badge_id", "") or "").strip().upper()
        return campaign_visible_gate_id(badge) or badge
    gate = str(entry.get("gate_id") or "").strip().upper()
    if gate.startswith("T"):
        return campaign_visible_gate_id(gate) or gate
    return gate


def _event_stage_token(entry: dict) -> str:
    spec = _event_transition_spec(entry)
    if spec is not None:
        source = str(getattr(spec, "source", "") or "").strip().upper()
        if source:
            return source
    try:
        step = int(entry.get("step", 0) or 0)
    except Exception:
        step = 0
    return f"E{step}" if step > 0 else ""


def _event_iteration(entry: dict) -> int:
    try:
        return int(entry.get("iteration", 0) or 0)
    except Exception:
        return 0


def _event_path_key(entry: dict) -> str:
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    path = str(details.get("path") or "").strip()
    if path:
        return path
    spec = _event_transition_spec(entry)
    if spec is not None:
        return str(getattr(spec, "path_key", "") or "").strip()
    return ""


def _iteration_path_map(entries: list[dict]) -> dict[int, str]:
    paths: dict[int, str] = {}
    priorities: dict[int, int] = {}
    for raw_entry in entries:
        entry = dict(raw_entry or {})
        if str(entry.get("status") or "ok").strip().lower() == "error":
            continue
        action = str(entry.get("action") or "").strip()
        iteration = _event_iteration(entry)
        path = _event_path_key(entry)
        explicit = bool((entry.get("details") or {}).get("path"))
        priority = (100 if explicit else 0) + (30 if action == "set_iteration_path" else 20 if action.startswith("approve_step1") else 10)
        if iteration > 0 and path in STEP1_ITERATION_PATHS and priority >= priorities.get(iteration, 0):
            paths[iteration] = path
            priorities[iteration] = priority
    return paths


def _iteration_path_label(path: str | None) -> str:
    normalized = str(path or "").strip()
    if not normalized:
        return ""
    path_def = STEP1_ITERATION_PATHS.get(normalized) or {}
    target = str(path_def.get("target") or iteration_path_target(normalized) or "").strip().lower()
    if target == "plate":
        return "tor tablic"
    if normalized == "char_from_ready_plates":
        return "tor znaków | istniejące tablice"
    if normalized == "char_from_images":
        return "tor znaków | obrazy"
    if target == "char":
        return "tor znaków"
    title = str(path_def.get("short_title") or path_def.get("title") or "").strip()
    return title


def _entry_text_blob(entry: dict) -> str:
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    return " ".join(
        str(value or "")
        for value in (
            entry.get("title"),
            entry.get("action"),
            entry.get("transition_id"),
            entry.get("gate_id"),
            details.get("message"),
            details.get("path"),
            details.get("target"),
        )
    ).casefold()


def _target_from_history_text(entry: dict) -> str:
    blob = _entry_text_blob(entry)
    if "plate_training" in blob:
        return "plate"
    if "char_from" in blob:
        return "char"
    if "model znak" in blob:
        return "char"
    if "model tablic" in blob:
        return "plate"
    if "znak" in blob and "model" in blob:
        return "char"
    if "tablic" in blob and "model" in blob:
        return "plate"
    return ""


def _event_iteration_target(entry: dict) -> str:
    path_target = iteration_path_target(_event_path_key(entry))
    if path_target in {"plate", "char"}:
        return path_target
    spec = _event_transition_spec(entry)
    if spec is not None:
        path_target = iteration_path_target(str(getattr(spec, "path_key", "") or ""))
        if path_target in {"plate", "char"}:
            return path_target
    return _target_from_history_text(entry)


def _iteration_target_map(entries: list[dict]) -> dict[int, str]:
    targets: dict[int, str] = {}
    priority_by_iteration: dict[int, int] = {}
    priorities = {
        "set_iteration_path": 30,
        "approve_step1": 20,
        "approve_step1_ready_plates": 20,
    }
    for raw_entry in entries:
        entry = dict(raw_entry or {})
        if str(entry.get("status") or "ok").strip().lower() == "error":
            continue
        iteration = _event_iteration(entry)
        if iteration <= 0:
            continue
        target = _event_iteration_target(entry)
        if target not in {"plate", "char"}:
            continue
        action = str(entry.get("action") or "").strip()
        priority = priorities.get(action, 10)
        if iteration_path_target(_event_path_key(entry)):
            priority += 100
        if priority >= priority_by_iteration.get(iteration, 0):
            targets[iteration] = target
            priority_by_iteration[iteration] = priority
    return targets


def _infer_training_run_iteration_from_history(
    run: dict,
    *,
    target: str,
    history_entries: list[dict],
    target_by_iteration: dict[int, str],
) -> int:
    normalized_target = str(target or "").strip().lower()
    if normalized_target not in {"plate", "char"}:
        return 0

    run_time = str(
        run.get("created_at")
        or run.get("started_at")
        or run.get("finished_at")
        or ""
    ).strip()
    candidates: list[tuple[str, int]] = []
    for raw_entry in history_entries:
        entry = dict(raw_entry or {})
        if str(entry.get("status") or "ok").strip().lower() == "error":
            continue
        if str(entry.get("action") or "").strip() != "approve_step4":
            continue
        iteration = _event_iteration(entry)
        if iteration <= 0:
            continue
        entry_target = target_by_iteration.get(iteration) or _event_iteration_target(entry)
        if entry_target != normalized_target:
            continue
        candidates.append((str(entry.get("created_at") or ""), iteration))

    if run_time:
        future_candidates = [
            (created_at, iteration)
            for created_at, iteration in candidates
            if created_at and created_at >= run_time
        ]
        if future_candidates:
            return min(future_candidates, key=lambda item: item[0])[1]
        prior_context = []
        for raw_entry in history_entries:
            entry = dict(raw_entry or {})
            if str(entry.get("status") or "ok").strip().lower() == "error":
                continue
            created_at = str(entry.get("created_at") or "").strip()
            if not created_at or created_at > run_time:
                continue
            iteration = _event_iteration(entry)
            if iteration <= 0:
                continue
            entry_target = target_by_iteration.get(iteration) or _event_iteration_target(entry)
            if entry_target == normalized_target:
                prior_context.append((created_at, iteration))
        if prior_context:
            return max(prior_context, key=lambda item: item[0])[1]

    if len(candidates) == 1:
        return candidates[0][1]

    matching_iterations = [
        iteration
        for iteration, iteration_target in target_by_iteration.items()
        if iteration_target == normalized_target
    ]
    if len(matching_iterations) == 1:
        return matching_iterations[0]
    return 0


def _training_run_target(run: dict) -> str:
    for key in ("training_target", "campaign_target", "target", "model_type"):
        value = str(run.get(key) or "").strip().lower()
        if value in {"plate", "plates", "pose"}:
            return "plate"
        if value in {"char", "chars", "character", "characters"}:
            return "char"
    snapshot = run.get("training_dataset_snapshot") or {}
    if snapshot.get("target") in {"plate", "char"}:
        return snapshot["target"]
    dataset_path = str(run.get("dataset_path") or "").strip()
    if dataset_path:
        path = Path(dataset_path)
        yaml_path = path if path.name.lower() == "data.yaml" else path / "data.yaml"
        try:
            config = safe_load_yaml(yaml_path) if yaml_path.is_file() else {}
            if config.get("kpt_shape"):
                return "plate"
            names = config.get("names") or []
            if isinstance(names, dict):
                names = list(names.values())
            if names and all(len(str(name)) == 1 and str(name).isalnum() for name in names):
                return "char"
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    blob = " ".join(
        str(run.get(key) or "")
        for key in ("name", "dataset_path", "base_model", "best_weights", "id")
    ).lower()
    if "plate" in blob or "pose" in blob:
        return "plate"
    if "char" in blob or "detect" in blob:
        return "char"
    return ""


def _training_run_iteration(run: dict) -> int:
    for key in ("trained_iteration", "iteration", "campaign_iteration", "step4_iteration", "iteration_num"):
        try:
            value = int(run.get(key, 0) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def _compact_artifact_text(entry: dict, fallback: str = "") -> str:
    pieces: list[str] = []
    for container_name in ("artifacts", "resources", "metrics", "details"):
        container = entry.get(container_name)
        if not isinstance(container, dict):
            continue
        for key, value in container.items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (str, int, float, bool)):
                pieces.append(f"{key}: {value}")
            elif isinstance(value, (list, tuple, set)):
                pieces.append(f"{key}: {len(value)}")
            elif isinstance(value, dict):
                pieces.append(f"{key}: {len(value)} pól")
            if len(pieces) >= 3:
                break
        if len(pieces) >= 3:
            break
    text = " | ".join(pieces).strip()
    return text or fallback or _event_title(entry)


def _first_int_value(entry: dict, keys: tuple[str, ...]) -> int | None:
    containers = [
        entry,
        entry.get("artifacts") if isinstance(entry.get("artifacts"), dict) else {},
        entry.get("resources") if isinstance(entry.get("resources"), dict) else {},
        entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {},
        entry.get("details") if isinstance(entry.get("details"), dict) else {},
    ]
    for key in keys:
        for container in containers:
            if not isinstance(container, dict):
                continue
            value = container.get(key)
            if value is None or value == "" or isinstance(value, bool):
                continue
            try:
                count = int(value)
                if count >= 0:
                    return count
            except (TypeError, ValueError, OverflowError):
                continue
    return None


def _plural_pl(number: int, singular: str, paucal: str, plural: str) -> str:
    try:
        value = abs(int(number))
    except Exception:
        value = 0
    if value == 1:
        return singular
    if value % 10 in {2, 3, 4} and value % 100 not in {12, 13, 14}:
        return paucal
    return plural


def _history_resource_snapshot(entry: dict, project_name: str, reader=None) -> dict:
    recorded = (entry.get("artifacts") or {}).get("resource_snapshot")
    if (
        isinstance(recorded, dict) and recorded.get("schema") == HISTORY_RESOURCE_SCHEMA
        and recorded.get("project") == project_name
        and recorded.get("iteration") == _event_iteration(entry)
    ):
        return recorded
    try:
        reader = reader or HistoryResourceReader(CAMPAIGN.get_project_root_dir(project_name), project_name)
        return reader.for_event(entry)
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _plate_history_counts(entry: dict, project_name: str, reader=None) -> dict:
    counts = _history_resource_snapshot(entry, project_name, reader).get("AT")
    if isinstance(counts, dict) and counts:
        return counts
    # Old graph events stored project totals, not the iteration's increment.
    images = _first_int_value(entry, ("approved_images", "approved_images_with_plates", "ok_images"))
    plates = _first_int_value(entry, ("approved_plates", "approved_total_plates", "ok_plates"))
    if (images or 0) > 0 or (plates or 0) > 0:
        return {"images": images, "plates": plates, "source": "legacy_project_totals"}
    return {}


def _format_plate_annotation_artifact(entry: dict, project_name: str = "", *, reader=None) -> str:
    counts = _plate_history_counts(entry, project_name, reader)
    if not counts:
        return "Brak wiarygodnych liczników AT dla tego wpisu."
    images, plates = counts.get("images"), counts.get("plates")
    image_word = _plural_pl(images, "zdjęcie", "zdjęcia", "zdjęć")
    frame_word = _plural_pl(plates, "ramka tablicy", "ramki tablic", "ramek tablic")
    return f"Łącznie: {images if images is not None else '?'} {image_word} [OK] / {plates if plates is not None else '?'} {frame_word}"


def _format_plate_annotation_increment(entry: dict, project_name: str = "", *, reader=None) -> str:
    counts = _plate_history_counts(entry, project_name, reader)
    images, plates = counts.get("iteration_images"), counts.get("iteration_plates")
    if images == 0 and plates == 0:
        return "Bez nowych AT w tej iteracji"
    if images is None and plates is None:
        return "Przyrost nie został zapisany"
    image_delta = f"{images:+d}" if images is not None else "?"
    plate_delta = f"{plates:+d}" if plates is not None else "?"
    return f"{image_delta} zdjęć [OK] / {plate_delta} ramek tablic"


def _build_project_product_rows(project_name: str, entries: list[dict], *, resource_reader=None) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, int, str]] = set()
    path_by_iteration = _iteration_path_map(entries)
    target_by_iteration = _iteration_target_map(entries)
    step4_gate_id = "T06"
    try:
        resource_reader = resource_reader or HistoryResourceReader(CAMPAIGN.get_project_root_dir(project_name), project_name)
    except (OSError, ValueError, TypeError):
        resource_reader = None

    def _add_row(
        *,
        code: str,
        name: str,
        iteration: int,
        stage: str,
        gate: str,
        status: str,
        increment: str,
        artifact: str,
        matches: set[tuple[str, str]],
        created_at: str = "",
    ) -> None:
        normalized_gate = str(gate or "").strip().upper()
        normalized_stage = str(stage or "").strip().upper()
        key = (
            str(code or "").strip().upper(),
            int(iteration or 0),
            normalized_gate or normalized_stage,
            str(created_at or "").strip(),
            str(artifact or "").strip(),
        )
        if key in seen:
            return
        seen.add(key)
        source_parts = []
        if iteration > 0:
            source_parts.append(f"IT{iteration}")
        if normalized_stage:
            source_parts.append(normalized_stage)
        if normalized_gate:
            source_parts.append(normalized_gate)
        rows.append(
            {
                "code": str(code or "").strip().upper(),
                "name": str(name or "").strip(),
                "iteration": int(iteration or 0),
                "source": " / ".join(source_parts) or "-",
                "status": str(status or "").strip() or "-",
                "increment": str(increment or "").strip() or "-",
                "artifact": str(artifact or "").strip() or "-",
                "matches": set(matches or set()),
                "created_at": str(created_at or "").strip(),
            }
        )

    for raw_entry in entries:
        entry = dict(raw_entry or {})
        if str(entry.get("status") or "ok").strip().lower() == "error":
            continue
        action = str(entry.get("action") or "").strip()
        iteration = _event_iteration(entry)
        stage = _event_stage_token(entry)
        gate = _event_badge_id(entry)
        matches = {(stage, str(iteration)) for stage in [stage] if stage and iteration > 0}
        if gate and iteration > 0:
            matches.add((gate, str(iteration)))
        if action == "approve_step2":
            _add_row(
                code="AT",
                name="Anotacje tablic",
                iteration=iteration,
                stage=stage or "E2",
                gate=gate,
                status="gotowe",
                increment=_format_plate_annotation_increment(entry, project_name, reader=resource_reader),
                artifact=_format_plate_annotation_artifact(entry, project_name, reader=resource_reader),
                matches=matches,
                created_at=str(entry.get("created_at") or ""),
            )
        elif action == "approve_step3":
            az = _history_resource_snapshot(entry, project_name, resource_reader).get("AZ") or {}
            source_iteration = int(az.get("source_iteration") or 0)
            az_artifact = (
                f"{az['plates']} tablic / {az['characters']} znaków | źródło IT{source_iteration}"
                if az else "Zatwierdzono źródło AZ; brak zapisanych liczników."
            )
            _add_row(
                code="AZ",
                name="Anotacje znaków",
                iteration=iteration,
                stage=stage or "E3",
                gate=gate,
                status="gotowe",
                increment=(
                    f"Zatwierdzono AZ z IT{source_iteration}"
                    if source_iteration == iteration else f"Bez nowych AZ; źródło IT{source_iteration}"
                    if source_iteration else "Zatwierdzono źródło AZ"
                ),
                artifact=az_artifact,
                matches=matches,
                created_at=str(entry.get("created_at") or ""),
            )

    for run in _load_project_training_runs(project_name, limit=None):
        target = _training_run_target(run)
        if target not in {"plate", "char"}:
            continue
        code = "MT" if target == "plate" else "MZ"
        stage_key = "E4T" if target == "plate" else "E4Z"
        iteration = _training_run_iteration(run)
        if iteration <= 0:
            iteration = _infer_training_run_iteration_from_history(
                run,
                target=target,
                history_entries=entries,
                target_by_iteration=target_by_iteration,
            )
        weights_path = str(run.get("best_weights") or "").strip()
        status = str(run.get("status") or "").strip().lower()
        if status != "completed" and (not weights_path or not Path(weights_path).is_file()):
            continue
        weights_name = _short_path_name(weights_path)
        metrics = (
            f"mAP50 {_metric_percent(run.get('best_map50'))}, "
            f"mAP50-95 {_metric_percent(run.get('best_map50_95'))}"
        )
        status_label = "gotowy" if status == "completed" else f"checkpoint ({status or 'nieukończony'})"
        if status == "completed" and (not weights_path or not Path(weights_path).is_file()):
            status_label = "plik modelu niedostępny"
        _add_row(
            code=code,
            name="Model tablic" if target == "plate" else "Model znaków",
            iteration=iteration,
            stage=stage_key,
            gate=step4_gate_id,
            status=status_label,
            increment=f"utworzono model {code}" if status == "completed" else "checkpoint nieukończonego treningu",
            artifact=f"{weights_name} | {metrics}",
            matches={
                (stage_key, str(iteration)),
                ("E4", str(iteration)),
                (step4_gate_id, str(iteration)),
            } if iteration > 0 else {(stage_key, ""), ("E4", ""), (step4_gate_id, "")},
            created_at=str(run.get("finished_at") or run.get("created_at") or ""),
        )

    has_model_for_iteration = {
        (row["code"], int(row.get("iteration", 0) or 0))
        for row in rows
        if row.get("code") in {"MT", "MZ"}
    }
    for raw_entry in entries:
        entry = dict(raw_entry or {})
        if str(entry.get("status") or "ok").strip().lower() == "error":
            continue
        if str(entry.get("action") or "").strip() != "approve_step4":
            continue
        iteration = _event_iteration(entry)
        path = path_by_iteration.get(iteration, "")
        target = "plate" if path == "plate_training" else ("char" if path else "")
        if target not in {"plate", "char"}:
            continue
        code = "MT" if target == "plate" else "MZ"
        stage_key = "E4T" if target == "plate" else "E4Z"
        if (code, iteration) in has_model_for_iteration:
            continue
        _add_row(
            code=code,
            name="Model tablic" if target == "plate" else "Model znaków",
            iteration=iteration,
            stage=stage_key,
            gate=step4_gate_id,
            status=f"zamknięto {stage_key}",
            increment="bez nowego modelu",
            artifact=_compact_artifact_text(
                entry,
                f"Zamknięto trening {'modelu tablic' if target == 'plate' else 'modelu znaków'}; "
                "szczegóły modelu nie są zapisane w historii.",
            ),
            matches={
                (stage_key, str(iteration)),
                ("E4", str(iteration)),
                (step4_gate_id, str(iteration)),
            } if iteration > 0 else {(stage_key, ""), ("E4", ""), (step4_gate_id, "")},
            created_at=str(entry.get("created_at") or ""),
        )

    rows.sort(
        key=lambda row: (
            int(row.get("iteration", 0) or 0),
            str(row.get("code") or ""),
            str(row.get("created_at") or ""),
        )
    )
    return rows


def _event_transition_spec(entry: dict):
    transition = str(entry.get("transition_id") or "").strip()
    if transition in _TRANSITION_SPEC_BY_KEY:
        return _TRANSITION_SPEC_BY_KEY[transition]
    # The edge key is the stable source of truth for historical entries. Gate
    # ids were renumbered during graph stabilization, so older rows can carry a
    # stale badge while still having the correct transition_id.
    gate = str(entry.get("gate_id") or "").strip().upper()
    if gate in _TRANSITION_SPEC_BY_BADGE:
        return _TRANSITION_SPEC_BY_BADGE[gate]
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    path = str(details.get("path") or "").strip()
    if path in _TRANSITION_SPEC_BY_KEY:
        return _TRANSITION_SPEC_BY_KEY[path]
    return None


def _path_token_kind(token: str) -> str:
    normalized = str(token or "").strip().upper()
    if normalized.startswith("E") and campaign_stage_step(normalized) > 0:
        return "stage"
    if normalized.startswith("T") and normalized[1:].isdigit():
        return "gate"
    if normalized.startswith("IT") and normalized[2:].isdigit():
        return "iteration"
    if normalized == "...":
        return "ellipsis"
    return "plain"


def _append_path_token(tokens: list[tuple[str, str]], token: str) -> None:
    normalized = str(token or "").strip().upper()
    if not normalized:
        return
    kind = _path_token_kind(normalized)
    if tokens and tokens[-1][1] == normalized:
        return
    tokens.append((kind, normalized))


def _append_transition_tokens(
    tokens: list[tuple[str, str]],
    spec,
    *,
    include_target: bool,
    current_stage: str | None = None,
) -> str:
    source = str(getattr(spec, "source", "") or "").strip().upper()
    raw_badge = str(getattr(spec, "badge_id", "") or "").strip().upper()
    badge = campaign_visible_gate_id(raw_badge) or raw_badge
    target = str(getattr(spec, "target", "") or "").strip().upper()
    current_stage_normalized = str(current_stage or "").strip().upper()
    if source == "E4" and current_stage_normalized.startswith("E4"):
        source = current_stage_normalized
    if not tokens:
        _append_path_token(tokens, source)
    elif badge and tokens[-1][1] == badge:
        pass
    elif len(tokens) >= 2 and tokens[-2][1] == source and tokens[-1][1] == badge:
        pass
    elif tokens[-1][1] != source:
        _append_path_token(tokens, source)
    _append_path_token(tokens, badge)
    if include_target:
        _append_path_token(tokens, target)
        return target or source
    return source


def _transition_accepts_current_stage(spec, current_stage: str | None) -> bool:
    source = str(getattr(spec, "source", "") or "").strip().upper()
    if not source:
        return False
    if not current_stage:
        return True
    normalized_current = str(current_stage or "").strip().upper()
    if normalized_current == source:
        return True
    return bool(campaign_stage_step(normalized_current) == campaign_stage_step(source))


def _history_bridge_specs(current_stage: str | None, spec, path_key: str | None = None) -> list:
    """Recover obvious missing graph steps from old/incomplete history rows."""
    normalized_current = str(current_stage or "").strip().upper()
    source = str(getattr(spec, "source", "") or "").strip().upper()
    target = str(getattr(spec, "target", "") or "").strip().upper()
    path_target = iteration_path_target(str(path_key or "").strip())
    if normalized_current == "E2" and source == "E3":
        bridge = _TRANSITION_SPEC_BY_KEY.get("e2_to_e3")
        return [bridge] if bridge is not None else []
    if normalized_current == "E2" and source == "E4":
        if path_target == "plate":
            bridge = _TRANSITION_SPEC_BY_KEY.get("e2_to_e4")
            return [bridge] if bridge is not None else []
        if path_target == "char":
            first = _TRANSITION_SPEC_BY_KEY.get("e2_to_e3")
            second = _TRANSITION_SPEC_BY_KEY.get("e3_to_e4")
            return [item for item in (first, second) if item is not None]
    if normalized_current == "E3" and source == "E4":
        bridge = _TRANSITION_SPEC_BY_KEY.get("e3_to_e4")
        if bridge is not None and (target == "E1" or path_target == "char"):
            return [bridge]
    return []


def _build_history_path_tokens(entries: list[dict], *, max_tokens: int = 34) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    current_iteration = None
    current_stage: str | None = None
    path_by_iteration = _iteration_path_map(entries)
    for raw_entry in entries:
        entry = dict(raw_entry or {})
        if str(entry.get("status") or "ok").strip().lower() == "error":
            continue
        action = str(entry.get("action") or "").strip()
        if action not in _PATH_ACTIONS:
            continue
        try:
            iteration = int(entry.get("iteration", 0) or 0)
        except Exception:
            iteration = 0
        if iteration > 0 and iteration != current_iteration:
            current_iteration = iteration
            if tokens:
                iteration_token = f"IT{iteration}"
                already_at_iteration_start = bool(
                    len(tokens) >= 2
                    and tokens[-2][1] == iteration_token
                    and tokens[-1][1] == "E1"
                )
                if not already_at_iteration_start:
                    _append_path_token(tokens, iteration_token)
                    current_stage = None
                else:
                    current_stage = "E1"
        spec = _event_transition_spec(entry)
        if spec is not None:
            path_key = _event_path_key(entry) or path_by_iteration.get(iteration, "")
            if not _transition_accepts_current_stage(spec, current_stage):
                bridge_specs = _history_bridge_specs(current_stage, spec, path_key)
                for bridge_spec in bridge_specs:
                    current_stage = _append_transition_tokens(
                        tokens,
                        bridge_spec,
                        include_target=True,
                        current_stage=current_stage,
                    )
                if not _transition_accepts_current_stage(spec, current_stage):
                    continue
            approve_action = str(getattr(spec, "approve_action", "") or "").strip()
            include_target = bool(
                action
                and (
                    action == approve_action
                    or action.startswith("approve_step")
                    or action == "start_next_iteration"
                )
            )
            target = str(getattr(spec, "target", "") or "").strip().upper()
            if include_target and target == "E1" and iteration > 0:
                _append_transition_tokens(tokens, spec, include_target=False, current_stage=current_stage)
                _append_path_token(tokens, f"IT{iteration + 1}")
                _append_path_token(tokens, "E1")
                current_stage = "E1"
            else:
                current_stage = _append_transition_tokens(
                    tokens,
                    spec,
                    include_target=include_target,
                    current_stage=current_stage,
                )
            continue

    if len(tokens) > max_tokens:
        head_count = max(8, max_tokens - 8)
        tokens = tokens[:head_count] + [("ellipsis", "...")] + tokens[-7:]
    return tokens


def _append_path_segment(
    segments: list[tuple[str, str, str]],
    token: str,
    iteration: int,
) -> None:
    normalized = str(token or "").strip().upper()
    if not normalized:
        return
    kind = _path_token_kind(normalized)
    if kind == "iteration":
        return
    iteration_label = str(iteration) if iteration > 0 and kind in {"stage", "gate"} else ""
    if segments and segments[-1][1] == normalized and segments[-1][2] == iteration_label:
        return
    segments.append((kind, normalized, iteration_label))


def _append_transition_segments(
    segments: list[tuple[str, str, str]],
    spec,
    *,
    include_target: bool,
    iteration: int,
    current_stage: str | None = None,
) -> str:
    source = str(getattr(spec, "source", "") or "").strip().upper()
    raw_badge = str(getattr(spec, "badge_id", "") or "").strip().upper()
    badge = campaign_visible_gate_id(raw_badge) or raw_badge
    target = str(getattr(spec, "target", "") or "").strip().upper()
    current_stage_normalized = str(current_stage or "").strip().upper()
    if source == "E4" and current_stage_normalized.startswith("E4"):
        source = current_stage_normalized
    if not segments:
        _append_path_segment(segments, source, iteration)
    elif badge and segments[-1][1] == badge:
        pass
    elif len(segments) >= 2 and segments[-2][1] == source and segments[-1][1] == badge:
        pass
    elif segments[-1][1] != source:
        _append_path_segment(segments, source, iteration)
    _append_path_segment(segments, badge, iteration)
    if include_target:
        _append_path_segment(segments, target, iteration)
        return target or source
    return source


def _normalize_iteration_start_segments(
    segments: list[tuple[str, str, str]]
) -> list[tuple[str, str, str]]:
    return list(segments or [])


def _build_history_path_segments(entries: list[dict], *, max_tokens: int = 34) -> list[tuple[str, str, str]]:
    segments: list[tuple[str, str, str]] = []
    current_iteration = None
    current_stage: str | None = None
    path_by_iteration = _iteration_path_map(entries)
    for raw_entry in entries:
        entry = dict(raw_entry or {})
        if str(entry.get("status") or "ok").strip().lower() == "error":
            continue
        action = str(entry.get("action") or "").strip()
        if action not in _PATH_ACTIONS:
            continue
        try:
            iteration = int(entry.get("iteration", 0) or 0)
        except Exception:
            iteration = 0
        if iteration > 0 and iteration != current_iteration:
            current_iteration = iteration
            current_stage = None
        spec = _event_transition_spec(entry)
        if spec is not None:
            path_key = _event_path_key(entry) or path_by_iteration.get(iteration, "")
            if not _transition_accepts_current_stage(spec, current_stage):
                bridge_specs = _history_bridge_specs(current_stage, spec, path_key)
                for bridge_spec in bridge_specs:
                    current_stage = _append_transition_segments(
                        segments,
                        bridge_spec,
                        include_target=True,
                        iteration=iteration,
                        current_stage=current_stage,
                    )
                if not _transition_accepts_current_stage(spec, current_stage):
                    continue
            approve_action = str(getattr(spec, "approve_action", "") or "").strip()
            include_target = bool(
                action
                and (
                    action == approve_action
                    or action.startswith("approve_step")
                    or action == "start_next_iteration"
                )
            )
            target = str(getattr(spec, "target", "") or "").strip().upper()
            if include_target and target == "E1" and iteration > 0:
                _append_transition_segments(
                    segments,
                    spec,
                    include_target=False,
                    iteration=iteration,
                    current_stage=current_stage,
                )
                _append_path_segment(segments, "E1", iteration + 1)
                current_stage = "E1"
                continue
            current_stage = _append_transition_segments(
                segments,
                spec,
                include_target=include_target,
                iteration=iteration,
                current_stage=current_stage,
            )
            continue

    segments = _normalize_iteration_start_segments(segments)
    if len(segments) > max_tokens:
        head_count = max(8, max_tokens - 8)
        segments = segments[:head_count] + [("ellipsis", "...", "")] + segments[-7:]
    return segments


def _format_history_path(entries: list[dict], *, max_tokens: int = 34) -> str:
    tokens = _build_history_path_tokens(entries, max_tokens=max_tokens)
    if not tokens:
        return "Ścieżka: brak rozpoznanych przejść grafu."
    return "Ścieżka: " + " → ".join(token for _kind, token in tokens)


def _format_event_details(entry: dict, *, reader=None) -> str:
    evidence = ""
    if str(entry.get("action") or "").startswith("approve_step") and not (entry.get("artifacts") or {}).get("resource_snapshot"):
        snapshot = _history_resource_snapshot(entry, str(entry.get("project") or ""), reader)
        if snapshot:
            evidence = (
                "Liczniki odtworzone z zapisów właściwej iteracji:\n"
                + json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n\nOryginalny wpis historii (bez zmian):\n"
            )
    header = [
        f"Czas: {entry.get('created_at', '-')}",
        f"Projekt: {entry.get('project', '-')}",
        f"Iteracja: {entry.get('iteration', '-')}",
        f"Krok: {_event_stage_label(entry)}",
        f"Bramka/przejście: {_event_gate(entry)}",
        f"Akcja: {entry.get('action', '-') or '-'}",
        f"Status: {_event_status(entry)}",
        "",
        "Pełny wpis:",
    ]
    return "\n".join(header) + "\n" + evidence + json.dumps(entry, ensure_ascii=False, indent=2, sort_keys=True)


def show_project_history_dialog(
    self,
    *,
    stage_filter: str | None = None,
    iteration_filter: int | None = None,
    transition_filter: str | None = None,
    stage_summary: dict | None = None,
):
    project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
    if not project_name:
        self.themed_info(
            "Historia projektu",
            "Historia jest dostępna po otwarciu projektu kampanii.",
            parent=self.root,
            tone="info",
        )
        return

    palette = getattr(self, "palette", {})
    bg = palette.get("bg", "#1e1e1e")
    panel = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    accent = palette.get("success", palette.get("accent", "#4ec9b0"))
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    normalized_stage = str(stage_filter or "").strip().upper()
    stage_step = campaign_stage_step(normalized_stage)
    stage_target_filter = ""
    if normalized_stage == "E4T":
        stage_target_filter = "plate"
    elif normalized_stage == "E4Z":
        stage_target_filter = "char"
    try:
        normalized_iteration = int(iteration_filter or 0)
    except Exception:
        normalized_iteration = 0
    normalized_transition = str(transition_filter or "").strip()
    stage_summary = dict(stage_summary or {})
    is_stage_dialog = bool(normalized_stage or stage_summary)

    title_suffix_parts = []
    if normalized_iteration > 0:
        title_suffix_parts.append(f"IT{normalized_iteration}")
    if normalized_stage:
        title_suffix_parts.append(normalized_stage)
    if normalized_transition:
        title_suffix_parts.append(normalized_transition)
    title_suffix = f" | {' / '.join(title_suffix_parts)}" if title_suffix_parts else ""

    dialog = tk.Toplevel(self.root)
    dialog.title(f"Historia projektu: {project_name}{title_suffix}")
    dialog.configure(bg=bg)
    dialog.transient(self.root)
    dialog.resizable(True, True)
    dialog.minsize(760, 420 if is_stage_dialog else 460)

    shell = tk.Frame(dialog, bg=bg, bd=0, highlightthickness=0, padx=12, pady=12)
    shell.pack(fill=tk.BOTH, expand=True)
    shell.rowconfigure(1, weight=0)
    shell.rowconfigure(2, weight=1)
    shell.rowconfigure(3, weight=0)
    if is_stage_dialog:
        shell.rowconfigure(2, weight=0)
    shell.columnconfigure(0, weight=1)

    header = tk.Frame(shell, bg=bg, bd=0, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    header.columnconfigure(0, weight=1)

    title_lbl = tk.Label(
        header,
        text=("Karta ścieżki" if is_stage_dialog else "Historia projektu"),
        bg=bg,
        fg=fg,
        font=("Segoe UI", 13, "bold"),
        anchor="w",
    )
    title_lbl.grid(row=0, column=0, sticky="ew")

    subtitle_lbl = tk.Label(
        header,
        text=(
            f"{project_name}{title_suffix} | ścieżka, wynik i najbliższy sens pracy"
            if title_suffix
            else f"{project_name} | ostatnie zdarzenia grafu i produkcji artefaktów"
        ),
        bg=bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
    )
    subtitle_lbl.grid(row=1, column=0, sticky="ew", pady=(3, 0))

    actions = tk.Frame(header, bg=bg, bd=0, highlightthickness=0)
    actions.grid(row=0, column=1, rowspan=2, sticky="e", padx=(10, 0))

    if normalized_stage or stage_summary:
        summary_shell = tk.Frame(shell, bg=panel, bd=0, highlightthickness=1, highlightbackground=border)
        summary_shell.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        summary_shell.columnconfigure(0, weight=1)

        summary_title = str(stage_summary.get("title") or "Karta etapu").strip()
        tk.Label(
            summary_shell,
            text=summary_title,
            bg=panel_alt,
            fg=accent,
            font=("Segoe UI", 10, "bold"),
            padx=9,
            pady=6,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        raw_rows = stage_summary.get("rows") if isinstance(stage_summary.get("rows"), (list, tuple)) else None
        if raw_rows:
            summary_rows = [
                (str(row[0] if isinstance(row, (list, tuple)) and len(row) > 0 else "").strip(),
                 str(row[1] if isinstance(row, (list, tuple)) and len(row) > 1 else "").strip())
                for row in raw_rows
            ]
            summary_rows = [(label, value) for label, value in summary_rows if label or value]
        else:
            summary_rows = [
                ("Wyprodukowano", stage_summary.get("current_effect") or "Brak szybkiego podsumowania dla tego etapu."),
                ("Ostatnia praca", stage_summary.get("last_effect") or "Brak istotnego wpisu produkcyjnego w historii."),
                ("Co odblokowuje", stage_summary.get("meaning") or "Sprawdź zasoby i aktywną bramkę na grafie."),
                ("Następny ruch", stage_summary.get("next_action") or "Wybierz aktywną bramkę albo otwórz jej kartę pracy."),
            ]
        for row_idx, (label, value) in enumerate(summary_rows, start=1):
            row_bg = panel if row_idx % 2 else bg
            row = tk.Frame(summary_shell, bg=row_bg, bd=0, highlightthickness=1, highlightbackground=border)
            row.grid(row=row_idx, column=0, sticky="ew")
            row.columnconfigure(1, weight=1)
            tk.Label(
                row,
                text=str(label),
                bg=row_bg,
                fg=muted,
                font=("Segoe UI", 8, "bold"),
                width=16,
                padx=8,
                pady=5,
                anchor="w",
            ).grid(row=0, column=0, sticky="nsw")
            tk.Label(
                row,
                text=str(value or "-"),
                bg=row_bg,
                fg=fg,
                font=("Segoe UI", 8),
                padx=8,
                pady=5,
                anchor="w",
                justify=tk.LEFT,
                wraplength=760,
            ).grid(row=0, column=1, sticky="ew")

    table_shell = tk.Frame(shell, bg=panel, bd=0, highlightthickness=1, highlightbackground=border)
    table_shell.grid(row=2, column=0, sticky="nsew")
    table_shell.rowconfigure(3, weight=1)
    table_shell.columnconfigure(0, weight=1)

    tk.Label(
        table_shell,
        text=("Przebieg i wynik" if is_stage_dialog else "Ślad i produkty projektu"),
        bg=panel_alt,
        fg=muted,
        font=("Segoe UI", 9, "bold"),
        padx=8,
        pady=5,
        anchor="w",
    ).grid(row=0, column=0, columnspan=2, sticky="ew")

    path_shell = tk.Frame(table_shell, bg=panel, bd=0, highlightthickness=1, highlightbackground=border)
    path_shell.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))
    path_shell.rowconfigure(0, weight=1)
    path_shell.columnconfigure(0, weight=1)
    path_text = tk.Text(
        path_shell,
        height=6,
        wrap=tk.WORD,
        bg=panel,
        fg=fg,
        insertbackground=fg,
        relief=tk.FLAT,
        bd=0,
        padx=4,
        pady=4,
        font=("Segoe UI", 8),
    )
    path_text.grid(row=0, column=0, sticky="ew", padx=(8, 0), pady=(4, 6))
    path_scrollbar = ttk.Scrollbar(path_shell, orient=tk.VERTICAL, command=path_text.yview)
    path_scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=(4, 6))
    path_text.configure(yscrollcommand=path_scrollbar.set, tabs=(94,))
    path_text.tag_configure("label", foreground=muted)
    path_text.tag_configure("stage", foreground=accent, font=("Segoe UI", 8, "bold"))
    path_text.tag_configure("gate", foreground=palette.get("warning", "#d7ba7d"), font=("Segoe UI", 8, "bold"))
    path_text.tag_configure("iteration", foreground=muted, font=("Segoe UI", 8, "bold"))
    path_text.tag_configure(
        "iteration_header",
        foreground=muted,
        font=("Segoe UI", 8, "bold"),
    )
    path_text.tag_configure("path_table_header", foreground=muted, font=("Segoe UI", 8, "bold"), spacing3=4)
    path_text.tag_configure("path_table_row", lmargin2=96, spacing1=2, spacing3=4)
    path_text.tag_configure("path_iteration_subscript", foreground=muted, font=("Segoe UI", 6), offset=-3)
    path_text.tag_configure("arrow", foreground=muted)
    path_text.tag_configure("plain", foreground=fg)
    path_text.tag_configure("ellipsis", foreground=muted)
    path_text.configure(cursor="arrow", takefocus=0)
    path_text.bind("<Key>", lambda _event: "break")

    result_shell = tk.Frame(table_shell, bg=panel, bd=0, highlightthickness=1, highlightbackground=border)
    result_shell.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 8))
    result_shell.columnconfigure(0, weight=1)
    tk.Label(
        result_shell,
        text="Produkty",
        bg=panel_alt,
        fg=muted,
        font=("Segoe UI", 8, "bold"),
        padx=8,
        pady=5,
        anchor="w",
    ).grid(row=0, column=0, sticky="ew")
    result_rows_host = tk.Frame(result_shell, bg=panel, bd=0, highlightthickness=0)
    result_rows_host.grid(row=1, column=0, sticky="ew", padx=8, pady=(6, 8))
    result_rows_host.columnconfigure(0, weight=1)
    product_columns = ("product", "iteration", "created_at", "source", "increment", "artifact")
    product_tree = ttk.Treeview(result_rows_host, columns=product_columns, show="headings", height=6, selectmode="browse")
    product_tree.heading("product", text="Produkt")
    product_tree.heading("iteration", text="IT")
    product_tree.heading("created_at", text="Data-czas")
    product_tree.heading("source", text="Źródło")
    product_tree.heading("increment", text="Przyrost")
    product_tree.heading("artifact", text="Artefakt / opis")
    product_tree.column("product", width=160, stretch=False, anchor=tk.W)
    product_tree.column("iteration", width=46, stretch=False, anchor=tk.CENTER)
    product_tree.column("created_at", width=138, stretch=False, anchor=tk.W)
    product_tree.column("source", width=110, stretch=False, anchor=tk.W)
    product_tree.column("increment", width=260, stretch=False, anchor=tk.W)
    product_tree.column("artifact", width=430, stretch=True, anchor=tk.W)
    product_tree.grid(row=0, column=0, sticky="ew")
    product_scrollbar = ttk.Scrollbar(result_rows_host, orient=tk.VERTICAL, command=product_tree.yview)
    product_scrollbar.grid(row=0, column=1, sticky="ns")
    product_tree.configure(yscrollcommand=product_scrollbar.set)
    product_tree.tag_configure("highlight", background=blend_hex_colors(accent, panel, 0.70), foreground=fg)
    product_tree.tag_configure("muted", foreground=muted)

    columns = ("time", "iteration", "gate", "action", "status", "title")
    tree = ttk.Treeview(table_shell, columns=columns, show="headings", height=12)
    tree.heading("time", text="Czas")
    tree.heading("iteration", text="IT")
    tree.heading("gate", text="Bramka")
    tree.heading("action", text="Akcja")
    tree.heading("status", text="Status")
    tree.heading("title", text="Co się stało")
    tree.column("time", width=142, stretch=False, anchor=tk.W)
    tree.column("iteration", width=46, stretch=False, anchor=tk.CENTER)
    tree.column("gate", width=76, stretch=False, anchor=tk.CENTER)
    tree.column("action", width=180, stretch=False, anchor=tk.W)
    tree.column("status", width=64, stretch=False, anchor=tk.CENTER)
    tree.column("title", width=340, stretch=True, anchor=tk.W)
    tree.grid(row=3, column=0, sticky="nsew")

    scrollbar = ttk.Scrollbar(table_shell, orient=tk.VERTICAL, command=tree.yview)
    scrollbar.grid(row=3, column=1, sticky="ns")
    tree.configure(yscrollcommand=scrollbar.set)

    details_shell = tk.Frame(shell, bg=panel, bd=0, highlightthickness=1, highlightbackground=border)
    details_shell.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    details_shell.columnconfigure(0, weight=1)

    details_lbl = tk.Label(
        details_shell,
        text="Szczegóły techniczne zaznaczonego wpisu",
        bg=panel_alt,
        fg=accent,
        font=("Segoe UI", 9, "bold"),
        padx=8,
        pady=5,
        anchor="w",
    )
    details_lbl.grid(row=0, column=0, sticky="ew")

    details_text = tk.Text(
        details_shell,
        height=8,
        wrap=tk.WORD,
        bg=panel,
        fg=fg,
        insertbackground=fg,
        relief=tk.FLAT,
        bd=0,
        padx=8,
        pady=8,
        font=("Consolas", 9),
    )
    details_text.grid(row=1, column=0, sticky="ew")
    details_text.configure(state=tk.DISABLED)
    if is_stage_dialog:
        try:
            table_shell.grid_remove()
            details_shell.grid_remove()
            shell.rowconfigure(2, weight=0)
        except Exception:
            pass
    else:
        try:
            tree.grid_remove()
            scrollbar.grid_remove()
            details_shell.grid_remove()
            table_shell.rowconfigure(3, weight=0)
            shell.rowconfigure(3, weight=0)
        except Exception:
            pass

    row_entries: dict[str, dict] = {}
    product_row_entries: dict[str, dict] = {}
    product_rows_cache: list[dict] = []
    history_resources = None
    product_sort_state = {"column": "product", "descending": False}
    history_target_by_iteration: dict[int, str] = {}
    history_path_by_iteration: dict[int, str] = {}
    product_names = {
        "AT": "Anotacje tablic",
        "AZ": "Anotacje znakĂłw",
        "MT": "Model tablic",
        "MZ": "Model znakĂłw",
    }
    product_code_rank = {code: index for index, code in enumerate(product_names)}
    path_segment_targets: dict[str, tuple[str, str]] = {}
    path_hover_tag = ""
    path_selected_tag = ""

    def _entry_matches_filters(entry: dict) -> bool:
        if normalized_iteration > 0:
            try:
                if int(entry.get("iteration", 0) or 0) != normalized_iteration:
                    return False
            except Exception:
                return False
        if stage_step > 0:
            try:
                if int(entry.get("step", 0) or 0) != stage_step:
                    return False
            except Exception:
                return False
        if stage_target_filter:
            iteration = _event_iteration(entry)
            entry_target = history_target_by_iteration.get(iteration) or _event_iteration_target(entry)
            if entry_target != stage_target_filter:
                return False
        if normalized_transition:
            transition = str(entry.get("transition_id") or entry.get("gate_id") or "").strip()
            if transition != normalized_transition:
                return False
        return True

    def _entry_matches_path_scope(entry: dict) -> bool:
        if normalized_iteration > 0:
            try:
                if int(entry.get("iteration", 0) or 0) != normalized_iteration:
                    return False
            except Exception:
                return False
        if normalized_transition:
            transition = str(entry.get("transition_id") or entry.get("gate_id") or "").strip()
            if transition != normalized_transition:
                return False
        return True

    def _set_details(text: str) -> None:
        try:
            details_text.configure(state=tk.NORMAL)
            details_text.delete("1.0", tk.END)
            details_text.insert("1.0", text)
            details_text.configure(state=tk.DISABLED)
        except Exception:
            pass

    def _highlight_products_for_path(token: str, iteration_label: str) -> None:
        normalized_token = str(token or "").strip().upper()
        normalized_iteration = str(iteration_label or "").strip()
        matched_ids: list[str] = []
        for item_id, row in product_row_entries.items():
            row_matches = set(row.get("matches") or set())
            if (normalized_token, normalized_iteration) in row_matches or (normalized_token, "") in row_matches:
                matched_ids.append(item_id)
        try:
            for item_id in product_tree.get_children():
                row = product_row_entries.get(str(item_id), {})
                tags = ["highlight"] if str(item_id) in matched_ids else []
                if not row:
                    tags.append("muted")
                product_tree.item(item_id, tags=tuple(tags))
            if matched_ids:
                first = matched_ids[0]
                product_tree.selection_set(first)
                product_tree.focus(first)
                product_tree.see(first)
            else:
                product_tree.selection_remove(product_tree.selection())
        except Exception:
            pass

    def _style_path_segment(segment_tag: str) -> None:
        tag = str(segment_tag or "").strip()
        if not tag:
            return
        try:
            if tag == path_selected_tag:
                path_text.tag_configure(
                    tag,
                    background=blend_hex_colors(accent, panel, 0.64),
                    underline=True,
                )
            elif tag == path_hover_tag:
                path_text.tag_configure(
                    tag,
                    background=blend_hex_colors(accent, panel, 0.82),
                    underline=True,
                )
            else:
                path_text.tag_configure(tag, background="", underline=True)
        except Exception:
            pass

    def _set_path_hover(segment_tag: str = "") -> None:
        nonlocal path_hover_tag
        normalized_tag = str(segment_tag or "").strip()
        if normalized_tag == path_hover_tag:
            return
        previous_tag = path_hover_tag
        try:
            path_hover_tag = normalized_tag
            _style_path_segment(previous_tag)
            _style_path_segment(path_hover_tag)
            path_text.configure(cursor=("hand2" if path_hover_tag else "arrow"))
        except Exception:
            pass

    def _set_path_selected(segment_tag: str = "") -> None:
        nonlocal path_selected_tag
        normalized_tag = str(segment_tag or "").strip()
        if normalized_tag == path_selected_tag:
            return
        previous_tag = path_selected_tag
        try:
            path_selected_tag = normalized_tag
            _style_path_segment(previous_tag)
            _style_path_segment(path_selected_tag)
        except Exception:
            pass

    def _path_segment_tag_at(event: tk.Event) -> str:
        try:
            index = path_text.index(f"@{int(event.x)},{int(event.y)}")
            tags = path_text.tag_names(index)
        except Exception:
            return ""
        for tag in tags:
            tag_text = str(tag or "")
            if tag_text.startswith("path_segment_") and tag_text in path_segment_targets:
                return tag_text
        return ""

    def _on_path_motion(event: tk.Event):
        _set_path_hover(_path_segment_tag_at(event))

    def _on_path_leave(_event: tk.Event):
        _set_path_hover("")

    def _on_path_click(event: tk.Event):
        segment_tag = _path_segment_tag_at(event)
        if not segment_tag:
            return "break"
        _set_path_selected(segment_tag)
        token, iteration_label = path_segment_targets.get(segment_tag, ("", ""))
        _highlight_products_for_path(token, iteration_label)
        return "break"

    path_text.bind("<Motion>", _on_path_motion, add="+")
    path_text.bind("<Leave>", _on_path_leave, add="+")
    path_text.bind("<Button-1>", _on_path_click, add="+")

    def _group_path_segments_by_iteration(
        segments: list[tuple[str, str, str]]
    ) -> list[tuple[str, list[tuple[str, str, str]]]]:
        groups: list[tuple[str, list[tuple[str, str, str]]]] = []
        current_label = ""
        current_items: list[tuple[str, str, str]] = []
        for kind, token, iteration_label in segments:
            normalized_label = str(iteration_label or "").strip()
            if kind == "ellipsis" and current_items:
                current_items.append((kind, token, iteration_label))
                continue
            if not normalized_label:
                normalized_label = current_label or "?"
            if current_items and normalized_label != current_label:
                groups.append((current_label, current_items))
                current_items = []
            current_label = normalized_label
            current_items.append((kind, token, iteration_label))
        if current_items:
            groups.append((current_label, current_items))
        return groups

    def _set_path_trace(entries: list[dict], *, label: str = "Ścieżka") -> None:
        nonlocal path_hover_tag, path_selected_tag
        try:
            segments = _build_history_path_segments(entries, max_tokens=160)
        except Exception:
            segments = []
        try:
            path_hover_tag = ""
            path_selected_tag = ""
            path_segment_targets.clear()
            for tag in list(path_text.tag_names()):
                if str(tag or "").startswith("path_segment_"):
                    try:
                        path_text.tag_delete(tag)
                    except Exception:
                        pass
            path_text.delete("1.0", tk.END)
            path_text.configure(cursor="arrow")
            path_text.insert(tk.END, "Iteracja\tŚlad\n", ("path_table_header",))
            if not segments:
                line_start = path_text.index(tk.END)
                path_text.insert(tk.END, "-\t", ("iteration_header",))
                path_text.insert(tk.END, "brak rozpoznanych przejść grafu", ("plain",))
                line_end = path_text.index("end-1c")
                path_text.tag_add("path_table_row", line_start, line_end)
            else:
                for group_index, (iteration_label, group_segments) in enumerate(
                    _group_path_segments_by_iteration(segments)
                ):
                    if group_index:
                        path_text.insert(tk.END, "\n", ("plain",))
                    iteration_caption = f"Iteracja {iteration_label}" if iteration_label != "?" else "Iteracja ?"
                    try:
                        iteration_number = int(iteration_label or 0)
                    except Exception:
                        iteration_number = 0
                    path_label = _iteration_path_label(history_path_by_iteration.get(iteration_number, ""))
                    if path_label:
                        iteration_caption = f"{iteration_caption} | {path_label}"
                    line_start = path_text.index(tk.END)
                    path_text.insert(tk.END, f"{iteration_caption}\t", ("iteration_header",))
                    for segment_index, (kind, token, segment_iteration_label) in enumerate(group_segments):
                        if segment_index:
                            path_text.insert(tk.END, "  →  ", ("arrow",))
                        segment_tag = f"path_segment_{group_index}_{segment_index}"
                        start_index = path_text.index(tk.END)
                        path_text.insert(tk.END, token, (kind,))
                        end_index = path_text.index(tk.END)
                        if kind in {"stage", "gate"}:
                            path_text.tag_add(segment_tag, start_index, end_index)
                            path_text.tag_configure(segment_tag, underline=True)
                            path_segment_targets[segment_tag] = (
                                token,
                                segment_iteration_label or iteration_label,
                            )
                    line_end = path_text.index("end-1c")
                    path_text.tag_add("path_table_row", line_start, line_end)
        except Exception:
            pass

    def _choose_path_entries(all_entries: list[dict]) -> tuple[list[dict], str]:
        try:
            project_tokens = _build_history_path_tokens(all_entries)
        except Exception:
            project_tokens = []
        if project_tokens:
            return list(all_entries), "Ślad projektu od początku"
        current_entries = [entry for entry in all_entries if _entry_matches_path_scope(entry)]
        return current_entries, "Ślad projektu"

    def _product_sort_value(row: dict, column: str):
        code = str(row.get("code") or "").strip().upper()
        if column == "product":
            return (
                product_code_rank.get(code, 99),
                str(row.get("name") or "").casefold(),
                int(row.get("iteration", 0) or 0),
                str(row.get("created_at") or ""),
            )
        if column == "iteration":
            return (
                int(row.get("iteration", 0) or 999999),
                product_code_rank.get(code, 99),
                str(row.get("created_at") or ""),
            )
        if column == "created_at":
            return (
                str(row.get("created_at") or ""),
                int(row.get("iteration", 0) or 0),
                product_code_rank.get(code, 99),
            )
        if column == "source":
            return str(row.get("source") or "").casefold()
        if column == "increment":
            return str(row.get("increment") or "").casefold()
        if column == "artifact":
            return str(row.get("artifact") or "").casefold()
        return str(row.get(column) or "").casefold()

    def _render_product_rows(rows: list[dict]) -> None:
        product_row_entries.clear()
        try:
            product_tree.delete(*product_tree.get_children())
        except Exception:
            pass
        sort_column = str(product_sort_state.get("column") or "product")
        descending = bool(product_sort_state.get("descending"))
        sorted_rows = sorted(
            list(rows or []),
            key=lambda row: _product_sort_value(dict(row or {}), sort_column),
            reverse=descending,
        )
        for index, row in enumerate(sorted_rows):
            item_id = f"product_{index}"
            product_row_entries[item_id] = dict(row)
            iteration = int(row.get("iteration", 0) or 0)
            tags = ("muted",) if row.get("status") == "brak" else ()
            product_tree.insert(
                "",
                tk.END,
                iid=item_id,
                values=(
                    f"{row.get('code', '')} | {row.get('name', '')}".strip(),
                    f"IT{iteration}" if iteration > 0 else "-",
                    _format_history_timestamp(row.get("created_at")),
                    row.get("source", "-"),
                    row.get("increment", "-"),
                    _short_text(row.get("artifact", "-"), 150),
                ),
                tags=tags,
            )

    def _sort_product_rows(column: str) -> None:
        normalized_column = str(column or "").strip()
        if not normalized_column:
            return
        if str(product_sort_state.get("column") or "") == normalized_column:
            product_sort_state["descending"] = not bool(product_sort_state.get("descending"))
        else:
            product_sort_state["column"] = normalized_column
            product_sort_state["descending"] = False
        _render_product_rows(product_rows_cache)

    for _product_column in product_columns:
        try:
            product_tree.heading(
                _product_column,
                command=lambda column=_product_column: _sort_product_rows(column),
            )
        except Exception:
            pass

    def _set_product_rows(all_entries: list[dict]) -> None:
        product_rows_cache.clear()
        try:
            product_tree.delete(*product_tree.get_children())
        except Exception:
            pass
        rows = _build_project_product_rows(project_name, all_entries, resource_reader=history_resources)
        product_names = {
            "AT": "Anotacje tablic",
            "AZ": "Anotacje znaków",
            "MT": "Model tablic",
            "MZ": "Model znaków",
        }
        present_codes = {str(row.get("code") or "").strip().upper() for row in rows}
        for code, name in product_names.items():
            if code in present_codes:
                continue
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "iteration": 0,
                    "created_at": "",
                    "source": "-",
                    "status": "brak",
                    "increment": "brak produktu",
                    "artifact": "Nie wytworzono jeszcze w historii projektu.",
                    "matches": set(),
                }
            )
        code_rank = {code: index for index, code in enumerate(product_names)}
        rows.sort(
            key=lambda row: (
                product_code_rank.get(str(row.get("code") or "").strip().upper(), 99),
                int(row.get("iteration", 0) or 999999),
                str(row.get("created_at") or ""),
                str(row.get("source") or ""),
            )
        )
        product_rows_cache.extend(dict(row or {}) for row in rows)
        _render_product_rows(product_rows_cache)

    def _load_rows(select_first: bool = True) -> None:
        nonlocal history_path_by_iteration, history_target_by_iteration, history_resources
        row_entries.clear()
        try:
            tree.delete(*tree.get_children())
        except Exception:
            pass
        all_entries = [dict(entry or {}) for entry in CAMPAIGN.load_project_history(project_name, limit=800)]
        history_resources = HistoryResourceReader(CAMPAIGN.get_project_root_dir(project_name), project_name)
        history_path_by_iteration = _iteration_path_map(all_entries)
        history_target_by_iteration = _iteration_target_map(all_entries)
        path_entries, path_label = _choose_path_entries(all_entries)
        _set_path_trace(path_entries, label=path_label)
        _set_product_rows(all_entries)
        entries = [entry for entry in all_entries if _entry_matches_filters(entry)]
        if not entries:
            if title_suffix:
                _set_details("Brak wpisów historii dla wybranego etapu lub filtra.")
            else:
                _set_details("Brak wpisów historii dla tego projektu.")
            return

        for index, entry in enumerate(entries):
            item_id = f"event_{index}"
            row_entries[item_id] = dict(entry)
            tree.insert(
                "",
                tk.END,
                iid=item_id,
                values=(
                    str(entry.get("created_at", "") or "-").replace("T", " ")[:19],
                    f"IT{int(entry.get('iteration', 1) or 1)}",
                    _event_gate(entry),
                    _short_text(entry.get("action") or entry.get("event_type"), 28),
                    _event_status(entry),
                    _short_text(_event_title(entry), 120),
                ),
            )

        if select_first:
            children = tree.get_children()
            if children:
                last = children[-1]
                tree.selection_set(last)
                tree.focus(last)
                tree.see(last)
                _set_details(_format_event_details(row_entries.get(last, {}), reader=history_resources))

    def _on_select(_event=None) -> None:
        selected = tree.selection()
        if not selected:
            return
        entry = row_entries.get(str(selected[0]), {})
        _set_details(_format_event_details(entry, reader=history_resources))

    def _copy_selected() -> None:
        selected = tree.selection()
        if not selected:
            return
        entry = row_entries.get(str(selected[0]), {})
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(json.dumps(entry, ensure_ascii=False, indent=2, sort_keys=True))
        except Exception:
            pass

    def _refresh() -> None:
        _load_rows(select_first=False)

    technical_visible = False

    def _toggle_technical_view() -> None:
        nonlocal technical_visible
        if is_stage_dialog:
            return
        technical_visible = not technical_visible
        try:
            if technical_visible:
                tree.grid(row=3, column=0, sticky="nsew")
                scrollbar.grid(row=3, column=1, sticky="ns")
                details_shell.grid(row=3, column=0, sticky="ew", pady=(10, 0))
                table_shell.rowconfigure(3, weight=1)
                shell.rowconfigure(3, weight=0)
                technical_btn.configure(text="Ukryj dziennik techniczny")
                copy_btn.pack(side=tk.LEFT, padx=(0, 6), before=close_btn)
            else:
                tree.grid_remove()
                scrollbar.grid_remove()
                details_shell.grid_remove()
                table_shell.rowconfigure(3, weight=0)
                shell.rowconfigure(3, weight=0)
                technical_btn.configure(text="Pokaż dziennik techniczny")
                copy_btn.pack_forget()
        except Exception:
            pass

    refresh_btn = ttk.Button(actions, text="Odśwież", command=_refresh)
    refresh_btn.pack(side=tk.LEFT, padx=(0, 6))
    copy_btn = ttk.Button(actions, text="Kopiuj wpis", command=_copy_selected)
    if not is_stage_dialog:
        technical_btn = ttk.Button(actions, text="Pokaż dziennik techniczny", command=_toggle_technical_view)
        technical_btn.pack(side=tk.LEFT, padx=(0, 6))
    else:
        technical_btn = ttk.Button(actions, text="Pokaż dziennik techniczny", command=_toggle_technical_view)
    close_btn = ttk.Button(actions, text="Zamknij", command=dialog.destroy)
    close_btn.pack(side=tk.LEFT)

    tree.bind("<<TreeviewSelect>>", _on_select, add="+")
    dialog.bind("<Escape>", lambda _event: dialog.destroy(), add="+")

    _load_rows(select_first=True)
    try:
        self._center_dialog_window(dialog, parent=self.root, width=1180, height=520 if is_stage_dialog else 620)
    except Exception:
        dialog.geometry("1180x520" if is_stage_dialog else "1180x620")
    try:
        dialog.lift()
        dialog.focus_force()
    except Exception:
        pass
