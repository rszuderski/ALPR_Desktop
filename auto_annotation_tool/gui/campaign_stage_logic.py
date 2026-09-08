#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Campaign wizard stage decision logic extracted from tab_campaign.py."""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from textwrap import shorten
from collections import Counter
from datetime import datetime
from time import perf_counter
import xml.etree.ElementTree as ET
import shutil
import threading

from ..config import CONFIG, logger, PIL_AVAILABLE, Image, ImageTk, ImageDraw, ImageFont
from ..campaign_manager import CAMPAIGN
from ..campaign_iteration_paths import normalize_iteration_path
from ..campaign_ingest_planner import CHAR_ALPHABET, CampaignIngestPlanner
from ..validators import validate_model_file, format_yolo_model_identity
from ..icons import IconManager
from ..project_cache import PROJECT_CACHE
from . import campaign_dashboard_cache
from . import campaign_ui_helpers
from . import campaign_project_browser
from . import campaign_model_status
from . import campaign_step1_assets
from . import campaign_step1_ingest
from . import campaign_stage_ui
from . import campaign_graph_actions
from .help_manager import HELP
from .run_display import build_run_display_ref
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .z3_view_models import Step3ViewModel
from .campaign_models import WizardStageStatus
from .z2_shared_ui import campaign_visible_gate_id, campaign_gate_id_for_edge


def _get_step2_disk_approval_fallback(
    self,
    *,
    iteration_target: str = "",
    plate_ready_source: dict | None = None,
    char_ready_source: dict | None = None,
    extra_run_dirs: list[object] | tuple[object, ...] | None = None,
) -> dict:
    target = self._normalize_iteration_target(iteration_target)
    if target not in {"plate", "char"}:
        return {"ready": False}

    annotation_tab = self.app.tabs.get("annotation") if getattr(self.app, "tabs", None) else None
    source_state = dict(plate_ready_source or {}) if target == "plate" else dict(char_ready_source or {})
    bootstrap = dict(source_state.get("bootstrap") or {})
    candidates: list[object] = []

    preferred_candidate_count = 0
    for raw_candidate in list(extra_run_dirs or []):
        if raw_candidate:
            candidates.append(raw_candidate)
            preferred_candidate_count += 1
    for raw_candidate in (
        CAMPAIGN.get_step2_staging_run(),
        source_state.get("restore_run_dir"),
        bootstrap.get("restore_run_dir"),
    ):
        if raw_candidate:
            candidates.append(raw_candidate)

    try:
        bundle = dict(
            CAMPAIGN.get_iteration_artifact_bundle(
                iteration_num=int(CAMPAIGN.get_current_iteration_num() or 1)
            ) or {}
        )
    except Exception:
        bundle = {}
    for entry_key in ("step2_active_run", "plate_source", "char_effective_source"):
        entry = dict(bundle.get(entry_key) or {})
        raw_candidate = entry.get("run_dir")
        if raw_candidate:
            candidates.append(raw_candidate)

    def resolve_run(raw_value) -> Path | None:
        if not raw_value:
            return None
        if annotation_tab is not None and hasattr(annotation_tab, "_resolve_safe_annotation_run_dir"):
            try:
                resolved = annotation_tab._resolve_safe_annotation_run_dir(raw_value, require_xml=True)
                if resolved is not None:
                    return Path(resolved)
            except Exception:
                pass
        try:
            candidate = Path(raw_value)
        except Exception:
            return None
        try:
            if candidate.exists() and candidate.is_dir() and (candidate / "annotations.xml").exists():
                return candidate
        except Exception:
            return None
        try:
            from . import z2_manifest_runtime

            resolved = z2_manifest_runtime._resolve_campaign_annotation_run_dir(
                candidate,
                str(CAMPAIGN.get_active_project_name() or "").strip(),
            )
            if resolved is not None:
                return Path(resolved)
        except Exception:
            pass
        return None

    def count_run_approved(run_dir: Path | None) -> tuple[int, int]:
        if run_dir is None:
            return 0, 0
        if annotation_tab is not None and hasattr(annotation_tab, "_get_run_plate_approved_counts"):
            try:
                return annotation_tab._get_run_plate_approved_counts(run_dir)
            except Exception:
                pass
        try:
            manifest = CAMPAIGN._load_annotation_run_manifest_file(run_dir)
            approved_names = {
                CAMPAIGN._normalize_image_set_name(name)
                for name in list(manifest.get("approved_filenames") or [])
                if CAMPAIGN._normalize_image_set_name(name)
            }
            counts = CAMPAIGN._load_run_plate_counts_by_image(run_dir, image_names=approved_names)
            image_count = sum(1 for value in counts.values() if int(value or 0) > 0)
            plate_count = sum(max(0, int(value or 0)) for value in counts.values())
            return int(image_count), int(plate_count)
        except Exception:
            return 0, 0

    staging_root = None
    auto_root = None
    try:
        staging_root = CAMPAIGN.get_staging_dir("auto_ann")
        staging_root = Path(staging_root) if staging_root is not None else None
    except Exception:
        staging_root = None
    try:
        auto_root = CAMPAIGN.get_dir("auto_ann")
        auto_root = Path(auto_root) if auto_root is not None else None
    except Exception:
        auto_root = None

    def path_is_within(candidate: Path | None, root: Path | None) -> bool:
        if candidate is None or root is None:
            return False
        try:
            Path(candidate).resolve().relative_to(Path(root).resolve())
            return True
        except Exception:
            return False

    def classify_run_source(run_dir: Path | None) -> str:
        if run_dir is None:
            return ""
        if path_is_within(run_dir, staging_root):
            return "staging"
        if path_is_within(run_dir, auto_root):
            return "existing"
        return ""

    resolved_run = None
    resolved_source_kind = ""
    run_images = 0
    run_plates = 0
    min_char_plates = int(getattr(self, "STEP3_CHAR_MIN_PLATES", 10) or 10)
    min_plate_plates = int(getattr(self, "STEP2_PLATE_MIN_PLATES", 10) or 10)
    seen: set[str] = set()
    for candidate_index, raw_candidate in enumerate(candidates):
        candidate = resolve_run(raw_candidate)
        if candidate is None:
            continue
        candidate_source_kind = classify_run_source(candidate)
        if not candidate_source_kind:
            continue
        try:
            key = str(candidate.resolve()).lower()
        except Exception:
            key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        images, plates = count_run_approved(candidate)
        if images > run_images or plates > run_plates or resolved_run is None:
            resolved_run = candidate
            resolved_source_kind = candidate_source_kind
            run_images = int(images or 0)
            run_plates = int(plates or 0)
        if candidate_index < preferred_candidate_count and (images > 0 or plates > 0):
            resolved_run = candidate
            resolved_source_kind = candidate_source_kind
            run_images = int(images or 0)
            run_plates = int(plates or 0)
            break
        if target == "char":
            if run_plates >= min_char_plates:
                break
        elif run_plates >= min_plate_plates:
            break

    try:
        approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
    except Exception:
        approved_stats = {}
    project_images = max(0, int(approved_stats.get("images", 0) or 0))
    project_plates = max(0, int(approved_stats.get("plates", 0) or 0))
    unpromoted_images = 0
    unpromoted_plates = 0
    unpromoted_names: set[str] = set()
    if resolved_run is not None and int(run_images or 0) > 0:
        try:
            approved_names = set()
            if annotation_tab is not None and hasattr(annotation_tab, "_load_annotation_run_approved_filenames"):
                approved_names = set(annotation_tab._load_annotation_run_approved_filenames(resolved_run) or set())
            if not approved_names:
                manifest = CAMPAIGN._load_annotation_run_manifest_file(resolved_run)
                approved_names = {
                    str(name or "").strip().lower()
                    for name in list(manifest.get("approved_filenames") or [])
                    if str(name or "").strip()
                }
        except Exception:
            approved_names = set()
        try:
            project_names = {
                CAMPAIGN._normalize_image_set_name(entry.get("image_name", ""))
                for entry in list(CAMPAIGN.list_plate_approved_entries() or [])
                if isinstance(entry, dict) and CAMPAIGN._normalize_image_set_name(entry.get("image_name", ""))
            }
        except Exception:
            project_names = set()
        try:
            normalized_approved_names = {
                CAMPAIGN._normalize_image_set_name(name)
                for name in set(approved_names or set())
                if CAMPAIGN._normalize_image_set_name(name)
            }
        except Exception:
            normalized_approved_names = {
                str(name or "").strip().lower()
                for name in set(approved_names or set())
                if str(name or "").strip()
            }
        unpromoted_names = set(normalized_approved_names or set()) - set(project_names or set())
        if unpromoted_names:
            try:
                pending_counts = dict(
                    CAMPAIGN._load_run_plate_counts_by_image(
                        resolved_run,
                        image_names=set(unpromoted_names),
                    )
                    or {}
                )
            except Exception:
                pending_counts = {}
            if pending_counts:
                unpromoted_images = sum(1 for value in pending_counts.values() if int(value or 0) > 0)
                unpromoted_plates = sum(max(0, int(value or 0)) for value in pending_counts.values())
            else:
                unpromoted_images = len(unpromoted_names)
                unpromoted_plates = int(run_plates or 0)
    total_images = project_images + int(run_images or 0)
    total_plates = project_plates + int(run_plates or 0)
    if target == "char":
        ready = bool(total_plates >= min_char_plates)
    else:
        ready = bool(total_plates >= min_plate_plates)

    action = ""
    if ready:
        action = "approve_stage"
        if target == "char" and int(run_images or 0) <= 0 and source_state:
            action = "continue_characters"

    return {
        "ready": bool(ready),
        "iteration_target": target,
        "action": action,
        "run_dir": resolved_run,
        "source_kind": resolved_source_kind,
        "run_approved_images": int(run_images or 0),
        "run_approved_plates": int(run_plates or 0),
        "unpromoted_approved_images": int(unpromoted_images or 0),
        "unpromoted_approved_plates": int(unpromoted_plates or 0),
        "unpromoted_approved_names": sorted(set(unpromoted_names or set())),
        "interrupted_work": bool(unpromoted_images > 0 or unpromoted_plates > 0),
        "project_approved_images": int(project_images or 0),
        "project_approved_plates": int(project_plates or 0),
        "total_images": int(total_images or 0),
        "total_plates": int(total_plates or 0),
    }


def _get_step2_current_iteration_contribution_state(
    self,
    *,
    approval_context: dict | None = None,
    source_state: dict | None = None,
    run_approved_images: int | None = None,
    run_approved_plates: int | None = None,
) -> dict:
    context = dict(approval_context or {})
    source = dict(source_state or {})

    try:
        iteration_stats = dict(CAMPAIGN.get_plate_approved_set_iteration_stats() or {})
    except Exception:
        iteration_stats = {}
    try:
        approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
    except Exception:
        approved_stats = {}

    def _int_value(payload: dict, key: str) -> int:
        try:
            return max(0, int(payload.get(key, 0) or 0))
        except Exception:
            return 0

    current_images = _int_value(iteration_stats, "images")
    current_plates = _int_value(iteration_stats, "plates")

    for image_key, plate_key in (
        ("current_images_with_plates", "current_total_plates"),
        ("pending_images_with_plates", "pending_total_plates"),
    ):
        current_images = max(current_images, _int_value(source, image_key))
        current_plates = max(current_plates, _int_value(source, plate_key))

    if run_approved_images is not None or run_approved_plates is not None:
        try:
            current_images = max(current_images, int(run_approved_images or 0))
        except Exception:
            pass
        try:
            current_plates = max(current_plates, int(run_approved_plates or 0))
        except Exception:
            pass
    else:
        run_dir = context.get("run_dir")
        source_kind = str(context.get("source_kind") or "").strip().lower()
        if run_dir is not None and source_kind == "staging":
            try:
                annotation_tab = self.app.tabs.get("annotation") if getattr(self.app, "tabs", None) else None
                counter = getattr(annotation_tab, "_get_run_plate_approved_counts", None)
                if callable(counter):
                    run_images, run_plates = counter(Path(run_dir))
                    current_images = max(current_images, int(run_images or 0))
                    current_plates = max(current_plates, int(run_plates or 0))
            except Exception:
                pass

    project_images = max(
        _int_value(approved_stats, "images"),
        _int_value(source, "project_images_with_plates"),
        _int_value(source, "images_with_plates"),
    )
    project_plates = max(
        _int_value(approved_stats, "plates"),
        _int_value(source, "project_total_plates"),
        _int_value(source, "total_plates"),
    )

    try:
        iteration = int(iteration_stats.get("iteration", CAMPAIGN.get_current_iteration_num() or 1) or 1)
    except Exception:
        iteration = 1

    return {
        "iteration": int(iteration),
        "current_images": int(current_images),
        "current_plates": int(current_plates),
        "project_images": int(project_images),
        "project_plates": int(project_plates),
        "iteration_target": str(
            context.get("iteration_target")
            or source.get("iteration_target")
            or CAMPAIGN.get_iteration_target()
            or ""
        ).strip().lower(),
    }


def _confirm_step2_without_current_iteration_contribution(
    self,
    *,
    approval_context: dict | None = None,
    source_state: dict | None = None,
    run_approved_images: int | None = None,
    run_approved_plates: int | None = None,
) -> bool:
    contribution = self._get_step2_current_iteration_contribution_state(
        approval_context=approval_context,
        source_state=source_state,
        run_approved_images=run_approved_images,
        run_approved_plates=run_approved_plates,
    )
    current_images = int(contribution.get("current_images", 0) or 0)
    current_plates = int(contribution.get("current_plates", 0) or 0)
    project_images = int(contribution.get("project_images", 0) or 0)
    project_plates = int(contribution.get("project_plates", 0) or 0)
    iteration = int(contribution.get("iteration", CAMPAIGN.get_current_iteration_num() or 1) or 1)
    target = str(contribution.get("iteration_target") or "").strip().lower()

    if current_images > 0 or current_plates > 0 or project_plates <= 0:
        return True

    next_step_text = (
        "E4T zostanie uruchomione na tej samej puli tablic co wcześniej."
        if target == "plate"
        else "E3/Z3 zostanie uruchomione na obecnej puli tablic projektu."
    )
    repair_hint = (
        "\n\nW torze znaków to nie zamyka drogi do dopisania tablic: w E3 możesz wrócić do Z2 "
        "w trybie naprawczym i powiększyć pulę przed finalnym zatwierdzeniem E3."
        if target == "char"
        else ""
    )
    message = (
        f"Iteracja {iteration:03d} spełnia bramkę E2 dzięki zatwierdzonej puli projektu, "
        "ale w tej iteracji nie dodano żadnych nowych zatwierdzonych obrazów z tablicami.\n\n"
        f"Aktualna pula projektu: {project_images} zdjęć / {project_plates} tablic.\n"
        "Nowy wkład tej iteracji: 0 zdjęć / 0 tablic.\n\n"
        f"Jeśli zatwierdzisz E2 mimo to, {next_step_text} To jest poprawne tylko wtedy, gdy świadomie "
        "chcesz kontynuować bez powiększania materiału wejściowego."
        f"{repair_hint}"
    )
    title = "Brak nowych tablic w tej iteracji" if target == "char" else "E2 bez nowych danych"
    confirm_label = "Kontynuuj bez nowych tablic" if target == "char" else "Zatwierdź mimo to"
    cancel_label = "Oznacz więcej tablic" if target == "char" else "Wróć do E2"
    try:
        return bool(
            self.app.themed_confirm(
                title,
                message,
                parent=self.frame,
                confirm_label=confirm_label,
                cancel_label=cancel_label,
                tone="warning",
            )
        )
    except Exception:
        return True


def _resolve_step2_wizard_action_command(self, action_id: str, *, context: dict | None = None):
    normalized = str(action_id or "").strip().lower()
    if not normalized:
        return None

    if normalized == "open_z2":
        return lambda: campaign_graph_actions.execute_campaign_graph_action(self, "open_z2_campaign_context")
    if normalized == "open_z2_step2_review":
        return lambda: campaign_graph_actions.execute_campaign_graph_action(self, "open_z2_step2_review")
    if normalized == "return_to_z2":
        return lambda: campaign_graph_actions.execute_campaign_graph_action(self, "open_z2_step3_repair")
    if normalized == "continue_z3":
        return lambda ctx=dict(context or {}): campaign_graph_actions.execute_campaign_graph_action(
            self,
            "continue_z3",
            payload={"context": ctx},
        )
    return None


def _approve_step2_from_wizard(self, context: dict | None = None):
    annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
    if annotation_tab is None or not callable(getattr(annotation_tab, "_approve_annotation_stage", None)):
        annotation_tab = self.app._ensure_tab_loaded("annotation", select=False)
    if annotation_tab is None or not callable(getattr(annotation_tab, "_approve_annotation_stage", None)):
        return False
    graph_context = dict(context or {})
    if graph_context:
        try:
            annotation_tab._campaign_graph_entry_context = graph_context
        except Exception:
            pass

    try:
        ensure_context = getattr(annotation_tab, "ensure_campaign_context_ready_for_active_project", None)
        if callable(ensure_context):
            ensure_context(force=False)
    except Exception:
        pass

    try:
        wizard_status_getter = getattr(annotation_tab, "get_campaign_step2_wizard_status", None)
        wizard_status = dict(wizard_status_getter() or {}) if callable(wizard_status_getter) else {}
    except Exception:
        wizard_status = {}

    approval_action = str(wizard_status.get("approval_action", "") or "").strip().lower()
    approval_iteration_target = str(wizard_status.get("approval_iteration_target", "") or "").strip().lower()

    try:
        approval_context_getter = getattr(annotation_tab, "_get_campaign_step2_approval_context", None)
        approval_context = dict(approval_context_getter() or {}) if callable(approval_context_getter) else {}
    except Exception:
        approval_context = {}
    approval_run_dir = approval_context.get("run_dir")
    if approval_iteration_target not in {"plate", "char"}:
        approval_iteration_target = self._get_iteration_target()

    graph_gate_id = campaign_gate_id_for_edge(graph_context.get("graph_edge_key"), graph_context.get("graph_gate_id"))
    graph_display_gate_id = campaign_visible_gate_id(graph_gate_id) if graph_gate_id else ""
    extra_run_dirs = []
    if graph_gate_id == "T04":
        try:
            iteration_state = dict(CAMPAIGN.get_iteration_state() or {})
            session = dict(iteration_state.get("t05_work_session") or {})
            session_state = str(session.get("state") or "").strip().lower()
            session_result = dict(session.get("last_return_result") or {})
            session_resolved = bool(session_result.get("ok")) or session_state in {
                "resolved",
                "closed",
                "complete",
                "completed",
            }
            if session_resolved:
                try:
                    session_run = str(session.get("run_dir", "") or "").strip()
                    now = datetime.now().isoformat(timespec="seconds")
                    CAMPAIGN.upsert_iteration_state(
                        updates={
                            "t05_work_session": {
                                **session,
                                "active": False,
                                "state": "resolved",
                                "resolved_at": str(session.get("resolved_at") or now),
                                "updated_at": now,
                                "last_return_result": session_result,
                                "run_dir": session_run,
                            }
                        }
                    )
                except Exception:
                    pass
                session_active = False
            else:
                session_active = bool(session.get("active")) or session_state in {"active", "started", "interrupted", "dirty"}
            session_run = str(session.get("run_dir", "") or "").strip()
            if session_active and session_run and session_state not in {"resolved", "closed", "complete", "completed"}:
                extra_run_dirs.append(session_run)
        except Exception:
            pass

    disk_fallback = {}
    try:
        disk_fallback = self._get_step2_disk_approval_fallback(
            iteration_target=approval_iteration_target,
            plate_ready_source=self._get_plate_route_ready_source() if approval_iteration_target == "plate" else {},
            char_ready_source=self._get_char_route_ready_source() if approval_iteration_target == "char" else {},
            extra_run_dirs=extra_run_dirs,
        )
    except Exception:
        disk_fallback = {}
    if bool(disk_fallback.get("ready")):
        if approval_iteration_target not in {"plate", "char"}:
            approval_iteration_target = str(disk_fallback.get("iteration_target") or "").strip().lower()
        if not approval_action:
            approval_action = str(disk_fallback.get("action") or "").strip().lower()
        if approval_run_dir is None and disk_fallback.get("run_dir") is not None:
            approval_run_dir = disk_fallback.get("run_dir")
            approval_context["run_dir"] = approval_run_dir
            approval_context["source_kind"] = str(disk_fallback.get("source_kind") or "staging")

    if approval_iteration_target in {"plate", "char"}:
        try:
            CAMPAIGN.set_iteration_target(approval_iteration_target)
        except Exception:
            pass

    if (
        approval_iteration_target == "plate"
        and graph_gate_id == "T04"
        and bool(disk_fallback.get("interrupted_work"))
        and disk_fallback.get("run_dir") is not None
    ):
        approval_run_dir = disk_fallback.get("run_dir")
        approval_context["run_dir"] = approval_run_dir
        approval_context["source_kind"] = str(disk_fallback.get("source_kind") or "staging")
        try:
            self.app.update_status(
                (
                    "Wykryto przerwaną pracę Z2: "
                    f"{int(disk_fallback.get('unpromoted_approved_images', 0) or 0)} obrazów [OK] / "
                    f"{int(disk_fallback.get('unpromoted_approved_plates', 0) or 0)} tablic czeka na dopisanie do puli YOLO."
                ),
                "warning",
            )
        except Exception:
            pass

    if approval_iteration_target == "char":
        ready_source = {}
        try:
            ready_source = self._get_char_route_ready_source() or {}
        except Exception:
            ready_source = {}
        should_continue_characters = bool(
            approval_action == "continue_characters"
            or (approval_run_dir is None and ready_source)
        )
        if should_continue_characters:
            if not self._confirm_step2_without_current_iteration_contribution(
                approval_context=approval_context,
                source_state=ready_source,
            ):
                try:
                    self.app.update_status(
                        "Zatwierdzenie E2 przerwane. Dodaj nowe zatwierdzone tablice albo świadomie zatwierdź E2 bez nowego wkładu.",
                        "warning",
                    )
                except Exception:
                    pass
                return
            try:
                self._finish_step2_char_and_focus_step3(
                    ready_source,
                    approve_step2=True,
                )
            except Exception as e:
                logger.error(f"Nie udało się zatwierdzić E2 z badge wizarda (char source): {e}")
            return

    if approval_iteration_target == "plate" and approval_run_dir is None:
        if not self._confirm_step2_without_current_iteration_contribution(
            approval_context=approval_context,
        ):
            try:
                self.app.update_status(
                    "Zatwierdzenie E2 przerwane. Dodaj nowe zatwierdzone tablice albo świadomie zatwierdź E2 bez nowego wkładu.",
                    "warning",
                )
            except Exception:
                pass
            return

        try:
            pending_images = int(disk_fallback.get("unpromoted_approved_images", 0) or 0)
            pending_plates = int(disk_fallback.get("unpromoted_approved_plates", 0) or 0)
        except Exception:
            pending_images, pending_plates = 0, 0
        if pending_images > 0 or pending_plates > 0:
            pending_run_dir = disk_fallback.get("run_dir") or approval_context.get("run_dir")
            if pending_run_dir is None:
                try:
                    self.app.update_status(
                        f"Nie udało się dopisać zaległych [OK] z {graph_display_gate_id or graph_gate_id or 'bramki'} do puli YOLO: brak ścieżki runu.",
                        "warning",
                    )
                except Exception:
                    pass
                return
            promote = getattr(annotation_tab, "_promote_run_to_campaign_plate_approved_set", None)
            if callable(promote):
                try:
                    promote_result = dict(
                        promote(
                            Path(pending_run_dir),
                            force_parse_xml=True,
                            project_name=str(CAMPAIGN.get_active_project_name() or "").strip() or None,
                        )
                        or {}
                    )
                except Exception as exc:
                    logger.debug(f"Nie udało się rozliczyć przerwanej pracy T05 przed zatwierdzeniem: {exc}")
                    promote_result = {"ok": False, "reason": "exception"}
                if bool(promote_result.get("ok")):
                    try:
                        now = datetime.now().isoformat(timespec="seconds")
                        resolved_run_dir = Path(str(promote_result.get("run_dir") or pending_run_dir))
                        CAMPAIGN.upsert_iteration_state(
                            updates={
                                "t05_work_session": {
                                    "active": False,
                                    "state": "resolved",
                                    "resolved_at": now,
                                    "updated_at": now,
                                    "run_dir": str(resolved_run_dir.resolve()),
                                    "approved_images": int(pending_images),
                                    "approved_plates": int(pending_plates),
                                    "last_return_result": promote_result,
                                }
                            }
                        )
                    except Exception as exc:
                        logger.debug(f"Nie udało się domknąć znacznika przerwanej pracy T05: {exc}")
                    try:
                        CAMPAIGN.invalidate_step3_char_source_state_cache()
                    except Exception:
                        pass
                    try:
                        self.app.update_status(
                            (
                                f"Dopisano zaległe [OK] z {graph_display_gate_id or graph_gate_id or 'bramki'} do puli YOLO: "
                                f"{pending_images} obrazów / {pending_plates} tablic."
                            ),
                            "success",
                        )
                    except Exception:
                        pass
                else:
                    try:
                        self.app.update_status(
                            f"Nie udało się dopisać zaległych [OK] z {graph_display_gate_id or graph_gate_id or 'bramki'} do puli YOLO. Bramka nie została zatwierdzona.",
                            "warning",
                        )
                    except Exception:
                        pass
                    return
            return
        try:
            CAMPAIGN.approve_step2()
            CAMPAIGN.set_current_step(4)
            self.request_wizard_stage_focus(step_num=4)
            self._refresh_dashboard()
            self.app.open_controlled_tab("campaign")
            self.app.update_campaign_tab_access()
            if graph_gate_id == "T04":
                self.app.update_status(
                    "Bramka T04 została zatwierdzona. Projekt przeszedł do E4T, czyli treningu modelu tablic; bramka T06 zamknięcia iteracji pozostaje osobną decyzją.",
                    "info",
                )
            else:
                self.app.update_status(
                    "E2 zostało zatwierdzone na podstawie zatwierdzonego zbioru projektu. Etap E4T jest już odblokowany.",
                    "info",
                )
        except Exception as e:
            logger.error(f"Nie udało się zatwierdzić E2 z badge wizarda (project approved set): {e}")
        return

    if approval_run_dir is not None:
        try:
            current_run = getattr(annotation_tab, "current_annotation_run_dir", None)
            same_run_loaded = bool(
                current_run is not None
                and hasattr(annotation_tab, "_paths_equivalent")
                and annotation_tab._paths_equivalent(current_run, approval_run_dir)
            )
        except Exception:
            same_run_loaded = False
        if not same_run_loaded and hasattr(annotation_tab, "_restore_preview_from_annotation_run"):
            try:
                annotation_tab._restore_preview_from_annotation_run(Path(approval_run_dir))
                annotation_tab._refresh_step2_action_states()
            except Exception as e:
                logger.debug(f"Nie udało się odtworzyć runu E2 przed zatwierdzeniem z badge: {e}")

    try:
        annotation_tab._approve_annotation_stage()
    except Exception as e:
        logger.error(f"Nie udało się zatwierdzić E2 z badge wizarda: {e}")


def _approve_step1_from_wizard(self):
    if not CAMPAIGN.get_active_project_name():
        return
    try:
        self._apply_current_ingest_plan()
    except Exception as e:
        logger.error(f"Nie udało się zatwierdzić E1 z badge wizarda: {e}")


def _get_annotation_step2_view_model(self):
    try:
        current_iteration_path = normalize_iteration_path(CAMPAIGN.get_iteration_path())
    except Exception:
        current_iteration_path = ""
    cache_key = (
        str(CAMPAIGN.get_active_project_name() or "").strip(),
        int(CAMPAIGN.get_current_iteration_num() or 1),
        int(CAMPAIGN.get_current_step() or 1),
        str(CAMPAIGN.get_iteration_target() or "").strip().lower(),
        str(current_iteration_path or "").strip().lower(),
        str(CAMPAIGN.get_step1_status() or "").strip().lower(),
        str(CAMPAIGN.get_step2_status() or "").strip().lower(),
        str(CAMPAIGN.get_step3_status() or "").strip().lower(),
    )
    vm_cache = self._get_dashboard_cache_bucket("step2_view_models")
    cached = vm_cache.get(cache_key)
    if isinstance(cached, Step2ViewModel):
        return cached

    current_step = int(CAMPAIGN.get_current_step() or 1)
    iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
    step1_status = str(CAMPAIGN.get_step1_status() or "").strip().lower()
    step2_status = str(CAMPAIGN.get_step2_status() or "").strip().lower()
    step3_status = str(CAMPAIGN.get_step3_status() or "").strip().lower()
    if step1_status != "approved":
        locked_vm = Step2ViewModel(
            stage_key="step2",
            iteration_target=iteration_target,
            current_step=current_step,
            step2_status=step2_status,
            state="locked",
            title="E2. Tablice",
            summary="E2 odblokuje się dopiero po zatwierdzeniu E1.",
            details="W E1 musi być wskazany katalog zdjęć oraz wybrany tor iteracji. Bez zatwierdzenia E1 nie można uruchomić STEP2-P1.",
            primary_cta=None,
            secondary_cta=None,
        )
        vm_cache[cache_key] = locked_vm
        return locked_vm
    if iteration_target == "char" and current_iteration_path == "char_from_ready_plates":
        skipped_vm = Step2ViewModel(
            stage_key="step2",
            iteration_target=iteration_target,
            current_step=current_step,
            step2_status=step2_status,
            state=("done" if step2_status == "approved" else "skipped"),
            title="E2. Tablice",
            summary="Ten tor korzysta z istniejącego źródła tablic i pomija pracę w Z2.",
            details=(
                "Bramka T03 nie otwiera listy obrazów w Z2. Jeśli źródło tablic jest poprawnie "
                "wskazane w zasobach, zatwierdź bramkę i przejdź dalej do pracy nad znakami."
            ),
            primary_cta=None,
            secondary_cta=None,
        )
        vm_cache[cache_key] = skipped_vm
        return skipped_vm
    if (
        current_step == 2
        and not iteration_target
        and step1_status == "approved"
        and step2_status == "pending"
        and step3_status == "pending"
    ):
        lightweight_vm = Step2ViewModel(
            stage_key="step2",
            iteration_target="",
            current_step=current_step,
            step2_status=step2_status,
            state="needs_attention",
            title="E2. Tablice",
            summary="Brak wybranego toru iteracji.",
            details="Wybór toru należy do E1. Wróć do E1 i wybierz tor, zanim uruchomisz E2.",
            primary_cta=None,
            secondary_cta=None,
        )
        vm_cache[cache_key] = lightweight_vm
        return lightweight_vm

    annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
    if annotation_tab is None:
        return None

    getter = getattr(annotation_tab, "get_campaign_step2_view_model", None)
    if not callable(getter):
        return None

    vm_started = perf_counter()
    try:
        view_model = getter()
    except Exception as e:
        logger.debug(f"Nie udało się pobrac modelu widoku E2 z Z2: {e}")
        return None

    if isinstance(view_model, Step2ViewModel):
        vm_cache[cache_key] = view_model

    self._log_perf(
        "step2_view_model",
        vm_started,
        threshold_ms=20.0,
        extra=f"target={str(getattr(view_model, 'iteration_target', '') or '').strip() or '-'}",
    )
    return view_model


def _resolve_step3_wizard_action_command(self, action_id: str, *, context: dict | None = None):
    normalized = str(action_id or "").strip().lower()
    if not normalized:
        return None

    if normalized == "approve_step3":
        return lambda: campaign_graph_actions.execute_campaign_graph_action(self, "approve_step3")
    if normalized == "open_z3":
        return lambda: campaign_graph_actions.execute_campaign_graph_action(self, "open_z3")
    if normalized == "open_z2_step3_repair":
        return lambda: campaign_graph_actions.execute_campaign_graph_action(self, "open_z2_step3_repair")
    if normalized == "continue_z3":
        return lambda ctx=dict(context or {}): campaign_graph_actions.execute_campaign_graph_action(
            self,
            "continue_z3",
            payload={"context": ctx},
        )
    if normalized == "open_z3_detect":
        return lambda ctx=dict(context or {}): campaign_graph_actions.execute_campaign_graph_action(
            self,
            "open_z3_detect",
            payload={"context": ctx},
        )
    return None


def _get_step3_current_iteration_contribution_state(self, source_state: dict | None = None) -> dict:
    state = dict(source_state or {})
    if not state:
        try:
            state = dict(self._get_char_route_source_state() or {})
        except Exception:
            state = {}

    def _int_value(key: str) -> int:
        try:
            return max(0, int(state.get(key, 0) or 0))
        except Exception:
            return 0

    current_images = max(
        _int_value("current_images_with_plates"),
        _int_value("pending_images_with_plates"),
    )
    current_plates = max(
        _int_value("current_total_plates"),
        _int_value("pending_total_plates"),
    )

    try:
        iteration_stats = dict(CAMPAIGN.get_plate_approved_set_iteration_stats() or {})
    except Exception:
        iteration_stats = {}

    iteration_images = int(iteration_stats.get("images", 0) or 0)
    iteration_plates = int(iteration_stats.get("plates", 0) or 0)
    current_images = max(current_images, iteration_images)
    current_plates = max(current_plates, iteration_plates)

    # Backward compatible fallback for projects created before ApprovedSet stored
    # first_approved_iteration. If the current step2 run is the source of entries,
    # treat it as current work rather than warning too aggressively.
    if current_plates <= 0:
        try:
            current_run = str(CAMPAIGN.get_step2_staging_run() or "").strip()
        except Exception:
            current_run = ""
        if current_run:
            matched_images = 0
            matched_plates = 0
            try:
                for entry in CAMPAIGN.list_plate_approved_entries():
                    if not isinstance(entry, dict):
                        continue
                    approved_from_run = str(entry.get("approved_from_run", "") or "").strip()
                    if not self._campaign_paths_equivalent(approved_from_run, current_run):
                        continue
                    valid_plate_count = 0
                    for plate_entry in list(entry.get("plates") or []):
                        if not isinstance(plate_entry, dict):
                            continue
                        polygon = list(plate_entry.get("polygon") or [])
                        if len(polygon) >= 4:
                            valid_plate_count += 1
                    if valid_plate_count <= 0:
                        valid_plate_count = int(entry.get("plate_count", 0) or 0)
                    if valid_plate_count <= 0:
                        continue
                    matched_images += 1
                    matched_plates += int(valid_plate_count)
            except Exception:
                matched_images = 0
                matched_plates = 0
            current_images = max(current_images, matched_images)
            current_plates = max(current_plates, matched_plates)

    try:
        approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
    except Exception:
        approved_stats = {}

    project_images = max(
        _int_value("project_images_with_plates"),
        int(approved_stats.get("images", 0) or 0),
    )
    project_plates = max(
        _int_value("project_total_plates"),
        int(approved_stats.get("plates", 0) or 0),
    )
    total_images = max(_int_value("images_with_plates"), project_images)
    total_plates = max(_int_value("total_plates"), project_plates)

    return {
        "current_images": int(current_images),
        "current_plates": int(current_plates),
        "project_images": int(project_images),
        "project_plates": int(project_plates),
        "total_images": int(total_images),
        "total_plates": int(total_plates),
        "iteration": int(iteration_stats.get("iteration", CAMPAIGN.get_current_iteration_num() or 1) or 1),
    }


def _confirm_step3_without_current_iteration_contribution(self, contribution_state: dict | None = None) -> bool:
    contribution = dict(contribution_state or self._get_step3_current_iteration_contribution_state())
    current_plates = int(contribution.get("current_plates", 0) or 0)
    project_plates = int(contribution.get("project_plates", 0) or 0)
    total_plates = int(contribution.get("total_plates", 0) or 0)
    iteration = int(contribution.get("iteration", CAMPAIGN.get_current_iteration_num() or 1) or 1)

    if current_plates > 0 or project_plates <= 0 or total_plates <= 0:
        return True

    try:
        previous_char_model = dict(
            CAMPAIGN.get_latest_trained_project_model(
                "char",
                before_iteration=iteration,
                project_name=str(CAMPAIGN.get_active_project_name() or "").strip() or None,
            )
            or {}
        )
    except Exception:
        previous_char_model = {}

    previous_model_path = str(
        previous_char_model.get("path")
        or previous_char_model.get("best_weights")
        or previous_char_model.get("model_path")
        or ""
    ).strip()
    previous_model_exists = False
    if previous_model_path:
        try:
            previous_model_exists = Path(previous_model_path).exists()
        except Exception:
            previous_model_exists = True
    if not previous_model_exists:
        return True

    try:
        previous_iteration = int(
            previous_char_model.get("trained_iteration")
            or previous_char_model.get("iteration")
            or 0
        )
    except Exception:
        previous_iteration = 0
    previous_model_label = str(previous_char_model.get("name") or "").strip()
    if not previous_model_label and previous_model_path:
        previous_model_label = Path(previous_model_path).name
    previous_info = ""
    if previous_model_label:
        previous_info = f" Ostatni wynik modelu znaków: {previous_model_label}"
        previous_info += f" z iteracji {previous_iteration:03d}." if previous_iteration > 0 else "."

    message = (
        f"Iteracja {iteration:03d} spełnia T05 dzięki tablicom przygotowanym wcześniej, "
        "ale w tej iteracji nie dodano nowych tablic do pracy nad znakami.\n\n"
        "Ponieważ w projekcie istnieje już wcześniejszy wytrenowany model znaków, kolejny trening znaków "
        "użyje praktycznie tej samej puli wejściowej. To ma sens głównie wtedy, gdy zmieniasz konfigurację "
        "treningu, split, augmentację albo chcesz wykonać run kontrolny."
        f"{previous_info}\n\n"
        "Możesz kontynuować mimo to albo wrócić do T05 i dodać lub poprawić materiał znaków."
    )
    try:
        return bool(
            self.app.themed_confirm(
                "T05 bez nowych danych znaków",
                message,
                parent=self.frame,
                confirm_label="Kontynuuj do treningu znaków",
                cancel_label="Wróć do T05",
                tone="warning",
            )
        )
    except Exception:
        return True


def _approve_step3_from_wizard(self):
    if not CAMPAIGN.get_active_project_name():
        return
    if self._get_iteration_target() != "char":
        return

    char_tab = self.app.tabs.get("characters") if getattr(self.app, "tabs", None) else None
    if char_tab is None:
        return

    readiness = {}
    try:
        getter = getattr(char_tab, "_get_campaign_step3_training_readiness", None)
        if callable(getter):
            readiness = dict(getter() or {})
    except Exception as e:
        logger.debug(f"Nie udało się sprawdzic gotowosci zatwierdzenia E3: {e}")
        readiness = {}

    has_outputs = False
    try:
        checker = getattr(char_tab, "_has_any_step3_export_outputs", None)
        if callable(checker):
            has_outputs = bool(checker())
    except Exception as e:
        logger.debug(f"Nie udało się sprawdzic artefaktow E3: {e}")
        has_outputs = False

    try:
        contract_readiness = dict(self._detect_campaign_char_ready_dataset_state() or {})
    except Exception as e:
        logger.debug(f"Nie udało się sprawdzic kontraktu eksportu E3/PZ3: {e}")
        contract_readiness = {}
    readiness = contract_readiness
    has_outputs = bool(
        bool(readiness.get("ok"))
        and str(readiness.get("ready_dataset") or readiness.get("dataset_hint") or "").strip()
    )

    perfect_count = int(readiness.get("perfect_count", 0) or 0)
    if not (has_outputs and bool(readiness.get("ok")) and perfect_count > 0):
        message = str(readiness.get("message") or "").strip() or (
            "E3 nie jest jeszcze gotowe do zatwierdzenia. W PZ2 przygotuj co najmniej jedną tablicę "
            "ze statusem perfect, a potem wykonaj eksport w PZ3."
        )
        try:
            self.app.update_status(message, "warning")
        except Exception:
            pass
        try:
            self.app.themed_info(
                "E3 jeszcze niegotowe",
                message,
                parent=self.frame,
                tone="warning",
            )
        except Exception:
            pass
        return

    contribution_state = self._get_step3_current_iteration_contribution_state()
    if not self._confirm_step3_without_current_iteration_contribution(contribution_state):
        try:
            self.app.update_status(
                "Zatwierdzenie E3 przerwane. Dodaj nowe tablice albo świadomie zatwierdź E3 bez nowego wkładu.",
                "warning",
            )
        except Exception:
            pass
        return

    CAMPAIGN.approve_step3()
    if int(CAMPAIGN.get_current_step() or 3) < 4:
        CAMPAIGN.set_current_step(4)

    try:
        self.request_wizard_stage_focus(step_num=4)
    except Exception:
        pass

    self._rebuild_wizard_stage_ui()
    self._refresh_dashboard()
    self.app.update_campaign_tab_access()

    try:
        self.app.open_controlled_tab("campaign")
    except Exception:
        pass

    try:
        self.app.update_status(
            "E3 zostało zatwierdzone. Odblokowano E4Z, czyli węzeł treningu modelu znaków.",
            "success",
        )
    except Exception:
        pass


def _get_campaign_step3_view_model(
    self,
    *,
    current_step: int,
    iteration_target: str,
    project_completed: bool,
    step2_status: str,
    step3_status: str,
    char_ready_source: dict | None = None,
    char_repair_guidance: dict | None = None,
    char_step4_gate: dict | None = None,
) -> Step3ViewModel:
    normalized_target = self._normalize_iteration_target(iteration_target)
    normalized_step2_status = str(step2_status or "").strip().lower()
    normalized_step3_status = str(step3_status or "").strip().lower()
    step2_approved = normalized_step2_status == "approved"
    ready_source = dict(char_ready_source or {})
    repair_guidance = dict(char_repair_guidance or {})
    step4_gate = dict(char_step4_gate or {})
    step3_training_gate = {}
    if normalized_target == "char":
        try:
            char_tab = self.app.tabs.get("characters") if getattr(self.app, "tabs", None) else None
            getter = getattr(char_tab, "_get_campaign_step3_training_readiness", None)
            if callable(getter):
                step3_training_gate = dict(getter() or {})
        except Exception as e:
            logger.debug(f"Nie udało się pobrać bramki perfectów E3: {e}")
            step3_training_gate = {}
    step3_perfect_count = int(
        step3_training_gate.get(
            "perfect_count",
            step4_gate.get("perfect_count", 0),
        )
        or 0
    )
    has_ready_char_dataset = bool(
        normalized_target == "char"
        and step3_perfect_count > 0
        and bool(step4_gate.get("ok"))
        and (
            str(step4_gate.get("ready_dataset") or "").strip()
            or str(step4_gate.get("dataset_hint") or "").strip()
            or int(step4_gate.get("train_images", 0) or 0) > 0
            or int(step4_gate.get("val_images", 0) or 0) > 0
        )
    )
    can_approve_step3 = bool(
        normalized_target == "char"
        and has_ready_char_dataset
        and (
            normalized_step3_status == "ready"
            or (int(current_step or 0) == 3 and has_ready_char_dataset)
        )
    )
    char_step4_blocked = bool(
        normalized_target == "char"
        and (int(current_step or 0) >= 4 or normalized_step3_status == "approved")
        and not bool(step4_gate.get("ok", True))
    )

    vm = Step3ViewModel(
        stage_key="step3",
        iteration_target=normalized_target,
        current_step=int(current_step or 0),
        step3_status=normalized_step3_status or "pending",
        state="locked",
        title="E3. Znaki i gold pack",
        summary="Z3 odblokuje się po przygotowaniu i zatwierdzeniu tablic w Z2.",
        details="Najpierw domknij E2.",
        body_mode="",
        body_visible=False,
        primary_cta=None,
        secondary_cta=None,
    )

    if normalized_target == "plate":
        return Step3ViewModel(
            stage_key="step3",
            iteration_target=normalized_target,
            current_step=int(current_step or 0),
            step3_status=normalized_step3_status or "pending",
            state="skipped",
            title="E3. Znaki i gold pack",
            summary="Tor tablic pomija Z3.",
            details="Po zatwierdzeniu Z2 projekt przechodzi od razu do Z4.",
        )

    if not normalized_target:
        return Step3ViewModel(
            stage_key="step3",
            iteration_target=normalized_target,
            current_step=int(current_step or 0),
            step3_status=normalized_step3_status or "pending",
            state="locked",
            title="E3. Znaki i gold pack",
            summary="Najpierw wybierz tor iteracji w E1.",
            details="Z3 dotyczy wyłącznie toru znaków, a tor jest decyzją wejściową E1.",
        )

    primary_cta = Step2CtaViewModel(
        label="Otwórz Z3",
        command_id=("continue_z3" if ready_source else "open_z3"),
        command_context=(dict(ready_source) if ready_source else {}),
    )
    secondary_cta = None
    state = "locked"
    summary = "Z3 odblokuje się po przygotowaniu i zatwierdzeniu tablic w Z2."
    details = "Najpierw domknij E2."
    body_mode = ""
    body_visible = False

    if can_approve_step3:
        state = "ready"
        summary = "Etap 3 jest gotowy do zamknięcia albo dalszej pracy."
        details = (
            "Możesz zamknąć E3, oznaczyć więcej tablic w Z2 albo pracować dalej na znakach tablic w Z3."
        )
        primary_cta = Step2CtaViewModel(
            label="Oznacz więcej tablic",
            command_id="open_z2_step3_repair",
        )
        secondary_cta = Step2CtaViewModel(
            label="Pracuj na znakach tablic",
            command_id=("continue_z3" if ready_source else "open_z3"),
            command_context=(dict(ready_source) if ready_source else {}),
            tone="secondary",
        )
    elif normalized_step3_status == "needs_rework":
        primary_command_id = "open_z3_detect" if str(repair_guidance.get("mode") or "").strip().lower() == "z3_pz2_repair" else "open_z2_step3_repair"
        primary_context = dict(ready_source or {}) if primary_command_id == "open_z3_detect" else {}
        primary_cta = Step2CtaViewModel(
            label=str(repair_guidance.get("primary_label") or "Oznacz więcej tablic"),
            command_id=primary_command_id,
            command_context=primary_context,
        )
        secondary_label = str(repair_guidance.get("secondary_label") or "").strip()
        secondary_command_id = "open_z3_detect" if str(repair_guidance.get("mode") or "").strip().lower() == "z3_pz2_repair" else ""
        secondary_context = dict(ready_source or {})
        if secondary_label:
            if secondary_label.lower().startswith(("przygotuj więcej tablic", "oznacz więcej tablic")):
                secondary_command_id = "open_z2_step3_repair"
                secondary_context = {}
            secondary_cta = Step2CtaViewModel(
                label=secondary_label,
                command_id=secondary_command_id,
                command_context=secondary_context,
            )
        state = "needs_attention"
        summary = "Dane znaków wymagają korekty przed treningiem."
        details = str(repair_guidance.get("details") or "Najpierw przygotuj poprawna sciezke naprawy dla toru znaków.")
        body_mode = "step3_rework"
        body_visible = not bool(project_completed)
    elif normalized_step3_status == "approved" or int(current_step or 0) > 3:
        state = "done"
        summary = "Z3 zostało zatwierdzone. Dataset znaków jest gotowy do Z4."
        details = "Możesz wrócić do Z3 albo przejść dalej do budowy datasetu i treningu."
    if (normalized_step3_status == "approved" or int(current_step or 0) > 3) and char_step4_blocked:
        primary_command_id = "open_z3_detect" if str(repair_guidance.get("mode") or "").strip().lower() == "z3_pz2_repair" else "open_z2_step3_repair"
        primary_context = dict(ready_source or {}) if primary_command_id == "open_z3_detect" else {}
        primary_cta = Step2CtaViewModel(
            label=str(repair_guidance.get("primary_label") or "Oznacz więcej tablic"),
            command_id=primary_command_id,
            command_context=primary_context,
        )
        secondary_label = str(repair_guidance.get("secondary_label") or "").strip()
        secondary_command_id = "open_z3_detect" if str(repair_guidance.get("mode") or "").strip().lower() == "z3_pz2_repair" else ""
        secondary_context = dict(ready_source or {})
        if secondary_label:
            if secondary_label.lower().startswith(("przygotuj więcej tablic", "oznacz więcej tablic")):
                secondary_command_id = "open_z2_step3_repair"
                secondary_context = {}
            secondary_cta = Step2CtaViewModel(
                label=secondary_label,
                command_id=secondary_command_id,
                command_context=secondary_context,
            )
        state = "needs_attention"
        summary = "Z3 jest formalnie zatwierdzone, ale dataset znaków nadal wymaga poprawy."
        gate_msg = str(step4_gate.get("message") or "").strip()
        repair_msg = str(repair_guidance.get("details") or "").strip()
        if gate_msg and repair_msg:
            details = gate_msg + "\n\n" + repair_msg
        else:
            details = gate_msg or repair_msg or "Wróć do Z3 i popraw dataset znaków, zanim przejdziesz do Z4."
        body_mode = ""
        body_visible = False
    elif int(current_step or 0) == 3 and not can_approve_step3:
        if ready_source and step2_approved:
            state = "in_progress"
            summary = "Pracuj na znakach tablic w Z3."
            primary_cta = Step2CtaViewModel(
                label="Pracuj na znakach tablic",
                command_id="continue_z3",
                command_context=dict(ready_source),
            )
            secondary_label = str(repair_guidance.get("secondary_label") or "").strip()
            secondary_command_id = (
                "open_z3_detect"
                if str(repair_guidance.get("mode") or "").strip().lower() == "z3_pz2_repair"
                else ""
            )
            secondary_context = dict(ready_source or {})
            if secondary_label:
                if secondary_label.lower().startswith(("przygotuj więcej tablic", "oznacz więcej tablic")):
                    secondary_command_id = "open_z2_step3_repair"
                    secondary_context = {}
                secondary_cta = Step2CtaViewModel(
                    label=secondary_label,
                    command_id=secondary_command_id,
                    command_context=secondary_context,
                )
            if secondary_cta is None:
                secondary_cta = Step2CtaViewModel(
                    label="Oznacz więcej tablic",
                    command_id="open_z2_step3_repair",
                    command_context={},
                    tone="secondary",
                )
            primary_label = str(getattr(primary_cta, "label", "") or "").strip()
            secondary_label = str(getattr(secondary_cta, "label", "") or "").strip()
            details = (
                f"„{primary_label}” otwiera Z3 i przebudowuje dane znaków z aktualnych anotacji tablic. "
                f"„{secondary_label}” wraca do Z2, jeśli chcesz najpierw dopisać albo poprawić tablice."
            )
        elif str(repair_guidance.get("primary_label") or "").strip():
            primary_command_id = (
                "open_z3_detect"
                if str(repair_guidance.get("mode") or "").strip().lower() == "z3_pz2_repair"
                else "open_z2_step3_repair"
            )
            primary_context = dict(ready_source or {}) if primary_command_id == "open_z3_detect" else {}
            primary_cta = Step2CtaViewModel(
                label=str(repair_guidance.get("primary_label") or "Oznacz więcej tablic"),
                command_id=primary_command_id,
                command_context=primary_context,
            )
            secondary_label = str(repair_guidance.get("secondary_label") or "").strip()
            secondary_command_id = (
                "open_z3_detect"
                if str(repair_guidance.get("mode") or "").strip().lower() == "z3_pz2_repair"
                else ""
            )
            secondary_context = dict(ready_source or {})
            if secondary_label:
                if secondary_label.lower().startswith(("przygotuj więcej tablic", "oznacz więcej tablic")):
                    secondary_command_id = "open_z2_step3_repair"
                    secondary_context = {}
                secondary_cta = Step2CtaViewModel(
                    label=secondary_label,
                    command_id=secondary_command_id,
                    command_context=secondary_context,
                )
            state = "needs_attention"
            summary = "Źródło tablic dla toru znaków nadal wymaga uwagi."
            details = str(
                repair_guidance.get("details")
                or "Najpierw przygotuj więcej tablic w Z2 albo wróć do Z3, jeśli źródło jest już wystarczające."
            )
        else:
            state = "in_progress"
            summary = "Pracujesz teraz w Z3: wycinanie tablic, OCR, korekty i eksport."
            details = "Wizard pokazuje tylko stan etapu. Cała praca dzieje się w zakładce Znaki."
    elif step2_approved:
        state = "ready"
        summary = "Z3 jest gotowe do uruchomienia."
        details = "Źródła tablic są już przygotowane i możesz zacząć pracę nad znakami."
        if secondary_cta is None:
            secondary_cta = Step2CtaViewModel(
                label="Oznacz więcej tablic",
                command_id="open_z2_step3_repair",
                command_context={},
                tone="secondary",
            )

    if normalized_target == "char" and state in {"in_progress", "ready"} and not can_approve_step3:
        try:
            char_model_info = dict(
                CAMPAIGN.get_effective_project_model(
                    "char",
                    before_iteration=int(CAMPAIGN.get_current_iteration_num() or 1),
                )
                or {}
            )
        except Exception:
            char_model_info = {}
        char_model_path = str(char_model_info.get("path") or "").strip()
        if char_model_path and Path(char_model_path).exists():
            if state == "in_progress":
                details = (
                    "Wizard pokazuje tylko stan etapu. Cala praca dzieje się w zakładce Znaki, "
                    "a aktywny model znaków projektu jest tam podstawiany automatycznie."
                )
            elif state == "ready":
                details = (
                    "Źródła tablic są już przygotowane i możesz zacząć pracę nad znakami. "
                    "Po wejsciu do Z3 model znaków projektu będzie już ustawiony automatycznie."
                )

    if normalized_target == "char" and not step2_approved and state in {"in_progress", "ready"}:
        state = "locked"
        summary = "Z3 odblokuje się po przygotowaniu i zatwierdzeniu E2."
        details = "Gotowe źródło tablic nie wystarcza jeszcze do wejścia w E3. Najpierw formalnie zamknij E2."
        body_mode = ""
        body_visible = False
        primary_cta = None
        secondary_cta = None

    if normalized_target == "char" and int(current_step or 0) < 3 and not step2_approved:
        state = "locked"
        summary = "Z3 odblokuje się po przygotowaniu i zatwierdzeniu E2."
        details = "Najpierw przygotuj albo zatwierdź w E2 tablice dla toru znaków."
        body_mode = ""
        body_visible = False
        primary_cta = None
        secondary_cta = None

    if state == "done" and not bool(char_step4_blocked):
        primary_cta = None
        secondary_cta = None
        body_mode = ""
        body_visible = False

    if (
        normalized_target == "char"
        and not bool(char_step4_blocked)
        and (int(current_step or 0) >= 4 or normalized_step3_status == "approved")
    ):
        state = "done"
        primary_cta = None
        secondary_cta = None
        body_mode = ""
        body_visible = False

    return Step3ViewModel(
        stage_key="step3",
        iteration_target=normalized_target,
        current_step=int(current_step or 0),
        step3_status=normalized_step3_status or "pending",
        state=state,
        title="E3. Znaki i gold pack",
        summary=summary,
        details=details,
        body_mode=body_mode,
        body_visible=bool(body_visible),
        primary_cta=primary_cta,
        secondary_cta=secondary_cta,
    )


def _get_char_route_ready_source(self) -> dict:
    source_state = self._get_char_route_source_state()
    if not bool(source_state.get("ready")):
        return {}

    def _source_context_is_usable(context: dict | None) -> bool:
        if not isinstance(context, dict) or not context:
            return False
        try:
            restore_run = (
                context.get("restore_run_dir")
                or context.get("run_dir")
                or context.get("annotation_run_dir")
            )
            xml_value = (
                context.get("xml_path")
                or context.get("source_xml")
                or context.get("source_xml_path")
            )
            images_value = (
                context.get("input_dir")
                or context.get("images_dir")
                or context.get("source_images_dir")
            )
            xml_path = Path(str(xml_value or "").strip()) if str(xml_value or "").strip() else None
            if xml_path is None and str(restore_run or "").strip():
                xml_path = Path(str(restore_run).strip()) / "annotations.xml"
            images_dir = Path(str(images_value or "").strip()) if str(images_value or "").strip() else None
            return bool(
                xml_path is not None
                and xml_path.exists()
                and xml_path.is_file()
                and images_dir is not None
                and images_dir.exists()
                and images_dir.is_dir()
            )
        except Exception:
            return False

    bootstrap = source_state.get("bootstrap")
    if _source_context_is_usable(bootstrap):
        return dict(bootstrap)

    try:
        bundle = dict(CAMPAIGN.get_iteration_artifact_bundle(
            iteration_num=int(CAMPAIGN.get_current_iteration_num() or 1)
        ) or {})
    except Exception:
        bundle = {}
    char_effective = dict(bundle.get("char_effective_source") or {})
    if char_effective:
        try:
            effective_plate_model = dict(
                CAMPAIGN.get_effective_project_model(
                    "plate",
                    before_iteration=int(CAMPAIGN.get_current_iteration_num() or 1),
                )
                or {}
            )
        except Exception:
            effective_plate_model = {}
        plate_model_path = str(effective_plate_model.get("path") or "").strip()
        registry_context = {
            "restore_run_dir": char_effective.get("run_dir"),
            "input_dir": char_effective.get("images_dir"),
            "xml_path": char_effective.get("xml_path"),
            "input_source": "campaign_char_effective_source",
            "manual_template": False,
            "plate_model_path": plate_model_path,
            "display_name": str(char_effective.get("display_name") or "Zatwierdzony zbiór projektu tablic").strip(),
            "run_name": str(char_effective.get("display_name") or "").strip(),
            "contributor_run_dir": str(char_effective.get("contributor_run_dir") or "").strip(),
            "source_scope": "campaign_char_effective_source",
        }
        if _source_context_is_usable(registry_context):
            return registry_context

    annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
    builder = getattr(annotation_tab, "_build_campaign_char_effective_source", None)
    if callable(builder):
        try:
            effective_source = dict(builder() or {})
        except Exception as e:
            logger.debug(f"Nie udało się zbudować efektywnego źródła znaków z ApprovedSet: {e}")
            effective_source = {}
        if effective_source:
            try:
                run_dir = Path(str(effective_source.get("run_dir") or "").strip())
            except Exception:
                run_dir = None
            try:
                images_dir = Path(str(effective_source.get("images_dir") or "").strip())
            except Exception:
                images_dir = None
            try:
                xml_path = Path(str(effective_source.get("xml_path") or "").strip())
            except Exception:
                xml_path = None
            if (
                run_dir is not None
                and images_dir is not None
                and xml_path is not None
                and run_dir.exists()
                and images_dir.exists()
                and xml_path.exists()
            ):
                try:
                    effective_plate_model = dict(
                        CAMPAIGN.get_effective_project_model(
                            "plate",
                            before_iteration=int(CAMPAIGN.get_current_iteration_num() or 1),
                        )
                        or {}
                    )
                except Exception:
                    effective_plate_model = {}
                plate_model_path = str(effective_plate_model.get("path") or "").strip()
                return {
                    "restore_run_dir": run_dir,
                    "input_dir": images_dir,
                    "xml_path": xml_path,
                    "input_source": "campaign_char_effective_source",
                    "manual_template": False,
                    "plate_model_path": plate_model_path,
                    "display_name": str(
                        effective_source.get("display_name")
                        or "Zatwierdzony zbiór projektu tablic"
                    ).strip(),
                    "run_name": str(
                        effective_source.get("display_name")
                        or getattr(run_dir, "name", "")
                        or ""
                    ).strip(),
                    "contributor_run_dir": str(effective_source.get("contributor_run_dir") or "").strip(),
                }

    return dict(bootstrap) if isinstance(bootstrap, dict) else {}


def _get_char_training_split_preview(self, total_plates: int) -> dict:
    total = max(0, int(total_plates or 0))
    train_pct = 80.0
    val_pct = 10.0
    test_pct = 10.0

    try:
        char_tab = self.app.tabs.get("characters") if getattr(self.app, "tabs", None) else None
        if char_tab is not None and hasattr(char_tab, "_get_gold_export_split_percentages"):
            train_pct, val_pct, test_pct = char_tab._get_gold_export_split_percentages()
    except Exception:
        train_pct, val_pct, test_pct = 80.0, 10.0, 10.0

    train_count = int(total * (float(train_pct) / 100.0))
    val_count = int(total * (float(val_pct) / 100.0))
    test_count = max(0, total - train_count - val_count)

    return {
        "total_plates": total,
        "train_pct": float(train_pct),
        "val_pct": float(val_pct),
        "test_pct": float(test_pct),
        "train": int(train_count),
        "val": int(val_count),
        "test": int(test_count),
        "ok": bool(total > 0 and train_count > 0 and val_count > 0),
    }


def _looks_like_campaign_char_dataset_dir(self, dataset_dir: Path | None) -> bool:
    if dataset_dir is None:
        return False
    try:
        dataset_dir = Path(dataset_dir)
        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.exists():
            return False
        text = data_yaml.read_text(encoding="utf-8", errors="ignore")
        compact = "".join(text.lower().split())
        if "kpt_shape" in text.lower():
            return False
        return "nc:36" in compact
    except Exception:
        return False


def _path_is_inside_project_root(path_like, project_root) -> bool:
    if not path_like or not project_root:
        return True
    try:
        Path(path_like).resolve().relative_to(Path(project_root).resolve())
        return True
    except Exception:
        return False


def _active_campaign_project_context() -> tuple[str, Path | None]:
    try:
        project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
    except Exception:
        project_name = ""
    try:
        project_root = CAMPAIGN.get_active_project_root_dir() if project_name else None
    except Exception:
        project_root = None
    return project_name, project_root


def _payload_matches_campaign_project(
    payload: dict | None,
    *,
    project_name: str = "",
    project_root=None,
    extra_paths: list[object] | tuple[object, ...] | None = None,
) -> bool:
    if not project_name:
        return True
    data = dict(payload or {})
    payload_project = str(data.get("project", "") or "").strip()
    if payload_project and payload_project != project_name:
        return False
    path_values = list(extra_paths or [])
    for key in ("dataset_path", "gold_dataset_path", "summary_path", "summary_dir", "_summary_path", "_summary_dir"):
        raw = str(data.get(key) or "").strip()
        if raw:
            path_values.append(raw)
    for raw in path_values:
        if raw and not _path_is_inside_project_root(raw, project_root):
            return False
    return True


def _read_latest_campaign_step3_export_summary() -> dict:
    active_project, active_project_root = _active_campaign_project_context()
    try:
        chars_root = CAMPAIGN.get_dir("chars")
    except Exception:
        chars_root = None
    if chars_root is None:
        return {}
    try:
        root = Path(chars_root)
    except Exception:
        return {}
    if not root.exists():
        return {}
    if active_project and not _path_is_inside_project_root(root, active_project_root):
        return {}
    try:
        candidates = sorted(
            root.rglob("export_summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        return {}
    for candidate in candidates:
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(loaded, dict) or not loaded:
            continue
        dataset_path = str(loaded.get("gold_dataset_path") or "").strip()
        if not (
            bool(loaded.get("gold_dataset_created"))
            and bool(loaded.get("gold_dataset_valid", True))
            and dataset_path
        ):
            continue
        try:
            if not Path(dataset_path).exists():
                continue
        except Exception:
            continue
        loaded["_summary_path"] = str(candidate)
        loaded["_summary_dir"] = str(candidate.parent)
        if _payload_matches_campaign_project(
            loaded,
            project_name=active_project,
            project_root=active_project_root,
            extra_paths=(dataset_path, candidate, candidate.parent),
        ):
            return loaded
    return {}


def _detect_campaign_char_ready_dataset_state(self) -> dict:
    result = {
        "ok": False,
        "reason": "missing_char_dataset",
        "message": (
            "Brakuje eksportu datasetu znaków z PZ3. Sama anotacja boxów w PZ2 nie otwiera T05: "
            "po oznaczeniu znaków i uzyskaniu tablic perfect przejdź do PZ3 i wyeksportuj dataset znaków YOLO Detect."
        ),
        "ready_dataset": "",
        "dataset_hint": "",
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "perfect_count": 0,
        "validation_message": "",
    }

    try:
        active_project, active_project_root = _active_campaign_project_context()
        contracts = dict((CAMPAIGN.get_iteration_state() or {}).get("t06_contracts") or {})
        pz2_contract = dict(contracts.get("pz2_char_boxes") or {})
        pz3_contract = dict(contracts.get("pz3_char_dataset") or {})
        session = dict((CAMPAIGN.get_iteration_state() or {}).get("t06_work_session") or {})
        dataset_path_raw = str(pz3_contract.get("dataset_path") or "").strip()
        dataset_path = Path(dataset_path_raw) if dataset_path_raw else None
        pz3_reason = str(pz3_contract.get("reason") or "").strip().lower()
        if dataset_path_raw and not _payload_matches_campaign_project(
            pz3_contract,
            project_name=active_project,
            project_root=active_project_root,
            extra_paths=(dataset_path_raw,),
        ):
            result.update(
                reason="stale_char_dataset",
                message=(
                    "Kontrakt PZ3 wskazuje dataset spoza bieżącego projektu. "
                    "Wróć do pracy bramki i utwórz dataset znaków w PZ3 dla aktualnego projektu."
                ),
                ready_dataset=dataset_path_raw,
                dataset_hint=dataset_path_raw,
            )
            return result
        if pz3_reason == "approve_step3_backfill":
            result.update(
                reason="stale_char_dataset",
                message=(
                    "Wpis PZ3 powstał podczas próby zatwierdzenia, a nie podczas realnego eksportu datasetu. "
                    "Wróć do pracy bramki i utwórz dataset znaków w PZ3."
                ),
                ready_dataset=dataset_path_raw,
                dataset_hint=dataset_path_raw,
            )
            return result

        def _backfill_pz3_contract_from_summary(reason: str) -> dict:
            summary = _read_latest_campaign_step3_export_summary()
            if not summary:
                return {}
            summary_dataset_raw = str(summary.get("gold_dataset_path") or "").strip()
            try:
                summary_dataset = Path(summary_dataset_raw) if summary_dataset_raw else None
            except Exception:
                summary_dataset = None
            if (
                summary_dataset is None
                or not summary_dataset.exists()
                or not self._looks_like_campaign_char_dataset_dir(summary_dataset)
                or not _payload_matches_campaign_project(
                    summary,
                    project_name=active_project,
                    project_root=active_project_root,
                    extra_paths=(summary_dataset,),
                )
            ):
                return {}
            try:
                current_iteration = int(CAMPAIGN.get_current_iteration_num() or 1)
            except Exception:
                current_iteration = 1
            try:
                summary_iteration = int(
                    summary.get("source_iteration")
                    or summary.get("created_iteration")
                    or summary.get("produced_iteration")
                    or summary.get("iteration")
                    or 0
                )
            except Exception:
                summary_iteration = 0
            if summary_iteration <= 0 or summary_iteration != current_iteration:
                return {}
            now = datetime.now().isoformat(timespec="seconds")
            contract = {
                "fulfilled": True,
                "project": active_project,
                "product": "char_yolo_dataset",
                "source": "PZ3",
                "reason": str(reason or "summary_backfill"),
                "dataset_path": str(summary_dataset),
                "exportable_plate_count": int(summary.get("exportable_plate_count", 0) or 0),
                "exportable_char_count": int(summary.get("exportable_char_count", 0) or 0),
                "perfect_count": int(summary.get("perfect_count", summary.get("exportable_plate_count", 0)) or 0),
                "gold_dataset_valid": bool(summary.get("gold_dataset_valid", True)),
                "summary_path": str(summary.get("_summary_path") or ""),
                "summary_dir": str(summary.get("_summary_dir") or ""),
                "iteration": int(summary_iteration or 0),
                "source_iteration": int(summary_iteration or 0),
                "created_iteration": int(summary_iteration or 0),
                "fulfilled_at": now,
                "updated_at": now,
            }
            try:
                CAMPAIGN.upsert_iteration_state(updates={"t06_contracts": {"pz3_char_dataset": contract}})
            except Exception as exc:
                logger.debug(f"Nie udalo sie odbudowac kontraktu T06/PZ3 z summary: {exc}")
            return contract

        if not (bool(pz3_contract.get("fulfilled")) and dataset_path_raw):
            backfilled_contract = _backfill_pz3_contract_from_summary("summary_backfill")
            if backfilled_contract:
                pz3_contract = backfilled_contract
                dataset_path_raw = str(pz3_contract.get("dataset_path") or "").strip()
                dataset_path = Path(dataset_path_raw) if dataset_path_raw else None

        def _contract_time(payload: dict) -> float:
            for field in ("fulfilled_at", "updated_at", "interrupted_at", "started_at"):
                raw = str((payload or {}).get(field) or "").strip()
                if not raw:
                    continue
                try:
                    return float(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
                except Exception:
                    continue
            return 0.0

        pz3_time = _contract_time(pz3_contract)
        pz2_time = _contract_time(pz2_contract)
        session_time = _contract_time(session)
        session_state = str(session.get("state") or "").strip().lower()
        session_gate = str(session.get("working_gate_id") or "").strip().upper()
        char_work_gate_session_ids = {"T05", "T06"}
        session_substep = str(session.get("substep") or session.get("target_substep") or "").strip().lower()
        session_targets_pz2 = session_substep in {"2", "detect", "pz2", "z3_pz2"}
        session_targets_pz3 = session_substep in {"3", "dataset", "pz3", "z3_pz3"}
        session_active = bool(session.get("active")) or session_state in {"active", "started", "interrupted", "dirty"}

        def _contract_int(payload: dict, key: str) -> int:
            try:
                return int((payload or {}).get(key, 0) or 0)
            except Exception:
                return 0

        pz2_reason = str(pz2_contract.get("reason") or "").strip().lower()
        pz2_marker_only = pz2_reason in {"enter_pz3", "return_ready_to_graph", "dataset_exported"}
        pz2_same_export_scope = bool(
            _contract_int(pz2_contract, "exportable_plate_count") <= _contract_int(pz3_contract, "exportable_plate_count")
            and _contract_int(pz2_contract, "exportable_char_count") <= _contract_int(pz3_contract, "exportable_char_count")
        )
        stale_after_pz2 = bool(
            pz2_time
            and pz3_time
            and pz2_time > pz3_time + 0.001
            and not (session_gate in char_work_gate_session_ids and session_active and session_targets_pz2)
            and not pz2_marker_only
        )
        stale_after_session = bool(
            session_gate in char_work_gate_session_ids
            and session_active
            and session_state not in {"resolved", "closed", "complete", "completed"}
            and not session_targets_pz2
            and session_targets_pz3
            and (not pz3_time or not session_time or session_time > pz3_time + 0.001)
        )
        if stale_after_pz2 or stale_after_session:
            if stale_after_session and not stale_after_pz2:
                backfilled_contract = _backfill_pz3_contract_from_summary("summary_backfill_after_pz3_session")
                if backfilled_contract:
                    pz3_contract = backfilled_contract
                    dataset_path_raw = str(pz3_contract.get("dataset_path") or "").strip()
                    dataset_path = Path(dataset_path_raw) if dataset_path_raw else None
                    try:
                        previous_interrupted_at = str(session.pop("interrupted_at", "") or "").strip()
                        session.update(
                            {
                                "active": False,
                                "state": "completed",
                                "working_gate_id": "T05",
                                "closed_at": datetime.now().isoformat(timespec="seconds"),
                                "reason": "pz3_summary_ready",
                            }
                        )
                        if previous_interrupted_at:
                            session.setdefault("resolved_interrupted_at", previous_interrupted_at)
                        CAMPAIGN.upsert_iteration_state(updates={"t06_work_session": session})
                        CAMPAIGN.invalidate_step3_char_source_state_cache()
                        CAMPAIGN.clear_project_iteration_ui_snapshots()
                    except Exception as exc:
                        logger.debug(f"Nie udalo sie domknac sesji T06/Z3 po summary: {exc}")
                    stale_after_session = False
        if stale_after_pz2 or stale_after_session:
            result.update(
                reason="stale_char_dataset",
                message=(
                    "Eksport datasetu znaków nie obejmuje najnowszej pracy T05. "
                    "Wróć do pracy bramki i ponownie utwórz dataset znaków w PZ3."
                ),
                ready_dataset=dataset_path_raw,
                dataset_hint=dataset_path_raw,
            )
            return result
        if (
            bool(pz3_contract.get("fulfilled"))
            and dataset_path is not None
            and dataset_path.exists()
            and self._looks_like_campaign_char_dataset_dir(dataset_path)
            and _payload_matches_campaign_project(
                pz3_contract,
                project_name=active_project,
                project_root=active_project_root,
                extra_paths=(dataset_path,),
            )
        ):
            session_covered_by_dataset = bool(
                not session_targets_pz2
                or (pz3_time > 0.0 and (session_time <= 0.0 or session_time <= pz3_time + 0.001))
            )
            if (
                session_gate in char_work_gate_session_ids
                and session_active
                and str(session.get("work_area") or "").strip().lower() == "z3"
                and session_covered_by_dataset
                and session_state not in {"resolved", "closed", "complete", "completed"}
            ):
                try:
                    previous_interrupted_at = str(session.pop("interrupted_at", "") or "").strip()
                    session.update(
                        {
                            "active": False,
                            "state": "completed",
                            "working_gate_id": "T05",
                            "closed_at": datetime.now().isoformat(timespec="seconds"),
                            "reason": "pz3_dataset_ready",
                        }
                    )
                    if previous_interrupted_at:
                        session.setdefault("resolved_interrupted_at", previous_interrupted_at)
                    CAMPAIGN.upsert_iteration_state(updates={"t06_work_session": session})
                    CAMPAIGN.invalidate_step3_char_source_state_cache()
                    CAMPAIGN.clear_project_iteration_ui_snapshots()
                except Exception as exc:
                    logger.debug(f"Nie udalo sie domknac gotowej sesji T06/Z3: {exc}")
            result.update(
                ok=True,
                reason="source_dataset_ready_for_split",
                message="",
                ready_dataset=str(dataset_path),
                dataset_hint=str(dataset_path),
                exportable_char_count=int(pz3_contract.get("exportable_char_count", 0) or 0),
                exportable_plate_count=int(pz3_contract.get("exportable_plate_count", 0) or 0),
                perfect_count=int(
                    pz2_contract.get(
                        "perfect_count",
                        pz3_contract.get("exportable_plate_count", 0),
                    )
                    or 0
                ),
                validation_message="Kontrakt PZ3 T05 potwierdza wyeksportowany dataset znaków.",
            )
            return result
    except Exception as e:
        logger.debug(f"Nie udało się użyć kontraktu T06/PZ3 jako źródła datasetu znaków: {e}")

    return result


def _get_char_repair_guidance(self, source_state: dict | None = None) -> dict:
    state = dict(source_state or self._get_char_route_source_state() or {})
    images_with_plates = int(state.get("images_with_plates", 0) or 0)
    total_plates = int(state.get("total_plates", 0) or 0)
    run_name = str(state.get("run_name", "") or "").strip()
    if run_name:
        try:
            run_name = build_run_display_ref({"run_name": run_name}, kind_hint="annotation").id
        except Exception:
            pass
    source_scope = str(state.get("source_scope", "") or "").strip().lower()
    split_preview = self._get_char_training_split_preview(total_plates)
    source_context = dict(state.get("bootstrap") or {})

    result = {
        "mode": "z2_more_tables",
        "primary_label": "Oznacz więcej tablic",
        "primary_command": lambda: campaign_graph_actions.execute_campaign_graph_action(self, "open_z2_step3_repair"),
        "secondary_label": "",
        "secondary_command": None,
        "details": "",
        "split_preview": split_preview,
        "images_with_plates": images_with_plates,
        "total_plates": total_plates,
    }

    if source_scope == "step3_pending_union":
        source_intro = (
            f"Aktywny zbiór Z3/PZ2 wraz z nowymi [OK] z Z2 ({run_name})"
            if run_name
            else "Aktywny zbiór Z3/PZ2 wraz z nowymi [OK] z Z2"
        )
    elif source_scope == "campaign_approved_set":
        source_intro = "Zatwierdzony zbiór tablic projektu"
    elif source_scope == "step3_preview":
        source_intro = (
            f"Aktywny zbiór Z3/PZ2 ({run_name})"
            if run_name
            else "Aktywny zbiór Z3/PZ2"
        )
    else:
        source_intro = (
            f"Źródło {run_name}"
            if run_name
            else "Biezace źródło"
        )

    if split_preview.get("ok"):
        result.update(
            mode="z3_pz2_repair",
            primary_label="Popraw anotacje znaków w Z3/PZ2",
            primary_command=(
                lambda ctx=dict(source_context): campaign_graph_actions.execute_campaign_graph_action(
                    self,
                    "open_z3_detect",
                    payload={"context": ctx},
                )
            ),
            secondary_label="Oznacz więcej tablic",
            secondary_command=lambda: campaign_graph_actions.execute_campaign_graph_action(self, "open_z2_step3_repair"),
        )
        result["details"] = (
            f"{source_intro} zawiera {total_plates} tablic na {images_with_plates} oznaczonych obrazach. "
            f"To wystarczy, aby po poprawie znaków w Z3/PZ2 przygotować około "
            f"train={split_preview['train']}, val={split_preview['val']}, test={split_preview['test']}."
        )
        return result

    result["details"] = (
        f"{source_intro} zawiera tylko {total_plates} tablic na {images_with_plates} oznaczonych obrazach. "
        f"Przy rozkladzie {split_preview['train_pct']:.0f}/{split_preview['val_pct']:.0f}/{split_preview['test_pct']:.0f} "
        f"dostaniesz tylko train={split_preview['train']}, val={split_preview['val']}, test={split_preview['test']}. "
        "To znaczy, ze sama poprawa znaków w Z3/PZ2 nie wystarczy i najpierw trzeba przygotować więcej tablic w Z2."
    )
    return result


def _get_pending_step3_char_union_state(self) -> dict:
    try:
        return dict(CAMPAIGN.get_step3_char_source_state() or {})
    except Exception:
        return {}


def _get_char_route_source_state(self) -> dict:
    result = {
        "ready": False,
        "has_source": False,
        "needs_more_tables": False,
        "images_with_plates": 0,
        "total_plates": 0,
        "restore_run_dir": None,
        "run_name": "",
        "bootstrap": {},
    }
    if bool(getattr(self, "_project_open_lightweight_refresh", False)):
        try:
            current_iteration = int(CAMPAIGN.get_current_iteration_num() or 1)
            registry = CAMPAIGN.load_artifact_registry()
            packages = registry.get("packages", {}) if isinstance(registry, dict) else {}
            iteration_index = registry.get("iteration_index", {}) if isinstance(registry, dict) else {}
            package_id = str((iteration_index or {}).get(str(current_iteration), "") or "").strip()
            package = packages.get(package_id) if isinstance(packages, dict) else {}
        except Exception:
            package = {}
        if isinstance(package, dict) and package:
            route_hints = dict(package.get("route_hints") or {})
            char_effective = dict(package.get("char_effective_source") or {})
            plate_source = dict(package.get("plate_source") or {})
            step2_active_run = dict(package.get("step2_active_run") or {})
            images_with_plates = max(
                int(route_hints.get("images_with_plates", 0) or 0),
                int(char_effective.get("images_with_plates", 0) or 0),
                int(plate_source.get("images_with_plates", 0) or 0),
                int(step2_active_run.get("images_with_plates", 0) or 0),
            )
            total_plates = max(
                int(route_hints.get("total_plates", 0) or 0),
                int(char_effective.get("total_plates", 0) or 0),
                int(plate_source.get("total_plates", 0) or 0),
                int(step2_active_run.get("total_plates", 0) or 0),
            )
            min_char_plates = int(getattr(self, "STEP3_CHAR_MIN_PLATES", 10) or 10)
            run_dir = (
                str(char_effective.get("run_dir") or "").strip()
                or str(step2_active_run.get("run_dir") or "").strip()
                or str(plate_source.get("run_dir") or "").strip()
            )
            xml_path = (
                str(char_effective.get("xml_path") or "").strip()
                or str(step2_active_run.get("xml_path") or "").strip()
                or str(plate_source.get("xml_path") or "").strip()
            )
            images_dir = (
                str(char_effective.get("images_dir") or "").strip()
                or str(step2_active_run.get("images_dir") or "").strip()
                or str(plate_source.get("images_dir") or "").strip()
            )
            run_name = str(
                char_effective.get("display_name")
                or char_effective.get("run_name")
                or Path(run_dir).name
                or ""
            ).strip()
            has_source = bool(
                route_hints.get("char_has_source")
                or char_effective
                or run_dir
                or xml_path
                or images_dir
                or total_plates > 0
            )
            ready = bool(route_hints.get("char_ready") or total_plates >= min_char_plates)
            result.update(
                {
                    "ready": ready,
                    "has_source": has_source,
                    "needs_more_tables": bool(has_source and not ready),
                    "images_with_plates": int(images_with_plates),
                    "total_plates": int(total_plates),
                    "restore_run_dir": Path(run_dir) if run_dir else None,
                    "run_name": run_name,
                    "input_source": "artifact_registry",
                    "source_scope": str(char_effective.get("source_scope") or "artifact_registry").strip(),
                    "bootstrap": {
                        "restore_run_dir": run_dir,
                        "run_dir": run_dir,
                        "input_dir": images_dir,
                        "images_dir": images_dir,
                        "xml_path": xml_path,
                        "source_xml": xml_path,
                        "input_source": "artifact_registry",
                        "display_name": run_name,
                        "run_name": run_name,
                        "source_scope": str(char_effective.get("source_scope") or "artifact_registry").strip(),
                    },
                }
            )
        return result

    source_state = self._get_annotation_step2_source_state("char")
    if not source_state:
        source_state = {}
    result.update(source_state)

    pending_union_state = self._get_pending_step3_char_union_state()
    if pending_union_state:
        result["has_source"] = True
        result["source_scope"] = str(pending_union_state.get("source_scope") or "step3_pending_union").strip()
        result["images_with_plates"] = int(pending_union_state.get("images_with_plates", 0) or 0)
        result["total_plates"] = int(pending_union_state.get("total_plates", 0) or 0)
        result["run_name"] = str(pending_union_state.get("run_name") or result.get("run_name") or "").strip()
        result["project_images_with_plates"] = int(pending_union_state.get("project_images_with_plates", 0) or 0)
        result["project_total_plates"] = int(pending_union_state.get("project_total_plates", 0) or 0)
        result["current_images_with_plates"] = int(pending_union_state.get("current_images_with_plates", 0) or 0)
        result["current_total_plates"] = int(pending_union_state.get("current_total_plates", 0) or 0)
        result["preview_images_with_plates"] = int(pending_union_state.get("preview_images_with_plates", 0) or 0)
        result["preview_total_plates"] = int(pending_union_state.get("preview_total_plates", 0) or 0)
        result["pending_images_with_plates"] = int(pending_union_state.get("pending_images_with_plates", 0) or 0)
        result["pending_total_plates"] = int(pending_union_state.get("pending_total_plates", 0) or 0)
        min_char_plates = int(getattr(self, "STEP3_CHAR_MIN_PLATES", 10) or 10)
        result["ready"] = bool(
            int(result.get("total_plates", 0) or 0) >= min_char_plates
        )
        result["needs_more_tables"] = bool(
            result["has_source"] and not result["ready"]
        )
    return result


def _get_plate_route_ready_source(self) -> dict:
    if CAMPAIGN.get_step2_status() == "generated":
        return {}
    source_state = self._get_annotation_step2_source_state("plate")
    bootstrap = source_state.get("bootstrap")
    return dict(bootstrap) if isinstance(bootstrap, dict) else {}


def _get_step2_jump_button_text(self, target: str) -> str:
    target = self._normalize_iteration_target(target)

    if target == "plate":
        if CAMPAIGN.get_step2_status() == "generated":
            return "Sprawdź tablice w Z2"
        plate_bootstrap = self._get_annotation_bootstrap_for_target("plate")
        if not bool(plate_bootstrap.get("manual_template", True)):
            return "Przygotuj tablice na modelu projektu (Z2)"
        return "Przejdź do anotacji tablic"

    if target == "char":
        if CAMPAIGN.get_step2_status() == "generated":
            return "Sprawdź tablice w Z2"
        char_source_state = self._get_char_route_source_state()
        preflight = self._get_step1_char_route_preflight_state()
        plate_source_state = self._get_annotation_step2_source_state("plate")
        plate_model_ready = bool(plate_source_state.get("plate_model_ready"))
        if bool(char_source_state.get("ready")) and bool(preflight.get("material_ready")):
            return "Pracuj na znakach tablic"
        if bool(char_source_state.get("has_source")):
            return "Uzupełnij tablice w Z2"
        if plate_model_ready:
            return "Przygotuj tablice na modelu projektu (Z2)"
        return "Przejdź do anotacji tablic"

    return "Najpierw wybierz tor"


def _iteration_target_label(target: str) -> str:
    if target == "plate":
        return "tor tablic"
    if target == "char":
        return "tor znaków"
    return "nie wybrano toru"


def _iteration_target_button_label(self, route: str, *, current_target: str = "", last_target: str = "") -> str:
    route = self._normalize_iteration_target(route)
    current_target = self._normalize_iteration_target(current_target)
    last_target = self._normalize_iteration_target(last_target)

    if route == "plate":
        if not current_target and last_target == "plate":
            return "Tor tablic (kontynuuj)"
        return "Tor tablic"

    if route == "char":
        if not current_target and last_target == "char":
            return "Tor znaków (kontynuuj)"
        return "Tor znaków"

    return ""


def _get_iteration_target_lock_reason(self) -> str:
    try:
        if bool(CAMPAIGN.is_t02_at_review_committed_current_iteration()):
            return (
                "Ta iteracja ma już zapisaną kontrolę AT w T02. "
                "Drugi tor będzie dostępny dopiero w kolejnej iteracji albo po osobnym cofnięciu tej kontroli."
            )
    except Exception:
        pass

    current_target = self._get_iteration_target()
    if current_target not in {"plate", "char"}:
        return ""

    active_route_label = self._iteration_target_button_label(current_target)
    if not active_route_label:
        active_route_label = (
            "Tor tablic"
            if current_target == "plate"
            else "Tor znaków"
        )

    current_step = int(CAMPAIGN.get_current_step() or 1)
    step1_status = str(CAMPAIGN.get_step1_status() or "").strip().lower()
    step2_status = str(CAMPAIGN.get_step2_status() or "").strip().lower()

    if step1_status == "approved":
        return (
            f"Tor tej iteracji: „{active_route_label}”. "
            "Drugi tor wybierzesz dopiero w kolejnej iteracji."
        )

    if current_step > 2:
        return (
            f"Ta iteracja trwa już w „{active_route_label}”. "
            "Drugi tor będzie dostępny dopiero w nowej iteracji."
        )

    if step2_status in {"approved"}:
        return (
            f"Ta iteracja trwa już w „{active_route_label}”. "
            "Drugi tor będzie dostępny dopiero w nowej iteracji."
        )

    return ""


def _get_step1_char_preflight_image_count(self) -> int:
    try:
        plan = dict(self.current_ingest_plan or {})
    except Exception:
        plan = {}

    if plan:
        try:
            selected_total = int(plan.get("selected_total", 0) or 0)
        except Exception:
            selected_total = 0
        if selected_total > 0:
            return selected_total
        try:
            selected_items = list(plan.get("selected") or [])
            if selected_items:
                return int(len(selected_items))
        except Exception:
            pass

    try:
        summary = dict(CAMPAIGN.load_latest_ingest_plan_summary() or {})
    except Exception:
        summary = {}
    if summary:
        try:
            summary_iter = int(summary.get("iteration", 0) or 0)
        except Exception:
            summary_iter = 0
        summary_project = str(summary.get("project", "") or "").strip()
        current_project = str(CAMPAIGN.get_active_project_name() or "").strip()
        try:
            current_iter = int(CAMPAIGN.get_current_iteration_num() or 1)
        except Exception:
            current_iter = 1
        if summary_iter == current_iter and (not summary_project or summary_project == current_project):
            for key in ("selected_total", "raw_total", "source_new_to_project_total"):
                try:
                    count = int(summary.get(key, 0) or 0)
                except Exception:
                    count = 0
                if count > 0:
                    return count

    try:
        image_source = dict(self._get_project_start_effective_images_source() or {})
    except Exception:
        image_source = {}

    counts = []
    for key in ("iteration_count", "effective_count", "master_count"):
        try:
            counts.append(int(image_source.get(key, 0) or 0))
        except Exception:
            counts.append(0)
    return max(counts) if counts else 0


def _count_plate_annotations_in_xml(xml_path: Path | str | None) -> int:
    try:
        path = Path(xml_path) if xml_path is not None else None
    except Exception:
        path = None
    if path is None or not path.exists() or not path.is_file():
        return 0

    try:
        root = ET.parse(path).getroot()
    except Exception:
        return 0

    negative_label_parts = ("vehicle", "car", "pojazd")
    positive_label_parts = ("plate", "tablic")
    count = 0
    for image_node in root.findall(".//image"):
        for det_node in list(image_node):
            tag_name = str(getattr(det_node, "tag", "") or "").strip().lower()
            if tag_name not in {"polygon", "box"}:
                continue
            label = str(det_node.get("label", "") or "").strip().lower()
            if label and any(part in label for part in negative_label_parts):
                continue
            if not label or any(part in label for part in positive_label_parts):
                count += 1
    return int(count)


def _count_step1_imported_plate_annotations(self) -> int:
    try:
        plate_source_info = dict(self._get_project_start_plate_source_info() or {})
    except Exception:
        plate_source_info = {}
    if str(plate_source_info.get("source_mode") or "").strip().lower() != "approved":
        return 0

    candidates: list[Path] = []
    xml_path = str(plate_source_info.get("xml_path") or "").strip()
    run_path = str(plate_source_info.get("run_path") or "").strip()
    if xml_path:
        try:
            candidates.append(Path(xml_path))
        except Exception:
            pass
    if run_path:
        try:
            candidates.append(Path(run_path) / "annotations.xml")
        except Exception:
            pass

    for candidate in candidates:
        count = self._count_plate_annotations_in_xml(candidate)
        if count > 0:
            return count
    return 0


def _get_step1_char_route_preflight_state(self) -> dict:
    min_images = int(getattr(self, "STEP1_CHAR_MIN_IMAGES", 10) or 10)
    min_plates = int(getattr(self, "STEP3_CHAR_MIN_PLATES", 10) or 10)
    image_count = self._get_step1_char_preflight_image_count()

    try:
        approved_stats = dict(self._get_plate_approved_set_stats() or {})
    except Exception:
        approved_stats = {}
    approved_plate_count = int(approved_stats.get("plates", 0) or 0)
    approved_image_count = int(approved_stats.get("images", 0) or 0)

    if bool(getattr(self, "_project_open_lightweight_refresh", False)):
        route_hints = {}
        char_effective = {}
        try:
            current_iteration = int(CAMPAIGN.get_current_iteration_num() or 1)
            registry = CAMPAIGN.load_artifact_registry()
            packages = registry.get("packages", {}) if isinstance(registry, dict) else {}
            iteration_index = registry.get("iteration_index", {}) if isinstance(registry, dict) else {}
            package_id = str((iteration_index or {}).get(str(current_iteration), "") or "").strip()
            package = packages.get(package_id) if isinstance(packages, dict) else {}
            if isinstance(package, dict):
                route_hints = dict(package.get("route_hints") or {})
                char_effective = dict(package.get("char_effective_source") or {})
        except Exception:
            route_hints = {}
            char_effective = {}

        source_plate_count = max(
            int(route_hints.get("total_plates", 0) or 0),
            int(char_effective.get("total_plates", 0) or 0),
            approved_plate_count,
        )
        source_image_count = max(
            int(route_hints.get("images_with_plates", 0) or 0),
            int(char_effective.get("images_with_plates", 0) or 0),
            approved_image_count,
        )
        imported_plate_count = 0
        plate_material_count = max(source_plate_count, approved_plate_count, imported_plate_count)
        material_ready = bool(plate_material_count >= min_plates)
        image_potential = bool(image_count >= min_images)
        allow_route = bool(material_ready or image_potential)
        if material_ready:
            mode = "material_ready"
        elif image_potential:
            mode = "image_potential"
        else:
            mode = "blocked"
        if allow_route:
            block_reason = ""
        elif image_count <= 0 and plate_material_count <= 0:
            block_reason = (
                "Najpierw wybierz katalog zdjec w E1 albo dodaj zgodne anotacje tablic. "
                f"Tor znakow potrzebuje co najmniej {min_images} obrazow na start lub "
                f"{min_plates} gotowych tablic z wczesniejszych albo importowanych anotacji."
            )
        else:
            block_reason = (
                f"Tor znakow nie ma jeszcze bezpiecznego minimum wejsciowego: E1 widzi {image_count} obrazow "
                f"oraz {plate_material_count} gotowych tablic. Wybierz co najmniej {min_images} obrazow "
                f"albo dostarcz {min_plates} gotowych tablic."
            )
        return {
            "mode": mode,
            "allow_route": allow_route,
            "block_reason": block_reason,
            "image_count": int(image_count),
            "min_images": int(min_images),
            "plate_material_count": int(plate_material_count),
            "source_plate_count": int(source_plate_count),
            "source_image_count": int(source_image_count),
            "approved_plate_count": int(approved_plate_count),
            "approved_image_count": int(approved_image_count),
            "imported_plate_count": int(imported_plate_count),
            "min_plates": int(min_plates),
            "material_ready": material_ready,
            "image_potential": image_potential,
        }

    try:
        char_source_state = dict(self._get_char_route_source_state() or {})
    except Exception:
        char_source_state = {}
    source_plate_count = max(
        int(char_source_state.get("total_plates", 0) or 0),
        int(char_source_state.get("project_total_plates", 0) or 0),
        int(char_source_state.get("current_total_plates", 0) or 0),
        int(char_source_state.get("pending_total_plates", 0) or 0),
        approved_plate_count,
    )
    source_image_count = max(
        int(char_source_state.get("images_with_plates", 0) or 0),
        int(char_source_state.get("project_images_with_plates", 0) or 0),
        int(char_source_state.get("current_images_with_plates", 0) or 0),
        int(char_source_state.get("pending_images_with_plates", 0) or 0),
        approved_image_count,
    )

    if bool(getattr(self, "_project_open_lightweight_refresh", False)):
        imported_plate_count = 0
    else:
        imported_plate_count = self._count_step1_imported_plate_annotations()
    plate_material_count = max(source_plate_count, approved_plate_count, imported_plate_count)
    material_ready = bool(plate_material_count >= min_plates)
    image_potential = bool(image_count >= min_images)
    allow_route = bool(material_ready or image_potential)

    if material_ready:
        mode = "material_ready"
    elif image_potential:
        mode = "image_potential"
    else:
        mode = "blocked"

    if allow_route:
        block_reason = ""
    elif image_count <= 0 and plate_material_count <= 0:
        block_reason = (
            "Najpierw wybierz katalog zdjęć w E1 albo dodaj zgodne anotacje tablic. "
            f"Tor znaków potrzebuje co najmniej {min_images} obrazów na start lub "
            f"{min_plates} gotowych tablic z wcześniejszych albo importowanych anotacji."
        )
    else:
        block_reason = (
            f"Tor znaków nie ma jeszcze bezpiecznego minimum wejściowego: E1 widzi {image_count} obrazów "
            f"oraz {plate_material_count} gotowych tablic. Wybierz co najmniej {min_images} obrazów "
            f"albo dostarcz {min_plates} gotowych tablic."
        )

    return {
        "mode": mode,
        "allow_route": allow_route,
        "block_reason": block_reason,
        "image_count": int(image_count),
        "min_images": int(min_images),
        "plate_material_count": int(plate_material_count),
        "source_plate_count": int(source_plate_count),
        "source_image_count": int(source_image_count),
        "approved_plate_count": int(approved_plate_count),
        "approved_image_count": int(approved_image_count),
        "imported_plate_count": int(imported_plate_count),
        "min_plates": int(min_plates),
        "material_ready": material_ready,
        "image_potential": image_potential,
    }


def _get_step1_char_preflight_signature(self, state: dict | None = None) -> tuple:
    state = dict(state or self._get_step1_char_route_preflight_state() or {})
    try:
        iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        iteration_num = 1
    return (
        str(CAMPAIGN.get_active_project_name() or "").strip(),
        int(iteration_num),
        int(state.get("image_count", 0) or 0),
        int(state.get("plate_material_count", 0) or 0),
        int(state.get("imported_plate_count", 0) or 0),
    )


def _confirm_step1_char_preflight_warning(self, state: dict | None = None) -> bool:
    state = dict(state or self._get_step1_char_route_preflight_state() or {})
    if str(state.get("mode") or "").strip() != "image_potential":
        return True

    signature = self._get_step1_char_preflight_signature(state)
    if getattr(self, "_step1_char_preflight_ack_signature", None) == signature:
        return True

    image_count = int(state.get("image_count", 0) or 0)
    min_plates = int(state.get("min_plates", self.STEP3_CHAR_MIN_PLATES) or self.STEP3_CHAR_MIN_PLATES)
    message = (
        f"E1 widzi {image_count} obrazów, ale nie ma jeszcze {min_plates} gotowych tablic do pracy w E3.\n\n"
        "Możesz wybrać tor znaków, ale E2/Z2 będzie wtedy etapem przygotowania tablic. "
        "Dopiero po oznaczeniu i zatwierdzeniu odpowiedniej liczby obrazów program będzie miał realny materiał "
        "do wycinania tablic i dalszej pracy nad znakami.\n\n"
        "Jeśli okaże się, że wybrany katalog zdjęć nie pozwala uzyskać minimum, wrócisz do E1 po większy katalog "
        "albo do E2/Z2, żeby przygotować więcej tablic."
    )
    try:
        confirmed = self.app.themed_confirm(
            "Tor znaków: preflight E1",
            message,
            parent=self.frame,
            confirm_label="Wybierz tor znaków",
            cancel_label="Zostań w E1",
            tone="warning",
        )
    except Exception:
        confirmed = messagebox.askyesno("Tor znaków: preflight E1", message, parent=self.frame)
    if confirmed:
        self._step1_char_preflight_ack_signature = signature
    return bool(confirmed)


def _validate_step1_char_preflight_for_approval(self) -> bool:
    if self._get_iteration_target() != "char":
        return True

    state = self._get_step1_char_route_preflight_state()
    if not bool(state.get("allow_route")):
        message = str(state.get("block_reason") or "").strip()
        if not message:
            message = (
                "Tor znaków wymaga wybranego katalogu zdjęć albo gotowych anotacji tablic. "
                "Uzupełnij E1 przed zatwierdzeniem."
            )
        try:
            self.app.themed_info(
                "E1 nie jest gotowe dla toru znaków",
                message,
                parent=self.frame,
                tone="warning",
            )
        except Exception:
            messagebox.showwarning("E1 nie jest gotowe dla toru znaków", message, parent=self.frame)
        return False

    return self._confirm_step1_char_preflight_warning(state)


def _format_step1_char_route_card_body(self, state: dict | None = None, block_reason: str = "") -> str:
    state = dict(state or self._get_step1_char_route_preflight_state() or {})
    if block_reason:
        return block_reason

    min_plates = int(state.get("min_plates", self.STEP3_CHAR_MIN_PLATES) or self.STEP3_CHAR_MIN_PLATES)
    min_images = int(state.get("min_images", getattr(self, "STEP1_CHAR_MIN_IMAGES", 10)) or 10)
    image_count = int(state.get("image_count", 0) or 0)
    plate_material_count = int(state.get("plate_material_count", 0) or 0)
    if bool(state.get("material_ready")):
        return (
            f"Cel: model znaków. Warunek wejścia: istniejące źródło tablic ({plate_material_count}/{min_plates}). "
            "Po zatwierdzeniu E1 możesz przejść przez E2 do wyodrębniania tablic i pracy nad znakami."
        )
    if bool(state.get("image_potential")):
        return (
            f"Cel: model znaków. Warunek wejścia: katalog zdjęć ({image_count}/{min_images}). "
            f"W E2/Z2 przygotujesz minimum {min_plates} tablic potrzebnych do E3/Z3."
        )
    return (
        f"Cel: model znaków. Brakuje katalogu zdjęć ({image_count}/{min_images}) albo gotowych tablic "
        f"({plate_material_count}/{min_plates}). Tor możesz wybrać teraz, ale E1 zatwierdzisz dopiero po spełnieniu minimum."
    )


def _get_step1_char_route_block_reason(self) -> str:
    try:
        if (
            str(CAMPAIGN.get_step1_status() or "").strip().lower() == "approved"
            or int(CAMPAIGN.get_current_step() or 1) > 1
        ):
            return ""
    except Exception:
        return ""

    state = self._get_step1_char_route_preflight_state()
    return "" if bool(state.get("allow_route")) else str(state.get("block_reason") or "")
