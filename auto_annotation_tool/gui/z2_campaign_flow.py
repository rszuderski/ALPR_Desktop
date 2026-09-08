from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import logger
from ..campaign_iteration_paths import normalize_iteration_path
from .z2_flow_models import (
    Z2CampaignRuntimeState,
    Z2CopyPayload,
    Z2CtaState,
    Z2LayoutState,
    Z2LeftPanelCopyContext,
)
from .z2_shared_ui import (
    campaign_gate_id_for_edge,
    campaign_visible_gate_id,
    is_campaign_t02_at_review_context,
)

if TYPE_CHECKING:
    from .tab_annotation import AnnotationTab


CHAR_WORK_GATE_DISPLAY_ID = "T05"


def apply_campaign_step2_workflow_preset(
    host: "AnnotationTab",
    *,
    iteration_target: str | None = None,
    manual_template: bool | None = None,
) -> None:
    if host._is_free_mode_session_context():
        return

    target = str(iteration_target or "").strip().lower()
    if target not in {"plate", "char"}:
        try:
            from ..campaign_manager import CAMPAIGN
            target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        except Exception:
            target = ""
    if target not in {"plate", "char"}:
        target = "plate"

    input_ready = bool(str(host.input_dir_var.get() or "").strip())
    campaign_default_manual = False
    try:
        from ..campaign_manager import CAMPAIGN

        campaign_default_manual = bool(
            CAMPAIGN.get_active_project_name()
            and int(CAMPAIGN.get_current_step() or 0) == 2
            and target in {"plate", "char"}
        )
    except Exception:
        campaign_default_manual = False

    if manual_template is not None:
        use_manual_route = bool(manual_template)
    else:
        use_manual_route = bool(campaign_default_manual or (target == "plate"))

    host._manual_review_active = False
    host._manual_review_from_auto = False
    host._manual_review_export_ready = False
    host._dataset_export_completed = False
    host._last_completed_workflow_route = ""
    host._auto_route_settings_pending = False
    host._reset_campaign_step2_runtime_state()

    if use_manual_route:
        host._set_workflow_route_state("manual", campaign_context=True)
        host._set_manual_entry_mode_state("new", campaign_context=True)
        host.manual_xml_template_var.set(True)
        try:
            host.manual_vehicle_assist_var.set(False)
        except Exception:
            pass
        host._set_workflow_step_state("manual_start" if input_ready else "manual_input", campaign_context=True)
    else:
        host._set_workflow_route_state("auto", campaign_context=True)
        host.manual_xml_template_var.set(False)
        try:
            auto_choice = "use" if str(host.mode_var.get() or "").strip() == "C: Pojazdy + tablice" else "skip"
            host._set_auto_vehicle_choice_state(auto_choice, campaign_context=True)
        except Exception:
            pass
        next_step = "auto_input" if not input_ready else "auto_start"
        host._set_workflow_step_state(next_step, campaign_context=True)
    try:
        host._set_screen_state("workflow", campaign_context=True)
    except Exception:
        pass

    host._refresh_left_panel_route_copy()
    host._refresh_detection_configuration_ui()
    host._refresh_step2_action_states()
    host._refresh_free_mode_workflow_ui()


def open_existing_run_for_campaign_review(
    host: "AnnotationTab",
    run_dir: Path | None = None,
    *,
    iteration_target: str | None = None,
    manual_template: bool = False,
    defer_ui_restore: bool = False,
) -> bool:
    target_run_dir = host._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    if target_run_dir is None:
        return False

    normalized_target = str(iteration_target or "").strip().lower()
    repair_mode = bool(
        (normalized_target == "char" and host._is_campaign_char_repair_return_mode())
        or (normalized_target == "plate" and host._is_campaign_plate_step4_repair_return_mode())
    )
    if defer_ui_restore:
        host.current_annotation_run_dir = target_run_dir
        host.current_annotation_xml_path = target_run_dir / "annotations.xml"
        host.last_staging_run_dir = target_run_dir
        try:
            host.plate_dataset_run_var.set(str(target_run_dir))
        except Exception:
            pass
        host._campaign_deferred_run_restore_in_progress = True
        host._campaign_deferred_run_restore_payload_applied = False
        host._campaign_deferred_run_restore_target_dir = str(target_run_dir)
        apply_campaign_step2_workflow_preset(
            host,
            iteration_target=iteration_target,
            manual_template=manual_template,
        )
    else:
        if not host._restore_preview_from_annotation_run(
            target_run_dir,
            defer_ui_restore=False,
        ):
            return False

        apply_campaign_step2_workflow_preset(
            host,
            iteration_target=iteration_target,
            manual_template=manual_template,
        )
    if manual_template:
        host._set_workflow_route_state("manual", campaign_context=True)
        host._set_manual_entry_mode_state("continue", campaign_context=True)
        host._manual_review_active = True
        host._manual_review_from_auto = False
        host._manual_review_origin_route = "manual"
        host._manual_review_export_ready = False
        host._dataset_export_completed = False
        host._last_completed_workflow_route = ""
        host._auto_route_settings_pending = False
        host._set_workflow_step_state("manual_history", campaign_context=True)
    else:
        host._set_workflow_route_state("auto", campaign_context=True)
        host.manual_xml_template_var.set(False)
        host._set_workflow_step_state("auto_start", campaign_context=True)
        host._manual_review_active = False
        host._manual_review_from_auto = False
        host._manual_review_origin_route = ""
        host._manual_review_export_ready = False
        host._dataset_export_completed = False
        host._last_completed_workflow_route = "" if repair_mode else "auto"
        host._auto_route_settings_pending = False
    host._refresh_left_panel_route_copy()
    host._refresh_detection_configuration_ui()
    host._refresh_run_output_info()
    if not defer_ui_restore:
        host._refresh_plate_dataset_export_sources()
        host._refresh_preview_list_summary()
        host._refresh_step2_action_states()
    else:
        host._refresh_step2_action_states()
    host._refresh_free_mode_workflow_ui()
    return True


def open_campaign_step2_entry(
    host: "AnnotationTab",
    *,
    iteration_target: str | None = None,
    entry_strategy: str | None = None,
    restore_preview: bool = True,
    open_existing_run: bool = True,
    defer_preview_load: bool = False,
    source_context: dict | None = None,
) -> dict[str, Any]:
    open_started = time.perf_counter()
    phase_started = open_started
    slow_phase_notes: list[str] = []

    def _mark_phase(label: str) -> None:
        nonlocal phase_started
        now = time.perf_counter()
        elapsed_ms = (now - phase_started) * 1000.0
        if elapsed_ms >= 250.0:
            slow_phase_notes.append(f"{label}={elapsed_ms:.0f}ms")
        phase_started = now

    host._begin_campaign_step2_transition()
    try:
        from ..campaign_manager import CAMPAIGN

        active_project = CAMPAIGN.get_active_project_name()
        incoming_context = dict(source_context or {})
        t02_at_review = bool(
            str(incoming_context.get("z2_work_mode") or "").strip().lower() == "t02_at_review"
            or (
                str(incoming_context.get("graph_edge_key") or "").strip() == "e1_to_e3"
                and str(incoming_context.get("graph_gate_id") or "").strip().upper() == "T02"
            )
        )
        if not active_project or (int(CAMPAIGN.get_current_step() or 0) < 2 and not t02_at_review):
            return {"ok": False, "reason": "campaign_inactive"}

        host._campaign_context_project_name = str(active_project or "").strip()
        host._campaign_graph_entry_context = dict(incoming_context or {})

        try:
            host.app.campaign_free_mode = False
            host.app.set_campaign_mode(True)
        except Exception:
            pass

        target = str(iteration_target or CAMPAIGN.get_iteration_target() or "").strip().lower()
        if target not in {"plate", "char"}:
            return {"ok": False, "reason": "missing_iteration_target"}
        try:
            current_step_for_graph = int(CAMPAIGN.get_current_step() or 0)
        except Exception:
            current_step_for_graph = 0
        try:
            current_iteration_path = normalize_iteration_path(CAMPAIGN.get_iteration_path())
        except Exception:
            current_iteration_path = ""
        if (
            target == "char"
            and current_iteration_path == "char_from_ready_plates"
            and current_step_for_graph < 3
            and not t02_at_review
        ):
            try:
                logger.warning(
                    "[Z2 GRAPH] blocked entry for char_from_ready_plates path; this gate skips Z2 "
                    "step=%s context=%s",
                    current_step_for_graph,
                    str(source_context or {}),
                )
            except Exception:
                pass
            return {"ok": False, "reason": "char_ready_plates_skips_z2"}
        graph_context = dict(getattr(host, "_campaign_graph_entry_context", {}) or {})
        if not str(graph_context.get("graph_gate_id") or "").strip():
            if target == "char" and current_step_for_graph >= 3:
                graph_context.update(
                    {
                        "source": "campaign_graph_inferred",
                        "graph_edge_key": "e3_to_e4",
                        "graph_gate_id": "T05",
                        "graph_visible_gate_id": "T05",
                        "graph_display_gate_id": "T05",
                        "graph_gate_label": "Dataset znaków",
                        "graph_transition_title": "Bramka datasetu znaków do treningu",
                        "graph_transition_source": "E3",
                        "graph_transition_target": "E4Z",
                        "graph_path_key": "",
                    }
                )
            elif target == "char":
                graph_context.update(
                    {
                        "source": "campaign_graph_inferred",
                        "graph_edge_key": "e2_to_e3",
                        "graph_gate_id": "T03",
                        "graph_visible_gate_id": "T03",
                        "graph_display_gate_id": "T03",
                        "graph_gate_label": "Przekazanie tablic do pracy nad znakami",
                        "graph_transition_title": "Przekazanie tablic do pracy nad znakami",
                        "graph_transition_source": "E2",
                        "graph_transition_target": "E3",
                        "graph_path_key": "char_from_images",
                    }
                )
            elif target == "plate":
                graph_context.update(
                    {
                        "source": "campaign_graph_inferred",
                        "graph_edge_key": "e2_to_e4",
                        "graph_gate_id": "T04",
                        "graph_visible_gate_id": "T04",
                        "graph_display_gate_id": "T04",
                        "graph_gate_label": "Trening modelu tablic",
                        "graph_transition_title": "Trenuj model tablic",
                        "graph_transition_source": "E2",
                        "graph_transition_target": "E4T",
                        "graph_path_key": "plate_training",
                    }
                )
            if graph_context:
                host._campaign_graph_entry_context = graph_context
                try:
                    logger.info(
                        "[Z2 GRAPH] inferred gate=%s edge=%s target=%s step=%s",
                        str(graph_context.get("graph_gate_id") or "-"),
                        str(graph_context.get("graph_edge_key") or "-"),
                        target,
                        current_step_for_graph,
                    )
                except Exception:
                    pass
        try:
            context_edge_key = str(graph_context.get("graph_edge_key") or "").strip()
            context_gate_id = campaign_gate_id_for_edge(
                context_edge_key,
                graph_context.get("graph_gate_id"),
            )
            repair_origin_edge_key = str(
                graph_context.get("repair_origin_edge_key")
                or graph_context.get("source_graph_edge_key")
                or ""
            ).strip()
            repair_origin_gate_id = str(
                graph_context.get("repair_origin_gate_id")
                or graph_context.get("source_graph_gate_id")
                or ""
            ).strip().upper()
            repair_origin_gate_id = campaign_gate_id_for_edge(
                repair_origin_edge_key,
                repair_origin_gate_id,
            )
        except Exception:
            context_gate_id = ""
            context_edge_key = ""
            repair_origin_gate_id = ""
        expected_context: dict[str, Any] = {}
        if t02_at_review:
            expected_context = {
                "source": str(graph_context.get("source") or "campaign_graph_t02_at_review"),
                "graph_edge_key": "e1_to_e3",
                "graph_gate_id": "T02",
                "graph_visible_gate_id": "T02",
                "graph_display_gate_id": "T02",
                "graph_gate_label": "Kontrola AT",
                "graph_transition_title": "Kontroluj import anotacji tablic",
                "graph_transition_source": "E1",
                "graph_transition_target": "E3",
                "graph_path_key": "char_from_ready_plates",
                "input_source": "t02_at_review",
                "z2_work_mode": "t02_at_review",
            }
        elif repair_origin_gate_id == "T06":
            expected_context = {}
        elif target == "char" and current_step_for_graph == 2:
            expected_context = {
                "source": str(graph_context.get("source") or "campaign_graph_canonicalized"),
                "graph_edge_key": "e2_to_e3",
                "graph_gate_id": "T03",
                "graph_visible_gate_id": "T03",
                "graph_display_gate_id": "T03",
                "graph_gate_label": "Przekazanie tablic do pracy nad znakami",
                "graph_transition_title": "Przekazanie tablic do pracy nad znakami",
                "graph_transition_source": "E2",
                "graph_transition_target": "E3",
                "graph_path_key": "char_from_images",
            }
        elif target == "plate" and current_step_for_graph == 2:
            expected_context = {
                "source": str(graph_context.get("source") or "campaign_graph_canonicalized"),
                "graph_edge_key": "e2_to_e4",
                "graph_gate_id": "T04",
                "graph_visible_gate_id": "T04",
                "graph_display_gate_id": "T04",
                "graph_gate_label": "Dataset i trening modelu tablic",
                "graph_transition_title": "Przygotuj dataset i trening modelu tablic",
                "graph_transition_source": "E2",
                "graph_transition_target": "E4T",
                "graph_path_key": "plate_training",
            }
        elif target == "char" and current_step_for_graph >= 3:
            expected_context = {
                "source": str(graph_context.get("source") or "campaign_graph_canonicalized"),
                "graph_edge_key": "e3_to_e4",
                "graph_gate_id": "T05",
                "graph_visible_gate_id": "T05",
                "graph_display_gate_id": "T05",
                "graph_gate_label": "Dataset znaków",
                "graph_transition_title": "Bramka datasetu znaków do treningu",
                "graph_transition_source": "E3",
                "graph_transition_target": "E4Z",
                "graph_path_key": "",
            }
        if expected_context:
            expected_gate_id = str(expected_context.get("graph_gate_id") or "").strip().upper()
            expected_edge_key = str(expected_context.get("graph_edge_key") or "").strip()
            if context_gate_id != expected_gate_id or context_edge_key != expected_edge_key:
                graph_context.update(expected_context)
                host._campaign_graph_entry_context = graph_context
                try:
                    logger.info(
                        "[Z2 GRAPH] canonicalized stale gate context old_gate=%s old_edge=%s new_gate=%s new_edge=%s target=%s step=%s",
                        context_gate_id or "-",
                        context_edge_key or "-",
                        expected_gate_id or "-",
                        expected_edge_key or "-",
                        target,
                        current_step_for_graph,
                    )
                except Exception:
                    pass
        try:
            graph_context_log = dict(getattr(host, "_campaign_graph_entry_context", {}) or {})
            logger.info(
                "[Z2 GRAPH] entry gate=%s edge=%s source=%s target=%s step=%s",
                str(graph_context_log.get("graph_gate_id") or "-"),
                str(graph_context_log.get("graph_edge_key") or "-"),
                str(graph_context_log.get("source") or "-"),
                target,
                current_step_for_graph,
            )
        except Exception:
            pass
        t04_plate_work_entry = bool(
            target == "plate"
            and int(current_step_for_graph or 0) == 2
            and campaign_gate_id_for_edge(
                graph_context.get("graph_edge_key"),
                graph_context.get("graph_gate_id"),
            )
            == "T04"
        )
        campaign_mode_text = "B: Tylko tablice"

        raw_dir = CAMPAIGN.get_dir("raw")
        auto_out = CAMPAIGN.get_staging_dir("auto_ann")
        if auto_out is not None:
            Path(auto_out).mkdir(parents=True, exist_ok=True)

        if raw_dir is None or auto_out is None:
            return {"ok": False, "reason": "missing_campaign_dirs"}

        vehicle_model_path = str(CAMPAIGN.get_global_model("vehicle") or "").strip()
        plate_model_path = str(CAMPAIGN.get_global_model("plate") or "").strip()
        char_source_state = {}
        char_has_existing_source = False
        plate_source_state = {}
        plate_model_ready = bool(plate_model_path and Path(plate_model_path).exists())
        if target == "char" and not restore_preview:
            plate_source_state = {}
        else:
            try:
                plate_source_state = dict(host.get_campaign_step2_source_state(iteration_target="plate") or {})
            except Exception:
                plate_source_state = {}
        plate_model_ready = bool(plate_source_state.get("plate_model_ready", plate_model_ready))
        if plate_model_ready and (not plate_model_path or not Path(plate_model_path).exists()):
            try:
                plate_model_path = str(
                    (plate_source_state.get("bootstrap") or {}).get("plate_model_path") or plate_model_path or ""
                ).strip()
            except Exception:
                plate_model_path = str(plate_model_path or "").strip()
        if target == "char" and not t02_at_review:
            if not restore_preview:
                try:
                    char_source_state = dict(CAMPAIGN.get_step3_char_source_state() or {})
                except Exception:
                    char_source_state = {}
            else:
                try:
                    char_source_state = dict(host.get_campaign_step2_source_state(iteration_target="char") or {})
                except Exception:
                    char_source_state = {}
            char_has_existing_source = bool(
                char_source_state.get("has_source")
                or int(char_source_state.get("total_plates", 0) or 0) > 0
            )
        char_manual_bootstrap = bool(target == "char" and not plate_model_ready and not char_has_existing_source)

        iter_num = CAMPAIGN.get_current_iteration_num()
        try:
            iteration_source_dir = (
                CAMPAIGN.get_iteration_image_source_dir(iter_num)
                or CAMPAIGN.get_iteration_raw_dir(iter_num)
            )
        except Exception:
            iteration_source_dir = None
        fallback_folder = Path(raw_dir) / f"Iteracja_{iter_num:03d}"
        if iteration_source_dir is not None and Path(iteration_source_dir).exists():
            input_dir = Path(iteration_source_dir)
        else:
            input_dir = fallback_folder if fallback_folder.exists() else Path(raw_dir)
        if t02_at_review:
            from .campaign_t02_source import resolve_t02_review_source

            source_result = resolve_t02_review_source(CAMPAIGN, graph_context)
            if not source_result.get("ok") or source_result.get("needs_approved_source"):
                return {"ok": False, "reason": source_result.get("reason", "missing_t02_source"),
                        "message": source_result.get("message", "Przygotuj źródło AT przez pole Praca bramki T02.")}
            graph_context.update(source_result["context"])
            input_dir = Path(graph_context.get("input_dir") or input_dir)
            host._campaign_graph_entry_context = graph_context
        base_input_dir = input_dir

        try:
            host._repair_campaign_step2_generated_state_for_input(base_input_dir)
        except Exception:
            pass
        _mark_phase("source_state")

        strategy = str(entry_strategy or "").strip().lower()
        snapshot_state = {} if t02_at_review else host._load_campaign_project_snapshot()
        _mark_phase("snapshot")
        char_repair_return = bool(
            target == "char"
            and host._is_campaign_char_repair_return_mode()
        )
        bootstrap = {}
        if t02_at_review:
            bootstrap = {"input_dir": base_input_dir, "restore_run_dir": graph_context.get("restore_run_dir"),
                         "input_source": graph_context.get("input_source", "t02_at_review"), "manual_template": False}
            char_manual_bootstrap = False
        elif char_repair_return and open_existing_run and strategy != "raw":
            snapshot_restore_run = None
            snapshot_input_dir = None
            registry_active_entry = host._get_campaign_step2_active_run_entry(
                images_dir=base_input_dir,
                iteration_num=iter_num,
            )
            try:
                registry_bundle = dict(
                    host._get_campaign_iteration_artifact_bundle(
                        images_dir=base_input_dir,
                        iteration_num=iter_num,
                    )
                    or {}
                )
            except Exception:
                registry_bundle = {}
            registry_plate_source = dict(registry_bundle.get("plate_source") or {})
            try:
                snapshot_restore_run = host._resolve_safe_annotation_run_dir(
                    registry_active_entry.get("run_dir") or registry_plate_source.get("run_dir"),
                    require_xml=True,
                )
            except Exception:
                snapshot_restore_run = None
            try:
                snapshot_input_dir = host._resolve_existing_dir(
                    registry_active_entry.get("images_dir") or registry_plate_source.get("images_dir")
                )
            except Exception:
                snapshot_input_dir = None
            if snapshot_restore_run is None:
                try:
                    snapshot_restore_run = host._resolve_safe_annotation_run_dir(
                        CAMPAIGN.get_step2_staging_run(),
                        require_xml=True,
                    )
                except Exception:
                    snapshot_restore_run = None
            if snapshot_restore_run is None:
                try:
                    stored_manual_source = CAMPAIGN.get_last_plate_manual_source()
                except Exception:
                    stored_manual_source = {}
                for raw_candidate in (
                    str((stored_manual_source or {}).get("source_run_path") or "").strip(),
                    str((stored_manual_source or {}).get("source_xml_path") or "").strip(),
                ):
                    if not raw_candidate:
                        continue
                    try:
                        run_candidate = Path(raw_candidate)
                        if run_candidate.suffix.lower() == ".xml":
                            run_candidate = run_candidate.parent
                    except Exception:
                        continue
                    snapshot_restore_run = host._resolve_safe_annotation_run_dir(
                        run_candidate,
                        require_xml=True,
                    )
                    if snapshot_restore_run is not None:
                        break
            if snapshot_restore_run is None and restore_preview:
                try:
                    latest_staging_run = host._find_latest_annotation_run_dir(Path(auto_out))
                except Exception:
                    latest_staging_run = None
                snapshot_restore_run = host._resolve_safe_annotation_run_dir(
                    latest_staging_run,
                    require_xml=True,
                )
            if snapshot_input_dir is None:
                try:
                    run_manifest = host._load_annotation_run_manifest(snapshot_restore_run)
                except Exception:
                    run_manifest = {}
                snapshot_input_dir = host._resolve_existing_dir(
                    run_manifest.get("input_dir")
                    or run_manifest.get("source_input_dir")
                    or run_manifest.get("imported_source_input_dir")
                )

            repair_context_matches_current_iteration = False
            if snapshot_input_dir is not None:
                try:
                    repair_context_matches_current_iteration = host._paths_equivalent(
                        snapshot_input_dir,
                        base_input_dir,
                    )
                except Exception:
                    repair_context_matches_current_iteration = False
            if not repair_context_matches_current_iteration and snapshot_restore_run is not None:
                try:
                    repair_context_matches_current_iteration = host._annotation_run_matches_expected_input_dir(
                        snapshot_restore_run,
                        base_input_dir,
                    )
                except Exception:
                    repair_context_matches_current_iteration = False

            if repair_context_matches_current_iteration:
                if snapshot_input_dir is None:
                    snapshot_input_dir = base_input_dir
            else:
                if snapshot_restore_run is not None or snapshot_input_dir is not None:
                    try:
                        logger.info(
                            "Pomijam stary kontekst Z2 przy powrocie E3 -> Z2: "
                            f"snapshot_input={snapshot_input_dir} restore_run={snapshot_restore_run} "
                            f"current_iteration_input={base_input_dir}"
                        )
                    except Exception:
                        pass
                snapshot_restore_run = None
                snapshot_input_dir = base_input_dir

            bootstrap = {
                "input_dir": (snapshot_input_dir or base_input_dir),
                "input_source": (
                    "campaign_step3_repair_snapshot"
                    if snapshot_restore_run is not None
                    else "raw"
                ),
                "manual_template": False,
                "plate_model_path": plate_model_path if plate_model_path and Path(plate_model_path).exists() else "",
                "restore_run_dir": snapshot_restore_run,
            }
        else:
            try:
                bootstrap = host._get_campaign_auto_annotation_bootstrap(target)
            except Exception:
                bootstrap = {}
        _mark_phase("bootstrap")

        if t02_at_review:
            try:
                start_source = dict(CAMPAIGN.get_project_start_plate_source() or {})
            except Exception:
                start_source = {}
            source_run = str(start_source.get("source_run_path") or "").strip()
            source_xml = str(start_source.get("source_xml_path") or "").strip()
            source_input = str(start_source.get("source_input_path") or "").strip()
            if source_run:
                graph_context.setdefault("restore_run_dir", source_run)
            elif source_xml:
                try:
                    graph_context.setdefault("restore_run_dir", str(Path(source_xml).parent))
                except Exception:
                    pass
            if source_input:
                graph_context.setdefault("input_dir", source_input)
            graph_context.setdefault("input_source", "t02_at_review")
            graph_context.setdefault("z2_work_mode", "t02_at_review")
            host._campaign_graph_entry_context = graph_context

        context_restore_run = None
        try:
            context_restore_run = graph_context.get("restore_run_dir")
        except Exception:
            context_restore_run = None
        try:
            context_gate_id = campaign_gate_id_for_edge(
                graph_context.get("graph_edge_key"),
                graph_context.get("graph_gate_id"),
            )
        except Exception:
            context_gate_id = ""
        if context_restore_run is not None and (target == "plate" or context_gate_id in {"T02", "T05"} or t02_at_review):
            try:
                context_run_dir = host._resolve_safe_annotation_run_dir(
                    context_restore_run,
                    require_xml=True,
                )
            except Exception:
                context_run_dir = None
            if context_run_dir is not None:
                bootstrap = dict(bootstrap) if isinstance(bootstrap, dict) else {}
                bootstrap["restore_run_dir"] = context_run_dir
                bootstrap["input_source"] = str(
                    graph_context.get("input_source")
                    or "campaign_graph_restore_run"
                ).strip()
                bootstrap["manual_template"] = False
                try:
                    manifest = host._load_annotation_run_manifest(context_run_dir)
                except Exception:
                    manifest = {}
                context_input = host._resolve_existing_dir(graph_context.get("input_dir")) if t02_at_review else None
                for manifest_key in ("source_input_dir", "input_dir", "imported_source_input_dir"):
                    if context_input is not None:
                        break
                    raw_input = str(manifest.get(manifest_key) or "").strip()
                    if not raw_input:
                        continue
                    try:
                        candidate_input = host._resolve_existing_dir(raw_input)
                    except Exception:
                        candidate_input = None
                    if candidate_input is not None:
                        context_input = candidate_input
                        break
                if context_input is not None:
                    bootstrap["input_dir"] = context_input

        if char_manual_bootstrap and not bootstrap.get("restore_run_dir"):
            bootstrap = dict(bootstrap) if isinstance(bootstrap, dict) else {}
            bootstrap["input_dir"] = base_input_dir
            bootstrap["restore_run_dir"] = None
            bootstrap["input_source"] = "raw_manual_for_char"
            bootstrap["manual_template"] = True
            bootstrap["plate_model_path"] = ""

        if target == "plate" and strategy == "raw":
            bootstrap = dict(bootstrap) if isinstance(bootstrap, dict) else {}
            bootstrap["input_dir"] = base_input_dir
            bootstrap["restore_run_dir"] = None
            bootstrap["input_source"] = "raw_forced"
            bootstrap["manual_template"] = bool(not (plate_model_path and Path(plate_model_path).exists()))
            bootstrap["plate_model_path"] = plate_model_path if plate_model_path and Path(plate_model_path).exists() else ""

        stale_restore_run = None
        restore_candidate = bootstrap.get("restore_run_dir") if isinstance(bootstrap, dict) else None
        if restore_candidate is not None:
            try:
                if not host._annotation_run_matches_expected_input_dir(restore_candidate, base_input_dir):
                    stale_restore_run = restore_candidate
            except Exception:
                stale_restore_run = None
        if stale_restore_run is not None:
            try:
                logger.info(
                    "Pomijam stary run Z2 przy wejsciu z grafu: "
                    f"run={stale_restore_run} expected_input={base_input_dir}"
                )
            except Exception:
                pass
            bootstrap = dict(bootstrap) if isinstance(bootstrap, dict) else {}
            bootstrap["input_dir"] = base_input_dir
            bootstrap["restore_run_dir"] = None
            bootstrap["skip_project_state_restore"] = True
            if target == "char" and not (plate_model_path and Path(plate_model_path).exists()):
                bootstrap["input_source"] = "raw_manual_for_char"
                bootstrap["manual_template"] = True
                bootstrap["plate_model_path"] = ""
            elif str(bootstrap.get("input_source") or "").strip().startswith("registry"):
                bootstrap["input_source"] = "raw_current_iteration"

        input_dir = Path(bootstrap.get("input_dir") or input_dir)
        manual_template = bool(bootstrap.get("manual_template", target == "plate"))
        plate_bootstrap_model = str(bootstrap.get("plate_model_path") or "").strip()
        restore_run_dir = bootstrap.get("restore_run_dir")
        input_source = str(bootstrap.get("input_source") or "raw").strip()
        if target == "plate" and restore_run_dir is not None:
            try:
                restore_manifest = host._load_annotation_run_manifest(Path(restore_run_dir))
            except Exception:
                restore_manifest = {}
            manual_template = bool(host._annotation_run_manifest_is_manual_template(restore_manifest))
            bootstrap["manual_template"] = manual_template
        if t04_plate_work_entry and restore_run_dir is None and not bool(
            graph_context.get("allow_auto_annotation_entry")
            or graph_context.get("force_auto_annotation_entry")
        ):
            manual_template = True
            bootstrap["manual_template"] = True
            bootstrap["input_source"] = str(bootstrap.get("input_source") or "t04_manual_work_entry")
            input_source = str(bootstrap.get("input_source") or input_source or "t04_manual_work_entry").strip()
            try:
                logger.info(
                    "[Z2 GRAPH] T04 entry forced to manual workflow; autoannotation requires explicit Z2 action."
                )
            except Exception:
                pass
        if not host._should_restore_existing_campaign_step2_run(target):
            restore_run_dir = None
            bootstrap["restore_run_dir"] = None
        elif not host._should_restore_campaign_generated_step2_run_preview(
            restore_run_dir,
            session_state=snapshot_state,
            iteration_target=target,
        ):
            restore_run_dir = None
            bootstrap["restore_run_dir"] = None
        char_repair_without_run = bool(
            target == "char"
            and host._is_campaign_char_repair_return_mode()
            and restore_run_dir is None
        )
        effective_manual_template = bool(manual_template)
        restored_snapshot = False
        open_detected_run = bool(
            open_existing_run
            and not restored_snapshot
            and restore_run_dir is not None
            and strategy != "raw"
            and (
                target == "plate"
                or (target == "char" and not restore_preview)
            )
        )
        defer_existing_run_ui_restore = bool(
            target == "char"
            and open_detected_run
            and not restore_preview
        )
        skip_project_state_restore = bool(
            open_detected_run
            or defer_existing_run_ui_restore
            or bool(bootstrap.get("skip_project_state_restore"))
        )
        restore_preview_during_context = bool(restore_preview and not open_detected_run)
        opened_existing_run = False
        defer_initial_preview_load = bool(
            defer_preview_load
            and not open_detected_run
        )
        # The context layer must never synchronously prime the source preview
        # when this graph transition is about to open an existing run. On large
        # projects (NEON: 9k+ images) that redundant source scan cost several
        # seconds before the real run restore even started.
        defer_context_source_preview = bool(
            defer_preview_load
            or open_detected_run
            or not restore_preview_during_context
        )
        reused_loaded_t02_run = False
        repopulate_loaded_t02_list = False
        if t02_at_review and restore_run_dir is not None:
            try:
                safe_restore_run = host._resolve_safe_annotation_run_dir(
                    restore_run_dir,
                    require_xml=True,
                )
            except Exception:
                safe_restore_run = None
            try:
                safe_current_run = host._resolve_safe_annotation_run_dir(
                    getattr(host, "current_annotation_run_dir", None),
                    require_xml=True,
                )
            except Exception:
                safe_current_run = None
            same_loaded_run = False
            if safe_restore_run is not None and safe_current_run is not None:
                try:
                    same_loaded_run = host._paths_equivalent(safe_restore_run, safe_current_run)
                except Exception:
                    same_loaded_run = bool(Path(safe_restore_run).resolve() == Path(safe_current_run).resolve())
            if (
                same_loaded_run
                and bool(getattr(host, "current_annotations", None))
                and not bool(getattr(host, "_campaign_deferred_run_restore_in_progress", False))
            ):
                try:
                    host.current_input_dir = Path(input_dir)
                    host.input_dir_var.set(str(input_dir))
                    host.plate_dataset_images_var.set(str(input_dir))
                    host.current_annotation_run_dir = safe_current_run
                    host.current_annotation_xml_path = Path(safe_current_run) / "annotations.xml"
                    host.last_staging_run_dir = safe_current_run
                    host.plate_dataset_run_var.set(str(safe_current_run))
                except Exception:
                    pass
                try:
                    visible_rows = int(getattr(host, "preview_listbox", None).size() or 0)
                except Exception:
                    visible_rows = 0
                try:
                    expected_rows = int(len(getattr(host, "current_annotations", []) or []))
                except Exception:
                    expected_rows = 0
                reused_loaded_t02_run = True
                repopulate_loaded_t02_list = bool(expected_rows > 0 and visible_rows < expected_rows)
                opened_existing_run = True
                open_detected_run = False
                defer_existing_run_ui_restore = False
                skip_project_state_restore = True
                defer_initial_preview_load = False
                try:
                    apply_campaign_step2_workflow_preset(
                        host,
                        iteration_target=target,
                        manual_template=manual_template,
                    )
                except Exception:
                    pass
                try:
                    host._refresh_free_mode_workflow_ui()
                except Exception:
                    pass
                try:
                    host._reset_main_pane_left_width_for_z2()
                except Exception:
                    pass
                try:
                    logger.info(
                        "[Z2 PERF] reuse_loaded_t02_run annotations=%s listbox=%s run=%s",
                        expected_rows,
                        visible_rows,
                        safe_current_run,
                    )
                except Exception:
                    pass
            else:
                try:
                    logger.info(
                        "[Z2 PERF] reuse_loaded_t02_run skipped same=%s annotations=%s deferred=%s restore=%s current=%s",
                        bool(same_loaded_run),
                        int(len(getattr(host, "current_annotations", []) or [])),
                        bool(getattr(host, "_campaign_deferred_run_restore_in_progress", False)),
                        safe_restore_run,
                        safe_current_run,
                    )
                except Exception:
                    pass

        if reused_loaded_t02_run:
            restored_snapshot = False
            _mark_phase("reuse_loaded_t02_run")
        else:
            restored_snapshot = host.apply_campaign_context(
                input_dir,
                Path(auto_out),
                manual_template=effective_manual_template,
                mode_text=campaign_mode_text,
                restore_project_state=bool(not skip_project_state_restore),
                restore_preview=restore_preview_during_context,
                defer_preview_load=defer_context_source_preview,
                refresh_export_sources=False,
                refresh_character_models=False,
            )
            _mark_phase("apply_context")

        if skip_project_state_restore and isinstance(snapshot_state, dict):
            try:
                host._preview_session_restore_index = int(snapshot_state.get("last_preview_index", -1))
            except (TypeError, ValueError):
                host._preview_session_restore_index = -1
            try:
                host._preview_session_restore_filename = str(snapshot_state.get("last_preview_filename") or "").strip()
            except Exception:
                host._preview_session_restore_filename = ""

        if restored_snapshot and open_detected_run:
            open_detected_run = False
            defer_existing_run_ui_restore = False

        if not restored_snapshot and not open_detected_run and not reused_loaded_t02_run:
            if vehicle_model_path and Path(vehicle_model_path).exists():
                host.vehicle_model_var.set("Custom")
                host.vehicle_custom_var.set(vehicle_model_path)
            else:
                try:
                    vehicle_values = list(host.vehicle_combo["values"]) if hasattr(host, "vehicle_combo") else []
                except Exception:
                    vehicle_values = []
                detect_values = [value for value in vehicle_values if value != "Custom"]
                default_vehicle = "yolo11s" if "yolo11s" in detect_values else (detect_values[0] if detect_values else "")
                if default_vehicle:
                    host.vehicle_model_var.set(default_vehicle)
                host.vehicle_custom_var.set("")

            if (
                target == "char" and plate_model_path and Path(plate_model_path).exists()
            ) or (
                target == "plate" and plate_bootstrap_model and Path(plate_bootstrap_model).exists()
            ):
                host.plate_custom_var.set(plate_bootstrap_model if target == "plate" else plate_model_path)
            else:
                host.plate_custom_var.set("")

            host.mode_var.set(campaign_mode_text)
            host._on_mode_change()

        if open_detected_run:
            try:
                if host._is_free_mode_session_context():
                    opened_existing_run = host._open_existing_run_for_manual_review(
                        run_dir=restore_run_dir,
                        allow_fallback=False,
                        from_auto=False,
                        show_dialog=False,
                    )
                else:
                    opened_existing_run = open_existing_run_for_campaign_review(
                        host,
                        run_dir=restore_run_dir,
                        iteration_target=target,
                        manual_template=manual_template,
                        defer_ui_restore=defer_existing_run_ui_restore,
                    )
            except Exception:
                opened_existing_run = False

        if defer_existing_run_ui_restore and opened_existing_run:
            host._campaign_step2_transition_skip_heavy_finalize = True
        _mark_phase("open_run")

        if target == "char" and not restore_preview and not opened_existing_run:
            # Nie budujemy listy obrazów ani roboczego XML synchronicznie w ścieżce
            # T04/T06 -> Z2. Na dużych projektach (np. NEON) samo przejście do Z2
            # blokowało GUI na kilkadziesiąt sekund. Lista i ewentualny run roboczy
            # zostaną doładowane przez deferred_preview_load po pokazaniu karty.
            defer_initial_preview_load = True

        current_route = ""
        try:
            current_route = host._get_workflow_route()
        except Exception:
            current_route = ""

        force_campaign_preset = bool(
            not reused_loaded_t02_run
            and not opened_existing_run
            and target in {"plate", "char"}
            and effective_manual_template
            and current_route != "manual"
        )

        if not opened_existing_run and (not restored_snapshot or force_campaign_preset):
            apply_campaign_step2_workflow_preset(
                host,
                iteration_target=target,
                manual_template=effective_manual_template,
            )
        elif not opened_existing_run and not current_route:
            apply_campaign_step2_workflow_preset(
                host,
                iteration_target=target,
                manual_template=effective_manual_template,
            )

        if reused_loaded_t02_run:
            try:
                host.frame.after_idle(host._sync_right_panel_scrollregion)
            except Exception:
                pass
        else:
            try:
                host._refresh_free_mode_workflow_ui()
            except Exception:
                pass

            try:
                host._reset_main_pane_left_width_for_z2()
            except Exception:
                pass

            try:
                host._scroll_left_panel_to_widget(
                    getattr(host, "workflow_start_section", None)
                    or getattr(host, "source_section", None)
                    or getattr(host, "actions_section", None)
                )
            except Exception:
                pass

            try:
                host._refresh_step2_action_states()
            except Exception:
                pass

        if (
            effective_manual_template
            and not opened_existing_run
            and str(host._get_workflow_route() or "").strip().lower() == "manual"
            and str(host._get_manual_entry_mode() or "").strip().lower() == "new"
            and Path(input_dir).exists()
            and not t04_plate_work_entry
            and not bool(getattr(host, "_campaign_manual_prepare_pending", False))
        ):
            try:
                host._campaign_manual_prepare_pending = True

                def _prepare_campaign_manual_package() -> None:
                    try:
                        host._ensure_campaign_manual_package_ready(iteration_target=target)
                    finally:
                        host._campaign_manual_prepare_pending = False

                host.frame.after_idle(_prepare_campaign_manual_package)
            except Exception:
                host._campaign_manual_prepare_pending = False
        _mark_phase("refresh_ui")

        host._campaign_context_project_name = str(active_project or "").strip()

        deferred_preview_load = bool(
            defer_initial_preview_load
            and not opened_existing_run
            and not bool(getattr(host, "current_annotations", None))
        )

        return {
            "ok": True,
            "iteration_target": target,
            "input_dir": str(input_dir),
            "auto_out": str(auto_out),
            "manual_template": bool(effective_manual_template),
            "input_source": input_source,
            "restored_snapshot": bool(restored_snapshot),
            "opened_existing_run": bool(opened_existing_run),
            "manual_prepare_deferred_to_user": bool(
                t04_plate_work_entry
                and effective_manual_template
                and not opened_existing_run
            ),
            "restore_run_dir": str(restore_run_dir or ""),
            "plate_model_path": plate_model_path,
            "deferred_preview_load": deferred_preview_load,
            "deferred_preview_input_dir": (str(input_dir) if deferred_preview_load else ""),
            "deferred_existing_run_restore": bool(
                defer_existing_run_ui_restore and opened_existing_run
            ),
            "reused_loaded_run": bool(reused_loaded_t02_run),
            "repopulate_preview_list": bool(repopulate_loaded_t02_list),
        }
    except Exception as e:
        logger.error(f"Nie udalo sie otworzyc punktu startowego Z2: {e}")
        return {"ok": False, "reason": "exception", "error": str(e)}
    finally:
        end_transition_started = time.perf_counter()
        try:
            host._end_campaign_step2_transition()
        except Exception:
            pass
        end_transition_ms = (time.perf_counter() - end_transition_started) * 1000.0
        if end_transition_ms >= 250.0:
            slow_phase_notes.append(f"end_transition={end_transition_ms:.0f}ms")
        elapsed_ms = max(0.0, (time.perf_counter() - open_started) * 1000.0)
        if elapsed_ms >= 500.0:
            target_label = str(locals().get("target") or "?")
            char_repair_label = int(bool(locals().get("char_repair_without_run", False)))
            deferred_label = int(bool(locals().get("deferred_preview_load", False)))
            opened_label = int(bool(locals().get("opened_existing_run", False)))
            phase_label = ", ".join(slow_phase_notes) if slow_phase_notes else "no_slow_phase"
            logger.info(
                "[Z2 PERF] open_campaign_step2_entry total="
                f"{elapsed_ms:.0f}ms target={target_label} "
                f"restore_preview={int(bool(restore_preview))} "
                f"open_existing_run={int(bool(open_existing_run))} "
                f"defer_preview_load={int(bool(defer_preview_load))} "
                f"char_repair={char_repair_label} opened_run={opened_label} "
                f"deferred_preview={deferred_label} phases=[{phase_label}]"
            )


def build_z2_layout_state_campaign(
    host: "AnnotationTab",
    *,
    route: str,
    campaign_stage: int,
    campaign_iteration_target: str,
    available_primary_action_ids: list[str],
    manual_review_active: bool,
    auto_completed: bool,
    has_existing_run: bool,
) -> dict[str, Any]:
    campaign_stage2 = int(campaign_stage or 0) == 2
    t02_at_review_context = bool(is_campaign_t02_at_review_context(host))
    show_export_followup = bool(host._manual_review_export_ready)
    show_auto_followup = bool(
        auto_completed
        and not manual_review_active
        and not show_export_followup
        and False
    )
    show_manual_review_followup = bool(manual_review_active and not show_export_followup)
    show_route_choice = bool(
        campaign_stage2
        and campaign_iteration_target in {"plate", "char"}
        and not route
        and len(available_primary_action_ids) > 1
        and not show_export_followup
    )
    show_stage_export_cta = False
    compact_export_followup = False
    show_workflow_steps = bool(
        (
            route
            and not show_manual_review_followup
            and not show_export_followup
            and (not auto_completed or (campaign_stage2 and route == "auto"))
        )
        or (
            campaign_stage2
            and not has_existing_run
            and not show_manual_review_followup
            and not show_auto_followup
            and not show_export_followup
        )
    )
    compact_single_route_layout = bool(
        campaign_stage2
        and len(available_primary_action_ids) == 1
        and show_workflow_steps
        and not show_manual_review_followup
        and not show_export_followup
    )
    compact_left_column_layout = bool(
        compact_single_route_layout
        or show_manual_review_followup
        or show_auto_followup
        or show_export_followup
    )
    show_campaign_context_header = bool(
        True
        and not show_workflow_steps
        and bool(
            route
            or manual_review_active
            or has_existing_run
            or campaign_stage >= 3
            or t02_at_review_context
        )
    )
    return Z2LayoutState(
        show_export_followup=show_export_followup,
        show_auto_followup=show_auto_followup,
        show_manual_review_followup=show_manual_review_followup,
        show_route_choice=show_route_choice,
        show_stage_export_cta=show_stage_export_cta,
        compact_export_followup=compact_export_followup,
        show_workflow_steps=show_workflow_steps,
        compact_single_route_layout=compact_single_route_layout,
        compact_left_column_layout=compact_left_column_layout,
        show_campaign_context_header=show_campaign_context_header,
        show_nav_panel=False,
        show_right_panel=bool(
            (
                int(campaign_stage or 0) == 2
                or int(campaign_stage or 0) == 3
                or t02_at_review_context
                or host._is_campaign_char_repair_return_mode()
                or host._is_campaign_plate_step4_repair_return_mode()
            )
            and not show_export_followup
        ),
    )


def prepare_campaign_workflow_runtime(
    host: "AnnotationTab",
    *,
    route: str,
    manual_review_active: bool,
    input_dir_ready: bool,
) -> Z2CampaignRuntimeState:
    campaign_stage = 0
    campaign_iteration_target = ""
    campaign_iteration_num = 1
    campaign_char_repair_mode = False
    campaign_plate_step4_repair_mode = False
    try:
        from ..campaign_manager import CAMPAIGN
        campaign_stage = int(CAMPAIGN.get_current_step() or 0)
        campaign_iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        campaign_iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
        campaign_char_repair_mode = bool(
            campaign_stage == 3
            and campaign_iteration_target == "char"
        )
        campaign_plate_step4_repair_mode = bool(
            campaign_stage == 4
            and campaign_iteration_target == "plate"
            and host._is_campaign_plate_step4_repair_return_mode()
        )
    except Exception:
        campaign_stage = 0
        campaign_iteration_target = ""
        campaign_iteration_num = 1
        campaign_char_repair_mode = False
        campaign_plate_step4_repair_mode = False
    try:
        secondary_ctx = host._build_z2_action_context()
        available_primary_action_ids = [
            action_id
            for action_id, action in (host._get_z2_primary_actions() or {}).items()
            if action is not None and action.is_available(secondary_ctx)
        ]
    except Exception:
        available_primary_action_ids = []

    current_route = str(route or "").strip().lower()
    try:
        graph_context = dict(getattr(host, "_campaign_graph_entry_context", {}) or {})
        graph_gate_id = campaign_gate_id_for_edge(
            graph_context.get("graph_edge_key"),
            graph_context.get("graph_gate_id"),
        )
    except Exception:
        graph_gate_id = ""
    t04_plate_work_entry = bool(
        int(campaign_stage or 0) == 2
        and campaign_iteration_target == "plate"
        and graph_gate_id == "T04"
    )
    if (
        int(campaign_stage or 0) == 2
        and campaign_iteration_target in {"plate", "char"}
        and "auto" in available_primary_action_ids
        and current_route != "manual"
        and not t04_plate_work_entry
        and (
            (campaign_iteration_target == "plate" and int(campaign_iteration_num or 1) > 1)
            or campaign_iteration_target == "char"
        )
    ):
        available_primary_action_ids = ["auto"]
        if not manual_review_active and current_route != "auto":
            host._set_workflow_route_state("auto", campaign_context=True)
            host.manual_xml_template_var.set(False)
            host._set_workflow_step_state("auto_start" if input_dir_ready else "auto_input", campaign_context=True)
            host._set_auto_vehicle_choice_state(host._get_auto_vehicle_choice(), campaign_context=True)
            current_route = "auto"

    single_available_primary_route = (
        str(available_primary_action_ids[0]).strip()
        if len(available_primary_action_ids) == 1
        else ""
    )
    if (
        int(campaign_stage or 0) == 2
        and not current_route
        and not manual_review_active
        and single_available_primary_route in {"auto", "manual"}
    ):
        if single_available_primary_route == "manual":
            host._set_workflow_route_state("manual", campaign_context=True)
            host._set_manual_entry_mode_state("new", campaign_context=True)
            host.manual_xml_template_var.set(True)
            try:
                host.manual_vehicle_assist_var.set(False)
            except Exception:
                pass
            host._set_workflow_step_state("manual_start" if input_dir_ready else "manual_input", campaign_context=True)
        else:
            host._set_workflow_route_state("auto", campaign_context=True)
            host.manual_xml_template_var.set(False)
            host._set_workflow_step_state("auto_start" if input_dir_ready else "auto_input", campaign_context=True)
            host._set_auto_vehicle_choice_state(host._get_auto_vehicle_choice(), campaign_context=True)
        current_route = single_available_primary_route

    return Z2CampaignRuntimeState(
        route=current_route,
        campaign_stage=campaign_stage,
        campaign_iteration_target=campaign_iteration_target,
        campaign_iteration_num=campaign_iteration_num,
        campaign_char_repair_mode=campaign_char_repair_mode,
        campaign_plate_step4_repair_mode=campaign_plate_step4_repair_mode,
        available_primary_action_ids=available_primary_action_ids,
    )


def build_z2_cta_state_campaign(
    host: "AnnotationTab",
    *,
    route: str,
    current_step: str,
    show_workflow_steps: bool,
    show_export_followup: bool,
    auto_completed: bool,
    manual_run_already_created: bool,
    input_dir_ready: bool,
    auto_setup_pending: bool,
    auto_vehicle_choice: str,
    manual_setup: bool,
    campaign_reused_manual_count: int,
) -> Z2CtaState:
    graph_context: dict[str, Any] = {}
    try:
        graph_context = dict(getattr(host, "_campaign_graph_entry_context", {}) or {})
        graph_gate_id = campaign_gate_id_for_edge(
            graph_context.get("graph_edge_key"),
            graph_context.get("graph_gate_id"),
        )
    except Exception:
        graph_gate_id = ""
    try:
        from ..campaign_manager import CAMPAIGN

        t04_plate_work_entry = bool(
            graph_gate_id == "T04"
            and int(CAMPAIGN.get_current_step() or 0) == 2
            and str(CAMPAIGN.get_iteration_target() or "").strip().lower() == "plate"
        )
    except Exception:
        t04_plate_work_entry = bool(graph_gate_id == "T04")
    manual_auto_bootstrap = bool(
        manual_setup
        and input_dir_ready
        and not manual_run_already_created
        and not t04_plate_work_entry
    )
    campaign_auto_start_ready = bool(
        route == "auto"
        and current_step == "auto_start"
        and input_dir_ready
    )
    show_start_controls = bool(
        show_workflow_steps
        and (
            current_step in {"auto_start", "manual_start"}
            or current_step in {"auto_input", "manual_input"}
        )
    )
    if route == "auto" and auto_completed and not show_export_followup:
        show_start_controls = True
    if campaign_auto_start_ready:
        show_start_controls = True
    if route == "manual" and manual_run_already_created and not host.is_processing:
        show_start_controls = False

    def _start_campaign_auto_action() -> None:
        try:
            if getattr(host, "is_processing", False):
                return
            host._set_workflow_route_state("auto", campaign_context=True)
            host.manual_xml_template_var.set(False)
            host._set_workflow_step_state("auto_start", campaign_context=True)
            host._set_auto_vehicle_choice_state(host._get_auto_vehicle_choice(), campaign_context=True)
            host._start_annotation()
        except Exception:
            try:
                host._start_annotation()
            except Exception:
                pass

    start_enabled = not host.is_processing and bool(route)
    start_command = host._start_annotation
    start_text = "Wybierz tor"
    if route == "auto":
        if not input_dir_ready:
            start_enabled = not host.is_processing
            start_text = "Wybierz obrazy do autoanotacji" if auto_setup_pending else "Wybierz obrazy"
            start_command = host._select_input_dir
        elif auto_setup_pending:
            start_text = "Uruchom autoanotację"
        elif auto_vehicle_choice == "skip":
            start_text = "Uruchom autoanotację tablic"
        else:
            start_text = "Uruchom autoanotację tablic i pojazdów"
    elif manual_setup:
        if manual_run_already_created:
            start_enabled = False
            start_text = "Run istnieje"
        elif not input_dir_ready:
            start_enabled = not host.is_processing
            start_text = "Wybierz obrazy"
            start_command = host._select_input_dir
        else:
            if t04_plate_work_entry:
                start_enabled = not host.is_processing
                start_text = "Przygotuj roboczy XML Z2"
                start_command = host._start_annotation
            else:
                start_enabled = False
                start_text = "Przygotowuję Z2"

    if route == "auto" and input_dir_ready:
        start_command = _start_campaign_auto_action

    return Z2CtaState(
        show_start_controls=show_start_controls,
        show_nav_controls=False,
        start_enabled=start_enabled,
        start_command=start_command,
        start_text=start_text,
        back_enabled=False,
        next_enabled=False,
        next_text="Dalej",
        suppress_duplicate_start_cta=bool(
            not host.is_processing and start_text in {"Wybierz tor", "Run istnieje"}
        )
        or manual_auto_bootstrap,
    )


def _apply_campaign_char_repair_copy_payload(
    payload: Z2CopyPayload,
    *,
    manual_review_active: bool = False,
) -> Z2CopyPayload:
    payload["run_title"] = "Przygotowanie większej liczby tablic"
    payload["badge_text"] = "Aktywny tor: świadomy powrót z E3 do Z2"
    payload["badge_tone"] = "success"
    payload["run_intro_text"] = (
        "Wróciłeś do Z2, żeby świadomie powiększyć projektowy zbiór tablic. "
        "Więcej poprawnych ramek daje lepszy materiał do treningu znaków. "
        "Zatwierdzone anotacje tablic [OK] wchodzą do wspólnej puli przyszłych iteracji "
        "w torze tablic bądź znaków, zgodnie z wyborem użytkownika. "
        "Pamiętaj o zatwierdzeniu uznanych za poprawnie anotowane obrazy na liście wyników: "
        "prawy przycisk myszy PPM na pozycji lub zaznaczonej grupie."
    )
    payload["route_text"] = (
        "Uzupełnij ramki na podglądzie. Poprawne zdjęcia zatwierdzaj z menu listy "
        "otwieranym prawym przyciskiem myszy: wybierz „Oznacz zaznaczone jako OK”."
    )
    payload["action_text"] = ""
    payload["workflow_start_title"] = ""
    payload["workflow_start_intro"] = ""
    payload["auto_plate_model_hint_text"] = (
        "W tym powrocie możesz użyć aktywnego modelu projektu albo podmienić model tylko dla tego runu Z2. "
        "Jeśli wolisz, możesz też pominąć autoanotację i poprawiać tablice ręcznie."
    )
    payload["auto_plate_model_hint_tone"] = "muted"

    if manual_review_active:
        payload["manual_hint"] = (
            "Ten powrót otwiera pełne Z2 dla tego samego katalogu zdjęć: możesz użyć autoanotacji aktywnym modelem projektu, "
            "podmienić model tylko dla tego runu albo poprawiać tablice ręcznie."
        )
        payload["manual_hint_tone"] = "muted"
        payload["workflow_input_title"] = "Aktywny run źródłowy"
        payload["workflow_input_hint"] = (
            "Bieżący run tablic jest już wczytany i gotowy do ręcznej poprawy przed powrotem do E3."
        )

    return payload


def _get_campaign_graph_entry_context(host: "AnnotationTab") -> dict[str, Any]:
    try:
        return dict(getattr(host, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        return {}


def _is_campaign_graph_t05_repair_from_t07(host: "AnnotationTab", graph_context: dict[str, Any] | None = None) -> bool:
    context = dict(graph_context or _get_campaign_graph_entry_context(host))
    gate_id = campaign_gate_id_for_edge(context.get("graph_edge_key"), context.get("graph_gate_id"))
    origin_gate_id = campaign_gate_id_for_edge(
        context.get("repair_origin_edge_key") or context.get("source_graph_edge_key"),
        context.get("repair_origin_gate_id") or context.get("source_graph_gate_id"),
    )
    if gate_id == "T04" and origin_gate_id == "T06":
        return True
    try:
        return bool(gate_id == "T04" and host._is_campaign_plate_step4_repair_return_mode())
    except Exception:
        return False


def _apply_campaign_graph_gate_copy_payload(
    host: "AnnotationTab",
    ctx: Z2LeftPanelCopyContext,
    payload: Z2CopyPayload,
) -> Z2CopyPayload:
    graph_context = _get_campaign_graph_entry_context(host)
    gate_id = campaign_gate_id_for_edge(graph_context.get("graph_edge_key"), graph_context.get("graph_gate_id"))
    if gate_id not in {"T03", "T04", "T05"}:
        return payload
    display_gate_id = campaign_visible_gate_id(gate_id) or gate_id

    default_gate_label = {
        "T03": "Przekazanie tablic do pracy nad znakami",
        "T04": "Dataset i trening modelu tablic",
        "T05": "Uzupełnienie tablic dla znaków",
    }.get(gate_id, "Praca w Z2")
    gate_label = str(graph_context.get("graph_gate_label") or default_gate_label).strip()
    target_label = "E3" if gate_id in {"T03", "T05"} else "E4T"
    try:
        target_label = (
            str(graph_context.get("graph_transition_target") or target_label).strip().upper()
            or target_label
        )
    except Exception:
            target_label = "E3" if gate_id in {"T03", "T05"} else "E4T"

    if gate_id == "T04" and _is_campaign_graph_t05_repair_from_t07(host, graph_context):
        payload["run_title"] = "Naprawa tablic przed decyzją T06"
        payload["badge_text"] = "Tryb naprawczy: powrót z T06 do Z2"
        payload["badge_tone"] = "warning"
        payload["route_tone"] = "muted"
        payload["run_intro_text"] = (
            "Uzupełniasz materiał po wcześniejszym zatwierdzeniu tablic. "
            "Zdjęcia już zatwierdzone nie wracają na listę; tutaj pracujesz tylko na tym, co wymaga poprawy lub dopisania."
        )
        payload["route_text"] = (
            "Dodaj brakujące ramki tablic albo popraw istniejące. Poprawne obrazy oznaczaj statusem [OK]; "
            "po wyjściu wrócisz bezpośrednio do bramki T06."
        )
        payload["action_text"] = (
            "Po zakończeniu pracy wróć do grafu i podejmij decyzję na T06."
        )
        payload["workflow_start_title"] = "Napraw lub dopisz tablice"
        payload["workflow_start_intro"] = (
            "Pracuj ręcznie na podglądzie albo uruchom autoanotację jako wsparcie. "
            "Nowe poprawne obrazy oznacz [OK], żeby mogły zasilić dalszy krok projektu."
        )
        payload["followup_title"] = "Powrót do T06"
        payload["followup_text"] = (
            "Po zapisaniu zmian wróć do grafu. T06 zdecyduje, czy przejść do treningu, czy zakończyć etap bez treningu."
        )
        payload["export_title"] = "Materiał tablic do dalszej pracy"
        payload["export_text"] = (
            "W tym trybie liczy się aktualna pula zatwierdzonych obrazów [OK] z poprawnymi ramkami tablic."
        )
        payload["auto_plate_model_hint_text"] = (
            "Model tablic jest tylko pomocą naprawczą. Możesz użyć modelu projektu albo wskazać inny model tylko dla tej pracy."
        )
        payload["auto_plate_model_hint_tone"] = "muted"
        return payload

    if gate_id == "T05":
        payload["run_title"] = f"Praca nad otwarciem bramki {CHAR_WORK_GATE_DISPLAY_ID}"
        payload["badge_text"] = f"Aktywna bramka: {CHAR_WORK_GATE_DISPLAY_ID} - {gate_label}"
        payload["badge_tone"] = "success"
        payload["route_tone"] = "muted"
        payload["run_intro_text"] = (
            f"Z2 zostało otwarte z bramki {CHAR_WORK_GATE_DISPLAY_ID}. Celem jest uzupełnienie lub poprawienie ramek tablic, "
            "które będą źródłem dalszej pracy nad znakami."
        )
        payload["route_text"] = (
            "Rysuj nowe ramki tablic albo koryguj istniejące na podglądzie. Poprawne obrazy oznaczaj "
            "statusem [OK] na liście wyników; tylko taka pula zasila dalszą pracę w Z3."
        )
        payload["action_text"] = (
            f"Po uzupełnieniu tablic wróć do mapy kampanii i kontynuuj bramkę {CHAR_WORK_GATE_DISPLAY_ID}. Program użyje "
            "zatwierdzonych ramek jako źródła wyodrębniania tablic i detekcji znaków."
        )
        payload["workflow_start_title"] = "Autoanotacja jako wsparcie"
        payload["workflow_start_intro"] = (
            "Możesz dalej rysować ramki ręcznie albo uruchomić model jako pomoc. "
            "Po autoanotacji sprawdź wynik i oznacz poprawne obrazy statusem [OK]."
        )
        payload["followup_title"] = f"Domknięcie bramki {CHAR_WORK_GATE_DISPLAY_ID}"
        payload["followup_text"] = (
            "Gdy masz wystarczającą pulę zatwierdzonych tablic, wróć do mapy kampanii. "
            f"Dalszy krok prowadzi do {target_label} i pracy nad znakami."
        )
        payload["export_title"] = "Źródło tablic dla pracy nad znakami"
        payload["export_text"] = (
            "W tej bramce najważniejsze są zatwierdzone obrazy [OK] z poprawnymi ramkami tablic. "
            "To one tworzą źródło dla Z3/PZ2."
        )
        payload["auto_plate_model_hint_text"] = (
            "Model tablic jest narzędziem pomocniczym. Możesz użyć modelu projektu albo wskazać "
            "inny model tylko dla bieżącej autoanotacji."
        )
        payload["auto_plate_model_hint_tone"] = "muted"
        return payload

    if gate_id == "T03":
        payload["run_title"] = "Praca nad otwarciem bramki T03"
        payload["badge_text"] = f"Aktywna bramka: T03 - {gate_label}"
        payload["badge_tone"] = "success"
        payload["route_tone"] = "muted"
        payload["run_intro_text"] = (
            "Z2 zostało otwarte z bramki T03. Celem jest przygotowanie zatwierdzonych "
            "anotacji tablic, z których Z3 wyodrębni tablice do pracy nad znakami."
        )
        payload["route_text"] = (
            "Rysuj lub poprawiaj ramki tablic, a poprawne obrazy oznaczaj statusem [OK] "
            "na liście wyników. To status [OK] zasila licznik bramki T03."
        )
        payload["action_text"] = (
            "Gdy bramka T03 będzie otwarta, wróć do mapy kampanii i użyj pola Zatwierdź "
            "na bramce T03. Jeśli chcesz powiększyć źródło znaków, pracuj dalej w tym samym Z2."
        )
        payload["workflow_start_title"] = "Praca nad otwarciem bramki T03"
        payload["workflow_start_intro"] = (
            "Możesz pracować ręcznie albo uruchomić autoanotację jako wsparcie. "
            "Autoanotacja nie zamyka bramki samodzielnie: wynik trzeba sprawdzić i oznaczyć "
            "poprawne obrazy statusem [OK]."
        )
        payload["followup_title"] = "Domknięcie bramki T03"
        payload["followup_text"] = (
            "Po osiągnięciu wymaganego minimum wróć do mapy kampanii. "
            f"Pole Zatwierdź na T03 przenosi dalej do {target_label} i pracy nad znakami."
        )
        payload["export_title"] = "Źródło tablic dla znaków"
        payload["export_text"] = (
            "Eksport datasetu tablic nie jest wymagany do przejścia przez T03. "
            "Najważniejsze są zatwierdzone obrazy [OK] z poprawnymi anotacjami tablic, "
            "bo z nich Z3 przygotuje tablice do oznaczania znaków."
        )
        payload["auto_plate_model_hint_text"] = (
            "Model tablic jest tutaj narzędziem pomocniczym. Możesz użyć modelu projektu "
            "albo wskazać inny model tylko dla bieżącej autoanotacji."
        )
        payload["auto_plate_model_hint_tone"] = "muted"
        return payload

    payload["run_title"] = "Bramka T04: trening modelu tablic"
    payload["badge_text"] = f"Aktywna bramka: T04 - {gate_label}"
    payload["badge_tone"] = "success"
    payload["route_tone"] = "muted"
    payload["run_intro_text"] = (
        "Z2 zostało otwarte z bramki T04. Celem tej pracy jest przygotowanie zatwierdzonych anotacji tablic, "
        f"z których w {target_label} powstanie dataset do treningu modelu tablic."
    )
    payload["route_text"] = (
        "Rysuj lub poprawiaj ramki tablic, a poprawne obrazy oznaczaj statusem [OK] na liście wyników. "
        "To status [OK] zasila licznik bramki T04."
    )
    payload["action_text"] = (
        "Gdy bramka T04 będzie otwarta, wróć do mapy kampanii i użyj pola Zatwierdź na bramce T04. "
        "Jeżeli chcesz tylko dopisać kolejne anotacje, pracuj dalej w tym samym widoku Z2."
    )
    payload["workflow_start_title"] = "Praca nad otwarciem bramki T04"
    payload["workflow_start_intro"] = (
        "Możesz pracować ręcznie albo uruchomić autoanotację jako wsparcie. "
        "Autoanotacja nie zamyka bramki samodzielnie: wynik trzeba sprawdzić i oznaczyć poprawne obrazy statusem [OK]."
    )
    payload["followup_title"] = "Domknięcie bramki T04"
    payload["followup_text"] = (
        "Po osiągnięciu wymaganego minimum wróć do mapy kampanii. "
        "Pole Zatwierdź na T04 przenosi dalej do przygotowania treningu modelu tablic."
    )
    payload["export_title"] = "Dataset tablic dla treningu"
    payload["export_text"] = (
        "Eksport datasetu YOLO Pose jest możliwy dopiero wtedy, gdy bramka T04 ma wystarczającą liczbę "
        "zatwierdzonych obrazów z anotacjami tablic."
    )
    payload["auto_plate_model_hint_text"] = (
        "Model tablic jest tylko narzędziem pomocniczym dla tej pracy. "
        "Możesz użyć modelu projektu albo wskazać inny model wyłącznie dla bieżącej autoanotacji."
    )
    payload["auto_plate_model_hint_tone"] = "muted"
    if (
        str(ctx.route or "").strip().lower() == "manual"
        and str(ctx.manual_entry_mode or "").strip().lower() == "new"
        and int(ctx.campaign_iteration_num or 0) <= 1
    ):
        payload["workflow_start_title"] = "Ręczna praca na tablicach"
        payload["workflow_start_intro"] = (
            "Z2 przygotuje roboczy XML automatycznie, jeśli jeszcze go nie ma. "
            "Możesz od razu rysować ramki tablic; autoanotację uruchamiasz osobno jako wsparcie."
        )
        payload["action_text"] = (
            "Rysuj ramki tablic i oznacz poprawne obrazy statusem [OK]. "
            "Te zatwierdzone obrazy zasilą bramkę T04 i późniejszy trening modelu tablic."
        )
        payload["manual_hint"] = (
            "XML jest technicznym plikiem roboczym tej bramki. Program zapisuje go w obszarze projektu, "
            "więc nie wybierasz osobnej ścieżki i nie zarządzasz nim ręcznie."
        )
        payload["manual_hint_tone"] = "muted"
    if gate_id and display_gate_id and display_gate_id != gate_id:
        for payload_key, payload_value in list(payload.items()):
            if isinstance(payload_value, str):
                payload[payload_key] = payload_value.replace(gate_id, display_gate_id)
    return payload


def build_z2_left_panel_copy_payload_campaign(
    host: "AnnotationTab",
    ctx: Z2LeftPanelCopyContext,
    payload: Z2CopyPayload,
) -> Z2CopyPayload:
    route = str(ctx.route or "")
    actual_route = str(ctx.actual_route or "")
    manual_entry_mode = str(ctx.manual_entry_mode or "")
    manual_setup = bool(ctx.manual_setup)
    manual_import = bool(ctx.manual_import)
    vehicle_assist_enabled = bool(ctx.vehicle_assist_enabled)
    auto_vehicle_choice = str(ctx.auto_vehicle_choice or "")
    current_step = str(ctx.current_step or "")
    plate_model_selected = bool(ctx.plate_model_selected)
    campaign_iteration_target = str(ctx.campaign_iteration_target or "")
    campaign_stage = int(ctx.campaign_stage or 0)
    campaign_iteration_num = int(ctx.campaign_iteration_num or 0)
    campaign_reused_manual_count = int(ctx.campaign_reused_manual_count or 0)
    campaign_char_repair_mode = bool(ctx.campaign_char_repair_mode)
    campaign_manual_skip_count = int(ctx.campaign_manual_skip_count or 0)
    has_existing_run = bool(ctx.has_existing_run)
    manual_review_active = bool(ctx.manual_review_active)
    has_manual_history = bool(ctx.has_manual_history)
    auto_completed = bool(ctx.auto_completed)
    try:
        graph_context = dict(getattr(host, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    t02_at_review = bool(
        str(graph_context.get("z2_work_mode") or "").strip().lower() == "t02_at_review"
        or graph_gate_id == "T02"
    )
    if not route and campaign_stage == 2 and campaign_iteration_target == "char":
        payload["run_title"] = "Anotacja i korekta tablic"
        payload["badge_text"] = "Aktywny tor: przygotowanie źródła dla znaków"
        payload["badge_tone"] = "success"
        payload["route_text"] = (
            "Ten etap przygotowuje anotacje tablic potrzebne później w torze znaków. "
            "Jeśli uruchomisz autoanotację, domyślnie użyje aktywnego modelu tablic projektu."
        )
        payload["action_text"] = (
            "Możesz od razu przejść do anotacji ręcznych tablic i nadać im status [OK]. "
            "Jeśli chcesz, możesz też uruchomić autoanotację na modelu projektu albo podmienić model tylko dla tego runu Z2. "
            "Obrazy edytowane ręcznie i/lub ze statusem [OK] nie będą procesowane przez autoanotację - "
            "model raczej nie poprawi ręcznej korekty."
        )
        payload["workflow_start_title"] = "Anotacja i korekta tablic"
        payload["workflow_start_intro"] = (
            "Możesz od razu przejść do anotacji ręcznych tablic i nadać im status [OK]. "
            "Jeśli chcesz, możesz też uruchomić autoanotację na modelu projektu albo podmienić model tylko dla tego runu Z2. "
            "Obrazy edytowane ręcznie i/lub ze statusem [OK] nie będą procesowane przez autoanotację - "
            "model raczej nie poprawi ręcznej korekty."
        )
    if not plate_model_selected:
        payload["route_text"] = (
            "Na tym etapie dostępna jest ręczna anotacja tablic. "
            "Autoanotacja pojawi się, gdy projekt będzie miał aktywny model tablic."
        )
        payload["action_text"] = (
            "Przygotuj pierwszy zatwierdzony zestaw ręcznie. Po treningu modelu Z2 pokaże też tor autoanotacji."
        )

    if route == "auto":
        payload["run_title"] = "Anotacja i korekta tablic"
        payload["badge_text"] = "Aktywny tor: przygotowanie tablic"
        payload["badge_tone"] = "success"
        payload["route_tone"] = "muted"
        payload["workflow_conf_title"] = "Ustaw pewność detekcji"
        payload["workflow_conf_hint"] = (
            "Ten próg dotyczy bieżącego runu anotacji Z2. Po wyborze modelu tablic możesz od razu go dopasować."
        )
        payload["workflow_vehicle_title"] = "Wskaż model pojazdów (YOLO Box)"
        payload["workflow_vehicle_hint"] = (
            "Ten model jest opcjonalny. Jeśli go pominiesz, run autoanotacji Z2 wykona tylko autoanotację tablic."
        )
        payload["workflow_input_title"] = "Wskaż folder obrazów"
        payload["workflow_input_hint"] = (
            "To jest katalog zdjęć bieżącej iteracji. Opcjonalnie możesz też dołączyć ręcznie anotowane zdjęcia z wcześniejszych iteracji, które mają pozostać widoczne na liście Z2. Model tablic, confidence i opcjonalne boxy pojazdów ustawisz przy starcie autoanotacji w modalu."
        )
        payload["workflow_start_title"] = "Autoanotacja bieżącego runu"
        payload["workflow_start_intro"] = (
            "Masz otwarty roboczy run Z2. Możesz uruchomić autoanotację jako wsparcie korekty albo dalej pracować ręcznie "
            "na liście i podglądzie. Przycisk startu otworzy modal wyboru zakresu, modelu tablic, confidence "
            "oraz opcjonalnych boxów pojazdów."
        )
        payload["auto_plate_model_hint_text"] = (
            "Aktywny model projektu został już podstawiony do tego kroku. "
            "Możesz zostawić go bez zmian albo wskazać inny model tylko dla bieżącego runu Z2. "
            "Sama podmiana w Z2 nie zmieni modelu projektu."
        )

        if campaign_iteration_target == "char":
            payload["workflow_start_title"] = "Anotacja i korekta tablic"
            payload["workflow_start_intro"] = (
                "Możesz od razu przejść do anotacji ręcznych tablic i nadać im status [OK]. "
                "Jeśli chcesz, możesz też uruchomić autoanotację na modelu projektu albo podmienić model tylko dla tego runu Z2. "
                "Obrazy edytowane ręcznie i/lub ze statusem [OK] nie będą procesowane przez autoanotację - "
                "model raczej nie poprawi ręcznej korekty."
            )

        if not plate_model_selected:
            payload["route_text"] = "Model tablic wybierzesz przy starcie autoanotacji."
            payload["action_text"] = "Kliknij Start, a w modalu wskażesz zakres pracy i model dla bieżącego runu Z2."
            payload["workflow_start_intro"] = (
                "Autoanotacja jest dostępna jako wsparcie pracy na bieżącym runie Z2. "
                "Przycisk startu otworzy modal wyboru zakresu obrazów, wymaganego modelu tablic, confidence "
                "oraz opcjonalnego wsparcia modelem pojazdów. Po zakończeniu wynik sprawdzisz i poprawisz "
                "na liście oraz podglądzie Z2."
            )
            payload["auto_plate_model_hint_text"] = (
                "Na tym etapie możesz wskazać aktywny model projektu albo podmienić go na inny model dla bieżącego runu Z2. "
                "Sama podmiana w Z2 nie zmieni modelu projektu."
            )
        elif auto_vehicle_choice == "skip":
            payload["route_text"] = "Najpierw uruchomisz autoanotację samych tablic na obrazach widocznych na liście wyników anotacji, a potem sprawdzisz i poprawisz wynik ręcznie w tym samym Z2."
            payload["action_text"] = "Po zapisaniu runu Z2 od razu otworzy listę wyników anotacji i podgląd do ręcznej korekty polygonów tablic dla obrazów z bieżącego katalogu. Jeśli wolisz, możesz też pominąć autoanotację i przejść od razu do ręcznej anotacji tych obrazów."
            if campaign_iteration_target == "char":
                payload["action_text"] += " W tym torze domyślnie użyty zostanie aktywny model tablic projektu, ale w kroku wyboru modelu możesz wskazać inny tylko dla tego runu."
        else:
            payload["route_text"] = "Najpierw uruchomisz autoanotację tablic i pojazdów na obrazach widocznych na liście wyników anotacji, a potem sprawdzisz i poprawisz wynik ręcznie w tym samym Z2."
            payload["action_text"] = "Po zapisaniu runu Z2 od razu otworzy listę wyników anotacji i podgląd do ręcznej korekty polygonów tablic dla obrazów z bieżącego katalogu. Jeśli wolisz, możesz też pominąć autoanotację i przejść od razu do ręcznej anotacji tych obrazów."
            if campaign_iteration_target == "char":
                payload["action_text"] += " W tym torze domyślnie użyty zostanie aktywny model tablic projektu, ale w kroku wyboru modelu możesz wskazać inny tylko dla tego runu."
            payload["auto_choice_hint"] = "Odznaczone pole oznacza wariant: pojazdy + tablice."

        payload["followup_title"] = "Po uruchomieniu przejdziesz do ręcznej korekty"
        payload["followup_text"] = (
            "Ten tor nie kończy się na samym uruchomieniu modeli. Po zapisaniu runu Z2 od razu sprawdzisz wynik, "
            "poprawisz polygony tablic ręcznie i dopiero potem domkniesz E2. To obejmuje również zdjęcia dołączone checkboxem z wcześniejszych iteracji."
        )
        payload["workflow_vehicle_title"] = "Wskaz model pojazdow do wsparcia tablic (YOLO Box)"
        payload["workflow_vehicle_hint"] = (
            "Ten model jest opcjonalny. Sluzy tylko do zawezenia szukania tablic do obszaru pojazdu. "
            "Boxy pojazdow sa pomocnicze i nie trafiaja do finalnego eksportu YOLO."
        )
        if auto_vehicle_choice == "skip":
            payload["auto_choice_hint"] = "Zaznaczone pole oznacza wariant: tylko tablice."
        else:
            payload["route_text"] = "Najpierw uruchomisz autoanotacje tablic ze wsparciem wykrywania pojazdow na obrazach widocznych na liscie wynikow anotacji, a potem sprawdzisz i poprawisz wynik recznie w tym samym Z2."
            payload["action_text"] = "W tym wariancie najpierw wykrywany jest pojazd, a model tablic szuka tablic tylko w jego obrebie. Boxy pojazdow pozostaja pomocnicze i nie trafiaja do finalnego YOLO."
            if campaign_iteration_target == "char":
                payload["action_text"] += " W tym torze domyslnie uzyty zostanie aktywny model tablic projektu, ale w kroku wyboru modelu mozesz wskazac inny tylko dla tego runu."
            payload["auto_choice_hint"] = "Odznaczone pole oznacza wariant: wsparcie pojazdami dla tablic."

        live_manual_skip_count = max(
            int(campaign_manual_skip_count or 0),
            len(host._collect_preview_manually_touched_filenames()),
        )
        if live_manual_skip_count > 0:
            payload["followup_text"] += (
                f" Obrazy już poprawione ręcznie w tym Z2 są automatycznie pomijane przy autoanotacji ({live_manual_skip_count})."
            )
        payload["export_text"] = "Split i eksport datasetu są kolejnym krokiem dopiero na gotowym, sprawdzonym runie Z2."

        if campaign_char_repair_mode:
            _apply_campaign_char_repair_copy_payload(payload)

    elif route == "manual":
        payload["run_title"] = (
            "Anotacja i korekta tablic" if manual_setup and not manual_review_active else "Korekta tablic"
        )
        payload["badge_text"] = "Aktywny etap: anotacja i korekta tablic"
        payload["badge_tone"] = "warning"
        payload["route_tone"] = "muted"
        payload["followup_title"] = "3. Ręczna korekta i stage"
        payload["followup_text"] = (
            "W tym torze pracujesz bez mieszania z autoanotacją. Po otwarciu runu anotacji możesz kasować obrazy, przenosić je do stage i poprawiać polygony."
        )
        payload["export_text"] = "Po zapisaniu zmian domkniesz E2, a dataset i trening wykonasz potem w [Z4]."
        payload["workflow_input_title"] = "2. Wskaż folder obrazów"
        payload["workflow_input_hint"] = "Najpierw wybierz folder obrazów. Ten folder będzie bazą nowego ręcznego runu anotacji Z2."

        if manual_review_active:
            if campaign_char_repair_mode:
                _apply_campaign_char_repair_copy_payload(payload, manual_review_active=True)
            else:
                payload["route_text"] = "Korygujesz bieżący run Z2 tej iteracji."
                payload["action_text"] = "Po prawej poprawiasz polygony aktywnego runu i po zakończeniu zmian domykasz E2."
                payload["workflow_start_intro"] = "Tutaj wracasz do aktywnego runu tej iteracji i poprawiasz jego wynik ręcznie."
                payload["manual_hint"] = "To nie jest osobny mini-workflow. Z2 jest już otwarte w trybie korekty aktywnego runu kampanii."
                payload["manual_hint_tone"] = "muted"
                payload["workflow_start_title"] = "Aktywna korekta runu Z2"
                payload["workflow_input_title"] = "Aktywny run kampanii"
                payload["workflow_input_hint"] = "Bieżący run Z2 jest już wczytany i gotowy do poprawiania."
        elif current_step == "manual_entry":
            if manual_import:
                payload["manual_hint"] = (
                    "Po kliknięciu Dalej wskażesz dowolny run Z2 do korekty. "
                    "Jeśli leży poza workspace, program bezpiecznie skopiuje go do lokalnego importu."
                )
            elif manual_entry_mode == "continue":
                payload["manual_hint"] = (
                    "Po kliknięciu Dalej przejdziesz do historii lokalnych runów Z2 "
                    "i wybierzesz run do wznowienia korekty."
                )
            else:
                payload["manual_hint"] = (
                    "Po kliknięciu Dalej przejdziesz do tworzenia nowego runu ręcznego Z2."
                )
            payload["manual_hint_tone"] = "muted"

        if manual_entry_mode == "continue":
            if current_step == "manual_history":
                payload["workflow_start_title"] = "Otwórz run anotacji do korekty"
                payload["route_text"] = "Na tym etapie wybierasz run Z2 z lokalnej historii i otwierasz go do dalszej pracy ręcznej."
                payload["action_text"] = (
                    "Zaznacz run Z2 z historii i kliknij Dalej, aby otworzyć go w edytorze."
                    if has_manual_history
                    else "Historia jest pusta. Wróć i wybierz nowy run ręczny albo wskaż dowolny run Z2."
                )
                payload["manual_hint"] = (
                    "Podgląd pozostaje wyłączony, dopóki nie otworzysz konkretnego runu Z2 z historii."
                    if has_manual_history
                    else "Jeśli nie masz jeszcze lokalnej historii runów, cofnij się i wybierz inny sposób wejścia."
                )
                payload["manual_hint_tone"] = "muted" if has_manual_history else "warning"
                payload["manual_template_hint"] = (
                    "Run Z2 do kontynuacji to katalog z annotations.xml i zgodnymi obrazami. "
                    f"Domyślny katalog runów Z2: {host._get_annotation_run_storage_display_path()}."
                )
                payload["manual_template_tone"] = "muted"
        elif manual_import:
            payload["route_text"] = "Wybrano tor wskazania dowolnego runu Z2 do korekty."
            payload["action_text"] = "Kliknij Dalej, aby wskazać run Z2 i ewentualnie zaimportować go do lokalnego workspace."
            payload["manual_hint"] = (
                "Program przyjmie tylko run Z2 z annotations.xml i zgodnymi obrazami. "
                "Jeśli taki run leży poza workspace, zostanie bezpiecznie skopiowany do lokalnego runu import_*."
            )
            payload["manual_hint_tone"] = "muted"
            payload["manual_template_hint"] = (
                f"Domyślny katalog runów Z2: {host._get_annotation_run_storage_display_path()}."
            )
            payload["manual_template_tone"] = "muted"
        else:
            payload["workflow_input_title"] = "Wskaż folder obrazów i opcje pomocy"
            payload["workflow_input_hint"] = (
                "Tutaj wybierasz obrazy dla nowego runu ręcznej anotacji Z2 oraz opcjonalnie włączasz pomocnicze boxy pojazdów."
            )
            payload["workflow_start_title"] = (
                "Otwórz bieżący run anotacji tablic"
                if campaign_reused_manual_count > 0
                else "Ręczna praca na tablicach"
            )
            payload["workflow_start_intro"] = (
                "Z2 przygotuje roboczy XML dla katalogu zdjęć z E1 automatycznie. "
                "Możesz rysować i poprawiać ramki tablic oraz oznaczać poprawne obrazy statusem [OK]."
            )
            payload["route_text"] = (
                "Pracujesz na katalogu zdjęć wybranym w E1. Wcześniejsze ręczne korekty zostaną zachowane."
                if campaign_reused_manual_count > 0
                else "Katalog zdjęć tej iteracji jest przygotowany do pracy w Z2. Nie musisz wskazywać dodatkowego źródła ani otwierać osobnego runu."
            )
            payload["action_text"] = (
                "Kliknij przycisk poniżej, aby wrócić do pracy na bieżącym runie. Wcześniejsze poprawki pozostaną zachowane."
                if campaign_reused_manual_count > 0
                else "Pracuj od razu na podglądzie Z2. Gdy oznaczysz poprawne obrazy jako [OK], zasilą one aktywną bramkę grafu."
            )
            payload["manual_hint"] = (
                "Ten tor pracuje już na jednym runie tej iteracji, więc nie wymaga ręcznego zarządzania osobnym XML-em."
                if campaign_reused_manual_count > 0
                else "Z2 zapisuje wynik w runie tej iteracji. Użytkownik nie musi ręcznie zarządzać katalogami runu."
            )
            payload["manual_hint_tone"] = "muted"
            payload["manual_template_hint"] = (
                "Po wejściu do pracy wcześniejsze ręczne oznaczenia pozostają zachowane, a dalsze zmiany zapiszą się w XML tej iteracji."
                if campaign_reused_manual_count > 0
                else "Wynik pracy zapisze się w workspace Z2 dla tej iteracji. Nie musisz ręcznie zarządzać plikami runu."
            )
            payload["manual_template_tone"] = "muted"
            payload["manual_vehicle_hint"] = "Opcjonalne boxy pojazdów są tylko pomocą przy ręcznej pracy."
            payload["manual_vehicle_tone"] = "muted"
            if campaign_iteration_num == 1 and not campaign_reused_manual_count:
                payload["run_intro_text"] = (
                    "To jest pierwsze bazowe przygotowanie anotacji tablic w projekcie. "
                    "Z2 przygotuje plik roboczy automatycznie; Ty oznaczasz ramki tablic i nadajesz poprawnym obrazom status [OK]."
                )
                payload["route_text"] = (
                    "To jest pierwszy bazowy zestaw anotacji tablic dla projektu. "
                    "W tej iteracji przygotowujesz pierwsze poprawne tablice do odblokowania E2."
                )
                payload["action_text"] = (
                    "Narysuj lub popraw tablice ręcznie i oznacz poprawne obrazy jako [OK]. "
                    "To otworzy właściwą bramkę grafu."
                )
            if vehicle_assist_enabled:
                payload["workflow_conf_title"] = "Pewność pomocniczych boxów pojazdów"
                payload["workflow_conf_hint"] = (
                    "Ten próg dotyczy tylko pomocniczego boxowania pojazdów w nowym XML."
                )
                payload["workflow_vehicle_title"] = "Model pojazdów do pomocniczego boxowania"
                payload["workflow_vehicle_hint"] = (
                    "Model pojazdów posłuży tylko jako wsparcie przy ręcznym rysowaniu tablic."
                )

        if not str(payload.route_text or "").strip():
            payload["route_text"] = (
                "Pracujesz ręcznie na katalogu zdjęć tej iteracji. "
                "Możesz przygotować nowe oznaczenia albo poprawiać aktywny run Z2."
            )
        if not str(payload.action_text or "").strip():
            payload["action_text"] = (
                "Po prawej stronie wykonujesz właściwą pracę ręczną na obrazach tej iteracji, "
                "a zatwierdzone wyniki zasilą projektowy zbiór danych do treningu."
            )
        if not str(payload.workflow_start_intro or "").strip():
            payload["workflow_start_intro"] = "To jest główny krok roboczy tej części Z2."
        if (
            not str(payload.run_intro_text or "").strip()
            and route == "manual"
            and not manual_review_active
            and manual_setup
            and int(campaign_iteration_num or 0) <= 1
        ):
            payload["run_intro_text"] = (
                "Tutaj przygotowujesz pierwszy bazowy zestaw tablic dla projektu. "
                "Na katalogu zdjęć tej iteracji ręcznie ustawiasz rogi tablic i budujesz startowy run Z2, "
                "który później zatwierdzi E2 i posłuży do dalszego treningu."
            )

    if t02_at_review:
        payload["run_title"] = "Kontrola AT w Z2"
        payload["run_intro_text"] = (
            "Kontrolujesz anotacje tablic z importu lub zatwierdzonej puli projektu w bramce T02."
        )
        payload["badge_text"] = "Bramka T02: kontrola AT względem aktualnego zbioru O"
        payload["badge_tone"] = "warning"
        payload["route_text"] = (
            "Z2 pokazuje run importu AT. Sprawdź ramki tablic tylko dla pasujących obrazów "
            "i oznacz [OK] pozycje, które mają zasilić projektowe źródło tablic."
        )
        payload["action_text"] = (
            "Po zakończeniu wróć do T02. Bramka nie zostanie zatwierdzona automatycznie; "
            "dopiero decyzja na grafie zamyka ten skrót i prowadzi do pracy nad znakami."
        )
        payload["workflow_start_title"] = "Kontroluj AT"
        payload["workflow_start_intro"] = (
            "Popraw lub zaakceptuj ramki tablic, a poprawnym obrazom nadaj status [OK]. "
            "Te [OK] są realnym wynikiem kontroli T02."
        )

    return _apply_campaign_graph_gate_copy_payload(host, ctx, payload)
