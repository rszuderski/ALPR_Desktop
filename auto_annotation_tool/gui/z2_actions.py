#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lekkie obiekty akcji procesu Z2.

To nie jest pełny framework workflow. To minimalna warstwa pośrednia
między logiką domenową Z2 a rendererami kampanii / free mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Z2ActionContext:
    mode: str
    campaign_step: int
    iteration_target: str
    campaign_repair_mode: bool
    route: str
    input_dir: str
    has_plate_model: bool
    has_existing_run: bool
    has_manual_review: bool
    has_export_ready_run: bool
    is_processing: bool
    auto_vehicle_choice: str


class Z2Action:
    id: str = ""
    label: str = ""

    def is_available(self, ctx: Z2ActionContext) -> bool:
        return True

    def is_enabled(self, ctx: Z2ActionContext) -> bool:
        return not ctx.is_processing

    def get_description(self, ctx: Z2ActionContext) -> str:
        return ""

    def activate(self, host, ctx: Z2ActionContext) -> None:
        raise NotImplementedError


def _is_manual_template_run_manifest(manifest: dict | None) -> bool:
    if not isinstance(manifest, dict):
        return False
    if bool(manifest.get("manual_xml_template", False)):
        return True
    return str(manifest.get("annotation_run_type") or "").strip().lower() == "manual_template"


class PlateManualAction(Z2Action):
    id = "plate_manual"

    def is_available(self, ctx: Z2ActionContext) -> bool:
        if ctx.mode == "campaign" and ctx.campaign_repair_mode:
            return True
        if ctx.mode == "campaign" and ctx.campaign_step == 2 and ctx.has_plate_model:
            if ctx.iteration_target == "char":
                return False
            if ctx.iteration_target == "plate":
                return True
        return True

    label = "Ręczna anotacja tablic"

    def get_description(self, ctx: Z2ActionContext) -> str:
        if ctx.mode == "campaign":
            if ctx.campaign_repair_mode:
                return (
                    "Otwiera Z2 w trybie ręcznej naprawy: możesz poprawić albo dopisać ramki tablic "
                    "dla obrazów przypisanych do aktywnej bramki."
                )
            return (
                "Ręczna praca jest dostępna od razu. Jeśli roboczy XML jeszcze nie istnieje, "
                "program przygotuje go automatycznie w tle, bez osobnego kroku użytkownika."
            )
        return (
            "Kontynuuj istniejący XML albo utwórz nowy run anotacji do ręcznego oznaczania tablic."
        )

    def activate(self, host, ctx: Z2ActionContext) -> None:
        if ctx.mode == "campaign":
            preferred_run_dir = None
            try:
                preferred_run_dir = host._get_preferred_annotation_run_dir(require_xml=True)
            except Exception:
                preferred_run_dir = None
            if preferred_run_dir is not None:
                try:
                    manifest = host._load_annotation_run_manifest(preferred_run_dir)
                except Exception:
                    manifest = {}
                if not _is_manual_template_run_manifest(manifest):
                    same_active_run = False
                    try:
                        current_run_dir = getattr(host, "current_annotation_run_dir", None)
                        same_active_run = (
                            current_run_dir is not None
                            and host._paths_equivalent(current_run_dir, preferred_run_dir)
                            and bool(getattr(host, "current_annotations", []) or [])
                        )
                    except Exception:
                        same_active_run = False
                    if same_active_run:
                        try:
                            if not host._ensure_preview_edits_saved("przełączenie do ręcznej korekty"):
                                return
                        except Exception:
                            pass
                        try:
                            host.workflow_route_var.set("manual")
                            host.manual_entry_mode_var.set("continue")
                            host.manual_xml_template_var.set(False)
                            host.manual_vehicle_assist_var.set(False)
                            host._manual_review_active = True
                            host._manual_review_from_auto = True
                            host._manual_review_export_ready = False
                            host._dataset_export_completed = False
                            host._last_completed_workflow_route = "manual"
                            if getattr(host, "current_input_dir", None) is not None:
                                host.input_dir_var.set(str(host.current_input_dir))
                            host._refresh_preview_list(preserve_selection=True, render_current=True)
                            host._refresh_preview_list_summary()
                            host._refresh_left_panel_route_copy()
                            host._refresh_detection_configuration_ui()
                            host._refresh_run_output_info()
                            host._refresh_step2_action_states()
                            host._refresh_free_mode_workflow_ui()
                        except Exception:
                            pass
                        try:
                            host.app.update_status(
                                "W kampanii przełączono istniejący run auto Z2 do ręcznej korekty.",
                                "info",
                            )
                        except Exception:
                            pass
                        return
                    try:
                        if not host._ensure_preview_edits_saved("przełączenie do ręcznej korekty"):
                            return
                    except Exception:
                        pass
                    if host._open_existing_run_for_campaign_review(
                        run_dir=preferred_run_dir,
                        iteration_target=ctx.iteration_target or "plate",
                        manual_template=False,
                    ):
                        try:
                            host._manual_review_from_auto = True
                            host._last_completed_workflow_route = "manual"
                            host._refresh_left_panel_route_copy()
                            host._refresh_detection_configuration_ui()
                            host._refresh_step2_action_states()
                            host._refresh_free_mode_workflow_ui()
                        except Exception:
                            pass
                        try:
                            host.app.update_status(
                                "W kampanii otwarto ręczną korektę istniejącego runu autoanotacji w Z2.",
                                "info",
                            )
                        except Exception:
                            pass
                        return
            host._apply_campaign_step2_workflow_preset(
                iteration_target=ctx.iteration_target or "plate",
                manual_template=True,
            )
            input_dir_value = str(ctx.input_dir or "").strip()
            try:
                if input_dir_value and not getattr(host, "current_annotation_xml_path", None):
                    host._prime_campaign_source_preview(Path(input_dir_value))
            except Exception:
                pass

            try:
                input_dir_ready = bool(input_dir_value and Path(input_dir_value).exists())
            except Exception:
                input_dir_ready = False

            if input_dir_ready and not bool(getattr(host, "_campaign_manual_prepare_pending", False)):
                try:
                    host._campaign_manual_prepare_pending = True

                    def _prepare_campaign_manual_package():
                        try:
                            host._ensure_campaign_manual_package_ready(
                                iteration_target=ctx.iteration_target or "plate"
                            )
                        finally:
                            host._campaign_manual_prepare_pending = False

                    host.frame.after_idle(_prepare_campaign_manual_package)
                except Exception:
                    host._campaign_manual_prepare_pending = False

            try:
                host.app.update_status(
                    "W kampanii wybrano ręczną anotację obrazów widocznych na liście wyników anotacji w Z2.",
                    "info",
                )
            except Exception:
                pass
            return

        host._select_free_mode_route("manual")


class PlateAutoAction(Z2Action):
    id = "plate_auto"
    label = "Autoanotacja tablic"

    def is_available(self, ctx: Z2ActionContext) -> bool:
        if ctx.mode == "campaign":
            return True
        return True

    def is_enabled(self, ctx: Z2ActionContext) -> bool:
        if ctx.is_processing:
            return False
        if ctx.mode == "campaign":
            return True
        return True

    def get_description(self, ctx: Z2ActionContext) -> str:
        if ctx.mode == "campaign":
            if ctx.campaign_repair_mode:
                return (
                    "Dostepna takze w trybie naprawczym. Mozesz uzyc modelu tablic i opcjonalnej asysty pojazdow, "
                    "aby szybciej powiekszyc albo poprawic zrodlo tablic przed powrotem do E3."
                )
            if ctx.campaign_repair_mode:
                return (
                    "W tym trybie naprawczym autoanotacja projektowym modelem jest ukryta celowo. "
                    "Powrót z E3 służy ręcznej naprawie źródła tablic dla tego samego zestawu zdjęć."
                )
            if ctx.has_plate_model:
                return (
                    "Opcja dodatkowa. Uruchamia model tablic na obrazach widocznych na liście wyników anotacji, "
                    "a potem od razu otwiera wynik do ręcznej korekty w tym samym Z2."
                )
            return (
                "Możesz uruchomić autoanotację nawet bez aktywnego modelu projektu. "
                "Przy starcie wybierzesz wtedy model tylko dla tego runu Z2 albo od razu ustawisz go jako model projektu."
            )
        return "Uruchom YOLO, zapisz run anotacji Z2 w workspace i przejdź potem do korekty oraz splitu."

    def activate(self, host, ctx: Z2ActionContext) -> None:
        if not self.is_enabled(ctx):
            return

        if ctx.mode == "campaign":
            manual_overlay_bundle = {}
            plate_model_meta = {}
            try:
                plate_model_meta = dict(host._get_effective_plate_model_runtime_meta() or {})
                selected_plate_model_path = str(plate_model_meta.get("path") or "").strip()
            except Exception:
                selected_plate_model_path = ""
            if not selected_plate_model_path:
                try:
                    selected_plate_model_path = str(host.plate_custom_var.get() or "").strip()
                except Exception:
                    selected_plate_model_path = ""

            def _restore_selected_plate_model() -> None:
                if not selected_plate_model_path:
                    return
                try:
                    if not Path(selected_plate_model_path).exists():
                        return
                except Exception:
                    return
                try:
                    host.plate_custom_var.set(selected_plate_model_path)
                    host._remember_plate_model_runtime_meta(
                        model_path=Path(selected_plate_model_path),
                        source=str(plate_model_meta.get("source") or "external"),
                        scope=str(plate_model_meta.get("scope") or "run"),
                    )
                    host._refresh_plate_model_runtime_info_ui()
                except Exception:
                    pass

            try:
                manual_overlay_bundle = dict(host._get_current_campaign_manual_preview_bundle() or {})
            except Exception:
                manual_overlay_bundle = {}
            if manual_overlay_bundle:
                host._campaign_auto_manual_overlay_bundle = manual_overlay_bundle
            host._apply_campaign_step2_workflow_preset(
                iteration_target=ctx.iteration_target or "plate",
                manual_template=False,
            )
            _restore_selected_plate_model()
            try:
                input_dir_value = str(ctx.input_dir or "").strip()
                if input_dir_value:
                    host._prime_campaign_source_preview(Path(input_dir_value))
            except Exception:
                pass
            if manual_overlay_bundle:
                try:
                    host._campaign_pending_batch_summary = dict(getattr(host, "_campaign_pending_batch_summary", {}) or {})
                    host._campaign_pending_batch_summary["current_manual_count"] = len(manual_overlay_bundle)
                except Exception:
                    pass
                try:
                    host._refresh_preview_list_summary()
                except Exception:
                    pass
                try:
                    host._refresh_step2_action_states()
                except Exception:
                    pass
            try:
                host._enforce_campaign_plate_only_auto_default()
            except Exception:
                pass
            _restore_selected_plate_model()
            try:
                host.app.update_status(
                    "W kampanii wybrano autoanotację obrazów widocznych na liście wyników anotacji w Z2.",
                    "info",
                )
            except Exception:
                pass
            def _start_campaign_auto_annotation():
                try:
                    if getattr(host, "is_processing", False):
                        return
                    try:
                        host.frame.update_idletasks()
                    except Exception:
                        pass
                    try:
                        host._set_workflow_route_state("auto", campaign_context=True)
                        host.manual_xml_template_var.set(False)
                        host._set_workflow_step_state("auto_start", campaign_context=True)
                        host._set_auto_vehicle_choice_state(host._get_auto_vehicle_choice(), campaign_context=True)
                    except Exception:
                        pass
                    _restore_selected_plate_model()
                    try:
                        host.app.update_status(
                            "Otwieram modal autoanotacji Z2 dla aktywnej bramki.",
                            "info",
                        )
                    except Exception:
                        pass
                    host._start_annotation()
                except Exception as exc:
                    try:
                        host.app.update_status(
                            f"Nie udało się uruchomić autoanotacji Z2: {exc}",
                            "error",
                        )
                    except Exception:
                        pass

            _start_campaign_auto_annotation()
            return

        host._select_free_mode_route("auto")


class PlateReviewAction(Z2Action):
    id = "plate_review"
    label = "Korekta istniejącego runu"

    def is_available(self, ctx: Z2ActionContext) -> bool:
        return bool(ctx.has_existing_run)

    def is_enabled(self, ctx: Z2ActionContext) -> bool:
        return bool(ctx.has_existing_run and not ctx.is_processing)

    def get_description(self, ctx: Z2ActionContext) -> str:
        return "Otwiera istniejący run Z2 do ręcznej korekty bez tworzenia nowego przebiegu."

    def activate(self, host, ctx: Z2ActionContext) -> None:
        if not self.is_enabled(ctx):
            return

        if ctx.mode == "campaign":
            run_dir = host._get_preferred_annotation_run_dir(require_xml=True)
            if run_dir is None:
                return
            host._open_existing_run_for_campaign_review(
                run_dir=run_dir,
                iteration_target=ctx.iteration_target or "plate",
                manual_template=False,
            )
            try:
                host.app.update_status("Otworzono istniejący run Z2 do korekty.", "info")
            except Exception:
                pass
            return

        from_auto = False
        try:
            current_screen = str(host._coerce_free_mode_screen() or "").strip().lower()
        except Exception:
            current_screen = ""
        try:
            thematic_route = str(host._get_z2_thematic_route() or "").strip().lower()
        except Exception:
            thematic_route = ""

        if thematic_route == "auto" and current_screen in {"auto_summary", "manual_review", "export"}:
            from_auto = True
        elif str(ctx.route or "").strip().lower() == "auto":
            from_auto = True

        host._open_existing_run_for_manual_review(show_dialog=False, from_auto=from_auto)


class PlateExportAction(Z2Action):
    id = "plate_export"
    label = "Split i eksport datasetu"

    def is_available(self, ctx: Z2ActionContext) -> bool:
        if ctx.mode == "campaign":
            return False
        return bool(ctx.has_existing_run)

    def is_enabled(self, ctx: Z2ActionContext) -> bool:
        return bool(ctx.has_export_ready_run and not ctx.is_processing)

    def get_description(self, ctx: Z2ActionContext) -> str:
        if ctx.has_export_ready_run:
            return "Przechodzi do przygotowania splitu i eksportu datasetu na gotowym runie Z2."
        return "Akcja odblokuje się, gdy run Z2 będzie miał co najmniej jedną tablicę oznaczoną statusem OK."

    def activate(self, host, ctx: Z2ActionContext) -> None:
        if not self.is_enabled(ctx):
            return
        host._jump_to_export_section()


def build_z2_primary_actions() -> dict[str, Z2Action]:
    return {
        "auto": PlateAutoAction(),
        "manual": PlateManualAction(),
    }


def build_z2_secondary_actions() -> dict[str, Z2Action]:
    return {
        "review": PlateReviewAction(),
        "export": PlateExportAction(),
    }
