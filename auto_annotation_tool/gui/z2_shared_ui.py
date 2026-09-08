from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

from .z2_flow_models import Z2CtaState, Z2WorkflowBaseContext
from .web_slim_scrollbar import blend_hex_colors

if TYPE_CHECKING:
    from .tab_annotation import AnnotationTab


CAMPAIGN_GATE_ID_BY_EDGE = {
    "e1_to_e2": "T01",
    "e1_to_e2_plate_training": "T01",
    "e1_to_e2_char_from_images": "T01",
    "e1_to_e3": "T02",
    "e2_to_e3": "T03",
    "e2_to_e4": "T04",
    "e3_to_e4": "T05",
    "e4_to_e1": "T06",
    "e4t_to_e1": "T06",
    "e4z_to_e1": "T06",
}


def campaign_visible_gate_id(gate_id: str | None) -> str:
    """Return the canonical gate id used by the current graph.

    The old UI translated bare ids like T04 -> T03. That is unsafe now,
    because T03/T04/T05 are real, current ids. If a stale context needs
    repair, use the edge-aware helper below.
    """
    normalized = str(gate_id or "").strip().upper()
    return normalized


def campaign_gate_id_for_edge(edge_key: str | None, fallback_gate_id: str | None = None) -> str:
    normalized_edge = str(edge_key or "").strip()
    if normalized_edge:
        resolved = CAMPAIGN_GATE_ID_BY_EDGE.get(normalized_edge)
        if resolved:
            return resolved
    return campaign_visible_gate_id(fallback_gate_id)


def short_campaign_quality_label(label: str) -> str:
    normalized = str(label or "").strip().upper()
    if normalized == "BARDZO DOBRY":
        return "B. DOBRY"
    if normalized == "PRZECIĘTNY":
        return "PRZEC."
    return normalized or "PROGU"


def build_campaign_gate_focus_state(
    missing_to_open: int,
    quality_info: dict | None,
    *,
    unit_label: str = "tablic",
) -> dict[str, object]:
    """Return concise, unambiguous Z2 gate/quality counter copy.

    The campaign gate minimum is a hard condition. Dataset quality thresholds
    are optional guidance and must not be described as gate-opening blockers.
    """
    try:
        missing_open = max(0, int(missing_to_open or 0))
    except Exception:
        missing_open = 0
    if missing_open > 0:
        return {
            "label": "DO MIN.",
            "row_label": "Do otwarcia bramki brakuje",
            "value": int(missing_open),
            "text": f"{missing_open} {unit_label} [OK]",
            "tone": "warning",
        }

    info = dict(quality_info or {})
    next_quality_label = str(info.get("next_label", "") or "").strip()
    try:
        missing_next = max(0, int(info.get("missing_next", 0) or 0))
    except Exception:
        missing_next = 0
    if missing_next > 0 and next_quality_label:
        return {
            "label": f"CEL {short_campaign_quality_label(next_quality_label)}",
            "row_label": f"Cel jakości: {next_quality_label}",
            "value": int(missing_next),
            "text": f"+{missing_next} {unit_label} [OK]",
            "tone": "info",
        }

    return {
        "label": "PROGI",
        "row_label": "Progi jakości",
        "value": 0,
        "text": "Najwyższy próg jakości",
        "tone": "success",
    }


def format_campaign_quality_goal_value(
    quality_info: dict | None,
    *,
    unit_label: str = "tablic",
) -> tuple[str, str]:
    info = dict(quality_info or {})
    next_quality_label = str(info.get("next_label", "") or "").strip()
    try:
        missing_next = max(0, int(info.get("missing_next", 0) or 0))
    except Exception:
        missing_next = 0
    if missing_next > 0 and next_quality_label:
        return f"+{missing_next} {unit_label} [OK]", "info"
    return "MAX", "success"


def get_campaign_display_gate_id(host: "AnnotationTab") -> str:
    try:
        graph_context = dict(getattr(host, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    return campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )


def is_campaign_t06_plate_repair_context(host: "AnnotationTab") -> bool:
    try:
        graph_context = dict(getattr(host, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    repair_origin_gate_id = campaign_gate_id_for_edge(
        graph_context.get("repair_origin_edge_key") or graph_context.get("source_graph_edge_key"),
        graph_context.get("repair_origin_gate_id") or graph_context.get("source_graph_gate_id"),
    )
    return bool(graph_gate_id == "T04" and repair_origin_gate_id == "T06")


def is_campaign_t05_char_source_context(host: "AnnotationTab") -> bool:
    try:
        graph_context = dict(getattr(host, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    return graph_gate_id == "T05"


def is_campaign_t03_char_route_context(host: "AnnotationTab") -> bool:
    try:
        graph_context = dict(getattr(host, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    return graph_gate_id == "T03"


def is_campaign_t02_at_review_context(host: "AnnotationTab") -> bool:
    try:
        graph_context = dict(getattr(host, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    return bool(
        str(graph_context.get("z2_work_mode") or "").strip().lower() == "t02_at_review"
        or graph_gate_id == "T02"
    )


def get_campaign_return_to_graph_copy(host: "AnnotationTab") -> dict[str, object]:
    try:
        graph_context = dict(getattr(host, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    display_gate_id = campaign_visible_gate_id(graph_gate_id)
    if is_campaign_t02_at_review_context(host):
        return {
            "section": " Kontrola importu AT ",
            "button": "Zapisz kontrolę AT i wróć do bramki T02",
            "width": 42,
        }
    if is_campaign_t06_plate_repair_context(host):
        return {
            "section": " Przekazanie do puli YOLO ",
            "button": "Przekaż [OK] do puli YOLO i wróć do grafu",
            "width": 32,
        }
    if is_campaign_t05_char_source_context(host):
        return {
            "section": " Przekazanie do źródła Z3 ",
            "button": "Przekaż [OK] do Z3 i wróć do grafu",
            "width": 34,
        }
    if is_campaign_t03_char_route_context(host):
        return {
            "section": " Powrót do bramki T03 ",
            "button": "Zapisz [OK] i wróć do bramki T03",
            "width": 38,
        }
    if graph_gate_id == "T04":
        return {
            "section": f" Przekazanie [OK] do bramki {display_gate_id or 'T04'} ",
            "button": f"Przekaż [OK] do puli YOLO tablic i wróć do bramki {display_gate_id or 'T04'}",
            "width": 48,
        }
    return {
        "section": " Powrót do grafu ",
        "button": "Wróć do grafu",
        "width": 18,
    }


is_campaign_t07_plate_repair_context = is_campaign_t06_plate_repair_context
is_campaign_t06_char_source_context = is_campaign_t05_char_source_context
is_campaign_t04_char_route_context = is_campaign_t03_char_route_context


def build_z2_workflow_base_context(host: "AnnotationTab", preferred_run_dir) -> Z2WorkflowBaseContext:
    actual_route = host._get_workflow_route()
    route = host._get_z2_thematic_route()
    campaign_context = not host._is_free_mode_session_context()
    if not campaign_context:
        try:
            from ..campaign_manager import CAMPAIGN

            active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
        except Exception:
            active_project = ""
        if active_project:
            campaign_context = True
            try:
                host.app.campaign_free_mode = False
            except Exception:
                pass
    free_mode_context = not campaign_context
    manual_entry_mode = host._get_manual_entry_mode()
    auto_vehicle_choice = host._get_auto_vehicle_choice()
    plate_model_selected = bool(str(host.plate_custom_var.get() or "").strip())
    input_dir_value = str(host.input_dir_var.get() or "").strip()
    input_dir_ready = host._annotation_input_dir_ready(input_dir_value)
    vehicle_assist_enabled = host._manual_vehicle_assist_enabled()
    has_existing_run = preferred_run_dir is not None
    manual_setup = actual_route == "manual" and manual_entry_mode == "new"
    manual_continue = actual_route == "manual" and manual_entry_mode == "continue"
    manual_import = actual_route == "manual" and manual_entry_mode == "import"
    manual_run_already_created = bool(manual_setup and host._has_active_manual_template_run())
    current_step = host._coerce_workflow_step()
    if host._get_workflow_step() != current_step:
        host._set_workflow_step_state(current_step, campaign_context=campaign_context)
    manual_review_screen_active = False
    try:
        manual_review_screen_active = bool(
            free_mode_context
            and host._coerce_free_mode_screen() == "manual_review"
            and host._manual_review_active
        )
    except Exception:
        manual_review_screen_active = False
    manual_review_active = bool(host._manual_review_active and (has_existing_run or manual_review_screen_active))
    manual_review_from_auto = bool(manual_review_active and host._manual_review_from_auto)
    auto_completed = host._is_z2_auto_flow_completed(
        route=route,
        has_existing_run=has_existing_run,
    )
    campaign_reused_manual_count = (
        host._get_campaign_reused_manual_annotation_count()
        if campaign_context
        else 0
    )
    return Z2WorkflowBaseContext(
        actual_route=actual_route,
        route=route,
        campaign_context=campaign_context,
        auto_setup_pending=bool(
            free_mode_context
            and route == "auto"
            and getattr(host, "_auto_route_settings_pending", False)
        ),
        manual_entry_mode=manual_entry_mode,
        auto_vehicle_choice=auto_vehicle_choice,
        plate_model_selected=plate_model_selected,
        input_dir_value=input_dir_value,
        input_dir_ready=input_dir_ready,
        vehicle_assist_enabled=vehicle_assist_enabled,
        has_existing_run=has_existing_run,
        manual_setup=manual_setup,
        manual_continue=manual_continue,
        manual_import=manual_import,
        manual_run_already_created=manual_run_already_created,
        current_step=current_step,
        manual_review_active=manual_review_active,
        manual_review_from_auto=manual_review_from_auto,
        auto_completed=auto_completed,
        campaign_reused_manual_count=campaign_reused_manual_count,
    )


def refresh_workflow_route_cards(host: "AnnotationTab", *, refresh_content: bool = True) -> None:
    palette = getattr(host.app, "palette", {})
    route = host._get_workflow_route()
    hover_route = getattr(host, "_workflow_route_hover_mode", None)
    cached_state = getattr(host, "_workflow_route_card_render_state", None)
    if not isinstance(cached_state, dict):
        cached_state = {}

    actions = {}
    ctx = None
    if refresh_content:
        # Budowanie kontekstu Z2 potrafi sprawdzać runy i XML. Nie robimy tego
        # podczas zwykłego hovera, bo blokuje główną pętlę Tkintera.
        actions = host._get_z2_primary_actions()
        ctx = host._build_z2_action_context()

    panel_alt = palette.get("panel_alt", "#2d2d30")
    hover_bg = palette.get("button_hover", panel_alt)
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    accent = palette.get("accent", "#0e639c")
    success = palette.get("success", "#4ec9b0")
    surface_info = palette.get("surface_info", hover_bg)
    surface_success = palette.get("surface_success", hover_bg)

    config = {
        "auto": {
            "card": getattr(host, "auto_route_card", None),
            "title": getattr(host, "auto_route_card_title", None),
            "desc": getattr(host, "auto_route_card_desc", None),
            "accent": accent,
            "active_bg": surface_info,
        },
        "manual": {
            "card": getattr(host, "manual_route_card", None),
            "title": getattr(host, "manual_route_card_title", None),
            "desc": getattr(host, "manual_route_card_desc", None),
            "accent": success,
            "active_bg": surface_success,
        },
    }

    for key, widgets in config.items():
        card = widgets["card"]
        title = widgets["title"]
        desc = widgets["desc"]
        if card is None:
            continue

        state = cached_state.get(key)
        if not isinstance(state, dict):
            state = {}
        available = bool(state.get("available", True))
        enabled = bool(state.get("enabled", True))

        action = actions.get(key) if refresh_content else None
        if refresh_content:
            available = bool(action.is_available(ctx)) if action is not None else True
            enabled = bool(action.is_enabled(ctx)) if action is not None else True
            cached_state[key] = {
                "available": available,
                "enabled": enabled,
            }
        widgets["available"] = available
        if action is not None:
            try:
                title.configure(text=action.label)
            except Exception:
                pass
            try:
                desc_text = action.get_description(ctx)
                if (
                    key == "auto"
                    and ctx.mode == "campaign"
                    and ctx.has_plate_model
                    and not bool(ctx.campaign_repair_mode)
                ):
                    desc_text = (
                        "Uruchamia autoanotacje na obrazach z listy wynikow anotacji. "
                        "Domyslnie korzysta z aktywnego modelu tablic projektu, ale w kroku wyboru modelu "
                        "mozesz wskazac inny tylko dla tego runu Z2. Potem od razu otwiera wynik do recznej korekty w tym samym Z2."
                    )
                desc.configure(text=desc_text)
            except Exception:
                pass

        is_active = route == key
        is_hover = hover_route == key
        accent_color = widgets.get("accent", accent)
        active_bg = widgets.get("active_bg", surface_info)
        frame_bg = active_bg if is_active else (hover_bg if is_hover else panel_alt)
        frame_border = accent_color if is_active else border
        title_fg = accent_color if is_active else (fg if enabled else muted)
        desc_fg = fg if is_active else (muted if enabled else blend_hex_colors(muted, panel_alt, 0.45))

        try:
            card.configure(
                bg=frame_bg,
                highlightbackground=frame_border,
                highlightcolor=frame_border,
                cursor=("hand2" if enabled else "arrow"),
            )
        except Exception:
            pass

        for widget, color in ((title, title_fg), (desc, desc_fg)):
            if widget is None:
                continue
            try:
                widget.configure(bg=frame_bg, fg=color, cursor=("hand2" if enabled else "arrow"))
            except Exception:
                pass

    if refresh_content:
        host._workflow_route_card_render_state = cached_state

    if refresh_content:
        host._set_widget_packed(
            config["auto"]["card"],
            bool(config["auto"].get("available")),
            fill="x",
            pady=(0, 8),
        )
        host._set_widget_packed(
            config["manual"]["card"],
            bool(config["manual"].get("available")),
            fill="x",
        )


def apply_z2_workflow_left_layout(
    host: "AnnotationTab",
    *,
    campaign_context: bool,
    route: str,
    current_step: str,
    manual_setup: bool,
    manual_continue: bool,
    manual_review_active: bool,
    manual_review_from_auto: bool,
    vehicle_assist_enabled: bool,
    campaign_stage: int,
    campaign_char_repair_mode: bool,
    campaign_plate_step4_repair_mode: bool,
    show_route_choice: bool,
    show_workflow_steps: bool,
    show_auto_followup: bool,
    show_manual_review_followup: bool,
    show_export_followup: bool,
    show_stage_export_cta: bool,
    show_nav_panel: bool,
    show_miniflow_panel: bool,
    show_campaign_context_header: bool,
    show_right_panel: bool,
    compact_single_route_layout: bool,
    compact_export_followup: bool,
    auto_settings_in_scope_modal: bool,
    nav_anchor,
    secondary_actions,
    secondary_ctx,
) -> None:
    show_actions_section = bool(
        (show_workflow_steps or show_campaign_context_header)
        and not (campaign_context and show_manual_review_followup)
    )
    host._set_widget_packed(
        host.actions_section,
        show_actions_section,
        fill=tk.X,
        before=host.workflow_nav_panel,
    )
    host._set_widget_packed(host.actions_section_separator, False)
    host._set_widget_packed(host.followup_section, show_auto_followup, fill=tk.X, pady=(0, 2))
    show_auto_followup_primary_actions = False
    host._set_widget_packed(
        host.followup_actions_row,
        show_auto_followup_primary_actions,
        fill=tk.X,
        pady=(0, 8),
    )
    host._set_widget_packed(
        host.manual_stage_section,
        show_manual_review_followup,
        fill=tk.X,
    )
    host._set_widget_packed(
        host.manual_stage_separator,
        show_stage_export_cta,
        fill=tk.X,
        pady=(18, 20),
    )
    host._set_widget_packed(
        host.manual_stage_export_box,
        show_stage_export_cta,
        fill=tk.X,
        pady=(0, 12),
    )
    try:
        if campaign_context:
            host.manual_stage_export_title_lbl.configure(text="Trening modelu YOLO Pose")
            host.manual_stage_export_help_lbl.configure(
                text=(
                    "Jesli chcesz od razu trenowac model YOLO Pose wykrywajacy obrys tablic "
                    "rejestracyjnych, ustaw split i wyeksportuj dataset. Potem przejdz do Z4, "
                    "gdzie uruchomisz trening."
                )
            )
        elif manual_review_active:
            host.manual_stage_export_title_lbl.configure(text="Trening modelu YOLO Pose")
            host.manual_stage_export_help_lbl.configure(
                text=(
                    "Jesli korekta runu anotacji jest gotowa i chcesz od razu trenowac model "
                    "YOLO Pose wykrywajacy obrys tablic rejestracyjnych, ustaw split i "
                    "wyeksportuj dataset. Potem przejdz do Z4, gdzie uruchomisz trening."
                )
            )
        else:
            host.manual_stage_export_title_lbl.configure(text="Trening modelu YOLO Pose")
            host.manual_stage_export_help_lbl.configure(
                text=(
                    "Jesli chcesz od razu trenowac model YOLO Pose wykrywajacy obrys tablic "
                    "rejestracyjnych, ustaw split i wyeksportuj dataset. Potem przejdz do Z4, "
                    "gdzie uruchomisz trening."
                )
            )
    except Exception:
        pass
    host._set_widget_packed(
        host.export_section,
        show_export_followup,
        fill=tk.X,
    )
    host._set_widget_packed(
        host.export_back_btn,
        show_export_followup and campaign_context,
        fill=tk.X,
        pady=(10, 0),
    )
    host._set_widget_packed(
        getattr(host, "plate_export_split_frame", None),
        show_export_followup and campaign_context,
        fill=tk.X,
        pady=(0, 8),
    )
    host._set_widget_packed(
        host.export_plate_dataset_btn,
        show_export_followup,
        fill=tk.X,
        pady=(14, 0),
    )
    host._set_widget_packed(
        getattr(host, "export_plate_annotations_btn", None),
        False,
    )
    progress_anchor = None
    if show_actions_section:
        progress_anchor = host.actions_section
    elif show_auto_followup:
        progress_anchor = host.followup_section
    elif show_manual_review_followup:
        progress_anchor = host.manual_stage_section
    elif show_export_followup:
        progress_anchor = host.export_section
    elif show_route_choice:
        progress_anchor = host.workflow_entry_section
    try:
        if show_miniflow_panel and str(host.workflow_nav_panel.winfo_manager()) == "pack":
            host.workflow_nav_panel.pack_forget()
    except Exception:
        pass
    host._set_widget_packed(
        host.workflow_nav_panel,
        show_miniflow_panel,
        fill=tk.X,
        pady=(4, 0),
        before=progress_anchor,
        after=(None if progress_anchor is not None else nav_anchor),
    )
    host._refresh_z2_miniflow_progress()
    try:
        if show_nav_panel and str(host.workflow_nav_bottom_panel.winfo_manager()) == "pack":
            host.workflow_nav_bottom_panel.pack_forget()
    except Exception:
        pass
    host._set_widget_packed(
        host.workflow_nav_bottom_panel,
        show_nav_panel,
        fill=tk.X,
        pady=(12, 0),
        after=nav_anchor,
    )

    host._set_widget_packed(host.route_selector_frame, False)
    host._set_widget_packed(
        host.run_title_lbl,
        (show_workflow_steps or show_campaign_context_header or ((not campaign_context) and show_nav_panel))
        and not (campaign_context and show_manual_review_followup),
        anchor=tk.W,
        fill=tk.X,
        pady=((6, 0) if (campaign_context and route == "manual" and manual_setup and not manual_review_active) else ((10, 0) if show_route_choice else (0, 0))),
    )
    hide_manual_setup_header_meta = bool(
        campaign_context
        and route == "manual"
        and manual_setup
        and not manual_review_active
    )
    hide_campaign_repair_header_meta = bool(
        campaign_context
        and campaign_plate_step4_repair_mode
    )
    if hide_campaign_repair_header_meta:
        host._set_widget_packed(host.run_title_lbl, False)
    compact_free_mode_auto_header = bool(
        (not campaign_context)
        and route == "auto"
        and (
            (current_step in {"auto_input", "auto_start"} and show_workflow_steps)
            or show_auto_followup
        )
    )
    compact_free_mode_manual_header = bool(
        (not campaign_context)
        and route == "manual"
        and manual_setup
        and current_step in {"manual_input", "manual_start"}
        and show_workflow_steps
    )
    host._set_widget_packed(
        host.run_intro_lbl,
        (show_workflow_steps or show_campaign_context_header or ((not campaign_context) and show_nav_panel))
        and not (campaign_context and show_manual_review_followup)
        and not hide_campaign_repair_header_meta
        and not hide_manual_setup_header_meta
        and not compact_free_mode_auto_header
        and not compact_free_mode_manual_header
        and bool(str(host.run_intro_var.get() or "").strip()),
        anchor=tk.W,
        fill=tk.X,
        pady=((0, 0) if (campaign_context and campaign_char_repair_mode) else (0, 6)),
    )
    host._set_widget_packed(
        host.route_badge_lbl,
        (show_workflow_steps or show_campaign_context_header)
        and not (campaign_context and show_manual_review_followup)
        and not hide_campaign_repair_header_meta
        and not hide_manual_setup_header_meta
        and not compact_free_mode_auto_header
        and not compact_free_mode_manual_header
        and not (campaign_context and campaign_char_repair_mode),
        anchor=tk.W,
        fill=tk.X,
        pady=(0, 2),
    )
    host._set_widget_packed(
        host.route_summary_lbl,
        (show_workflow_steps or show_campaign_context_header)
        and not (campaign_context and show_manual_review_followup)
        and not hide_campaign_repair_header_meta
        and not hide_manual_setup_header_meta
        and not compact_free_mode_auto_header
        and not compact_free_mode_manual_header,
        anchor=tk.W,
        fill=tk.X,
        pady=((1, 0) if (campaign_context and campaign_char_repair_mode) else (0, 2)),
    )
    host._set_widget_packed(
        host.workflow_action_hint_lbl,
        (show_workflow_steps or show_campaign_context_header)
        and not (campaign_context and show_manual_review_followup)
        and not hide_campaign_repair_header_meta
        and not hide_manual_setup_header_meta
        and not compact_free_mode_auto_header
        and not compact_free_mode_manual_header
        and bool(str(host.workflow_action_hint_var.get() or "").strip()),
        anchor=tk.W,
        fill=tk.X,
        pady=((2, 0) if (campaign_context and campaign_char_repair_mode) else (0, 3)),
    )
    hide_compact_route_meta = bool(
        compact_single_route_layout
        and not (campaign_context and route == "manual")
    )
    if hide_compact_route_meta:
        host._set_widget_packed(host.run_title_lbl, False)
        host._set_widget_packed(host.run_intro_lbl, False)
        host._set_widget_packed(host.route_badge_lbl, False)
        host._set_widget_packed(host.route_summary_lbl, False)
        host._set_widget_packed(host.workflow_action_hint_lbl, False)
    t02_at_review_context = bool(campaign_context and is_campaign_t02_at_review_context(host))
    if campaign_context and (campaign_stage >= 3 or t02_at_review_context):
        try:
            return_copy = get_campaign_return_to_graph_copy(host)
            return_label = str(return_copy.get("button") or "Wróć do grafu")
            return_width = int(return_copy.get("width", 18) or 18)
            host.return_to_campaign_btn.configure(text=return_label, width=return_width)
            host.return_to_campaign_right_btn.configure(text=return_label, width=return_width)
        except Exception:
            pass
    host._set_widget_packed(
        host.return_to_campaign_btn,
        False,
        anchor=tk.W,
        fill=tk.X,
        pady=(0, 8),
    )
    host._set_widget_packed(
        getattr(host, "return_to_campaign_right_btn", None),
        bool(campaign_context and host._should_show_right_panel()),
        fill=tk.X,
        pady=(0, 0),
    )
    show_campaign_auto_persistent_cards = bool(
        campaign_context and show_workflow_steps and route == "auto"
    )
    section_reorder_anchor_candidates = (
        getattr(host, "workflow_input_section", None),
        getattr(host, "workflow_start_section", None),
        getattr(host, "run_output_info_lbl", None),
    )

    def _get_workflow_reorder_anchor(*preferred_widgets):
        for widget in preferred_widgets:
            if widget is not None and host._widget_is_packed(widget):
                return widget
        for widget in section_reorder_anchor_candidates:
            if widget is not None and host._widget_is_packed(widget):
                return widget
        return None

    manual_start_vehicle_options = bool(
        (not campaign_context)
        and show_workflow_steps
        and route == "manual"
        and manual_setup
        and current_step == "manual_start"
        and not bool(getattr(host, "_manual_template_ready_for_review", False))
    )
    campaign_manual_input_vehicle_options = bool(
        campaign_context
        and show_workflow_steps
        and route == "manual"
        and manual_setup
        and current_step == "manual_input"
    )

    host._set_widget_packed(
        host.auto_plate_model_section,
        (not auto_settings_in_scope_modal) and show_workflow_steps and (
            current_step == "auto_plate_model"
            or show_campaign_auto_persistent_cards
        ),
        fill=tk.X,
        pady=(0, 10),
        before=_get_workflow_reorder_anchor(
            getattr(host, "workflow_conf_section", None),
            getattr(host, "auto_vehicle_choice_section", None),
            getattr(host, "workflow_vehicle_model_section", None),
        ),
    )
    host._refresh_auto_vehicle_choice_ui()
    host._set_widget_packed(
        host.auto_vehicle_choice_section,
        (not auto_settings_in_scope_modal) and show_workflow_steps and (
            current_step == "auto_vehicle_choice"
            or show_campaign_auto_persistent_cards
        ),
        fill=tk.X,
        pady=(0, 10),
        before=_get_workflow_reorder_anchor(
            getattr(host, "workflow_vehicle_model_section", None),
        ),
    )
    host._set_widget_packed(
        host.manual_entry_section,
        show_workflow_steps and current_step == "manual_entry",
        fill=tk.X,
        pady=(0, 10),
    )
    host._set_widget_packed(
        host.manual_history_section,
        show_workflow_steps
        and current_step == "manual_history"
        and manual_continue,
        fill=tk.X,
        pady=(0, 10),
    )
    host._set_widget_packed(host.manual_history_open_btn, False)
    host._set_widget_packed(host.manual_history_import_btn, False)
    host._set_widget_packed(host.manual_xml_template_hint_lbl, False)
    host._set_widget_packed(
        host.workflow_manual_vehicle_assist_check,
        campaign_manual_input_vehicle_options,
        anchor=tk.W,
        pady=(6, 2),
    )
    host._set_widget_packed(
        host.workflow_manual_vehicle_assist_hint_lbl,
        campaign_manual_input_vehicle_options,
        fill=tk.X,
        pady=(0, 6),
    )
    host._set_widget_packed(
        getattr(host, "workflow_start_manual_vehicle_assist_check", None),
        manual_start_vehicle_options,
        anchor=tk.W,
        pady=(6, 2),
        before=getattr(host, "start_btn_row", None),
    )
    host._set_widget_packed(
        getattr(host, "workflow_start_manual_vehicle_assist_hint_lbl", None),
        manual_start_vehicle_options,
        fill=tk.X,
        pady=(0, 6),
        before=getattr(host, "start_btn_row", None),
    )
    host._set_widget_packed(
        host.workflow_conf_section,
        show_workflow_steps
        and (
            current_step == "manual_conf"
            or (
                (campaign_manual_input_vehicle_options or manual_start_vehicle_options)
                and vehicle_assist_enabled
            )
        ),
        fill=tk.X,
        pady=(0, 10),
        before=_get_workflow_reorder_anchor(
            getattr(host, "auto_vehicle_choice_section", None),
            getattr(host, "workflow_vehicle_model_section", None),
        ),
    )
    host._set_widget_packed(
        host.workflow_vehicle_model_section,
        show_workflow_steps
        and (
            current_step == "manual_vehicle_model"
            or (
                (campaign_manual_input_vehicle_options or manual_start_vehicle_options)
                and vehicle_assist_enabled
            )
        ),
        fill=tk.X,
        pady=(0, 10),
        before=_get_workflow_reorder_anchor(),
    )
    campaign_auto_input_ui_state = (
        host._get_campaign_manual_reuse_ui_state()
        if campaign_context and host._get_workflow_route() == "auto"
        else {}
    )
    suppress_campaign_fixed_input_section = bool(
        campaign_context
        and host._get_workflow_route() == "auto"
        and str(host.input_dir_var.get() or "").strip()
    )
    show_campaign_auto_input_section = bool(
        campaign_context
        and current_step == "auto_start"
        and host._get_workflow_route() == "auto"
        and bool(campaign_auto_input_ui_state.get("show_option"))
        and not suppress_campaign_fixed_input_section
    )
    host._set_widget_packed(
        host.workflow_input_section,
        (
            bool(show_workflow_steps and current_step in {"auto_input", "manual_input"})
            and not suppress_campaign_fixed_input_section
        )
        or show_campaign_auto_input_section
        or (
            show_campaign_auto_persistent_cards
            and not suppress_campaign_fixed_input_section
        ),
        fill=tk.X,
        pady=(0, 2),
        before=_get_workflow_reorder_anchor(),
    )
    host._refresh_campaign_workflow_input_lock_state(
        campaign_context=campaign_context,
        current_step=current_step,
    )
    host._refresh_campaign_manual_reuse_option_ui()
    host._set_widget_packed(
        host.run_output_info_lbl,
        False,
        fill=tk.X,
        pady=(0, 8),
    )
    host._set_widget_packed(
        host.open_run_dir_btn,
        show_auto_followup and not campaign_context,
        anchor=tk.W,
        pady=(0, 8),
    )
    post_hint_label = getattr(host, "post_annotation_hint_lbl", None)
    post_hint_visible = False
    if post_hint_label is not None and not (show_auto_followup and not campaign_context):
        try:
            post_hint_visible = bool(str(post_hint_label.cget("text") or "").strip())
        except Exception:
            post_hint_visible = False
    host._set_widget_packed(
        post_hint_label,
        post_hint_visible,
        fill=tk.X,
        pady=(6, 0),
    )
    try:
        if show_auto_followup and not campaign_context:
            cut_ready = False
            try:
                run_dir = host._get_preferred_annotation_run_dir(require_xml=True)
                cut_ready = bool(
                    run_dir is not None
                    and host._get_run_plate_strict_approved_state(run_dir).get("ok")
                )
            except Exception:
                cut_ready = False
            host.open_run_dir_btn.configure(
                text="Wyodrębnij tablice",
                command=host._open_step3_from_z2_annotation_source,
                state=(tk.NORMAL if cut_ready and not host.is_processing else tk.DISABLED),
            )
        else:
            host.open_run_dir_btn.configure(
                text="Otwórz folder runu",
                command=host._open_current_run_dir,
                state=(tk.NORMAL if not host.is_processing else tk.DISABLED),
            )
    except Exception:
        pass
    try:
        if show_auto_followup and not campaign_context:
            if str(host.followup_title_lbl.winfo_manager()) == "pack":
                host.followup_title_lbl.pack_configure(anchor=tk.W, fill=tk.X, pady=(0, 4))
            if str(host.followup_intro_lbl.winfo_manager()) == "pack":
                host.followup_intro_lbl.pack_configure(anchor=tk.W, fill=tk.X, pady=(0, 4))
        else:
            if str(host.followup_title_lbl.winfo_manager()) == "pack":
                host.followup_title_lbl.pack_configure(anchor=tk.W, fill=tk.X, pady=(0, 6))
            if str(host.followup_intro_lbl.winfo_manager()) == "pack":
                host.followup_intro_lbl.pack_configure(anchor=tk.W, fill=tk.X, pady=(0, 8))
    except Exception:
        pass

    try:
        if (
            (not campaign_context)
            and route == "manual"
            and manual_setup
            and current_step == "manual_input"
        ):
            ordered_widgets = [
                (getattr(host, "workflow_input_section", None), {"fill": tk.X, "pady": (0, 2)}),
            ]
            visible_widgets = []
            for widget, pack_kwargs in ordered_widgets:
                if widget is None:
                    continue
                try:
                    if str(widget.winfo_manager()) == "pack":
                        visible_widgets.append((widget, pack_kwargs))
                except Exception:
                    continue
            for widget, _pack_kwargs in visible_widgets:
                try:
                    widget.pack_forget()
                except Exception:
                    continue
            for widget, pack_kwargs in visible_widgets:
                try:
                    widget.pack(**pack_kwargs)
                except Exception:
                    continue
            if str(host.workflow_input_title_lbl.winfo_manager()) == "pack":
                host.workflow_input_title_lbl.pack_configure(anchor=tk.W, fill=tk.X, pady=(0, 4))
            if str(host.workflow_input_hint_lbl.winfo_manager()) == "pack":
                host.workflow_input_hint_lbl.pack_configure(anchor=tk.W, fill=tk.X, pady=(0, 4))
            input_group_widgets = [
                (getattr(host, "workflow_input_row", None), {"fill": tk.X, "pady": (2, 6)}),
            ]
            visible_input_group_widgets = []
            for widget, pack_kwargs in input_group_widgets:
                if widget is None:
                    continue
                try:
                    if str(widget.winfo_manager()) == "pack":
                        visible_input_group_widgets.append((widget, pack_kwargs))
                except Exception:
                    continue
            for widget, _pack_kwargs in visible_input_group_widgets:
                try:
                    widget.pack_forget()
                except Exception:
                    continue
            for widget, pack_kwargs in visible_input_group_widgets:
                try:
                    widget.pack(**pack_kwargs)
                except Exception:
                    continue
        elif str(host.workflow_input_title_lbl.winfo_manager()) == "pack":
            host.workflow_input_title_lbl.pack_configure(anchor=tk.W, fill=tk.X, pady=(0, 6))
    except Exception:
        pass

    try:
        inline_manual_vehicle_conf = bool(
            (not campaign_context)
            and route == "manual"
            and manual_setup
            and current_step == "manual_start"
            and vehicle_assist_enabled
        )
        conf_title_lbl = getattr(host, "workflow_conf_title_lbl", None)
        conf_hint_lbl = getattr(host, "workflow_conf_hint_lbl", None)
        if conf_title_lbl is not None and conf_hint_lbl is not None:
            if inline_manual_vehicle_conf:
                if str(conf_title_lbl.winfo_manager()) == "pack":
                    conf_title_lbl.pack_forget()
                if str(conf_hint_lbl.winfo_manager()) == "pack":
                    conf_hint_lbl.pack_configure(anchor=tk.W, fill=tk.X, pady=(0, 6))
            else:
                if str(conf_title_lbl.winfo_manager()) != "pack":
                    conf_title_lbl.pack(anchor=tk.W, fill=tk.X, before=conf_hint_lbl)
                else:
                    conf_title_lbl.pack_configure(anchor=tk.W, fill=tk.X)
                if str(conf_hint_lbl.winfo_manager()) == "pack":
                    conf_hint_lbl.pack_configure(anchor=tk.W, fill=tk.X, pady=(0, 6))
    except Exception:
        pass

    try:
        header_widgets = [
            (getattr(host, "run_title_lbl", None), {"anchor": tk.W, "fill": tk.X, "pady": (0, 6)}),
            (getattr(host, "run_intro_lbl", None), {"anchor": tk.W, "fill": tk.X, "pady": (0, 6)}),
            (getattr(host, "route_badge_lbl", None), {"anchor": tk.W, "fill": tk.X, "pady": (0, 2)}),
            (getattr(host, "route_summary_lbl", None), {"anchor": tk.W, "fill": tk.X, "pady": (0, 2)}),
            (getattr(host, "workflow_action_hint_lbl", None), {"anchor": tk.W, "fill": tk.X, "pady": (0, 3)}),
        ]
        visible_headers = []
        for widget, pack_kwargs in header_widgets:
            if widget is None:
                continue
            try:
                if str(widget.winfo_manager()) == "pack":
                    visible_headers.append((widget, pack_kwargs))
            except Exception:
                continue

        if visible_headers:
            content_anchor = None
            for candidate in (
                getattr(host, "return_to_campaign_btn", None),
                getattr(host, "route_selector_frame", None),
                getattr(host, "manual_entry_section", None),
                getattr(host, "manual_history_section", None),
                getattr(host, "auto_plate_model_section", None),
                getattr(host, "workflow_conf_section", None),
                getattr(host, "auto_vehicle_choice_section", None),
                getattr(host, "workflow_vehicle_model_section", None),
                getattr(host, "workflow_input_section", None),
                getattr(host, "workflow_start_section", None),
                getattr(host, "run_output_info_lbl", None),
            ):
                try:
                    if candidate is not None and str(candidate.winfo_manager()) == "pack":
                        content_anchor = candidate
                        break
                except Exception:
                    continue

            for widget, _pack_kwargs in visible_headers:
                try:
                    widget.pack_forget()
                except Exception:
                    continue

            for widget, pack_kwargs in visible_headers:
                safe_kwargs = dict(pack_kwargs)
                if content_anchor is not None:
                    safe_kwargs["before"] = content_anchor
                try:
                    widget.pack(**safe_kwargs)
                except Exception:
                    safe_kwargs.pop("before", None)
                    try:
                        widget.pack(**safe_kwargs)
                    except Exception:
                        pass
    except Exception:
        pass

    show_free_mode_right_panel = bool((not campaign_context) and show_right_panel)
    host._set_widget_packed(host.right_scroll_host, False)
    host._set_widget_packed(
        host.approve_btn_row,
        bool(show_right_panel and (campaign_context or show_free_mode_right_panel)),
        fill=tk.X,
        pady=(8, 0),
    )
    if show_free_mode_right_panel:
        try:
            host._refresh_free_mode_manual_right_panel()
        except Exception:
            pass
    host._set_widget_packed(host.mode_title_lbl, False)
    host._set_widget_packed(host.mode_combo, False)
    host._set_widget_packed(host.mode_hint_lbl, False)
    host._set_widget_packed(host.pla_frame, False)
    host._set_widget_packed(host.veh_frame, False)
    host._set_widget_packed(host.param_frame, False)
    host._refresh_manual_review_followup_ui(
        from_auto=manual_review_from_auto,
        active_run=manual_review_active,
    )
    host._refresh_preview_workspace_visibility(manual_review_active=manual_review_active)
    host._refresh_export_followup_ui(compact_active_run=compact_export_followup)
    try:
        review_action = secondary_actions.get("review")
        if review_action is not None:
            host.enter_manual_review_btn.configure(
                text=review_action.label,
                state=(tk.NORMAL if review_action.is_enabled(secondary_ctx) else tk.DISABLED),
            )
    except Exception:
        pass
    try:
        export_action = secondary_actions.get("export")
        if export_action is not None:
            host.jump_to_export_btn.configure(
                text=export_action.label,
                state=(tk.NORMAL if export_action.is_enabled(secondary_ctx) else tk.DISABLED),
            )
    except Exception:
        pass
    try:
        if campaign_context:
            if str(host.jump_to_export_btn.winfo_manager()) == "grid":
                host.jump_to_export_btn.grid_remove()
            if str(host.enter_manual_review_btn.winfo_manager()) == "grid":
                host.enter_manual_review_btn.grid_remove()
        else:
            if str(host.enter_manual_review_btn.winfo_manager()) != "grid":
                host.enter_manual_review_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
            if str(host.jump_to_export_btn.winfo_manager()) != "grid":
                host.jump_to_export_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))
            host.enter_manual_review_btn.grid_configure(column=0, columnspan=1, sticky="ew", padx=(0, 6))
    except Exception:
        pass
    try:
        host.manual_stage_export_btn.configure(
            state=(tk.NORMAL if show_stage_export_cta and not host.is_processing else tk.DISABLED)
        )
    except Exception:
        pass
    try:
        host.export_back_btn.configure(
            state=(tk.NORMAL if show_export_followup and not host.is_processing else tk.DISABLED)
        )
    except Exception:
        pass
    host._refresh_manual_stage_export_box_style()


def apply_z2_workflow_cta_ui(
    host: "AnnotationTab",
    *,
    campaign_context: bool,
    route: str,
    current_step: str,
    show_workflow_steps: bool,
    show_auto_followup: bool,
    show_export_followup: bool,
    manual_review_active: bool,
    auto_completed: bool,
    cta_state: Z2CtaState,
) -> None:
    host._set_widget_packed(
        host.progress_info_row,
        False,
        fill=tk.X,
        pady=(10, 4),
        after=host.start_btn_row,
    )
    host._set_widget_packed(
        host.progress,
        False,
        fill=tk.X,
        pady=(0, 2),
        after=host.progress_info_row,
    )

    show_start_controls = bool(cta_state.show_start_controls)
    compact_campaign_auto_start = bool(
        campaign_context
        and show_workflow_steps
        and str(route or "").strip().lower() == "auto"
    )
    host._set_widget_packed(
        host.workflow_start_section,
        show_start_controls,
        fill=tk.X,
        pady=(
            (4, 10)
            if compact_campaign_auto_start
            else ((0, 2) if ((not campaign_context) and route == "auto" and current_step == "auto_start") else (0, 10))
        ),
    )
    try:
        workflow_start_title_text = str(host.workflow_start_title_lbl.cget("text") or "").strip()
    except Exception:
        workflow_start_title_text = ""
    host._set_widget_packed(
        host.workflow_start_title_lbl,
        show_start_controls
        and bool(workflow_start_title_text),
        anchor=tk.W,
        fill=tk.X,
    )
    host._set_widget_packed(
        host.workflow_start_intro_lbl,
        show_start_controls
        and bool(str(host.workflow_start_intro_var.get() or "").strip()),
        anchor=tk.W,
        fill=tk.X,
        pady=(4, 0),
    )
    host._set_widget_packed(
        host.start_btn_row,
        show_start_controls,
        fill=tk.X,
        pady=(6, 0),
    )
    host._set_widget_packed(
        host.workflow_start_action_hint_lbl,
        show_start_controls
        and bool(str(host.workflow_action_hint_var.get() or "").strip()),
        anchor=tk.W,
        fill=tk.X,
        pady=(6, 0),
        after=host.start_btn_row,
    )
    try:
        host.start_btn_frame.grid_configure(row=0, column=0, columnspan=2, sticky="ew", padx=(0, 0))
    except Exception:
        pass
    try:
        host.stop_btn.grid_remove()
    except Exception:
        pass
    try:
        host._normalize_workflow_start_section_order()
    except Exception:
        pass

    start_enabled = bool(cta_state.start_enabled)
    start_command = cta_state.start_command or host._start_annotation
    start_text = str(cta_state.start_text or "Wybierz tor")
    try:
        host.start_btn.configure(
            text=start_text,
            state=(tk.NORMAL if start_enabled else tk.DISABLED),
            command=start_command,
        )
    except Exception:
        pass

    suppress_campaign_duplicate_start_cta = bool(cta_state.suppress_duplicate_start_cta)
    if suppress_campaign_duplicate_start_cta:
        host._set_widget_packed(host.start_btn_row, False)

    back_enabled = bool(cta_state.back_enabled)
    next_enabled = bool(cta_state.next_enabled)
    back_text = str(getattr(cta_state, "back_text", "Wstecz") or "Wstecz")
    next_text = str(cta_state.next_text or "Dalej")
    show_next_button = bool(str(cta_state.next_text or "").strip())
    try:
        back_label = back_text.strip()
        back_width = max(10, min(22, len(back_label) + 2))
        if back_label.casefold() == "wstecz":
            back_width = 10
        host.workflow_back_btn.configure(
            state=(tk.NORMAL if back_enabled else tk.DISABLED),
            text=back_text,
            width=back_width,
        )
        host.workflow_next_btn.configure(
            state=(tk.NORMAL if next_enabled else tk.DISABLED),
            text=next_text,
            width=max(18, min(30, len(next_text) + 2)),
        )
        if show_next_button:
            host.workflow_next_btn.grid()
        else:
            host.workflow_next_btn.grid_remove()
    except Exception:
        pass
