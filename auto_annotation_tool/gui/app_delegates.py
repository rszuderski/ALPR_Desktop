"""Delegate bindings for ``AutoAnnotationApp``.

The main app class owns the public method names used by tabs and callbacks.
Implementation details live in focused app_* modules; this binder keeps those
names available without keeping one-line forwarding methods in app.py.
"""

from __future__ import annotations

from . import app_global_terminal
from . import app_menu_dropdown
from . import app_project_history
from . import app_shutdown
from . import app_startup
from . import app_tooltips
from . import app_window_recovery
from .app_style_setup import setup_style


_INSTANCE_METHODS = (
    ("_release_window_grabs_for_recovery", app_window_recovery.release_window_grabs_for_recovery),
    ("_release_preview_fullscreens_for_recovery", app_window_recovery.release_preview_fullscreens_for_recovery),
    ("_register_recoverable_toplevel", app_window_recovery.register_recoverable_toplevel),
    ("_restore_app_window_stack", app_window_recovery.restore_app_window_stack),
    ("_on_root_unmap", app_window_recovery.on_root_unmap),
    ("_on_root_map", app_window_recovery.on_root_map),
    ("_on_root_visibility", app_window_recovery.on_root_visibility),
    ("_on_root_focus_in", app_window_recovery.on_root_focus_in),
    ("_schedule_root_recovery", app_window_recovery.schedule_root_recovery),
    ("_recover_root_after_map", app_window_recovery.recover_root_after_map),
    ("_bind_simple_tooltip", app_tooltips.bind_simple_tooltip),
    ("_schedule_simple_tooltip", app_tooltips.schedule_simple_tooltip),
    ("_show_simple_tooltip", app_tooltips.show_simple_tooltip),
    ("_hide_simple_tooltip", app_tooltips.hide_simple_tooltip),
    ("_enable_fatal_crash_logging", app_startup.enable_fatal_crash_logging),
    ("_raise_startup_overlay", app_startup.raise_startup_overlay),
    ("_flush_startup_overlay", app_startup.flush_startup_overlay),
    ("_prepare_main_window_for_startup", app_startup.prepare_main_window_for_startup),
    ("_reveal_main_window_after_startup", app_startup.reveal_main_window_after_startup),
    ("_show_startup_overlay", app_startup.show_startup_overlay),
    ("_set_startup_progress", app_startup.set_startup_progress),
    ("_hide_startup_overlay", app_startup.hide_startup_overlay),
    ("_wait_for_startup_marker", app_startup.wait_for_startup_marker),
    ("_drain_startup_pending_events", app_startup.drain_startup_pending_events),
    ("_startup_tabs_ready", app_startup.startup_tabs_ready),
    ("_schedule_startup_finalize", app_startup.schedule_startup_finalize),
    ("_finalize_startup_after_tabs_ready", app_startup.finalize_startup_after_tabs_ready),
    ("_setup_style", setup_style),
    ("_apply_global_terminal_visual_state", app_global_terminal._apply_global_terminal_visual_state),
    ("_normalize_global_terminal_entry", app_global_terminal._normalize_global_terminal_entry),
    ("_refresh_global_terminal_plain_lines", app_global_terminal._refresh_global_terminal_plain_lines),
    ("_configure_global_terminal_tags", app_global_terminal._configure_global_terminal_tags),
    ("_insert_global_terminal_entry", app_global_terminal._insert_global_terminal_entry),
    ("_set_global_terminal_hold_position", app_global_terminal._set_global_terminal_hold_position),
    ("_sync_global_terminal_hold_position_from_view", app_global_terminal._sync_global_terminal_hold_position_from_view),
    ("_schedule_global_terminal_hold_position_sync", app_global_terminal._schedule_global_terminal_hold_position_sync),
    ("_on_global_terminal_pointer_event", app_global_terminal._on_global_terminal_pointer_event),
    ("_on_global_terminal_scroll_event", app_global_terminal._on_global_terminal_scroll_event),
    ("_populate_global_terminal_widget", app_global_terminal._populate_global_terminal_widget),
    ("_position_global_terminal_window", app_global_terminal._position_global_terminal_window),
    ("_ensure_global_terminal_window", app_global_terminal._ensure_global_terminal_window),
    ("is_global_terminal_visible", app_global_terminal.is_global_terminal_visible),
    ("show_global_terminal", app_global_terminal.show_global_terminal),
    ("hide_global_terminal", app_global_terminal.hide_global_terminal),
    ("toggle_global_terminal", app_global_terminal.toggle_global_terminal),
    ("clear_global_terminal", app_global_terminal.clear_global_terminal),
    ("append_global_terminal_entries", app_global_terminal.append_global_terminal_entries),
    ("append_global_terminal", app_global_terminal.append_global_terminal),
    ("_close_menu_dropdown", app_menu_dropdown._close_menu_dropdown),
    ("_toggle_menu_dropdown", app_menu_dropdown._toggle_menu_dropdown),
    ("_open_menu_dropdown", app_menu_dropdown._open_menu_dropdown),
    ("show_project_history_dialog", app_project_history.show_project_history_dialog),
    ("_cancel_after_handle_safely", app_shutdown._cancel_after_handle_safely),
    ("_cancel_dynamic_after_callbacks", app_shutdown._cancel_dynamic_after_callbacks),
    ("_release_runtime_references", app_shutdown._release_runtime_references),
    ("_flush_loaded_tab_runtime_state", app_shutdown._flush_loaded_tab_runtime_state),
    ("_on_closing", app_shutdown._on_closing),
)


_STATIC_METHODS = (
    ("_consume_startup_overlay_event", app_startup.consume_startup_overlay_event),
    ("_get_expected_startup_tab_keys", app_startup.get_expected_startup_tab_keys),
    ("_capture_terminal_widget_view_state", app_global_terminal._capture_terminal_widget_view_state),
    ("_restore_terminal_widget_view_state", app_global_terminal._restore_terminal_widget_view_state),
)


def bind_app_delegates(app_cls):
    """Attach extracted app runtime methods to ``AutoAnnotationApp``."""

    for method_name, target in _INSTANCE_METHODS:
        setattr(app_cls, method_name, target)

    for method_name, target in _STATIC_METHODS:
        setattr(app_cls, method_name, staticmethod(target))

    return app_cls
