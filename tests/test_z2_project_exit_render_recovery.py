from types import SimpleNamespace
from unittest.mock import Mock, patch
import tkinter as tk

import pytest

from auto_annotation_tool.gui import z2_panel_workflow as panel
from auto_annotation_tool.gui.tab_annotation import AnnotationTab
from auto_annotation_tool.campaign_manager import CAMPAIGN


def test_z2_created_inside_project_has_route_cards_after_project_exit():
    root = tk.Tk()
    root.withdraw()
    app = SimpleNamespace(campaign_free_mode=False)
    try:
        with patch.object(CAMPAIGN, "get_active_project_name", return_value="test-project"):
            tab = AnnotationTab(root, app)
        app.campaign_free_mode = True
        tab._apply_free_mode_session_snapshot = lambda **kwargs: tab.free_mode_screen_var.set("route_choice")
        tab._pre_campaign_free_mode_snapshot = None
        with patch.object(CAMPAIGN, "get_active_project_name", return_value=""):
            tab.clear_campaign_context(restore_free_mode_preview=False)
            root.update_idletasks()
        assert tab.workflow_entry_shell.winfo_manager() == "pack"
        assert tab.auto_route_card is not None
        assert tab.manual_route_card is not None
        assert tab.auto_route_card.winfo_manager() == "pack"
        assert tab.manual_route_card.winfo_manager() == "pack"
        tab.auto_route_card.event_generate("<Button-1>")
        assert tab.workflow_route_var.get() == "auto"
    finally:
        root.destroy()


def test_project_exit_restores_real_entry_shell_after_interrupted_transition():
    root = tk.Tk()
    root.withdraw()
    try:
        tab = AnnotationTab(root, SimpleNamespace(campaign_free_mode=True))
        # Keep this probe independent of the user's saved session.
        tab._apply_free_mode_session_snapshot = lambda **kwargs: tab.free_mode_screen_var.set("route_choice")
        tab._pre_campaign_free_mode_snapshot = None
        tab._begin_campaign_step2_transition()
        assert tab.workflow_entry_shell.winfo_manager() == ""

        tab.clear_campaign_context(restore_free_mode_preview=False)
        root.update_idletasks()

        assert tab._campaign_step2_transition_in_progress is False
        assert tab._campaign_step2_transition_refresh_pending is False
        assert tab.workflow_entry_shell.winfo_manager() == "pack"
        assert tab.workflow_entry_section.winfo_manager() == "pack"
        assert tab._workflow_ui_refresh_in_progress is False
    finally:
        root.destroy()


def test_render_exception_does_not_block_next_refresh():
    owner = SimpleNamespace(_is_free_mode_session_context=lambda: True)
    with patch.object(panel, "_render_free_mode_workflow_ui", side_effect=RuntimeError("render failed")):
        with pytest.raises(RuntimeError, match="render failed"):
            panel._refresh_free_mode_workflow_ui(owner)
    assert owner._workflow_ui_refresh_in_progress is False

    with patch.object(panel, "_render_free_mode_workflow_ui", Mock()) as render:
        panel._refresh_free_mode_workflow_ui(owner)
        render.assert_called_once_with(owner)
