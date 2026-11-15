from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import tkinter as tk

import pytest

from auto_annotation_tool.gui.app import AutoAnnotationApp
from auto_annotation_tool.gui.app_theme_definitions import THEME_DEFINITIONS
from auto_annotation_tool.gui.z3_extraction_tab_ui import (
    _ensure_campaign_detect_splash_widgets,
    hide_campaign_detect_splash,
    show_campaign_detect_splash,
)
from auto_annotation_tool.gui.z3_loading_progress import LoadingProgressBar


@pytest.fixture
def root():
    window = tk.Tk()
    window.geometry("900x580")
    yield window
    window.destroy()


def host(root, theme="light_visual_cs"):
    app = AutoAnnotationApp.__new__(AutoAnnotationApp)
    app.root = root
    app.palette = deepcopy(THEME_DEFINITIONS[theme]["palette"])
    app.style = tk.ttk.Style(root)
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)
    content = tk.Frame(frame)
    content.pack(fill=tk.BOTH, expand=True)
    owner = SimpleNamespace(app=app, frame=frame, detect_content_frame=content,
                            _refresh_campaign_step3_navigation_visibility=Mock(),
                            _return_to_t05_work_after_step3_pz1=Mock())
    owner._show_campaign_detect_splash = lambda **options: show_campaign_detect_splash(owner, **options)
    return owner


def show(owner, progress=0., **options):
    show_campaign_detect_splash(owner, title="Wyodrębniam tablice dla Z3",
                                body="Wycinanie: 45/120 obrazów · 67 tablic", progress=progress, **options)


@pytest.mark.parametrize("theme", ["light_visual_cs", "dark_visual_cs"])
def test_card_and_zero_progress_remain_visible_after_generic_theme_refresh(root, theme):
    owner = host(root, theme)
    owner._campaign_detect_splash_force_root_surface = True
    show(owner)
    root.update()
    overlay = owner.campaign_detect_splash_overlay
    card = owner.campaign_detect_splash_card
    progress = owner.campaign_detect_splash_progress
    for _ in range(3):
        owner.app.style_panel_surface(owner.frame)
    assert card.cget("bg") != overlay.cget("bg")
    assert card.cget("bg") == owner.campaign_detect_splash_body_lbl.cget("bg")
    assert card.cget("bg") == owner.campaign_detect_splash_title_lbl.cget("bg")
    assert int(card.cget("highlightthickness")) == 1
    assert owner.campaign_detect_splash_progress_pct_lbl.cget("text") == "0%"
    assert progress.winfo_ismapped()
    assert progress.itemcget(progress._trough, "fill") != progress.cget("bg")
    assert progress.coords(progress._trough)[2] > 200
    assert progress.itemcget(progress._fill, "state") == tk.HIDDEN


def test_later_panel_refresh_preserves_animation_and_styles_ordinary_panels(root):
    owner = host(root)
    ordinary_panel = tk.Frame(owner.frame, bg="#123456")
    show(owner, progress=None)
    bar = owner.campaign_detect_splash_progress
    job = bar._job
    text = owner.campaign_detect_splash_body_lbl.cget("text")
    for surface in (owner.frame, owner.campaign_detect_splash_card,
                    owner.campaign_detect_splash_content, owner.campaign_detect_splash_progress_header):
        owner.app.style_panel_surface(surface)
        assert owner.campaign_detect_splash_card.cget("bg") == owner.app.palette["field"]
        assert owner.campaign_detect_splash_content.cget("bg") == owner.app.palette["field"]
        assert owner.campaign_detect_splash_body_lbl.cget("bg") == owner.app.palette["field"]
        assert bar._job == job
        assert bar.cget("mode") == "indeterminate"
        assert owner.campaign_detect_splash_body_lbl.cget("text") == text
    assert ordinary_panel.cget("bg") == owner.app.palette["panel"]


@pytest.mark.parametrize("theme", ["light_visual_cs", "dark_visual_cs"])
def test_app_style_setup_and_later_refresh_keep_complete_loading_card(root, theme):
    owner = host(root, theme)
    owner.app.themes = deepcopy(THEME_DEFINITIONS)
    owner.app.current_theme_key = theme
    owner.app._setup_style(theme)
    assert owner.app.style.theme_use() == "clam"
    owner._campaign_detect_splash_force_root_surface = True
    show(owner, progress=None)
    owner.app.style_panel_surface(owner.frame)
    owner.app._apply_theme_to_widget_tree(owner.frame)
    owner.app.style_panel_surface(owner.frame)
    root.update_idletasks()
    assert owner.campaign_detect_splash_content.cget("bg") == owner.app.palette["field"]
    assert owner.campaign_detect_splash_title_lbl.cget("bg") == owner.app.palette["field"]
    assert owner.campaign_detect_splash_body_lbl.cget("bg") == owner.app.palette["field"]
    assert owner.campaign_detect_splash_progress.cget("bg") == owner.app.palette["field"]
    assert int(owner.campaign_detect_splash_card.cget("highlightthickness")) == 1
    assert owner.campaign_detect_splash_progress._job is not None


@pytest.mark.parametrize("width,height", [(1400, 900), (900, 580), (440, 420)])
def test_card_centers_and_wraps_inside_loading_surface(root, width, height):
    root.geometry(f"{width}x{height}")
    owner = host(root)
    show(owner, progress=37.5)
    root.update()
    overlay = owner.campaign_detect_splash_overlay
    card = owner.campaign_detect_splash_card
    assert card.winfo_width() <= min(640, width - 48)
    assert abs(card.winfo_x() + card.winfo_width() / 2 - overlay.winfo_width() / 2) <= 1
    assert abs(card.winfo_y() + card.winfo_height() / 2 - overlay.winfo_height() / 2) <= 1
    assert card.winfo_y() >= 0
    assert card.winfo_y() + card.winfo_height() <= overlay.winfo_height()
    assert int(owner.campaign_detect_splash_body_lbl.cget("wraplength")) <= card.winfo_width() - 50
    assert owner.campaign_detect_splash_progress_pct_lbl.cget("text") == "37.5%"
    bar = owner.campaign_detect_splash_progress
    trough = bar.coords(bar._trough)
    fill = bar.coords(bar._fill)
    assert (fill[2] - fill[0]) / (trough[2] - trough[0]) == pytest.approx(.375)


def test_indeterminate_start_switches_to_real_progress_and_cleans_up_timer(root):
    owner = host(root)
    show(owner, progress=None)
    root.update()
    bar = owner.campaign_detect_splash_progress
    first_job = bar._job
    assert first_job
    assert owner.campaign_detect_splash_progress_pct_lbl.cget("text") == "Przygotowanie…"
    assert owner._campaign_detect_splash_progress_value is None
    show(owner, progress=50)
    assert first_job not in root.tk.call("after", "info")
    assert bar._job is None
    assert bar.cget("mode") == "determinate"
    assert bar.cget("value") == 50
    assert owner.campaign_detect_splash_progress_pct_lbl.cget("text") == "50%"
    show(owner, progress=None)
    second_job = bar._job
    hide_campaign_detect_splash(owner)
    assert second_job not in root.tk.call("after", "info")
    assert not owner._campaign_detect_splash_visible
    assert not owner.campaign_detect_splash_overlay.winfo_manager()


def test_destroying_running_bar_cancels_animation(root):
    bar = LoadingProgressBar(root)
    bar.configure(mode="indeterminate")
    bar.start()
    job = bar._job
    bar.destroy()
    assert job not in root.tk.call("after", "info")


def test_progress_redraw_does_not_reenter_tk_event_loop(root):
    bar = LoadingProgressBar(root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bar, "update_idletasks", Mock(side_effect=AssertionError("reentrant draw")))
        bar.configure(value=50)
        bar._redraw()
        bar.update_idletasks.assert_not_called()


def test_theme_switch_preserves_live_progress_and_text(root):
    owner = host(root)
    show(owner, progress=73)
    owner.app.palette = deepcopy(THEME_DEFINITIONS["dark_visual_cs"]["palette"])
    owner.app.style_panel_surface(owner.frame)
    assert owner.campaign_detect_splash_progress.cget("value") == 73
    assert owner.campaign_detect_splash_progress_pct_lbl.cget("text") == "73%"
    assert owner.campaign_detect_splash_body_lbl.cget("text") == "Wycinanie: 45/120 obrazów · 67 tablic"
    assert owner.campaign_detect_splash_title_lbl.cget("fg") == owner.app.palette["fg"]


def test_error_stops_progress_and_retains_existing_return_action(root):
    owner = host(root)
    show(owner, progress=None)
    bar = owner.campaign_detect_splash_progress
    job = bar._job
    show_campaign_detect_splash(owner, title="Nie udało się przygotować tablic dla Z3",
                                body="Popraw anotacje i spróbuj ponownie.", tone="error",
                                show_progress=False, show_return=True)
    root.update()
    assert job not in root.tk.call("after", "info")
    assert not bar.winfo_manager()
    assert not owner.campaign_detect_splash_progress_header.winfo_manager()
    assert owner.campaign_detect_splash_return_btn.winfo_ismapped()
    owner.campaign_detect_splash_return_btn.invoke()
    owner._return_to_t05_work_after_step3_pz1.assert_called_once_with()


def test_building_detection_ui_retains_pinned_root_splash_and_its_widgets(root):
    owner = host(root)
    owner._campaign_detect_splash_force_root_surface = True
    show(owner, progress=None)
    original = owner.campaign_detect_splash_overlay
    bar = owner.campaign_detect_splash_progress
    owner._campaign_detect_splash_force_root_surface = False
    _ensure_campaign_detect_splash_widgets(owner, return_text="Wróć do E3", return_command=Mock())
    assert owner.campaign_detect_splash_overlay is original
    assert owner.campaign_detect_splash_progress is bar
    assert owner._campaign_detect_splash_surface is owner.frame
    assert owner._campaign_detect_splash_visible
    assert bar._job


def test_hidden_splash_can_move_to_detection_surface_without_leaving_old_timer(root):
    owner = host(root)
    owner._campaign_detect_splash_force_root_surface = True
    show(owner, progress=None)
    old = owner.campaign_detect_splash_overlay
    old_bar = owner.campaign_detect_splash_progress
    hide_campaign_detect_splash(owner)
    action = Mock()
    _ensure_campaign_detect_splash_widgets(owner, return_text="Wróć do E3", return_command=action)
    assert not old.winfo_exists()
    assert old_bar._job is None
    assert owner._campaign_detect_splash_surface is owner.detect_content_frame
    owner.campaign_detect_splash_return_btn.invoke()
    action.assert_called_once_with()
