import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette, THEME_DEFINITIONS
from auto_annotation_tool.gui.campaign_sidebar import ProjectSidebarToggle, SlidingProjectSidebar, STYLE
from auto_annotation_tool.gui import campaign_shell_ui as shell


class CampaignSidebarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
        except tk.TclError as exc:
            raise unittest.SkipTest(f"Tk unavailable: {exc}")
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.clock = [0.0]
        self.container = ttk.Frame(self.root, width=1200, height=760)
        self.panel = ttk.Frame(self.container, width=340)
        self.panel.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        self.panel.grid_propagate(False)
        self.listbox = tk.Listbox(self.panel)
        self.listbox.pack(fill="both", expand=True)
        self.listbox.insert("end", *[f"Project {i}" for i in range(30)])
        self.listbox.selection_set(12)
        self.command = Mock()
        self.button = ProjectSidebarToggle(self.container, get_theme_palette(), self.command)
        self.motion_events = []
        self.settled = Mock()
        self.motion = SlidingProjectSidebar(self.container, self.panel, self.button,
            on_motion=self.motion_events.append, on_settled=self.settled, clock=lambda: self.clock[0])

    def tearDown(self):
        self.container.destroy()
        self.root.update_idletasks()

    def advance(self, seconds):
        self.motion._cancel_tick()
        self.clock[0] += seconds
        self.motion._tick()

    def test_slide_keeps_width_content_selection_and_button_position(self):
        button_position = self.button.place_info()
        identity = str(self.listbox)
        self.motion.set_collapsed(True)
        self.assertTrue(self.motion.running)
        self.assertEqual(self.panel.winfo_manager(), "place")
        positions = [int(self.panel.place_info()["x"])]
        for _ in range(3):
            self.advance(0.05)
            positions.append(int(self.panel.place_info()["x"]))
            self.assertEqual(int(self.panel.place_info()["width"]), 340)
            self.assertEqual(self.button.place_info(), button_position)
        self.assertEqual(positions, sorted(positions))
        self.assertGreater(positions[-1], positions[0])
        self.assertEqual(self.settled.call_count, 0)
        self.advance(0.2)
        self.assertFalse(self.motion.running)
        self.assertEqual(self.panel.winfo_manager(), "")
        self.assertEqual(self.container.grid_columnconfigure(1)["minsize"], 0)
        self.assertEqual(self.button.winfo_manager(), "place")
        self.assertEqual(self.listbox.curselection(), (12,))
        self.assertEqual(str(self.listbox), identity)
        self.settled.assert_called_once()

    def test_reopen_slides_left_and_restores_grid(self):
        self.motion.set_collapsed(True, animate=False)
        self.settled.reset_mock()
        self.motion.set_collapsed(False)
        self.assertEqual(int(self.panel.place_info()["x"]), 0)
        self.advance(0.12)
        self.assertLess(int(self.panel.place_info()["x"]), 0)
        self.advance(0.2)
        self.assertEqual(self.motion.visible, 1.0)
        self.assertEqual(self.panel.winfo_manager(), "grid")
        self.assertEqual(int(self.panel.grid_info()["column"]), 1)
        self.assertEqual(self.container.grid_columnconfigure(1)["minsize"], STYLE["width"])
        self.assertEqual(self.listbox.size(), 30)
        self.settled.assert_called_once()

    def test_reverse_mid_slide_has_no_position_jump_or_duplicate_timers(self):
        self.motion.set_collapsed(True)
        self.advance(0.1)
        position = self.panel.place_info()["x"]
        visible = self.motion.visible
        old_job = self.motion._job
        self.motion.set_collapsed(False)
        self.assertEqual(self.panel.place_info()["x"], position)
        self.assertEqual(self.motion.visible, visible)
        self.assertNotIn(old_job, self.root.tk.call("after", "info"))
        self.advance(0.05)
        self.assertGreater(self.motion.visible, visible)
        self.advance(0.3)
        self.assertEqual(self.panel.winfo_manager(), "grid")
        self.assertFalse(self.button.collapsed)
        self.settled.assert_called_once()

    def test_repeated_request_for_same_target_does_not_restart_animation(self):
        self.motion.set_collapsed(True)
        job = self.motion._job
        self.clock[0] = 0.1
        self.motion.set_collapsed(True)
        self.assertEqual(self.motion._job, job)
        self.assertEqual(self.motion._started, 0.0)
        self.advance(0.2)
        self.motion.set_collapsed(True)
        self.settled.assert_called_once()

    def test_no_animation_for_hidden_tab(self):
        self.motion.set_collapsed(True, animate=False)
        self.assertIsNone(self.motion._job)
        self.assertEqual(self.motion.visible, 0.0)
        self.assertEqual(self.motion_events, [True, False])
        self.settled.assert_called_once()

    def test_stall_finishes_animation_without_replaying_queued_frames(self):
        self.motion.set_collapsed(True)
        self.advance(2.0)
        self.assertFalse(self.motion.running)
        self.assertIsNone(self.motion._job)
        self.assertEqual(self.motion.visible, 0.0)
        self.settled.assert_called_once()

    def test_slide_configure_events_do_not_relayout_root_overlays(self):
        self.assertIsNone(self.motion._on_configure(None))
        self.motion.set_collapsed(True)
        self.assertEqual(self.motion._on_configure(None), "break")
        self.advance(0.3)
        self.assertIsNone(self.motion._on_configure(None))

    def test_parent_resize_keeps_panel_attached_to_right_edge(self):
        self.motion.set_collapsed(True)
        self.advance(0.1)
        self.container.configure(width=1400, height=900)
        self.advance(0.03)
        placement = self.panel.place_info()
        self.assertEqual(float(placement["relx"]), 1.0)
        self.assertEqual(float(placement["relheight"]), 1.0)
        self.assertEqual(int(placement["width"]), 340)

    def test_destroy_cancels_timer_and_does_not_refresh_dead_ui(self):
        self.motion.set_collapsed(True)
        job = self.motion._job
        self.panel.destroy()
        self.assertIsNone(self.motion._job)
        self.assertNotIn(job, self.root.tk.call("after", "info"))
        self.assertFalse(self.motion_events[-1])
        self.settled.assert_not_called()
        self.motion.set_collapsed(False)
        self.assertIsNone(self.motion._job)

    def test_child_destroy_does_not_cancel_animation(self):
        self.motion.set_collapsed(True)
        self.listbox.destroy()
        self.assertTrue(self.motion.running)
        self.advance(0.3)
        self.settled.assert_called_once()

    def test_button_copy_icon_and_keyboard_invoke(self):
        direction = self.button.coords("direction")
        for _ in range(3):
            self.button.set_collapsed(True)
            self.assertNotEqual(self.button.coords("direction"), direction)
            self.assertEqual(self.button.itemcget("label", "text"), "Pokaż projekty")
            self.button.set_collapsed(False)
            self.assertEqual(self.button.coords("direction"), direction)
            self.assertEqual(self.button.itemcget("label", "text"), "Ukryj panel")
        self.assertTrue(self.button.bind("<Return>"))
        self.assertTrue(self.button.bind("<space>"))
        self.assertEqual(self.button._invoke(), "break")
        self.command.assert_called_once()

    def test_all_themes_keep_label_within_button_and_use_palette(self):
        for theme in THEME_DEFINITIONS:
            palette = get_theme_palette(theme)
            self.button.set_palette(palette)
            for collapsed in (False, True):
                self.button.set_collapsed(collapsed)
                bounds = self.button.bbox("label")
                self.assertLess(bounds[2], self.button._width)
                self.assertLess(bounds[3], self.button._height)
                self.assertNotEqual(self.button.itemcget("label", "fill"), self.button.itemcget("shell", "fill"))
                self.assertEqual(self.button.itemcget("shell", "outline"), palette["accent"])

    def test_shell_preserves_graph_origin_and_forwards_visibility(self):
        graph = tk.Canvas(self.container)
        owner = SimpleNamespace(
            _project_side_panel_collapsed=False, _project_sidebar_motion=self.motion,
            _campaign_graph_view={"layout_width": 900.0, "layout_offset_x": -87.0},
            campaign_transition_graph_canvas=graph, frame=self.container, app=Mock(),
        )
        shell._set_project_side_panel_collapsed(owner, True, from_graph=True)
        self.assertTrue(owner._project_side_panel_collapsed)
        self.assertEqual(owner._campaign_graph_view["layout_offset_x"], -87.0)
        self.assertEqual(self.panel.winfo_manager(), "")
        owner.app.update_status.assert_called_once()
        shell._toggle_project_side_panel(owner)
        self.assertFalse(owner._project_side_panel_collapsed)
        self.assertEqual(self.panel.winfo_manager(), "grid")

    def test_graph_fullscreen_restores_both_sidebar_states(self):
        for initial_collapsed in (False, True):
            owner = SimpleNamespace(_campaign_graph_fullscreen=False,
                _project_side_panel_collapsed=initial_collapsed, _project_sidebar_motion=Mock(),
                frame=Mock(), main_container=Mock(), app=Mock(),
                _apply_campaign_graph_fullscreen_layout=Mock())
            with patch.object(shell, "_apply_campaign_graph_fullscreen_layout"):
                shell._toggle_campaign_graph_fullscreen(owner)
                self.assertTrue(owner._project_side_panel_collapsed)
                shell._toggle_campaign_graph_fullscreen(owner)
                self.assertEqual(owner._project_side_panel_collapsed, initial_collapsed)
                self.assertFalse(owner._campaign_graph_fullscreen)

    def test_build_actual_shell_and_hide_show_without_rebuilding_projects(self):
        frame = ttk.Frame(self.root)
        owner = SimpleNamespace(frame=frame, app=SimpleNamespace(palette=get_theme_palette()),
            _project_side_panel_collapsed=False)
        for name in ("_open_selected_project", "_delete_project", "_build_projects_browser",
                     "_build_model_status", "_toggle_project_completion", "_exit_project_mode",
                     "_rebuild_wizard_stage_ui", "_on_global_mousewheel", "_get_campaign_green_accent"):
            setattr(owner, name, Mock())
        for name in ("_sync_left_panel_canvas_width", "_sync_right_panel_canvas_width",
                     "_sync_left_panel_scrollregion", "_sync_right_panel_scrollregion"):
            setattr(owner, name, lambda event=None, n=name: getattr(shell, n)(owner, event))
        owner._campaign_graph_refresh_after_layout_change = Mock()
        try:
            with patch.object(shell.HELP, "bind_help"):
                shell._build_ui(owner)
            self.root.update_idletasks()
            owner._campaign_graph_refresh_after_layout_change.reset_mock()
            shell._set_project_side_panel_collapsed(owner, True)
            self.root.update_idletasks()
            self.assertEqual(owner.right_sidebar_tab_host.winfo_manager(), "place")
            self.assertEqual(owner.left_panel_host.winfo_manager(), "")
            shell._set_project_side_panel_collapsed(owner, False)
            self.root.update_idletasks()
            self.assertEqual(owner.left_panel_host.winfo_manager(), "grid")
            owner._build_projects_browser.assert_called_once()
            self.assertEqual(owner._campaign_graph_refresh_after_layout_change.call_count, 2)
        finally:
            frame.destroy()


if __name__ == "__main__":
    unittest.main()
