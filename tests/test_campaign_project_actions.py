import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import campaign_ui_helpers as ui
from auto_annotation_tool.gui import campaign_project_browser as browser
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette, THEME_DEFINITIONS


class ProjectActionLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
        except tk.TclError as exc:
            raise unittest.SkipTest(f"Tk unavailable: {exc}")
        cls.root.withdraw()
        cls.original_scaling = cls.root.tk.call("tk", "scaling")

    @classmethod
    def tearDownClass(cls):
        cls.root.tk.call("tk", "scaling", cls.original_scaling)
        cls.root.destroy()

    def setUp(self):
        self.palette = get_theme_palette()
        self.canvas = tk.Canvas(self.root, width=260, height=38, bg=self.palette["panel"], highlightthickness=0)
        self.owner = SimpleNamespace(app=SimpleNamespace(palette=self.palette),
            project_add_button_canvas=self.canvas, wizard_exit_button_canvas=None,
            _icon_button_state={}, _icon_button_images={})
        self.owner._draw_icon_button = lambda role: ui._draw_icon_button(self.owner, role)

    def tearDown(self):
        self.canvas.destroy()
        self.root.tk.call("tk", "scaling", self.original_scaling)

    def draw(self, width):
        with patch.object(self.canvas, "winfo_width", return_value=width):
            ui._draw_icon_button(self.owner, "project_add")

    def test_full_label_fits_available_width_at_different_windows_scalings(self):
        for scale in (1.0, 1.25, 1.5, 2.0, 2.5):
            self.root.tk.call("tk", "scaling", 96.0/72.0*scale)
            for width in (150, 200, 260, 290):
                with self.subTest(scale=scale, width=width):
                    self.draw(width)
                    left, top, right, bottom = self.canvas.bbox("project_action_label")
                    self.assertEqual(self.canvas.itemcget("project_action_label", "text"), "Utwórz projekt")
                    self.assertLessEqual(right, width-7)
                    self.assertGreaterEqual(top, 7)
                    self.assertLessEqual(bottom, int(self.canvas.cget("height"))-7)
                    self.assertGreater(left, self.canvas.bbox("project_action_icon")[2])

    def test_height_grows_for_wrapped_text_and_shrinks_after_widening(self):
        self.root.tk.call("tk", "scaling", 96.0/72.0*2.0)
        self.draw(150)
        narrow_height = int(self.canvas.cget("height"))
        self.draw(320)
        self.assertLess(int(self.canvas.cget("height")), narrow_height)

    def test_rerender_does_not_repeat_height_changes(self):
        self.draw(200)
        with patch.object(self.canvas, "configure", wraps=self.canvas.configure) as configure:
            for _ in range(5):
                self.draw(200)
        self.assertFalse(any("height" in call.kwargs for call in configure.call_args_list))

    def test_initial_one_pixel_viewport_does_not_grow_a_tall_button(self):
        self.draw(1)
        self.assertEqual(int(self.canvas.cget("height")), 38)
        self.assertEqual(self.canvas.find_all(), ())

    def test_pillow_availability_does_not_change_project_action_rendering(self):
        for available in (False, True):
            with patch.object(ui, "PIL_AVAILABLE", available):
                self.draw(200)
                self.assertIn("text", [self.canvas.type(item) for item in self.canvas.find_all()])
                self.assertNotIn("image", [self.canvas.type(item) for item in self.canvas.find_all()])

    def test_enabled_and_disabled_colors_follow_all_themes(self):
        for theme in THEME_DEFINITIONS:
            palette = get_theme_palette(theme)
            self.owner.app.palette = palette
            self.canvas.configure(bg=palette["panel"])
            for enabled in (True, False):
                self.owner._icon_button_state["project_add"] = {"enabled": enabled}
                self.draw(260)
                expected = palette["success"] if enabled else palette["muted"]
                self.assertEqual(self.canvas.itemcget("project_action_label", "fill"), expected)

    def test_click_still_respects_enabled_state(self):
        callback = Mock()
        self.owner._widget_contains_point = Mock(return_value=True)
        event = SimpleNamespace(widget=self.canvas, x_root=10, y_root=10)
        for enabled in (False, True):
            self.owner._icon_button_state["project_add"] = {"enabled": enabled}
            ui._on_icon_button_release(self.owner, event, role="project_add", command=callback)
            self.assertEqual(callback.call_count, int(enabled))

    def test_browser_gives_action_its_own_full_width_row(self):
        frame = ttk.Frame(self.root)
        owner = SimpleNamespace(frame=frame, app=SimpleNamespace(palette=self.palette),
            _bind_icon_button=Mock(), _draw_icon_button=Mock(), _add_new_project=Mock(),
            _get_campaign_green_accent=lambda: self.palette["success"],
            _get_project_list_selection_bg=lambda: self.palette["selection_bg"],
            _get_project_list_selection_fg=lambda: self.palette["selection_fg"],
            _on_project_changed=Mock(), _open_selected_project=Mock(), _show_project_context_menu=Mock(),
            _on_listbox_mousewheel=Mock(), _delete_project=Mock())
        try:
            with patch.object(browser.HELP, "bind_help"):
                browser._build_projects_browser(owner, frame)
            self.root.update_idletasks()
            owner.project_list_status_labels[0].configure(text="Projekty: 999 | Długi opis aktywnego projektu")
            action = owner.project_add_button_canvas
            self.assertIs(action.master, owner.project_list_status_lbl)
            self.assertIsNot(action.master, owner.project_status_top_row)
            self.assertEqual(action.pack_info()["fill"], "x")
            self.assertEqual(action.pack_info()["side"], "top")
            self.assertEqual(int(action.cget("width")), 1)
        finally:
            frame.destroy()


if __name__ == "__main__":
    unittest.main()
