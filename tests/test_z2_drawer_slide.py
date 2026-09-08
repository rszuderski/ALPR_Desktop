import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from auto_annotation_tool.gui.z2_drawer_slide import PreviewDrawerSlide
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette


class DrawerSlideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.host = tk.Frame(self.root, width=1000, height=700)
        self.panel = tk.Frame(self.host)
        self.status = tk.Label(self.panel, text="T04 / 23")
        self.status.pack()
        self.clock = [0.0]
        self.owner = SimpleNamespace(preview_overlay_dock=self.panel, canvas_frame=self.host,
                                     preview_canvas=Mock(), app=SimpleNamespace(palette=get_theme_palette()))
        self.slide = PreviewDrawerSlide(self.owner, clock=lambda: self.clock[0])
        self.place()

    def tearDown(self):
        self.host.destroy()
        self.root.update_idletasks()

    def place(self, fullscreen=True):
        self.slide.place(1000, 720, 46, 270, 400, fullscreen=fullscreen)

    def advance(self, seconds):
        self.slide._cancel()
        self.clock[0] += seconds
        self.slide._tick()

    def test_slide_preserves_panel_and_status(self):
        identity = str(self.status)
        self.slide.toggle()
        self.advance(0.1)
        self.assertGreater(int(self.panel.place_info()["x"]), 720)
        self.assertEqual(int(self.panel.place_info()["width"]), 270)
        self.advance(0.2)
        self.assertEqual(self.panel.winfo_manager(), "")
        self.assertEqual(self.slide.button.winfo_manager(), "place")
        self.assertEqual(str(self.status), identity)
        self.assertEqual(self.status.cget("text"), "T04 / 23")
        self.slide.toggle()
        self.advance(0.3)
        self.assertEqual(int(self.panel.place_info()["x"]), 720)

    def test_repeated_placement_does_not_reopen_hidden_panel(self):
        self.slide.toggle()
        self.advance(0.3)
        for _ in range(20):
            self.place()
        self.assertEqual(self.panel.winfo_manager(), "")
        self.assertIsNone(self.slide._job)

    def test_exit_fs_shows_normal_drawer_without_toggle(self):
        self.slide.toggle()
        self.advance(0.3)
        self.place(fullscreen=False)
        self.assertEqual(self.panel.winfo_manager(), "place")
        self.assertEqual(self.slide.button.winfo_manager(), "")
        self.place()
        self.assertEqual(self.panel.winfo_manager(), "")

    def test_transition_suspends_motion_but_remembers_choice(self):
        self.slide.toggle()
        self.advance(0.05)
        self.slide.suspend()
        self.assertIsNone(self.slide._job)
        self.assertEqual(self.slide.button.winfo_manager(), "")
        self.place()
        self.assertEqual(self.panel.winfo_manager(), "")

    def test_rapid_reversal_does_not_jump(self):
        self.slide.toggle()
        self.advance(0.1)
        x = self.panel.place_info()["x"]
        self.slide.toggle()
        self.assertEqual(self.panel.place_info()["x"], x)
        self.advance(0.3)
        self.assertEqual(int(self.panel.place_info()["x"]), 720)

    def test_destroy_cancels_animation(self):
        self.slide.toggle()
        job = self.slide._job
        self.panel.destroy()
        self.assertIsNone(self.slide._job)
        self.assertNotIn(job, self.root.tk.call("after", "info"))

    def test_non_fs_toggle_does_not_change_panel(self):
        self.place(fullscreen=False)
        self.slide.toggle()
        self.assertFalse(self.slide.hidden)
        self.assertIsNone(self.slide._job)
