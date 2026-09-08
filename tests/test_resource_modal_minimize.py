import ast
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import Mock

from auto_annotation_tool.gui import app_window_recovery as recovery
from auto_annotation_tool.gui.native_modal_minimize import NativeModalMinimize, SC_MINIMIZE, WM_SYSCOMMAND, WM_NCDESTROY


class FakeWindow:
    def __init__(self, state="normal"):
        self.current_state = state
        self._aat_skip_window_recovery = False
        self.winfo_exists = Mock(return_value=True)
        self.winfo_toplevel = Mock(return_value=self)
        self.winfo_children = Mock(return_value=[])
        self.winfo_viewable = Mock(return_value=state in {"normal", "zoomed"})
        self.grab_current = Mock(return_value=None)
        self.grab_set = Mock()
        self.grab_release = Mock()
        self.lift = Mock()
        self.focus_force = Mock()
        self.deiconify = Mock()
        self.bindings = {}

    def state(self):
        return self.current_state

    def bind(self, name, callback, add=None):
        self.bindings.setdefault(name, []).append(callback)

    def emit(self, name, widget=None):
        for callback in self.bindings.get(name, []):
            callback(SimpleNamespace(widget=self if widget is None else widget))


class ResourceModalMinimizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gui = Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui"
        module = ast.parse((gui / "app.py").read_text(encoding="utf-8-sig"))
        app_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "AutoAnnotationApp")
        names = {"_restore_free_mode_assistant_owner", "_place_free_mode_assistant"}
        body = [node for node in app_class.body if isinstance(node, ast.FunctionDef) and node.name in names]
        cls.assistant = {"tk": tk}
        exec(compile(ast.Module(body=body, type_ignores=[]), "assistant_window_callbacks", "exec"), cls.assistant)
        graph = ast.parse((gui / "campaign_dashboard_ui.py").read_text(encoding="utf-8-sig"))
        renderer = next(node for node in graph.body if isinstance(node, ast.FunctionDef) and node.name == "_render_step1_route_actions")
        cls.resources = next(node for node in renderer.body if isinstance(node, ast.FunctionDef) and node.name == "_open_resources")

    def setUp(self):
        self.root = FakeWindow()
        self.dialog = FakeWindow()
        self.app = SimpleNamespace(root=self.root, _iter_loaded_tabs=lambda: [],
            _free_mode_assistant_context_override_owner=self.dialog)
        self.root.winfo_children.return_value = [self.dialog]

    def test_resources_installs_native_minimize_policy(self):
        calls = [node for node in self.resources.body if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == "configure_minimizable_modal"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].value.args[0].id, "dialog")

    def test_root_focus_does_not_restore_minimized_resources(self):
        recovery.configure_minimizable_modal(self.dialog)
        self.dialog.current_state = "iconic"
        recovery.on_root_focus_in(self.app, SimpleNamespace(widget=self.root))
        self.dialog.deiconify.assert_not_called()
        self.dialog.lift.assert_not_called()
        self.dialog.focus_force.assert_not_called()

    def test_unmap_releases_only_own_grab_and_does_not_raise_windows(self):
        recovery.configure_minimizable_modal(self.dialog)
        self.dialog.current_state = "iconic"
        self.dialog.grab_current.return_value = self.dialog
        self.dialog.emit("<Unmap>")
        self.dialog.grab_release.assert_called_once()
        self.dialog.deiconify.assert_not_called()
        self.dialog.lift.assert_not_called()
        self.root.deiconify.assert_not_called()

    def test_restore_reacquires_grab_without_forced_activation(self):
        recovery.configure_minimizable_modal(self.dialog)
        self.dialog.current_state = "normal"
        self.dialog.emit("<Map>")
        self.dialog.grab_set.assert_called_once()
        self.dialog.deiconify.assert_not_called()
        self.dialog.focus_force.assert_not_called()

    def test_child_events_do_not_change_parent_capture(self):
        recovery.configure_minimizable_modal(self.dialog)
        self.dialog.current_state = "iconic"
        self.dialog.grab_current.return_value = self.dialog
        self.dialog.emit("<Unmap>", widget=FakeWindow())
        self.dialog.current_state = "normal"
        self.dialog.emit("<Map>", widget=FakeWindow())
        self.dialog.grab_release.assert_not_called()
        self.dialog.grab_set.assert_not_called()

    def test_restoring_parent_does_not_steal_child_grab(self):
        recovery.configure_minimizable_modal(self.dialog)
        self.dialog.grab_current.return_value = FakeWindow()
        self.dialog.emit("<Map>")
        self.dialog.grab_set.assert_not_called()

    def test_policy_installation_does_not_duplicate_bindings(self):
        recovery.configure_minimizable_modal(self.dialog)
        recovery.configure_minimizable_modal(self.dialog)
        self.assertEqual(len(self.dialog.bindings["<Map>"]), 1)
        self.assertEqual(len(self.dialog.bindings["<Unmap>"]), 1)

    def test_child_completion_does_not_unminimize_parent(self):
        self.dialog.current_state = "iconic"
        body = [node for node in self.resources.body if isinstance(node, ast.FunctionDef)
                and node.name == "_restore_resource_dialog_after_child"]
        namespace = {"dialog": self.dialog, "restore_visible_modal": recovery.restore_visible_modal}
        exec(compile(ast.Module(body=body, type_ignores=[]), "resource_child_return", "exec"), namespace)
        namespace["_restore_resource_dialog_after_child"]()
        self.dialog.deiconify.assert_not_called()
        self.dialog.lift.assert_not_called()
        self.dialog.grab_set.assert_not_called()

    def test_return_to_visible_parent_still_works(self):
        self.assertTrue(recovery.restore_visible_modal(self.dialog))
        self.dialog.lift.assert_called_once()
        self.dialog.grab_set.assert_called_once()
        self.dialog.deiconify.assert_not_called()

    def test_assistant_does_not_restore_or_place_itself_over_hidden_owner(self):
        recovery.configure_minimizable_modal(self.dialog)
        self.app._normalize_free_mode_assistant_owner = lambda owner: owner
        self.app._free_mode_assistant_enabled = True
        self.app._get_free_mode_assistant_owner = lambda: self.dialog
        self.app._ensure_free_mode_assistant_overlay = Mock()
        for state in ("iconic", "withdrawn"):
            self.dialog.current_state = state
            self.assistant["_restore_free_mode_assistant_owner"](self.app, self.dialog)
            self.assistant["_place_free_mode_assistant"](self.app)
        self.app._ensure_free_mode_assistant_overlay.assert_not_called()
        self.dialog.deiconify.assert_not_called()
        self.dialog.lift.assert_not_called()

    def test_assistant_still_places_after_user_restores_dialog(self):
        recovery.configure_minimizable_modal(self.dialog)
        self.app._free_mode_assistant_enabled = True
        self.app._get_free_mode_assistant_owner = lambda: self.dialog
        self.app._ensure_free_mode_assistant_overlay = Mock()
        self.app._restore_free_mode_assistant_owner = Mock()
        self.assistant["_place_free_mode_assistant"](self.app)
        self.app._ensure_free_mode_assistant_overlay.assert_called_once_with(self.dialog)


class NativeMinimizeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.window = Mock()
        self.window.grab_current.return_value = self.window
        self.window.winfo_toplevel.return_value = self.window
        self.window.after.return_value = "poll-job"
        self.window.after_idle.return_value = "post-job"
        self.bridge = NativeModalMinimize.__new__(NativeModalMinimize)
        self.bridge.window = self.window
        self.bridge.hwnd = 123
        self.bridge.closed = False
        self.bridge._minimize_job = None
        self.bridge._poll_job = None
        self.bridge._requested_command = None
        self.bridge._forwarding = False
        self.bridge._id = 42
        self.bridge._callback = object()
        self.bridge._user32 = Mock()
        self.bridge._user32.PostMessageW.return_value = True
        self.bridge._comctl32 = Mock()
        self.bridge._comctl32.DefSubclassProc.return_value = 99

    def send(self, message=WM_SYSCOMMAND, command=SC_MINIMIZE):
        return self.bridge._dispatch(123, message, command, 0, 42, 0)

    def test_native_callback_only_enqueues_and_never_calls_tk(self):
        self.assertEqual(self.send(command=SC_MINIMIZE | 2), 0)
        self.assertEqual(self.window.method_calls, [])
        self.bridge._comctl32.DefSubclassProc.assert_not_called()
        self.assertEqual(self.bridge._requested_command, (123, SC_MINIMIZE | 2, 0))

    def test_release_on_tk_callback_then_forward_once_after_idle(self):
        self.send()
        self.bridge._poll()
        self.window.grab_release.assert_called_once()
        self.bridge._user32.PostMessageW.assert_not_called()
        self.window.grab_current.return_value = None
        self.window.after_idle.call_args.args[0]()
        self.bridge._user32.PostMessageW.assert_called_once_with(123, WM_SYSCOMMAND, SC_MINIMIZE, 0)
        self.assertEqual(self.send(), 99)
        self.assertFalse(self.bridge._forwarding)
        self.bridge._comctl32.DefSubclassProc.assert_called_once()

    def test_other_native_commands_are_not_changed(self):
        for command in (0xF030, 0xF120, 0xF060, 0xF010):
            self.assertEqual(self.send(command=command), 99)
        self.assertEqual(self.window.method_calls, [])
        self.assertIsNone(self.bridge._requested_command)

    def test_cannot_minimize_parent_while_child_has_grab(self):
        child = FakeWindow()
        self.window.grab_current.return_value = child
        self.send()
        self.bridge._poll()
        self.window.grab_release.assert_not_called()
        child.grab_release.assert_not_called()
        self.window.after_idle.assert_not_called()

    def test_new_child_before_idle_prevents_minimize(self):
        self.send()
        self.bridge._poll()
        self.window.grab_current.return_value = FakeWindow()
        self.window.after_idle.call_args.args[0]()
        self.bridge._user32.PostMessageW.assert_not_called()

    def test_repeated_clicks_are_coalesced(self):
        self.send()
        self.send()
        self.bridge._poll()
        self.window.after_idle.assert_called_once()

    def test_close_cancels_pending_jobs_and_removes_native_hook(self):
        self.send()
        self.bridge._poll()
        self.bridge.close()
        self.assertTrue(self.bridge.closed)
        self.assertIsNone(self.bridge.hwnd)
        self.assertIsNone(self.bridge._poll_job)
        self.assertIsNone(self.bridge._minimize_job)
        self.assertEqual(self.window.after_cancel.call_count, 2)
        self.bridge._comctl32.RemoveWindowSubclass.assert_called_once()
        self.bridge._forward_minimize(123, SC_MINIMIZE, 0)
        self.bridge._user32.PostMessageW.assert_not_called()

    def test_wrapper_destruction_never_calls_tk_from_native_code(self):
        self.send(message=WM_NCDESTROY, command=0)
        self.assertEqual(self.window.method_calls, [])
        self.assertIsNone(self.bridge.hwnd)
        self.bridge._comctl32.RemoveWindowSubclass.assert_called_once()

    def test_stale_wrapper_request_is_discarded(self):
        self.bridge.hwnd = 456
        self.bridge._forward_minimize(123, SC_MINIMIZE, 0)
        self.bridge._user32.PostMessageW.assert_not_called()

    def test_failed_post_restores_modal_capture(self):
        self.window.grab_current.return_value = None
        self.window.state.return_value = "normal"
        self.bridge._user32.PostMessageW.return_value = False
        self.bridge._forward_minimize(123, SC_MINIMIZE, 0)
        self.window.grab_set.assert_called_once()
        self.assertFalse(self.bridge._forwarding)


if __name__ == "__main__":
    unittest.main()
