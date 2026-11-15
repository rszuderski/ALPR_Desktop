from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import app_window_recovery as recovery
from auto_annotation_tool.gui import z4_campaign_flow as flow


class FakeRoot:
    def __init__(self, state="iconic", viewable=False):
        self.current_state = state
        self.viewable = viewable
        self.deiconify_calls = 0
        self.after_calls = []

    def state(self):
        return self.current_state

    def deiconify(self):
        self.deiconify_calls += 1
        self.current_state = "normal"
        self.viewable = True

    def update_idletasks(self):
        pass

    def lift(self, *args):
        pass

    def focus_force(self):
        pass

    def focus_set(self):
        pass

    def grab_current(self):
        return None

    def winfo_exists(self):
        return True

    def winfo_children(self):
        return []

    def winfo_viewable(self):
        return self.viewable

    def after(self, delay, callback):
        token = f"after-{len(self.after_calls)+1}"
        self.after_calls.append((token, delay, callback))
        return token

    def after_cancel(self, token):
        pass

    @property
    def tk(self):
        return SimpleNamespace(call=lambda *args: "win32")


def make_app(root):
    return SimpleNamespace(
        root=root,
        _native_file_dialog_active=False,
        _window_recovery_suspended=False,
        _window_restore_after_id=None,
        _window_restore_topmost_after_id=None,
        _window_restore_attempts=0,
        _window_restore_pending=False,
        _window_restore_requested_by_map=False,
        _help_overlay_forced_visible=False,
        _iter_loaded_tabs=lambda: [],
        _close_menu_dropdown=lambda: None,
        _hide_simple_tooltip=lambda: None,
    )


def test_unmap_does_not_mark_user_restore_request():
    root = FakeRoot("iconic", False)
    app = make_app(root)
    recovery.on_root_unmap(app, SimpleNamespace(widget=root))
    assert app._window_restore_pending is True
    assert app._window_restore_requested_by_map is False


def test_map_marks_restore_and_forces_deiconify_when_tk_is_still_iconic():
    root = FakeRoot("iconic", False)
    app = make_app(root)
    recovery.on_root_map(app, SimpleNamespace(widget=root))
    assert app._window_restore_requested_by_map is True
    recovery.recover_root_after_map(app)
    assert root.deiconify_calls >= 1
    assert root.state() == "normal"
    assert app._window_restore_pending is False
    assert app._window_restore_requested_by_map is False


def test_training_completion_summary_is_deferred_while_root_is_minimized():
    root = FakeRoot("iconic", False)
    app = SimpleNamespace(root=root, themed_info=Mock(), themed_error=Mock())
    host = SimpleNamespace(
        app=app,
        current_run_id="20261008_101010",
        _last_training_completion_summary_run_id=None,
        _training_completion_summary_after_id=None,
    )
    flow.show_training_completion_summary(host, promoted=False, can_finish_step4=False)
    app.themed_info.assert_not_called()
    app.themed_error.assert_not_called()
    assert host._training_completion_summary_after_id is not None
    assert len(root.after_calls) == 1
