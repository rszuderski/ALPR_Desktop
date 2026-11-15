"""Z4LIB002: static regression contract for native library window controls.

The native Win32 caption must additionally be checked manually on Windows.
"""
from pathlib import Path
import ast

SOURCE = Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/z4_mobile_artifact_library.py"


def _window_calls():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
            owner = func.value
            if isinstance(owner.value, ast.Name) and owner.value.id == "self" and owner.attr == "window":
                calls.append((func.attr, node))
    return calls


def test_library_is_modeless_with_native_window_controls():
    source = SOURCE.read_text(encoding="utf-8-sig")
    assert 'self.window = tk.Toplevel(parent)' in source
    calls = _window_calls()
    names = [name for name, _ in calls]
    assert "transient" not in names
    assert "grab_set" not in names
    assert "overrideredirect" not in names
    controls = [node for name, node in calls if name == "resizable"]
    assert len(controls) == 1
    assert len(controls[0].args) == 2
    assert all(isinstance(arg, ast.Constant) and arg.value is True for arg in controls[0].args)


def test_library_keeps_minimum_size_and_close_protocol():
    source = SOURCE.read_text(encoding="utf-8-sig")
    assert 'self.window.minsize(940, 570)' in source
    assert 'self.window.geometry("1300x790")' in source
    assert 'self.window.protocol("WM_DELETE_WINDOW", self.close)' in source
