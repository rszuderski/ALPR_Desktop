from pathlib import Path
import inspect

from auto_annotation_tool.gui import z4_fine_tune_protocol as ft


def test_panel_is_not_removed_from_geometry():
    source = inspect.getsource(ft.refresh_visibility)
    assert "pack_forget" not in source
    assert "_ensure_panel_managed(host)" in source


def test_build_panel_registers_runtime_controls():
    source = inspect.getsource(ft.build_panel)
    for token in ("AMP", "Mosaic", "Cache", "Workers", "Wykresy", "Seed", "Optimizer"):
        assert token in source
    assert "_ft_protocol_controls" in source
    assert "_ensure_panel_managed(host)" in source


def test_refresh_disables_instead_of_hiding():
    source = inspect.getsource(ft.refresh_visibility)
    assert "widget.configure" in source
    assert "tk.DISABLED" in source


def test_runtime_forces_refresh_after_selecting_fine_tune_parent():
    root = Path(__file__).resolve().parents[1]
    runtime = (root / "auto_annotation_tool/gui/z4_training_runtime.py").read_text(encoding="utf-8-sig")
    start = runtime.index("def _select_selected_run_as_fine_tune_base")
    end = runtime.find("\ndef ", start + 10)
    chunk = runtime[start:end if end > start else len(runtime)]
    assert "z4_fine_tune_protocol.refresh_visibility(self)" in chunk
