from pathlib import Path


def _run_details_chunk():
    root = Path(__file__).resolve().parents[1]
    path = root / "auto_annotation_tool" / "gui" / "z4_training_runtime.py"
    text = path.read_text(encoding="utf-8-sig")
    start = text.index("def _open_run_details_modal")
    end = text.find("\ndef ", start + 10)
    return text[start:end if end > start else len(text)]


def test_run_details_window_is_not_transient():
    chunk = _run_details_chunk()
    assert "dialog.transient(self.frame.winfo_toplevel())" not in chunk
    assert 'dialog.wm_transient("")' in chunk


def test_run_details_window_keeps_native_min_max_frame():
    chunk = _run_details_chunk()
    assert 'dialog.attributes("-toolwindow", False)' in chunk
    assert "dialog.resizable(True, True)" in chunk


def test_run_details_reapplies_normal_window_style_on_show():
    chunk = _run_details_chunk()
    show_pos = chunk.rfind('dialog.title(f"Szczegóły runu')
    assert show_pos > 0
    before = chunk[max(0, show_pos - 500):show_pos]
    assert 'dialog.wm_transient("")' in before
