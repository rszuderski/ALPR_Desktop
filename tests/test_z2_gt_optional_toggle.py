from pathlib import Path


def test_z2_entry_has_no_gt_on_off_checkbox():
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "auto_annotation_tool/gui/z2_main_widgets.py",
        root / "auto_annotation_tool/gui/tab_annotation.py",
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in sources
    )

    assert "gt_capture_check" not in combined
    assert "gt_capture_hint_lbl" not in combined
    assert "Zapisuj GT znaków" not in combined

def test_z2_has_no_canvas_gt_toggle_anywhere():
    root = Path(__file__).resolve().parents[1]
    preview = (root / "auto_annotation_tool/gui/z2_preview_editor.py").read_text(encoding="utf-8-sig")
    inline = (root / "auto_annotation_tool/gui/z2_plate_gt_inline.py").read_text(encoding="utf-8-sig")
    assert "gt_icon_x1" not in preview
    assert "gt_icon_x2" not in preview
    assert "toggle_gt_mode(self)" not in preview
    assert "_plate_gt_inline_enabled" not in inline
    assert "_plate_gt_inline_mode_target" not in inline
    start = inline.index("def draw_gt_mode_toggle")
    end = inline.find("\ndef ", start + 1)
    if end < 0:
        end = len(inline)
    body = inline[start:end]
    assert "create_rectangle" not in body
    assert "create_text" not in body
    assert "create_oval" not in body

def test_z2_canvas_hides_gt_pack_infrastructure():
    root = Path(__file__).resolve().parents[1]
    preview = (root / "auto_annotation_tool/gui/z2_preview_editor.py").read_text(encoding="utf-8-sig")
    assert "z2_gt_pack_dialog" not in preview
    assert "preview_gt_pack_status" not in preview
    assert "pack_text" not in preview
    assert "pack_box_w" not in preview
    assert "open_gt_pack_manager(self)" not in preview
