from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8-sig")


def _function_source(text: str, name: str) -> str:
    marker = f"def {name}("
    start = text.index(marker)
    next_def = text.find("\ndef ", start + len(marker))
    return text[start:] if next_def < 0 else text[start:next_def]


def test_cvat_import_reopens_review_before_replacing_characters():
    source = _source("auto_annotation_tool/gui/z3_cvat_tab_ui.py")
    body = _function_source(source, "run_cvat_import")

    mark = body.index("host._mark_review_edit_started(target_data)")
    write = body.index('target_data["characters"] = clean_chars')

    assert mark < write


def test_gt_assist_accept_reopens_review_before_replacing_characters():
    source = _source("auto_annotation_tool/gui/z3_gt_assist_runtime.py")
    body = _function_source(source, "accept_gt_assist_suggestion")

    mark = body.index("host._mark_review_edit_started(data)")
    write = body.index('data["characters"] = accepted')

    assert mark < write


def test_manual_layout_override_reopens_review_before_layout_mutation():
    source = _source("auto_annotation_tool/gui/z3_plate_layout_runtime.py")
    body = _function_source(source, "_apply_preview_plate_layout_override")

    mark = body.index("self._mark_review_edit_started(data)")
    mutation = body.index('data["plate_layout_override"] = next_override')

    assert mark < mutation


def test_separator_release_reopens_review_and_uses_canonical_status_gate():
    source = _source("auto_annotation_tool/gui/z3_preview_events.py")
    body = _function_source(source, "on_preview_canvas_release")

    separator_start = body.index(
        'separator_drag_state = getattr(self, "_preview_layout_separator_drag_state", None)'
    )
    char_drag_start = body.index(
        'char_drag_state = getattr(self, "_preview_char_drag_state", None)'
    )
    separator_branch = body[separator_start:char_drag_start]

    mark = separator_branch.index("self._mark_review_edit_started(data)")
    mutation = separator_branch.index('data["layout_separator"] = dict(separator)')

    assert mark < mutation
    assert "self._derive_preview_status_from_data(" in separator_branch
    assert "ordered_chars" in separator_branch
    assert "self._derive_preview_status_from_characters(" not in separator_branch
