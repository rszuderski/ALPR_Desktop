from types import SimpleNamespace

from auto_annotation_tool.gui.z3_gt_assist_runtime import apply_live_gt_assist


def _host():
    return SimpleNamespace(
        _get_preview_ground_truth_text=lambda data: data.get("ground_truth_text", ""),
        _preview_layout_separator_conflicts_with_chars=lambda data, chars: False,
        _sort_character_records_by_x=lambda chars, data=None: list(chars),
    )


def _rec(character, *, sign_source="yolo_symbol", symbol_source="", symbol_method=""):
    return {
        "character": character,
        "bbox": [0.0, 0.0, 10.0, 20.0],
        "sign_source": sign_source,
        "symbol_source": symbol_source,
        "symbol_method": symbol_method,
    }


def _data(chars):
    return {
        "ground_truth_text": "ABC",
        "review_state": {"status": "in_progress"},
        "characters": chars,
        "plate_layout": "single_row",
    }


def test_live_gt_assist_never_overwrites_manual_sign():
    chars = [
        _rec("A"),
        _rec("X", sign_source="manual_sign", symbol_source="manual", symbol_method="manual"),
        _rec("C"),
    ]
    data = _data(chars)

    result = apply_live_gt_assist(_host(), data)

    assert result["reason"] == "ready"
    assert chars[1]["character"] == "X"
    assert chars[1]["sign_source"] == "manual_sign"


def test_live_gt_assist_still_corrects_non_manual_sign():
    chars = [_rec("A"), _rec("X"), _rec("C")]
    data = _data(chars)

    result = apply_live_gt_assist(_host(), data)

    assert result["changed"] is True
    assert chars[1]["character"] == "B"
    assert chars[1]["sign_source"] == "gt_assisted"
    assert chars[1]["correction_source"] == "gt_assisted"


def test_manual_symbol_source_is_protected_when_legacy_sign_source_is_missing():
    chars = [
        _rec("A"),
        _rec("X", sign_source="", symbol_source="manual", symbol_method="manual"),
        _rec("C"),
    ]
    data = _data(chars)

    apply_live_gt_assist(_host(), data)

    assert chars[1]["character"] == "X"
    assert chars[1]["symbol_source"] == "manual"


def test_explicit_auto_sign_source_wins_over_manual_box_metadata():
    chars = [
        _rec("A"),
        _rec("X", sign_source="yolo_symbol", symbol_source="manual", symbol_method="manual"),
        _rec("C"),
    ]
    data = _data(chars)

    apply_live_gt_assist(_host(), data)

    assert chars[1]["character"] == "B"
    assert chars[1]["sign_source"] == "gt_assisted"
