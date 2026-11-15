from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab


def _host():
    return CharacterAnnotationTab.__new__(CharacterAnnotationTab)


def _two_row_data(separator=None):
    return {
        "plate_layout": "two_row",
        "plate_layout_override": "two_row",
        "plate_image_width": 120.0,
        "plate_image_height": 100.0,
        "layout_separator": separator or {
            "x1": 0.0,
            "y1": 50.0,
            "x2": 120.0,
            "y2": 50.0,
            "source": "manual",
        },
    }


def test_hard_separator_clips_top_and_bottom_boxes_to_their_rows():
    host = _host()
    data = _two_row_data()
    chars = [
        {"character": "A", "bbox": [10, 10, 30, 65], "reading_row": 1},
        {"character": "B", "bbox": [50, 35, 70, 90], "reading_row": 2},
    ]

    result = host._apply_preview_layout_separator_constraints_to_chars(data, chars)

    assert result[0]["bbox"][3] <= 49.0
    assert result[1]["bbox"][1] >= 51.0
    assert host._preview_layout_separator_conflicts_with_chars(data, result) is False


def test_hard_separator_uses_entire_sloped_line_not_only_box_center():
    host = _host()
    data = _two_row_data({
        "x1": 0.0,
        "y1": 40.0,
        "x2": 120.0,
        "y2": 60.0,
        "source": "manual",
    })

    top = host._constrain_preview_char_bbox_to_layout_separator(
        [10, 5, 110, 70], data=data, row=1, min_size=4.0
    )
    bottom = host._constrain_preview_char_bbox_to_layout_separator(
        [10, 30, 110, 95], data=data, row=2, min_size=4.0
    )

    assert top[3] < 50.0
    assert bottom[1] > 50.0


def test_separator_candidate_is_clamped_to_existing_row_gap():
    host = _host()
    data = _two_row_data()
    data["characters"] = [
        {"character": "A", "bbox": [10, 10, 30, 35], "reading_row": 1},
        {"character": "B", "bbox": [50, 65, 70, 90], "reading_row": 2},
    ]

    candidate = {
        "x1": 0.0,
        "y1": 20.0,
        "x2": 120.0,
        "y2": 20.0,
        "source": "manual",
    }
    clamped = host._clamp_preview_layout_separator_to_existing_rows(data, candidate)

    assert clamped["y1"] >= 37.0
    assert clamped["y2"] >= 37.0


def test_raw_review_materialization_applies_hard_separator_constraints():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "auto_annotation_tool" / "gui" / "z3_review_runtime.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def prepare_working_annotation_from_raw")
    end = source.index("\ndef start_review_from_raw", start)
    body = source[start:end]

    assert "_apply_preview_layout_separator_constraints_to_chars(" in body
