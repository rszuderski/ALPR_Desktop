import inspect
from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool import plate_ground_truth as pgt
from auto_annotation_tool.gui import z2_plate_gt_inline
from auto_annotation_tool.gui import z3_extraction_sources
from auto_annotation_tool.character_recognition import plate_generator


def test_layout_gt_contract_defaults_to_1r():
    attrs = {}
    assert pgt.get_plate_layout_gt(attrs) == pgt.PLATE_LAYOUT_SINGLE_ROW
    assert pgt.has_explicit_plate_layout_gt(attrs) is False


def test_detection_contract_materializes_default_1r():
    det = SimpleNamespace(attributes={})
    attrs = pgt.ensure_plate_detection_contract(det)
    assert attrs[pgt.PLATE_LAYOUT_GT_ATTR] == pgt.PLATE_LAYOUT_SINGLE_ROW
    assert det.attributes[pgt.PLATE_LAYOUT_GT_ATTR] == pgt.PLATE_LAYOUT_SINGLE_ROW


def test_layout_gt_contract_accepts_ui_aliases():
    attrs = {}
    assert pgt.set_plate_layout_gt(attrs, "2R") == pgt.PLATE_LAYOUT_TWO_ROW
    assert attrs[pgt.PLATE_LAYOUT_GT_ATTR] == pgt.PLATE_LAYOUT_TWO_ROW
    assert pgt.set_plate_layout_gt(attrs, "1R") == pgt.PLATE_LAYOUT_SINGLE_ROW
    assert attrs[pgt.PLATE_LAYOUT_GT_ATTR] == pgt.PLATE_LAYOUT_SINGLE_ROW


def test_z2_toggle_defaults_from_1r_to_2r_and_back():
    det = SimpleNamespace(attributes={"ground_truth_text": "WX12345"})
    ann = SimpleNamespace()
    button = Mock()
    layout_var = Mock()
    frame = Mock()
    host = SimpleNamespace(
        frame=frame,
        app=SimpleNamespace(palette={}),
        _plate_gt_inline_widgets={},
        _preview_is_editable=lambda: True,
        _mark_preview_image_dirty=Mock(),
        _save_preview_edits=Mock(return_value=True),
        _update_preview_edit_status=Mock(),
    )
    key = "plate:test"
    host._plate_gt_inline_widgets[key] = {
        "det": det,
        "ann": ann,
        "layout_button": button,
        "layout_var": layout_var,
        "layout_preview": "",
    }
    assert z2_plate_gt_inline._toggle_record_layout(host, key) == "break"
    assert det.attributes[pgt.PLATE_LAYOUT_GT_ATTR] == pgt.PLATE_LAYOUT_TWO_ROW
    assert z2_plate_gt_inline._toggle_record_layout(host, key) == "break"
    assert det.attributes[pgt.PLATE_LAYOUT_GT_ATTR] == pgt.PLATE_LAYOUT_SINGLE_ROW


def test_gt_text_save_materializes_visible_default_1r():
    source = inspect.getsource(z2_plate_gt_inline.save_inline_plate_gt_value)
    assert "ensure_plate_layout_gt(det.attributes)" in source


def test_z2_editor_reads_default_without_hidden_save_side_effect():
    source = inspect.getsource(z2_plate_gt_inline._create_editor)
    assert "layout_button" in source
    assert "layout_var" in source
    assert "_toggle_record_layout" in source
    assert "layout_value = get_plate_layout_gt(" in source


def test_pz1_gt_hash_includes_layout_gt():
    base = {
        "plate_annotation_id": "plate-ann-test",
        "ground_truth_text": "WX12345",
        "ground_truth_source": "manual_z2",
    }
    missing = z3_extraction_sources.build_plate_source_gt_hash("img.jpg", base)
    single = z3_extraction_sources.build_plate_source_gt_hash(
        "img.jpg",
        {**base, pgt.PLATE_LAYOUT_GT_ATTR: pgt.PLATE_LAYOUT_SINGLE_ROW},
    )
    two = z3_extraction_sources.build_plate_source_gt_hash(
        "img.jpg",
        {**base, pgt.PLATE_LAYOUT_GT_ATTR: pgt.PLATE_LAYOUT_TWO_ROW},
    )
    assert missing != single
    assert single != two


def test_plate_generator_propagates_layout_gt_to_pz2_override():
    source = inspect.getsource(
        plate_generator.PlateGenerator.generate_from_annotations
    )
    assert "plate_layout_gt" in source
    assert "'plate_layout_override':" in source
    assert "'z2_ground_truth'" in source
    assert "PLATE_LAYOUT_TWO_ROW" in source


def test_plate_generator_prefers_explicit_layout_over_aspect_ratio():
    source = inspect.getsource(
        plate_generator.PlateGenerator.generate_from_annotations
    )
    assert "if layout_gt_explicit:" in source
    assert "is_square = plate_layout_gt == PLATE_LAYOUT_TWO_ROW" in source
