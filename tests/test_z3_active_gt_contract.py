from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.data_models import Detection
from auto_annotation_tool.gui import z3_preview_metadata_runtime as runtime
from auto_annotation_tool.gui.z3_extraction_sources import (
    prepare_plate_cut_detections_for_source,
)


class Host:
    def __init__(self, parser):
        self.parser = parser

    def _get_preview_active_data(self, create=False):
        return {}

    def _get_true_texts_from_filename(self, candidate):
        return self.parser(candidate)

    def _get_preview_reference_text_values(self, data=None):
        return runtime._get_preview_reference_text_values(self, data)

    def _get_preview_ground_truth_text(self, data=None):
        return runtime._get_preview_ground_truth_text(self, data)

    def _preview_uses_plate_gt_contract(self, data=None):
        return runtime._preview_uses_plate_gt_contract(self, data)

    def _get_preview_filename_expected_texts(self, data=None):
        return runtime._get_preview_filename_expected_texts(self, data)

    def _get_preview_expected_texts(self, data=None):
        return runtime._get_preview_expected_texts(self, data)

    def _resolve_preview_expected_text_for_crop(self, data=None, chars=None):
        return runtime._resolve_preview_expected_text_for_crop(self, data, chars)

    def _characters_to_text(self, chars, data=None):
        return "".join(str(rec.get("character", "") or "") for rec in chars)


def plate(attributes=None):
    return Detection(
        label="plate",
        confidence=1.0,
        bbox=(0.0, 0.0, 20.0, 10.0),
        polygon=[
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 10.0),
            (0.0, 10.0),
        ],
        attributes=dict(attributes or {}),
    )


def chars(text):
    return [
        {
            "character": char,
            "bbox": [idx * 10, 0, (idx * 10) + 8, 20],
        }
        for idx, char in enumerate(text)
    ]


def test_pz1_never_creates_gt_from_filename_anymore():
    parser = Mock(side_effect=AssertionError("filename parser must not run"))
    host = SimpleNamespace(
        _plate_cut_reading_order_key=lambda pair: pair[0],
        _extract_source_plate_tokens_from_filename=parser,
    )
    det = plate()

    result = prepare_plate_cut_detections_for_source(
        host,
        "ABC123_001.jpg",
        [det],
    )

    assert result == [det]
    assert det.attributes["plate_annotation_id"].startswith("plate-ann-")
    assert det.attributes["source_plate_index"] == "0"
    assert det.attributes["source_plate_count"] == "1"
    assert "source_expected_text" not in det.attributes
    assert "source_expected_texts" not in det.attributes
    assert "source_expected_text_source" not in det.attributes
    parser.assert_not_called()


def test_pz1_removes_old_filename_gt_when_creating_current_contract():
    parser = Mock(side_effect=AssertionError("filename parser must not run"))
    host = SimpleNamespace(
        _plate_cut_reading_order_key=lambda pair: pair[0],
        _extract_source_plate_tokens_from_filename=parser,
    )
    det = plate(
        {
            "source_expected_text": "ABC123",
            "source_expected_texts": '["ABC123"]',
            "source_expected_text_source": "filename_order",
        }
    )

    prepare_plate_cut_detections_for_source(
        host,
        "ABC123_001.jpg",
        [det],
    )

    assert det.attributes["plate_annotation_id"].startswith("plate-ann-")
    assert "source_expected_text" not in det.attributes
    assert "source_expected_texts" not in det.attributes
    assert "source_expected_text_source" not in det.attributes
    parser.assert_not_called()


def test_gt_resolver_has_simple_non_ambiguous_semantics():
    parser = Mock(side_effect=AssertionError("filename parser must not run"))
    host = Host(parser)
    data = {
        "source_image": "AAA_BBB_001.jpg",
        "source_annotation_id": "plate-ann-one",
        "ground_truth_text": "WI905PW",
        "ground_truth_source": "manual_z2",
    }

    unresolved = host._resolve_preview_expected_text_for_crop(data, [])
    assert unresolved["target_text"] == "WI905PW"
    assert unresolved["target_length"] == 7
    assert unresolved["target_lengths"] == [7]
    assert unresolved["count_resolved"] is True
    assert unresolved["text_resolved"] is False
    assert unresolved["ambiguous"] is False
    assert unresolved["resolution"] == "ground_truth_expected"

    resolved = host._resolve_preview_expected_text_for_crop(
        data,
        chars("WI905PW"),
    )
    assert resolved["matched_text"] == "WI905PW"
    assert resolved["text_resolved"] is True
    assert resolved["resolution"] == "ground_truth_exact"
    parser.assert_not_called()


def test_missing_gt_in_current_contract_is_explicit_not_ambiguous():
    parser = Mock(side_effect=AssertionError("filename parser must not run"))
    host = Host(parser)
    data = {
        "source_image": "ABC123_001.jpg",
        "source_annotation_id": "plate-ann-no-gt",
        "ground_truth_text": None,
    }

    result = host._resolve_preview_expected_text_for_crop(
        data,
        chars("ABC123"),
    )

    assert result["expected_texts"] == []
    assert result["target_text"] == ""
    assert result["target_length"] == 0
    assert result["count_resolved"] is False
    assert result["text_resolved"] is False
    assert result["ambiguous"] is False
    assert result["expected_source"] == "ground_truth"
    assert result["resolution"] == "missing_ground_truth"
    parser.assert_not_called()
