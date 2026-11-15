from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.data_models import Detection
from auto_annotation_tool.gui import z3_preview_metadata_runtime as runtime
from auto_annotation_tool.gui.z3_extraction_sources import (
    backfill_preview_expected_texts_from_sources,
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

    def _preview_has_reference_text_source(self, data=None):
        return runtime._preview_has_reference_text_source(self, data)

    def _characters_to_text(self, chars, data=None):
        return "".join(str(rec.get("character", "") or "") for rec in chars)

    def _sanitize_preview_char_symbol(self, symbol):
        return str(symbol or "").strip().upper()

    def _char_record_bbox(self, rec):
        return rec.get("bbox") if isinstance(rec, dict) else None

    def _is_exportable_character_record(self, rec):
        return runtime._is_exportable_character_record(self, rec)

    def _derive_preview_status_from_characters(self, chars):
        return runtime._derive_preview_status_from_characters(self, chars)

    def _preview_layout_separator_conflicts_with_chars(self, data, chars):
        return False

    def _derive_preview_status_from_data(self, data, chars):
        return runtime._derive_preview_status_from_data(self, data, chars)


def chars(text):
    return [
        {
            "character": char,
            "bbox": [idx * 10, 0, (idx * 10) + 8, 20],
        }
        for idx, char in enumerate(text)
    ]


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


def test_ground_truth_is_primary_and_filename_parser_is_not_called():
    parser = Mock(side_effect=AssertionError("filename parser must not run"))
    host = Host(parser)
    data = {
        "source_image": "AAA_BBB_001.jpg",
        "source_annotation_id": "plate-ann-one",
        "ground_truth_text": "BBB",
        "ground_truth_source": "manual_z2",
    }

    assert host._get_preview_expected_texts(data) == ["BBB"]
    resolution = host._resolve_preview_expected_text_for_crop(data, [])
    assert resolution["target_text"] == "BBB"
    assert resolution["target_length"] == 3
    assert resolution["count_resolved"] is True
    assert resolution["expected_source"] == "ground_truth"
    parser.assert_not_called()


def test_ground_truth_controls_perfect_status():
    parser = Mock(side_effect=AssertionError("filename parser must not run"))
    host = Host(parser)
    data = {
        "source_image": "AAA_BBB_001.jpg",
        "source_annotation_id": "plate-ann-two",
        "ground_truth_text": "BBB",
        "ground_truth_source": "manual_z2",
    }

    assert host._derive_preview_status_from_data(data, chars("BBB")) == "perfect"
    assert host._derive_preview_status_from_data(data, chars("AAA")) == "needs_fix"
    parser.assert_not_called()


def test_new_contract_without_gt_never_falls_back_to_filename():
    parser = Mock(side_effect=AssertionError("filename parser must not run"))
    host = Host(parser)
    data = {
        "source_image": "AAA_BBB_001.jpg",
        "source_annotation_id": "plate-ann-missing-gt",
        "ground_truth_text": None,
        "ground_truth_source": None,
    }

    assert host._preview_uses_plate_gt_contract(data) is True
    assert host._get_preview_expected_texts(data) == []
    assert host._derive_preview_status_from_data(data, chars("AAA")) == "needs_fix"
    parser.assert_not_called()


def test_legacy_crop_without_contract_keeps_filename_fallback():
    parser = Mock(return_value=["ABC123"])
    host = Host(parser)
    data = {
        "source_image": "ABC123_001.jpg",
    }

    assert host._preview_uses_plate_gt_contract(data) is False
    assert host._get_preview_expected_texts(data) == ["ABC123"]
    parser.assert_called()


def test_pz1_contract_without_gt_does_not_create_filename_gt():
    parser = Mock(side_effect=AssertionError("filename parser must not run"))
    host = SimpleNamespace(
        _plate_cut_reading_order_key=lambda pair: pair[0],
        _extract_source_plate_tokens_from_filename=parser,
    )
    det = plate({"plate_annotation_id": "plate-ann-no-gt"})

    prepare_plate_cut_detections_for_source(
        host,
        "AAA_BBB_001.jpg",
        [det],
    )

    assert det.attributes["plate_annotation_id"] == "plate-ann-no-gt"
    assert "source_expected_text" not in det.attributes
    assert "source_expected_texts" not in det.attributes
    assert "source_expected_text_source" not in det.attributes
    parser.assert_not_called()


def test_backfill_skips_new_contract_even_when_gt_is_missing():
    parser = Mock(side_effect=AssertionError("filename parser must not run"))
    host = SimpleNamespace(
        _extract_source_plate_tokens_from_filename=parser,
        _plate_cut_reading_order_key=lambda pair: pair[0],
    )
    metadata = {
        "plate_000001": {
            "source_image": "AAA_BBB_001.jpg",
            "source_annotation_id": "plate-ann-no-gt",
            "ground_truth_text": None,
            "plate_attributes": {
                "plate_annotation_id": "plate-ann-no-gt",
            },
        }
    }

    assert not backfill_preview_expected_texts_from_sources(host, metadata)
    parser.assert_not_called()
