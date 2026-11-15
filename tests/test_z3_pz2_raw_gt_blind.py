import inspect

from auto_annotation_tool.gui import z3_detection_runtime
from auto_annotation_tool.gui import z3_preview_metadata_runtime as metadata_runtime


def chars(text):
    return [
        {
            "character": char,
            "bbox": [idx * 10, 0, (idx * 10) + 8, 20],
        }
        for idx, char in enumerate(text)
    ]


class Host:
    def _characters_to_text(self, records, data=None):
        return "".join(
            str(record.get("character", "") or "")
            for record in records
        )

    def _preview_uses_plate_gt_contract(self, data=None):
        return bool(
            isinstance(data, dict)
            and data.get("source_annotation_id")
        )

    def _get_preview_ground_truth_text(self, data=None):
        return str(
            (data or {}).get("ground_truth_text", "")
            or ""
        ).strip().upper()

    def _get_preview_expected_texts(self, data=None):
        gt = self._get_preview_ground_truth_text(data)
        return [gt] if gt else []

    def _is_exportable_character_record(self, rec):
        bbox = list(rec.get("bbox", []) or [])
        return bool(
            str(rec.get("character", "") or "").strip()
            and len(bbox) >= 4
            and float(bbox[2]) > float(bbox[0])
            and float(bbox[3]) > float(bbox[1])
        )

    def _preview_layout_separator_conflicts_with_chars(
        self,
        data,
        records,
    ):
        return False


def validate(data, records):
    return metadata_runtime._build_raw_detection_validation(
        Host(),
        data,
        records,
    )


def test_raw_validation_exact_plate_match():
    data = {
        "source_annotation_id": "plate-ann-one",
        "ground_truth_text": "WI905PW",
    }

    result = validate(data, chars("WI905PW"))

    assert result["raw_contract"] == "gt_blind.v1"
    assert result["status"] == "perfect"
    assert result["exact_text_match"] is True
    assert result["exact_count_match"] is True
    assert result["geometry_ok"] is True
    assert result["edit_distance"] == 0
    assert result["cer"] == 0.0
    assert result["reason_codes"] == []


def test_raw_validation_reports_text_mismatch_without_repair():
    data = {
        "source_annotation_id": "plate-ann-one",
        "ground_truth_text": "WI905PW",
    }

    result = validate(data, chars("WI9O5PW"))

    assert result["status"] == "needs_fix"
    assert result["prediction_text"] == "WI9O5PW"
    assert result["ground_truth_text"] == "WI905PW"
    assert "text_mismatch" in result["reason_codes"]
    assert result["detected_char_count"] == 7
    assert result["expected_char_count"] == 7
    assert result["edit_distance"] == 1


def test_raw_validation_reports_missing_and_extra_boxes():
    data = {
        "source_annotation_id": "plate-ann-one",
        "ground_truth_text": "ABC123",
    }

    missing = validate(data, chars("ABC12"))
    extra = validate(data, chars("ABC1234"))

    assert "missing_boxes" in missing["reason_codes"]
    assert "extra_boxes" in extra["reason_codes"]


def test_raw_validation_missing_gt_is_explicit():
    data = {
        "source_annotation_id": "plate-ann-one",
        "ground_truth_text": None,
    }

    result = validate(data, chars("ABC123"))

    assert result["status"] == "needs_fix"
    assert "missing_ground_truth" in result["reason_codes"]
    assert result["ground_truth_text"] is None
    assert result["edit_distance"] is None
    assert result["cer"] is None


def test_detection_runtime_does_not_read_gt_before_detector_call():
    source = inspect.getsource(
        z3_detection_runtime.run_fast_ocr_test
    )
    detect_marker = "chars = detector.detect(img)"
    detect_pos = source.find(detect_marker)
    assert detect_pos > 0

    block_start = source.rfind(
        'source_image = local_meta[pid].get("source_image"',
        0,
        detect_pos,
    )
    assert block_start > 0
    pre_detection = source[block_start:detect_pos]

    assert "_get_preview_expected_texts" not in pre_detection
    assert "_get_true_texts_from_filename" not in pre_detection
    assert "_resolve_preview_expected_text_for_crop" not in pre_detection
    assert "expected_character_count = 0" in pre_detection


def test_detection_runtime_marks_saved_raw_result_as_gt_blind():
    source = inspect.getsource(
        z3_detection_runtime.run_fast_ocr_test
    )

    assert '"raw_detection"' in source
    assert '"gt_blind.v1"' in source
    assert "_build_raw_detection_validation" in source

def test_raw_validation_carries_revision_and_raw_result_contract():
    data = {
        "source_annotation_id": "plate-ann-one",
        "source_gt_hash": "gt-hash-v1",
        "source_gt_revision_ids": ["rev-b", "rev-a"],
        "source_geometry_revision_ids": ["geom-one"],
        "ground_truth_text": "ABC123",
        "raw_detection": {
            "schema": "alpr.pz2.raw_detection.v1",
            "contract": "gt_blind.v1",
            "prediction_text": "ABC123",
            "characters": chars("ABC123"),
        },
    }

    result = validate(data, chars("ABC123"))

    assert result["gt_hash"] == "gt-hash-v1"
    assert result["gt_revision_id"] is None
    assert result["gt_revision_ids"] == ["rev-a", "rev-b"]
    assert result["geometry_revision_id"] == "geom-one"
    assert result["geometry_revision_ids"] == ["geom-one"]
    assert len(result["raw_result_hash"]) == 64


def test_detection_runtime_persists_canonical_raw_result_hash():
    source = inspect.getsource(
        z3_detection_runtime.run_fast_ocr_test
    )

    assert 'raw_detection["result_hash"]' in source
    assert "canonical_raw_detection_hash(raw_detection)" in source

def test_raw_validation_reports_exact_plate_and_box_count_error():
    exact = validate(
        {
            "source_annotation_id": "plate-ann-one",
            "ground_truth_text": "ABC123",
        },
        chars("ABC123"),
    )
    missing = validate(
        {
            "source_annotation_id": "plate-ann-one",
            "ground_truth_text": "ABC123",
        },
        chars("ABC12"),
    )
    extra = validate(
        {
            "source_annotation_id": "plate-ann-one",
            "ground_truth_text": "ABC123",
        },
        chars("ABC1234"),
    )

    assert exact["exact_plate_match"] is True
    assert exact["box_count_error"] == 0
    assert exact["box_count_abs_error"] == 0

    assert missing["exact_plate_match"] is False
    assert missing["box_count_error"] == -1
    assert missing["box_count_abs_error"] == 1

    assert extra["exact_plate_match"] is False
    assert extra["box_count_error"] == 1
    assert extra["box_count_abs_error"] == 1
