import copy

from auto_annotation_tool.gui import z3_gt_assist_runtime as runtime


def chars(text):
    return [
        {
            "character": char,
            "bbox": [idx * 10, 0, (idx * 10) + 8, 20],
            "confidence": 0.9,
            "method": "yolo",
        }
        for idx, char in enumerate(text)
    ]


class Host:
    def _characters_to_text(self, records, data=None):
        return "".join(
            str(record.get("character", "") or "")
            for record in records
        )

    def _get_preview_ground_truth_text(self, data=None):
        return str((data or {}).get("ground_truth_text", "") or "")

    def _serialize_character_records(
        self,
        records,
        fusion_strategy="",
        fusion_details=None,
        data=None,
    ):
        return copy.deepcopy(list(records or []))

    def _fit_detection_count_to_truths(self, records, truths):
        records = list(records or [])
        truth = str(list(truths or [""])[0] or "")
        target = len(truth)
        if len(records) <= target:
            return records, None

        best = None
        for start in range(len(records) - target + 1):
            window = records[start:start + target]
            text = self._characters_to_text(window)
            score = sum(
                1
                for left, right in zip(text, truth)
                if left != right
            )
            if best is None or score < best[0]:
                best = (score, start, window)

        _, start, window = best
        return list(window), {
            "original_box_count": len(records),
            "trimmed_box_count": len(window),
            "trimmed_extra_boxes": len(records) - len(window),
            "trimmed_left_boxes": start,
            "trimmed_right_boxes": len(records) - start - len(window),
        }

    def _repair_ocr_with_yolo_boxes(
        self,
        current,
        yolo,
        truths,
        max_mismatch_count=999,
    ):
        truth = str(list(truths or [""])[0] or "")
        current = copy.deepcopy(list(current or []))
        yolo = list(yolo or [])
        if len(current) != len(truth):
            return None, None

        changed = []
        for index, expected in enumerate(truth):
            current_char = str(current[index].get("character", "") or "")
            if current_char == expected:
                continue
            candidates = [
                rec
                for rec in yolo
                if str(rec.get("character", "") or "") == expected
                and abs(
                    float(rec["bbox"][0])
                    - float(current[index]["bbox"][0])
                ) <= 2.0
            ]
            if not candidates:
                return None, None
            current[index] = copy.deepcopy(candidates[0])
            current[index]["character"] = expected
            changed.append(index)

        if self._characters_to_text(current) != truth:
            return None, None
        return current, {
            "mismatch_positions": changed,
            "expected_text": truth,
        }

    def _derive_preview_status_from_data(self, data, records):
        return (
            "perfect"
            if self._characters_to_text(records)
            == str(data.get("ground_truth_text") or "")
            else "needs_fix"
        )


def test_gt_assist_does_not_restore_nms_candidates_rejected_by_geometry_guards():
    data = metadata("XX", "AB", yolo_text="AB")
    data["yolo_box_detections"] = []
    before = copy.deepcopy(data)
    result = runtime.build_gt_assist_suggestion(Host(), data)
    assert result["status"] == "no_suggestion"
    assert not result["operations"]
    assert data == before


def metadata(raw_text, gt, *, yolo_text=None):
    return {
        "source_annotation_id": "plate-ann-one",
        "source_gt_hash": "gt-hash-one",
        "ground_truth_text": gt,
        "characters": chars(raw_text),
        "raw_detection": {
            "schema": "alpr.pz2.raw_detection.v1",
            "contract": "gt_blind.v1",
            "prediction_text": raw_text,
            "characters": chars(raw_text),
        },
        "yolo_detections": chars(
            yolo_text if yolo_text is not None else raw_text
        ),
    }


def test_assist_does_not_mutate_raw_detection():
    host = Host()
    data = metadata("WI9O5PW", "WI905PW", yolo_text="WI905PW")
    raw_before = copy.deepcopy(data["raw_detection"])

    suggestion = runtime.store_gt_assist_suggestion(host, data)

    assert suggestion["status"] == "suggested"
    assert suggestion["suggestion_text"] == "WI905PW"
    assert suggestion["exact_text_match"] is True
    assert data["raw_detection"] == raw_before
    assert data["characters"][3]["character"] == "O"


def test_assist_can_trim_extra_boxes_without_touching_raw():
    host = Host()
    data = metadata("XABC123", "ABC123", yolo_text="XABC123")
    raw_before = copy.deepcopy(data["raw_detection"])

    suggestion = runtime.build_gt_assist_suggestion(host, data)

    assert suggestion["status"] == "suggested"
    assert suggestion["suggestion_text"] == "ABC123"
    assert any(
        op["type"] == "trim_extra_boxes"
        for op in suggestion["operations"]
    )
    assert data["raw_detection"] == raw_before


def test_accept_marks_review_but_preserves_raw():
    host = Host()
    data = metadata("WI9O5PW", "WI905PW", yolo_text="WI905PW")
    raw_before = copy.deepcopy(data["raw_detection"])

    runtime.store_gt_assist_suggestion(host, data)
    result = runtime.accept_gt_assist_suggestion(host, data)

    assert result["ok"] is True
    assert result["text"] == "WI905PW"
    assert data["correction_source"] == "gt_assist_confirmed"
    assert data["review_source"] == "gt_assist_confirmed"
    assert data["status"] == "perfect"
    assert data["raw_detection"] == raw_before
    assert data["gt_assist"]["status"] == "accepted"


def test_accept_rejects_stale_raw_suggestion():
    host = Host()
    data = metadata("WI9O5PW", "WI905PW", yolo_text="WI905PW")

    runtime.store_gt_assist_suggestion(host, data)
    data["raw_detection"]["prediction_text"] = "CHANGED"

    result = runtime.accept_gt_assist_suggestion(host, data)

    assert result["ok"] is False
    assert result["reason"] == "stale_raw_detection"
    assert data["characters"][3]["character"] == "O"


def test_accept_rejects_stale_ground_truth():
    host = Host()
    data = metadata("WI9O5PW", "WI905PW", yolo_text="WI905PW")

    runtime.store_gt_assist_suggestion(host, data)
    data["ground_truth_text"] = "WI905PX"

    result = runtime.accept_gt_assist_suggestion(host, data)

    assert result["ok"] is False
    assert result["reason"] == "stale_ground_truth"


def test_raw_exact_produces_not_needed():
    host = Host()
    data = metadata("ABC123", "ABC123")

    suggestion = runtime.build_gt_assist_suggestion(host, data)

    assert suggestion["status"] == "not_needed"
    assert suggestion["reason"] == "raw_exact"
    assert suggestion["operations"] == []

def test_assist_snapshots_gt_revision_set():
    host = Host()
    data = metadata(
        "WI9O5PW",
        "WI905PW",
        yolo_text="WI905PW",
    )
    data["source_gt_revision_ids"] = ["rev-b", "rev-a"]

    suggestion = runtime.store_gt_assist_suggestion(host, data)

    assert suggestion["source_gt_revision_id"] is None
    assert suggestion["source_gt_revision_ids"] == [
        "rev-a",
        "rev-b",
    ]
    assert len(suggestion["source_raw_result_hash"]) == 64


def test_accept_rejects_stale_gt_revision_even_when_text_and_hash_match():
    host = Host()
    data = metadata(
        "WI9O5PW",
        "WI905PW",
        yolo_text="WI905PW",
    )
    data["source_gt_revision_ids"] = ["rev-one"]

    runtime.store_gt_assist_suggestion(host, data)
    data["source_gt_revision_ids"] = ["rev-two"]

    result = runtime.accept_gt_assist_suggestion(host, data)

    assert result["ok"] is False
    assert result["reason"] == "stale_gt_revision"
