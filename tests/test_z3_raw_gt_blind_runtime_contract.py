from pathlib import Path


def _function_source(text: str, name: str) -> str:
    marker = f"def {name}("
    start = text.index(marker)
    next_def = text.find("\ndef ", start + len(marker))
    return text[start:] if next_def < 0 else text[start:next_def]


def test_run_fast_ocr_test_contains_no_gt_driven_raw_plumbing():
    root = Path(__file__).resolve().parents[1]
    runtime = (
        root / "auto_annotation_tool/gui/z3_detection_runtime.py"
    ).read_text(encoding="utf-8-sig")
    body = _function_source(runtime, "run_fast_ocr_test")

    assert "true_texts" not in body
    assert "expected_char_count" not in body
    assert "_pick_best_true_text" not in body
    assert "_best_text_distance" not in body
    assert "_apply_final_truth_count_guard" not in body
    assert "gt_count_guard" not in body
    assert "YOLO_BOX_RECALL_CONFIDENCE" not in body

    assert "detector.expected_character_count = 0" in body
    assert "[],  # RAW contract: no reference texts" in body


def test_raw_snapshot_precedes_evaluation_and_gt_assist():
    root = Path(__file__).resolve().parents[1]
    runtime = (
        root / "auto_annotation_tool/gui/z3_detection_runtime.py"
    ).read_text(encoding="utf-8-sig")
    body = _function_source(runtime, "run_fast_ocr_test")

    raw_idx = body.index('local_meta[pid]["raw_detection"] = raw_detection')
    validation_idx = body.index("self._build_raw_detection_validation(")
    assist_idx = body.index("self._build_gt_assist_suggestion(")

    assert raw_idx < validation_idx
    assert raw_idx < assist_idx
    assert '"contract": "gt_blind.v1"' in body


def test_gt_dependent_tools_remain_in_explicit_assist_layer():
    root = Path(__file__).resolve().parents[1]
    algorithms = (
        root / "auto_annotation_tool/gui/z3_detection_algorithms.py"
    ).read_text(encoding="utf-8-sig")
    assist = (
        root / "auto_annotation_tool/gui/z3_gt_assist_runtime.py"
    ).read_text(encoding="utf-8-sig")

    assert "def fit_detection_count_to_truths(" in algorithms
    assert "def apply_final_truth_count_guard(" in algorithms
    assert "host._fit_detection_count_to_truths(" in assist
    assert 'GT_ASSIST_SOURCE = "gt_assisted"' in assist
