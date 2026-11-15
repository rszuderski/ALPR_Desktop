from pathlib import Path


def _source(path: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / path).read_text(encoding="utf-8-sig")


def _function_source(text: str, name: str) -> str:
    marker = f"def {name}("
    start = text.index(marker)
    next_def = text.find("\ndef ", start + len(marker))
    return text[start:] if next_def < 0 else text[start:next_def]


def test_main_detection_freezes_raw_then_prepares_working_annotation():
    runtime = _source("auto_annotation_tool/gui/z3_detection_runtime.py")
    stage = _function_source(runtime, "run_detection_stage")

    assert "_prompt_pz2_detection_guard_options" not in stage
    assert '"raw_only": True' in stage
    assert '"process_scope": "all"' in stage
    assert '"prepare_working": True' in stage
    assert '"campaign"' in stage
    assert '"free"' in stage


def test_raw_experiment_exits_before_review_gold_mutation():
    runtime = _source("auto_annotation_tool/gui/z3_detection_runtime.py")
    body = _function_source(runtime, "run_fast_ocr_test")

    raw_store = body.index('local_meta[pid]["raw_detection"] = raw_detection')
    raw_branch = body.index("if raw_only:", raw_store)
    raw_continue = body.index("continue", raw_branch)
    review_write = body.index('local_meta[pid]["characters"] = final_chars')

    assert raw_store < raw_branch < raw_continue < review_write
    # Sprawdzamy semantykę kontraktu, nie techniczne napisy UI.
    assert 'local_meta[pid]["raw_detection"] = raw_detection' in body
    assert 'local_meta[pid]["characters"] = final_chars' in body
    assert "if raw_only:" in body
    assert "Twoje poprawki i zatwierdzenia pozostają bez zmian" in body


def test_raw_mode_does_not_unlock_dataset_or_use_review_scope():
    runtime = _source("auto_annotation_tool/gui/z3_detection_runtime.py")
    body = _function_source(runtime, "run_fast_ocr_test")

    assert 'process_scope = (' in body
    assert '"all"' in body
    assert "elif not raw_only:" in body
    assert "self.unlock_dataset_subtab()" in body
    assert '"execution_mode": (' in body
    assert '"annotation"' in body
    assert "self._sync_step3_access_from_preview_state(self.preview_metadata)" in body
    assert '"workflow_context": workflow_context' in body


def test_z3_main_button_uses_plain_language_but_keeps_raw_runtime_contract():
    ui = _source("auto_annotation_tool/gui/z3_detection_tab_ui.py")
    runtime = _source("auto_annotation_tool/gui/z3_detection_runtime.py")

    assert 'text="Uruchom wykrywanie"' in ui
    assert 'text="Uruchom RAW"' not in ui

    stage = _function_source(runtime, "run_detection_stage")
    assert '"raw_only": True' in stage
    assert '"process_scope": "all"' in stage
    assert '"prepare_working": True' in stage
