from pathlib import Path


def test_confirm_gold_only_protects_manual_z2_number():
    source = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z3_review_runtime.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def confirm_review_gold(")
    block = source[start:]

    assert 'inherited_from_z2 = bool(saved_text and number_source == "manual_z2")' in block
    assert 'local_z3_number = bool(saved_text and number_source == "manual_z3")' not in block
    assert "if not inherited_from_z2:" in block
    assert "save_plate_ground_truth(host, data, candidate_text, prepare=False)" in block


def test_review_quality_and_confirmation_share_same_z2_only_protection_contract():
    source = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z3_review_runtime.py"
    ).read_text(encoding="utf-8-sig")

    assert source.count('number_source == "manual_z2"') >= 2
    assert "if not protected_z2_gt:" in source
    assert "if not inherited_from_z2:" in source


def test_non_z2_fallback_mismatch_is_not_sent_to_number_mismatch_branch():
    source = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z3_review_runtime.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("if saved_text and candidate_text and candidate_text != saved_text:")
    end = source.index("\n    try:\n        resolved_status", start)
    block = source[start:end]

    update_pos = block.index("if not inherited_from_z2:")
    mismatch_pos = block.index('"reason": "number_mismatch"')
    assert update_pos < mismatch_pos
    assert "save_plate_ground_truth(host, data, candidate_text, prepare=False)" in block
