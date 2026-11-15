from __future__ import annotations

from types import SimpleNamespace

from auto_annotation_tool.gui import z3_goldpack_ui as gold


def _rec(ch="A"):
    return {"character": ch, "bbox": [0.0, 0.0, 10.0, 20.0]}


def _legacy_perfect():
    return {
        "status": "perfect",
        "characters": [_rec()],
    }


def _review(status, *, approved=False):
    data = {
        "status": "perfect",
        "ground_truth_text": "A",
        "characters": [_rec()],
        "review_state": {
            "schema": "alpr.pz2.review.v1",
            "status": status,
        },
        "gold_state": {
            "candidate": bool(approved),
            "approved": bool(approved),
        },
    }
    if status == "approved" and approved:
        data["review_state"]["approved_reference"] = (
            gold.z3_review_runtime.build_review_reference_snapshot(data)
        )
    return data


def test_gold_eligibility_keeps_legacy_compatibility_but_requires_new_approval():
    assert gold.is_gold_export_eligible_data(_legacy_perfect()) is True
    assert gold.is_gold_export_eligible_data(_review("in_progress", approved=False)) is False
    assert gold.is_gold_export_eligible_data(_review("approved", approved=False)) is False
    assert gold.is_gold_export_eligible_data(_review("approved", approved=True)) is True

    bad = _review("approved", approved=True)
    bad["status"] = "needs_fix"
    assert gold.is_gold_export_eligible_data(bad) is False

    excluded_legacy = _legacy_perfect()
    excluded_legacy["gold_state"] = {
        "candidate": False,
        "approved": True,
        "excluded": True,
        "excluded_reason": "unreadable",
    }
    assert gold.is_gold_export_eligible_data(excluded_legacy) is False

    excluded_approved = _review("approved", approved=True)
    excluded_approved["gold_state"]["excluded"] = True
    excluded_approved["gold_state"]["excluded_reason"] = "unreadable"
    assert gold.is_gold_export_eligible_data(excluded_approved) is False


def test_exportable_perfect_counter_uses_same_gold_contract():
    metadata = {
        "legacy": _legacy_perfect(),
        "pending": _review("in_progress", approved=False),
        "approved_without_gold": _review("approved", approved=False),
        "approved": _review("approved", approved=True),
    }
    host = SimpleNamespace(
        _count_exportable_characters_in_data=lambda data: len(data.get("characters", [])),
    )

    assert gold.count_exportable_perfect_plates_in_metadata(host, metadata) == 2


def test_contextual_gold_counts_use_same_gold_contract(monkeypatch):
    metadata = {
        "legacy": _legacy_perfect(),
        "pending": _review("in_progress", approved=False),
        "approved_without_gold": _review("approved", approved=False),
        "approved": _review("approved", approved=True),
    }

    monkeypatch.setattr(gold, "_get_contextual_gold_metadata", lambda host: metadata)

    host = SimpleNamespace(
        _empty_perfect_strategy_counts=lambda: {"other_perfect": 0},
        _empty_gold_source_counts=lambda: {"auto_preview": 0},
        _empty_plate_layout_counts=lambda: {"1R": 0},
        _get_perfect_strategy_bucket=lambda data: "other_perfect",
        _get_plate_source_bucket=lambda data: "auto_preview",
        _count_exportable_characters_in_data=lambda data: len(data.get("characters", [])),
        _increment_plate_layout_counts=lambda counts, data, amount=1: counts.__setitem__(
            "1R", int(counts.get("1R", 0)) + int(amount)
        ),
    )

    result = gold._compute_contextual_gold_export_counts(
        host,
        selected_strategies={"other_perfect"},
        selected_sources={"auto_preview"},
    )

    assert result["selected_plate_count"] == 2
    assert result["selected_char_count"] == 2
    assert result["strategy_counts"]["other_perfect"] == 2
    assert result["source_counts"]["auto_preview"] == 2


def test_source_metadata_does_not_mark_pending_review_as_gold_candidate():
    data = _review("in_progress", approved=False)
    data["gold_state"]["candidate"] = True

    host = SimpleNamespace(
        _normalize_plate_source_bucket=lambda bucket, data=None, meta_path=None: "auto_preview",
        _normalize_plate_source_origin=lambda origin, bucket=None: "pz2_detect",
        _normalize_character_source_kind=lambda rec, data=None, plate_source_bucket=None: "auto_preview",
    )

    changed = gold.ensure_plate_source_metadata(
        host,
        data,
        plate_id="p1",
        default_bucket="auto_preview",
        default_origin="pz2_detect",
        include_diagnostics=False,
    )

    assert changed is True
    assert data["gold_state"]["candidate"] is False
    assert data["gold_state"]["approved"] is False
