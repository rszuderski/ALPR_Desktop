from types import SimpleNamespace

from auto_annotation_tool.gui import z3_preview_metadata_runtime
from auto_annotation_tool.registry.pz2_az_reuse import (
    pz2_az_reuse_protection_reason,
)


def _revision():
    return {
        "az_revision_id": "AZR-IMPORT",
        "effective_status": "imported_pending_review",
    }


def _reuse_state():
    return {
        "az_revision_id": "AZR-IMPORT",
        "effective_status": "imported_pending_review",
        "requires_review": True,
    }


def test_stale_same_revision_pending_materialization_is_reapplied():
    data = {
        "status": "perfect",
        "fusion_strategy": "az_reuse",
        "gold_state": {
            "approved": False,
            "candidate": True,
            "excluded": False,
        },
        "az_reuse": _reuse_state(),
    }

    assert pz2_az_reuse_protection_reason(data, _revision()) == ""


def test_correct_pending_materialization_is_already_current():
    data = {
        "status": "needs_fix",
        "fusion_strategy": "az_reuse",
        "gold_state": {
            "approved": False,
            "candidate": False,
            "excluded": False,
        },
        "review_state": {
            "schema": "alpr.pz2.review.v1",
            "status": "in_progress",
            "source": "az_project_import",
            "human_edited": False,
            "approved_at": None,
        },
        "az_reuse": _reuse_state(),
    }

    assert (
        pz2_az_reuse_protection_reason(data, _revision())
        == "already_current"
    )


def test_same_revision_pending_preserves_human_edit():
    data = {
        "status": "needs_fix",
        "fusion_strategy": "az_reuse",
        "gold_state": {
            "approved": False,
            "candidate": False,
            "excluded": False,
        },
        "review_state": {
            "schema": "alpr.pz2.review.v1",
            "status": "in_progress",
            "source": "az_project_import",
            "human_edited": True,
        },
        "az_reuse": _reuse_state(),
    }

    assert (
        pz2_az_reuse_protection_reason(data, _revision())
        == "human_review_edit"
    )


def test_in_progress_review_blocks_generic_perfect_recalculation():
    host = SimpleNamespace(
        _derive_preview_status_from_characters=lambda chars: "perfect",
        _review_approval_is_current=lambda data: False,
    )
    data = {
        "status": "needs_fix",
        "review_state": {
            "schema": "alpr.pz2.review.v1",
            "status": "in_progress",
            "source": "az_project_import",
            "human_edited": False,
        },
    }

    assert (
        z3_preview_metadata_runtime._derive_preview_status_from_data(
            host,
            data,
            [{"character": "A", "bbox": [0, 0, 1, 1]}],
        )
        == "needs_fix"
    )
