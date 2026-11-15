from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import campaign_graph_actions as actions


def test_continue_z3_propagates_false_mapping_to_graph_action():
    failure = {
        "ok": False,
        "reason": "approved_source_invalid",
        "message": "Brakuje zatwierdzonego obrazu: X.jpg",
    }
    host = SimpleNamespace(
        _step_continue_characters_from_ready_source=Mock(return_value=failure),
    )

    result = actions._execute_continue_z3(
        host,
        {
            "context": {
                "graph_edge_key": "e3_to_e4",
                "graph_gate_id": "T05",
                "target_substep": "pz3",
                "force_pz3": "1",
            }
        },
    )

    assert result.ok is False
    assert result.action == "continue_z3"
    assert "Brakuje zatwierdzonego obrazu" in result.message


def test_continue_z3_propagates_success_mapping_to_graph_action():
    success = {
        "ok": True,
        "reason": "current_pz2_checkpoint",
        "direct_pz3_checkpoint": True,
    }
    host = SimpleNamespace(
        _step_continue_characters_from_ready_source=Mock(return_value=success),
    )

    result = actions._execute_continue_z3(
        host,
        {
            "context": {
                "graph_edge_key": "e3_to_e4",
                "graph_gate_id": "T05",
                "target_substep": "pz3",
                "force_pz3": "1",
            }
        },
    )

    assert result.ok is True
    assert result.action == "continue_z3"


def test_navigation_source_returns_nested_z3_result():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (
        root / "auto_annotation_tool" / "gui" / "campaign_navigation.py"
    ).read_text(encoding="utf-8-sig")

    assert (
        "return self._step_goto_characters(preferred_source_context=source_context)"
        in source
    )

    failure_block = 'if not result.get("ok"):\n        try:\n            tab_char._campaign_pz2_sync_loading = False'
    failure_at = source.index(failure_block)
    return_at = source.index("return result", failure_at)
    latest_at = source.index("latest_xml =", failure_at)
    assert failure_at < return_at < latest_at

    marker = "# Z3NAVRET001: propagate real Z3 entry result"
    marker_at = source.index(marker)
    training_at = source.index("def _step_goto_training", marker_at)
    assert "return result" in source[marker_at:training_at]
