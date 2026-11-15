from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import campaign_graph_actions as actions


def test_t05_continue_z3_forces_pz3():
    receiver = Mock(return_value={"ok": True})
    host = SimpleNamespace(
        _step_continue_characters_from_ready_source=receiver,
    )

    result = actions._execute_continue_z3(
        host,
        {
            "context": {
                "graph_edge_key": "e3_to_e4",
                "graph_gate_id": "T05",
            }
        },
    )

    assert result.ok
    receiver.assert_called_once()

    context = receiver.call_args.args[0]
    assert context["graph_gate_id"] == "T05"
    assert context["graph_edge_key"] == "e3_to_e4"
    assert context["target_substep"] == "pz3"
    assert context["force_pz3"] == "1"
    assert "force_pz2" not in context


def test_step2_continue_z3_does_not_force_pz3():
    receiver = Mock(return_value={"ok": True})
    host = SimpleNamespace(
        _step_continue_characters_from_ready_source=receiver,
    )

    result = actions._execute_continue_z3(
        host,
        {
            "context": {
                "graph_edge_key": "e2_to_e3",
                "graph_gate_id": "T03",
            }
        },
    )

    assert result.ok
    receiver.assert_called_once()

    context = receiver.call_args.args[0]
    assert context["graph_gate_id"] == "T03"
    assert context["graph_edge_key"] == "e2_to_e3"
    assert "target_substep" not in context
    assert "force_pz3" not in context


def test_navigation_accepts_current_t05_and_legacy_t06():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "auto_annotation_tool"
        / "gui"
        / "campaign_navigation.py"
    ).read_text(encoding="utf-8-sig")

    assert 'if gate_hint in {"T05", "T06"}:' in source
    assert 'if gate_hint == "T06":' not in source
