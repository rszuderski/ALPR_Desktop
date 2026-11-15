from pathlib import Path

from auto_annotation_tool.gui import campaign_dashboard_ui as dashboard
from auto_annotation_tool.gui import campaign_ui_helpers


def _source() -> str:
    return Path(dashboard.__file__).read_text(encoding="utf-8-sig")


def test_campaign_workflow_semantic_tone_contract():
    assert campaign_ui_helpers.campaign_workflow_semantic_tone("done") == "success"
    assert campaign_ui_helpers.campaign_workflow_semantic_tone("current") == "warning"
    assert campaign_ui_helpers.campaign_workflow_semantic_tone("attention") == "warning"
    assert campaign_ui_helpers.campaign_workflow_semantic_tone("blocked") == "muted"
    assert campaign_ui_helpers.campaign_workflow_semantic_tone("error") == "error"
    assert campaign_ui_helpers.campaign_workflow_semantic_tone("neutral") == "info"


def test_current_and_recommended_actions_are_not_green():
    source = _source()
    assert '"tone": success if is_recommended else muted,' not in source
    assert "fg=success if recommended_label else muted," not in source
    assert (
        'bg=blend_hex_colors(body_bg, success, 0.09) '
        'if recommended_label else body_bg,'
    ) not in source
    assert '"tone": warning if is_recommended else muted,' in source


def test_completion_badges_follow_shared_semantics():
    source = _source()
    assert 'DO WYKONANIA IT{current_iter}", warning' in source
    assert 'DO WYKONANIA IT{current_iter}", muted' not in source
    assert 'PRZERWANE IT{current_iter}", warning' in source
    assert 'PRZERWANE IT{current_iter}", error' not in source
    assert 'WYKONANE IT{current_iter}", success' in source
    assert 'NIEDOSTĘPNE IT{current_iter}", muted' in source


def test_graph_current_actions_use_warning_not_success():
    source = _source()
    assert "row_fill = blend_hex_colors(fill, graph_card_warning, 0.16)" in source
    assert 'copy.tone == "success" and not (is_work_row or is_approve_row)' in source
    assert "primary_color = graph_card_warning" in source


def test_t05_t06_row_uses_done_current_blocked_semantics():
    source = _source()
    assert "action_done = bool(" in source
    assert "campaign_workflow_semantic_color(" in source



def _nested_function_source(function_name: str) -> str:
    source = _source()
    ast_mod = __import__("ast")
    tree = ast_mod.parse(source)
    node = next(
        item for item in ast_mod.walk(tree)
        if isinstance(item, ast_mod.FunctionDef) and item.name == function_name
    )
    return ast_mod.get_source_segment(source, node) or ""


def _literal_return_statuses(function_name: str):
    source = _source()
    ast_mod = __import__("ast")
    tree = ast_mod.parse(source)
    fn = next(
        item for item in ast_mod.walk(tree)
        if isinstance(item, ast_mod.FunctionDef) and item.name == function_name
    )
    rows = []
    for ret in ast_mod.walk(fn):
        if not isinstance(ret, ast_mod.Return) or not isinstance(ret.value, ast_mod.Dict):
            continue
        fields = {}
        for key, value in zip(ret.value.keys, ret.value.values):
            if isinstance(key, ast_mod.Constant) and isinstance(key.value, str):
                fields[key.value] = value
        title_node = fields.get("title")
        title = title_node.value if isinstance(title_node, ast_mod.Constant) else ""
        tone = ast_mod.get_source_segment(source, fields.get("tone")) if fields.get("tone") else ""
        mark_tone = ast_mod.get_source_segment(source, fields.get("mark_tone")) if fields.get("mark_tone") else ""
        if title:
            rows.append((title, tone or "", mark_tone or ""))
    return rows


def test_t06_status_producer_reserves_success_for_completed_work():
    rows = _literal_return_statuses("_t06_action_status")
    offenders = [
        (title, tone)
        for title, tone, _mark in rows
        if tone.strip() == "success" and not title.startswith("WYKONANE:")
    ]
    assert offenders == []


def test_t06_current_steps_are_orange_at_the_status_source():
    rows = dict((title, tone) for title, tone, _mark in _literal_return_statuses("_t06_action_status"))
    assert rows["KROK 1: ANOTACJE ZNAKÓW"] == "warning"
    assert rows["KROK 2: DATASET ZNAKÓW"] == "warning"
    assert rows["CEL: UŻYJ [OK]"] == "warning"


def test_interrupted_attention_marks_are_not_red_errors():
    tokens = ("PRZERWANE", "WZNÓW", "DO ROZLICZENIA", "ROZLICZ [OK]", "NIEDOMKNIĘT")
    offenders = []
    for function_name in ("_t06_action_status", "_t07_action_status"):
        for title, _tone, mark_tone in _literal_return_statuses(function_name):
            if any(token in title for token in tokens) and mark_tone.strip() == "error":
                offenders.append((function_name, title))
    assert offenders == []



def test_t05_graph_ready_export_is_done_even_with_optional_pz2_draft():
    compact = _nested_function_source("_edge_t06_work_compact_status")
    assert 'pending_t06.get("valid_export_exists")' in compact
    assert 'return "WYKONANE"' in compact
    assert 'current_work_status or "WYKONANE"' in compact


def test_t05_graph_does_not_render_ready_export_draft_as_interruption():
    draw_gate = _nested_function_source("_draw_gate")
    assert 'gate_t06_state.get("valid_export_exists")' in draw_gate
    assert "and not bool(gate_t06_state.get" in draw_gate
    assert "t06_work_done = bool(" in draw_gate
    assert "row_fill = blend_hex_colors(fill, graph_card_success, 0.10)" in draw_gate
    assert "primary_color = graph_card_success" in draw_gate


def test_t05_graph_completed_work_copy_is_green_done_state():
    from auto_annotation_tool.gui.campaign_gate_fields import gate_field_copy

    copy = gate_field_copy("PRACA", "WYKONANE", enabled=True)

    assert copy.caption == "Praca"
    assert copy.primary == "Praca wykonana"
    assert copy.detail == "Dataset znaków jest gotowy"
    assert copy.tone == "success"
