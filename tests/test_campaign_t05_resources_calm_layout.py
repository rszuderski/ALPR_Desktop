import ast
from pathlib import Path

from auto_annotation_tool.gui import campaign_dashboard_ui as dashboard


def _t05_resources_source() -> str:
    source = Path(dashboard.__file__).read_text(encoding="utf-8-sig")
    module = ast.parse(source)
    outer = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_render_step1_route_actions"
    )
    open_resources = next(
        node for node in outer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_open_resources"
    )
    modal = next(
        node for node in ast.walk(open_resources)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_open_t06_resources_modal"
    )
    return ast.get_source_segment(source, modal) or ""


def test_t05_resources_uses_calm_material_layout():
    source = _t05_resources_source()
    assert 'geometry="1080x350"' in source
    assert "table.pack(fill=tk.X" in source
    assert "balance_lines = list" in source
    assert "fg=fg if is_total else muted" in source
    assert 'font=("Segoe UI", 8, "bold" if is_total else "normal")' in source
    assert "balance_fills =" not in source
    assert "segment_bg =" not in source


def test_t05_resources_has_single_primary_reuse_accent():
    source = _t05_resources_source()
    assert '"Użyj wcześniejszych anotacji "' in source
    assert "importable_count" in source
    assert "reuse_button = tk.Button(" in source
    assert "reuse_button.bind(" in source


def test_t05_resources_calm_ux_does_not_change_reuse_logic():
    source = _t05_resources_source()
    for token in (
        "list_project_az_import_sources",
        "summarize_project_az_resource",
        'state["mode"] = "hidden"',
        'state["mode"] = "browse"',
        'state["mode"] = "importable"',
        "importable_count > 0",
        "source_count = len(candidates)",
        "pending_review_count",
        "command=_open_t05_az_project_import_browser if action_enabled",
    ):
        assert token in source
