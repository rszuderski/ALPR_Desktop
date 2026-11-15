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


def test_t05_reuse_uses_single_cta_with_count():
    source = _t05_resources_source()
    assert 'geometry="1080x350"' in source
    assert '"Użyj wcześniejszych anotacji "' in source
    assert "importable_count" in source
    assert "Dostępne wcześniejsze anotacje:" not in source


def test_t05_single_cta_keeps_hover_and_backend_contract():
    source = _t05_resources_source()
    for token in (
        "reuse_button.pack(side=tk.RIGHT)",
        "reuse_button.bind(",
        '"<Enter>"',
        '"<Leave>"',
        "list_project_az_import_sources",
        "summarize_project_az_resource",
        "command=_open_t05_az_project_import_browser if action_enabled",
    ):
        assert token in source
