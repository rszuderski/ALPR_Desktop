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
        node for node in open_resources.body
        if isinstance(node, ast.If)
        for child in node.body
        if isinstance(child, ast.FunctionDef)
        and child.name == "_open_t06_resources_modal"
        for modal in [child]
    )
    return ast.get_source_segment(source, modal) or ""


def test_t05_resources_exposes_cross_project_az_import():
    source = _t05_resources_source()

    assert "def _open_t05_az_project_import_browser" in source
    assert "list_project_az_import_sources" in source
    assert "import_project_az_bindings" in source
    assert '"Użyj wcześniejszych anotacji "' in source
    assert "command=_open_t05_az_project_import_browser if action_enabled" in source
    assert "program rozpoznał" in source
    assert "dokładnie te same w obu projektach" in source


def test_t05_az_import_preserves_target_review_contract():
    source = _t05_resources_source()

    assert "wymagały lokalnej kontroli w projekcie docelowym" in source
    assert "Status OK z projektu źródłowego nie zostanie przeniesiony" in source
    assert "Istniejące konflikty nie zostaną nadpisane" in source
    assert "Do lokalnej kontroli w PZ2" in source
    assert "imported_pending_review" not in source
    assert "logical crop_id" not in source
    assert "GOLD" not in source


def test_t05_az_browser_shows_required_import_diagnostics():
    source = _t05_resources_source()

    for label in (
        "AZ źródła",
        "Do importu",
        "Już przypięte",
        "Konflikty",
        "Brak cropa targetu",
    ):
        assert label in source
