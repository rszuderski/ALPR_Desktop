import ast
from pathlib import Path

from auto_annotation_tool.gui import campaign_dashboard_ui as dashboard


def _src() -> str:
    source = Path(dashboard.__file__).read_text(encoding="utf-8-sig")
    module = ast.parse(source)
    outer = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_render_step1_route_actions")
    open_resources = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == "_open_resources")
    modal = next(n for n in ast.walk(open_resources) if isinstance(n, ast.FunctionDef) and n.name == "_open_t06_resources_modal")
    return ast.get_source_segment(source, modal) or ""


def test_pending_import_overrides_pz2_copy_before_rows_are_built():
    source = _src()
    assert "pending_import_review_count = int(" in source
    assert 'az_reuse_ui.get("pending_review_count", 0)' in source
    assert "Zaimportowane anotacje ({pending_import_review_count})" in source
    assert "czekają na sprawdzenie." in source
    assert '"Przejdź do pracy T05"' in source
    assert source.index("pending_import_review_count = int(") < source.index("rows = [")


def test_pending_import_keeps_reuse_browser_available():
    source = _src()
    assert 'state["mode"] = "browse"' in source
    assert 'action_text = "Wcześniejsze anotacje…"' in source
    assert "command=_open_t05_az_project_import_browser if action_enabled" in source


def test_no_material_fallback_still_exists():
    source = _src()
    assert "Brak gotowych anotacji znaków." in source
