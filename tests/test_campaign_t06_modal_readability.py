import ast
from pathlib import Path

from auto_annotation_tool.gui import campaign_dashboard_ui as dashboard


def _nested_function(source: str, name: str) -> ast.FunctionDef:
    module = ast.parse(source)
    renderer = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_render_step1_route_actions"
    )
    return next(
        node for node in renderer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    )


def _font_sizes(node: ast.AST) -> list[int]:
    result = []
    for item in ast.walk(node):
        if not isinstance(item, ast.keyword) or item.arg != "font":
            continue
        value = item.value
        if not isinstance(value, (ast.Tuple, ast.List)) or len(value.elts) < 2:
            continue
        size = value.elts[1]
        if isinstance(size, ast.Constant) and isinstance(size.value, int):
            result.append(size.value)
    return result


def test_visible_t06_uses_t03_typography_scale():
    source = Path(dashboard.__file__).read_text(encoding="utf-8-sig")
    t03_shell = _nested_function(source, "_open_small_modal")
    t06 = _nested_function(source, "_open_t07_actions_modal")

    t03_segment = ast.get_source_segment(source, t03_shell) or ""
    t06_segment = ast.get_source_segment(source, t06) or ""

    # T03: nagłówek 11 pt; zwykły tekst/przyciski korzystają z aplikacyjnego 10 pt;
    # drobny nagłówek notice ma 9 pt.
    assert 'font=("Segoe UI", 11, "bold")' in t03_segment
    assert 'font=("Segoe UI", 9, "bold")' in t03_segment

    # Widoczna Praca T06 ma korzystać z tej samej skali typograficznej.
    assert 'font=("Segoe UI", 11, "bold")' in t06_segment
    assert 'font=("Segoe UI", 10)' in t06_segment
    assert 'font=("Segoe UI", 10, "bold")' in t06_segment
    assert 'font=("Segoe UI Semibold", 9)' in t06_segment

    sizes = _font_sizes(t06)
    assert sizes
    assert min(sizes) >= 9
    assert max(sizes) <= 11
