import ast
import re
from pathlib import Path

from auto_annotation_tool.gui import campaign_dashboard_ui as dashboard


def _visible_t05_source() -> str:
    source = Path(dashboard.__file__).read_text(encoding="utf-8-sig")
    module = ast.parse(source)
    outer = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_render_step1_route_actions"
    )
    nested = next(
        node for node in outer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_open_t06_actions_modal"
    )
    return ast.get_source_segment(source, nested) or ""


def test_visible_t05_uses_compact_work_gate_typography():
    source = _visible_t05_source()

    sizes = [
        int(match.group(1))
        for match in re.finditer(
            r'font=\("Segoe UI(?: Semibold)?",\s*(\d+)',
            source,
        )
    ]

    assert sizes == [11, 9, 10, 10, 9, 10, 9, 9, 9, 10, 10]
    assert max(sizes) <= 11
    assert min(sizes) >= 9


def test_visible_t05_keeps_existing_modal_geometry():
    source = _visible_t05_source()

    assert "modal_width = 940" in source
    assert "modal_height = 620" in source
