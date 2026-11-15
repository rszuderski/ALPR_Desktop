import ast
from pathlib import Path

from auto_annotation_tool.gui import campaign_dashboard_ui as dashboard


def _t05_resources_source() -> str:
    source = Path(dashboard.__file__).read_text(encoding="utf-8-sig")
    module = ast.parse(source)
    outer = next(
        n for n in module.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_render_step1_route_actions"
    )
    open_resources = next(
        n for n in outer.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_open_resources"
    )
    modal = next(
        n for n in ast.walk(open_resources)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_open_t06_resources_modal"
    )
    return ast.get_source_segment(source, modal) or ""


def test_pending_review_does_not_disable_reuse_browser():
    source = _t05_resources_source()
    assert "source_count = len(candidates)" in source
    assert 'state["mode"] = "browse"' in source
    assert 'action_text = "Wcześniejsze anotacje…"' in source
    assert "action_enabled = True" in source


def test_pending_review_is_material_state_not_browser_lock():
    source = _t05_resources_source()
    assert "wymagają lokalnej kontroli w PZ2" in source
    assert "pending_review_count" in source


def test_buttons_have_final_spacing_and_close_style():
    source = _t05_resources_source()
    assert "padx=14" in source
    assert "pady=7" in source
    assert "close_button = tk.Button(" in source
    assert "close_bg = blend_hex_colors(body_bg, accent, 0.06)" in source
    assert "close_button.bind(" in source
