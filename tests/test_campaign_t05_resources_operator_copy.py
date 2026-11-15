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


def test_t05_resources_uses_operator_language():
    source = _t05_resources_source()

    for text in (
        "Stan pracy",
        "Status: {operator_status_text}",
        "Warunki jeszcze niespełnione",
        "Potrzebny materiał",
        "Stan materiału",
        "Co dalej",
        '"Anotacje znaków"',
        '"Dataset znaków"',
        "Brak gotowych anotacji znaków",
        "Dataset nie został jeszcze utworzony",
        "Użyj wcześniejszych anotacji ",
        "Użyj wcześniejszych anotacji",
    ):
        assert text in source


def test_t05_resources_hides_old_technical_copy():
    source = _t05_resources_source()

    for text in (
        "Status bramki:",
        "Przyrost / dziedziczone",
        "Czeka na kontrakt PZ2",
        "Kontrakt PZ2 spełniony",
        "Minimum PZ2 jest spełnione",
        "AZ z innego projektu",
        "Masz już anotacje w innym projekcie?",
    ):
        assert text not in source
