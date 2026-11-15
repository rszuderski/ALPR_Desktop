import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "auto_annotation_tool" / "gui" / "z3_detection_tab_ui.py"


def _source():
    return SOURCE.read_text(encoding="utf-8-sig")


def test_project_material_ctas_use_plain_language_and_compact_padding():
    source = _source()

    assert 'text="Dodaj tablice lub zdjęcia…"' in source
    assert 'text="Eksportuj wycięcia tablic…"' in source
    assert 'text="Dodaj materiał…"' not in source
    assert 'text="Dodaj materiał do zbioru…"' not in source
    assert 'text="Eksportuj wycięcia…"' not in source
    assert 'text="Eksportuj cropy + AZ…"' not in source

    assert (
        'self.preview_add_material_btn.grid('
        'row=0, column=0, columnspan=2, sticky="ew")'
        in source
    )
    assert "self.preview_export_material_btn.grid(" in source
    assert "row=1," in source
    assert "columnspan=2," in source
    assert 'sticky="ew",' in source
    tree = ast.parse(source)
    focus_grid = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "grid"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "preview_import_focus_btn"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "self"
        ):
            continue
        focus_grid = node
        break

    assert focus_grid is not None
    kwargs = {kw.arg: kw.value for kw in focus_grid.keywords if kw.arg}
    assert isinstance(kwargs.get("row"), ast.Constant)
    assert kwargs["row"].value == 2
    assert isinstance(kwargs.get("column"), ast.Constant)
    assert kwargs["column"].value == 0
def test_import_focus_button_is_compact_too():
    source = _source()
    start = source.index('text="Pokaż tylko dodane"')
    block = source[start:start + 500]
    assert "preview_import_focus_btn.configure(padding=(5, 2))" in block
    assert 'pady=(3, 0)' in block
