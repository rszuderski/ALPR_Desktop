import ast
from pathlib import Path

from auto_annotation_tool.gui import z4_dataset_panels


def _splitter_ui_source() -> str:
    path = Path(z4_dataset_panels.__file__)
    source = path.read_text(encoding="utf-8-sig")
    module = ast.parse(source)
    fn = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_splitter_ui"
    )
    return ast.get_source_segment(source, fn) or ""


def test_technical_source_details_use_structured_colored_table():
    source = _splitter_ui_source()

    assert 'text="Szczegóły techniczne źródła"' in source
    assert '("Pole", "Wartość")' in source
    assert "table.grid_columnconfigure(0" in source
    assert "table.grid_columnconfigure(1" in source
    assert "Powtórny eksport — ten sam materiał logiczny." in source
    assert '"Równoważne eksporty"' in source
    assert '"materiału logicznego"' in source
    assert '"kontrakt"' in source
    assert '"fingerprint"' in source
    assert "wraplength=625" in source
    assert "dialog.grab_set()" in source


def test_technical_details_keep_raw_dialog_only_as_fallback():
    source = _splitter_ui_source()

    start = source.index("def _show_split_source_technical_details")
    end = source.index("self.split_source_technical_row", start)
    segment = source[start:end]

    assert "tk.Toplevel" in segment
    assert "self.app.themed_info" in segment
    assert segment.index("tk.Toplevel") < segment.rindex("self.app.themed_info")
