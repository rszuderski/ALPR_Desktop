import ast
from pathlib import Path

from auto_annotation_tool.gui import z3_shared_ui


def test_pz3_dataset_mode_refresh_also_refreshes_status_panel():
    source = Path(z3_shared_ui.__file__).read_text(encoding="utf-8-sig")
    module = ast.parse(source)
    fn = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "refresh_pz3_dataset_mode_ui"
    )

    calls = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            calls.append(func.id)
        elif isinstance(func, ast.Attribute):
            calls.append(func.attr)

    # Zmiana źródła/preview/filtrów odświeża lewy panel przez
    # refresh_pz3_dataset_mode_ui; ten sam refresh musi odświeżyć
    # prawy Status PZ3, żeby nie został w stanie sprzed wczytania PZ2.
    assert "_refresh_pz3_status_panel_ui" in source[source.find("def refresh_pz3_dataset_mode_ui"):source.find("def refresh_pz3_dataset_card")]
    assert calls.count("callable") >= 2
