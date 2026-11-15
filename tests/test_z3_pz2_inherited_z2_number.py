from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EDITOR = ROOT / "auto_annotation_tool" / "gui" / "z3_preview_editor_runtime.py"
GT = ROOT / "auto_annotation_tool" / "gui" / "z3_plate_gt_runtime.py"


def test_editor_modes_warn_about_number_inherited_from_z2():
    source = EDITOR.read_text(encoding="utf-8-sig")
    assert "Numer zapisany wcześniej w Z2" in source
    assert "Edycja boxów i znaków nie zmienia tego numeru" in source
    assert "Jeśli numer z Z2 jest błędny, użyj „Zmień numer”" in source


def test_manual_change_of_number_from_z2_has_extra_confirmation():
    source = GT.read_text(encoding="utf-8-sig")
    assert "Numer odziedziczony z Z2" in source
    assert "Zmienić numer odziedziczony z Z2?" in source
    assert "Ta operacja zapisze nową wersję numeru w PZ2" in source
