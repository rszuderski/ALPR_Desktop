import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative):
    return (ROOT / relative).read_text(encoding="utf-8-sig")


def test_sort_bar_is_one_compact_cta_not_two_row_button_grid():
    source = _read("auto_annotation_tool/gui/z3_detection_tab_ui.py")

    assert "self.preview_sort_cta = tk.Button(" in source
    assert "command=self._open_preview_sort_modal" in source
    assert 'text="Sortowanie: Domyślne  ▾"' in source
    assert "buttons_per_row = 4" not in source
    assert "btn_row = idx // buttons_per_row" not in source
    assert "self.preview_sort_buttons_frame = None" in source


def test_sort_modal_is_single_choice_checklist_and_closes_after_choice():
    source = _read("auto_annotation_tool/gui/z3_preview_sort_modal.py")

    assert 'mark = "✓" if selected else " "' in source
    assert "active_key = host._get_preview_sort_mode_key()" in source
    assert "host._on_preview_sort_mode_change(target_key)" in source
    assert "close()" in source
    assert "dialog.grab_set()" in source
    assert "Wybierz jeden sposób uporządkowania tablic." in source


def test_sort_cta_shows_current_active_mode():
    source = _read("auto_annotation_tool/gui/z3_preview_list_ui.py")

    assert 'text=f"Sortowanie: {active_label}  ▾"' in source
    assert "active_key = host._get_preview_sort_mode_key()" in source
    assert 'active_label = str(sort_labels.get(active_key, active_key)' in source


def test_character_tab_exposes_sort_modal_wrapper():
    source = _read("auto_annotation_tool/gui/tab_character_annotation.py")

    assert "from .z3_preview_sort_modal import open_preview_sort_modal" in source
    assert "def _open_preview_sort_modal(self):" in source
    assert "sort_options=PREVIEW_SORT_OPTIONS" in source
    assert "sort_color_keys=PREVIEW_SORT_COLOR_KEYS" in source


def test_sort_option_contract():
    source = _read("auto_annotation_tool/gui/tab_character_annotation.py")
    tree = ast.parse(source)

    options = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PREVIEW_SORT_OPTIONS":
                    options = ast.literal_eval(node.value)
                    break

    assert options == [
        ("DEFAULT", "Domyślne"),
        ("PLATE_ID", "Po plate_id"),
        ("OK", "Po OK"),
        ("GT_Z2", "Po GT z Z2"),
        ("1R", "Po 1R"),
        ("2R", "Po 2R"),
        ("M", "Po M"),
        ("YOLO", "Po YOLO"),
        ("OCR", "Po OCR"),
        ("BOXES", "Po boxach"),
    ]
