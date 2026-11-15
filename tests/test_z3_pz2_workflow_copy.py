from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "auto_annotation_tool" / "gui" / "z3_detection_tab_ui.py"
EVENTS = ROOT / "auto_annotation_tool" / "gui" / "z3_preview_events.py"
STATUS = ROOT / "auto_annotation_tool" / "gui" / "z3_preview_ui.py"


def _copy_block():
    source = UI.read_text(encoding="utf-8-sig")
    start = source.index("self.preview_list_intro_lbl = tk.Label(")
    end = source.index("self._set_inline_status_label_state(", start)
    return source[start:end]


def test_left_panel_explains_real_flow_and_number_from_z2():
    block = _copy_block()
    for label in ("Sprawdź i popraw", "Alt+W", "Numer z Z2", "Zmień numer", "Krok 2: dataset PZ3"):
        assert label in block
    assert "naciśnij O" in block
    assert "status OK" in block


def test_left_panel_avoids_internal_jargon():
    block = _copy_block()
    for jargon in ("GT", "GOLD", "perfect", "REVIEW"):
        assert jargon not in block


def test_o_copy_mentions_number_from_z2_on_mismatch():
    source = EVENTS.read_text(encoding="utf-8-sig")
    assert "wpisane znaki różnią się od numeru z Z2" in source
    assert "Zmień numer" in source


def test_status_copy_exposes_z2_mismatch_without_gt_jargon():
    source = STATUS.read_text(encoding="utf-8-sig")
    assert "Różni się od numeru z Z2" in source
    assert "zgodna z GT" not in source
