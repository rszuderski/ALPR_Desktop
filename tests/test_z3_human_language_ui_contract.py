from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")

def test_primary_character_actions_use_human_language():
    source = _read("auto_annotation_tool/gui/z3_detection_tab_ui.py")
    assert 'text="Uruchom wykrywanie"' in source
    assert 'text="Sprawdź i popraw"' in source
    assert 'text="Zatwierdź tablicę"' in source
    assert 'text="Uruchom RAW"' not in source
    assert 'text="RAW → REVIEW"' not in source
    assert 'text="Zatwierdź GOLD"' not in source
    assert 'text="Perfect"' not in source
    assert 'text="Perf."' not in source

def test_review_dialogs_do_not_expose_internal_state_names():
    source = _read("auto_annotation_tool/gui/z3_review_runtime.py")
    banned = (
        'messagebox.showinfo("REVIEW"',
        'messagebox.showinfo("GOLD"',
        '"Nie można zatwierdzić GOLD"',
        '"REVIEW nadal wymaga korekty"',
        '"Tablica została jawnie zatwierdzona jako GOLD."',
    )
    for item in banned:
        assert item not in source

def test_primary_help_uses_human_language():
    data = json.loads(_read("auto_annotation_tool/help_db.json"))
    joined = "\n".join(
        str(data.get(key, {}).get("full_help", ""))
        for key in ("btn_export_yolo", "t2_pz3_goldpack", "t2_pz3_source", "camp_step3")
    ).lower()
    banned_phrases = (
        "perfect",
        "gold pack",
        "raw ->",
        "raw →",
        "review/",
        "review ",
        "gold approved",
        "zatwierdź gold",
        "uruchom raw",
    )
    for banned in banned_phrases:
        assert banned not in joined

def test_campaign_title_is_plain_language():
    for rel in (
        "auto_annotation_tool/gui/campaign_stage_logic.py",
        "auto_annotation_tool/gui/tab_campaign.py",
    ):
        source = _read(rel)
        assert "E3. Znaki i gold pack" not in source
        assert "E3. Przygotowanie znaków" in source
