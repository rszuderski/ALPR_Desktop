"""Z4LIB028 after Z4LIB029: checkpoint versus package sources stay explicit.

The earlier global radio choice is superseded by separate per-role choices.
No Tk window, conversion or disk mutation.
"""
import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_model_export.py'


def test_training_sources_are_shown_before_role_choices():
    source = SOURCE.read_text(encoding='utf-8-sig')
    assert '''text=f"{marker} | {candidate.get('model_label') or '-'}"''' in source
    assert 'presentation.add_role(state["marker"])' in source
    assert 'export_plan_var' in source
    ast.parse(source)


def test_paths_are_for_existing_mobile_packages_only():
    source = SOURCE.read_text(encoding='utf-8-sig')
    for expected in (
        'Paczka MT: wybierz plik .alprmodel z biblioteki',
        'Paczka MZ: wybierz plik .alprmodel z biblioteki',
        'Przygotuj nowy MT z wybranego best.pt',
        'Przygotuj nowy MZ z wybranego best.pt',
    ):
        assert expected in source


def test_no_duplicate_global_mode_or_bypassed_safety():
    source = SOURCE.read_text(encoding='utf-8-sig')
    assert 'export_strategy_frame' not in source
    assert 'f4_no_conversion = tk.BooleanVar(master=executor_modal, value=False)' in source
    assert 'f4_no_conversion.trace_add("write", _mark_executor_dirty)' in source
    for guard in (
        'require_f4_selection(',
        'require_f4_no_conversion_request(bundle_request)',
        'require_f4_no_conversion_request(request, require_fresh_destination=True)',
        '"f4_zero_conversion_only": bool(f4_no_conversion.get())',
        'export_f4_atomic_no_overwrite(',
    ):
        assert guard in source
