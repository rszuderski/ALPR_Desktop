"""Z4LIB029: per-role export choices, summary and guard regression, no Tk loop."""
import ast
from pathlib import Path
import pytest
from auto_annotation_tool.gui.z4_mobile_export_plan import summarize_export_plan

GUI = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_model_export.py'


def entry(role, *, reuse=False, path='', precision='int8', fp32=True, int8=False, size=512):
    return dict(role=role, reuse=reuse, path=path, precision=precision,
                formats=dict(litert=fp32, int8=int8), image_size=size)


def test_new_fp32_pair_is_clear_and_not_reuse_only():
    text, reuse = summarize_export_plan([entry('MT'), entry('MZ', size=320)])
    assert not reuse
    assert 'MT: przygotuję nowy model z best.pt — TFLite FP32, imgsz=512' in text
    assert 'MZ: przygotuję nowy model z best.pt — TFLite FP32, imgsz=320' in text


def test_existing_pair_is_reuse_only_even_when_missing_path():
    text, reuse = summarize_export_plan([entry('MT', reuse=True), entry('MZ', reuse=True)])
    assert reuse
    assert text.count('WYBIERZ PLIK') == 2  # Missing paths are NOT auto-filled.


def test_mixed_sources_are_reported_separately_without_false_reuse_only():
    text, reuse = summarize_export_plan([
        entry('MT', reuse=True, path='C:/models/pinned-mt.alprmodel', precision='int8'),
        entry('MZ', size=320),
    ])
    assert not reuse
    assert 'pinned-mt.alprmodel' in text
    assert 'MZ: przygotuję nowy model z best.pt' in text


def test_multi_format_is_not_silently_described_as_fp32_only():
    text, reuse = summarize_export_plan([entry('MT', int8=True), entry('MZ', size=320)])
    assert 'TFLite FP32, TFLite INT8' in text
    assert not reuse


@pytest.mark.parametrize('rows', [[], [entry('MT')], [entry('MT'), entry('MT')]])
def test_invalid_role_sets_are_rejected(rows):
    with pytest.raises(ValueError):
        summarize_export_plan(rows)


def test_gui_uses_one_wide_column_of_scrollable_cards():
    source = GUI.read_text(encoding='utf-8-sig')
    assert 'model_card_columns = 1  # One wide scrollable card per role tab.' in source
    assert 'presentation.add_role(state["marker"])' in source
    assert '_create_model_panel(model_cards_content, state, 0)' in source


def test_each_role_has_exactly_one_action_choice_with_two_options():
    source = GUI.read_text(encoding='utf-8-sig')
    for role in ('MT', 'MZ'):
        assert f'Przygotuj nowy {role} z wybranego best.pt' in source
        assert f'Wykorzystaj gotowy {role} z biblioteki' in source
    assert 'export_strategy_frame' not in source
    assert source.count('value=False, variable=state["reuse_') == 2
    assert source.count('value=True, variable=state["reuse_') == 2


def test_precision_and_paths_only_visible_on_reuse_and_summary_tracks_changes():
    source = GUI.read_text(encoding='utf-8-sig')
    for role in ('mt', 'mz'):
        assert f'state["_reuse_{role}_precision_row"]' in source
    assert 'def _sync_f4_reuse_plan(*_args)' in source
    assert 'summarize_export_plan' in source
    assert 'f4_no_conversion.set(reuse_only)' in source
    ast.parse(source)


def test_old_preflight_and_no_conversion_guards_still_exist():
    source = GUI.read_text(encoding='utf-8-sig')
    for token in ('require_f4_selection(',
                  'require_f4_no_conversion_request(bundle_request)',
                  'require_f4_no_conversion_request(request, require_fresh_destination=True)',
                  'validate_existing_mt_package(', 'validate_existing_mz_package(',
                  'export_f4_atomic_no_overwrite(',
                  'f4_no_conversion.trace_add("write", _mark_executor_dirty)'):
        assert token in source
