"""Z4LIB034 — operator understands library status; no fake Android approval."""
from pathlib import Path
from types import SimpleNamespace
import ast

from auto_annotation_tool.gui.z4_mobile_artifact_status_ui import (
    artifact_status_label, artifact_status_guidance, validation_result_text,
)
from auto_annotation_tool.gui.z4_mobile_artifact_details import format_artifact_card

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / 'auto_annotation_tool/gui/z4_mobile_artifact_library.py'
REGISTRY = ROOT / 'auto_annotation_tool/gui/z4_mobile_artifact_registry.py'


def row(state='DISCOVERED_NOT_VERIFIED', **kwargs):
    values = dict(name='pair.alprmodel', role='MT+MZ', kind='PACKAGE', model_id='pair',
                  state=state, error='', manifest={'schema': 'alpr.package.v1'},
                  variants='MT: TFLITE/FP32 | MZ: TFLITE/FP32',
                  child_metadata={}, schema='alpr.package.v1', size_bytes=1000000,
                  modified_utc='2026-10-15T12:00:00+00:00')
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_discovered_is_not_desktop_validated():
    item=row()
    assert artifact_status_label(item) == 'Nie sprawdzono paczki'
    assert 'Sprawdź paczkę' in artifact_status_guidance(item).next_step
    assert 'Nie sprawdzono paczki' in format_artifact_card(item)
    assert 'DISCOVERED_NOT_VERIFIED' in format_artifact_card(item)  # technical source preserved


def test_desktop_success_is_ephemeral_and_not_phone_approval():
    item=row()
    assert artifact_status_label(item, desktop_validated=True) == 'Kontrola na PC zaliczona'
    assert artifact_status_label(item) == 'Nie sprawdzono paczki'
    msg=validation_result_text(ok=True, path='C:/models/pair.alprmodel', sha256='f'*64)
    assert 'ZALICZONA' in msg and 'ALPR Mobile' in msg and 'Nie potwierdzono' in msg
    assert 'w chwili kontroli' in artifact_status_guidance(item, desktop_validated=True).meaning


def test_failed_check_does_not_approve_and_gives_action():
    item=row()
    assert artifact_status_label(item, validation_failed=True) == 'Kontrola na PC niezaliczona'
    msg=validation_result_text(ok=False, path='C:/models/pair.alprmodel', problem='checksum mismatch')
    assert 'NIEZALICZONA' in msg and 'checksum mismatch' in msg and 'Nie używaj' in msg
    card=format_artifact_card(item, validation_failure='checksum mismatch')
    assert 'checksum mismatch' in card and 'Kontrola na PC niezaliczona' in card


def test_provenance_blocker_not_hidden_by_desktop_check():
    item=row('REGISTERED_SHA_MISMATCH')
    assert artifact_status_label(item, desktop_validated=True) == 'Niezgodny z historią'
    assert 'Nie używaj' in artifact_status_guidance(item, desktop_validated=True).next_step


def test_error_and_incomplete_states_have_different_meaning():
    assert artifact_status_label(row('MANIFEST_ERROR', kind='INVALID')) == 'Problem z paczką'
    assert artifact_status_label(row('CHILD_METADATA_INCOMPLETE_NOT_VERIFIED')) == 'Niepełny opis paczki'
    assert artifact_status_label(row('REGISTERED_SHA_MATCH_NOT_VERIFIED')) == 'Zgodny z historią eksportu'
    assert artifact_status_label(row('RESEARCH_ONLY_NOT_PACKAGED',kind='RESEARCH_TFLITE')) == 'Plik badawczy — nie paczka'
    assert 'zatwierdzonego' in artifact_status_guidance(row('unrecognized')).next_step


def test_status_explains_what_where_and_why_without_pretending_phone_approval():
    item = row()
    info = artifact_status_guidance(item)
    assert info.label == 'Nie sprawdzono paczki'  # stan, nie polecenie
    assert all(token in info.meaning for token in (
        'kompletności paczki', 'sum kontrolnych', 'precyzji modeli', 'przed testami na telefonie'))
    assert all(token in info.next_step for token in (
        'Pakiety i modele', 'na dole okna', 'Sprawdź paczkę (pełna kontrola)',
        'nie rozpocznie testu na telefonie'))
    card = format_artifact_card(item)
    assert 'Nie sprawdzono paczki' in card
    assert 'DISCOVERED_NOT_VERIFIED' in card
    assert 'Android' in card


def test_other_ambiguous_statuses_are_descriptions_not_commands():
    assert artifact_status_label(row('CHILD_METADATA_INCOMPLETE_NOT_VERIFIED')) == 'Niepełny opis paczki'
    assert artifact_status_label(row('REGISTERED_SHA_MATCH_NOT_VERIFIED')) == 'Zgodny z historią eksportu'
    for state in ('CHILD_METADATA_INCOMPLETE_NOT_VERIFIED', 'REGISTERED_SHA_MATCH_NOT_VERIFIED'):
        info = artifact_status_guidance(row(state))
        assert 'Pakiety i modele' in info.next_step
        assert 'Sprawdź paczkę (pełna kontrola)' in info.next_step
    assert artifact_status_label(row('REGISTERED_SHA_MISMATCH')) == 'Niezgodny z historią'


def test_gui_wires_status_without_changing_registry_codes():
    gui=LIB.read_text(encoding='utf-8-sig')
    reg=REGISTRY.read_text(encoding='utf-8-sig')
    assert 'artifact_status_label(row)' in gui
    assert 'validation_result_text(' in gui
    assert 'self._refresh_artifact_status_cells()' in gui
    assert 'self._verification_error = None' in gui
    assert 'verify_unchanged_mobile_package(' in gui
    assert 'self._verified_selection_path = ticket["path"]' in gui
    assert 'row.state' in reg and 'DISCOVERED_NOT_VERIFIED' in reg
    assert 'f4_zero_conversion_only' not in gui  # packaging logic not moved here
    ast.parse(gui)
