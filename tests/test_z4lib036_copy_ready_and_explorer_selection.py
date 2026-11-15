# -*- coding: utf-8 -*-
"""Z4LIB036: copy readiness is explicitly bounded to Desktop evidence.

Only pure UI state logic and static GUI wiring; no conversion, Android or Tk loop.
"""
from pathlib import Path
from types import SimpleNamespace
import ast

from auto_annotation_tool.gui.z4_mobile_artifact_validation_history import SCHEMA, SCOPE
from auto_annotation_tool.gui.z4_mobile_artifact_transfer_ui import transfer_guidance


def row(*, kind="PACKAGE", state="DISCOVERED_NOT_VERIFIED", error="", name="C:/models/pair.alprmodel"):
    return SimpleNamespace(kind=kind, state=state, error=error, path=name)


def past(row_obj=None, *, compare="MATCH", outcome="PASSED", sha="a" * 64, scope=SCOPE,
         kind="PACKAGE"):
    row_obj = row_obj or row()
    return {"comparison": compare, "record": {
        "path": row_obj.path, "kind": kind, "sha256": sha, "schema": SCHEMA,
        "outcome": outcome, "scope": scope,
        "android_verified": False, "experiment_verified": False,
    }}


def test_existing_complete_package_pass_is_ready_to_copy_not_android_approved():
    p = row()
    guide = transfer_guidance(p, history_entry=past(p))
    assert guide.copy_ready is True
    assert guide.label == "Gotowa do skopiowania"
    assert "Androidzie" in guide.explanation
    assert "Pokaż wybrany plik w folderze" in guide.next_step
    assert "ALPR Mobile" in guide.next_step


def test_current_desktop_pass_is_ready_only_as_file_copy():
    info = transfer_guidance(row(), desktop_validated=True)
    assert info.copy_ready
    assert "kontrolę techniczną PC" in info.explanation
    assert "nie potwierdza" in info.explanation


def test_unchecked_package_requires_validation_not_new_conversion():
    info = transfer_guidance(row())
    assert not info.copy_ready
    assert info.label == "Najpierw kontrola PC"
    assert "Sprawdź paczkę (pełna kontrola)" in info.next_step
    assert "telefon" in info.next_step


def test_blockers_take_priority_even_over_historical_pass():
    for state in ("MANIFEST_ERROR", "READ_ERROR", "REGISTERED_SHA_MISMATCH",
                  "CHILD_METADATA_INCOMPLETE_NOT_VERIFIED", "REGISTERED_SHA_UNREADABLE",
                  "AN_UNRECOGNIZED_STATE"):
        p = row(state=state)
        info = transfer_guidance(p, history_entry=past(p), desktop_validated=True)
        assert not info.copy_ready, state
        assert "Nie kopiuj" in info.label
    p = row(error="brak manifestu potomnego")
    assert not transfer_guidance(p, history_entry=past(p)).copy_ready


def test_changed_sha_never_advertises_copy_readiness():
    p = row()
    info = transfer_guidance(p, history_entry=past(p, compare="CHANGED"))
    assert not info.copy_ready
    assert "plik zmieniony" in info.label


def test_unreadable_sha_blocks_copy():
    p = row()
    info = transfer_guidance(p, history_entry=past(p, compare="UNREADABLE"))
    assert not info.copy_ready
    assert "SHA" in info.label


def test_failed_current_or_historical_validation_blocks_copy():
    p = row()
    assert not transfer_guidance(p, validation_failed=True).copy_ready
    assert not transfer_guidance(p, history_entry=past(p, outcome="FAILED")).copy_ready
    assert "Nie kopiuj" in transfer_guidance(p, validation_failed=True).label


def test_history_not_accepted_from_wrong_scope_path_kind_or_unknown_hash():
    p = row()
    for entry in (past(p, scope="WRONG"), past(p, kind="MODEL"),
                  past(p, sha="no sha"), past(row(name="C:/other/pair.alprmodel"))):
        assert not transfer_guidance(p, history_entry=entry).copy_ready


def test_single_model_is_not_misrepresented_as_complete_pair():
    p = row(kind="MODEL", name="C:/models/mt.alprmodel")
    info = transfer_guidance(p, desktop_validated=True)
    assert not info.copy_ready
    assert "niepełny ALPR" in info.label
    assert "MT+MZ" in info.next_step


def test_research_tflite_is_not_deployment_package():
    p = row(kind="RESEARCH_TFLITE", name="C:/models/probe.tflite")
    info = transfer_guidance(p, desktop_validated=True)
    assert not info.copy_ready
    assert "nie jest obsługiwaną paczką" in info.explanation


def test_running_never_advertises_readiness():
    assert not transfer_guidance(row(), validation_running=True).copy_ready


def test_no_selected_package():
    assert transfer_guidance(None).label == "Wybierz paczkę"


def test_gui_wires_explorer_highlight_without_new_copy_mechanism():
    repo = Path(__file__).resolve().parents[1]
    gui = (repo / "auto_annotation_tool/gui/z4_mobile_artifact_library.py").read_text(encoding="utf-8-sig")
    card = (repo / "auto_annotation_tool/gui/z4_mobile_artifact_details.py").read_text(encoding="utf-8-sig")
    assert 'text="Pokaż wybrany plik w folderze", command=self.open_location' in gui
    assert 'subprocess.Popen(["explorer.exe", "/select,", str(path)])' in gui
    assert 'self._transfer_status_label(row)' in gui
    assert 'label = state.label' in gui
    view = (repo / 'auto_annotation_tool/gui/z4_mobile_library_view.py').read_text(encoding='utf-8')
    assert 'verdict = transfer_guidance(' in view
    assert 'history_entry=history' in view
    assert 'transfer_guidance(' in card
    assert 'PRZENIESIENIE NA TELEFON' in card
    assert 'history_entry=history_entry' in card
    assert 'verify_unchanged_mobile_package(' in gui
    assert 'observe_last_checks(self.packages_root, scan_result[0])' in gui
    assert 'record_desktop_check(' in gui
    assert 'self._verified_selection_path = ticket["path"]' in gui
    assert 'f4_zero_conversion_only' not in gui
    ast.parse(gui)
    ast.parse(card)
