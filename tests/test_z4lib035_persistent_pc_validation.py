"""Z4LIB035: persistent but non-authoritative Desktop checks, no GUI or conversion."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import ast

import pytest

from auto_annotation_tool.gui.z4_mobile_artifact_validation_history import (
    FILENAME, SCHEMA, SCOPE, history_key, record_desktop_check,
    read_last_checks, observe_last_checks, historical_status_label, historical_detail_text,
)


def row(path, *, state="DISCOVERED_NOT_VERIFIED", kind="PACKAGE"):
    return SimpleNamespace(path=str(path), state=state, kind=kind)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_pass_is_recorded_and_restored_after_reopening(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    file = root / "pair.alprmodel"
    file.write_bytes(b"stable-MT-MZ")
    proof = record_desktop_check(root, path=str(file), kind="PACKAGE", passed=True,
                                 sha256=digest(file.read_bytes()))
    assert proof["schema"] == SCHEMA and proof["scope"] == SCOPE
    assert proof["android_verified"] is False and proof["experiment_verified"] is False
    history = read_last_checks(root)  # simulate a new application process
    assert history[history_key(file)]["outcome"] == "PASSED"
    observed = observe_last_checks(root, [row(file)])
    info = observed[history_key(file)]
    assert info["comparison"] == "MATCH"
    assert historical_status_label(row(file), info, fallback="Nie sprawdzono paczki") == "Poprzednia kontrola PC: OK"
    detail = historical_detail_text(info)
    assert "Data (UTC)" in detail and digest(b"stable-MT-MZ") in detail
    assert "NIE zatwierdzenie działania na Androidzie" in detail
    assert "Przed ponownym użyciem uruchom pełną kontrolę" in detail


def test_mutated_package_disables_previous_success(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    f = root / "pair.alprmodel"
    f.write_bytes(b"before")
    record_desktop_check(root, path=str(f), kind="PACKAGE", passed=True, sha256=digest(f.read_bytes()))
    f.write_bytes(b"after")
    value = observe_last_checks(root, [row(f)])[history_key(f)]
    assert value["comparison"] == "CHANGED"
    assert historical_status_label(row(f), value, fallback="---") == "Zmieniony od kontroli"
    assert "Poprzedni wynik nie dotyczy obecnej wersji" in historical_detail_text(value)


def test_failed_check_is_preserved_without_faking_approval(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    f = root / "invalid.alprmodel"
    f.write_bytes(b"wrong")
    record_desktop_check(root, path=str(f), kind="PACKAGE", passed=False,
                         sha256=digest(f.read_bytes()), detail="zły manifest")
    obs = observe_last_checks(root, [row(f)])[history_key(f)]
    assert historical_status_label(row(f), obs, fallback="---") == "Poprzednia kontrola PC: błąd"
    assert "niezaliczona" in historical_detail_text(obs)


def test_provenance_mismatch_always_wins_over_old_pass(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    f = root / "pair.alprmodel"
    f.write_bytes(b"stable")
    record_desktop_check(root, path=str(f), kind="PACKAGE", passed=True, sha256=digest(f.read_bytes()))
    problematic = row(f, state="REGISTERED_SHA_MISMATCH")
    obs = observe_last_checks(root, [problematic])[history_key(f)]
    assert historical_status_label(problematic, obs, fallback="Niezgodny z historią") == "Niezgodny z historią"


def test_corrupt_history_and_unreadable_or_no_hash_are_fail_closed(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    f = root / "pair.alprmodel"
    f.write_bytes(b"a")
    record_desktop_check(root, path=str(f), kind="PACKAGE", passed=False, sha256="", detail="cannot read")
    obs = observe_last_checks(root, [row(f)])[history_key(f)]
    assert obs["comparison"] == "UNCONFIRMED"
    assert historical_status_label(row(f), obs, fallback="---") != "Poprzednia kontrola PC: OK"
    journal = root / "_registry" / FILENAME
    with journal.open("ab") as fd:
        fd.write(b"garbage-not-json\n")
        fd.write(b'{"schema":"broken"}\n')
    assert history_key(f) in read_last_checks(root)
    assert all(x.get("scope") == SCOPE for x in read_last_checks(root).values())
    journal.write_bytes(b"x" * (6*1024*1024+1))
    assert read_last_checks(root) == {}


def test_record_is_append_only_and_does_not_touch_export_ledger_or_models(tmp_path):
    root = tmp_path / "models"
    (root / "_registry").mkdir(parents=True)
    export_log = root / "_registry" / "mobile_export_events.v1.jsonl"
    export_log.write_bytes(b"historic-data\n")
    f = root / "pair.alprmodel"
    f.write_bytes(b"model-data")
    original = f.read_bytes()
    for _ in range(2):
        record_desktop_check(root, path=str(f), kind="PACKAGE", passed=True, sha256=digest(original))
    assert len((root / "_registry" / FILENAME).read_text(encoding="utf-8").splitlines()) == 2
    assert export_log.read_bytes() == b"historic-data\n"
    assert f.read_bytes() == original


def test_symlink_ledger_is_rejected(tmp_path):
    root = tmp_path / "models"
    (root / "_registry").mkdir(parents=True)
    original = tmp_path / "outside"
    original.write_bytes(b"untouched")
    pointer = root / "_registry" / FILENAME
    try:
        pointer.symlink_to(original)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable")
    f = root / "pair.alprmodel"
    f.write_bytes(b"bytes")
    with pytest.raises(ValueError):
        record_desktop_check(root, path=str(f), kind="PACKAGE", passed=True, sha256=digest(b"bytes"))
    assert read_last_checks(root) == {}
    assert original.read_bytes() == b"untouched"


def test_existing_validator_and_status_guards_remain_in_gui():
    root = Path(__file__).resolve().parents[1]
    code = (root / "auto_annotation_tool/gui/z4_mobile_artifact_library.py").read_text(encoding="utf-8-sig")
    detail = (root / "auto_annotation_tool/gui/z4_mobile_artifact_details.py").read_text(encoding="utf-8-sig")
    assert 'verify_unchanged_mobile_package(' in code
    assert 'model_validator=MobileModelExporter().validate_package' in code
    assert 'bundle_validator=MobileAlprPackageExporter().validate_package' in code
    assert 'self._verified_selection_path = ticket["path"]' in code
    assert 'observe_last_checks(self.packages_root, scan_result[0])' in code
    assert 'record_desktop_check(' in code
    assert 'self._refresh_artifact_status_cells()' in code
    assert 'history_entry=self._previous_record(row)' in code
    assert 'historical_detail_text(' in detail
    assert 'f4_zero_conversion_only' not in code
    ast.parse(code)
    ast.parse(detail)
