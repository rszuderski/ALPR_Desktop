"""Read-only diagnostics must preserve failure evidence and fail closed."""
import errno
from pathlib import Path

import pytest

from auto_annotation_tool.gui import mobile_comparison_core as core
from auto_annotation_tool.gui.mobile_compare_review_gate import imported_verification
from test_z4_akt87_verified_mt_mz import make_export, SID, SHA


@pytest.mark.parametrize("operation", ["resolve_manifest_file", "read_and_verify_sha256"])
def test_permission_failure_is_unreadable_and_has_file_operation_identity(tmp_path, monkeypatch, operation):
    directory = make_export(tmp_path)
    file = directory / "canvas_review_summary.json"
    original = Path.resolve if operation == "resolve_manifest_file" else core._file_hash
    def deny(path, *args, **kwargs):
        if Path(path).name == file.name:
            raise PermissionError(errno.EACCES, "test access denied", str(path))
        return original(path, *args, **kwargs)
    if operation == "resolve_manifest_file":
        monkeypatch.setattr(Path, "resolve", deny)
    else:
        monkeypatch.setattr(core, "_file_hash", deny)
    trace = {}
    result = imported_verification(SID, SHA, tmp_path, diagnostics=trace)
    assert result.state == "UNREADABLE" and not result.ready
    assert result.label == "Brak odczytu"
    assert "uszkodzony" not in result.reason
    detail = trace["exports"][0]
    assert detail["error_type"] == "PermissionError" and detail["errno"] == errno.EACCES
    assert Path(detail["error_file"]) == file
    assert detail["operation"] == operation
    assert detail["session_id_matches"] and detail["source_sha_matches"]
    assert detail["loader_module"] == core.__name__
    assert Path(detail["loader_module_path"]) == Path(core.__file__)
    assert detail["traceback"]


def test_hash_error_retains_invalid_and_does_not_report_access_error(tmp_path):
    directory = make_export(tmp_path)
    file = directory / "canvas_review_summary.json"
    contents = file.read_bytes()
    file.write_bytes(contents.replace(b"ABC1234", b"XYZ1234"))
    trace = {}
    result = imported_verification(SID, SHA, tmp_path, diagnostics=trace)
    assert result.state == "INVALID"
    detail = trace["exports"][0]
    assert detail["error_type"] == "InvalidExport" and detail["errno"] is None
    assert detail["operation"] == "read_and_verify_sha256"
    assert "SHA-256 mismatch" in result.reason


def test_newer_unreadable_export_cannot_fall_back_to_older_ready(tmp_path, monkeypatch):
    make_export(tmp_path)
    newer = make_export(tmp_path, rev=8)
    original = core._file_hash
    def deny(path):
        if Path(path).parent == newer:
            raise PermissionError(errno.EACCES, "test access denied", str(path))
        return original(path)
    monkeypatch.setattr(core, "_file_hash", deny)
    trace = {}
    result = imported_verification(SID, SHA, tmp_path, diagnostics=trace)
    assert result.state == "UNREADABLE" and not result.ready
    assert len(trace["exports"]) == 2
    assert trace["exports"][0]["manifest_verified"] is True
    assert trace["exports"][1]["error_type"] == "PermissionError"


def test_diagnostics_keep_success_and_mismatched_identity_without_writing(tmp_path):
    directory = make_export(tmp_path)
    frozen = {p.name:p.read_bytes() for p in directory.iterdir()}
    trace = {}
    assert imported_verification(SID, SHA, tmp_path, diagnostics=trace).ready
    detail = trace["exports"][0]
    assert detail["manifest_verified"] and detail["files_verified"] == detail["files_expected"] == 3
    assert detail["operation"] == "complete" and "error_type" not in detail
    mismatch = {}
    assert not imported_verification(SID, "c"*64, tmp_path, diagnostics=mismatch).ready
    assert mismatch["exports"][0]["session_id_matches"]
    assert not mismatch["exports"][0]["source_sha_matches"]
    assert frozen == {p.name:p.read_bytes() for p in directory.iterdir()}
