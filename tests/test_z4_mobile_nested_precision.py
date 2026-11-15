"""Z4LIB004: nested per-role precision comes from root-SHA-checked child manifests.

Tests use synthetic ZIPs. No conversions, graph imports, or writes to real packages.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from auto_annotation_tool.gui import z4_mobile_artifact_registry as catalog


def child(role, precision):
    return {"schema": "alpr.model.v1", "role": role,
            "task": "pose" if role == "plate" else "detect",
            "model_id": f"{role}-frozen-test", "input": {"width": 512 if role == "plate" else 320},
            "source": {"checkpoint_sha256": "b"*64},
            "variants": [{"runtime": "tflite", "precision": precision}]}


def make_set(path, *, roles=("plate", "character"), bad_sha=None,
             bad_role=None, oversized=None, bad_path=None, missing=None,
             root_only=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    models = {}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for role in roles:
            item = child(role, "int8" if role == "plate" else "fp32")
            if role == bad_role:
                item["role"] = "plate" if role != "plate" else "character"
            rel = f"models/{role}/manifest.json"
            data = json.dumps(item).encode("utf-8")
            if role == oversized:
                data = b" " * (catalog.MANIFEST_CAP + 1)
            member = f"../outside/{role}.json" if role == bad_path else rel
            if role != missing and not root_only:
                archive.writestr(rel, data)
            declared_sha = hashlib.sha256(data).hexdigest()
            if role == bad_sha:
                declared_sha = "0"*64
            entry = {"role":role, "task": "pose" if role=="plate" else "detect",
                     "model_id": f"{role}-frozen-test"}
            if root_only:
                entry["variants"] = item["variants"]
            else:
                entry.update(manifest_file=member, sha256={member: declared_sha})
            models[role] = entry
        archive.writestr("manifest.json", json.dumps({"schema":"alpr.package.v1",
                                                        "package_id":"SYNTH", "models": models}))
    return path


def test_nested_sidecars_supply_mt_and_mz_precisions_without_graph_claim(tmp_path):
    path = make_set(tmp_path / "alpr" / "set.alprmodel")
    row = catalog.read_archive(path)
    assert row.role == "MT+MZ" and row.kind == "PACKAGE"
    assert row.variants == "MT: TFLITE/INT8 | MZ: TFLITE/FP32"
    assert row.state == "DISCOVERED_NOT_VERIFIED"
    assert row.child_metadata["plate"]["origin"].endswith("NOT_GRAPH_VERIFIED")
    assert row.child_metadata["character"]["model_id"] == "character-frozen-test"
    assert row.child_metadata["plate"]["checkpoint_sha256"] == "b"*64
    assert not catalog.history_file(tmp_path).exists()


def test_three_role_package(tmp_path):
    row = catalog.read_archive(make_set(tmp_path / "with-mp.alprmodel",
                                            roles=("vehicle","plate","character")))
    assert row.role == "MP+MT+MZ"
    assert row.variants == "MP: TFLITE/FP32 | MT: TFLITE/INT8 | MZ: TFLITE/FP32"


def test_tampered_sidecar_hash_suppresses_precision(tmp_path):
    row = catalog.read_archive(make_set(tmp_path / "bad.alprmodel", bad_sha="plate"))
    assert row.variants == "MT: - | MZ: TFLITE/FP32"
    assert row.state == "CHILD_METADATA_INCOMPLETE_NOT_VERIFIED"
    assert "SHA-256" in row.error and "plate" not in row.child_metadata


def test_wrong_role_suppresses_metadata_even_with_valid_sha(tmp_path):
    row = catalog.read_archive(make_set(tmp_path / "role.alprmodel", bad_role="character"))
    assert row.variants == "MT: TFLITE/INT8 | MZ: -"
    assert "rola" in row.error


def test_oversized_compressed_sidecar_is_never_decompressed(tmp_path):
    row = catalog.read_archive(make_set(tmp_path / "oversize.alprmodel", oversized="character"))
    assert row.variants.endswith("MZ: -")
    assert "przekracza limit" in row.error


def test_missing_sidecar_does_not_invent_precision(tmp_path):
    row = catalog.read_archive(make_set(tmp_path / "missing.alprmodel", missing="plate"))
    assert row.variants.startswith("MT: -")
    assert "nie znaleziono" in row.error


def test_root_only_legacy_variant_is_labeled_unverified(tmp_path):
    row = catalog.read_archive(make_set(tmp_path / "root-only.alprmodel", root_only=True))
    assert row.variants == "MT: TFLITE/INT8 | MZ: TFLITE/FP32"
    assert row.child_metadata["plate"]["origin"] == "ROOT_ONLY_UNVERIFIED"
    assert row.state == "DISCOVERED_NOT_VERIFIED"


def test_unexpected_sidecar_path_not_followed(tmp_path):
    p = tmp_path / "path.alprmodel"
    row = catalog.read_archive(make_set(p, bad_path="plate"))
    assert row.variants == "MT: - | MZ: TFLITE/FP32"
    assert row.state == "CHILD_METADATA_INCOMPLETE_NOT_VERIFIED"
    assert "nieobsługiwana ścieżka" in row.error


def test_sha_mismatch_is_still_detected_for_package_with_bad_child_metadata(tmp_path):
    root = tmp_path / "managed"
    path = make_set(root / "alpr" / "bad-child.alprmodel", bad_sha="plate")
    catalog.append_event(root, operation_id="Z4LIB004_SHA_12345", operation="COMPLETE_PACKAGE_EXPORT",
                         outcome="SUCCEEDED", artifact_path=str(path))
    rows, events, _ = catalog.library_snapshot(root, tmp_path)
    assert rows[0].state == "REGISTERED_SHA_MATCH_NOT_VERIFIED"
    assert "SHA-256" in rows[0].error
    with path.open("ab") as stream:
        stream.write(b"TAMPER")
    rows, events, _ = catalog.library_snapshot(root, tmp_path)
    assert rows[0].state == "REGISTERED_SHA_MISMATCH"


def test_package_file_unchanged_after_discovery(tmp_path):
    path = make_set(tmp_path / "intact.alprmodel")
    before = catalog.sha256_file(path)
    row = catalog.read_archive(path)
    assert row.variants.startswith("MT: TFLITE")
    assert catalog.sha256_file(path) == before


def test_library_details_show_child_provenance():
    path = Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/z4_mobile_artifact_library.py"
    source = path.read_text(encoding="utf-8-sig")
    assert '"child_metadata": row.child_metadata' in source
    assert "SHA manifestu potomnego" in source
