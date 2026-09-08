"""D1–D9: one logical model versus a complete mobile ALPR pipeline.

Conversion is stubbed; archive assembly, manifests, hashes, validation and
ranking import run normally. No training data or installed runtime is changed.
"""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import pytest

from auto_annotation_tool.exporters import mobile_model_exporter as mobile
from auto_annotation_tool.gui.z4_mobile_export_contract import describe_mobile_export_selection
from auto_annotation_tool.ranking.mobile_package_experiments import read_alpr_package_manifest
from test_mobile_ncnn_output_contract import output_spec


def export_fixture_model(root, role):
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / f"{role}.pt"
    checkpoint.write_bytes(f"fixture checkpoint {role}".encode())
    count = {"plate": 1, "character": 36, "vehicle": 80}[role]
    metadata = {"training": {"best_epoch": 17, "dataset_id": "DS-contract-fixture"},
                "lineage": {"parent": "baseline"}, "metrics": {"map50": 0.82}}
    request = mobile.MobileExportRequest(
        checkpoint=checkpoint, destination=root / f"{role}.alprmodel", role=role,
        formats=("ncnn",), quantizations=("fp32",), image_size=512, metadata=metadata,
    )
    exporter = mobile.MobileModelExporter()
    info = {"labels": [f"class{i}" for i in range(count)], "class_count": count,
            "keypoint_count": 4 if role == "plate" else 0, "keypoint_dimensions": 2 if role == "plate" else 0}
    def convert(_model, *, package_root, **_kwargs):
        relative = ("variants/ncnn/model.param", "variants/ncnn/model.bin")
        files = tuple(package_root / name for name in relative)
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"converted runtime fixture")
        return mobile.ExportedVariant(
            id="ncnn-fp32", runtime="ncnn", precision="fp32", files=files, relative_files=relative,
            input_spec={"width": 512, "height": 512, "channels": 3, "layout": "NCHW",
                        "color": "RGB", "data_type": "FLOAT32", "scale": 1 / 255, "offset": 0},
            output_spec=output_spec(raw=True, role=role, classes=count),
        )
    messages = []
    with patch.object(exporter, "preflight", return_value=[]), \
            patch.object(exporter, "_load_yolo_model", return_value=object()), \
            patch.object(exporter, "_inspect_yolo_checkpoint", return_value=info), \
            patch.object(exporter, "_export_variant", side_effect=convert):
        result = exporter.export(request, progress=lambda _pct, message: messages.append(message))
    exporter.validate_package(result)
    assert messages[-1].startswith("Model mobilny gotowy:")
    return result, metadata


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("mobile_contract")
    return {role: export_fixture_model(root / role, role) for role in ("vehicle", "plate", "character")}


@pytest.mark.parametrize("role", ["plate", "character", "vehicle"], ids=["D1-MT", "D2-MZ", "D3-MP"])
def test_single_model_export_preserves_role_and_provenance(models, role):
    path, metadata = models[role]
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["schema"] == "alpr.model.v1"
    assert manifest["role"] == role
    assert manifest["task"] == ("pose" if role == "plate" else "detect")
    assert "models" not in manifest
    assert all(manifest[key] == value for key, value in metadata.items())
    assert manifest["source"]["checkpoint_sha256"] == mobile.sha256_file(path.with_suffix(".pt"))
    assert manifest["output"]["output_format"] == "raw_yolo"
    with pytest.raises(ValueError, match="not a complete ALPR package"):
        read_alpr_package_manifest(path)


@pytest.mark.parametrize("roles", [("plate", "character"), ("vehicle", "plate", "character")], ids=["D4", "D5"])
def test_complete_package_exports_and_preserves_each_child(models, tmp_path, roles):
    request = mobile.MobileAlprPackageRequest(destination=tmp_path / "complete.alprmodel",
        package_id="complete", **{role + "_package": models[role][0] for role in roles})
    exporter = mobile.MobileAlprPackageExporter()
    path = exporter.export(request)
    exporter.validate_package(path)
    manifest = read_alpr_package_manifest(path)
    assert manifest["schema"] == "alpr.package.v1"
    assert set(manifest["models"]) == set(roles)
    with zipfile.ZipFile(path) as archive:
        for role in roles:
            assert archive.read(manifest["models"][role]["package_file"]) == models[role][0].read_bytes()
    for missing in ("plate", "character"):
        invalid = copy.deepcopy(manifest)
        invalid["models"].pop(missing)
        with pytest.raises(mobile.MobileExportError, match="Kompletny pakiet ALPR wymaga modeli MT i MZ"):
            exporter._validate_manifest_basic(invalid)
        invalid_path = tmp_path / f"missing-{missing}.alprmodel"
        with zipfile.ZipFile(invalid_path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(invalid))
        with pytest.raises(ValueError, match="required MT\\+MZ"):
            read_alpr_package_manifest(invalid_path)


@pytest.mark.parametrize("roles", [("plate",), ("vehicle", "plate"), ("character",), ("vehicle", "character"), ("vehicle",), ()])
def test_d6_d7_incomplete_package_is_rejected_before_writing(models, tmp_path, roles):
    request = mobile.MobileAlprPackageRequest(destination=tmp_path / "incomplete.alprmodel",
        **{role + "_package": models[role][0] for role in roles})
    exporter = mobile.MobileAlprPackageExporter()
    for method in (exporter.export, exporter.bundle_existing):
        with pytest.raises(mobile.MobileExportError, match="użyj eksportu modelu mobilnego"):
            method(request)
        assert not request.destination.exists()
    with pytest.raises(mobile.MobileExportError, match="Kompletny pakiet ALPR wymaga modeli MT i MZ"):
        exporter.bundle_existing(request, _skip_preflight=True)


@pytest.mark.parametrize("markers,schema", [
    (("MP",), "alpr.model.v1"), (("MT",), "alpr.model.v1"), (("MZ",), "alpr.model.v1"),
    (("MT", "MZ"), "alpr.package.v1"), (("MP", "MT", "MZ"), "alpr.package.v1"),
    (("MP", "MT"), ""), (("MP", "MZ"), ""), (("MT", "MT"), ""), ((), ""), (("unknown",), ""),
])
def test_d8_selection_names_the_actual_export_contract(markers, schema):
    selection = describe_mobile_export_selection(markers)
    assert selection.valid == bool(schema)
    assert selection.schema == schema
    if schema:
        assert schema in selection.message
        noun = "pakiet ALPR" if schema == "alpr.package.v1" else "model mobilny"
        assert noun in selection.label
        assert selection.export_label == f"Eksportuj {noun} (.alprmodel)"
        assert selection.save_title == f"Zapisz {noun} (.alprmodel)"
        assert noun.lower() in selection.success_title.lower()


def test_d9_documentation_does_not_allow_single_model_as_alpr_package():
    root = Path(__file__).resolve().parents[1]
    paths = ["auto_annotation_tool/gui/tab_help.py", "docs/specyfikacja_agenta_aplikacji_mobilnej_alpr.md",
             "docs/eksport_mobilny_kwantyzacja.md", "alpr_python_exporter_handoff.md",
             "DZIENNIK_ARCHITEKTURY_I_ZMIAN.md", "docs/freeze_smoke_test_checklist.md"]
    for name in paths:
        text = (root / name).read_text(encoding="utf-8")
        assert "alpr.package.v1 może zawierać jeden model" not in text, name
        assert "Pakiet może zawierać jeden model logiczny" not in text, name
        assert "alpr.model.v1" in text and "alpr.package.v1" in text, name
