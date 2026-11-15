import hashlib
import json
import zipfile

import pytest

from auto_annotation_tool.exporters import mobile_model_exporter as mobile
from test_mobile_model_package_contract import export_fixture_model


def rewrite_package(source, destination, edit):
    with zipfile.ZipFile(source, "r") as archive:
        entries = {
            name: archive.read(name)
            for name in archive.namelist()
        }

    manifest = json.loads(
        entries["manifest.json"].decode("utf-8")
    )

    edit(manifest, entries)

    entries["manifest.json"] = json.dumps(
        manifest
    ).encode("utf-8")

    with zipfile.ZipFile(
        destination,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)

    return destination


def make_complete_package(tmp_path):
    plate = export_fixture_model(
        tmp_path / "plate",
        "plate",
    )[0]

    character = export_fixture_model(
        tmp_path / "character",
        "character",
    )[0]

    exporter = mobile.MobileAlprPackageExporter()

    package = exporter.export(
        mobile.MobileAlprPackageRequest(
            destination=tmp_path / "complete.alprmodel",
            plate_package=plate,
            character_package=character,
        )
    )

    return exporter, package


def test_rejects_outer_model_id_different_from_nested_model(
    tmp_path,
):
    exporter, package = make_complete_package(tmp_path)

    def edit(manifest, entries):
        manifest["models"]["plate"]["model_id"] = (
            "different-model-id"
        )

    invalid = rewrite_package(
        package,
        tmp_path / "wrong-model-id.alprmodel",
        edit,
    )

    with pytest.raises(
        mobile.MobileExportError,
        match="model_id",
    ):
        exporter.validate_package(invalid)


def test_rejects_sidecar_manifest_different_from_nested_model(
    tmp_path,
):
    exporter, package = make_complete_package(tmp_path)

    def edit(manifest, entries):
        item = manifest["models"]["plate"]

        manifest_file = item["manifest_file"]

        sidecar = json.loads(
            entries[manifest_file].decode("utf-8")
        )

        sidecar["name"] = "different-sidecar-name"

        entries[manifest_file] = json.dumps(
            sidecar
        ).encode("utf-8")

        # Aktualizujemy hash, żeby test nie zatrzymał się
        # wcześniej na kontroli SHA-256.
        item["sha256"][manifest_file] = hashlib.sha256(
            entries[manifest_file]
        ).hexdigest()

    invalid = rewrite_package(
        package,
        tmp_path / "wrong-sidecar.alprmodel",
        edit,
    )

    with pytest.raises(
        mobile.MobileExportError,
        match="kopia manifestu",
    ):
        exporter.validate_package(invalid)