import json
import zipfile

from auto_annotation_tool.ranking.mobile_package_experiments import (
    read_mobile_report_bundle,
)


def test_unknown_manifest_schema_is_rejected(tmp_path):
    path = tmp_path / "unknown.alprsession"

    manifest = {
        "schema": "some.future.or.foreign.schema.v99",
    }

    report = {
        "schema": "alpr.mobile_benchmark_report.v1",
        "report_id": "schema-test",
    }

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(manifest),
        )
        archive.writestr(
            "report.json",
            json.dumps(report),
        )

    bundle = read_mobile_report_bundle(path)

    assert bundle.bundle_kind == "unknown"
    assert bundle.validation.ok is False
    assert any(
        "Nieobsługiwany schemat" in error
        for error in bundle.validation.errors
    )


def test_manifestless_legacy_zip_stays_supported(tmp_path):
    path = tmp_path / "legacy.zip"

    report = {
        "schema": "alpr.mobile_benchmark_report.v1",
        "report_id": "legacy-test",
    }

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "report.json",
            json.dumps(report),
        )

    bundle = read_mobile_report_bundle(path)

    assert bundle.bundle_kind == "legacy_zip"
    assert bundle.validation.ok is True

def test_research_bundle_exposes_explicit_capabilities(tmp_path):
    import hashlib

    path = tmp_path / "research.alprsession"

    report = {
        "schema": "alpr.mobile_benchmark_report.v1",
        "report_id": "capability-test",
    }

    session = {
        "schema": "alpr.mobile_research_session.v1",
        "session_id": "session-1",
        "collection_complete": False,
    }

    sample_schema = {
        "schema": "alpr.mobile_research_samples.v2",
        "normalization_policy": "uppercase_alphanumeric.v1",
        "capabilities": {
            "attempt_registry": True,
            "mt_invocation_identity": True,
            "raw_prediction": True,
            "registration_key": True,
            "mt_input_evidence_references": True,
            "source_timestamp_domain": True,
            "actual_backend_fields": True,
        },
    }

    model_refs = {
        "plate": {"model_id": "plate-test"},
        "character": {"model_id": "character-test"},
    }

    entries = {
        "report.json": json.dumps(report).encode(),
        "session.json": json.dumps(session).encode(),
        "samples/schema.json": json.dumps(
            sample_schema
        ).encode(),
        "pipeline/model_refs.json": json.dumps(
            model_refs
        ).encode(),
    }

    manifest = {
        "schema": "alpr.mobile_research_bundle.v1",
        "entry_sha256": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in entries.items()
        },
    }

    with zipfile.ZipFile(
        path,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(manifest),
        )

        for name, data in entries.items():
            archive.writestr(name, data)

    bundle = read_mobile_report_bundle(path)

    assert bundle.validation.ok is True

    capabilities = bundle.capabilities

    assert capabilities["attempt_registry"] is True
    assert capabilities["mt_invocation_identity"] is True
    assert capabilities["raw_prediction"] is True
    assert capabilities["registration_key"] is True

    assert capabilities["collection_complete"] is False
    assert capabilities["model_refs"] is True

    # Android tego jeszcze jawnie nie deklaruje:
    assert capabilities["fresh_mz_result"] is None

    hashes = capabilities["hash_coverage"]

    assert hashes["declared_entries"] == 4
    assert hashes["checked_entries"] == 4
    assert hashes["complete"] is True
