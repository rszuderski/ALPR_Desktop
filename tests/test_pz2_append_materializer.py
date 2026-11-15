import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from auto_annotation_tool.registry import pz2_append_materializer as materializer

from auto_annotation_tool.registry import RegistryDatabase
from auto_annotation_tool.registry.az_package_transport import (
    AZ_PACKAGE_SCHEMA,
    append_az_package,
    load_az_package,
)
from auto_annotation_tool.registry.az_registry import (
    AZRegistry,
    crop_contract_sha256,
)
from auto_annotation_tool.registry.az_revision_store import (
    compute_az_payload_sha256,
)
from auto_annotation_tool.registry.crop_identity import (
    build_pz1_crop_identity,
    compute_crop_identity_sha256,
)
from auto_annotation_tool.registry.pz2_append_materializer import (
    PZ2AppendMaterializationError,
    materialize_az_append_to_preview,
)


class PZ2AppendMaterializerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "Workspace"
        self.workspace.mkdir()

        self.db = RegistryDatabase(
            self.workspace / "_registry" / "alpr_registry.sqlite3"
        )
        self.registry = AZRegistry(
            self.db,
            workspace_dir=self.workspace,
        )
        self.registry.initialize()

        with self.db.transaction() as con:
            con.execute(
                """
                INSERT INTO projects(
                    project_id, display_name, folder_name
                ) VALUES (?, ?, ?)
                """,
                ("PRJ-TARGET", "Target", "Target_F00"),
            )

        self.preview = (
            self.workspace
            / "9_projects"
            / "Target_F00"
            / "3_cropped_characters"
            / "run_001"
        )
        self.images = self.preview / "images"
        self.images.mkdir(parents=True)

        self.old_row = {
            "characters": [{
                "character": "X",
                "bbox": [10, 10, 20, 30],
                "method": "manual",
            }],
            "status": "perfect",
            "ground_truth_text": "OLD123",
            "gold_state": {
                "approved": True,
                "excluded": False,
                "candidate": True,
            },
            "review_state": {
                "status": "approved",
                "human_edited": True,
            },
            "crop_id": "CROP-OLD",
            "crop_identity_sha256": "f" * 64,
        }
        self.initial_metadata = {
            "existing_plate": copy.deepcopy(self.old_row)
        }
        (self.preview / "metadata.json").write_text(
            json.dumps(
                self.initial_metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.images / "existing_plate.jpg").write_bytes(b"old")

        self.package_dir = self.root / "package"
        (self.package_dir / "crops").mkdir(parents=True)

        self.crop_bytes = b"portable-crop-A"
        self.package_crop_path = (
            self.package_dir / "crops" / "plate_A.jpg"
        )
        self.package_crop_path.write_bytes(self.crop_bytes)

        self.source_sha = "a" * 64
        self.geometry_sha = "b" * 64
        self.identity = build_pz1_crop_identity(
            source_image_id="img-sha256-" + self.source_sha,
            source_annotation_id="plate-ann-A",
            source_geometry_hash=self.geometry_sha,
            interpolation="lanczos4",
            output_width=256,
            output_height=64,
        )
        self.identity_sha = compute_crop_identity_sha256(
            self.identity
        )
        self.contract_sha = crop_contract_sha256(
            self.identity["crop_contract"]
        )

    def tearDown(self):
        self.tmp.cleanup()

    def az_payload(self):
        return {
            "crop_identity_sha256": self.identity_sha,
            "characters": [{
                "character": "Q",
                "bbox": [0.10, 0.20, 0.20, 0.80],
                "row": 0,
                "method": "manual",
                "source_kind": "local_manual",
                "confidence": 1.0,
            }],
            "layout": {
                "kind": "1R",
                "confirmed": True,
            },
            "gold_state": {
                "approved": True,
                "excluded": False,
                "candidate": True,
            },
            "status": "perfect",
            "expected_text": "Q123",
        }

    def crop_entry(self, *, with_az=True):
        payload = self.az_payload()
        entry = {
            "entry_id": "donor-A",
            "crop_identity": self.identity,
            "crop_identity_sha256": self.identity_sha,
            "crop_contract_sha256": self.contract_sha,
            "source_file_sha256": self.source_sha,
            "artifact": {
                "path": "crops/plate_A.jpg",
                "sha256": hashlib.sha256(
                    self.crop_bytes
                ).hexdigest(),
                "size_bytes": len(self.crop_bytes),
                "width": 256,
                "height": 64,
            },
        }
        if with_az:
            entry["az"] = {
                "payload": payload,
                "payload_sha256": compute_az_payload_sha256(
                    payload
                ),
                "source_kind": "project_export",
                "source_status": "perfect",
                "trust_state": "source_local",
                "origin_project_id": "PRJ-SOURCE",
                "origin_iteration": 1,
                "source_az_revision_id": "AZR-SOURCE-1",
                "created_at": "2026-09-29T12:00:00+00:00",
            }
        return entry

    def write_package(self, *, with_az=True):
        manifest = {
            "schema": AZ_PACKAGE_SCHEMA,
            "package_id": "AZPKG-TEST",
            "created_at": "2026-09-29T12:00:00+00:00",
            "source": {
                "project_id": "PRJ-SOURCE",
                "project_name": "Donor",
                "iteration": 1,
            },
            "crops": [
                self.crop_entry(with_az=with_az)
            ],
        }
        (self.package_dir / "manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return load_az_package(self.package_dir)

    def append(self, package, *, destination=None):
        return append_az_package(
            self.registry,
            package,
            target_project_id="PRJ-TARGET",
            iteration_num=1,
            target_artifact_dir=(
                destination
                if destination is not None
                else self.images
            ),
        )

    def load_metadata(self):
        return json.loads(
            (self.preview / "metadata.json").read_text(
                encoding="utf-8"
            )
        )

    def materialize(self, package, append_result=None):
        return materialize_az_append_to_preview(
            self.registry,
            package,
            append_result=append_result,
            preview_dir=self.preview,
            target_project_id="PRJ-TARGET",
            iteration_num=1,
        )

    def test_append_with_az_adds_one_pending_row_and_preserves_old(self):
        package = self.write_package(with_az=True)
        append_result = self.append(package)

        result = self.materialize(package, append_result)

        self.assertEqual(result.before_count, 1)
        self.assertEqual(result.candidates, 1)
        self.assertEqual(result.added, 1)
        self.assertEqual(result.already_materialized, 0)
        self.assertEqual(result.with_az, 1)
        self.assertEqual(result.without_az, 0)
        self.assertEqual(result.after_count, 2)
        self.assertIsNotNone(result.backup_path)
        self.assertTrue(Path(result.backup_path).is_file())

        metadata = self.load_metadata()
        self.assertEqual(
            metadata["existing_plate"],
            self.initial_metadata["existing_plate"],
        )

        item = result.items[0]
        row = metadata[item.plate_id]
        self.assertEqual(row["crop_id"], item.crop_id)
        self.assertEqual(
            row["crop_identity_sha256"],
            self.identity_sha,
        )
        self.assertEqual(
            row["characters"][0]["character"],
            "Q",
        )
        self.assertEqual(row["ground_truth_text"], "Q123")
        self.assertEqual(row["status"], "needs_fix")
        self.assertFalse(row["gold_state"]["approved"])
        self.assertFalse(row["gold_state"]["candidate"])
        self.assertEqual(
            row["review_state"]["status"],
            "in_progress",
        )
        self.assertEqual(
            row["review_state"]["source"],
            "az_project_import",
        )
        self.assertTrue(
            row["az_reuse"]["requires_review"]
        )
        self.assertEqual(
            row["append_provenance"]["package_id"],
            "AZPKG-TEST",
        )
        self.assertEqual(
            row["append_provenance"]["target_project_id"],
            "PRJ-TARGET",
        )
        self.assertEqual(
            row["append_provenance"]["iteration_num"],
            1,
        )

        backup = json.loads(
            Path(result.backup_path).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(backup, self.initial_metadata)

    def test_append_without_az_creates_empty_pending_row(self):
        package = self.write_package(with_az=False)
        append_result = self.append(package)

        result = self.materialize(package, append_result)

        self.assertEqual(result.added, 1)
        self.assertEqual(result.with_az, 0)
        self.assertEqual(result.without_az, 1)

        row = self.load_metadata()[result.items[0].plate_id]
        self.assertEqual(row["characters"], [])
        self.assertEqual(row["status"], "needs_fix")
        self.assertEqual(
            row["review_state"]["source"],
            "az_package_append",
        )
        self.assertFalse(row["gold_state"]["approved"])

    def test_second_materialization_is_idempotent(self):
        package = self.write_package(with_az=True)
        append_result = self.append(package)

        first = self.materialize(package, append_result)
        first_metadata = self.load_metadata()

        second = self.materialize(package, append_result)

        self.assertEqual(first.added, 1)
        self.assertEqual(second.added, 0)
        self.assertEqual(second.already_materialized, 1)
        self.assertIsNone(second.backup_path)
        self.assertEqual(self.load_metadata(), first_metadata)

    def test_recovery_after_restart_uses_registry_without_append_result(self):
        package = self.write_package(with_az=True)
        self.append(package)  # AZ009B commit happened.
        self.assertEqual(
            len(self.load_metadata()),
            1,
        )

        # Symulacja restartu: nie mamy już obiektu append_result.
        recovered = self.materialize(
            package,
            append_result=None,
        )

        self.assertEqual(recovered.candidates, 1)
        self.assertEqual(recovered.added, 1)
        self.assertEqual(recovered.after_count, 2)

    def test_second_az009b_noop_still_allows_recovery(self):
        package = self.write_package(with_az=True)
        first_append = self.append(package)
        self.assertEqual(first_append.appended, 1)

        # Program "padł" przed AZ009C. Po restarcie użytkownik uruchamia
        # append jeszcze raz; AZ009B jest poprawnie no-op.
        second_append = self.append(package)
        self.assertEqual(second_append.appended, 0)

        recovered = self.materialize(
            package,
            append_result=second_append,
        )

        self.assertEqual(recovered.candidates, 1)
        self.assertEqual(recovered.added, 1)

    def test_artifact_outside_active_images_is_rejected_without_write(self):
        package = self.write_package(with_az=True)
        outside = self.workspace / "_registry" / "staging"
        append_result = self.append(
            package,
            destination=outside,
        )
        before = self.load_metadata()

        with self.assertRaisesRegex(
            PZ2AppendMaterializationError,
            "images/",
        ):
            self.materialize(package, append_result)

        self.assertEqual(self.load_metadata(), before)
        self.assertFalse(
            (self.preview / "_az_append_backups").exists()
        )

    def test_large_nested_old_payload_is_preserved_and_result_matches_verified_file(self):
        metadata = self.load_metadata()
        metadata["existing_plate"]["yolo_raw_detections"] = [
            {"bbox": [1.25, 2.5, 3.75, 4.0], "scores": [0.12, 0.34], "symbol": "Ł"}
            for _ in range(1000)
        ]
        path = self.preview / "metadata.json"
        path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        before_bytes = path.read_bytes()
        package = self.write_package(with_az=True)
        result = self.materialize(package, self.append(package))
        self.assertEqual(self.load_metadata()["existing_plate"], metadata["existing_plate"])
        self.assertEqual(Path(result.backup_path).read_bytes(), before_bytes)
        self.assertEqual(result.verified_metadata, self.load_metadata())
        self.assertEqual(result.metadata_signature, (path.stat().st_mtime_ns, path.stat().st_size))
        self.assertFalse(b'\n  ' in path.read_bytes())

    def test_postcheck_restores_backup_when_written_old_record_is_changed(self):
        package = self.write_package(with_az=True)
        appended = self.append(package)
        before = (self.preview / "metadata.json").read_bytes()
        original_writer = materializer._atomic_write_json

        def corrupt_writer(path, payload):
            original_writer(path, payload)
            written = json.loads(path.read_text(encoding="utf-8"))
            written["existing_plate"]["ground_truth_text"] = "CORRUPT"
            path.write_text(json.dumps(written), encoding="utf-8")

        with patch.object(materializer, "_atomic_write_json", side_effect=corrupt_writer):
            with self.assertRaisesRegex(PZ2AppendMaterializationError, "Post-check"):
                self.materialize(package, appended)
        self.assertEqual((self.preview / "metadata.json").read_bytes(), before)

    def test_existing_row_with_same_identity_is_not_duplicated(self):
        package = self.write_package(with_az=True)
        append_result = self.append(package)
        appended = append_result.items[0]

        metadata = self.load_metadata()
        metadata["already_here"] = {
            "crop_id": appended.crop_id,
            "crop_identity_sha256": self.identity_sha,
            "characters": [{
                "character": "Z",
                "bbox": [1, 2, 3, 4],
            }],
            "status": "perfect",
            "gold_state": {
                "approved": True,
                "excluded": False,
                "candidate": True,
            },
        }
        (self.preview / "metadata.json").write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        before = self.load_metadata()

        result = self.materialize(package, append_result)

        self.assertEqual(result.added, 0)
        self.assertEqual(result.already_materialized, 1)
        self.assertEqual(self.load_metadata(), before)


if __name__ == "__main__":
    unittest.main()
