import json
import tempfile
import unittest
from pathlib import Path

from auto_annotation_tool.registry import RegistryDatabase
from auto_annotation_tool.registry.az_registry import AZRegistry
from auto_annotation_tool.registry.pz1_run_registry import (
    PZ1RunRegistryError,
    register_pz1_preview_run,
)


SOURCE_SHA = "a" * 64
GEOM_A = "b" * 64
GEOM_B = "c" * 64


class PZ1RunRegistryTests(unittest.TestCase):
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

    def tearDown(self):
        self.tmp.cleanup()

    def make_run(self, name="run_001", *, second=True):
        run = self.workspace / "3_cropped_characters" / name
        images = run / "images"
        images.mkdir(parents=True)
        (images / "plate_000000.jpg").write_bytes(b"crop-A")
        metadata = {
            "plate_000000": {
                "source_image_id": "img-sha256-" + SOURCE_SHA,
                "source_file_sha256": SOURCE_SHA,
                "source_annotation_id": "plate-ann-A",
                "source_geometry_hash": GEOM_A,
                "is_square": False,
            }
        }
        if second:
            (images / "plate_000001.jpg").write_bytes(b"crop-B")
            metadata["plate_000001"] = {
                "source_image_id": "img-sha256-" + SOURCE_SHA,
                "source_file_sha256": SOURCE_SHA,
                "source_annotation_id": "plate-ann-B",
                "source_geometry_hash": GEOM_B,
                "is_square": True,
            }
        (run / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return run

    def test_free_mode_registers_whole_run_and_enriches_metadata(self):
        run = self.make_run()
        report = register_pz1_preview_run(
            self.registry,
            run,
            interpolation="lanczos4",
        )

        self.assertEqual(report.total, 2)
        self.assertEqual(report.registered, 2)
        self.assertEqual(report.new_crops, 2)
        self.assertEqual(report.new_artifacts, 2)
        self.assertEqual(report.project_memberships, 0)
        self.assertTrue(report.metadata_updated)

        metadata = json.loads(
            (run / "metadata.json").read_text(encoding="utf-8")
        )
        for row in metadata.values():
            self.assertTrue(row["crop_id"].startswith("CROP-"))
            self.assertEqual(len(row["crop_identity_sha256"]), 64)
            self.assertEqual(row["crop_identity_schema"], "alpr.crop_identity.v1")
            self.assertTrue(row["artifact_id"].startswith("ART-CROP-"))
            self.assertTrue(row["registry_source_image_id"].startswith("SRC-SHA256-"))

        with self.db.read_connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM plate_crops").fetchone()[0], 2)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM crop_artifacts").fetchone()[0], 2)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM project_crop_members").fetchone()[0], 0)

    def test_rerun_same_preview_is_idempotent(self):
        run = self.make_run()
        first = register_pz1_preview_run(self.registry, run)
        second = register_pz1_preview_run(self.registry, run)

        self.assertEqual(first.new_crops, 2)
        self.assertEqual(second.new_crops, 0)
        self.assertEqual(second.reused_crops, 2)
        self.assertEqual(second.new_artifacts, 0)
        self.assertEqual(second.reused_artifacts, 2)
        self.assertFalse(second.metadata_updated)

        with self.db.read_connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM plate_crops").fetchone()[0], 2)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM image_artifacts WHERE kind='plate_crop'").fetchone()[0], 2)

    def test_same_logical_crop_in_second_run_reuses_crop_but_adds_artifact(self):
        run1 = self.make_run("run_001", second=False)
        run2 = self.make_run("run_002", second=False)
        (run2 / "images" / "plate_000000.jpg").write_bytes(
            (run1 / "images" / "plate_000000.jpg").read_bytes()
        )

        first = register_pz1_preview_run(self.registry, run1)
        second = register_pz1_preview_run(self.registry, run2)

        self.assertEqual(first.new_crops, 1)
        self.assertEqual(second.new_crops, 0)
        self.assertEqual(second.reused_crops, 1)
        self.assertEqual(second.new_artifacts, 1)

        with self.db.read_connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM plate_crops").fetchone()[0], 1)
            crop_id = con.execute("SELECT crop_id FROM plate_crops").fetchone()[0]
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM crop_artifacts WHERE crop_id=?",
                    (crop_id,),
                ).fetchone()[0],
                2,
            )

    def test_campaign_registers_project_and_iteration_memberships(self):
        run = self.make_run()
        with self.db.transaction() as con:
            con.execute(
                "INSERT INTO projects(project_id, display_name) VALUES (?, ?)",
                ("PRJ-1", "Project"),
            )

        report = register_pz1_preview_run(
            self.registry,
            run,
            project_id="PRJ-1",
            iteration_num=4,
            source_mode="pz1",
            source_at_ref="AT-REF",
        )

        self.assertEqual(report.project_memberships, 2)
        self.assertEqual(report.iteration_memberships, 2)
        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM project_crop_members").fetchone()[0],
                2,
            )
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM iteration_crop_members WHERE iteration_num=4").fetchone()[0],
                2,
            )

    def test_missing_required_metadata_fails_before_any_db_write(self):
        run = self.make_run()
        metadata_path = run / "metadata.json"
        before = metadata_path.read_text(encoding="utf-8")
        metadata = json.loads(before)
        metadata["plate_000001"].pop("source_geometry_hash")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        bad_before = metadata_path.read_text(encoding="utf-8")

        with self.assertRaises(PZ1RunRegistryError):
            register_pz1_preview_run(self.registry, run)

        self.assertEqual(
            metadata_path.read_text(encoding="utf-8"),
            bad_before,
        )
        with self.db.read_connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM plate_crops").fetchone()[0], 0)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM image_artifacts WHERE kind='plate_crop'").fetchone()[0], 0)

    def test_missing_crop_file_fails_before_any_db_write(self):
        run = self.make_run()
        (run / "images" / "plate_000001.jpg").unlink()
        with self.assertRaises(PZ1RunRegistryError):
            register_pz1_preview_run(self.registry, run)
        with self.db.read_connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM plate_crops").fetchone()[0], 0)

    def test_missing_source_image_id_is_derived_from_source_sha(self):
        run = self.make_run(second=False)
        metadata_path = run / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["plate_000000"].pop("source_image_id")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report = register_pz1_preview_run(self.registry, run)
        self.assertEqual(report.registered, 1)
        enriched = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(len(enriched["plate_000000"]["crop_identity_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
