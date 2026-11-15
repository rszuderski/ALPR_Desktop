import tempfile
import unittest
from pathlib import Path

from auto_annotation_tool.registry import RegistryDatabase
from auto_annotation_tool.registry.az_registry import (
    AZRegistry,
    PLATE_CROP_KIND,
    file_sha256,
    source_image_id_from_sha256,
)
from auto_annotation_tool.registry.crop_identity import (
    build_pz1_crop_identity,
    compute_crop_identity_sha256,
)


SOURCE_SHA = "a" * 64
GEOMETRY_SHA = "b" * 64


def identity():
    payload = build_pz1_crop_identity(
        source_image_id="img-sha256-" + SOURCE_SHA,
        source_annotation_id="plate-ann-1",
        source_geometry_hash=GEOMETRY_SHA,
        interpolation="lanczos4",
        output_width=256,
        output_height=64,
    )
    return payload, compute_crop_identity_sha256(payload)


class AZRegistryRegistrationTests(unittest.TestCase):
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

        self.crop1 = self.workspace / "run1" / "images" / "plate_000001.jpg"
        self.crop1.parent.mkdir(parents=True)
        self.crop1.write_bytes(b"same-crop-file")

    def tearDown(self):
        self.tmp.cleanup()

    def register(self, path=None, **overrides):
        payload, digest = identity()
        crop_path = Path(path or self.crop1)
        values = dict(
            crop_identity_sha256=digest,
            identity_mode=payload["identity_mode"],
            source_file_sha256=SOURCE_SHA,
            artifact_path=crop_path,
            artifact_sha256=file_sha256(crop_path),
            size_bytes=crop_path.stat().st_size,
            width=256,
            height=64,
            source_annotation_id="plate-ann-1",
            source_geometry_hash=GEOMETRY_SHA,
            project_id=None,
            iteration_num=None,
            source_mode=None,
            source_plate_key="plate_000001",
            source_at_ref=None,
            created_at="2026-09-25T00:00:00+00:00",
        )
        values.update(overrides)
        return self.registry.register_crop_artifact(**values)

    def test_register_creates_source_artifact_crop_and_link(self):
        result = self.register()

        self.assertTrue(result.crop_created)
        self.assertTrue(result.artifact_created)
        self.assertEqual(
            result.source_image_id,
            source_image_id_from_sha256(SOURCE_SHA),
        )

        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM source_images").fetchone()[0],
                1,
            )
            artifact = con.execute(
                """
                SELECT kind, relative_path, external_path
                FROM image_artifacts
                WHERE artifact_id = ?
                """,
                (result.artifact_id,),
            ).fetchone()
            self.assertEqual(artifact["kind"], PLATE_CROP_KIND)
            self.assertIsNotNone(artifact["relative_path"])
            self.assertIsNone(artifact["external_path"])

            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM plate_crops").fetchone()[0],
                1,
            )
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM crop_artifacts").fetchone()[0],
                1,
            )

    def test_same_identity_and_same_path_is_idempotent(self):
        first = self.register()
        second = self.register(
            created_at="2026-09-25T00:01:00+00:00"
        )

        self.assertEqual(first.crop_id, second.crop_id)
        self.assertEqual(first.artifact_id, second.artifact_id)
        self.assertFalse(second.crop_created)
        self.assertFalse(second.artifact_created)

        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM plate_crops").fetchone()[0],
                1,
            )
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM image_artifacts WHERE kind = ?",
                    (PLATE_CROP_KIND,),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM crop_artifacts").fetchone()[0],
                1,
            )

    def test_same_logical_crop_can_have_second_physical_artifact(self):
        second_path = (
            self.workspace / "run2" / "images" / "plate_000777.jpg"
        )
        second_path.parent.mkdir(parents=True)
        second_path.write_bytes(self.crop1.read_bytes())

        first = self.register()
        second = self.register(path=second_path)

        self.assertEqual(first.crop_id, second.crop_id)
        self.assertNotEqual(first.artifact_id, second.artifact_id)
        self.assertFalse(second.crop_created)
        self.assertTrue(second.artifact_created)

        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM crop_artifacts WHERE crop_id = ?",
                    (first.crop_id,),
                ).fetchone()[0],
                2,
            )
            primary = con.execute(
                """
                SELECT COUNT(*)
                FROM crop_artifacts
                WHERE crop_id = ? AND is_primary = 1
                """,
                (first.crop_id,),
            ).fetchone()[0]
            self.assertEqual(primary, 1)

    def test_existing_source_row_with_same_canonical_sha_is_reused(self):
        with self.db.transaction() as con:
            con.execute(
                """
                INSERT INTO source_images(
                    source_image_id,
                    canonical_sha256,
                    origin_status
                ) VALUES (?, ?, ?)
                """,
                ("LEGACY-SRC-ID", SOURCE_SHA, "legacy_partial"),
            )

        result = self.register()
        self.assertEqual(result.source_image_id, "LEGACY-SRC-ID")

        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM source_images").fetchone()[0],
                1,
            )

    def test_campaign_membership_and_iteration_are_written(self):
        with self.db.transaction() as con:
            con.execute(
                """
                INSERT INTO projects(project_id, display_name)
                VALUES (?, ?)
                """,
                ("PRJ-1", "Project"),
            )

        result = self.register(
            project_id="PRJ-1",
            iteration_num=3,
            source_mode="pz1",
            source_at_ref="AT-REF",
        )

        self.assertTrue(result.project_attached)
        self.assertTrue(result.iteration_attached)

        with self.db.read_connection() as con:
            project_row = con.execute(
                """
                SELECT first_seen_iteration, last_seen_iteration, source_mode
                FROM project_crop_members
                WHERE project_id = ? AND crop_id = ?
                """,
                ("PRJ-1", result.crop_id),
            ).fetchone()
            self.assertEqual(project_row["first_seen_iteration"], 3)
            self.assertEqual(project_row["last_seen_iteration"], 3)
            self.assertEqual(project_row["source_mode"], "pz1")

            iteration_row = con.execute(
                """
                SELECT artifact_id, source_at_ref
                FROM iteration_crop_members
                WHERE project_id = ?
                  AND iteration_num = ?
                  AND crop_id = ?
                """,
                ("PRJ-1", 3, result.crop_id),
            ).fetchone()
            self.assertEqual(
                iteration_row["artifact_id"],
                result.artifact_id,
            )
            self.assertEqual(iteration_row["source_at_ref"], "AT-REF")

    def test_free_mode_does_not_create_project_membership(self):
        result = self.register()
        self.assertFalse(result.project_attached)
        self.assertFalse(result.iteration_attached)

        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM project_crop_members"
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM iteration_crop_members"
                ).fetchone()[0],
                0,
            )

    def test_invalid_project_rolls_back_whole_registration(self):
        with self.assertRaises(Exception):
            self.register(
                project_id="MISSING-PROJECT",
                iteration_num=1,
            )

        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM source_images").fetchone()[0],
                0,
            )
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM image_artifacts WHERE kind = ?",
                    (PLATE_CROP_KIND,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM plate_crops").fetchone()[0],
                0,
            )

    def test_iteration_without_project_is_rejected_before_write(self):
        with self.assertRaises(ValueError):
            self.register(iteration_num=1)

        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM plate_crops").fetchone()[0],
                0,
            )

    def test_existing_crop_identity_with_conflicting_lineage_is_rejected(self):
        self.register()
        with self.assertRaises(RuntimeError):
            self.register(
                source_geometry_hash="c" * 64,
            )

        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM plate_crops").fetchone()[0],
                1,
            )


if __name__ == "__main__":
    unittest.main()
