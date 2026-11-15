import sqlite3
import tempfile
import unittest
from pathlib import Path

from auto_annotation_tool.registry import RegistryDatabase, SCHEMA_VERSION
from auto_annotation_tool.registry.migrations import migrate_database


NEW_TABLES = {
    "plate_crops",
    "crop_artifacts",
    "project_crop_members",
    "iteration_crop_members",
    "az_revisions",
    "project_crop_az",
}


class AZRegistrySchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "_registry" / "alpr_registry.sqlite3"
        self.database = RegistryDatabase(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_schema_v6_creates_az_tables(self):
        self.assertEqual(SCHEMA_VERSION, 6)
        self.assertEqual(self.database.initialize(), 6)
        with self.database.read_connection() as db:
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            fk_errors = db.execute("PRAGMA foreign_key_check").fetchall()
        self.assertTrue(NEW_TABLES.issubset(tables))
        self.assertEqual(fk_errors, [])

    def test_v5_to_v6_preserves_existing_rows(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db_path))
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            self.assertEqual(migrate_database(connection, target_version=5), 5)
            connection.execute(
                "INSERT INTO projects(project_id, display_name) VALUES (?, ?)",
                ("PRJ-OLD", "Old project"),
            )
            connection.execute(
                "INSERT INTO source_images(source_image_id, canonical_sha256) VALUES (?, ?)",
                ("SRC-OLD", "a" * 64),
            )
            connection.execute(
                """
                INSERT INTO image_artifacts(
                    artifact_id, source_image_id, sha256, kind
                ) VALUES (?, ?, ?, ?)
                """,
                ("ART-OLD", "SRC-OLD", "b" * 64, "dataset_image"),
            )
            connection.commit()
            before = {
                "projects": connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0],
                "source_images": connection.execute("SELECT COUNT(*) FROM source_images").fetchone()[0],
                "image_artifacts": connection.execute("SELECT COUNT(*) FROM image_artifacts").fetchone()[0],
            }
        finally:
            connection.close()

        self.assertEqual(self.database.initialize(), 6)
        self.assertEqual(self.database.initialize(), 6)

        with self.database.read_connection() as db:
            after = {
                "projects": db.execute("SELECT COUNT(*) FROM projects").fetchone()[0],
                "source_images": db.execute("SELECT COUNT(*) FROM source_images").fetchone()[0],
                "image_artifacts": db.execute("SELECT COUNT(*) FROM image_artifacts").fetchone()[0],
            }
            version = db.execute("PRAGMA user_version").fetchone()[0]
            fk_errors = db.execute("PRAGMA foreign_key_check").fetchall()
        self.assertEqual(before, after)
        self.assertEqual(version, 6)
        self.assertEqual(fk_errors, [])

    def test_one_logical_crop_can_reference_multiple_physical_artifacts(self):
        self.database.initialize()
        with self.database.transaction() as db:
            db.execute(
                "INSERT INTO source_images(source_image_id) VALUES (?)",
                ("SRC-1",),
            )
            for artifact_id in ("ART-1", "ART-2"):
                db.execute(
                    """
                    INSERT INTO image_artifacts(
                        artifact_id, source_image_id, sha256, kind
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (artifact_id, "SRC-1", "same-file-sha", "plate_crop"),
                )
            db.execute(
                """
                INSERT INTO plate_crops(
                    crop_id, identity_sha256, identity_mode,
                    source_image_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("CROP-1", "crop-identity", "lineage_v1", "SRC-1", "2026-09-24T00:00:00Z"),
            )
            db.execute(
                "INSERT INTO crop_artifacts(crop_id, artifact_id) VALUES (?, ?)",
                ("CROP-1", "ART-1"),
            )
            db.execute(
                "INSERT INTO crop_artifacts(crop_id, artifact_id) VALUES (?, ?)",
                ("CROP-1", "ART-2"),
            )

        with self.database.read_connection() as db:
            links = db.execute(
                "SELECT COUNT(*) FROM crop_artifacts WHERE crop_id = ?",
                ("CROP-1",),
            ).fetchone()[0]
            artifacts = db.execute(
                "SELECT COUNT(*) FROM image_artifacts WHERE sha256 = ?",
                ("same-file-sha",),
            ).fetchone()[0]
        self.assertEqual(links, 2)
        self.assertEqual(artifacts, 2)

    def test_logical_crop_identity_is_unique_per_schema(self):
        self.database.initialize()
        with self.database.transaction() as db:
            db.execute(
                """
                INSERT INTO plate_crops(
                    crop_id, identity_sha256, identity_mode, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                ("CROP-1", "same-identity", "lineage_v1", "2026-09-24T00:00:00Z"),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as db:
                db.execute(
                    """
                    INSERT INTO plate_crops(
                        crop_id, identity_sha256, identity_mode, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    ("CROP-2", "same-identity", "lineage_v1", "2026-09-24T00:00:01Z"),
                )

    def test_project_delete_removes_bindings_but_keeps_global_crop_and_revision(self):
        self.database.initialize()
        with self.database.transaction() as db:
            db.execute(
                "INSERT INTO projects(project_id, display_name) VALUES (?, ?)",
                ("PRJ-1", "Project 1"),
            )
            db.execute(
                "INSERT INTO source_images(source_image_id) VALUES (?)",
                ("SRC-1",),
            )
            db.execute(
                """
                INSERT INTO image_artifacts(
                    artifact_id, source_image_id, sha256, kind
                ) VALUES (?, ?, ?, ?)
                """,
                ("ART-1", "SRC-1", "file-sha", "plate_crop"),
            )
            db.execute(
                """
                INSERT INTO plate_crops(
                    crop_id, identity_sha256, identity_mode,
                    source_image_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("CROP-1", "crop-sha", "lineage_v1", "SRC-1", "2026-09-24T00:00:00Z"),
            )
            db.execute(
                "INSERT INTO crop_artifacts(crop_id, artifact_id) VALUES (?, ?)",
                ("CROP-1", "ART-1"),
            )
            db.execute(
                """
                INSERT INTO project_crop_members(
                    project_id, crop_id, first_seen_iteration
                ) VALUES (?, ?, ?)
                """,
                ("PRJ-1", "CROP-1", 1),
            )
            db.execute(
                """
                INSERT INTO az_revisions(
                    az_revision_id, crop_id, payload_sha256, payload_json,
                    source_kind, trust_state, origin_project_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "AZ-1", "CROP-1", "payload-sha", "{}",
                    "local_manual", "local", "PRJ-1", "2026-09-24T00:00:00Z",
                ),
            )
            db.execute(
                """
                INSERT INTO project_crop_az(
                    project_id, crop_id, az_revision_id,
                    effective_status, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("PRJ-1", "CROP-1", "AZ-1", "ok", "2026-09-24T00:00:00Z"),
            )
            db.execute("DELETE FROM projects WHERE project_id = ?", ("PRJ-1",))

        with self.database.read_connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM project_crop_members").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM project_crop_az").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM plate_crops").fetchone()[0], 1)
            row = db.execute(
                "SELECT origin_project_id FROM az_revisions WHERE az_revision_id = ?",
                ("AZ-1",),
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertIsNone(row[0])

    def test_invalid_crop_artifact_foreign_key_is_rejected(self):
        self.database.initialize()
        with self.database.transaction() as db:
            db.execute(
                """
                INSERT INTO plate_crops(
                    crop_id, identity_sha256, identity_mode, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                ("CROP-1", "crop-sha", "pixels_v1", "2026-09-24T00:00:00Z"),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as db:
                db.execute(
                    "INSERT INTO crop_artifacts(crop_id, artifact_id) VALUES (?, ?)",
                    ("CROP-1", "MISSING-ART"),
                )


if __name__ == "__main__":
    unittest.main()
