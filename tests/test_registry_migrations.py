import sqlite3
import tempfile
import unittest
from pathlib import Path

from auto_annotation_tool.registry import RegistryDatabase, SCHEMA_VERSION
from auto_annotation_tool.registry.schema import SCHEMA_V1_STATEMENTS


class RegistryMigrationTests(unittest.TestCase):
    def test_existing_v1_database_migrates_to_v2_without_losing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "alpr_registry.sqlite3"
            connection = sqlite3.connect(str(db_path))
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                for statement in SCHEMA_V1_STATEMENTS:
                    connection.execute(statement)
                connection.execute("PRAGMA user_version = 1")
                connection.execute(
                    """
                    INSERT INTO projects (
                        project_id,
                        campaign_key,
                        folder_name,
                        display_name
                    ) VALUES (?, ?, ?, ?)
                    """,
                    ("PRJ-OLD", "old", "old_ABC123", "Old"),
                )
                connection.execute(
                    """
                    INSERT INTO datasets (
                        dataset_id,
                        owner_project_id,
                        target,
                        name,
                        purpose,
                        provenance_status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "DS-MT-OLD",
                        "PRJ-OLD",
                        "plate",
                        "old_dataset",
                        "training",
                        "partial",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            database = RegistryDatabase(db_path)
            version = database.initialize()
            self.assertEqual(version, SCHEMA_VERSION)

            with database.read_connection() as connection:
                project_count = connection.execute(
                    "SELECT COUNT(*) FROM projects"
                ).fetchone()[0]
                dataset_count = connection.execute(
                    "SELECT COUNT(*) FROM datasets"
                ).fetchone()[0]
                locations_exists = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM sqlite_master
                    WHERE type='table' AND name='dataset_locations'
                    """
                ).fetchone()[0]

            self.assertEqual(project_count, 1)
            self.assertEqual(dataset_count, 1)
            self.assertEqual(locations_exists, 1)



    def test_v4_adds_reverse_reference_indexes_without_changing_rows(self):
        from auto_annotation_tool.registry.migrations import migrate_database
        with sqlite3.connect(":memory:") as db:
            db.execute("PRAGMA foreign_keys = ON")
            migrate_database(db, target_version=4)
            db.execute("INSERT INTO source_images(source_image_id) VALUES ('source')")
            db.execute(
                """INSERT INTO image_artifacts(artifact_id, source_image_id, sha256)
                   VALUES ('parent', 'source', 'hash')"""
            )
            db.execute(
                """INSERT INTO image_artifacts(artifact_id, source_image_id, sha256, derived_from_artifact_id)
                   VALUES ('child', 'source', 'hash', 'parent')"""
            )
            before = db.execute("SELECT * FROM image_artifacts ORDER BY artifact_id").fetchall()
            db.commit()
            self.assertEqual(migrate_database(db), SCHEMA_VERSION)
            self.assertEqual(migrate_database(db), SCHEMA_VERSION)
            self.assertEqual(before, db.execute("SELECT * FROM image_artifacts ORDER BY artifact_id").fetchall())
            for table, column in (
                ("image_artifacts", "derived_from_artifact_id"),
                ("evaluation_track_members", "source_artifact_id"),
                ("evaluation_track_members", "track_artifact_id"),
                ("dataset_members", "artifact_id"),
            ):
                with self.subTest(table=table, column=column):
                    plan = db.execute(
                        f"EXPLAIN QUERY PLAN SELECT 1 FROM {table} WHERE {column} = ?", ("parent",)
                    ).fetchall()
                    self.assertTrue(any("SEARCH" in row[3] and "INDEX" in row[3] for row in plan), plan)
            self.assertFalse(db.execute("PRAGMA foreign_key_check").fetchall())


if __name__ == "__main__":
    unittest.main()
