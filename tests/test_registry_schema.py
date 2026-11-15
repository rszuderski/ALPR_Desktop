import sqlite3
import tempfile
import unittest
from pathlib import Path

from auto_annotation_tool.registry import RegistryDatabase, SCHEMA_VERSION


EXPECTED_TABLES = {
    "projects",
    "source_images",
    "image_artifacts",
    "datasets",
    "dataset_locations",
    "dataset_members",
    "training_runs",
    "models",
    "model_locations",
    "evaluation_tracks",
    "evaluation_track_members",
    "reservations",
    "experiments",
    "experiment_participants",
    "experiment_overlap_items",
    "experiment_results",
}


class RegistrySchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "_registry" / "alpr_registry.sqlite3"
        self.database = RegistryDatabase(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialize_creates_versioned_schema(self):
        version = self.database.initialize()
        self.assertEqual(version, SCHEMA_VERSION)
        self.assertTrue(self.db_path.exists())

        with self.database.read_connection() as connection:
            stored_version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
            table_rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()

        tables = {row[0] for row in table_rows}
        self.assertEqual(stored_version, SCHEMA_VERSION)
        self.assertTrue(EXPECTED_TABLES.issubset(tables))

    def test_connections_enable_foreign_keys_and_wal(self):
        self.database.initialize()
        with self.database.read_connection() as connection:
            foreign_keys = connection.execute(
                "PRAGMA foreign_keys"
            ).fetchone()[0]
            journal_mode = connection.execute(
                "PRAGMA journal_mode"
            ).fetchone()[0]

        self.assertEqual(foreign_keys, 1)
        self.assertEqual(str(journal_mode).lower(), "wal")

    def test_foreign_key_violation_is_rejected(self):
        self.database.initialize()
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO datasets (
                        dataset_id,
                        owner_project_id,
                        target,
                        purpose,
                        provenance_status
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    ("DS-MT-TEST", "missing-project", "plate", "training", "complete"),
                )

    def test_transaction_rolls_back_on_failure(self):
        self.database.initialize()
        with self.assertRaises(RuntimeError):
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO projects(project_id, display_name) VALUES (?, ?)",
                    ("project-a", "Project A"),
                )
                raise RuntimeError("stop")

        with self.database.read_connection() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM projects WHERE project_id = ?",
                ("project-a",),
            ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_initialize_is_idempotent(self):
        first = self.database.initialize()
        second = self.database.initialize()
        self.assertEqual(first, SCHEMA_VERSION)
        self.assertEqual(second, SCHEMA_VERSION)

    def test_model_content_can_have_multiple_locations(self):
        self.database.initialize()
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO models (
                    model_id,
                    target,
                    yolo_scale,
                    sha256,
                    provenance_status
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("MT-1", "plate", "n", "abc123", "complete"),
            )
            connection.execute(
                """
                INSERT INTO model_locations (
                    model_id,
                    location_key,
                    relative_path,
                    is_primary
                ) VALUES (?, ?, ?, ?)
                """,
                ("MT-1", "global", "6_models/trained/plates/a.pt", 1),
            )
            connection.execute(
                """
                INSERT INTO model_locations (
                    model_id,
                    location_key,
                    relative_path,
                    is_primary
                ) VALUES (?, ?, ?, ?)
                """,
                ("MT-1", "project-copy", "9_projects/p/6_models/trained/plates/a.pt", 0),
            )

        with self.database.read_connection() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM model_locations WHERE model_id = ?",
                ("MT-1",),
            ).fetchone()[0]
        self.assertEqual(count, 2)

    def test_backup_preserves_schema_and_data(self):
        self.database.initialize()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO projects(project_id, display_name) VALUES (?, ?)",
                ("project-a", "Project A"),
            )

        backup_path = self.root / "backups" / "registry.sqlite3"
        self.database.backup_to(backup_path)
        self.assertTrue(backup_path.exists())

        connection = sqlite3.connect(str(backup_path))
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            count = connection.execute(
                "SELECT COUNT(*) FROM projects WHERE project_id = ?",
                ("project-a",),
            ).fetchone()[0]
        finally:
            connection.close()

        self.assertEqual(version, SCHEMA_VERSION)
        self.assertEqual(count, 1)

    def test_integrity_check_reports_ok(self):
        self.database.initialize()
        self.assertEqual(self.database.integrity_check().lower(), "ok")


if __name__ == "__main__":
    unittest.main()
