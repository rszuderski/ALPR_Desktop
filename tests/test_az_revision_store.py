import tempfile
import unittest
from pathlib import Path
import hashlib

from auto_annotation_tool.registry import RegistryDatabase
from auto_annotation_tool.registry.az_registry import AZRegistry
from auto_annotation_tool.registry.az_revision_store import (
    AZ_PAYLOAD_SCHEMA,
    AZRevisionStore,
    canonicalize_az_payload,
    compute_az_payload_sha256,
)
from auto_annotation_tool.registry.crop_identity import (
    build_pz1_crop_identity,
    compute_crop_identity_sha256,
)


SOURCE_SHA = "a" * 64
GEOMETRY_SHA = "b" * 64


class AZRevisionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name) / "Workspace"
        self.workspace.mkdir()

        self.db = RegistryDatabase(
            self.workspace / "_registry" / "alpr_registry.sqlite3"
        )
        self.registry = AZRegistry(
            self.db,
            workspace_dir=self.workspace,
        )
        self.registry.initialize()
        self.store = AZRevisionStore(self.db)

        self.crop_file = self.workspace / "run" / "images" / "plate_000001.jpg"
        self.crop_file.parent.mkdir(parents=True)
        self.crop_file.write_bytes(b"crop")

        identity_payload = build_pz1_crop_identity(
            source_image_id="img-sha256-" + SOURCE_SHA,
            source_annotation_id="plate-ann-1",
            source_geometry_hash=GEOMETRY_SHA,
            interpolation="lanczos4",
            output_width=256,
            output_height=64,
        )
        self.identity_sha = compute_crop_identity_sha256(identity_payload)

        artifact_sha = hashlib.sha256(self.crop_file.read_bytes()).hexdigest()
        result = self.registry.register_crop_artifact(
            crop_identity_sha256=self.identity_sha,
            identity_mode=identity_payload["identity_mode"],
            source_file_sha256=SOURCE_SHA,
            artifact_path=self.crop_file,
            artifact_sha256=artifact_sha,
            size_bytes=self.crop_file.stat().st_size,
            width=256,
            height=64,
            source_annotation_id="plate-ann-1",
            source_geometry_hash=GEOMETRY_SHA,
            created_at="2026-09-25T00:00:00+00:00",
        )
        self.crop_id = result.crop_id

        with self.db.transaction() as con:
            con.execute(
                """
                INSERT INTO projects(project_id, display_name)
                VALUES (?, ?)
                """,
                ("PRJ-A", "A"),
            )
            con.execute(
                """
                INSERT INTO projects(project_id, display_name)
                VALUES (?, ?)
                """,
                ("PRJ-B", "B"),
            )

    def tearDown(self):
        self.tmp.cleanup()

    def payload(self, char="A", x1=0.1):
        return {
            "crop_identity_sha256": self.identity_sha,
            "characters": [
                {
                    "character": char,
                    "bbox": [x1, 0.2, x1 + 0.1, 0.8],
                    "row": 0,
                    "method": "manual",
                    "source_kind": "local_manual",
                    "confidence": 1.0,
                }
            ],
            "layout": {"kind": "1R", "confirmed": True},
            "gold_state": {
                "approved": True,
                "excluded": False,
                "candidate": True,
            },
            "status": "perfect",
        }

    def test_canonical_payload_is_stable_for_character_order(self):
        a = self.payload()
        a["characters"].append(
            {
                "character": "B",
                "bbox": [0.4, 0.2, 0.5, 0.8],
                "row": 0,
                "method": "manual",
                "source_kind": "local_manual",
                "confidence": 1.0,
            }
        )
        b = dict(a)
        b["characters"] = list(reversed(a["characters"]))

        ca = canonicalize_az_payload(a)
        cb = canonicalize_az_payload(b)

        self.assertEqual(ca, cb)
        self.assertEqual(ca["schema"], AZ_PAYLOAD_SCHEMA)
        self.assertEqual(
            compute_az_payload_sha256(a),
            compute_az_payload_sha256(b),
        )

    def test_save_revision_and_bind_project(self):
        result = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload(),
            source_kind="local_manual",
            trust_state="local",
            source_status="perfect",
            origin_project_id="PRJ-A",
            origin_iteration=1,
            bind_project_id="PRJ-A",
            effective_status="ok",
            created_at="2026-09-25T00:01:00+00:00",
        )

        self.assertTrue(result.created)
        current = self.store.get_project_az(
            project_id="PRJ-A",
            crop_id=self.crop_id,
        )
        self.assertIsNotNone(current)
        self.assertEqual(current["az_revision_id"], result.az_revision_id)
        self.assertEqual(current["effective_status"], "ok")
        self.assertEqual(
            current["payload"]["characters"][0]["character"],
            "A",
        )

    def test_identical_payload_is_deduplicated(self):
        first = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload(),
            source_kind="local_manual",
            trust_state="local",
        )
        second = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload(),
            source_kind="cvat_manual",
            trust_state="external_pending_review",
        )

        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(first.az_revision_id, second.az_revision_id)

        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM az_revisions").fetchone()[0],
                1,
            )

    def test_new_project_revision_auto_parents_current_binding(self):
        first = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("A"),
            source_kind="local_manual",
            trust_state="local",
            bind_project_id="PRJ-A",
            effective_status="ok",
        )
        second = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("B"),
            source_kind="local_manual",
            trust_state="local",
            bind_project_id="PRJ-A",
            effective_status="ready",
        )

        self.assertTrue(second.created)
        self.assertEqual(second.parent_revision_id, first.az_revision_id)

    def test_free_mode_revision_creates_no_project_binding(self):
        result = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload(),
            source_kind="local_manual",
            trust_state="free_mode",
        )
        self.assertTrue(result.created)
        self.assertIsNone(result.project_id)

        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM project_crop_az").fetchone()[0],
                0,
            )

    def test_same_revision_can_be_bound_to_second_project_with_other_status(self):
        first = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload(),
            source_kind="local_manual",
            trust_state="local",
            bind_project_id="PRJ-A",
            effective_status="ok",
        )
        second = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload(),
            source_kind="project_import",
            trust_state="external_pending_review",
            bind_project_id="PRJ-B",
            effective_status="imported_pending_review",
        )

        self.assertEqual(first.az_revision_id, second.az_revision_id)
        project_b = self.store.get_project_az(
            project_id="PRJ-B",
            crop_id=self.crop_id,
        )
        self.assertEqual(
            project_b["effective_status"],
            "imported_pending_review",
        )

    def test_parent_from_other_crop_is_rejected(self):
        second_file = self.workspace / "run" / "images" / "plate_000002.jpg"
        second_file.write_bytes(b"crop2")
        second_identity_payload = build_pz1_crop_identity(
            source_image_id="img-sha256-" + ("c" * 64),
            source_annotation_id="plate-ann-2",
            source_geometry_hash="d" * 64,
            interpolation="lanczos4",
            output_width=256,
            output_height=64,
        )
        second_identity = compute_crop_identity_sha256(second_identity_payload)
        second_crop = self.registry.register_crop_artifact(
            crop_identity_sha256=second_identity,
            identity_mode=second_identity_payload["identity_mode"],
            source_file_sha256="c" * 64,
            artifact_path=second_file,
            artifact_sha256=hashlib.sha256(
                second_file.read_bytes()
            ).hexdigest(),
            size_bytes=second_file.stat().st_size,
            width=256,
            height=64,
            source_annotation_id="plate-ann-2",
            source_geometry_hash="d" * 64,
        )
        parent = self.store.save_revision(
            crop_id=second_crop.crop_id,
            payload={
                **self.payload(),
                "crop_identity_sha256": second_identity,
            },
            source_kind="manual",
            trust_state="local",
        )

        with self.assertRaises(ValueError):
            self.store.save_revision(
                crop_id=self.crop_id,
                payload=self.payload("B"),
                source_kind="manual",
                trust_state="local",
                parent_revision_id=parent.az_revision_id,
            )

    def test_payload_crop_identity_must_match_registry(self):
        bad = self.payload()
        bad["crop_identity_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            self.store.save_revision(
                crop_id=self.crop_id,
                payload=bad,
                source_kind="manual",
                trust_state="local",
            )

    def test_invalid_normalized_bbox_is_rejected(self):
        bad = self.payload()
        bad["characters"][0]["bbox"] = [10, 0, 20, 30]
        with self.assertRaises(ValueError):
            canonicalize_az_payload(bad)

    def test_get_revision_returns_payload(self):
        result = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload(),
            source_kind="manual",
            trust_state="local",
        )
        loaded = self.store.get_revision(result.az_revision_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["payload_sha256"], result.payload_sha256)
        self.assertEqual(
            loaded["payload"]["crop_identity_sha256"],
            self.identity_sha,
        )

    def test_free_reuse_returns_latest_crop_revision(self):
        first = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("A"),
            source_kind="local_manual",
            trust_state="free_mode",
            created_at="2026-09-25T00:01:00+00:00",
        )
        second = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("B"),
            source_kind="local_manual",
            trust_state="free_mode",
            created_at="2026-09-25T00:02:00+00:00",
        )

        resolved = self.store.resolve_reusable_az(crop_id=self.crop_id)

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["az_revision_id"], second.az_revision_id)
        self.assertNotEqual(resolved["az_revision_id"], first.az_revision_id)
        self.assertEqual(resolved["payload"]["characters"][0]["character"], "B")
        self.assertIsNone(resolved["project_id"])
        self.assertIsNone(resolved["effective_status"])

    def test_project_reuse_prefers_bound_revision_over_newer_global_revision(self):
        project_revision = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("A"),
            source_kind="local_manual",
            trust_state="local",
            origin_project_id="PRJ-A",
            origin_iteration=1,
            bind_project_id="PRJ-A",
            effective_status="approved",
            created_at="2026-09-25T00:01:00+00:00",
        )
        self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("B"),
            source_kind="local_manual",
            trust_state="free_mode",
            created_at="2026-09-25T00:02:00+00:00",
        )

        resolved = self.store.resolve_reusable_az(
            crop_id=self.crop_id,
            project_id="PRJ-A",
        )

        self.assertIsNotNone(resolved)
        self.assertEqual(
            resolved["az_revision_id"],
            project_revision.az_revision_id,
        )
        self.assertEqual(
            resolved["payload"]["characters"][0]["character"],
            "A",
        )
        self.assertEqual(resolved["project_id"], "PRJ-A")
        self.assertEqual(resolved["effective_status"], "approved")

    def test_project_reuse_does_not_fall_back_to_global_revision(self):
        self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("A"),
            source_kind="local_manual",
            trust_state="free_mode",
            created_at="2026-09-25T00:01:00+00:00",
        )

        resolved = self.store.resolve_reusable_az(
            crop_id=self.crop_id,
            project_id="PRJ-B",
        )

        self.assertIsNone(resolved)

    def test_free_reuse_returns_none_when_crop_has_no_az_revision(self):
        second_file = self.workspace / "run" / "images" / "plate_000099.jpg"
        second_file.write_bytes(b"crop-without-az")
        second_identity_payload = build_pz1_crop_identity(
            source_image_id="img-sha256-" + ("e" * 64),
            source_annotation_id="plate-ann-99",
            source_geometry_hash="f" * 64,
            interpolation="lanczos4",
            output_width=256,
            output_height=64,
        )
        second_identity = compute_crop_identity_sha256(second_identity_payload)
        second_crop = self.registry.register_crop_artifact(
            crop_identity_sha256=second_identity,
            identity_mode=second_identity_payload["identity_mode"],
            source_file_sha256="e" * 64,
            artifact_path=second_file,
            artifact_sha256=hashlib.sha256(second_file.read_bytes()).hexdigest(),
            size_bytes=second_file.stat().st_size,
            width=256,
            height=64,
            source_annotation_id="plate-ann-99",
            source_geometry_hash="f" * 64,
        )

        resolved = self.store.resolve_reusable_az(
            crop_id=second_crop.crop_id,
        )

        self.assertIsNone(resolved)



if __name__ == "__main__":
    unittest.main()
