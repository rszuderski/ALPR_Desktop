import hashlib
import tempfile
import unittest
from pathlib import Path

from auto_annotation_tool.registry import RegistryDatabase
from auto_annotation_tool.registry.az_registry import AZRegistry
from auto_annotation_tool.registry.az_revision_store import AZRevisionStore
from auto_annotation_tool.registry.crop_identity import (
    build_pz1_crop_identity,
    compute_crop_identity_sha256,
)
from auto_annotation_tool.registry.pz2_az_reuse import (
    apply_reusable_az_to_metadata,
    pz2_az_reuse_protection_reason,
)


SOURCE_SHA = "a" * 64
GEOMETRY_SHA = "b" * 64


class PZ2AZReuseTests(unittest.TestCase):
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

        self.crop_file = (
            self.workspace / "3_cropped_characters" / "run_001"
            / "images" / "plate_000001.jpg"
        )
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

        registered = self.registry.register_crop_artifact(
            crop_identity_sha256=self.identity_sha,
            identity_mode=identity_payload["identity_mode"],
            source_file_sha256=SOURCE_SHA,
            artifact_path=self.crop_file,
            artifact_sha256=hashlib.sha256(
                self.crop_file.read_bytes()
            ).hexdigest(),
            size_bytes=self.crop_file.stat().st_size,
            width=256,
            height=64,
            source_annotation_id="plate-ann-1",
            source_geometry_hash=GEOMETRY_SHA,
        )
        self.crop_id = registered.crop_id

        with self.db.transaction() as con:
            con.execute(
                "INSERT INTO projects(project_id, display_name) VALUES (?, ?)",
                ("PRJ-A", "A"),
            )
            con.execute(
                "INSERT INTO projects(project_id, display_name) VALUES (?, ?)",
                ("PRJ-B", "B"),
            )

    def tearDown(self):
        self.tmp.cleanup()

    def payload(self, char="A", *, excluded=False, approved=True):
        return {
            "crop_identity_sha256": self.identity_sha,
            "characters": [
                {
                    "character": char,
                    "bbox": [0.1, 0.2, 0.2, 0.8],
                    "row": 1,
                    "method": "manual",
                    "source_kind": "local_manual",
                    "confidence": 1.0,
                }
            ],
            "layout": {"kind": "single_row", "confirmed": True},
            "gold_state": {
                "approved": approved,
                "excluded": excluded,
                "candidate": bool(approved and not excluded),
            },
            "status": "perfect" if approved and not excluded else "needs_fix",
            "expected_text": char,
        }

    def fresh_metadata(self):
        return {
            "crop_id": self.crop_id,
            "crop_identity_sha256": self.identity_sha,
            "status": "needs_fix",
            "characters": [],
            "gold_state": {
                "approved": False,
                "excluded": False,
                "candidate": False,
            },
            "source_info": {
                "bucket": "auto_preview",
                "origin": "pz2_detect",
                "last_modified_by": "system",
            },
        }

    def test_free_mode_applies_latest_revision_to_fresh_record(self):
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
        metadata = {"plate_000001": self.fresh_metadata()}

        summary = apply_reusable_az_to_metadata(
            self.registry,
            metadata,
        )

        self.assertEqual(summary.applied, 1)
        self.assertEqual(
            metadata["plate_000001"]["characters"][0]["character"],
            "B",
        )
        self.assertEqual(
            metadata["plate_000001"]["az_reuse"]["az_revision_id"],
            second.az_revision_id,
        )
        self.assertNotEqual(
            metadata["plate_000001"]["az_reuse"]["az_revision_id"],
            first.az_revision_id,
        )

    def test_automatic_raw_review_does_not_block_reuse(self):
        saved = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("A"),
            source_kind="local_manual",
            trust_state="free_mode",
        )
        row = self.fresh_metadata()
        row["review_state"] = {
            "status": "in_progress",
            "source": "raw_detection",
            "human_edited": False,
            "approved_at": None,
        }
        row["fusion_strategy"] = "review_from_raw"
        metadata = {"plate_000001": row}

        summary = apply_reusable_az_to_metadata(self.registry, metadata)

        self.assertEqual(summary.applied, 1)
        self.assertEqual(
            metadata["plate_000001"]["az_reuse"]["az_revision_id"],
            saved.az_revision_id,
        )
        self.assertNotIn("review_state", metadata["plate_000001"])

    def test_human_edited_review_is_protected(self):
        saved = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("A"),
            source_kind="local_manual",
            trust_state="free_mode",
        )
        row = self.fresh_metadata()
        row["characters"] = [{"character": "X", "bbox": [1, 2, 3, 4]}]
        row["review_state"] = {
            "status": "in_progress",
            "source": "raw_detection",
            "human_edited": True,
        }
        row["fusion_strategy"] = "manual_correction"
        metadata = {"plate_000001": row}

        summary = apply_reusable_az_to_metadata(self.registry, metadata)

        self.assertEqual(summary.protected, 1)
        self.assertEqual(metadata["plate_000001"]["characters"][0]["character"], "X")
        self.assertNotIn("az_reuse", metadata["plate_000001"])
        self.assertTrue(saved.az_revision_id)

    def test_local_n_decision_is_protected(self):
        saved = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("A"),
            source_kind="local_manual",
            trust_state="free_mode",
        )
        row = self.fresh_metadata()
        row["gold_state"]["excluded"] = True
        metadata = {"plate_000001": row}

        summary = apply_reusable_az_to_metadata(self.registry, metadata)

        self.assertEqual(summary.protected, 1)
        self.assertTrue(metadata["plate_000001"]["gold_state"]["excluded"])
        self.assertTrue(saved.az_revision_id)

    def test_existing_az_reuse_can_advance_to_newer_revision(self):
        first = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("A"),
            source_kind="local_manual",
            trust_state="free_mode",
            created_at="2026-09-25T00:01:00+00:00",
        )
        metadata = {"plate_000001": self.fresh_metadata()}
        apply_reusable_az_to_metadata(self.registry, metadata)
        self.assertEqual(
            metadata["plate_000001"]["az_reuse"]["az_revision_id"],
            first.az_revision_id,
        )

        second = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("B"),
            source_kind="local_manual",
            trust_state="free_mode",
            created_at="2026-09-25T00:02:00+00:00",
        )

        summary = apply_reusable_az_to_metadata(self.registry, metadata)

        self.assertEqual(summary.applied, 1)
        self.assertEqual(
            metadata["plate_000001"]["az_reuse"]["az_revision_id"],
            second.az_revision_id,
        )
        self.assertEqual(
            metadata["plate_000001"]["characters"][0]["character"],
            "B",
        )

    def test_same_az_revision_is_not_reapplied(self):
        saved = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("A"),
            source_kind="local_manual",
            trust_state="free_mode",
        )
        metadata = {"plate_000001": self.fresh_metadata()}
        apply_reusable_az_to_metadata(self.registry, metadata)

        summary = apply_reusable_az_to_metadata(self.registry, metadata)

        self.assertEqual(summary.already_current, 1)
        self.assertEqual(summary.applied, 0)
        self.assertEqual(
            metadata["plate_000001"]["az_reuse"]["az_revision_id"],
            saved.az_revision_id,
        )

    def test_campaign_uses_only_project_binding(self):
        project_revision = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("P"),
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
            payload=self.payload("G"),
            source_kind="local_manual",
            trust_state="free_mode",
            created_at="2026-09-25T00:02:00+00:00",
        )
        metadata = {"plate_000001": self.fresh_metadata()}

        summary = apply_reusable_az_to_metadata(
            self.registry,
            metadata,
            project_id="PRJ-A",
        )

        self.assertEqual(summary.applied, 1)
        self.assertEqual(
            metadata["plate_000001"]["characters"][0]["character"],
            "P",
        )
        self.assertEqual(
            metadata["plate_000001"]["az_reuse"]["az_revision_id"],
            project_revision.az_revision_id,
        )

    def test_campaign_without_binding_does_not_fall_back_to_global(self):
        self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("G"),
            source_kind="local_manual",
            trust_state="free_mode",
        )
        metadata = {"plate_000001": self.fresh_metadata()}

        summary = apply_reusable_az_to_metadata(
            self.registry,
            metadata,
            project_id="PRJ-B",
        )

        self.assertEqual(summary.applied, 0)
        self.assertEqual(summary.missing_revision, 1)
        self.assertNotIn("az_reuse", metadata["plate_000001"])


class PZ2AZReusePolicyTests(unittest.TestCase):
    def test_manual_source_is_protected(self):
        data = {
            "source_info": {
                "bucket": "local_manual",
                "origin": "preview_editor",
                "last_modified_by": "human",
            }
        }
        reason = pz2_az_reuse_protection_reason(
            data,
            {"az_revision_id": "AZR-X"},
        )
        self.assertEqual(reason, "manual_source")


def test_registry_az_reuse_after_load_persists_when_applied(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from auto_annotation_tool.gui import z3_preview_ui

    meta_path = tmp_path / "metadata.json"
    meta_path.write_text('{"plate": {}}', encoding="utf-8")

    host = SimpleNamespace(
        preview_metadata={"plate": {}},
        _loaded_meta_path=meta_path,
        _loaded_meta_mtime=None,
        _atomic_write_json=Mock(
            side_effect=lambda path, payload: Path(path).write_text(
                __import__("json").dumps(payload),
                encoding="utf-8",
            )
        ),
    )

    with patch.object(
        z3_preview_ui,
        "_apply_registry_az_reuse_best_effort",
        return_value={"ok": True, "applied": 1},
    ) as apply_reuse:
        result = z3_preview_ui._apply_registry_az_reuse_after_load(
            host,
            meta_path,
        )

    self_payload = __import__("json").loads(
        meta_path.read_text(encoding="utf-8")
    )
    assert result["applied"] == 1
    assert apply_reuse.call_count == 1
    host._atomic_write_json.assert_called_once_with(
        meta_path,
        host.preview_metadata,
    )
    assert self_payload == host.preview_metadata
    assert host._loaded_meta_path == meta_path
    assert host._loaded_meta_mtime is not None


def test_registry_az_reuse_after_load_does_not_write_when_no_change(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from auto_annotation_tool.gui import z3_preview_ui

    meta_path = tmp_path / "metadata.json"
    meta_path.write_text('{"plate": {}}', encoding="utf-8")

    host = SimpleNamespace(
        preview_metadata={"plate": {}},
        _loaded_meta_path=meta_path,
        _loaded_meta_mtime=meta_path.stat().st_mtime,
        _atomic_write_json=Mock(),
    )

    with patch.object(
        z3_preview_ui,
        "_apply_registry_az_reuse_best_effort",
        return_value={"ok": True, "applied": 0},
    ):
        result = z3_preview_ui._apply_registry_az_reuse_after_load(
            host,
            meta_path,
        )

    assert result["applied"] == 0
    host._atomic_write_json.assert_not_called()


    def test_campaign_import_pending_review_materializes_for_review(self):
        revision = self.store.save_revision(
            crop_id=self.crop_id,
            payload=self.payload("I"),
            source_kind="local_manual",
            trust_state="local_manual",
            origin_project_id="PRJ-A",
            origin_iteration=1,
            bind_project_id="PRJ-B",
            effective_status="imported_pending_review",
        )
        metadata = {"plate_000001": self.fresh_metadata()}

        summary = apply_reusable_az_to_metadata(
            self.registry,
            metadata,
            project_id="PRJ-B",
        )

        row = metadata["plate_000001"]
        self.assertEqual(summary.applied, 1)
        self.assertEqual(row["characters"][0]["character"], "I")
        self.assertEqual(row["status"], "needs_fix")
        self.assertFalse(row["gold_state"]["approved"])
        self.assertFalse(row["gold_state"]["candidate"])
        self.assertFalse(row["gold_state"]["excluded"])
        self.assertEqual(
            row["az_reuse"]["az_revision_id"],
            revision.az_revision_id,
        )
        self.assertEqual(
            row["az_reuse"]["effective_status"],
            "imported_pending_review",
        )
        self.assertTrue(row["az_reuse"]["requires_review"])
        self.assertEqual(
            row["fusion_details"]["source"],
            "az_project_import",
        )


if __name__ == "__main__":
    unittest.main()
