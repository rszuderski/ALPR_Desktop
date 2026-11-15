import hashlib
import tempfile
import unittest
from pathlib import Path

from auto_annotation_tool.registry import RegistryDatabase
from auto_annotation_tool.registry.az_campaign_resource import (
    summarize_project_az_resource,
)
from auto_annotation_tool.registry.az_registry import AZRegistry
from auto_annotation_tool.registry.az_revision_store import AZRevisionStore
from auto_annotation_tool.registry.crop_identity import (
    build_pz1_crop_identity,
    compute_crop_identity_sha256,
)


class AZCampaignResourceTests(unittest.TestCase):
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

        with self.db.transaction() as con:
            con.execute(
                """
                INSERT INTO projects(project_id, display_name)
                VALUES (?, ?)
                """,
                ("PRJ-A", "A"),
            )

    def tearDown(self):
        self.tmp.cleanup()

    def _create_crop(self, label: str) -> tuple[str, str]:
        source_sha = hashlib.sha256(
            f"source:{label}".encode("utf-8")
        ).hexdigest()
        geometry_sha = hashlib.sha256(
            f"geometry:{label}".encode("utf-8")
        ).hexdigest()
        identity_payload = build_pz1_crop_identity(
            source_image_id="img-sha256-" + source_sha,
            source_annotation_id=f"plate-{label}",
            source_geometry_hash=geometry_sha,
            interpolation="lanczos4",
            output_width=256,
            output_height=64,
        )
        identity = compute_crop_identity_sha256(identity_payload)

        image = self.workspace / label / f"{label}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"crop:{label}".encode("utf-8"))

        result = self.registry.register_crop_artifact(
            crop_identity_sha256=identity,
            identity_mode=identity_payload["identity_mode"],
            source_file_sha256=source_sha,
            artifact_path=image,
            artifact_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            size_bytes=image.stat().st_size,
            width=256,
            height=64,
            source_annotation_id=f"plate-{label}",
            source_geometry_hash=geometry_sha,
            project_id="PRJ-A",
            iteration_num=1,
            source_mode="campaign",
        )
        return result.crop_id, identity

    def _payload(self, identity: str, char: str = "A") -> dict:
        return {
            "crop_identity_sha256": identity,
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
                "approved": True,
                "excluded": False,
                "candidate": True,
            },
            "status": "perfect",
        }

    def test_unknown_project_is_read_only_empty_state(self):
        state = summarize_project_az_resource(
            self.registry,
            project_id="PRJ-NOT-REGISTERED",
        )

        self.assertFalse(state.project_exists)
        self.assertEqual(state.crop_count, 0)
        self.assertEqual(state.az_count, 0)
        self.assertEqual(state.coverage_status, "no_project")

        with self.db.read_connection() as con:
            count = con.execute(
                "SELECT COUNT(*) FROM projects"
            ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_project_with_crops_and_no_az_is_missing(self):
        self._create_crop("one")
        self._create_crop("two")

        state = summarize_project_az_resource(
            self.registry,
            project_id="PRJ-A",
        )

        self.assertEqual(state.crop_count, 2)
        self.assertEqual(state.az_count, 0)
        self.assertEqual(state.missing_count, 2)
        self.assertEqual(state.coverage_status, "missing")
        self.assertFalse(state.contract_ready)

    def test_partial_project_az_reports_pending_review(self):
        crop_a, identity_a = self._create_crop("one")
        self._create_crop("two")

        self.store.save_revision(
            crop_id=crop_a,
            payload=self._payload(identity_a, "A"),
            source_kind="project_import",
            trust_state="external_pending_review",
            bind_project_id="PRJ-A",
            effective_status="imported_pending_review",
            created_at="2026-09-26T00:01:00+00:00",
        )

        state = summarize_project_az_resource(
            self.registry,
            project_id="PRJ-A",
        )

        self.assertEqual(state.crop_count, 2)
        self.assertEqual(state.az_count, 1)
        self.assertEqual(state.usable_count, 1)
        self.assertEqual(state.pending_review_count, 1)
        self.assertEqual(state.missing_count, 1)
        self.assertEqual(state.coverage_status, "partial")
        self.assertTrue(state.review_required)
        self.assertFalse(state.contract_ready)

    def test_full_project_az_counts_ready_and_excluded(self):
        crop_a, identity_a = self._create_crop("one")
        crop_b, identity_b = self._create_crop("two")

        self.store.save_revision(
            crop_id=crop_a,
            payload=self._payload(identity_a, "A"),
            source_kind="local_manual",
            trust_state="local_manual",
            bind_project_id="PRJ-A",
            effective_status="approved",
            created_at="2026-09-26T00:01:00+00:00",
        )
        excluded_payload = self._payload(identity_b, "B")
        excluded_payload["gold_state"] = {
            "approved": False,
            "excluded": True,
            "candidate": False,
        }
        excluded_payload["status"] = "needs_fix"
        self.store.save_revision(
            crop_id=crop_b,
            payload=excluded_payload,
            source_kind="local_manual",
            trust_state="local_manual",
            bind_project_id="PRJ-A",
            effective_status="excluded",
            created_at="2026-09-26T00:02:00+00:00",
        )

        state = summarize_project_az_resource(
            self.registry,
            project_id="PRJ-A",
        )

        self.assertEqual(state.crop_count, 2)
        self.assertEqual(state.az_count, 2)
        self.assertEqual(state.usable_count, 1)
        self.assertEqual(state.ready_count, 1)
        self.assertEqual(state.excluded_count, 1)
        self.assertEqual(state.coverage_status, "full")
        self.assertTrue(state.contract_ready)
        self.assertEqual(
            state.latest_updated_at,
            "2026-09-26T00:02:00+00:00",
        )

    def test_full_pending_review_is_visible_but_not_contract_ready(self):
        crop_a, identity_a = self._create_crop("pending-one")
        crop_b, identity_b = self._create_crop("pending-two")

        for crop_id, identity, char in (
            (crop_a, identity_a, "A"),
            (crop_b, identity_b, "B"),
        ):
            self.store.save_revision(
                crop_id=crop_id,
                payload=self._payload(identity, char),
                source_kind="project_import",
                trust_state="external_pending_review",
                bind_project_id="PRJ-A",
                effective_status="imported_pending_review",
                created_at="2026-09-26T00:03:00+00:00",
            )

        state = summarize_project_az_resource(
            self.registry,
            project_id="PRJ-A",
        )

        self.assertEqual(state.coverage_status, "full")
        self.assertEqual(state.az_count, 2)
        self.assertEqual(state.pending_review_count, 2)
        self.assertEqual(state.ready_count, 0)
        self.assertEqual(state.reviewed_count, 0)
        self.assertTrue(state.review_required)
        self.assertFalse(state.contract_ready)

    def test_partial_reviewed_az_is_not_full_contract(self):
        crop_a, identity_a = self._create_crop("reviewed-one")
        self._create_crop("reviewed-two")

        self.store.save_revision(
            crop_id=crop_a,
            payload=self._payload(identity_a, "A"),
            source_kind="local_manual",
            trust_state="local_manual",
            bind_project_id="PRJ-A",
            effective_status="approved",
            created_at="2026-09-26T00:04:00+00:00",
        )

        state = summarize_project_az_resource(
            self.registry,
            project_id="PRJ-A",
        )

        self.assertEqual(state.coverage_status, "partial")
        self.assertEqual(state.ready_count, 1)
        self.assertEqual(state.missing_count, 1)
        self.assertEqual(state.reviewed_count, 1)
        self.assertFalse(state.review_required)
        self.assertFalse(state.contract_ready)

    def test_pending_snapshot_cannot_satisfy_required_char_run_contract(self):
        from auto_annotation_tool.campaign_resource_contracts import (
            resource_contract_ready,
        )
        from auto_annotation_tool.campaign_resource_state import (
            CampaignResourceSnapshot,
        )

        snapshot = CampaignResourceSnapshot(
            key="char_run",
            canonical_key="char_run",
            code="AZ",
            label="AZ - anotacje znaków",
            source="Registry projektu",
            validation="Do kontroli",
            tone="warning",
            counter_text="2/2",
            meta={
                "contract_ready": False,
                "contract_enforced": True,
                "review_required": True,
                "pending_review_count": 2,
            },
        )

        self.assertFalse(
            resource_contract_ready(snapshot, required=True)
        )


def test_char_run_row_is_registry_backed_not_planowane_placeholder():
    path = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "campaign_step1_assets.py"
    )
    source = path.read_text(encoding="utf-8-sig")

    assert "Planowane | import AZ nie jest jeszcze dostępny w zasobach bramki." not in source
    assert "_get_project_start_az_resource_state" in source
    assert 'coverage_status = str(az_state.get("coverage_status") or "")' in source


def test_char_run_pending_review_is_warning_not_success():
    path = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "campaign_step1_assets.py"
    )
    source = path.read_text(encoding="utf-8-sig")

    assert 'elif az_pending_count > 0 or az_other_count > 0:' in source
    assert 'Do kontroli |' in source
    assert 'char_run_validation_tone = "warning"' in source
    assert 'review_complete=bool(az_state.get("review_complete"))' in source


if __name__ == "__main__":
    unittest.main()
