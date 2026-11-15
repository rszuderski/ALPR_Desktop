import unittest

from auto_annotation_tool.registry.pz2_az_adapter import (
    az_revision_to_pz2_metadata,
    derive_pz2_revision_context,
    pz2_metadata_to_az_payload,
)


IDENTITY = "a" * 64


class PZ2AZAdapterTests(unittest.TestCase):
    def base(self):
        return {
            "crop_identity_sha256": IDENTITY,
            "plate_image_width": 256,
            "plate_image_height": 64,
            "characters": [
                {
                    "character": "b",
                    "bbox": [100, 8, 140, 56],
                    "confidence": 0.91,
                    "method": "manual",
                    "source_kind": "local_manual",
                    "reading_row": 1,
                    "reading_col": 2,
                },
                {
                    "character": "a",
                    "bbox": [20, 8, 60, 56],
                    "confidence": 0.95,
                    "method": "manual",
                    "source_kind": "local_manual",
                    "reading_row": 1,
                    "reading_col": 1,
                },
            ],
            "plate_layout": "single_row",
            "layout_row_count": 1,
            "layout_source": "manual_override",
            "plate_layout_override": "single_row",
            "gold_state": {
                "approved": True,
                "excluded": False,
                "candidate": True,
            },
            "status": "perfect",
            "ground_truth_text": "AB",
            "source_info": {
                "bucket": "local_manual",
                "origin": "preview_editor",
            },
        }

    def test_pixel_bboxes_are_normalized_and_sorted(self):
        payload = pz2_metadata_to_az_payload(self.base())

        self.assertEqual(
            [c["character"] for c in payload["characters"]],
            ["A", "B"],
        )
        self.assertEqual(
            payload["characters"][0]["bbox"],
            [0.078125, 0.125, 0.234375, 0.875],
        )
        self.assertEqual(payload["characters"][0]["row"], 1)

    def test_layout_and_gold_state_are_preserved_semantically(self):
        payload = pz2_metadata_to_az_payload(self.base())

        self.assertEqual(payload["layout"]["kind"], "single_row")
        self.assertTrue(payload["layout"]["confirmed"])
        self.assertTrue(payload["gold_state"]["approved"])
        self.assertFalse(payload["gold_state"]["excluded"])
        self.assertTrue(payload["gold_state"]["candidate"])
        self.assertEqual(payload["status"], "perfect")
        self.assertEqual(payload["expected_text"], "AB")

    def test_explicit_dimensions_can_be_used_without_runtime_metadata(self):
        data = self.base()
        data.pop("plate_image_width")
        data.pop("plate_image_height")

        payload = pz2_metadata_to_az_payload(
            data,
            image_width=256,
            image_height=64,
        )
        self.assertEqual(len(payload["characters"]), 2)

    def test_nonempty_characters_require_dimensions(self):
        data = self.base()
        data.pop("plate_image_width")
        data.pop("plate_image_height")

        with self.assertRaises(ValueError):
            pz2_metadata_to_az_payload(data)

    def test_empty_characters_do_not_require_dimensions(self):
        data = self.base()
        data["characters"] = []
        data.pop("plate_image_width")
        data.pop("plate_image_height")

        payload = pz2_metadata_to_az_payload(data)
        self.assertEqual(payload["characters"], [])

    def test_requires_az003_crop_identity(self):
        data = self.base()
        data.pop("crop_identity_sha256")

        with self.assertRaises(ValueError):
            pz2_metadata_to_az_payload(data)

    def test_two_row_reading_rows_survive_canonicalization(self):
        data = self.base()
        data["plate_image_height"] = 128
        data["plate_layout"] = "two_row"
        data["plate_layout_override"] = "two_row"
        data["characters"][0]["bbox"] = [100, 70, 140, 120]
        data["characters"][0]["reading_row"] = 2
        data["characters"][0]["reading_col"] = 1
        data["characters"][1]["bbox"] = [20, 8, 60, 56]
        data["characters"][1]["reading_row"] = 1
        data["characters"][1]["reading_col"] = 1

        payload = pz2_metadata_to_az_payload(data)

        self.assertEqual(payload["layout"]["kind"], "two_row")
        self.assertEqual(
            [c["row"] for c in payload["characters"]],
            [1, 2],
        )

    def test_cvat_context_is_external_reviewed(self):
        data = self.base()
        data["characters"][0]["source_kind"] = "cvat_manual"
        data["source_info"] = {
            "bucket": "cvat_manual",
            "origin": "cvat_import",
        }

        context = derive_pz2_revision_context(data)

        self.assertEqual(context.source_kind, "cvat_manual")
        self.assertEqual(context.trust_state, "external_reviewed")
        self.assertEqual(context.effective_status, "approved")

    def test_excluded_wins_effective_status(self):
        data = self.base()
        data["gold_state"]["approved"] = False
        data["gold_state"]["excluded"] = True
        data["status"] = "needs_fix"

        context = derive_pz2_revision_context(data)

        self.assertEqual(context.effective_status, "excluded")

    def test_detected_context_defaults_to_auto(self):
        data = self.base()
        for rec in data["characters"]:
            rec["source_kind"] = "yolo"
            rec["method"] = "yolo"
        data["source_info"] = {
            "bucket": "auto_preview",
            "origin": "pz2_detect",
        }
        data["gold_state"]["approved"] = False

        context = derive_pz2_revision_context(data)

        self.assertEqual(context.source_kind, "pz2_detect")
        self.assertEqual(context.trust_state, "auto")
        self.assertEqual(context.effective_status, "ready")


    def test_az_revision_restores_pixel_boxes_without_mutating_base(self):
        base = self.base()
        original_first_bbox = list(base["characters"][0]["bbox"])
        payload = pz2_metadata_to_az_payload(base)
        revision = {
            "az_revision_id": "AZR-1",
            "crop_id": "CROP-1",
            "payload_sha256": "f" * 64,
            "payload": payload,
            "source_kind": "local_manual",
            "trust_state": "local_manual",
        }
        base["crop_id"] = "CROP-1"

        restored = az_revision_to_pz2_metadata(
            base,
            revision,
            image_width=256,
            image_height=64,
        )

        self.assertEqual(
            restored["characters"][0]["bbox"],
            [20.0, 8.0, 60.0, 56.0],
        )
        self.assertEqual(
            restored["characters"][1]["bbox"],
            [100.0, 8.0, 140.0, 56.0],
        )
        self.assertEqual(base["characters"][0]["bbox"], original_first_bbox)
        self.assertEqual(restored["characters"][0]["reading_row"], 1)
        self.assertEqual(restored["characters"][0]["reading_col"], 1)
        self.assertEqual(restored["characters"][0]["reading_index"], 1)

    def test_az_revision_restores_layout_gold_expected_text_and_provenance(self):
        base = self.base()
        base["crop_id"] = "CROP-1"
        base["review_state"] = {
            "status": "approved",
            "approved_reference": {"stale": True},
        }
        payload = pz2_metadata_to_az_payload(base)
        revision = {
            "az_revision_id": "AZR-ABC",
            "crop_id": "CROP-1",
            "payload_sha256": "1" * 64,
            "payload": payload,
            "source_kind": "local_manual",
            "trust_state": "local_manual",
            "origin_project_id": "PRJ-A",
            "origin_iteration": 3,
            "created_at": "2026-09-25T12:00:00+00:00",
        }

        restored = az_revision_to_pz2_metadata(
            base,
            revision,
            image_width=256,
            image_height=64,
        )

        self.assertEqual(restored["plate_layout"], "single_row")
        self.assertEqual(restored["plate_layout_override"], "single_row")
        self.assertEqual(restored["layout_source"], "manual_override")
        self.assertTrue(restored["gold_state"]["approved"])
        self.assertFalse(restored["gold_state"]["excluded"])
        self.assertEqual(restored["status"], "perfect")
        self.assertEqual(restored["ground_truth_text"], "AB")
        self.assertNotIn("review_state", restored)
        self.assertEqual(restored["fusion_strategy"], "az_reuse")
        self.assertEqual(
            restored["az_reuse"]["az_revision_id"],
            "AZR-ABC",
        )
        self.assertEqual(restored["az_reuse"]["origin_project_id"], "PRJ-A")
        self.assertEqual(restored["az_reuse"]["origin_iteration"], 3)

    def test_az_revision_restores_two_row_order(self):
        base = self.base()
        base["crop_id"] = "CROP-1"
        base["plate_image_height"] = 128
        base["plate_layout"] = "two_row"
        base["plate_layout_override"] = "two_row"
        base["characters"][0]["bbox"] = [100, 70, 140, 120]
        base["characters"][0]["reading_row"] = 2
        base["characters"][0]["reading_col"] = 1
        base["characters"][1]["bbox"] = [20, 8, 60, 56]
        base["characters"][1]["reading_row"] = 1
        base["characters"][1]["reading_col"] = 1

        payload = pz2_metadata_to_az_payload(base)
        revision = {
            "az_revision_id": "AZR-2R",
            "crop_id": "CROP-1",
            "payload": payload,
        }

        restored = az_revision_to_pz2_metadata(
            base,
            revision,
            image_width=256,
            image_height=128,
        )

        self.assertEqual(restored["plate_layout"], "two_row")
        self.assertEqual(restored["layout_row_count"], 2)
        self.assertEqual(
            [rec["reading_row"] for rec in restored["characters"]],
            [1, 2],
        )
        self.assertEqual(
            [rec["reading_col"] for rec in restored["characters"]],
            [1, 1],
        )

    def test_az_revision_rejects_other_crop_identity(self):
        base = self.base()
        base["crop_id"] = "CROP-1"
        payload = pz2_metadata_to_az_payload(base)
        payload["crop_identity_sha256"] = "b" * 64
        revision = {
            "az_revision_id": "AZR-BAD",
            "crop_id": "CROP-1",
            "payload": payload,
        }

        with self.assertRaises(ValueError):
            az_revision_to_pz2_metadata(
                base,
                revision,
                image_width=256,
                image_height=64,
            )

    def test_pz2_to_az_to_pz2_to_az_roundtrip_is_semantically_stable(self):
        base = self.base()
        base["crop_id"] = "CROP-1"
        first = pz2_metadata_to_az_payload(base)
        revision = {
            "az_revision_id": "AZR-ROUNDTRIP",
            "crop_id": "CROP-1",
            "payload": first,
        }

        restored = az_revision_to_pz2_metadata(
            base,
            revision,
            image_width=256,
            image_height=64,
        )
        second = pz2_metadata_to_az_payload(
            restored,
            image_width=256,
            image_height=64,
        )

        self.assertEqual(first, second)



    def test_project_import_pending_review_keeps_content_but_resets_local_decisions(self):
        base = self.base()
        base["crop_id"] = "CROP-1"
        payload = pz2_metadata_to_az_payload(base)
        revision = {
            "az_revision_id": "AZR-IMPORT",
            "crop_id": "CROP-1",
            "payload_sha256": "2" * 64,
            "payload": payload,
            "source_kind": "local_manual",
            "trust_state": "local_manual",
            "origin_project_id": "PRJ-A",
            "origin_iteration": 2,
            "project_id": "PRJ-B",
            "effective_status": "imported_pending_review",
        }

        restored = az_revision_to_pz2_metadata(
            base,
            revision,
            image_width=256,
            image_height=64,
        )

        self.assertEqual(
            [rec["character"] for rec in restored["characters"]],
            ["A", "B"],
        )
        self.assertEqual(restored["plate_layout"], "single_row")
        self.assertEqual(restored["ground_truth_text"], "AB")
        self.assertFalse(restored["gold_state"]["approved"])
        self.assertFalse(restored["gold_state"]["candidate"])
        self.assertFalse(restored["gold_state"]["excluded"])
        self.assertEqual(restored["status"], "needs_fix")
        self.assertEqual(restored["review_state"]["status"], "in_progress")
        self.assertEqual(restored["review_state"]["source"], "az_project_import")
        self.assertFalse(restored["review_state"]["human_edited"])
        self.assertIsNone(restored["review_state"]["approved_at"])
        self.assertEqual(restored["fusion_strategy"], "az_reuse")
        self.assertEqual(
            restored["fusion_details"]["source"],
            "az_project_import",
        )
        self.assertEqual(
            restored["az_reuse"]["effective_status"],
            "imported_pending_review",
        )
        self.assertEqual(restored["az_reuse"]["project_id"], "PRJ-B")
        self.assertTrue(restored["az_reuse"]["requires_review"])

    def test_project_import_pending_review_does_not_inherit_source_exclusion(self):
        base = self.base()
        base["crop_id"] = "CROP-1"
        base["gold_state"] = {
            "approved": False,
            "excluded": True,
            "candidate": False,
        }
        base["status"] = "needs_fix"
        payload = pz2_metadata_to_az_payload(base)
        revision = {
            "az_revision_id": "AZR-IMPORT-N",
            "crop_id": "CROP-1",
            "payload": payload,
            "project_id": "PRJ-B",
            "effective_status": "imported_pending_review",
        }

        restored = az_revision_to_pz2_metadata(
            base,
            revision,
            image_width=256,
            image_height=64,
        )

        self.assertFalse(restored["gold_state"]["excluded"])
        self.assertFalse(restored["gold_state"]["approved"])
        self.assertFalse(restored["gold_state"]["candidate"])
        self.assertEqual(restored["status"], "needs_fix")
        self.assertTrue(restored["az_reuse"]["requires_review"])

    def test_project_binding_approved_preserves_normal_reuse_semantics(self):
        base = self.base()
        base["crop_id"] = "CROP-1"
        payload = pz2_metadata_to_az_payload(base)
        revision = {
            "az_revision_id": "AZR-SAME-PROJECT",
            "crop_id": "CROP-1",
            "payload": payload,
            "project_id": "PRJ-A",
            "effective_status": "approved",
        }

        restored = az_revision_to_pz2_metadata(
            base,
            revision,
            image_width=256,
            image_height=64,
        )

        self.assertTrue(restored["gold_state"]["approved"])
        self.assertTrue(restored["gold_state"]["candidate"])
        self.assertFalse(restored["gold_state"]["excluded"])
        self.assertEqual(restored["status"], "perfect")
        self.assertFalse(restored["az_reuse"]["requires_review"])


if __name__ == "__main__":
    unittest.main()
