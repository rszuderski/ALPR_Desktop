import unittest

from auto_annotation_tool.registry.crop_identity import (
    CROP_IDENTITY_MODE_LINEAGE,
    CROP_IDENTITY_SCHEMA,
    CROP_PIXEL_IDENTITY_SCHEMA,
    PZ1_OUTPUT_WIDTH,
    PZ1_SINGLE_ROW_HEIGHT,
    PZ1_TWO_ROW_HEIGHT,
    build_pixel_fallback_descriptor,
    build_pz1_crop_identity,
    canonical_json_bytes,
    compute_crop_identity_sha256,
    compute_pixel_fallback_sha256,
    compute_pz1_crop_identity_sha256,
    verify_crop_identity_sha256,
)


BASE = {
    "source_image_id": "img-sha256:" + "a" * 64,
    "source_annotation_id": "plate-ann-123",
    "source_geometry_hash": "B" * 64,
    "rectify": True,
    "do_deskew": False,
    "enhance_contrast": False,
    "interpolation": "lanczos4",
    "output_width": PZ1_OUTPUT_WIDTH,
    "output_height": PZ1_SINGLE_ROW_HEIGHT,
}


class AZCropIdentityTests(unittest.TestCase):
    def test_canonical_json_ignores_mapping_key_order(self):
        left = {"b": 2, "a": {"y": 2, "x": 1}}
        right = {"a": {"x": 1, "y": 2}, "b": 2}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(
            compute_crop_identity_sha256(left),
            compute_crop_identity_sha256(right),
        )

    def test_pz1_identity_contract_contains_no_path_project_iteration_or_gt(self):
        payload = build_pz1_crop_identity(**BASE)

        self.assertEqual(payload["schema"], CROP_IDENTITY_SCHEMA)
        self.assertEqual(
            payload["identity_mode"],
            CROP_IDENTITY_MODE_LINEAGE,
        )

        serialized = canonical_json_bytes(payload).decode("utf-8")
        for forbidden in (
            "relative_path",
            "external_path",
            "filename",
            "project_id",
            "iteration",
            "ground_truth",
            "expected_text",
            "created_at",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_same_lineage_gives_same_identity(self):
        first = compute_pz1_crop_identity_sha256(**BASE)
        second = compute_pz1_crop_identity_sha256(**dict(BASE))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_interpolation_alias_is_normalized(self):
        a = compute_pz1_crop_identity_sha256(
            **{**BASE, "interpolation": "LANCZOS"}
        )
        b = compute_pz1_crop_identity_sha256(
            **{**BASE, "interpolation": "lanczos4"}
        )
        self.assertEqual(a, b)

    def test_geometry_change_changes_identity(self):
        first = compute_pz1_crop_identity_sha256(**BASE)
        second = compute_pz1_crop_identity_sha256(
            **{**BASE, "source_geometry_hash": "c" * 64}
        )
        self.assertNotEqual(first, second)

    def test_source_image_change_changes_identity(self):
        first = compute_pz1_crop_identity_sha256(**BASE)
        second = compute_pz1_crop_identity_sha256(
            **{
                **BASE,
                "source_image_id": "img-sha256:" + "d" * 64,
            }
        )
        self.assertNotEqual(first, second)

    def test_plate_annotation_change_changes_identity(self):
        first = compute_pz1_crop_identity_sha256(**BASE)
        second = compute_pz1_crop_identity_sha256(
            **{**BASE, "source_annotation_id": "plate-ann-999"}
        )
        self.assertNotEqual(first, second)

    def test_crop_contract_change_changes_identity(self):
        first = compute_pz1_crop_identity_sha256(**BASE)

        changes = (
            {"rectify": False},
            {"do_deskew": True},
            {"enhance_contrast": True},
            {"interpolation": "cubic"},
            {"output_width": 320},
            {"output_height": PZ1_TWO_ROW_HEIGHT},
            {"contract_version": "pz1_crop.v2"},
        )
        for change in changes:
            with self.subTest(change=change):
                second = compute_pz1_crop_identity_sha256(
                    **{**BASE, **change}
                )
                self.assertNotEqual(first, second)

    def test_single_and_two_row_have_distinct_identity(self):
        single = compute_pz1_crop_identity_sha256(
            **{**BASE, "output_height": PZ1_SINGLE_ROW_HEIGHT}
        )
        double = compute_pz1_crop_identity_sha256(
            **{**BASE, "output_height": PZ1_TWO_ROW_HEIGHT}
        )
        self.assertNotEqual(single, double)

    def test_verify_crop_identity(self):
        payload = build_pz1_crop_identity(**BASE)
        digest = compute_crop_identity_sha256(payload)

        self.assertTrue(verify_crop_identity_sha256(payload, digest))
        self.assertTrue(
            verify_crop_identity_sha256(payload, digest.upper())
        )
        self.assertFalse(
            verify_crop_identity_sha256(payload, "0" * 64)
        )
        self.assertFalse(
            verify_crop_identity_sha256(payload, "invalid")
        )

    def test_missing_required_lineage_is_rejected(self):
        for field in (
            "source_image_id",
            "source_annotation_id",
            "source_geometry_hash",
        ):
            with self.subTest(field=field):
                values = dict(BASE)
                values[field] = ""
                with self.assertRaises(ValueError):
                    build_pz1_crop_identity(**values)

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            build_pz1_crop_identity(
                **{**BASE, "output_width": 0}
            )
        with self.assertRaises(ValueError):
            build_pz1_crop_identity(
                **{**BASE, "output_height": -1}
            )

    def test_unknown_interpolation_is_rejected(self):
        with self.assertRaises(ValueError):
            build_pz1_crop_identity(
                **{**BASE, "interpolation": "magic"}
            )

    def test_pixel_fallback_is_deterministic(self):
        pixels = bytes(range(12))
        first = compute_pixel_fallback_sha256(
            width=2,
            height=2,
            channels=3,
            decoded_rgb_bytes=pixels,
        )
        second = compute_pixel_fallback_sha256(
            width=2,
            height=2,
            channels=3,
            decoded_rgb_bytes=pixels,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_pixel_fallback_changes_with_pixels_or_shape(self):
        pixels = bytes(range(12))
        changed = bytearray(pixels)
        changed[-1] ^= 1

        base = compute_pixel_fallback_sha256(
            width=2,
            height=2,
            channels=3,
            decoded_rgb_bytes=pixels,
        )
        other_pixels = compute_pixel_fallback_sha256(
            width=2,
            height=2,
            channels=3,
            decoded_rgb_bytes=changed,
        )
        other_shape = compute_pixel_fallback_sha256(
            width=1,
            height=4,
            channels=3,
            decoded_rgb_bytes=pixels,
        )

        self.assertNotEqual(base, other_pixels)
        self.assertNotEqual(base, other_shape)

    def test_pixel_fallback_rejects_invalid_byte_count(self):
        with self.assertRaises(ValueError):
            compute_pixel_fallback_sha256(
                width=2,
                height=2,
                channels=3,
                decoded_rgb_bytes=b"too-short",
            )

    def test_pixel_descriptor_is_serializable_contract(self):
        pixels = bytes(range(12))
        digest = compute_pixel_fallback_sha256(
            width=2,
            height=2,
            channels=3,
            decoded_rgb_bytes=pixels,
        )
        descriptor = build_pixel_fallback_descriptor(
            width=2,
            height=2,
            channels=3,
            pixel_sha256=digest,
        )

        self.assertEqual(
            descriptor["schema"],
            CROP_PIXEL_IDENTITY_SCHEMA,
        )
        self.assertEqual(descriptor["pixel_sha256"], digest)


if __name__ == "__main__":
    unittest.main()
