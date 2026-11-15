import json
from pathlib import Path

from PIL import Image

from auto_annotation_tool.gt_pack import (
    ALPRGTPack,
    IMAGE_ID_PREFIX,
    merge_gt_packs,
)


def make_image(path: Path, *, value=(10, 20, 30), fmt=None):
    image = Image.new("RGB", (40, 20), value)
    image.save(path, format=fmt)
    return path


def polygon():
    return [
        (4.0, 4.0),
        (30.0, 4.0),
        (30.0, 14.0),
        (4.0, 14.0),
    ]


def shifted_polygon():
    return [
        (5.0, 5.0),
        (31.0, 5.0),
        (31.0, 15.0),
        (5.0, 15.0),
    ]


def test_image_identity_is_independent_from_filename(tmp_path):
    source = make_image(tmp_path / "first.png")
    renamed = tmp_path / "renamed.png"
    renamed.write_bytes(source.read_bytes())

    pack = ALPRGTPack.create(tmp_path / "pack.alprgt")
    first = pack.add_image(source)
    second = pack.add_image(renamed)

    assert first["image_id"].startswith(IMAGE_ID_PREFIX)
    assert first["image_id"] == second["image_id"]
    assert first["source_file_sha256"] == second["source_file_sha256"]

    stored = pack.get_image(first["image_id"])
    assert stored["aliases"] == ["first.png", "renamed.png"]


def test_reencoded_same_pixels_are_not_silently_same_image(tmp_path):
    png = make_image(tmp_path / "same.png", value=(1, 2, 3), fmt="PNG")
    bmp = make_image(tmp_path / "same.bmp", value=(1, 2, 3), fmt="BMP")

    pack = ALPRGTPack.create(tmp_path / "pack.alprgt")
    png_rec = pack.add_image(png)
    bmp_rec = pack.add_image(bmp)

    assert png_rec["image_id"] != bmp_rec["image_id"]
    assert png_rec["source_file_sha256"] != bmp_rec["source_file_sha256"]
    # Recovery metadata can still reveal that decoded pixels are identical.
    assert png_rec["pixel_sha256s"] == bmp_rec["pixel_sha256s"]


def test_z2_plate_roundtrip_keeps_plate_id_and_gt(tmp_path):
    image = make_image(tmp_path / "plate.png")
    pack = ALPRGTPack.create(tmp_path / "pack.alprgt")

    result = pack.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text=" wi 905-pw ",
    )

    assert result["plate"]["plate_id"] == "plate-ann-fixed"
    assert result["ground_truth"]["resolved"] is True
    assert result["ground_truth"]["has_ground_truth"] is True
    assert result["ground_truth"]["text"] == "WI905PW"
    assert result["geometry"]["resolved"] is True
    assert len(result["geometry"]["points"]) == 4
    assert pack.validate()["ok"] is True


def test_repeated_same_gt_does_not_create_extra_revision(tmp_path):
    image = make_image(tmp_path / "plate.png")
    pack = ALPRGTPack.create(tmp_path / "pack.alprgt")
    result = pack.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text="ABC123",
    )
    before = pack.summary()["revisions"]

    pack.set_ground_truth(
        result["plate"]["plate_id"],
        "abc-123",
        source="manual_z2",
    )

    assert pack.summary()["revisions"] == before
    assert pack.resolve_ground_truth("plate-ann-fixed")["text"] == "ABC123"


def test_clear_gt_is_an_append_only_revision(tmp_path):
    image = make_image(tmp_path / "plate.png")
    pack = ALPRGTPack.create(tmp_path / "pack.alprgt")
    pack.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text="ABC123",
    )
    before = pack.summary()["revisions"]

    clear_revision = pack.clear_ground_truth("plate-ann-fixed")
    state = pack.resolve_ground_truth("plate-ann-fixed")

    assert pack.summary()["revisions"] == before + 1
    assert clear_revision["operation"] == "clear"
    assert clear_revision["value"] is None
    assert state["resolved"] is True
    assert state["conflict"] is False
    assert state["has_ground_truth"] is False
    assert state["operation"] == "clear"
    assert state["text"] == ""


def test_setting_gt_after_clear_creates_new_descendant(tmp_path):
    image = make_image(tmp_path / "plate.png")
    pack = ALPRGTPack.create(tmp_path / "pack.alprgt")
    pack.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text="ABC123",
    )
    clear_rev = pack.clear_ground_truth("plate-ann-fixed")
    set_rev = pack.set_ground_truth("plate-ann-fixed", "ABC124")

    assert clear_rev["revision_id"] in set_rev["parents"]
    state = pack.resolve_ground_truth("plate-ann-fixed")
    assert state["resolved"] is True
    assert state["text"] == "ABC124"
    assert state["revision_ids"] == [set_rev["revision_id"]]


def test_geometry_update_keeps_plate_id_and_creates_revision(tmp_path):
    image = make_image(tmp_path / "plate.png")
    pack = ALPRGTPack.create(tmp_path / "pack.alprgt")
    first = pack.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text="ABC123",
    )
    before = pack.summary()["geometries"]

    second = pack.upsert_z2_plate(
        image_path=image,
        polygon=shifted_polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text="ABC123",
    )

    assert second["plate"]["plate_id"] == first["plate"]["plate_id"]
    assert pack.summary()["geometries"] == before + 1
    assert second["geometry"]["resolved"] is True
    assert pack.validate()["ok"] is True


def test_merge_is_idempotent_for_same_pack(tmp_path):
    image = make_image(tmp_path / "plate.png")
    source = ALPRGTPack.create(tmp_path / "source.alprgt")
    source.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text="ABC123",
    )

    merged = merge_gt_packs(
        [source.root, source.root],
        tmp_path / "merged.alprgt",
    )

    summary = merged.summary()
    assert summary["images"] == 1
    assert summary["plates"] == 1
    assert summary["geometries"] == 1
    assert summary["revisions"] == 1
    assert summary["conflicts"] == 0
    assert merged.validate()["ok"] is True


def test_merge_prunes_ancestor_head(tmp_path):
    image = make_image(tmp_path / "plate.png")

    base = ALPRGTPack.create(tmp_path / "base.alprgt")
    base.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text="ABC123",
    )

    old = ALPRGTPack.create(tmp_path / "old.alprgt")
    old.merge_from(base)

    newer = ALPRGTPack.create(tmp_path / "newer.alprgt")
    newer.merge_from(base)
    child = newer.set_ground_truth("plate-ann-fixed", "ABC124")

    merged = merge_gt_packs(
        [old.root, newer.root],
        tmp_path / "merged.alprgt",
    )

    state = merged.resolve_ground_truth("plate-ann-fixed")
    plate = merged.get_plate("plate-ann-fixed")
    assert state["resolved"] is True
    assert state["conflict"] is False
    assert state["text"] == "ABC124"
    assert state["revision_ids"] == [child["revision_id"]]
    assert plate["revision_heads"] == [child["revision_id"]]


def test_merge_preserves_concurrent_gt_as_explicit_conflict(tmp_path):
    image = make_image(tmp_path / "plate.png")

    base = ALPRGTPack.create(tmp_path / "base.alprgt")
    base.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text="ABC123",
    )

    left = ALPRGTPack.create(tmp_path / "left.alprgt")
    left.merge_from(base)
    right = ALPRGTPack.create(tmp_path / "right.alprgt")
    right.merge_from(base)

    left.set_ground_truth("plate-ann-fixed", "ABC124")
    right.set_ground_truth("plate-ann-fixed", "ABC125")

    merged = merge_gt_packs(
        [left.root, right.root],
        tmp_path / "merged.alprgt",
    )

    resolved = merged.resolve_ground_truth("plate-ann-fixed")
    assert resolved["resolved"] is False
    assert resolved["conflict"] is True
    assert resolved["values"] == ["ABC124", "ABC125"]
    assert merged.validate()["ok"] is True


def test_concurrent_same_gt_is_semantically_resolved(tmp_path):
    image = make_image(tmp_path / "plate.png")

    base = ALPRGTPack.create(tmp_path / "base.alprgt")
    base.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text="ABC123",
    )

    left = ALPRGTPack.create(tmp_path / "left.alprgt")
    left.merge_from(base)
    right = ALPRGTPack.create(tmp_path / "right.alprgt")
    right.merge_from(base)

    left.set_ground_truth(
        "plate-ann-fixed",
        "ABC124",
        source="manual_z2",
    )
    right.set_ground_truth(
        "plate-ann-fixed",
        "ABC124",
        source="mobile_human_review",
    )

    merged = merge_gt_packs(
        [left.root, right.root],
        tmp_path / "merged.alprgt",
    )

    state = merged.resolve_ground_truth("plate-ann-fixed")
    assert state["resolved"] is True
    assert state["conflict"] is False
    assert state["text"] == "ABC124"
    assert len(state["revision_ids"]) == 2


def test_clear_vs_concurrent_set_is_conflict(tmp_path):
    image = make_image(tmp_path / "plate.png")

    base = ALPRGTPack.create(tmp_path / "base.alprgt")
    base.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text="ABC123",
    )

    left = ALPRGTPack.create(tmp_path / "left.alprgt")
    left.merge_from(base)
    right = ALPRGTPack.create(tmp_path / "right.alprgt")
    right.merge_from(base)

    left.clear_ground_truth("plate-ann-fixed")
    right.set_ground_truth("plate-ann-fixed", "ABC124")

    merged = merge_gt_packs(
        [left.root, right.root],
        tmp_path / "merged.alprgt",
    )

    state = merged.resolve_ground_truth("plate-ann-fixed")
    assert state["resolved"] is False
    assert state["conflict"] is True
    assert state["values"] == ["<CLEAR>", "ABC124"]


def test_merge_order_does_not_change_semantic_result(tmp_path):
    image = make_image(tmp_path / "plate.png")

    pack_a = ALPRGTPack.create(
        tmp_path / "a.alprgt",
        producer="desktop",
    )
    pack_a.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-a",
        ground_truth_text="AAA111",
        producer="desktop",
    )

    pack_b = ALPRGTPack.create(
        tmp_path / "b.alprgt",
        producer="android",
    )
    pack_b.upsert_z2_plate(
        image_path=image,
        polygon=shifted_polygon(),
        plate_annotation_id="plate-ann-b",
        ground_truth_text="BBB222",
        producer="android",
    )

    ab = merge_gt_packs(
        [pack_a.root, pack_b.root],
        tmp_path / "ab.alprgt",
    )
    ba = merge_gt_packs(
        [pack_b.root, pack_a.root],
        tmp_path / "ba.alprgt",
    )

    assert ab.summary() == ba.summary()
    assert sorted(
        (
            plate["plate_id"],
            ab.resolve_ground_truth(plate["plate_id"]).get("text"),
        )
        for plate in ab.list_plates()
    ) == sorted(
        (
            plate["plate_id"],
            ba.resolve_ground_truth(plate["plate_id"]).get("text"),
        )
        for plate in ba.list_plates()
    )


def test_rebuild_manifest_repairs_counts(tmp_path):
    image = make_image(tmp_path / "plate.png")
    pack = ALPRGTPack.create(tmp_path / "pack.alprgt")
    pack.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text="ABC123",
    )

    manifest = json.loads(pack.manifest_path.read_text(encoding="utf-8"))
    manifest["record_counts"]["plates"] = 999
    pack.manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    pack.manifest = manifest

    before = pack.validate(deep=False)
    assert before["ok"] is False
    assert any("licznik plates" in issue for issue in before["issues"])

    pack.rebuild_manifest()
    after = pack.validate(deep=False)
    assert after["ok"] is True


def test_validate_detects_missing_parent(tmp_path):
    image = make_image(tmp_path / "plate.png")
    pack = ALPRGTPack.create(tmp_path / "pack.alprgt")
    pack.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text="ABC123",
    )
    revision = pack.list_revisions()[0]
    revision["parents"] = ["rev-sha256-missing"]
    revision_path = pack.revisions_dir / f"{revision['revision_id']}.json"
    revision_path.write_text(
        json.dumps(revision),
        encoding="utf-8",
    )

    result = pack.validate(deep=False)
    assert result["ok"] is False
    assert any("brak parent GT" in issue for issue in result["issues"])


def test_validate_detects_revision_cycle(tmp_path):
    image = make_image(tmp_path / "plate.png")
    pack = ALPRGTPack.create(tmp_path / "pack.alprgt")
    pack.upsert_z2_plate(
        image_path=image,
        polygon=polygon(),
        plate_annotation_id="plate-ann-fixed",
        ground_truth_text="ABC123",
    )
    second = pack.set_ground_truth("plate-ann-fixed", "ABC124")

    revisions = {
        item["revision_id"]: item
        for item in pack.list_revisions()
    }
    first_id = next(
        rid
        for rid in revisions
        if rid != second["revision_id"]
    )
    first = revisions[first_id]
    second_rec = revisions[second["revision_id"]]

    first["parents"] = [second_rec["revision_id"]]
    second_rec["parents"] = [first["revision_id"]]

    for record in (first, second_rec):
        path = pack.revisions_dir / f"{record['revision_id']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")

    result = pack.validate(deep=False)
    assert result["ok"] is False
    assert any("cykl grafu GT" in issue for issue in result["issues"])


def test_deep_validate_checks_blob_hash(tmp_path):
    image = make_image(tmp_path / "plate.png")
    pack = ALPRGTPack.create(tmp_path / "pack.alprgt")
    record = pack.add_image(image, copy_blob=True)

    assert pack.validate(deep=True)["ok"] is True

    blob_ref = record["blob_refs"][0]
    blob_path = pack.root / blob_ref
    blob_path.write_bytes(b"corrupted")

    result = pack.validate(deep=True)
    assert result["ok"] is False
    assert any("ma inny SHA-256" in issue for issue in result["issues"])
