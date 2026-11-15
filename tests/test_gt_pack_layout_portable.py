from PIL import Image

from auto_annotation_tool.gt_pack import ALPRGTPack, merge_gt_packs


def _image(path):
    Image.new("RGB", (80, 40), (30, 60, 90)).save(path)
    return path


def _seed(pack, image, layout):
    item = pack.add_image(image)
    pack.ensure_plate(
        image_id=item["image_id"],
        polygon=[(10, 10), (60, 10), (60, 26), (10, 26)],
        plate_id="plate-ann-layout",
    )
    pack.set_plate_layout_gt("plate-ann-layout", layout)


def test_layout_gt_roundtrip(tmp_path):
    image = _image(tmp_path / "source.png")
    pack = ALPRGTPack.create(tmp_path / "pack.alprgt")
    _seed(pack, image, "two_row")
    state = pack.resolve_plate_layout_gt("plate-ann-layout")
    assert state["resolved"] is True
    assert state["conflict"] is False
    assert state["layout"] == "two_row"
    assert pack.summary()["layout_revisions"] == 1
    assert pack.validate(deep=False)["ok"]


def test_concurrent_layout_edits_merge_as_explicit_conflict(tmp_path):
    image = _image(tmp_path / "source.png")
    left = ALPRGTPack.create(tmp_path / "left.alprgt")
    right = ALPRGTPack.create(tmp_path / "right.alprgt")
    _seed(left, image, "single_row")
    _seed(right, image, "two_row")

    merged = merge_gt_packs(
        [left.root, right.root],
        tmp_path / "merged.alprgt",
    )
    state = merged.resolve_plate_layout_gt("plate-ann-layout")
    assert state["resolved"] is False
    assert state["conflict"] is True
    assert sorted(state["values"]) == ["single_row", "two_row"]
