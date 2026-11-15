import copy
import io
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from auto_annotation_tool.mobile_acquisition import (
    CROP_SCHEMA, CropReview, acquisition_provenance, export_annotation_preview,
    file_sha256, read_crop_session,
)
from auto_annotation_tool.gui.z3_split_utils import build_split_entries
from auto_annotation_tool.ranking.mobile_package_experiments import read_mobile_report_bundle


def make_crop(text="ab123", index=1):
    return {
        "text": text, "image": f"crop-{index:06d}.jpg", "image_width": 200, "image_height": 60,
        "characters": [{"label": "a", "confidence": .8, "left": .1, "top": .2, "right": .3, "bottom": .9}],
        "observations": [{"observation_id": f"observation-{index}", "text": text,
                          "entity_id": index, "plate_track_id": 10 + index, "frame_id": 100 + index,
                          "scene_generation": 2, "captured_at_ms": 1700000000000 + index,
                          "captured_elapsed_ns": 100000000 + index, "source": "camera_cache",
                          "timing": {"mz_ms": 12}, "hud": {"models": [{"stage": "MZ", "backend": "ncnn"}]}}],
    }


def make_archive(path, crops=None, *, manifest_changes=None, extra_entries=()):
    crops = crops if crops is not None else [make_crop()]
    manifest = {"schema": CROP_SCHEMA, "session_id": "capture-1", "started_at_ms": 1700000000000,
                "crops": crops}
    manifest.update(manifest_changes or {})
    image = io.BytesIO()
    Image.new("RGB", (200, 60), "white").save(image, format="JPEG")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("session.json", json.dumps(manifest))
        for crop in crops:
            archive.writestr(crop["image"], image.getvalue())
        for name, data in extra_entries:
            info = zipfile.ZipInfo()
            info.filename = name
            archive.writestr(info, data)
    return path


def test_case_grouping_preserves_images_and_observations(tmp_path):
    crops = [make_crop("ab123", 1), make_crop("AB123", 2), make_crop("AB 123", 3)]
    crops[0]["observations"].append({**crops[0]["observations"][0], "observation_id": "earlier",
                                     "entity_id": 8, "captured_at_ms": 1699999999000})
    session = read_crop_session(make_archive(tmp_path / "normal.zip", crops))
    assert [group.key for group in session.groups] == ["AB123", "AB 123"]
    assert session.groups[0].observation_count == 3
    assert session.groups[0].crops == tuple(crops[:2])
    assert session.groups[0].base["text"] == "ab123"
    assert session.read_image(session.groups[0]).size == (200, 60)
    assert session.read_image(session.groups[0], 1).size == (200, 60)


@pytest.mark.parametrize("mutate", [
    lambda crop: crop.update(text=""),
    lambda crop: crop.update(image_width=100),
    lambda crop: crop.update(image_height=True),
    lambda crop: crop.update(observations=[]),
    lambda crop: crop["observations"][0].update(text="OTHER"),
    lambda crop: crop["observations"].append(copy.deepcopy(crop["observations"][0])),
    lambda crop: crop["characters"][0].update(left=-.1),
    lambda crop: crop["characters"][0].update(right=.01),
    lambda crop: crop["characters"][0].update(top=float("nan")),
])
def test_invalid_crop_contract_rejected(tmp_path, mutate):
    crop = make_crop()
    mutate(crop)
    with pytest.raises(ValueError):
        read_crop_session(make_archive(tmp_path / "invalid.zip", [crop]))


@pytest.mark.parametrize("entry", ["../escape.jpg", "/absolute.jpg", "C:/escape.jpg", "a\\b.jpg", "a:stream", "SESSION.JSON"])
def test_unsafe_or_ambiguous_zip_rejected(tmp_path, entry):
    path = make_archive(tmp_path / "unsafe.zip", extra_entries=[(entry, b"x")])
    with pytest.raises(ValueError, match="Unsafe or duplicate"):
        read_crop_session(path)


def test_zip_limits_and_wrong_schema(tmp_path):
    path = make_archive(tmp_path / "crop.zip")
    for limits in ({"max_entries": 1}, {"max_total_bytes": 1}, {"max_manifest_bytes": 1}):
        with pytest.raises(ValueError):
            read_crop_session(path, **limits)
    path = make_archive(tmp_path / "research.zip", manifest_changes={"schema": "alpr.mobile_research_bundle.v1"})
    with pytest.raises(ValueError, match="Expected"):
        read_crop_session(path)


def test_review_reopen_drafts_and_binding(tmp_path):
    path = make_archive(tmp_path / "source.zip")
    session = read_crop_session(path)
    before = file_sha256(path)
    sidecar = tmp_path / "review.json"
    review = CropReview(session, sidecar)
    review.set_decision("plate_000001", "AB123", "draft")
    reopened = CropReview(session, sidecar)
    assert reopened.decisions["plate_000001"]["status"] == "draft"
    with pytest.raises(ValueError, match="No confirmed"):
        export_annotation_preview(reopened, tmp_path / "preview")
    reopened.set_decision("plate_000001", "ab124", "confirmed")
    assert reopened.decisions["plate_000001"]["gt"] == "AB124"
    with pytest.raises(ValueError, match="another window"):
        review.set_decision("plate_000001", "AB125", "confirmed")
    assert file_sha256(path) == before
    other = read_crop_session(make_archive(tmp_path / "other.zip", [make_crop("DIFFERENT")]))
    with pytest.raises(ValueError, match="does not belong"):
        CropReview(other, sidecar)


def test_export_preserves_provenance_and_converts_geometry(tmp_path):
    crops = [make_crop("ab123", 1), make_crop("AB123", 2), make_crop("UNREVIEWED", 3)]
    source = make_archive(tmp_path / "source.zip", crops)
    session = read_crop_session(source)
    review = CropReview(session, tmp_path / "review.json")
    with pytest.raises(ValueError, match="requires ground truth"):
        review.set_decision("plate_000001", " ", "confirmed")
    review.set_decision("plate_000001", "ab124", "confirmed")
    result = export_annotation_preview(review, tmp_path / "preview")
    metadata = json.loads((result / "metadata.json").read_text(encoding="utf-8"))
    assert list(metadata) == ["plate_000001"]
    row = metadata["plate_000001"]
    assert row["ocr_text"] == "ab123"
    assert row["source_expected_text"] == "AB124"
    assert row["source_expected_text_source"] == "mobile_crop_human_review"
    assert row["characters"][0]["bbox"] == pytest.approx([20, 12, 60, 54])
    assert row["characters"][0]["method"] == "yolo"
    assert row["mobile_acquisition"]["crops"] == crops[:2]
    assert row["dataset_group"] == "mobile-session:capture-1"
    assert row["status"] == "needs_fix"
    assert file_sha256(result / "source.zip") == file_sha256(source)
    assert "attempt_id" not in row and "mt_invocation_id" not in row
    assert not (result / "report.json").exists()
    imported = json.loads((result / "acquisition_import.json").read_text(encoding="utf-8"))
    assert imported["pipeline_quality_available"] is False
    with zipfile.ZipFile(source) as archive:
        assert (result / "images" / "plate_000001.jpg").read_bytes() == archive.read(crops[0]["image"])
    with pytest.raises(FileExistsError):
        export_annotation_preview(review, result)
    provenance = acquisition_provenance(row)
    assert provenance["mobile_acquisition"] == row["mobile_acquisition"]
    provenance["mobile_acquisition"]["crops"].clear()
    assert row["mobile_acquisition"]["crops"] == crops[:2]


def test_changed_source_does_not_publish_partial_preview(tmp_path):
    source = make_archive(tmp_path / "source.zip")
    review = CropReview(read_crop_session(source), tmp_path / "review.json")
    review.set_decision("plate_000001", "AB123", "confirmed")
    make_archive(source, [make_crop("CHANGED")])
    with pytest.raises(ValueError, match="changed"):
        export_annotation_preview(review, tmp_path / "preview")
    assert not (tmp_path / "preview").exists()
    assert not list(tmp_path.glob(".acquisition-*"))


@pytest.mark.parametrize("extension", [".zip", ".alprsession"])
def test_acquisition_cannot_be_used_as_benchmark_even_with_report(tmp_path, extension):
    fake_report = json.dumps({"schema": "alpr.mobile_benchmark_report.v1", "session_id": "capture-1"})
    path = make_archive(tmp_path / ("crop" + extension), extra_entries=[("report.json", fake_report)])
    bundle = read_mobile_report_bundle(path)
    assert not bundle.validation.ok
    assert any("Akwizycji" in error for error in bundle.validation.errors)


def test_dataset_split_keeps_entire_mobile_session_together():
    entries = [{"id": index, "dataset_group": f"session-{index // 3}"} for index in range(12)]
    splits, counts = build_split_entries(entries, {"train": .5, "val": .25, "test": .25})
    assignments = {}
    for split, items in splits.items():
        for item in items:
            assignments.setdefault(item["dataset_group"], set()).add(split)
    assert all(len(values) == 1 for values in assignments.values())
    assert counts == {"train": 6, "val": 3, "test": 3}
    single, counts = build_split_entries(entries[:3], {"train": .8, "val": .1, "test": .1})
    assert counts == {"train": 3, "val": 0, "test": 0}
    assert len(single["train"]) == 3


def test_plain_dataset_split_and_zero_weight_remain_valid():
    entries = [{"id": index} for index in range(10)]
    splits, counts = build_split_entries(entries, {"train": .8, "val": .1, "test": .1})
    assert counts == {"train": 8, "val": 1, "test": 1}
    assert sorted(item["id"] for items in splits.values() for item in items) == list(range(10))
    entries[0]["dataset_group"] = "capture-1"
    _, counts = build_split_entries(entries, {"train": 1, "val": 0, "test": 0})
    assert counts == {"train": 10, "val": 0, "test": 0}


def test_z3_uses_operator_gt_without_filename_backfill(tmp_path):
    from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab

    session = read_crop_session(make_archive(tmp_path / "source.zip"))
    review = CropReview(session, tmp_path / "review.json")
    review.set_decision("plate_000001", "A", "confirmed")
    preview = export_annotation_preview(review, tmp_path / "preview")
    metadata = json.loads((preview / "metadata.json").read_text(encoding="utf-8"))
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    assert host._backfill_preview_expected_texts_from_sources(metadata) is False
    row = metadata["plate_000001"]
    assert host._get_preview_expected_texts(row) == ["A"]
    assert host._derive_preview_status_from_data(row, row["characters"]) == "perfect"
    row["characters"][0]["character"] = "B"
    assert host._derive_preview_status_from_data(row, row["characters"]) == "needs_fix"
    first_key = host._build_gold_export_plate_unique_key(row)
    row["source_image"] = "another copy of the same imported archive"
    assert host._build_gold_export_plate_unique_key(row) == first_key


def test_acquisition_preview_readiness_is_independent_of_xml(tmp_path):
    from types import SimpleNamespace
    from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab

    session = read_crop_session(make_archive(tmp_path / "source.zip"))
    review = CropReview(session, tmp_path / "review.json")
    review.set_decision("plate_000001", "AB123", "confirmed")
    preview = export_annotation_preview(review, tmp_path / "preview")
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    host.app = SimpleNamespace(campaign_free_mode=True)
    host._step3_linear_mode = False
    host._is_campaign_char_step3_context = lambda: False
    host.preview_dir_var = SimpleNamespace(get=lambda: str(preview))
    source = {"xml_path": "old.xml", "images_dir": "old-images", "annotation_run_dir": ""}
    for key in source:
        setattr(host, key + "_var", SimpleNamespace(get=lambda name=key: source[name]))
    assert host._is_extract_preview_ready() is False
    for key in source:
        source[key] = ""
    assert host._is_extract_preview_ready() is True
    assert host._is_extract_preview_ready_fast() is True
    assert host._get_extract_preview_ready_count() == 1
    assert not (preview / "extract_manifest.json").exists()
