from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from PIL import Image

from auto_annotation_tool.campaign_resource_state import CampaignResourceSnapshot
from auto_annotation_tool.campaign_transition_evaluator import CampaignTransitionEvalContext, is_transition_ready
from auto_annotation_tool.campaign_transition_specs import get_transition_specs_for_edge
from auto_annotation_tool.gui.z2_gt_readiness import summarize_char_gt_entries, approval_block_reason
from auto_annotation_tool.gui.z3_extraction_sources import prepare_plate_cut_detections_for_source
from auto_annotation_tool.character_recognition.plate_generator import PlateGenerator
from auto_annotation_tool.data_models import ImageAnnotation
from test_z3_pz1_ground_truth import plate


@pytest.mark.parametrize("geometries,present,ready", [(10, 0, True), (10, 6, True), (0, 0, False)])
def test_t03_uses_geometry_ready_not_gt_complete(geometries, present, ready):
    entries = [{"entry_key": "image", "image_name": "image.jpg", "plates": [
        {"polygon": [[0, 0], [20, 0], [20, 10], [0, 10]],
         "ground_truth_text": "ABC123" if i < present else ""} for i in range(geometries)]}]
    summary = summarize_char_gt_entries(entries)
    snapshot = CampaignResourceSnapshot("approved_plates", "approved_plates", "AT", "Tablice",
                                        counter_text=str(geometries), tone="success", meta=summary)
    ctx = CampaignTransitionEvalContext(selected_path="char_from_images", current_step=2, image_count=1000,
                                        resource_snapshots={"approved_plates": snapshot})
    spec = next(spec for spec in get_transition_specs_for_edge("e2_to_e3") if spec.key == "e2_to_e3_chars")
    assert is_transition_ready(spec, ctx) is ready
    assert summary["plate_gt_present_count"] == present
    assert summary["plate_gt_missing_count"] == geometries - present


def test_invalid_geometry_is_still_blocked_without_gt():
    det = plate()
    det.polygon = [(0, 0)] * 4
    assert approval_block_reason(SimpleNamespace(), ImageAnnotation("x.jpg", 100, 50, [det]))
    assert not summarize_char_gt_entries([{"plates": [{"polygon": det.polygon}]}])["geometry_ready"]


def test_pz1_extracts_plate_without_ground_truth_text(tmp_path):
    image = tmp_path / "ordinary_photo.png"
    Image.new("RGB", (100, 50), (100, 120, 150)).save(image)
    det = plate({"plate_annotation_id": "plate-no-gt"})
    host = SimpleNamespace(_plate_cut_reading_order_key=lambda pair: pair[0],
                           _extract_source_plate_tokens_from_filename=Mock(side_effect=AssertionError("no filename GT")))
    prepared = prepare_plate_cut_detections_for_source(host, image.name, [det])
    generator = PlateGenerator(tmp_path / "crops")
    results = generator.generate_from_annotations(image, ImageAnnotation(image.name, 100, 50, prepared), rectify=False)
    assert len(results) == 1
    record = generator.metadata["plate_000000"]
    assert not record.get("ground_truth_text")
    assert record["source_annotation_id"] == "plate-no-gt"
    assert (tmp_path / "crops/images/plate_000000.jpg").is_file()
