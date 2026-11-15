from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_annotation_tool.ranking.mz_whole_plate_evaluation import (
    aggregate_plate_metrics, approved_manifest_text, evaluate_plate_records, evaluate_mz_test_split,
)


def records(text):
    return [{"character":ch,"bbox":[i*12,0,i*12+10,20]} for i,ch in enumerate(text)]


@pytest.mark.parametrize("prediction,field,expected", [
    ("ABC123","exact_match",True),("ABC823","incorrect_characters",1),
    ("ABC12","missing_characters",1),("ABC1234","extra_characters",1),
    ("","no_read",True),
])
def test_plate_alignment_reuses_the_existing_cer_contract(prediction,field,expected):
    result=evaluate_plate_records("ABC123",records(prediction))
    assert result[field]==expected
    if prediction=="ABC123":
        assert result["cer"]==0 and result["edit_distance"]==0
    if not prediction:
        assert result["missing_characters"]==6 and result["cer"]==1


def test_predictions_are_read_by_geometry_not_input_order_or_gt_positions():
    result=evaluate_plate_records("ABC123",list(reversed(records("ABC123"))))
    assert result["prediction"]=="ABC123"
    assert result["exact_match"]


def test_two_row_reading_uses_shared_application_policy():
    predictions = [{"character":"2","bbox":[20,50,30,70]},
        {"character":"B","bbox":[20,0,30,20]},
        {"character":"1","bbox":[0,50,10,70]},
        {"character":"A","bbox":[0,0,10,20]}]
    result=evaluate_plate_records("AB12",predictions)
    assert result["prediction"]=="AB12" and result["exact_match"]
    assert [row["reading_row"] for row in result["ordered_predictions"]]==[1,1,2,2]


def test_prediction_is_not_trimmed_to_ground_truth_character_count():
    result=evaluate_plate_records("AB",records("ABX"))
    assert result["extra_characters"]==1 and result["prediction"]=="ABX"


def test_aggregation_uses_micro_cer_and_includes_no_read():
    rows=[evaluate_plate_records("AB",records("AB")),evaluate_plate_records("ABCD",[])]
    summary=aggregate_plate_metrics(rows)
    assert summary["plates"]==2 and summary["exact_match_rate"]==.5
    assert summary["no_read"]==1 and summary["cer"]==pytest.approx(4/6)


def test_gt_is_the_final_pz3_text_not_an_old_pack_or_filename():
    assert approved_manifest_text({"layout":{"plate_text_reading_order":"1TF8664"},
        "characters":[{"character":ch} for ch in "1TF8664"],"source_image":"1TF866.jpg"})=="1TF8664"
    with pytest.raises(ValueError,match="inconsistent"):
        approved_manifest_text({"layout":{"plate_text_reading_order":"OLD"},"characters":[{"character":"A"}]})


def test_wrong_model_classes_cannot_be_reported_as_mz_quality(tmp_path):
    # Checked after assignment validation; a pretrained COCO model cannot pass
    # as a character checkpoint merely by producing no detections.
    import json
    from auto_annotation_tool.dataset_split_assignment import build_group_assignment
    items=[{"pid":f"p{i}","image_path":f"images/p{i}.jpg","label_path":f"labels/p{i}.txt"} for i in range(20)]
    assignment=build_group_assignment({"items":items},source_dataset_id="test",source_manifest_sha256="a"*64,
        ratios={"train":.8,"val":.1,"test":.1})
    (tmp_path/"split_assignment_manifest.json").write_text(json.dumps(assignment),encoding="utf-8")
    (tmp_path/"metadata_manifest.json").write_text(json.dumps({"items":items}),encoding="utf-8")
    with pytest.raises(ValueError,match="36 character"):
        evaluate_mz_test_split(SimpleNamespace(task="detect",names={0:"person"}),tmp_path,{})
