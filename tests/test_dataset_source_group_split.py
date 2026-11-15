from copy import deepcopy
import json
from pathlib import Path

import pytest

from auto_annotation_tool.dataset_split_assignment import (
    assignment_sha256, build_group_assignment, source_group_id, validate_group_assignment,
)
from auto_annotation_tool.training.dataset_splitter import DatasetSplitter


RATIOS = {"train": .8, "val": .1, "test": .1}


def item(index, **extra):
    return {"pid":f"p{index}", "source_pid":f"plate_{index}", "image_path":f"images/p{index}.jpg",
        "label_path":f"labels/p{index}.txt", "characters":[{"character":"A", "class_id":0}], **extra}


def assign(items, seed=42):
    return build_group_assignment({"items":items}, source_dataset_id="source",
        source_manifest_sha256="a"*64, ratios=RATIOS, seed=seed)


def test_dataset_group_stays_in_one_split_and_wins_over_image_identity():
    manifest = assign([item(i,dataset_group="A",provenance={"source_image_id":str(i)}) for i in range(3)]
        + [item(i) for i in range(3,30)])
    rows = [row for row in manifest["items"] if row["group_id"]=="dataset_group:A"]
    assert len(rows)==3 and len({row["split"] for row in rows})==1


def test_source_image_fallback_stays_together():
    manifest = assign([item(i,provenance={"source_image_id":"image"}) for i in range(3)]
        + [item(i) for i in range(3,30)])
    rows = [row for row in manifest["items"] if row["group_id"]=="source_image_id:image"]
    assert len(rows)==3 and len({row["split"] for row in rows})==1


def test_missing_group_creates_individual_crop_groups():
    manifest = assign([item(i) for i in range(10)])
    assert manifest["counts"]["groups_total"]==10
    assert len({row["group_id"] for row in manifest["items"]})==10


def test_explicit_session_group_is_supported_without_fuzzy_names():
    assert source_group_id(item(1,session_group="capture"))=="session_group:capture"
    assert source_group_id(item(2,source_image="same-looking-file.jpg"))=="crop:plate_2"


def test_assignment_is_deterministic_and_not_sensitive_to_input_iteration_order():
    items = [item(i,provenance={"source_image_id":str(i//3)}) for i in range(30)]
    first = assign(items)
    second = assign(list(reversed(items)))
    assert first["assignment_sha256"]==second["assignment_sha256"]
    assert first["items"]==second["items"]
    assert "C:" not in json.dumps(first)


def test_seed_can_change_assignments_without_splitting_any_group():
    items = [item(i,provenance={"source_image_id":str(i//3)}) for i in range(90)]
    first, second = assign(items,42), assign(items,43)
    assert first["assignment_sha256"]!=second["assignment_sha256"]
    assert validate_group_assignment(first)["ok"]
    assert validate_group_assignment(second)["ok"]


def test_large_source_groups_are_not_systematically_absorbed_by_train():
    # An absolute-deficit greedy allocator put all large groups in the 80%
    # fold, leaving val/test with singleton sources regardless of seed.
    items=[item(i,provenance={"source_image_id":str(i//3)}) for i in range(60)]
    items += [item(i) for i in range(60,360)]
    manifest=assign(items)
    grouped=[row for row in manifest["items"] if row["group_id"].startswith("source_image_id:")]
    assert {row["split"] for row in grouped}=={"train","val","test"}
    assert validate_group_assignment(manifest)["ok"]


def test_every_item_occurs_exactly_once_and_ratios_can_be_approximate():
    items = [item(i,dataset_group="large") for i in range(9)] + [item(10),item(11)]
    manifest = assign(items)
    assert len(manifest["items"])==11
    assert {row["crop_identity"] for row in manifest["items"]}=={f"plate_{i}" for i in [*range(9),10,11]}
    assert validate_group_assignment(manifest,source_manifest={"items":items})["ok"]


def test_validator_detects_cross_split_group_leakage_even_with_recomputed_fingerprint():
    manifest = assign([item(i,dataset_group="A") for i in range(3)]+[item(i) for i in range(3,30)])
    changed = deepcopy(manifest)
    row = next(row for row in changed["items"] if row["group_id"]=="dataset_group:A")
    row["split"] = "test" if row["split"]!="test" else "train"
    changed["assignment_sha256"] = assignment_sha256(changed)
    result = validate_group_assignment(changed)
    assert not result["ok"]
    assert "dataset_group:A" in result["cross_split_group_leakage"]


def test_validator_detects_missing_and_duplicate_crops():
    items = [item(i) for i in range(10)]
    manifest = assign(items)
    manifest["items"][0] = deepcopy(manifest["items"][1])
    manifest["assignment_sha256"] = assignment_sha256(manifest)
    result = validate_group_assignment(manifest,source_manifest={"items":items})
    assert not result["ok"] and result["missing_items"] and result["duplicate_crop_identities"]


@pytest.mark.parametrize("path", ["../outside.jpg", "C:\\data\\p.jpg", "/data/p.jpg"])
def test_source_paths_cannot_escape_dataset(path):
    with pytest.raises(ValueError,match="relative"):
        assign([item(0,image_path=path)])


def write_source(root, with_manifest=True):
    (root/"images").mkdir(parents=True)
    (root/"labels").mkdir()
    items = [item(i,provenance={"source_image_id":str(i//3)}) for i in range(30)]
    for row in items:
        (root/row["image_path"]).write_bytes(row["pid"].encode())
        (root/row["label_path"]).write_text("0 .5 .5 .2 .2",encoding="utf-8")
    (root/"data.yaml").write_text("nc: 1\nnames: ['A']\ntrain: images\nval: images\n",encoding="utf-8")
    if with_manifest:
        (root/"metadata_manifest.json").write_text(json.dumps({"dataset_type":"char_yolo_detect",
            "items":items,"plate_count":30,"character_count":30}),encoding="utf-8")
    return items


def test_native_splitter_writes_complete_assignment_and_remapped_manifest(tmp_path):
    source = tmp_path/"source"
    items = write_source(source)
    before = {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    output = tmp_path/"output"
    ok, message, stats = DatasetSplitter(42).split_dataset(source,output,RATIOS)
    assert ok and stats["total"]==30 and stats["group_aware"]
    assignment = json.loads((output/"split_assignment_manifest.json").read_text(encoding="utf-8"))
    assert validate_group_assignment(assignment,source_manifest={"items":items})["ok"]
    derived = json.loads((output/"metadata_manifest.json").read_text(encoding="utf-8"))
    assert derived["split_enabled"]
    for row in derived["items"]:
        assert (output/row["image_path"]).read_bytes()==row["pid"].encode()
        assert (output/row["label_path"]).is_file()
    assert before=={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}


def test_legacy_split_api_and_counts_remain_compatible(tmp_path):
    source = tmp_path/"source"
    write_source(source,with_manifest=False)
    ok, message, stats = DatasetSplitter().split_dataset(source,tmp_path/"output",RATIOS)
    assert ok and stats=={"train":24,"val":3,"test":3,"total":30}


def test_module_has_one_canonical_splitter_definition():
    import ast
    import auto_annotation_tool.training.dataset_splitter as module
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8-sig"))
    assert sum(isinstance(n,ast.ClassDef) and n.name=="DatasetSplitter" for n in tree.body)==1


def test_existing_output_is_not_overwritten(tmp_path):
    source = tmp_path/"source"
    write_source(source)
    output = tmp_path/"output"
    output.mkdir()
    marker = output/"keep.txt"
    marker.write_text("keep")
    ok,message,_ = DatasetSplitter().split_dataset(source,output,RATIOS)
    assert not ok and marker.read_text()=="keep"
