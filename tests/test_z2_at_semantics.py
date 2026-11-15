from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from auto_annotation_tool.annotators import combined_annotator, plate_annotator
from auto_annotation_tool.annotators.plate_model_contract import plate_class_ids
from auto_annotation_tool.data_models import Detection, ImageAnnotation
from auto_annotation_tool.gui import z2_import_workflow, z2_preview_state, z2_preview_workflow, z2_preview_editor
from auto_annotation_tool.gui.z2_auto_run_result import protected_view_bundle, vehicle_assistance_visible
from auto_annotation_tool.plate_import_validation import plate_import_error
from auto_annotation_tool.gui.z2_restore_semantics import exclude_restored_vehicle_plate_conflicts


def plate(box, **attrs):
    x1, y1, x2, y2 = box
    return Detection("plate", .9, box, polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)], attributes=attrs)


def test_protected_manual_edit_does_not_resurrect_old_auto_vehicle_sized_plate():
    automatic = plate((130, 188, 531, 556))
    manual = plate((183, 436, 257, 470), manually_edited="true", manual_source="preview", ground_truth_text="BI360FE")
    ann = ImageAnnotation("BI360FE.jpg", 640, 640, [automatic, manual])
    state = {"annotations": [ann], "input_dir": "images"}
    host = SimpleNamespace(_is_plate_detection_label=lambda name: name == "plate",
                           _preview_annotation_has_manual_touch=lambda a: True)
    normalizer = lambda a: z2_preview_state._build_manual_override_annotation(host, a)
    bundle = protected_view_bundle(state, {"bi360fe.jpg"}, build_manual_override=normalizer)
    assert bundle[ann.filename][0].detections == [manual]
    assert ann.detections == [automatic, manual]
    bundle[ann.filename][0].detections[0].attributes["ground_truth_text"] = "CHANGED"
    assert manual.attributes["ground_truth_text"] == "BI360FE"


def test_explicitly_approved_mixed_image_remains_wholly_protected():
    ann = ImageAnnotation("ok.jpg", 640, 640, [plate((10, 10, 30, 20)), plate((50, 50, 70, 60), manually_edited="true")])
    normalizer = Mock()
    bundle = protected_view_bundle({"annotations": [ann], "input_dir": "images", "approved_filenames": {"ok.jpg"}},
                                   {"ok.jpg"}, build_manual_override=normalizer)
    normalizer.assert_not_called()
    assert bundle["ok.jpg"][0].detections == ann.detections
    assert bundle["ok.jpg"][0]._approved_for_training
    assert not hasattr(ann, "_approved_for_training")


def test_manual_overlay_also_preserves_other_plate_with_explicit_gt():
    reviewed = plate((100, 200, 200, 220), ground_truth_text="ABC123")
    edited = plate((300, 200, 400, 220), manually_edited="true")
    ann = ImageAnnotation("many.jpg", 640, 640, [reviewed, edited, plate((0, 0, 600, 600))])
    host = SimpleNamespace(_is_plate_detection_label=lambda name: name == "plate",
                           _preview_annotation_has_manual_touch=lambda a: True)
    preserved = z2_preview_state._build_manual_override_annotation(host, ann)
    assert preserved.detections == [reviewed, edited]


def test_import_rejects_vehicle_polygons_before_allocating_workspace(tmp_path):
    xml = tmp_path / "annotations.xml"
    xml.write_text('''<annotations><image name="one.jpg" width="640" height="640">
      <box label="vehicle" xtl="130" ytl="188" xbr="531" ybr="556"/>
      <polygon label="plate" source="auto" points="131,188;531,188;531,556;131,556"/>
      </image></annotations>''', encoding="utf-8")
    before = xml.read_bytes()
    host = SimpleNamespace(_bbox_from_polygon=lambda p: (min(x for x,y in p), min(y for x,y in p), max(x for x,y in p), max(y for x,y in p)),
                           _keypoints_from_polygon=lambda p: [(x,y,1.) for x,y in p],
                           _resolve_existing_run_dir=lambda p: Path(p), _allocate_annotation_run_dir=Mock())
    host._parse_cvat_preview_annotations = lambda p: z2_preview_workflow._parse_cvat_preview_annotations(host, p)
    run, error, needs_images = z2_import_workflow._import_external_annotation_run_to_workspace(host, tmp_path)
    assert run is None and not needs_images
    assert "pojazdy" in error and "one.jpg" in error
    host._allocate_annotation_run_dir.assert_not_called()
    assert xml.read_bytes() == before


def test_import_accepts_real_plate_inside_vehicle_and_preserves_manual_geometry():
    vehicle = Detection("vehicle", .9, (10, 10, 400, 400))
    assert not plate_import_error([ImageAnnotation("a.jpg", 640, 640, [vehicle, plate((100, 200, 200, 220))])])
    reviewed = plate(vehicle.bbox, manually_edited="true", ground_truth_text="ABC123")
    assert not plate_import_error([ImageAnnotation("a.jpg", 640, 640, [vehicle, reviewed])])


def test_vehicle_assistance_off_hides_imported_boxes_and_their_hit_targets():
    owner = SimpleNamespace(_manual_xml_template_enabled=lambda: False, _get_auto_vehicle_choice=lambda: "skip")
    owner._get_preview_annotation = Mock()
    assert not vehicle_assistance_visible(owner)
    assert z2_preview_editor._find_preview_vehicle_hit(owner, 100, 100) is None
    owner._get_preview_annotation.assert_not_called()
    owner._get_auto_vehicle_choice = lambda: "use"
    assert vehicle_assistance_visible(owner)
    owner._manual_xml_template_enabled = lambda: True
    owner._manual_vehicle_assist_enabled = lambda: False
    assert not vehicle_assistance_visible(owner)


@pytest.mark.parametrize("module,kind", [(plate_annotator, "plate"), (combined_annotator, "combined")])
def test_vehicle_or_person_model_cannot_load_as_plate_model(monkeypatch, module, kind):
    wrong = SimpleNamespace(names={0: "person", 1: "car", 2: "truck"})
    monkeypatch.setattr(module, "YOLO_AVAILABLE", True)
    monkeypatch.setattr(module, "get_yolo_class", lambda: lambda path: wrong)
    annotator = (module.PlateAnnotator(Path("wrong.pt")) if kind == "plate"
                 else module.CombinedAnnotator(Path("vehicle.pt"), Path("wrong.pt")))
    ok, message = annotator.load_models()
    assert not ok and "klasy tablicy" in message


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


@pytest.mark.parametrize("kind", ["plate", "combined"])
def test_multiclass_model_only_converts_plate_class_and_keeps_keypoint_index(monkeypatch, kind):
    names = {0: "car", 1: "license_plate"}
    result = SimpleNamespace(names=names, boxes=SimpleNamespace(
        xyxy=Tensor([[10, 10, 500, 500], [100, 200, 200, 220]]), conf=Tensor([.99, .9]), cls=Tensor([0, 1])),
        keypoints=SimpleNamespace(data=Tensor([[[10,10], [500,10], [500,500], [10,500]],
                                             [[101,201], [199,201], [199,219], [101,219]]])))
    model = Mock(return_value=[result])
    model.names = names
    if kind == "plate":
        annotator = plate_annotator.PlateAnnotator(Path("mixed.pt"))
        annotator.model = model
        annotator.is_pose_model = True
        monkeypatch.setattr(plate_annotator, "get_image_size", lambda p: (640, 640))
        monkeypatch.setattr(plate_annotator, "CV2_AVAILABLE", True)
        annotator._read_image_for_yolo = lambda p: np.zeros((640, 640, 3), dtype=np.uint8)
        detections = annotator.process_image(Path("image.jpg")).detections
    else:
        annotator = combined_annotator.CombinedAnnotator(Path("vehicle.pt"), Path("mixed.pt"))
        annotator.plate_model = model
        annotator.is_plate_pose_model = True
        detections = annotator._detect_plates(np.zeros((640,640,3), dtype=np.uint8), 640, 640)
    assert len(detections) == 1
    assert detections[0].bbox == (100., 200., 200., 220.)
    assert set(detections[0].polygon) == {(101.,201.), (199.,201.), (199.,219.), (101.,219.)}


def test_plate_model_class_aliases_are_accepted():
    assert plate_class_ids(["number_plate", "vehicle", "tablica"]) == {0, 2}


def test_restore_filters_stale_vehicle_plate_even_after_vehicle_boxes_were_removed(tmp_path):
    import json
    from auto_annotation_tool.gui import z2_restore_workflow
    project = tmp_path / "project"
    source = project / "import" / "annotations.xml"
    run = project / "result"
    source.parent.mkdir(parents=True)
    run.mkdir()
    (project / "_campaign_state").mkdir()
    (project / "_campaign_state/project.json").write_text(json.dumps({
        "project": {"project_start_plate_source_xml": str(source)}}), encoding="utf-8")
    bad = '<polygon label="plate" source="auto" points="131,188;531,188;531,556;131,556"><attribute name="plate_annotation_id">stale</attribute></polygon>'
    vehicle = '<box label="vehicle" xtl="130" ytl="188" xbr="531" ybr="556"/>'
    manual = '<polygon label="plate" source="manual" points="180,430;250,430;250,470;180,470"><attribute name="ground_truth_text">BI360FE</attribute></polygon>'
    template = '<annotations><image name="BI360FE.jpg" width="640" height="640">{}</image></annotations>'
    source.write_text(template.format(vehicle + bad), encoding="utf-8")
    xml = run / "annotations.xml"
    xml.write_text(template.format(bad + manual), encoding="utf-8")
    before = xml.read_bytes()
    manifest = {"annotation_run_type": "auto_annotation"}
    owner = SimpleNamespace(
        _bbox_from_polygon=lambda p: (min(x for x,y in p), min(y for x,y in p), max(x for x,y in p), max(y for x,y in p)),
        _keypoints_from_polygon=lambda p: [(x,y,1.) for x,y in p],
        _resolve_safe_annotation_run_dir=lambda p, **kw: p,
        _load_annotation_run_manifest=lambda p: manifest,
        _mark_auto_plate_origin_for_annotations=lambda a: None,
        _is_plate_detection_label=lambda name: name == "plate",
    )
    owner._parse_cvat_preview_annotations = lambda p: z2_preview_workflow._parse_cvat_preview_annotations(owner, p)
    restored = []
    # Stop after the actual restore-preparation path has parsed and repaired
    # shapes, before unrelated image discovery or UI state is needed.
    owner._resolve_run_image_dir_for_annotations = lambda annotations, *a, **kw: restored.extend(annotations)
    z2_restore_workflow._prepare_annotation_run_restore_payload(owner, run)
    assert len(restored) == 1 and len(restored[0].plates) == 1
    assert restored[0].plates[0].attributes["ground_truth_text"] == "BI360FE"
    assert z2_preview_state._get_plate_detections(owner, restored[0]) == restored[0].plates
    assert len(restored[0]._excluded_vehicle_plate_conflicts) == 1
    assert xml.read_bytes() == before
    assert exclude_restored_vehicle_plate_conflicts(owner, restored, run, manifest) == 0
    reviewed = owner._parse_cvat_preview_annotations(xml)
    assert exclude_restored_vehicle_plate_conflicts(owner, reviewed, run, {"approved_filenames": ["BI360FE.jpg"]}) == 1
    assert len(reviewed[0].plates) == 1
    for attr, value in (("ground_truth_text", "EXPLICIT"), ("manually_edited", "true"),
                        ("plate_annotation_id", "new-detection")):
        corrected = owner._parse_cvat_preview_annotations(xml)
        corrected[0].plates[0].attributes[attr] = value
        assert exclude_restored_vehicle_plate_conflicts(owner, corrected, run, manifest) == 0
    corrected = owner._parse_cvat_preview_annotations(xml)
    corrected[0].plates[0].polygon[0] = (150, 200)
    assert exclude_restored_vehicle_plate_conflicts(owner, corrected, run, manifest) == 0
