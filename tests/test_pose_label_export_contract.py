import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from auto_annotation_tool.data_models import Detection, ImageAnnotation
from auto_annotation_tool.converters import PlateAnnotation as ConverterPlate, CVATToYOLOPoseConverter
from auto_annotation_tool.training.dataset_creator import PlateAnnotation as DatasetPlate, DatasetCreator
from auto_annotation_tool.exporters.yolo_exporter import YOLOPosePlate4Exporter


UPRIGHT = [(10., 10.), (90., 10.), (90., 40.), (10., 40.)]
PERSPECTIVE = [(0., 0.), (40., 0.), (50., 20.), (60., 130.)]
BOWTIE = [PERSPECTIVE[0], PERSPECTIVE[1], PERSPECTIVE[3], PERSPECTIVE[2]]


@pytest.mark.parametrize("kind", [ConverterPlate, DatasetPlate])
def test_upright_legacy_label_indices_remain_unchanged(kind):
    values = list(map(float, kind('image.jpg', 100, 200, UPRIGHT).to_yolo_pose().split()))
    assert len(values) == 13
    assert values[5:] == [coordinate / (100 if i % 2 == 0 else 200)
                           for i, coordinate in enumerate(v for p in UPRIGHT for v in p)]


@pytest.mark.parametrize("kind", [ConverterPlate, DatasetPlate])
@pytest.mark.parametrize("points", [BOWTIE, UPRIGHT[::-1], UPRIGHT[:3], UPRIGHT + [(1, 2)],
                                    [(0, 0), (40, 0), (10, 10), (0, 40)], PERSPECTIVE,
                                    [(10, 10), (20, 10), (20, 180), (10, 180)]])
def test_bad_or_unknown_pose_labels_raise_instead_of_sorting_or_dropping(kind, points):
    with pytest.raises(ValueError):
        kind('image.jpg', 100, 200, points).to_yolo_pose()


@pytest.mark.parametrize("kind", [ConverterPlate, DatasetPlate])
def test_only_explicit_new_identity_flag_allows_perspective_preserving_indices(kind):
    with pytest.raises(ValueError, match="CORNER_IDENTITY_AMBIGUOUS"):
        kind('image.jpg', 100, 200, PERSPECTIVE, {'gt_status': 'approved', 'manually_edited': 'true'}).to_yolo_pose()
    result = kind('image.jpg', 100, 200, PERSPECTIVE, {'pose_corner_identity': 'confirmed'}).to_yolo_pose()
    values = list(map(float, result.split()))
    assert values[5:] == [0, 0, .4, 0, .5, .1, .6, .65]
    with pytest.raises(ValueError, match="SELF_INTERSECTING"):
        kind('image.jpg', 100, 200, BOWTIE, {'pose_corner_identity': 'confirmed'}).to_yolo_pose()


def detection(points, attributes=None):
    return Detection('plate', .9, (0, 0, 100, 200), polygon=points, attributes=attributes or {})


@pytest.mark.parametrize("points", [BOWTIE, None, [(-1, 10), (90, 10), (90, 40), (-1, 40)]])
def test_export_prevalidates_all_rows_before_touching_existing_output(tmp_path, points):
    destination = tmp_path / 'export'
    (destination / 'labels').mkdir(parents=True)
    existing = destination / 'labels/good.txt'
    existing.write_text('prior label', encoding='utf-8')
    rows = [ImageAnnotation('good.jpg', 100, 200, [detection(UPRIGHT)]),
            ImageAnnotation('bad.jpg', 100, 200, [detection(points)])]
    with pytest.raises(ValueError):
        YOLOPosePlate4Exporter().export(rows, tmp_path, destination)
    assert existing.read_text(encoding='utf-8') == 'prior label'
    assert list(destination.rglob('*')) == [destination / 'labels', existing]


@pytest.mark.parametrize("flag", ['bbox_fallback', 'reordered', 'manual_reordered'])
def test_export_does_not_disguise_corrected_display_as_verified_pose(tmp_path, flag):
    det = detection(UPRIGHT)
    det.geometry = {'schema': 'alpr.pose_geometry.v1', flag: True}
    with pytest.raises(ValueError, match="CORNER_IDENTITY_AMBIGUOUS"):
        YOLOPosePlate4Exporter().export([ImageAnnotation('a.jpg', 100, 200, [det])], tmp_path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_accepted_export_keeps_plate4_contract_and_original_input(tmp_path):
    det = detection(UPRIGHT)
    before = list(det.polygon)
    destination = YOLOPosePlate4Exporter().export([ImageAnnotation('a.jpg', 100, 200, [det])], tmp_path, tmp_path/'out')
    assert det.polygon == before
    assert len((destination/'labels/a.txt').read_text().split()) == 13
    yaml = (destination/'data.yaml').read_text()
    assert 'kpt_shape: [4, 2]' in yaml and 'flip_idx: [1, 0, 3, 2]' in yaml


def xml_fixture(path, points):
    root = ET.Element('annotations')
    for name, quad in [('good.jpg', UPRIGHT), ('bad.jpg', points)]:
        im = ET.SubElement(root, 'image', name=name, width='100', height='200')
        ET.SubElement(im, 'polygon', label='plate', points=';'.join(f'{x},{y}' for x, y in quad))
    ET.ElementTree(root).write(path)


def test_dataset_creator_partial_xml_error_cannot_export_remaining_valid_plate(tmp_path):
    xml_fixture(tmp_path/'input.xml', BOWTIE)
    creator = DatasetCreator()
    ok, message, stats = creator.parse_cvat_xml(tmp_path/'input.xml')
    assert not ok and stats['errors']
    with pytest.raises(ValueError):
        creator.create_dataset(tmp_path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_converter_partial_xml_error_fails_before_any_outputs(tmp_path):
    xml_fixture(tmp_path/'input.xml', UPRIGHT[:3])
    with pytest.raises(ValueError, match='KEYPOINTS_MISSING'):
        CVATToYOLOPoseConverter().convert(tmp_path/'input.xml', tmp_path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize("kind", [ConverterPlate, DatasetPlate])
def test_output_precision_cannot_silently_collapse_valid_tiny_plate(kind):
    points = [(1, 1), (1.00001, 1), (1.00001, 1.000001), (1, 1.000001)]
    with pytest.raises(ValueError, match='zaokrągleniu'):
        kind('tiny.jpg', 10000, 10000, points).to_yolo_pose()


@pytest.mark.parametrize("kind", [ConverterPlate, DatasetPlate])
def test_cvat_metadata_survives_dataset_path_and_reordered_label_requires_identity(kind):
    metadata = {'schema': 'alpr.pose_geometry.v1', 'reordered': True}
    attrs = {'alpr_quad_geometry': json.dumps(metadata)}
    with pytest.raises(ValueError, match='CORNER_IDENTITY_AMBIGUOUS'):
        kind('a.jpg', 100, 200, UPRIGHT, attrs).to_yolo_pose()
    attrs['pose_corner_identity'] = 'confirmed'
    assert kind('a.jpg', 100, 200, UPRIGHT, attrs).to_yolo_pose()
