import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.data_models import Detection
from auto_annotation_tool.gui.z3_extraction_sources import (
    backfill_preview_expected_texts_from_sources,
    prepare_plate_cut_detections_for_source,
    read_xml_plate_attributes,
)


def plate(attributes=None):
    return Detection(
        label="plate",
        confidence=1.0,
        bbox=(0.0, 0.0, 20.0, 10.0),
        polygon=[
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 10.0),
            (0.0, 10.0),
        ],
        attributes=dict(attributes or {}),
    )


def host_with_parser(parser):
    return SimpleNamespace(
        _plate_cut_reading_order_key=lambda pair: pair[0],
        _extract_source_plate_tokens_from_filename=parser,
    )


def test_xml_plate_attributes_preserve_gt_and_annotation_identity():
    polygon = ET.fromstring(
        "<polygon label='plate' source='manual' points='0,0;20,0;20,10;0,10'>"
        "<attribute name='confidence'>0.875</attribute>"
        "<attribute name='plate_annotation_id'>plate-ann-fixed</attribute>"
        "<attribute name='ground_truth_text'>WI905PW</attribute>"
        "<attribute name='ground_truth_source'>manual_z2</attribute>"
        "</polygon>"
    )

    confidence, attributes = read_xml_plate_attributes(polygon)

    assert confidence == 0.875
    assert attributes == {
        "plate_annotation_id": "plate-ann-fixed",
        "ground_truth_text": "WI905PW",
        "ground_truth_source": "manual_z2",
    }


def test_explicit_gt_bypasses_filename_parser_in_pz1():
    parser = Mock(side_effect=AssertionError("filename parser must not run"))
    host = host_with_parser(parser)
    det = plate(
        {
            "plate_annotation_id": "plate-ann-one",
            "ground_truth_text": " wi 905-pw ",
            "ground_truth_source": "manual_z2",
        }
    )

    result = prepare_plate_cut_detections_for_source(
        host,
        "AAA_BBB_001.jpg",
        [det],
    )

    assert result == [det]
    assert det.attributes["ground_truth_text"] == "WI905PW"
    assert det.attributes["ground_truth_source"] == "manual_z2"
    assert det.attributes["plate_annotation_id"] == "plate-ann-one"
    assert det.attributes["source_plate_index"] == "0"
    assert det.attributes["source_plate_count"] == "1"
    assert "source_expected_text" not in det.attributes
    assert "source_expected_texts" not in det.attributes
    assert "source_expected_text_source" not in det.attributes
    parser.assert_not_called()


def test_filename_backfill_skips_crop_with_explicit_gt():
    parser = Mock(side_effect=AssertionError("filename parser must not run"))
    host = host_with_parser(parser)
    metadata = {
        "plate_000001": {
            "source_image": "AAA_BBB_001.jpg",
            "ground_truth_text": "BBB",
            "ground_truth_source": "manual_z2",
            "source_annotation_id": "plate-ann-one",
            "plate_attributes": {
                "plate_annotation_id": "plate-ann-one",
                "ground_truth_text": "BBB",
                "ground_truth_source": "manual_z2",
            },
        }
    }

    assert not backfill_preview_expected_texts_from_sources(host, metadata)
    parser.assert_not_called()
    assert "source_expected_text" not in metadata["plate_000001"]
    assert "source_expected_texts" not in metadata["plate_000001"]
