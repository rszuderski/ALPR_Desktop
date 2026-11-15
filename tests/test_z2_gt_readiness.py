from auto_annotation_tool.gui.z2_gt_readiness import (
    summarize_char_gt_entries,
)


def plate(gt=None):
    attrs = {}
    if gt is not None:
        attrs["ground_truth_text"] = gt
    return {
        "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]],
        "attributes": attrs,
    }


def entry(key, image, plates):
    return {
        "entry_key": key,
        "image_name": image,
        "plates": plates,
    }


def test_all_gt_is_ready():
    result = summarize_char_gt_entries(
        [
            entry("a", "a.jpg", [plate("AA111"), plate("BB222")]),
            entry("b", "b.jpg", [plate("CC333")]),
        ]
    )

    assert result["total_plates"] == 3
    assert result["gt_plates"] == 3
    assert result["missing_gt"] == 0
    assert result["ready"] is True
    assert result["missing_images"] == []


def test_missing_gt_does_not_block_geometry_readiness():
    result = summarize_char_gt_entries(
        [
            entry("a", "a.jpg", [plate("AA111"), plate(None)]),
            entry("b", "b.jpg", [plate("CC333")]),
        ]
    )

    assert result["total_plates"] == 3
    assert result["gt_plates"] == 2
    assert result["missing_gt"] == 1
    assert result["ready"] is True
    assert result["geometry_ready"] is True
    assert result["gt_complete"] is False
    assert result["missing_images"] == ["a.jpg"]


def test_current_run_overrides_older_project_entry():
    project = [
        entry("same", "a.jpg", [plate(None), plate("BB222")]),
    ]
    current_run = [
        entry("same", "a.jpg", [plate("AA111"), plate("BB222")]),
    ]

    result = summarize_char_gt_entries(project, current_run)

    assert result["total_plates"] == 2
    assert result["gt_plates"] == 2
    assert result["missing_gt"] == 0
    assert result["ready"] is True


def test_direct_ground_truth_field_is_supported():
    result = summarize_char_gt_entries(
        [
            {
                "entry_key": "direct",
                "image_name": "direct.jpg",
                "plates": [
                    {
                        "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]],
                        "ground_truth_text": "wi 905-pw",
                        "attributes": {},
                    }
                ],
            }
        ]
    )

    assert result["total_plates"] == 1
    assert result["gt_plates"] == 1
    assert result["ready"] is True
