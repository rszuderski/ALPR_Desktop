from pathlib import Path
from types import SimpleNamespace
import ast

from auto_annotation_tool.gui.z3_detection_pipeline_ui import compile_detection_pipeline_blocks
from auto_annotation_tool.gui.z3_detection_runtime import build_yolo_box_only_records
from auto_annotation_tool.gui.z3_preview_badges import get_preview_source_badge_layers
from auto_annotation_tool.gui.z3_preview_records import normalize_character_box_source
from auto_annotation_tool.gui.z3_plate_layout_runtime import _count_character_sources
from auto_annotation_tool.gui.z3_preview_ui import format_preview_source_counts_line


def _runtime_host():
    return SimpleNamespace(
        _sort_character_records_by_x=lambda records, data=None: sorted(
            list(records or []),
            key=lambda rec: float((rec.get("bbox") or [0])[0]),
        ),
        _char_record_bbox=lambda rec: list(rec.get("bbox") or []),
        _char_record_confidence=lambda rec: float(rec.get("confidence", 0.0) or 0.0),
    )


def _provenance_host():
    return SimpleNamespace(
        _get_character_box_source_tag=lambda rec, data=None, fallback_index=0: rec.get("box_source", ""),
        _get_character_sign_source_tag=lambda rec, data=None, fallback_index=0: rec.get("sign_source", ""),
        _get_character_source_tag=lambda rec, data=None, fallback_index=0: rec.get("source_tag", ""),
    )


def _badge_host():
    return SimpleNamespace(
        app=SimpleNamespace(palette={}),
        _normalize_character_source_tag=lambda raw_tag="", **kwargs: str(raw_tag or "ocr").strip().lower(),
        _get_readable_text_color=lambda fill, preferred=None: preferred or "#ffffff",
    )


def test_single_yb_pipeline_compiles_to_yolo_box_only():
    compiled = compile_detection_pipeline_blocks(["yolo_box"])
    assert compiled["valid"]
    assert compiled["method_key"] == "YOLO_BOX"
    assert compiled["blocks"] == ["yolo_box"]
    assert compiled["requires_yolo"]


def test_yb_records_keep_explicit_yolo_geometry_provenance():
    host = _runtime_host()
    records = build_yolo_box_only_records(
        host,
        [
            {"character": "A", "bbox": [30, 2, 40, 20], "confidence": 0.81},
            {"character": "B", "bbox": [10, 2, 20, 20], "confidence": 0.92},
        ],
    )
    assert len(records) == 2
    assert [rec["bbox"][0] for rec in records] == [10.0, 30.0]
    assert all(rec["box_source"] == "yolo_box" for rec in records)
    assert all(rec["geometry_source"] == "yolo" for rec in records)
    assert all(rec["geometry_method"] == "yolo_box" for rec in records)
    assert all(rec["box_backend"] == "yolo" for rec in records)
    assert all(rec["sign_source"] == "" for rec in records)

    normalizer_host = SimpleNamespace()
    assert normalize_character_box_source(normalizer_host, records[0]) == "yolo_box"


def test_yb_plus_gt_renders_as_yb_and_gt_not_gb():
    layers = get_preview_source_badge_layers(
        _badge_host(),
        "yolo_box",
        include_confidence=False,
        has_symbol=True,
        box_source="yolo_box",
        sign_source="gt_assisted",
    )
    texts = [layer["text"] for layer in layers]
    assert texts == ["YB", "GT"]
    assert "GB" not in texts


def test_source_counts_include_gt_assisted_symbols():
    host = _provenance_host()
    chars = [
        {
            "character": "A",
            "bbox": [1, 2, 10, 20],
            "source_tag": "yolo_box",
            "box_source": "yolo_box",
            "sign_source": "gt_assisted",
        },
        {
            "character": "B",
            "bbox": [11, 2, 20, 20],
            "source_tag": "generated_box",
            "box_source": "generated_box",
            "sign_source": "ocr_symbol",
        },
    ]
    counts = _count_character_sources(host, chars)
    assert counts["yolo_box"] == 1
    assert counts["generated_box"] == 1
    assert counts["gt_assisted"] == 1
    assert counts["ocr_symbol"] == 1

    line = format_preview_source_counts_line(host, counts)
    assert "YB=1" in line
    assert "GB=1" in line
    assert "GT=1" in line


def test_badge_helpers_have_single_top_level_definition():
    import auto_annotation_tool.gui.z3_preview_badges as badges

    tree = ast.parse(Path(badges.__file__).read_text(encoding="utf-8"))
    wanted = {
        "get_preview_badge_component_style",
        "get_preview_source_component_legend_items",
        "get_preview_source_badge_layers",
    }
    counts = {name: 0 for name in wanted}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in counts:
            counts[node.name] += 1
    assert counts == {name: 1 for name in wanted}
