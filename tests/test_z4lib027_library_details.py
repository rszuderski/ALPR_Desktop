"""Z4LIB027: no model mutation, only manifest-based readable library cards."""
from types import SimpleNamespace
from pathlib import Path
from auto_annotation_tool.gui.z4_mobile_artifact_details import (
    artifact_search_text, checkpoint_sha_by_role, format_artifact_card,
    format_event_card,
)

MT_SHA = "25636b5d97d9562f7381b719fbb5ec63ecd7630e16d31a824438171d889978cf"
MZ_SHA = "631bb1453554cdb6b15eca724baa392e2acffb350899107f892dc11f937dbbf8"


def sample_row(**override):
    d = dict(name="model.alprmodel", path="C:/models/model.alprmodel",
             kind="MODEL", role="MZ", model_id="MZ-261007-2336-390B",
             schema="alpr.model.v1", variants="TFLITE/FP32", size_bytes=10485760,
             modified_utc="2026-10-15T00:00:00+00:00", state="DISCOVERED_NOT_VERIFIED",
             error="", child_metadata={}, manifest={
                "role": "character", "task": "detect", "source": {"checkpoint_sha256": MZ_SHA},
                "input": {"width": 320, "height": 320, "channels": 3,
                          "layout": "NHWC", "color_space": "RGB", "data_type": "FLOAT32"},
                "output": {"output_format": "raw_yolo", "decoder": "ultralytics_raw",
                           "class_count": 36}})
    d.update(override)
    return SimpleNamespace(**d)


def test_readable_card_has_model_provenance_without_false_approval():
    row = sample_row()
    card = format_artifact_card(row)
    assert MZ_SHA in card and "320 × 320" in card and "FLOAT32" in card
    assert "DISCOVERED_NOT_VERIFIED" in card
    assert "NIEPOTWIERDZONA" in card
    assert "nie" in card.lower() and "android" in card.lower()


def test_only_valid_checkpoint_sha_is_searchable():
    row = sample_row()
    assert checkpoint_sha_by_role(row) == [("MZ", MZ_SHA)]
    assert MZ_SHA in artifact_search_text(row)
    bad = sample_row(manifest={"source": {"checkpoint_sha256": "XYZ"}})
    assert checkpoint_sha_by_role(bad) == []
    assert "BRAK" in format_artifact_card(bad)


def test_composite_manifest_sidecars_keep_roles_separate():
    row = sample_row(kind="PACKAGE", role="MT+MZ", model_id="bundle",
        manifest={"schema":"alpr.package.v1"},child_metadata={
            "plate": {"model_id":"MT-DAY", "checkpoint_sha256":MT_SHA,
                      "variants":"TFLITE/INT8", "input":{"width":512,"height":512}},
            "character": {"model_id":"MZ-DAY", "checkpoint_sha256":MZ_SHA,
                          "variants":"TFLITE/INT8", "input":{"width":320,"height":320}},
        })
    card=format_artifact_card(row)
    assert MT_SHA in card and MZ_SHA in card
    assert "MT: MT-DAY" in card and "MZ: MZ-DAY" in card
    assert "512 × 512" in card and "320 × 320" in card


def test_event_summary_is_only_historical():
    card=format_event_card({"operation":"SINGLE_MODEL_EXPORT", "outcome":"SUCCEEDED",
                            "artifact_sha256":MT_SHA})
    assert MT_SHA in card and "Historyczny" in card


def test_library_ui_keeps_full_validation_and_read_only_controls():
    text=(Path(__file__).resolve().parents[1]/
          "auto_annotation_tool/gui/z4_mobile_artifact_library.py").read_text(encoding="utf-8")
    modal=(Path(__file__).resolve().parents[1]/
           "auto_annotation_tool/gui/z4_mobile_details_dialog.py").read_text(encoding="utf-8")
    assert 'ttk.PanedWindow' not in text
    assert 'text="Podsumowanie"' in modal and 'json_label = "Manifest i źródła (JSON)"' in text
    assert 'text="Szczegóły…", command=self.show_details' in text
    assert 'command=self.validate_selected' in text
    assert 'sha_label = "Kopiuj SHA checkpointu"' in text
    assert 's.sha_value' in modal
    assert 'return self.copy_checkpoint_sha()' in text
    assert 'verify_unchanged_mobile_package(' in text
