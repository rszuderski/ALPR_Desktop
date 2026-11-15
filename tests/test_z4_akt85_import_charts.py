"""AKT85 — self-contained tests for the integrated mobile charts/comparison.

No .alprsession files, GT records, Desktop store or research outputs are changed.
"""
from __future__ import annotations
from types import SimpleNamespace
from pathlib import Path
from tkinter import ttk
import tkinter as tk
import hashlib
import json
import pytest

from auto_annotation_tool.gui.mobile_import_comparison import record_from_import, export_imported_charts
from auto_annotation_tool.gui.mobile_comparison_core import figures, validate_selection
from auto_annotation_tool.gui.mobile_import_trace_charts import trace_points


def make_report(sid="exp-akt85-one", sha="a"*64, variant="fp32", scenario="PAIR_DAY_01"):
    report=SimpleNamespace(report_id=sid,source_archive_sha256=sha)
    rows=[{"elapsed_ms":"1000", "status":"preliminary", "plate_inference_ms":"450",
           "character_inference_ms":"800", "mz_runs":"1", "total_ms":"1800"},
          {"elapsed_ms":"2000", "status":"no_plate", "plate_inference_ms":"500",
           "character_inference_ms":"", "mz_runs":"0", "total_ms":"1550"},
          {"elapsed_ms":"3000", "status":"scene_superseded", "plate_inference_ms":"",
           "character_inference_ms":"0", "mz_runs":"0", "total_ms":"500"}]
    b=SimpleNamespace(report=report,source_archive_sha256=sha,path="",
        report_payload={"device":{"model":"SM-A125F"},
                        "app_build":{"apk_identity":{"base_apk_sha256":"c"*64}},
                        "execution":{"plate":{"precision":variant,"checkpoint_sha256":"d"*64},
                                     "character":{"precision":variant,"checkpoint_sha256":"e"*64}}},
        collection_session={"session_id":sid,"scenario_id":scenario,"replicate_index":1},
        experiment_session={},model_refs={},pipeline_manifests={},trace_rows=rows,trace_total=len(rows),
        validation=SimpleNamespace(ok=True))
    return report,b


def test_precisions_and_runtime_denominators():
    r,b=make_report();s,rows,w=record_from_import(r,b)
    assert s.precision=="fp32" and s.apk_sha256=="c"*64
    assert (s.latency["mt"].n,s.latency["mz"].n,s.latency["pipeline"].n)==(2,1,2)
    assert s.latency["mt"].p50_ms==475
    assert not s.canvas_ready and s.canvas_exact_pct is None
    assert w==[]


def test_invalid_sources_block_before_charts():
    r,b=make_report()
    b.validation.ok=False
    with pytest.raises(ValueError,match="walidacji"):
        record_from_import(r,b)
    b.validation.ok=True
    b.source_archive_sha256="b"*64
    with pytest.raises(ValueError,match="SHA"):
        record_from_import(r,b)


def test_mz_without_inference_not_zero():
    assert trace_points([{"elapsed_ms":"100","mz_runs":"0","character_inference_ms":"0"}],
                        "character_inference_ms")==[]


def test_day_and_night_are_separate_and_quality_unavailable():
    a,b=make_report()
    a2,b2=make_report("exp-akt85-two","b"*64,"int8","PAIR_NIGHT_01")
    x,_,_=record_from_import(a,b)
    y,_,_=record_from_import(a2,b2)
    import matplotlib.pyplot as plt
    plotted=list(figures([x,y]))
    try:
        assert len(plotted)==12
        assert all("assisted" not in n for n,_ in plotted)
        assert any("DAY/NIGHT" in s for s in validate_selection([x,y]))
    finally:
        for _,fig in plotted:plt.close(fig)


def test_export_is_versioned_and_checks_hashes(tmp_path):
    dest=tmp_path/"03_wyniki";dest.mkdir();(dest/"porownanie").mkdir()
    a,b=make_report();r,rows,_=record_from_import(a,b)
    one=export_imported_charts([r],[rows],dest)
    two=export_imported_charts([r],[rows],dest)
    assert one!=two and one.parent==dest/r.session_id
    manifest=json.loads((one/"wykresy_manifest.json").read_text("utf-8"))
    assert manifest["sources"][0]["source_archive_sha256"]==a.source_archive_sha256
    assert (one/"czas_seria_plate_inference_ms.svg").exists()
    for obj in manifest["files"]:
        target=one/obj["name"]
        assert hashlib.sha256(target.read_bytes()).hexdigest()==obj["sha256"]
    with pytest.raises(ValueError,match="samej"):
        export_imported_charts([r,r],[rows,rows],dest)


def test_live_tk_comparison_dialog_and_import_source_choice():
    try:root=tk.Tk()
    except tk.TclError:pytest.skip("No Tk display")
    root.withdraw()
    try:
        from auto_annotation_tool.gui.mobile_import_comparison_gui import ImportedMobileComparisonWindow
        r,b=make_report();s,rows,_=record_from_import(r,b)
        dialog=ImportedMobileComparisonWindow(root,[s],[rows],imported_sources=[(r,b)])
        root.update()
        assert dialog.plot_choice.current()==0
        assert len(dialog.figure_items)>=9
        dialog._close()
    finally:
        root.destroy()
