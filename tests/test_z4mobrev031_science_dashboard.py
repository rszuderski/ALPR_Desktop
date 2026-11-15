"""Z4MOBREV031: present canonical ALPR metrics clearly and without reinterpretation."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import pytest

from auto_annotation_tool.gui.mobile_science_viewmodel import (
    STATUS_DISPLAY, MetricRow, make_science_dashboard,
)
from auto_annotation_tool.gui.mobile_science_summary_ui import MobileScienceSummaryPanel


def report():
    return {
        "source_archive_sha256": "a" * 64,
        "session_id": "test-31",
        "review_revision": 3,
        "review_mode": "assisted",
        "review_status": "IN_PROGRESS",
        "scientific_state": "PRELIMINARY",
        "conditions": {"scenario": "TV", "variant": "P1-R0-B", "device": "Galaxy A12"},
        "progress": {"subjects_total": 9, "subjects_mt_done": 0,
                     "mt_executed_active": 8, "mt_invocations_cardinality_reviewed": 0},
        "mt": {"single_plate_n": None, "localized_n": None, "success_rate": None,
               "no_detection_n": None},
        "pose": {"measured_n": 0, "polygon_iou_mean_pct": None, "nmce_mean_pct": None,
                 "manual_assisted_gt_n": 0, "assisted_corrections": {"assisted_correction_n": 0}},
        "mz": {"evaluable_reads": 0, "exact_n": 0, "exact_rate": None,
               "gt_characters": 0, "error_characters": 0, "cer": None},
        "alpr": {"evaluable_subjects": 0, "exact_n": 0, "exact_rate": None},
        "performance": {"latency": {
            "mt": {"n": 8, "p50_ms": 639.3, "p95_ms": 992.8},
            "mz": {"n": 8, "p50_ms": 228.5, "p95_ms": 459.1},
            "pipeline": {"n": 8, "p50_ms": 1384.3, "p95_ms": 2299.4},
        }},
        "caveats": ["Operator widział predykcje", "Brak niezależnego GT"],
        "methods": ["GT może być asystowany"],
    }


def form():
    return {"status": "IN_PROGRESS", "total": 9, "verified": 0,
            "issues": ["Brak GT numeru"]}


def row(model, name):
    return next(x for x in model["rows"] if x.metric_id == name)


def test_no_gt_does_not_mean_zero_effectiveness():
    vm = make_science_dashboard(report(), form())
    assert [c["value"] for c in vm["cards"]] == ["Brak wyniku", "Brak porównania", "Brak wyniku", "Brak wyniku"]
    assert row(vm, "mt_localization").formatted_value == "—"
    assert row(vm, "mz_exact").formatted_value == "—"
    assert row(vm, "mz_cer").formatted_value == "—"
    assert vm["progress"][0]["text"] == "0 z 8"
    assert vm["progress"][1]["text"] == "0 z 9"
    assert all("0%" not in entry.formatted_value for entry in vm["rows"])


def test_valid_measured_zero_is_zero_not_missing():
    r = report()
    r["mt"].update(single_plate_n=4, localized_n=0, success_rate=0.0, no_detection_n=4)
    r["mz"].update(evaluable_reads=3, exact_n=0, exact_rate=0.0)
    r["alpr"].update(evaluable_subjects=3, exact_n=0, exact_rate=0.0)
    vm = make_science_dashboard(r, form())
    assert row(vm, "mt_localization").formatted_value == "0 z 4 (0,00%)"
    assert row(vm, "mz_exact").formatted_value == "0 z 3 (0,00%)"
    assert row(vm, "alpr_exact").formatted_value == "0 z 3 (0,00%)"
    assert row(vm, "mt_no_detection").formatted_value == "4"


def test_missing_or_inconsistent_denominator_fail_closed():
    r = report()
    r["mt"].update(single_plate_n=0, localized_n=0, success_rate=0)
    r["mz"].update(evaluable_reads=2, exact_n=3, exact_rate=1.5)
    vm = make_science_dashboard(r, form())
    assert row(vm, "mt_localization").formatted_value == "—"
    assert row(vm, "mz_exact").formatted_value == "—"
    assert row(vm, "mz_exact").status == "MISSING"


def test_only_independent_pose_gt_is_eligible_for_main_iou_nmce():
    r = report()
    r["pose"].update(measured_n=2, polygon_iou_mean_pct=82.972,
                     nmce_mean_pct=4.503, manual_assisted_gt_n=3)
    vm = make_science_dashboard(r, form())
    assert row(vm, "pose_iou").formatted_value == "82,97%"
    assert row(vm, "pose_nmce").formatted_value == "4,50%"
    assert row(vm, "pose_iou").denominator == 2
    assert row(vm, "pose_iou").review_provenance == "independent"
    r["pose"].update(measured_n=0, polygon_iou_mean_pct=None, nmce_mean_pct=None)
    vm = make_science_dashboard(r, form())
    assert row(vm, "pose_iou").formatted_value == "—"
    assert vm["cards"][1]["value"] == "Brak porównania"


def test_mz_not_run_is_not_fabricated_as_ocr_error():
    r = report()
    r["progress"]["mz_not_run_subjects"] = 9
    vm = make_science_dashboard(r, form())
    assert row(vm, "mz_exact").formatted_value == "—"
    assert row(vm, "mz_cer").formatted_value == "—"
    assert not any("błąd modelu" in x for x in vm["why"])


def test_multiplate_operator_links_do_not_become_formal_tp():
    r = report()
    r["multi_plate"] = {"total_multi_invocations": 2, "verified_multi_invocations": 2,
                        "formal_status": "NOT_EVALUABLE",
                        "operator_diagnostic": {"plate_count": 5,
                                                "linked_predictions": 2,
                                                "unmatched_plate_annotations": 3}}
    f = {"status": "READY_FOR_EVALUATION", "total": 9, "verified": 9, "issues": []}
    vm = make_science_dashboard(r, f)
    formal = row(vm, "multi_formal")
    assert formal.formatted_value == "—"
    assert formal.status == "NOT_EVALUABLE"
    assert formal.numerator is None
    assert row(vm, "multi_links").formatted_value == "2"
    assert row(vm, "multi_links").status == "DIAGNOSTIC"
    assert vm["science_status"] == "Ocena naukowa nieukończona"
    assert vm["form_status"] == "Formularz zatwierdzony"


def test_only_form_complete_does_not_make_science_formal():
    r = report()
    r.update(review_status="COMPLETED", scientific_state="FORM_COMPLETE_SCIENCE_PENDING")
    f = {"status": "READY_FOR_EVALUATION", "total": 1, "verified": 1}
    vm = make_science_dashboard(r, f)
    assert vm["science_status"] == "Raport wstępny"
    assert vm["form_status"] == "Formularz zatwierdzony"


def test_cer_above_100_percent_is_not_clamped():
    r = report()
    r["mz"].update(gt_characters=5, error_characters=8, cer=1.6)
    vm = make_science_dashboard(r, form())
    assert row(vm, "mz_cer").formatted_value == "160,00%"


def test_latencies_available_without_gt_but_not_from_unverified_archive():
    vm = make_science_dashboard(report(), form())
    assert "639,3" in vm["latencies"][0]["value"]
    unverified = make_science_dashboard(report(), form(), source_verified=False)
    assert all(r.formatted_value == "—" for r in unverified["rows"])
    assert all(c["value"] == "Brak wyniku" for c in unverified["cards"])
    assert all(t["value"] == "Brak pomiaru" for t in unverified["latencies"])
    assert unverified["science_status"] == "Niezweryfikowane źródło"


def test_empty_old_report_is_safe_and_pure():
    assert make_science_dashboard({})["cards"][0]["value"] == "Brak wyniku"
    source = report()
    before = deepcopy(source)
    vm = make_science_dashboard(source, form())
    assert source == before
    assert all(isinstance(r, MetricRow) for r in vm["rows"])
    assert all(row.status in STATUS_DISPLAY for row in vm["rows"])


def test_progress_unknown_is_not_imputed_as_zero():
    r = report()
    r["progress"]["mt_executed_active"] = None
    assert make_science_dashboard(r, {})["progress"][0]["text"] == "Brak danych"


def test_no_fields_added_to_archives_or_gt_outputs():
    r = report()
    snapshot = deepcopy(r)
    make_science_dashboard(r, form())
    assert r == snapshot
    assert "derived_metrics" not in r


def _new_ui():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Brak wyświetlacza Tk")
    root.geometry("1050x650+0+0")
    outer = tk.Frame(root)
    outer.pack(fill="both", expand=True)
    panel = MobileScienceSummaryPanel(
        outer, background="#F5F5ED", surface="#EEEFE7", foreground="#24322A",
        muted="#596C60", accent="#527948", border="#B4B9AA")
    panel.frame.pack(fill="both", expand=True)
    return root, panel


def test_tk_table_is_selectable_and_displays_real_metrics():
    root, panel = _new_ui()
    try:
        panel.render(report(), form())
        root.update()
        assert panel.tree.exists("pose_iou")
        assert panel.tree.exists("mz_cer")
        assert "Brak danych" in panel.tree.item("pose_iou", "values")[4]
        panel.tree.selection_set("mz_exact")
        panel.tree.event_generate("<<TreeviewSelect>>")
        root.update()
        assert "mz.exact_n" in panel._table_description.cget("text")
        assert panel.scrollbar.winfo_ismapped()
        assert panel.canvas.cget("yscrollcommand")
    finally:
        root.destroy()


def test_tk_small_window_single_outer_scroll_and_expanders():
    root, panel = _new_ui()
    try:
        root.geometry("660x380+0+0")
        panel.render(report(), form())
        root.update()
        assert panel.canvas.bbox("all")[3] > panel.canvas.winfo_height()
        before = panel.canvas.yview()[0]
        panel._wheel(type("E", (), {"delta": -120, "num": "??"})())
        root.update()
        assert panel.canvas.yview()[0] >= before
        # The technical tree is NOT nested inside the new dashboard.
        assert sum(isinstance(w, ttk.Treeview) for w in panel.body.winfo_children()) == 0
    finally:
        root.destroy()


def test_science_tab_contains_backward_compatible_technical_toggle():
    source = Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/z4_mobile_sample_review.py"
    text = source.read_text(encoding="utf-8")
    assert "MobileScienceSummaryPanel" in text
    assert "self._science_raw_host.grid(" in text
    assert "self.science_dashboard.render(self._science_data, form, source_verified=source_valid)" in text
    assert "self.science_dashboard.frame.grid_remove()" in text
    assert "display_scientific_summary" in text  # underlying export-compatible summary retained
    assert "def _toggle_scientific_details(self):" in text
