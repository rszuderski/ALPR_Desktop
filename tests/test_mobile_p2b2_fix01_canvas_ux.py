"""P2B2 FIX01: removal/reset, persistent corner handles, checkbox and live UI.

All images/archives here are synthetic. No model, training, or user data needed.
Native GUI tests require a desktop display (or Xvfb); core tests do not.
"""
from __future__ import annotations

import copy
import io
import json
import zipfile
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from test_mobile_p2b2_canvas_review import source_session, MANUAL, text_all
from test_mobile_p2b2_canvas_gui import gui, pump, key, click, select
from test_mobile_human_review import attempt, make_bundle, load
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory
from auto_annotation_tool.ranking.mobile_verification_policy import observation_verification_cells, verification_issues
from auto_annotation_tool.ranking.mobile_package_experiments import _file_sha256


def two_scene_session(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows, samples = [], []
    for i in range(2):
        row = attempt(f"t{i}", subject=f"s/sg-1/entity-{i+1}", prediction=f"WA{i}1234")
        row.update(mt_invocation_id=f"g{i}", mt_executed=True, mt_detection_index=0,
            mt_detection_count=1, source_sequence=17+i, source_timestamp_nanos=100000+i,
            scene_generation=1, visual_epoch=1, camera_transform_generation=1,
            source_frame_expected=True, source_frame_entry=f"samples/frames/source{i}.jpg",
            source_frame_status="WRITTEN", source_frame_coordinate_space="mt_source_bitmap_xy",
            source_frame_origin="pipeline_bitmap_before_roi", source_frame_width_px=1000,
            source_frame_height_px=700, roi_left=100, roi_top=60, roi_right=900, roi_bottom=640,
            input_width=800, input_height=580, input_scale=1, input_pad_x=0, input_pad_y=0,
            mt_input_evidence_entry=row["evidence_entry"], mt_pose_status="VALID_QUAD",
            mt_pose_coordinate_space="source_frame_normalized_v1", mt_pose_corner_order="TL_TR_BR_BL",
            mt_pose_source_width_px=1000, mt_pose_source_height_px=700,
            mt_pose_corners_norm="0.15:0.2;0.30:0.2;0.30:0.3;0.15:0.3")
        rows.append(row)
        samples.append(dict(capture_id=f"c{i}", attempt_id=f"t{i}", subject_key=row["subject_key"],
                            prediction=row["prediction"], mz_status="READ"))
    path=make_bundle(tmp_path/"source.alprsession", attempts=rows, samples=samples, image_size=(800,580))
    with zipfile.ZipFile(path,"a") as z:
        for i in range(2):
            out=io.BytesIO();Image.new("RGB", (1000,700), (80+i*20,95,105)).save(out,format="JPEG")
            z.writestr(f"samples/frames/source{i}.jpg",out.getvalue())
    session=load(path,tmp_path);session.set_review_mode("assisted")
    return session


def test_remove_manual_is_not_a_flag_and_leaves_auditable_event(tmp_path):
    s=source_session(tmp_path);before_parents=core.scene(s,"mt-one")["parents"]
    sha=_file_sha256(s.path)
    oid=core.add_plate(s,"mt-one",MANUAL)
    core.set_text(s,"mt-one",oid,"TEST123")
    core.toggle_error(s,"mt-one",oid)
    deleted=copy.deepcopy(core._object(core.scene(s,"mt-one"),oid))
    assert core.remove_operator_geometry(s,"mt-one",oid)=="remove_manual_plate"
    scene=core.scene(s,"mt-one")
    assert oid not in {o["object_id"] for o in scene["objects"]}
    assert core.counts(scene)==dict(expected=2,mt=2,linked=2,missed=0,false_positive=0,operator_errors=0)
    event=scene["events"][-1]
    assert event["object_before"]==deleted and event["object_after"] is None
    assert event["action"]=="remove_manual_plate"
    assert scene["parents"]==before_parents and _file_sha256(s.path)==sha
    reopened=load(s.path,tmp_path)
    assert core.scene(reopened,"mt-one")["events"][-1]==event
    assert all(r["object_id"]!=oid for r in core.summary(reopened)["rows"])


def test_reset_child_preserves_parent_gt_and_false_positive_decision(tmp_path):
    s=source_session(tmp_path);text_all(s)
    original=copy.deepcopy(core.scene(s,"mt-one")["parents"])
    quad=copy.deepcopy(original[0]["quad"]);quad[0][0]+=14
    core.correct_plate(s,"mt-one","GT:a0",quad)
    core.toggle_error(s,"mt-one","GT:a0")
    prior=core._object(core.scene(s,"mt-one"),"GT:a0")
    before=core.counts(core.scene(s,"mt-one"))
    frozen=s.sidecar_path.read_bytes()
    with pytest.raises(ValueError,match='wspólnej GT'):core.remove_operator_geometry(s,"mt-one","GT:a0")
    scene=core.scene(s,"mt-one");obj=core._object(scene,"GT:a0")
    assert scene["parents"]==original and not scene["checked"]
    assert obj["gt_text"]==prior["gt_text"] and obj["error"] is True
    assert core.counts(scene)==before and s.sidecar_path.read_bytes()==frozen
    assert scene['physical_gt_v1']['plates'][0]['quad']==quad


@pytest.mark.parametrize("oid",["unknown","MT:a0",""])
def test_remove_invalid_target_is_atomic(tmp_path,oid):
    s=source_session(tmp_path);before=s.review.to_dict()
    with pytest.raises(ValueError):core.remove_operator_geometry(s,"mt-one",oid)
    assert s.review.to_dict()==before


def test_deletion_excludes_manual_from_csv_but_not_from_audit(tmp_path):
    s=source_session(tmp_path);text_all(s)
    oid=core.add_plate(s,"mt-one",MANUAL);core.remove_operator_geometry(s,"mt-one",oid)
    core.check_scene(s,"mt-one")
    core.export(s,tmp_path)
    csv=(tmp_path/"canvas_review_plates.csv").read_text(encoding="utf-8-sig")
    assert oid not in csv
    assert oid in json.dumps(s.review.to_dict())
    assert core.summary(s)["missed_plates"]==0


def test_checked_with_missing_gt_is_inspected_but_cannot_complete_or_enter_metrics(tmp_path):
    s=source_session(tmp_path)
    core.set_scene_review_state(s,"mt-one",checked=True)
    assert core.scene(s,"mt-one")["checked"]
    assert verification_issues(s)
    with pytest.raises(ValueError):s.complete()
    d=core.summary(s)
    assert d["inspected_scenes"]==1 and d["incomplete_checked_scenes"]==1
    assert d["checked_scenes"]==0 and d["evaluated_reads"]==0 and d["cer"] is None


def test_excluded_implies_inspected_unexclude_keeps_inspection_and_missing_gt(tmp_path):
    s=source_session(tmp_path)
    core.set_scene_review_state(s,"mt-one",checked=True,exclusion_reason="Odbicie uniemożliwia ocenę")
    assert not verification_issues(s)
    d=core.summary(s)
    assert d["inspected_scenes"]==1 and d["excluded_scenes"]==1 and d["checked_scenes"]==0
    core.set_scene_review_state(s,"mt-one",checked=True,exclusion_reason="")
    assert core.scene(s,"mt-one")["checked"] and verification_issues(s)
    assert core.summary(s)["incomplete_checked_scenes"]==1


@pytest.mark.parametrize("checked,reason",[(False,"excluded"),(1,""),(True,"x"*241)])
def test_checkbox_inconsistent_state_is_rejected_atomically(tmp_path,checked,reason):
    s=source_session(tmp_path);before=s.review.to_dict()
    with pytest.raises(ValueError):core.set_scene_review_state(s,"mt-one",checked=checked,exclusion_reason=reason)
    assert s.review.to_dict()==before


def test_uncheck_and_recheck_preserve_per_plate_text_and_original(tmp_path):
    s=source_session(tmp_path);text_all(s);core.check_scene(s,"mt-one")
    prior=copy.deepcopy(core.scene(s,"mt-one"))
    core.set_scene_review_state(s,"mt-one",checked=False)
    assert core.scene(s,"mt-one")["objects"]==prior["objects"]
    assert core.group_issues(s,"mt-one")
    core.set_scene_review_state(s,"mt-one",checked=True)
    assert not core.group_issues(s,"mt-one")
    assert core.summary(s)["evaluated_reads"]==2


def test_matrix_shows_partial_gt_before_whole_roi_inspection(tmp_path):
    s=source_session(tmp_path)
    key=next(iter(s.subjects))
    core.set_text(s,"mt-one","GT:a0","WA01234")
    cells=observation_verification_cells(s,key)
    assert cells[0]=="○" and cells[2]=="◐" and cells[-1]=="○"
    core.set_text(s,"mt-one","GT:a1","WA11234")
    cells=observation_verification_cells(s,key)
    assert cells[2]=="✓" and cells[0]=="○" and cells[-1]=="○"
    core.set_scene_review_state(s,"mt-one",checked=True)
    assert observation_verification_cells(s,key)==("✓",)*5


def test_excluded_matrix_does_not_pretend_to_have_gt(tmp_path):
    s=source_session(tmp_path)
    core.exclude_scene(s,"mt-one","Nieczytelny obraz")
    assert observation_verification_cells(s,next(iter(s.subjects)))==("✓","—","—","—","✓")


def test_inventory_includes_unopened_frames_without_writing(tmp_path):
    s=two_scene_session(tmp_path);before=s.review.to_dict()
    initial=build_inventory(s)
    assert initial["total"]==2 and initial["editable"]==2 and initial["remaining"]==2
    assert initial["inspected"]==0 and s.review.to_dict()==before
    core.set_scene_review_state(s,"g0",checked=True,exclusion_reason="Nie można ocenić")
    d=build_inventory(s)
    assert d["total"]==2 and d["inspected"]==1 and d["excluded"]==1 and d["remaining"]==1
    assert not core.has_scene(s,"g1")


def test_inventory_reports_missing_source_without_certifying_it(tmp_path):
    s=two_scene_session(tmp_path)
    s.entry_names.remove("samples/frames/source1.jpg")
    d=build_inventory(s)
    assert d["total"]==2 and d["editable"]==1 and d["unavailable"]==1
    assert d["inspected"]==0 and d["groups"]["g1"]["status"]=="UNAVAILABLE"


def test_native_r_removes_only_selected_manual_and_undo_restores_it(gui):
    root,h,ed,s=gui
    key(h,"d")
    for p in MANUAL:click(h,ed,p)
    oid=ed.selected
    key(h,"r");pump(root)
    assert core.counts(core.scene(s,"mt-one"))["expected"]==2
    assert oid not in ed.badges
    assert len(h.canvas.find_withtag("mt_original"))==2
    assert not any(target[2]==oid for target in ed.tree_targets.values())
    ed.undo();pump(root)
    assert oid in ed.badges and core.counts(core.scene(s,"mt-one"))["expected"]==3
    ed.undo(redo=True);pump(root)
    assert oid not in ed.badges


def test_native_r_reset_keeps_two_overlays_gt_and_is_immediately_draggable(gui):
    root,h,ed,s=gui
    ed.start_entry("GT:a0");ed.entry_var.set("WA01234");ed.entry_enter();pump(root)
    select(h,ed,(225,175))
    orig=copy.deepcopy(core.scene(s,"mt-one")["parents"])
    x,y=ed.to_canvas(orig[0]["quad"][0])
    h.canvas.event_generate("<ButtonPress-1>",x=round(x),y=round(y))
    h.canvas.event_generate("<B1-Motion>",x=round(x+12),y=round(y+5))
    h.canvas.event_generate("<ButtonRelease-1>",x=round(x+12),y=round(y+5));pump(root)
    assert core.scene(s,"mt-one")["objects"][0]["quad"]!=orig[0]["quad"]
    key(h,"r");pump(root)
    obj=core.scene(s,"mt-one")["objects"][0]
    assert obj["quad"]==orig[0]["quad"] and obj["gt_text"]=="WA01234"
    assert ed.mode=="edit" and len(h.canvas.find_withtag("corner_handle"))>=4
    assert ed.hit(x,y)[1]=="corner"
    assert len(h.canvas.find_withtag("operator_quad"))==2


def test_native_selected_corner_stays_visible_outside_quad_without_hover_selection(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175))
    x,y=ed.to_canvas(core.scene(s,"mt-one")["parents"][0]["quad"][0])
    # First hover inside, then move to the OUTSIDE half of the circular handle.
    for dx,dy in ((3,3),(-2,-2),(-6,-5),(-1,-4)):
        h.canvas.event_generate("<Motion>",x=round(x+dx),y=round(y+dy));pump(root)
        assert ed.selected=="GT:a0" and not ed.hover
        assert len(h.canvas.find_withtag("corner_handle"))>=4
    h.canvas.event_generate("<ButtonPress-1>",x=round(x-2),y=round(y-2));pump(root)
    assert ed.drag and ed.drag["oid"]=="GT:a0" and ed.mode=="edit"
    h.canvas.event_generate("<B1-Motion>",x=round(x+10),y=round(y+5))
    h.canvas.event_generate("<ButtonRelease-1>",x=round(x+10),y=round(y+5));pump(root)
    assert core.scene(s,"mt-one")["objects"][0]["geometry_state"]=="operator_corrected"
    assert len(core.scene(s,"mt-one")["objects"])==2


def test_selected_handles_have_hit_and_paint_priority_over_badges(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175))
    x,y=ed.to_canvas(core.scene(s,"mt-one")["objects"][0]["quad"][0])
    ed.badges["GT:a1"]=(x-10,y-10,x+10,y+10)
    assert ed.hit(x,y)==("GT:a0","corner",0)
    all_items=list(h.canvas.find_all())
    assert min(all_items.index(i) for i in h.canvas.find_withtag("corner_handle")) > max(
        all_items.index(i) for i in h.canvas.find_withtag("gt_badge"))


def test_r_typed_in_gt_is_text_not_delete(gui):
    root,h,ed,s=gui
    select(h,ed,(225,175))
    ed.start_entry("GT:a0");pump(root)
    ed.entry.delete(0,"end");ed.entry.focus_force();pump(root)
    ed.entry.event_generate("<KeyPress-r>");ed.entry.event_generate("<KeyRelease-r>");pump(root)
    assert ed.entry_var.get().upper()=="R"
    assert len(core.scene(s,"mt-one")["objects"])==2
    ed.entry_enter();pump(root)
    assert core._object(core.scene(s,"mt-one"),"GT:a0")["gt_text"]=="R"


def test_r_outside_selection_mode_does_not_remove(gui):
    root,h,ed,s=gui
    ed.selected="GT:a0";ed.mode="select";before=s.review.to_dict()
    key(h,"r")
    assert s.review.to_dict()==before
    key(h,"d");click(h,ed,MANUAL[0]);key(h,"r")
    assert len(ed.points)==1 and s.review.to_dict()==before


def test_native_checkboxes_inspection_and_exclusion_are_consistent_and_live(gui):
    root,h,ed,s=gui
    assert isinstance(ed.approve,tk.Checkbutton) and isinstance(ed.exclude,tk.Checkbutton)
    ed.exclude.invoke();pump(root)
    assert ed.checked_var.get() and ed.excluded_var.get()
    assert core.scene(s,"mt-one")["checked"] and core.scene(s,"mt-one")["exclusion_reason"]
    row=next(iter(h._subject_ids))
    assert h.subject_tree.item(row,"values")[1:]==("✓","—","—","—","✓")
    assert "1/1" in h.progress_label.cget("text")
    ed.exclude.invoke();pump(root)
    assert ed.checked_var.get() and not ed.excluded_var.get()
    assert core.group_issues(s,"mt-one")  # GT still required; no fabricated success.
    ed.approve.invoke();pump(root)
    assert not ed.checked_var.get() and not ed.excluded_var.get()
    assert "0/1" in h.progress_label.cget("text")


def test_native_gt_updates_left_partial_cells_without_waiting_for_worker(gui):
    root,h,ed,s=gui
    root_iid=next(iter(h._subject_ids))
    ed.start_entry("GT:a0");ed.entry_var.set("WA01234");ed.entry_enter();pump(root)
    assert h.subject_tree.item(root_iid,"values")[3]=="◐"
    target=next(i for i,t in ed.tree_targets.items() if t[2]=="GT:a0")
    assert "WA01234" in h.subject_tree.item(target,"text")
    ed.start_entry("GT:a1");ed.entry_var.set("WA11234");ed.entry_enter();pump(root)
    assert h.subject_tree.item(root_iid,"values")[3]=="✓"
    assert h.subject_tree.item(root_iid,"values")[-1]=="○"


def test_left_child_selection_selects_canvas_overlay_not_historical_form(gui):
    root,h,ed,s=gui
    target=next(i for i,t in ed.tree_targets.items() if t[2]=="GT:a1")
    h.subject_tree.selection_set(target);h.subject_tree.event_generate("<<TreeviewSelect>>");pump(root)
    assert ed.selected=="GT:a1" and not h._multi_host.winfo_ismapped()
    assert ed.mode=="edit"  # explicit list selection enables its existing handles
    assert len(h.canvas.find_withtag("corner_handle"))>=4


def test_sidebar_wheel_on_new_child_and_focus_misdelivery_do_not_zoom_image(gui):
    root,h,ed,s=gui
    h.window.geometry("1180x760+0+0");pump(root)
    assert not h.review_sidebar_scroll.winfo_ismapped() and ed.contexts.winfo_ismapped()
    zoom=h._view_zoom
    ed.contexts.fields['image'].event_generate("<MouseWheel>",delta=-120);pump(root)
    assert h._view_zoom==zoom
    # Windows can deliver wheel to focused canvas while pointer is over sidebar.
    x=ed.contexts.winfo_rootx()+20-h.canvas.winfo_rootx()
    y=ed.contexts.winfo_rooty()+50-h.canvas.winfo_rooty()
    h.canvas.event_generate("<Motion>",x=x,y=y,warp=True);pump(root)
    try:
        h.canvas.focus_force()
        h.canvas.event_generate("<MouseWheel>",x=x,y=y,delta=-120);pump(root)
        assert h._view_zoom==zoom
    finally:
        h.canvas.event_generate("<Motion>",x=25,y=25,warp=True);pump(root)


def test_canvas_report_has_kpi_cards_and_short_units_not_text_wall(gui):
    root,h,ed,s=gui
    text_all(s);core.check_scene(s,"mt-one")
    h.stats=s.statistics();ed.cache_key=None;ed.refresh_tree();ed.update_report()
    ed.report_host.master.select(ed.report_host);pump(root)
    cards=ed.report_host.cards
    assert len(cards)==11 and cards["cer"].value.cget("text")=="0,00%"
    assert cards["inspected"].value.cget("text")=="1 / 1"
    assert "14" in cards["cer"].note.cget("text")
    assert not hasattr(ed,"report_view")
    assert cards["cer"].value.cget("fg")!=cards["iou"].value.cget("fg")
    # Restore tab for fixture navigation/disposal.
    ed.report_host.master.select(0);pump(root)


def test_report_reflows_without_losing_measures(gui):
    root,h,ed,s=gui
    tabs=ed.report_host.master;tabs.select(ed.report_host)
    h.window.geometry("1080x670+0+0");pump(root)
    assert all(grid._columns<=maximum for grid,cards,maximum in ed.report_host.grids)
    assert ed.report_host.canvas.bbox("all")[3]>ed.report_host.canvas.winfo_height()
    before=ed.report_host.canvas.yview()[0]
    ed.report_host.cards["cer"].title.event_generate("<MouseWheel>",delta=-120);pump(root)
    assert ed.report_host.canvas.yview()[0]>before
    tabs.select(0);pump(root)


@pytest.fixture
def multiple_gui(tmp_path,monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    monkeypatch.setattr(native,"source_session",two_scene_session)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)


def test_selecting_different_image_child_preserves_target_and_previous_gt(multiple_gui):
    root,h,ed,s=multiple_gui
    ed.start_entry("GT:t0");ed.entry_var.set("NEWGT0");ed.entry_enter();pump(root)
    target=next(i for i,t in ed.tree_targets.items() if t[1:3]==("g1","GT:t1"))
    h.subject_tree.selection_set(target);h.subject_tree.event_generate("<<TreeviewSelect>>")
    pump(root,lambda:h.current_invocation=="g1" and h._image_entry=="samples/frames/source1.jpg")
    assert ed.selected=="GT:t1"
    assert core._object(core.scene(s,"g0"),"GT:t0")["gt_text"]=="NEWGT0"
    assert ed.mode=="edit" and len(h.canvas.find_withtag("corner_handle"))>=4


def test_inventory_header_counts_all_images_before_editing(multiple_gui):
    root,h,ed,s=multiple_gui
    assert "0/2" in h.progress_label.cget("text")
    assert ed.inventory()["total"]==2 and ed.inventory()["remaining"]==2
    ed.exclude.invoke();pump(root)
    assert "1/2" in h.progress_label.cget("text")
    assert "nie do oceny: 1" in h.progress_label.cget("text")


def test_changed_revision_hides_old_metrics_until_worker_publishes(gui):
    root,h,ed,s=gui
    text_all(s);core.check_scene(s,"mt-one")
    h.stats=s.statistics();ed.cache_key=None;ed.refresh_tree();ed.update_report()
    assert ed.report_host.cards["cer"].value.cget("text")=="0,00%"
    select(h,ed,(225,175));key(h,"r");pump(root)
    assert ed.report_host.cards["cer"].value.cget("text")=="—"
    assert "Przeliczanie" in ed.report_host.notice.cget("text")
    pump(root,lambda: not ed.refresh_after and not h._saving and not ed.in_refresh)
    assert "Przeliczanie" not in ed.report_host.notice.cget("text")
    assert ed.report_host.cards["cer"].value.cget("text")=="—"  # now unconfirmed ROI, not a stale 0%.


def test_pending_exclusion_reason_saved_before_keyboard_navigation(multiple_gui):
    root,h,ed,s=multiple_gui
    ed.exclude.invoke();pump(root)
    ed.reason_entry.focus_force();pump(root)
    ed.reason_var.set("Nieczytelny obraz — odbicia")
    ed.navigate(1)
    pump(root,lambda:h.current_invocation=="g1" and h._image_entry=="samples/frames/source1.jpg")
    assert core.scene(s,"g0")["exclusion_reason"]=="Nieczytelny obraz — odbicia"
    assert core.scene(s,"g0")["checked"] is True
    assert not core.scene(s,"g1")["checked"]


def test_invalid_pending_exclusion_reason_does_not_get_lost_on_navigation(multiple_gui):
    root,h,ed,s=multiple_gui
    ed.exclude.invoke();pump(root)
    ed.reason_entry.focus_force();pump(root);ed.reason_var.set("x"*241)
    revision=s.review.review_revision
    assert ed.prepare_navigation() is False
    assert s.review.review_revision==revision and h.current_invocation=="g0"
    assert ed.reason_var.get()=="x"*241
    ed.reason_var.set("Poprawiono opis");ed.save_exclusion_reason();pump(root)


def test_historical_dashboard_uses_disclosures_and_scrolls(gui):
    from auto_annotation_tool.gui.mobile_canvas_ui import Disclosure
    root,h,ed,s=gui
    tabs=ed.report_host.master;tabs.select(2)
    h.window.geometry("1080x670+0+0");pump(root)
    panel=h.science_dashboard
    def descendants(widget):
        for child in widget.winfo_children():
            yield child;yield from descendants(child)
    sections=[w for w in descendants(panel.body) if isinstance(w,Disclosure)]
    assert len(sections)>=5
    assert all(isinstance(section.header,tk.Label) for section in sections)
    assert not any(isinstance(w,ttk.Button) for w in descendants(panel.body))
    section=next(w for w in sections if "metodyka" in w.title.lower())
    section.toggle();pump(root)
    assert section.opened and section.body.winfo_ismapped()
    panel.canvas.yview_moveto(0)
    section.header.event_generate("<MouseWheel>",delta=-120);pump(root)
    assert panel.canvas.yview()[0]>0
    tabs.select(0);pump(root)
