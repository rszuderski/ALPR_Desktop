"""Native Tk Z4MOBREV008 probe: 4 clicks, drag, zoom, persistence and clearing.

Synthetic in-memory MT input; no archives, GT sidecars or real experiments touched.
Run separately on Windows: py -3 .\\tests\\gui_mobile_pose_gt_probe.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_annotation_tool.gui.z4_mobile_sample_review import (
    MobileSampleReviewWindow, _linked_pose_quad_on_mt_input, _pose_geometry_rows,
)


class FakeSession:
    def __init__(self):
        entry = "samples/evidence/a1.jpg"
        self.attempts = {"a1": dict(id="a1", attempt_id="a1", mt_invocation_id="i1", mt_status="VALID_QUAD",
            mt_detection_index="0", input_width="512", input_height="512", roi_left="0", roi_top="0",
            roi_right="512", roi_bottom="512", input_scale="1", input_pad_x="0", input_pad_y="0",
            plate_left="100", plate_top="190", plate_right="340", plate_bottom="250",
            mt_input_evidence_entry=entry)}
        self.samples = {"c1": dict(attempt_id="a1", image_entry="samples/crops/c1.jpg",
            plate_geometry={"available":True,"source_width_px":512,"source_height_px":512,
                "plate_corners_norm":[[100/512,190/512],[340/512,190/512],
                    [340/512,250/512],[100/512,250/512]]})}
        self.sample_ids_by_attempt = {"a1":["c1"]}
        self.mt_invocations = {"i1":SimpleNamespace(records=(self.attempts["a1"],),
            evidence_entries=(entry,),executed=True,cancelled=False,execution_failed=False,
            detection_count=1)}
        self.review = SimpleNamespace(source_archive_sha256="f"*64, attempt_annotations={})
        self.session_id = "gui-pose-mock"
        self.calls = []

    def cancelled(self,row): return False
    def annotate(self,kind,key,**kwargs):
        assert kind == "attempt" and key == "a1"
        self.calls.append(kwargs)
        self.review.attempt_annotations.setdefault(key,{}).update(kwargs)


def run():
    root=tk.Tk()
    root.title("Pose GT native GUI probe")
    root.geometry("900x800")
    ui = MobileSampleReviewWindow.__new__(MobileSampleReviewWindow)
    ui.session = FakeSession()
    ui.subject_key = "mock"
    ui.current_record = ("sample","c1")
    ui.current_invocation = "i1"
    ui._saving = False
    ui._image_entry = "samples/evidence/a1.jpg"
    ui._image_source_size = (512,512)
    ui._image = Image.new("RGB",(512,512),"#242e38")
    d=ImageDraw.Draw(ui._image); d.polygon([(90,184),(345,180),(350,257),(94,258)],fill="#b5a994")
    ui._view_zoom = 1.0
    ui._view_center = None
    ui._view_drag_at = None
    ui._view_fit_scale = 1.0
    ui._view_pixel_scale = 1.0
    ui._pose_gt_cache_token = None
    ui._pose_gt_points = []
    ui._pose_gt_drag_index = None
    ui._pose_gt_before_drag = None
    ui.pose_edit_var = tk.BooleanVar(root,value=True)
    ui.pose_edit_checkbox = ttk.Checkbutton(root,text="Edit", variable=ui.pose_edit_var)
    ui.pose_gt_reset = ttk.Button(root,text="Reset")
    ui.pose_gt_hint = ttk.Label(root)
    ui.pose_gt_progress = ttk.Label(root)
    ui.pose_gt_metric = ttk.Label(root)
    ui.status_var = tk.StringVar(root,value="")
    ui.zoom_value = ttk.Label(root)
    ui.canvas = tk.Canvas(root,width=600,height=580,bg="#11171d",highlightthickness=0)
    ui.canvas.pack(expand=True,fill="both")
    ui._change = lambda work,**kwargs: work()
    root.update_idletasks();root.update()
    ui._paint_image()
    assert _linked_pose_quad_on_mt_input(ui.session,ui.session.attempts["a1"]) is not None
    assert ui._pose_gt_attempt_context()=="a1"
    assert not ui.canvas.find_withtag("detection_polygon"), "Model must be hidden during first GT labels"
    gt=((95,186),(348,186),(348,258),(95,258))
    for i,point in enumerate(gt,1):
        cx,cy=ui._pose_mt_to_canvas(point,"a1")
        ev=SimpleNamespace(x=cx,y=cy)
        ui._pan_begin(ev)
        ui._pan_end(ev)
        assert len(ui._pose_gt_points)==i
    assert ui.session.calls and ui.session.review.attempt_annotations["a1"]["is_plate"] is True
    assert len(ui.session.review.attempt_annotations["a1"]["pose_gt_quad_mt_input_px"])==4
    assert len(ui.canvas.find_withtag("pose_gt_corner")) == 4
    assert ui.canvas.find_withtag("detection_polygon")
    rows=_pose_geometry_rows(ui.session)
    assert rows[0]["metric_status"]=="measured",rows
    assert rows[0]["nmce_pct"] > 0
    assert 0 < rows[0]["polygon_iou"] < 1
    first_nmce=rows[0]["nmce_pct"]
    ui.zoom_image(1.5)
    assert len(ui.canvas.find_withtag("pose_gt_corner"))==4
    px,py=ui._pose_mt_to_canvas(ui._pose_gt_points[0],"a1")
    ui._pan_begin(SimpleNamespace(x=px,y=py))
    newpoint=(ui._pose_gt_points[0][0]-5,ui._pose_gt_points[0][1]-5)
    newx,newy=ui._pose_mt_to_canvas(newpoint,"a1")
    ui._pan_move(SimpleNamespace(x=newx,y=newy))
    ui._pan_end(SimpleNamespace(x=newx,y=newy))
    assert len(ui.session.calls)==2
    assert _pose_geometry_rows(ui.session)[0]["metric_status"] == "measured"
    assert _pose_geometry_rows(ui.session)[0]["nmce_pct"] != first_nmce
    # Invalid GT must revert without changing persisted decision.
    before=ui.session.review.attempt_annotations["a1"]["pose_gt_quad_mt_input_px"]
    ui._pose_gt_before_drag=list(ui._pose_gt_points)
    ui._pose_gt_drag_index=0
    ui._pose_gt_points[0]=(500,500)  # crossed polygon
    ui._finish_pose_gt_drag()
    assert ui.session.review.attempt_annotations["a1"]["pose_gt_quad_mt_input_px"]==before
    ui._clear_pose_gt()
    assert ui.session.review.attempt_annotations["a1"]["pose_gt_quad_mt_input_px"] is None
    assert _pose_geometry_rows(ui.session)[0]["metric_status"]=="missing_manual_gt"
    root.destroy()
    print("GT_POSE_GUI_PASS: click 4, automatic save, drag, zoom, invalid revert, clear")


if __name__=='__main__':
    run()
