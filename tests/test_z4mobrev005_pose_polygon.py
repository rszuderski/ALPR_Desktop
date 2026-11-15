"""Z4MOBREV005: zachowanie oryginalnego Pose/quad i jawnego fallbacku BBOX."""
from types import SimpleNamespace
import pytest
from PIL import Image
from auto_annotation_tool.gui import z4_mobile_sample_review as module


ATTEMPT = dict(id="attempt-1", mt_status="VALID_QUAD", input_width="512", input_height="512",
    roi_left="0", roi_top="0", roi_right="480", roi_bottom="640",
    input_scale="0.8", input_pad_x="64", input_pad_y="0",
    plate_left="200.858032", plate_top="244.450058",
    plate_right="285.867920", plate_bottom="266.070740",
    mt_detection_index="0", mt_input_evidence_entry="samples/evidence/a.jpg")
SAMPLE = dict(attempt_id="attempt-1", plate_geometry=dict(available=True,
    source_width_px=480, source_height_px=640,
    plate_corners_norm=[(0.4188046455, 0.3822625279), (0.5985187292, 0.3919177651),
                        (0.5957919359, 0.4160884917), (0.4185177982, 0.4056845605)]))


def test_real_session_pose_coordinates_project_through_letterbox():
    quad = module._pose_quad_on_mt_input(ATTEMPT, SAMPLE)
    assert quad is not None and len(quad) == 4
    assert quad[0] == pytest.approx((224.820984, 195.718414), abs=.01)
    assert quad[2] == pytest.approx((292.7841, 213.0373), abs=.02)
    assert quad[0][1] < quad[1][1]  # na rzeczywistym zapisie krawędź jest ukośna


def test_csv_string_and_invalid_provenance_fall_back_safely():
    sample = dict(SAMPLE, plate_geometry=dict(SAMPLE['plate_geometry']))
    sample['plate_geometry']['plate_corners_norm'] = None
    sample['plate_corners_norm'] = '0.4188046455:0.3822625279;0.5985187292:0.3919177651;0.5957919359:0.4160884917;0.4185177982:0.4056845605'
    assert module._pose_quad_on_mt_input(ATTEMPT, sample) is not None
    assert module._pose_quad_on_mt_input(ATTEMPT, dict(sample, attempt_id='other')) is None
    assert module._pose_quad_on_mt_input(ATTEMPT, dict(SAMPLE, plate_geometry={'available':False})) is None
    degenerate = dict(SAMPLE, plate_geometry=dict(SAMPLE['plate_geometry']))
    degenerate['plate_geometry']['plate_corners_norm'] = [(0.5,0.5)] * 4
    assert module._pose_quad_on_mt_input(ATTEMPT, degenerate) is None
    wrong_size = dict(SAMPLE, plate_geometry=dict(SAMPLE['plate_geometry']))
    wrong_size['plate_geometry']['source_width_px'] = 1000
    assert module._pose_quad_on_mt_input(ATTEMPT, wrong_size) is None
    assert module._linked_pose_quad_on_mt_input(SimpleNamespace(samples={'c1':SAMPLE}, sample_ids_by_attempt={'attempt-1':['c1']}), ATTEMPT) is not None


class RecordingCanvas:
    def __init__(self):
        self.calls=[]
    def create_polygon(self, *coords, **kwargs):
        self.calls.append(('polygon',coords,kwargs)); return len(self.calls)
    def create_rectangle(self, *coords, **kwargs):
        self.calls.append(('rectangle',coords,kwargs)); return len(self.calls)
    def create_oval(self, *coords, **kwargs):
        self.calls.append(('oval',coords,kwargs)); return len(self.calls)
    def create_text(self, *coords, **kwargs):
        self.calls.append(('text',coords,kwargs)); return len(self.calls)
    def bbox(self, item):
        return (0,0,150,20)
    def move(self, *args): pass
    def tag_lower(self, *args): pass


def fake_ui(monkeypatch, sample=SAMPLE, zoom=1.0, center=(256.0,256.0)):
    monkeypatch.setattr(module, 'detection_box_on_mt_input', lambda row:(224.6864,195.5600,292.6943,212.8566))
    group = SimpleNamespace(records=[ATTEMPT], evidence_entries={'samples/evidence/a.jpg'}, detection_count=1)
    session = SimpleNamespace(attempts={'attempt-1': ATTEMPT}, samples={'c1':sample},
        sample_ids_by_attempt={'attempt-1':['c1']}, mt_invocations={'inv-1':group})
    view = module.MobileSampleReviewWindow.__new__(module.MobileSampleReviewWindow)
    view.current_record=('sample','c1')
    view.current_invocation='inv-1'
    view._image=Image.new('RGB',(512,512),'black')
    view._image_source_size=(512.0,512.0)
    view._image_entry='samples/evidence/a.jpg'
    view._view_pixel_scale=zoom
    view._view_center=center
    view.canvas=RecordingCanvas()
    view.session=session
    return view


def test_draws_actual_polygon_at_scale_and_pan(monkeypatch):
    view = fake_ui(monkeypatch)
    view._paint_detection_overlay(512,512,0,0)
    poly = [x for x in view.canvas.calls if x[0]=='polygon' and 'detection_polygon' in x[2].get('tags',())]
    assert len(poly)==1
    quad = module._pose_quad_on_mt_input(ATTEMPT,SAMPLE)
    assert poly[0][1]==pytest.approx(tuple(v for pt in quad for v in pt),abs=.001)
    assert len([x for x in view.canvas.calls if x[0]=='oval'])==4
    assert any('POSE 4 pkt' in x[2].get('text','') for x in view.canvas.calls if x[0]=='text')
    view._view_center=(240,260)
    view._view_pixel_scale=2.0
    view.canvas.calls.clear()
    view._paint_detection_overlay(512,512,0,0)
    moved = [x for x in view.canvas.calls if x[0]=='polygon' and 'detection_polygon' in x[2].get('tags',())]
    assert moved[0][1][:2] == pytest.approx(((quad[0][0]-240)*2+256, (quad[0][1]-260)*2+256),abs=.001)


def test_missing_pose_has_no_bbox_fallback(monkeypatch):
    sample=dict(SAMPLE,plate_geometry={'available':False})
    view=fake_ui(monkeypatch,sample)
    view._paint_detection_overlay(512,512,0,0)
    assert not any(c[0]=='polygon' for c in view.canvas.calls)
    assert not any(c[0]=='rectangle' and 'detection_box' in c[2].get('tags',()) for c in view.canvas.calls)
    assert any('narożników' in c[2].get('text','') for c in view.canvas.calls if c[0]=='text')
