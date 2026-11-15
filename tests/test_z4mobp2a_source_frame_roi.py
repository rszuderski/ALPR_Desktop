"""P2A: read-only Android source Bitmap+ROI inspection; no GT editing.

The module is deliberately testable without Tk or actual ALPR models.
"""
from __future__ import annotations

import ast
import importlib.util
import io
import os
from pathlib import Path
import sys
import types
import zipfile

import pytest
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
SRC = HERE / 'auto_annotation_tool/ranking/mobile_source_frame_roi.py'
GUI = HERE / 'auto_annotation_tool/gui/z4_mobile_sample_review.py'

spec = importlib.util.spec_from_file_location('auto_annotation_tool.ranking.mobile_source_frame_roi', SRC)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def row(index=0, *, entry='samples/frames/sample.jpg'):
    return dict(mt_invocation_id='invoke-7', id=f'attempt-{index}',
        source_frame_expected='true', source_frame_entry=entry,
        source_frame_status='WRITTEN',
        source_frame_coordinate_space='mt_source_bitmap_xy',
        source_frame_origin='pipeline_bitmap_before_roi',
        source_frame_width_px='480', source_frame_height_px='640',
        roi_left='20', roi_top='40', roi_right='440', roi_bottom='600',
        source_sequence='991', source_timestamp_nanos='123456789',
        mt_status='VALID_QUAD', mt_detection_index=str(index), mt_detection_count='2',
        mt_pose_status='VALID_QUAD',
        mt_pose_coordinate_space='source_frame_normalized_v1',
        mt_pose_corner_order='TL_TR_BR_BL',
        mt_pose_source_width_px='480', mt_pose_source_height_px='640',
        mt_pose_corners_norm='0.2:0.2;0.4:0.2;0.4:0.3;0.2:0.3')


def context(records=None):
    return core.source_invocation_context(records or [row()], 'invoke-7',
       {'samples/frames/sample.jpg'})


def test_verified_invocation_two_detections_share_one_source_frame_and_roi():
    c = context([row(0), row(1)])
    assert c['entry'] == 'samples/frames/sample.jpg'
    assert c['size'] == (480.0, 640.0)
    assert c['roi'] == (20, 40, 440, 600)
    result = core.verified_invocation_detections(c)
    assert [r['detection_index'] for r in result] == [0, 1]
    assert {r['attempt_id'] for r in result} == {'attempt-0', 'attempt-1'}


@pytest.mark.parametrize('change', [
    {'source_frame_entry':'samples/frames/other.jpg'},
    {'roi_right':'460'},
    {'source_frame_width_px':'479'},
    {'source_sequence':'992'},
    {'source_timestamp_nanos':'98765'},
    {'mt_invocation_id':'another-invoke'},
    {'source_frame_status':'MISSING'},
    {'source_frame_coordinate_space':'camera_raw_sensor'},
    {'source_frame_origin':'camera_sensor'},
])
def test_invocation_conflicting_multidetection_metadata_fails_closed(change):
    secondary = row(1)
    secondary.update(change)
    assert core.source_invocation_context([row(0), secondary], 'invoke-7',
        {'samples/frames/sample.jpg', 'samples/frames/other.jpg'}) is None


@pytest.mark.parametrize('invalid', [
    '../samples/frames/sample.jpg','samples/evidence/sample.jpg',
    'samples/frames/../sample.jpg', 'samples/frames/subfolder/sample.jpg',
    'samples/frames/evil\\x.jpg','C:/samples/frames/sample.jpg',
    'samples/frames/sample.jpg/../../x.jpg',
])
def test_source_entry_unsafe_path_is_rejected(invalid):
    assert core.source_frame_entry(row(entry=invalid), {invalid}) is None


def test_source_missing_or_wrong_size_rejected():
    assert context() is not None
    assert core.source_invocation_context([row()], 'invoke-7', set()) is None
    assert core.source_invocation_context([row()], 'invoke-7', {'samples/frames/sample.jpg'}, (512,512)) is None
    assert core.source_invocation_context([row()], 'invoke-7', {'samples/frames/sample.jpg'}, (480,640)) is not None
    assert core.frame_roi(dict(row(),roi_right='481'), (480,640)) is None
    assert core.frame_roi(dict(row(),roi_right='not-a-number'), (480,640)) is None
    assert core.frame_roi(row(),(512,512)) is None


@pytest.mark.parametrize('tamper',[
    {'mt_pose_status':'NO_DETECTION'},
    {'mt_pose_coordinate_space':'roi_normalized'},
    {'mt_pose_corner_order':'UNVERIFIED'},
    {'mt_pose_source_width_px':'512'},
    {'mt_pose_corners_norm':'0.2:0.2;0.4:0.4;0.2:0.4;0.4:0.2'},
    {'mt_pose_corners_norm':'0.2:0.2;0.2:0.2;0.2:0.2;0.2:0.2'},
    {'mt_pose_corners_norm':'0.2:0.2;0.4:0.2;0.4:0.3'},
])
def test_bad_model_pose_never_turns_into_synthetic_quad(tamper):
    r = row()
    r.update(tamper)
    assert core.source_pose_quad(r, (480,640)) is None


def test_valid_original_pose_and_hit_test():
    quad = core.source_pose_quad(row(), (480,640))
    assert quad is not None
    assert quad[0] == (96, 128)
    assert core.quad_contains_point(quad, 130,150)
    assert not core.quad_contains_point(quad, 300,150)


def test_no_detection_record_still_keeps_source_frame():
    record = row()
    record.update(mt_status='NO_DETECTION', mt_pose_status='NO_DETECTION', mt_detection_index='')
    c=context([record])
    assert c and core.verified_invocation_detections(c) == []


def test_missing_and_duplicate_detection_indices_never_invent_new_detections():
    bad = row(1)
    bad['mt_detection_index']='0'
    c = context([row(0),bad])
    assert len(core.verified_invocation_detections(c)) == 1


def test_archive_reader_uses_shared_zip_validator_and_limited_frame_path(tmp_path, monkeypatch):
    path = tmp_path/'sample.alprsession'
    img = Image.new('RGB',(480,640),color=(25,75,100))
    buf = io.BytesIO()
    img.save(buf,format='JPEG')
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('samples/frames/sample.jpg',buf.getvalue())
        z.writestr('samples/evidence/mz.jpg',buf.getvalue())
    assert not path.stat().st_size == 0
    # Production function imports these three guarded routines from existing
    # mobile_package_experiments; mock only the shared validator surface.
    packages = ['auto_annotation_tool','auto_annotation_tool.ranking']
    for name in packages:
        module = types.ModuleType(name)
        module.__path__ = [str(HERE / name.replace('.', '/'))]
        monkeypatch.setitem(sys.modules, name, module)
    m = types.ModuleType('auto_annotation_tool.ranking.mobile_package_experiments')
    m._archive_name_safe = lambda s: (not ('..' in s or '\\' in s), s)
    m._full_report_source_path = lambda bundle: bundle
    def validated(z):
        return {info.filename: info for info in z.infolist()}
    m._normalized_zip_entries_for_full_read = validated
    monkeypatch.setitem(sys.modules, m.__name__, m)
    returned = core.read_mobile_source_frame(path,'samples/frames/sample.jpg')
    assert returned.size == (480,640)
    with pytest.raises(ValueError):
        core.read_mobile_source_frame(path,'samples/evidence/mz.jpg')
    with pytest.raises(ValueError):
        core.read_mobile_source_frame(path,'samples/frames/../../bad.jpg')
    with pytest.raises(FileNotFoundError):
        core.read_mobile_source_frame(path,'samples/frames/missing.jpg')
    with pytest.raises(ValueError):
        core.read_mobile_source_frame(path,'samples/frames/sample.jpg',max_entry_bytes=150)


def test_gui_reviewer_adds_frame_without_changing_default_mt_view():
    tree=ast.parse(GUI.read_text(encoding='utf-8'))
    desired=['_review_image_sources','_primary_review_image_index']
    defs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in desired]
    assert len(defs)==2
    scope=dict(source_invocation_context=core.source_invocation_context)
    exec(compile(ast.Module(body=defs,type_ignores=[]),str(GUI),'exec'),scope)
    class Group:
        evidence_entries={'samples/evidence/mt1.jpg'}
        records=[row()]
    current=dict(row(),mt_input_evidence_entry='samples/evidence/mt1.jpg',evidence_entry='')
    names={'samples/frames/sample.jpg','samples/evidence/mt1.jpg'}
    result=scope['_review_image_sources']('attempt',current,current,Group(),[],names)
    assert result[0][0].startswith('Pełna klatka')
    assert result[0][1]['source_frame_view'] is True
    selected=scope['_primary_review_image_index'](result,current,Group())
    assert result[selected][1]['image_entry']=='samples/evidence/mt1.jpg'
    legacy=dict(current,source_frame_status='MISSING')
    class LegacyGroup:
        evidence_entries={'samples/evidence/mt1.jpg'}
        records=[legacy]
    assert all(not x[1].get('source_frame_view') for x in
      scope['_review_image_sources']('attempt',legacy,legacy,LegacyGroup(),[],names))


def test_source_view_keeps_manual_actions_off_full_frame():
    tree=ast.parse(GUI.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MobileSampleReviewWindow')
    def f(name):return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    names={n.name for n in cls.body if isinstance(n,ast.FunctionDef)}
    for method in ('_source_frame_context','_paint_source_frame_overlays','_toggle_source_frame_view',
                   '_select_source_frame_detection','_start_mt_redraw','_sync_source_frame_button'):
        assert method in names
    code=ast.get_source_segment(GUI.read_text(encoding='utf-8'), f('_start_mt_redraw'))
    assert code.index('startswith("samples/frames/")') < code.index('self._visual_target()')
    paint=ast.get_source_segment(GUI.read_text(encoding='utf-8'), f('_paint_source_frame_overlays'))
    assert 'session.annotate_' not in paint and 'save_' not in paint


def _gui_method(name, **extra):
    tree=ast.parse(GUI.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MobileSampleReviewWindow')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    env=dict(verified_invocation_detections=core.verified_invocation_detections,
             quad_contains_point=core.quad_contains_point, **extra)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(GUI),'exec'),env)
    return env[name]


def _mock_frame_window(source, selected='attempt-0'):
    from types import SimpleNamespace
    class Canvas:
        def __init__(self):self.draws=[]
        def record(self,kind,*a,**kw):
            self.draws.append((kind,a,kw));return len(self.draws)
        def create_rectangle(self,*a,**kw):return self.record('rectangle',*a,**kw)
        def create_polygon(self,*a,**kw):return self.record('polygon',*a,**kw)
        def create_text(self,*a,**kw):return self.record('text',*a,**kw)
        def winfo_width(self):return 480
        def winfo_height(self):return 640
    canvas=Canvas()
    label=SimpleNamespace(configure=lambda **kw:None)
    return SimpleNamespace(canvas=canvas, image_help=label, _image=SimpleNamespace(width=480,height=640),
        _view_center=(240,320), _view_pixel_scale=1.0,
        _is_source_frame_view=lambda:True,
        _source_frame_context=lambda:source,
        _image_entry=source['entry'],
        _current_mt_attempt=lambda:{'id':selected},
        _canvas_text_background=lambda *a,**kw:None,
        _multi_focus_attempt=lambda aid:None)


def test_source_overlays_draw_one_roi_and_all_real_detections_not_gt():
    r2=row(1)
    r2['mt_pose_corners_norm']='0.55:0.55;0.75:0.55;0.75:0.7;0.55:0.7'
    c=context([row(0),r2])
    win=_mock_frame_window(c)
    draw=_gui_method('_paint_source_frame_overlays')
    draw(win,480,640)
    tags=[x[2].get('tags',()) for x in win.canvas.draws]
    assert sum('source_mt_all_detections' in t and x[0]=='polygon'
               for x,t in zip(win.canvas.draws,tags))==2
    assert sum('mt_source_roi' in t and x[0]=='rectangle'
               for x,t in zip(win.canvas.draws,tags))==1
    assert sum('source_outside_roi' in t and x[0]=='rectangle'
               for x,t in zip(win.canvas.draws,tags))==4
    assert not any('operator_quad' in t or 'manual_gt' in t for t in tags)


def test_frame_click_selects_correct_detection_and_preserves_source_request():
    from types import SimpleNamespace
    r2=row(1)
    r2['mt_pose_corners_norm']='0.55:0.55;0.75:0.55;0.75:0.7;0.55:0.7'
    win=_mock_frame_window(context([row(0),r2]))
    selected=[]
    win._multi_focus_attempt=lambda aid:selected.append(aid)
    click=_gui_method('_select_source_frame_detection')
    assert click(win,SimpleNamespace(x=320,y=400)) is True
    assert selected==['attempt-1']
    assert win._source_frame_keep_on_selection=='samples/frames/sample.jpg'
    assert click(win,SimpleNamespace(x=10,y=20)) is False
    assert selected==['attempt-1']


def test_source_pose_on_real_android_p1_archive_if_provided():
    """Optional local inspection; automated suite remains independent of user data."""
    import csv
    example=os.environ.get('ALPR_P2A_EXAMPLE_ARCHIVE')
    if not example or not Path(example).exists():
        pytest.skip('Set ALPR_P2A_EXAMPLE_ARCHIVE to inspect source session')
    with zipfile.ZipFile(example) as z:
        attempts=list(csv.DictReader(io.StringIO(z.read('samples/attempts.csv').decode('utf-8'))))
        names=set(z.namelist())
        by_invocation={}
        for raw in attempts:
            raw['id']=raw['attempt_id']
            by_invocation.setdefault(raw['mt_invocation_id'],[]).append(raw)
        seen=0
        for inv,records in by_invocation.items():
            context=core.source_invocation_context(records,inv,names)
            assert context
            image=Image.open(io.BytesIO(z.read(context['entry'])))
            assert core.source_invocation_context(records,inv,names,image.size)
            seen+=len(core.verified_invocation_detections(context))
        assert len(by_invocation)==8
        assert seen==9
