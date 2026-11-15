"""Three host checks on the protected five-photo archive; no TV or inference."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from types import SimpleNamespace as NS
import hashlib,json,math,os,shutil,time,zipfile
import cv2
import numpy as np
from PIL import Image
import pytest
from auto_annotation_tool.ranking import final180_markers as mk,final180_reference as ref,final180_timeline as tl,photo_reference as photo
from auto_annotation_tool.gui import final180_tv as tv


@pytest.fixture(scope='module')
def source():
    location=os.environ.get('F6_ROI_EVIDENCE')
    if not location:pytest.skip('Private real F2 evidence fixture not supplied')
    out=Path(location);paths=photo.read_json(out/'paths.json');report=photo.read_json(out/'private/evaluation_original.json')
    old=photo.read_json(Path(paths['technical'])/'manifest.json');delivery=photo.read_json(out/'delivery.json')
    return NS(out=out,paths=paths,report=report,old=old,delivery=delivery)


def box(points):
    q=np.asarray(points);return [float(q[:,0].min()),float(q[:,1].min()),float(q[:,0].max()),float(q[:,1].max())]


def quantized(points):
    b=box(points);return [math.floor(b[0]),math.floor(b[1]),math.ceil(b[2]),math.ceil(b[3])]


def expand(b,n):return [b[0]-n,b[1]-n,b[2]+n,b[3]+n]


def geometries(source):return [f for f in source.report['frames'] if 'matrix_tv_to_camera' in f['marker_decoding']]


def projected_roi(board,frames):
    rois=[quantized(mk.project(mk.rect_points(board['photo_rect']),f['marker_decoding']['matrix_tv_to_camera'])) for f in frames]
    envelope=[min(r[0] for r in rois),min(r[1] for r in rois),max(r[2] for r in rois),max(r[3] for r in rois)]
    return rois,envelope,expand(envelope,6)  # 4 px geometric bound + 2 px operator error; validator adds its unchanged 2 px.


def test_1_real_overlap_and_coordinate_chain(source):
    archive=Path(source.paths['private_archive']);assert photo.digest(archive)==source.report['source_archive_sha256']
    cv2.setRNGSeed(0);reproduced=[];missing=[];input_count=0;projection_delta=[]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for f in source.report['frames']:
            seq=f['source_stamp']['source_sequence'];record=f['source_image'];raw=z.read(record['entry'])
            assert hashlib.sha256(raw).hexdigest()==record['sha256']
            image=Image.open(source.out/'source_frames'/f'seq_{seq}.jpg').convert('RGB');g=mk.geometry(image,source.old['layout'])
            assert image.size==(960,1280)
            if g['status']=='PASS':
                prior=f['marker_decoding'];q=mk.rect_points(source.old['layout']['photo_rect'])
                delta=float(np.linalg.norm(mk.project(q,g['matrix_tv_to_camera'])-mk.project(q,prior['matrix_tv_to_camera']),axis=1).max())
                projection_delta.append(delta);assert delta<.01
                if f['geometry_reason']=='MARKER_INSIDE_MODEL_INPUT':
                    result=mk.input_exclusion(g,source.old['layout'],f['input_region'])
                    expected={tuple(x['marker_rect_tv']):x['intersection_area_px'] for x in f['input_exclusion']['intersections']}
                    actual={tuple(x['marker_rect_tv']):x['intersection_area_px'] for x in result['intersections']}
                    assert result['status']=='N/A' and actual.keys()==expected.keys()
                    assert all(abs(actual[k]-expected[k])<.01 for k in actual)
                    no_clearance=mk.input_exclusion(g,source.old['layout'],f['input_region'],clearance=0)
                    reproduced.append(dict(sequence=seq,frame_key=f['frame_key'],policy_intersections=result['intersections'],unexpanded_roi_intersections=no_clearance['intersections']))
            else:
                assert seq in (59303,59614,60394) and g['reason']=='ARUCO_MISSING_OR_DUPLICATED';missing.append(seq)
            for item in f['actual_mt_inputs']:
                assert item['roi_camera_xy']==f['input_region']==[135,462,790,845]
                assert hashlib.sha256(z.read(item['entry'])).hexdigest()==item['sha256']
                assert Image.open(source.out/'source_frames'/f'seq_{seq}_MT.jpg').size==(512,512)
                assert abs(item['input_scale']-512/655)<1e-6 and abs(item['input_pad_y']-(512-383*(512/655))/2)<1e-4 and item['input_pad_x']==0
                assert f['processing_transform']=='CROP_ONLY_TRANSLATION' and f['H_TV']=='NOT_APPLIED'
                assert f['crop_translation']=={'x':39,'y':417};input_count+=1
    assert len(reproduced)==6 and len(projection_delta)==14 and input_count==13
    (source.out/'check1.json').write_text(json.dumps(dict(status='PASS_REPRODUCED_EXISTING_FAILURE',frames=17,valid_homographies=14,missing_aruco=sorted(missing),mt_inputs=13,
        overlap_frames=reproduced,max_homography_delta_px=max(projection_delta),offset_or_double_scale=False,tensor_bit_identity='NOT_AVAILABLE_JPEG94_ONLY'),indent=2),'utf-8')


def test_2_versioned_gap_on_all_homographies_and_full_material(source):
    old=source.old['layout'];frames=geometries(source)
    technical=Path(source.delivery['technical']);final=Path(source.delivery['final'])
    tm,_,tgt=ref.load(technical);fm,material,gt=ref.load(final);board=fm['layout']
    assert board['layout_version']==mk.F2_SAFE_LAYOUT and board['photo_rect']==[250,100,1670,980]
    assert board['qr_rects']==old['qr_rects'] and board['aruco']==old['aruco'] and board['qr']==old['qr']
    assert tgt is None and tm['reference_mode']=='none' and not(technical/'reference.json').exists()
    assert ref.reference_preflight(final)['counts']['GT_COMPLETE']==176
    for name in ('reference.json','source_annotations.xml','material.json'):assert photo.digest(final/name)==photo.digest(Path(source.paths['final'])/name)
    assert not(final/'runs').exists() and not(technical/'runs').exists()
    roi_candidates,envelope,stress=projected_roi(board,frames)
    for f in frames:
        g=dict(f['marker_decoding'],status='PASS')
        assert mk.input_exclusion(g,board,stress)['status']=='PASS'
    candidate=deepcopy(old);candidate['photo_rect']=[240,100,1680,980];_,_,stress240=projected_roi(candidate,frames)
    areas240=[x['intersection_area_px'] for f in frames for x in mk.input_exclusion(dict(f['marker_decoding'],status='PASS'),candidate,stress240)['intersections']]
    assert areas240
    unchanged=changed=0;slack=float('inf');rows=[]
    previous=photo.read_json(Path(source.paths['final'])/'manifest.json')
    for e,old_e in zip(fm['exposures'],previous['exposures']):
        assert (e['image_id'],e['photo_sha256'])==(old_e['image_id'],old_e['photo_sha256'])
        assert photo.digest(material.verify(e['image_id']))==e['photo_sha256']
        a,b=e['actual_image_rect'],old_e['actual_image_rect'];size=[a[2]-a[0],a[3]-a[1]];old_size=[b[2]-b[0],b[3]-b[1]]
        if size==old_size:unchanged+=1
        else:changed+=1
        rows.append(dict(exposure_id=e['exposure_id'],source_size=e['photo_placement']['source_size'],old_size=old_size,new_size=size))
        for f in frames:
            q=box(mk.project(mk.rect_points(a),f['marker_decoding']['matrix_tv_to_camera']))
            for r in roi_candidates:slack=min(slack,q[0]-r[0]-2,q[1]-r[1]-2,r[2]-q[2]-2,r[3]-q[3]-2)
    assert slack>0 and unchanged+changed==176
    canonical=[]
    for size in [(1500,1000),(1200,900)]:
        a=mk.photo_placement(size,old)['rect'];b=mk.photo_placement(size,board)['rect']
        assert [a[2]-a[0],a[3]-a[1]]==[b[2]-b[0],b[3]-b[1]]
        canonical.append(dict(source=size,rendered=[b[2]-b[0],b[3]-b[1]]))
    retained=mk.input_exclusion(geometries(source)[0]['marker_decoding'],board,[135,462,790,845])
    assert retained['status']=='N/A'  # Merely installing code cannot fix the old calibration.
    (source.out/'check2.json').write_text(json.dumps(dict(status='PASS_CONDITIONAL_GEOMETRIC_DESIGN',photo_rect=board['photo_rect'],gap_tv_px=46,
        observed_homographies=14,predicted_envelope_roi=envelope,stress_roi=stress,operator_error_px=2,geometry_buffer_px=4,unchanged_validator_clearance_px=2,
        quantization='OUTWARD_FLOOR_CEIL',new_intersections=0,rejected_240_max_intersection=max(areas240),actual_roi_must_be_recalibrated=True,
        jpeg_containment_min_slack_px=slack,jpeg_containment_scope='ALL_176_PHOTOS_ALL_14_OBSERVED_POSES_AND_CALIBRATION_POSITIONS_PLUS_2PX_INWARD_EDGE_ERROR',
        unchanged_image_sizes=unchanged,reduced_image_sizes=changed,canonical=canonical,placements=rows,A12_TV='NOT_VERIFIED'),indent=2),'utf-8')


def test_3_native_host_preroll_handoff_postroll_and_independent_qr(source,tmp_path,monkeypatch):
    import tkinter as tk
    from auto_annotation_tool import config
    from auto_annotation_tool.session import SessionManager
    old_sha=source.report['manifest_sha256'];old_run,old_events,_=tl.read_run(source.paths['run'],source.old,old_sha)
    assert not any(e['identity']=='POSTROLL' for e in old_run['qr_binding']['entries'])
    old_context=tl.marker_context(old_run,old_events,source.old,old_sha,prepared=source.paths['technical'])
    cv2.setRNGSeed(0)
    rerun=mk.decode(Image.open(source.out/'source_frames/seq_60262.jpg'),source.old,old_sha,old_run['run_id'],run_context=old_context)
    if not all(rerun.get('decoded_tokens',[])):assert rerun['status']=='N/A'
    prepared=tmp_path/'host_only'/'prepared_copy';shutil.copytree(source.delivery['technical'],prepared)
    m,_,_=ref.load(prepared);monkeypatch.setattr(config,'SESSION',SessionManager(tmp_path/'preferences.json'))
    monkeypatch.setattr(tv.monitor_dpi,'final_monitors',lambda:[dict(name='HOST_ONLY',x=0,y=0,w=1920,h=1080,verified=True,geometry_source='ISOLATED_TEST')])
    clock=NS(value=1_000_000_000);now=lambda:clock.value
    monkeypatch.setattr(tv,'time',NS(perf_counter_ns=now,monotonic=time.monotonic))
    controller=tv.PresentationRun;monkeypatch.setattr(tv,'PresentationRun',lambda *a,**kw:controller(*a,clock=now,**kw))
    root=tk.Tk();root.withdraw();errors=[];root.report_callback_exception=lambda *a:errors.append(str(a))
    app=tv.make_app(tv.legacy_module(tv.LEGACY_DEFAULT),NS(xml=None,positives=None,negatives=None,output=tmp_path,dwell=5,negatives_declared=False,unreadable_ids=[]),parent=root);app.window.withdraw()
    made=[];draws=[]
    def hidden(screen):
        made.append(screen);app.viewer=tk.Toplevel(root);app.viewer.withdraw();app.canvas=tk.Canvas(app.viewer)
    monkeypatch.setattr(app,'_make_viewer',hidden);draw=app.draw
    def capture(image):draws.append(image.copy());draw(image)
    monkeypatch.setattr(app,'draw',capture)
    try:
        app.loaded(prepared);app.show_calibration();assert app.mode=='CALIBRATION',app.status.get()
        viewer=app.viewer;app.begin_countdown();assert app.mode=='COUNTDOWN',app.status.get()
        assert app.viewer is viewer and len(made)==1
        clock.value+=int(m['countdown_seconds']*1e9)
        for index in range(5):
            deadline=time.monotonic()+8
            while app.mode!='RUNNING' and time.monotonic()<deadline:app._tick();root.update();time.sleep(.01)
            assert app.mode=='RUNNING' and app.run.current==index,app.status.get()
            clock.value+=int(m['dwell_seconds']*1e9);app._tick()
            if index<4:clock.value+=int(m['transition_seconds']*1e9)
        assert app.mode=='COMPLETED' and app._post_roll_hold and app.viewer is viewer and len(made)==1
        assert str(app.stop_btn.cget('state'))=='normal' and str(app.start_btn.cget('state'))=='disabled'
        state,events,_=tl.read_run(app.run.directory/'run.json',m,app.run.manifest_sha)
        context=tl.marker_context(state,events,m,app.run.manifest_sha,prepared=prepared)
        decoded=[mk.decode(im,m,app.run.manifest_sha,state['run_id'],run_context=context) for im in draws]
        assert all(len(v.get('decoded_tokens',[]))==2 and all(v['decoded_tokens']) and len(set(v['decoded_tokens']))==1 for v in decoded)
        assert [v['exposure_id'] for v in decoded if v['status']=='PASS']==[f'e{i:04d}' for i in range(1,6)]
        assert decoded[0]['reason']==decoded[-1]['reason']=='TECHNICAL_TRANSITION_FRAME'
        post=next(v for v in state['qr_binding']['entries'] if v['identity']=='POSTROLL')['token']
        assert decoded[-1]['decoded_tokens']==[post,post]
        assert any(e.get('display_token')==post and e['event']=='REPLACEMENT_VIEW_UPDATED' for e in events)
        assert sum(e['event']=='PHOTO_VIEW_UPDATED' for e in events)==5 and state['status']=='COMPLETED'
        draws[0].save(source.out/'HOST_new_calibration_T.png');draws[-1].save(source.out/'HOST_new_postroll_T.png')
        draws[next(i for i,v in enumerate(decoded) if v['status']=='PASS')].save(source.out/'HOST_new_technical_P.png')
        journal=app.run.directory/'run.json';before=photo.digest(journal);app.stop_btn.invoke()
        assert app.viewer is None and not app._post_roll_hold and photo.digest(journal)==before and not errors
        (source.out/'check3.json').write_text(json.dumps(dict(status='PASS_HOST_NATIVE_TK_AND_PIXELS',viewer_creations=1,black_viewer_recreation_between_preroll_and_start=False,
            synthetic_photographs=5,synthetic_frames_decoded=len(decoded),two_independent_qr=True,first_and_last_phase='T',postroll_token=post,
            old_60262_rerun=dict(status=rerun['status'],reason=rerun.get('reason'),tokens=rerun.get('decoded_tokens')),
            no_tv=True,A12_TV='NOT_VERIFIED',historical_unlocated_mt_not_reclassified=True),indent=2),'utf-8')
    finally:
        for token in root.tk.call('after','info'):
            if root.tk.call('after','info',token)[0] in (app.window._tclCommands or []):app.window.after_cancel(token)
        app._close();root.destroy()
