"""Three focused v3 layout checks; no TV, phone, inference or production run."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace as NS
import ast,json,math,os,shutil,time
import numpy as np
import pytest
from auto_annotation_tool.ranking import final180_markers as mk,final180_reference as ref,final180_timeline as tl,photo_reference as photo
from auto_annotation_tool.gui import final180_tv as tv
from test_final180_final_blockers import app_fixture,close_app


@pytest.fixture(scope='module')
def evidence():
    folder=os.environ.get('F6_CLEARANCE_EVIDENCE')
    if not folder:pytest.skip('Private evidence paths not provided')
    out=Path(folder);return NS(out=out,paths=photo.read_json(out/'paths.json'),delivery=photo.read_json(out/'delivery.json'))


def save(out,name,data):(out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),'utf-8')
def bounds(q):return [float(q[:,0].min()),float(q[:,1].min()),float(q[:,0].max()),float(q[:,1].max())]
def quantized(q):
    b=bounds(q);return [math.floor(b[0]),math.floor(b[1]),math.ceil(b[2]),math.ceil(b[3])]
def expand(b,n):return [b[0]-n,b[1]-n,b[2]+n,b[3]+n]
def size(b):return [b[2]-b[0],b[3]-b[1]]


def test_T1_raster_clearance_shared_calibration_and_legacy(evidence):
    prepared=Path(evidence.delivery['technical']);m,material,_=ref.load(prepared);b=m['layout'];sha=photo.digest(prepared/'manifest.json')
    old=photo.read_json(Path(evidence.paths['technical'])/'manifest.json')
    assert b['photo_rect']==[300,100,1620,980] and size(b['photo_rect'])==[1320,880]
    assert b['qr_rects']==[[45,441,243,639],[1677,441,1875,639]]
    assert b['aruco']==old['layout']['aruco'] and b['qr']==old['layout']['qr']
    assert b['image_inset_px']==16 and b['qr']['module_px']==6
    for r in b['qr_rects']:assert size(r)==[198,198] and r[0]>=45 and r[2]<=1875
    assert b['photo_rect'][0]-b['qr_rects'][0][2]==b['qr_rects'][1][0]-b['photo_rect'][2]==57
    symbols=[[a['rect'][0]-a['quiet'],a['rect'][1]-a['quiet'],a['rect'][2]+a['quiet'],a['rect'][3]+a['quiet']] for a in b['aruco']]+b['qr_rects']
    assert min(min(r[0],r[1],1920-r[2],1080-r[3]) for r in symbols)==29
    for r in symbols:
        assert 0<r[0]<r[2]<1920 and 0<r[1]<r[3]<1080
        p=b['photo_rect'];assert min(r[2],p[2])<=max(r[0],p[0]) or min(r[3],p[3])<=max(r[1],p[1])
    calibration,identity=tv.calibration_frame(m,sha,technical_id='c'*32)
    same,_=mk.render(material.image(material.order[0]),b,identity['token'])
    for r in symbols:assert np.array_equal(np.asarray(calibration.crop(r)),np.asarray(same.crop(r)))
    assert identity['photo_rect']==b['photo_rect']
    for image in (calibration,same):
        geometry=mk.geometry(image,b);assert geometry['status']=='PASS' and sorted(geometry['detected_ids'])==[0,1,2,3]
        codes=mk.decode_codes(image,b,geometry);assert codes['decoded_tokens']==[identity['token']]*2
    calibration.save(evidence.out/'HOST_v3_calibration_T.png');same.save(evidence.out/'HOST_v3_photo_placement.png')
    old_image,_=tv.calibration_frame(old,photo.digest(Path(evidence.paths['technical'])/'manifest.json'),technical_id='c'*32)
    old_image.save(evidence.out/'HOST_v2_calibration_T.png')
    assert mk.layout_f2(1920,1080,version=mk.F2_SAFE_LAYOUT)==old['layout']
    legacy=photo.read_json(Path(evidence.paths['old_evidence'])/'private/evaluation_original.json')
    prior_manifest=photo.read_json(Path(legacy['prepared_path'])/'manifest.json')
    assert mk.layout_f2(1920,1080,version=mk.F2_LAYOUT)==prior_manifest['layout']
    before=ast.parse((evidence.out/'before/final180_markers.py').read_text('utf-8'));after=ast.parse(Path(mk.__file__).read_text('utf-8'))
    functions=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)}
    a,c=functions(before),functions(after)
    for name in ('geometry','decode_codes','decode','qualify_f2','qr_f2','input_exclusion','render','photo_placement'):assert a[name]==c[name]
    assert m['geometry_policy']==old['geometry_policy']==mk.GEOMETRY_POLICY
    save(evidence.out,'T1.json',dict(status='PASS',photo_rect=b['photo_rect'],qr_rects=b['qr_rects'],qr_size=198,qr_gap=57,
        minimum_symbol_edge_margin_px=29,qr_edge_margin_px=45,shared_raster_regions=True,legacy_v1_v2_unchanged=True,decoder_and_thresholds_unchanged=True))


def test_T2_fourteen_camera_homographies_and_all_176_photographs(evidence):
    previous=photo.read_json(Path(evidence.paths['old_evidence'])/'private/evaluation_original.json')
    frames=[f for f in previous['frames'] if 'matrix_tv_to_camera' in f.get('marker_decoding',{})];assert len(frames)==14
    final=Path(evidence.delivery['final']);m,material,gt=ref.load(final);board=m['layout']
    old=photo.read_json(Path(evidence.paths['final'])/'manifest.json');new5=Path(evidence.delivery['technical']);m5,_,gt5=ref.load(new5)
    assert len(m['exposures'])==176 and len(m5['exposures'])==5 and m['dwell_seconds']==m5['dwell_seconds']==5
    assert gt5 is None and not(new5/'reference.json').exists() and m5['layout']==board
    assert not(final/'runs').exists() and not(new5/'runs').exists()
    for name in ('reference.json','material.json','source_annotations.xml'):assert photo.digest(final/name)==photo.digest(Path(evidence.paths['final'])/name)
    preflight=ref.reference_preflight(final);assert preflight['counts']['GT_COMPLETE']==176 and preflight['complete_gt_plates']==209
    rois=[quantized(mk.project(mk.rect_points(board['photo_rect']),f['marker_decoding']['matrix_tv_to_camera'])) for f in frames]
    envelope=[min(r[0] for r in rois),min(r[1] for r in rois),max(r[2] for r in rois),max(r[3] for r in rois)]
    assert envelope==[172,461,752,847]
    per_click_slack=envelope_slack=float('inf');retained_roi_intersections=0
    for f in frames:
        d=dict(f['marker_decoding'],status='PASS')
        # Quantization outward; 4 px geometry bound + 2 px point error.
        # input_exclusion adds its UNCHANGED extra 2 px clearance itself.
        assert mk.input_exclusion(d,board,expand(envelope,6))['status']=='PASS'
        for r in rois:assert mk.input_exclusion(d,board,expand(r,6))['status']=='PASS'
        retained_roi_intersections+=len(mk.input_exclusion(d,board,[151,461,773,845])['intersections'])
        for e in m['exposures']:
            b=bounds(mk.project(mk.rect_points(e['actual_image_rect']),d['matrix_tv_to_camera']))
            for r in rois:per_click_slack=min(per_click_slack,b[0]-r[0]-2,b[1]-r[1]-2,r[2]-b[2]-2,r[3]-b[3]-2)
            r=envelope;envelope_slack=min(envelope_slack,b[0]-r[0]-6,b[1]-r[1]-6,r[2]-b[2]-6,r[3]-b[3]-6)
    assert per_click_slack>0 and envelope_slack>0
    unchanged=reduced=0;placements=[]
    for e,prior in zip(m['exposures'],old['exposures']):
        assert (e['image_id'],e['photo_sha256'])==(prior['image_id'],prior['photo_sha256'])
        assert photo.digest(material.verify(e['image_id']))==e['photo_sha256']
        assert e['actual_image_rect']==mk.photo_placement(e['photo_placement']['source_size'],board)['rect']
        if size(e['actual_image_rect'])==size(prior['actual_image_rect']):unchanged+=1
        else:reduced+=1
        placements.append(dict(exposure_id=e['exposure_id'],source_size=e['photo_placement']['source_size'],before=size(prior['actual_image_rect']),after=size(e['actual_image_rect'])))
    assert (unchanged,reduced)==(142,34)
    for s in ((1500,1000),(1200,900)):
        assert size(mk.photo_placement(s,board)['rect'])==size(mk.photo_placement(s,old['layout'])['rect'])
    save(evidence.out,'T2.json',dict(status='PASS_HOST_REPROJECTION',homographies=14,photographs=176,photo_rect=board['photo_rect'],
        projected_rois=rois,common_envelope=envelope,stress_roi=expand(envelope,6),marker_intersections=0,geometry_error_px=4,operator_error_px=2,validator_clearance_px=2,
        min_per_calibration_photo_slack_2px_inward=per_click_slack,min_envelope_photo_slack_6px_inward=envelope_slack,
        old_roi_intersections=retained_roi_intersections,old_roi_reuse='NOT_ACCEPTED_RECALIBRATE',sizes_unchanged=unchanged,sizes_reduced=reduced,placements=placements,
        identical_GT_sha256=photo.digest(final/'reference.json'),A12_TV='NOT_VERIFIED'))


def test_T3_native_tk_selection_p_t_p_postroll(evidence,tmp_path,monkeypatch):
    prepared=tmp_path/'plans/prepared_private';shutil.copytree(evidence.delivery['technical'],prepared)
    clock=NS(value=1_000_000_000);now=lambda:clock.value;controller=tv.PresentationRun
    monkeypatch.setattr(tv,'time',NS(perf_counter_ns=now,monotonic=time.monotonic))
    monkeypatch.setattr(tv,'PresentationRun',lambda *a,**kw:controller(*a,clock=now,**kw))
    root,app,errors=app_fixture(tmp_path,monkeypatch);draws=[];draw=app.draw
    monkeypatch.setattr(app,'draw',lambda im:(draws.append(im.copy()),draw(im)))
    monkeypatch.setattr(tv.filedialog,'askdirectory',lambda **kw:str(prepared))
    try:
        app.open_prepared();deadline=time.monotonic()+10
        while app._selection_loading and time.monotonic()<deadline:root.update();time.sleep(.01)
        assert app.prepared==prepared and not app._selection_error,app.status.get()
        app.show_calibration();viewer=app.viewer;app.begin_countdown()
        assert app.mode=='COUNTDOWN' and app.viewer is viewer,app.status.get()
        clock.value=app.count_end
        for i in range(5):
            deadline=time.monotonic()+10
            while app.mode!='RUNNING' and time.monotonic()<deadline:app._tick();root.update();time.sleep(.01)
            assert app.mode=='RUNNING' and app.run.current==i,app.status.get()
            clock.value=app.run.deadline;app._tick()
            if i<4:clock.value=app.transition_end
        assert app.mode=='COMPLETED' and app._post_roll_hold and app.viewer is viewer
        m=app.manifest;state,events,_=tl.read_run(app.run.directory/'run.json',m,app.run.manifest_sha)
        context=tl.marker_context(state,events,m,app.run.manifest_sha,prepared=prepared)
        decoded=[mk.decode(im,m,app.run.manifest_sha,state['run_id'],run_context=context) for im in draws]
        assert all(len(d.get('decoded_tokens',[]))==2 and all(d['decoded_tokens']) and len(set(d['decoded_tokens']))==1 and sorted(d['detected_ids'])==[0,1,2,3] for d in decoded)
        phases=['P' if d['status']=='PASS' else 'T' for d in decoded]
        # TEST_ONLY, countdown 5, countdown 0, then the five photographs.
        assert phases==['T','T','T','P','T','P','T','P','T','P','T','P','T'],phases
        post=next(e['token'] for e in state['qr_binding']['entries'] if e['identity']=='POSTROLL')
        assert decoded[-1]['decoded_tokens']==[post,post] and state['status']=='COMPLETED'
        before={p.name:photo.digest(p) for p in app.run.directory.iterdir() if p.is_file()}
        app.stop_btn.invoke();assert app.mode=='COMPLETED' and app.viewer is None
        assert before=={p.name:photo.digest(p) for p in app.run.directory.iterdir() if p.is_file()} and not errors
        draws[phases.index('P')].save(evidence.out/'HOST_v3_first_P.png');draws[-1].save(evidence.out/'HOST_v3_POSTROLL.png')
        save(evidence.out,'T3.json',dict(status='PASS_NATIVE_TK_HOST',selected_without_error=True,phases=phases,independent_qr_pairs=len(decoded),
            source_photos=5,completed=5,no_tv=True,A12_TV='NOT_VERIFIED',postroll_preserved=True))
    finally:close_app(root,app)
