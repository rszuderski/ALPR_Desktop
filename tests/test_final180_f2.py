"""F2 private/synthetic acceptance; never a physical A12/TV acceptance."""
from copy import deepcopy
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace as NS
import cv2
import numpy as np
import pytest
from PIL import Image,ImageOps
from auto_annotation_tool.ranking import final180_markers as mk,final180_reference as ref,final180_timeline as tl,photo_reference as photo
from auto_annotation_tool.gui import final180_tv as tv
from test_final180_desktop import frozen,material


def plan(size=(1920,1080),n=176):
    return dict(layout=mk.layout_f2(*size),geometry_policy=deepcopy(mk.GEOMETRY_POLICY),mode='TECHNICAL_NON_FINAL',
        exposures=[dict(position=i,exposure_id=f'e{i:04d}') for i in range(1,n+1)])


def context(m,sha='a'*64,run='b'*32):
    binding=mk.f2_binding(m,sha,run)
    return dict(binding=binding,observed_tokens=[e['token'] for e in binding['entries']])


def test_all_176_photos_177_transitions_and_test_only_qr_roundtrip():
    m=plan();binding=mk.f2_binding(m,'a'*64,'b'*32)
    assert len(binding['entries'])==353 and len({e['token'] for e in binding['entries']})==353
    entries=binding['entries']+mk.f2_binding(m,'a'*64,'c'*32,test_only=True)['entries']
    first,_=mk.render(None,m['layout'],entries[0]['token']);geometry=mk.geometry(first,m['layout'])
    assert geometry['status']=='PASS'
    for e in entries:
        q=mk.qr_f2(e['token']);assert len(e['token'])==20 and q.shape==(33,33)
        assert np.all(q[:4]==255) and np.all(q[-4:]==255) and np.all(q[:,:4]==255) and np.all(q[:,-4:]==255)
        # Exercise the actual source-image rectifier/decoder, not a standalone
        # detector on a tightly cropped QR (which has different raster borders).
        image,_=mk.render(None,m['layout'],e['token'])
        decoded=mk.decode_codes(image,m['layout'],geometry)
        assert decoded['decoded_tokens']==[e['token'],e['token']]


@pytest.mark.parametrize('size',[(1280,720),(1920,1080),(2400,1350),(2560,1440)])
def test_layout_pair_pixel_modules_geometry_and_safety_margin(size,monkeypatch):
    m=plan(size);b=m['layout'];token=mk.payload(m,'a'*64,'b'*32,'e0001','P')
    original=mk.qr_f2;calls=[]
    monkeypatch.setattr(mk,'qr_f2',lambda v:calls.append(v) or original(v))
    im,fit=mk.render(Image.new('RGB',(1600,900),'#233344'),b,token)
    assert calls==[token,token]
    module=b['qr']['module_px'];q=original(token)
    for rect in b['qr_rects']:
        pixels=np.asarray(im.crop(rect).convert('L'));assert set(np.unique(pixels))=={0,255}
        assert np.array_equal(pixels,np.repeat(np.repeat(q,module,0),module,1))
    d=mk.decode(im,m,'a'*64,'b'*32,run_context=context(m))
    assert d['status']=='PASS' and d['exposure_id']=='e0001' and d['decoded_tokens']==[token,token]
    assert mk.input_exclusion(d,b,b['photo_rect'])['status']=='PASS'
    projected,_=mk.transform_gt([[0,0],[1600,0],[1600,900],[0,900]],(1600,900),b,d)
    from auto_annotation_tool.ranking.final180_mobile import _inside,_bbox
    assert _inside(_bbox(projected),b['photo_rect'])
    neutral,_=mk.render(None,b,token)
    board,identity=tv.calibration_frame(m,'a'*64,'c'*32)
    expected,_=mk.render(None,b,identity['token'])
    for rect in b['qr_rects']+[a['rect'] for a in b['aruco']]:assert board.crop(rect).tobytes()==expected.crop(rect).tobytes()
    assert identity['purpose']=='TEST_ONLY' and identity['phase']=='T'
    assert mk.decode(board,m,'a'*64,'b'*32,run_context=context(m))['status']=='N/A'
    assert fit['rect']==mk.photo_placement((1600,900),b)['rect']


@pytest.mark.parametrize('size',[(640,480),(1279,720),(1280,719)])
def test_unsafe_display_is_blocked(size):
    with pytest.raises(ValueError,match='F2_MONITOR_MINIMUM'):mk.layout_f2(*size)


@pytest.mark.parametrize('ratio',[(1500,1000),(1200,900),(1600,900),(600,1000)])
@pytest.mark.parametrize('orientation',range(1,9))
def test_fit_exif_and_actual_rect(ratio,orientation):
    m=plan();im=Image.new('RGB',ratio,'#123456');exif=im.getexif();exif[274]=orientation
    im.info['exif']=exif.tobytes();oriented=ImageOps.exif_transpose(im)
    rendered,fit=mk.render(im,m['layout'],mk.payload(m,'a'*64,'b'*32,'e0001','P'))
    l,t,r,b=fit['rect'];w,h=oriented.size
    assert fit['source_size']==[w,h] and abs((r-l)-w*(b-t)/h)<=1.1
    field=m['layout']['photo_rect'];inset=m['layout']['image_inset_px']
    assert l>=field[0]+inset and t>=field[1]+inset and r<=field[2]-inset and b<=field[3]-inset
    assert rendered.crop((l,t,r,b)).getextrema()==((18,18),(52,52),(86,86))


@pytest.mark.parametrize('fault',['erase_left','erase_right','conflict','check','foreign','wrong_phase','missing_aruco','duplicate_aruco','missing_index','unlogged'])
def test_bad_evidence_never_qualifies(fault):
    m=plan();b=m['layout'];c=context(m);token=mk.payload(m,'a'*64,'b'*32,'e0001','P')
    im,_=mk.render(None,b,token)
    if fault.startswith('erase'):im.paste('white',tuple(b['qr_rects'][0 if fault=='erase_left' else 1]))
    elif fault in ('conflict','foreign','wrong_phase','check'):
        v=mk.payload(m,'a'*64,'c'*32,'e0001','P') if fault in ('foreign','conflict') else mk.payload(m,'a'*64,'b'*32,'e0001','T')
        if fault=='check':v=token[:-1]+('A' if token[-1]!='A' else 'B')
        other,_=mk.render(None,b,v)
        if fault=='conflict':rect=b['qr_rects'][1];im.paste(other.crop(rect),tuple(rect[:2]))
        else:im=other
    elif fault=='missing_aruco':im.paste('white',tuple(b['aruco'][0]['rect']))
    elif fault=='duplicate_aruco':a=b['aruco'];im.paste(im.crop(a[0]['rect']),tuple(a[1]['rect'][:2]))
    elif fault=='missing_index':c=None
    else:c['observed_tokens']=[]
    d=mk.decode(im,m,'a'*64,'b'*32,run_context=c)
    assert d['status']=='N/A' and not d.get('exposure_id')
    if fault.startswith('erase'):assert sum(bool(v) for v in d['decoded_tokens'])==1


def test_relayout_preserves_reference_bytes_and_checks_frozen_geometry(tmp_path):
    old=frozen(tmp_path/'legacy');before={str(p.relative_to(old)):photo.digest(p) for p in old.rglob('*') if p.is_file()}
    d=ref.relayout(old,(1280,720));m,_,_=ref.load(d)
    assert m['layout_version']==mk.F2_LAYOUT and m['qr_version']==mk.F2_QR
    for name,sha in before.items():
        assert photo.digest(old/name)==sha
        if name!='manifest.json':assert photo.digest(d/name)==sha
    assert not (d/'runs').exists() and len(m['exposures'])==3
    assert m['exposures'][0]['actual_image_rect']==m['exposures'][0]['photo_placement']['rect']
    later=ref.replan(d,7.5);assert ref.load(later)[0]['layout']==m['layout']
    changed=deepcopy(m);changed['exposures'][0]['actual_image_rect'][0]+=1
    (d/'manifest.json').write_text(json.dumps(changed),'utf-8')
    with pytest.raises(ValueError,match='F2_FROZEN_LAYOUT'):ref.load(d)


def test_176_order_stop_resume_tokens_and_journal_proof(tmp_path):
    old=frozen(tmp_path/'legacy',n=166,negative=10,dwell_seconds=5);d=ref.relayout(old,(1280,720));m,mat,_=ref.load(d)
    now=[0];run=tv.PresentationRun(d,m,clock=lambda:now[0]);sha=run.manifest_sha;rid=run.state['run_id']
    for i in range(176):
        if i==73:
            saved=run.directory;run.stop();run=tv.PresentationRun(d,m,resume=saved,clock=lambda:now[0]);assert run.next_index==73 and run.state['run_id']==rid
        e=m['exposures'][i];tok=mk.payload(m,sha,rid,e['exposure_id'],'P')
        run.present(i,{'display_token':tok},lambda:None);now[0]+=5_000_000_000
        if i==175:run.complete(lambda:None)
        else:run.end_photo();run.replace(lambda:None,display_token=mk.payload(m,sha,rid,m['exposures'][i+1]['exposure_id'],'T'))
    state,events,_=tl.read_run(run.directory/'run.json',m,sha);c=tl.marker_context(state,events,m,sha)
    projected=tl.project(m,events)
    assert len(projected)==176 and all(p['status']=='COMPLETED' and p['duration_pc_s']==5 for p in projected)
    assert len([e for e in events if e['event']=='PHOTO_VIEW_UPDATED'])==176
    assert mk.qualify_f2(mk.payload(m,sha,rid,'e0001','P'),m,sha,rid,c)['status']=='PASS'
    other=tv.PresentationRun(d,m);assert other.state['qr_binding']['run_tag']!=state['qr_binding']['run_tag'];other.stop()
    assert mk.qualify_f2(mk.payload(m,sha,other.state['run_id'],'e0001','P'),m,sha,rid,c)['status']=='N/A'


def test_tag_collision_and_corrupt_run_index_fail_closed(tmp_path,monkeypatch):
    d=ref.relayout(frozen(tmp_path/'legacy'),(1280,720));m,_,_=ref.load(d)
    original=mk._short;monkeypatch.setattr(mk,'_short',lambda value,n:'A'*n if n==10 else original(value,n))
    a=tv.PresentationRun(d,m);a.stop()
    with pytest.raises(ValueError,match='F2_RUN_TAG_COLLISION'):tv.PresentationRun(d,m)
    state=photo.read_json(a.directory/'run.json');state['qr_binding']['entries'][0]['token']='INVALID'
    (a.directory/'run.json').write_text(json.dumps(state),'utf-8')
    with pytest.raises(ValueError,match='F2_RUN_INDEX_MISMATCH'):tl.read_run(a.directory/'run.json',m,a.manifest_sha)


def test_native_hidden_tk_calibration_start_gate_and_technical_resume(tmp_path,monkeypatch):
    if os.environ.get('F2_TK_ISOLATED')!='1':
        child=subprocess.run([sys.executable,'-m','pytest',str(Path(__file__).resolve())+'::test_native_hidden_tk_calibration_start_gate_and_technical_resume','-q'],env=dict(os.environ,F2_TK_ISOLATED='1'),capture_output=True,text=True,timeout=90)
        assert child.returncode==0,child.stdout+child.stderr;return
    import tkinter as tk
    legacy=tv.legacy_module(tv.LEGACY_DEFAULT);monkeypatch.setattr(tv.monitor_dpi,'final_monitors',lambda: [dict(name='PRIVATE_TEST',w=1280,h=720,x=0,y=0,verified=True,geometry_source='ISOLATED_TEST')])
    root=tk.Tk();root.withdraw();args=NS(xml=None,positives=None,negatives=None,output=tmp_path/'plans',dwell=5,negatives_declared=True,unreadable_ids=[])
    app=tv.make_app(legacy,args,parent=root);app.window.withdraw()
    def hidden(_):app.viewer=tk.Toplevel(root);app.viewer.withdraw();app.canvas=tk.Canvas(app.viewer)
    monkeypatch.setattr(app,'_make_viewer',hidden);monkeypatch.setattr(app,'_tick',lambda:None)
    d=ref.relayout(frozen(tmp_path/'source'),(1280,720));m,_,_=ref.load(d);before={p:photo.digest(p) for p in d.rglob('*') if p.is_file()}
    try:
        app.loaded(d);app.show_calibration();root.update()
        assert app.run is None and app.mode=='CALIBRATION' and not (d/'runs').exists()
        assert app.viewer.bind('<Escape>') and not app.viewer.bind('<space>')
        app.calibration_btn.invoke();assert app.mode=='IDLE' and app.viewer is None
        assert all(photo.digest(p)==sha for p,sha in before.items())
        app.begin_countdown();assert app.mode=='COUNTDOWN' and app.run.state['qr_binding']
        saved=app.run.directory;app.abort();app.begin_countdown(resume=saved);assert app.run.directory==saved;app.abort()
        # V3: presentation readiness is independent of research/device acceptance.
        final_d=ref.relayout(d,(1280,720));m,_,_=ref.load(final_d)
        m['mode']='FINAL';m['status']='NO_GO';(final_d/'manifest.json').write_text(json.dumps(m),'utf-8');app.loaded(final_d)
        count=len(list((final_d/'runs').glob('*')));app.begin_countdown()
        assert app.mode=='COUNTDOWN' and len(list((final_d/'runs').glob('*')))==count+1,app.status.get()
        assert app.manifest['acceptance']['a12_tv']=='NOT_VERIFIED';app.abort()
    finally:app._close();root.destroy()


def test_technical_no_gt_never_claims_reference_or_final_ready(tmp_path):
    a=material(tmp_path/'source',empty=(0,1),negative=0)
    d=ref.freeze(a,tmp_path/'plans',technical=True,layout_version=mk.F2_LAYOUT)
    m,_,gt=ref.load(d)
    assert m['input_errors'] and ref.display_ready(m) and m['status']=='TECHNICAL_ONLY'
    assert all(gt.state(e['image_id'])['presence']=='UNRESOLVED' for e in m['exposures'])
    final=deepcopy(m);final['mode']='FINAL';assert not ref.display_ready(final)
    bad=deepcopy(m);bad['input_errors'].append('PHOTO_SHA_CHANGED');assert not ref.display_ready(bad)


def test_r0_final_alias_is_only_for_f2_and_explicit_native_contract():
    from auto_annotation_tool.ranking import final180_mobile as mobile
    stages=dict(mp=dict(enabled=False),mt=dict(runtime='tflite',precision='fp32',enabled=True),mz=dict(runtime='tflite',precision='fp32',enabled=True))
    coll=dict(variant='R0_FINAL',execution=dict(variant='R0_FINAL',roi_budget_policy='r0_full_frame',capture_profile='f6_final180_v1',stages=stages))
    bundle=NS(collection_session=coll,model_refs={},report_payload={})
    assert mobile.configuration(bundle,{})['status']=='N/A'  # historical F6v1 unchanged
    assert mobile.configuration(bundle,{},layout_version=mk.F2_LAYOUT)['status']=='PASS'
    for field,value in [('variant','R1'),('roi_budget_policy','r1_one_roi'),('capture_profile','other')]:
        b=deepcopy(bundle);b.collection_session['execution'][field]=value
        assert mobile.configuration(b,{},layout_version=mk.F2_LAYOUT)['status']=='N/A'


def test_registered_calibration_is_technical_without_mutating_plan_or_creating_run(tmp_path):
    d=ref.relayout(frozen(tmp_path/'legacy'),(1280,720));m,_,_=ref.load(d);sha=photo.digest(d/'manifest.json')
    before={p:photo.digest(p) for p in d.rglob('*') if p.is_file()};frame,identity=tv.calibration_frame(m,sha)
    proof=tl.register_calibration(d,identity)
    assert proof.parent==d.parent/'calibration_previews' and not(d/'runs').exists()
    assert all(photo.digest(p)==s for p,s in before.items())
    run=tv.PresentationRun(d,m);run.stop();state,events,_=tl.read_run(run.directory/'run.json',m,sha)
    context=tl.marker_context(state,events,m,sha)
    decoded=mk.decode(frame,m,sha,state['run_id'],run_context=context)
    assert decoded['reason']=='TECHNICAL_TRANSITION_FRAME' and decoded['technical_purpose']=='TEST_ONLY'
    assert not decoded.get('exposure_id') and decoded['decoded_tokens']==[identity['token']]*2
    other=deepcopy(context);other['technical_tokens']=[]
    assert mk.decode(frame,m,sha,state['run_id'],run_context=other)['reason']=='FOREIGN_MANIFEST_OR_RUN'


def test_native_f2_source_pixels_to_mt_final_mz_without_geometry_stub(tmp_path):
    import io,zipfile
    from auto_annotation_tool.ranking import final180_mobile as mobile
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    from test_final180_timeline import FIXTURES,mutate_fixture
    d=ref.relayout(frozen(tmp_path/'legacy'),(1920,1080));m,_,_=ref.load(d);run=tv.PresentationRun(d,m)
    token=mk.payload(m,run.manifest_sha,run.state['run_id'],'e0001','P');image,_=mk.render(None,m['layout'],token)
    source=MobileReviewSession.open(FIXTURES/'r0-positive.alprsession',sidecar_path=tmp_path/'original.absent.json')
    record=source.tv_bbox_evidence['rows'][0]['source_image'];entry=record['entry'];oldsha=record['sha256']
    buffer=io.BytesIO();image.resize((record['width_px'],record['height_px']),Image.Resampling.LANCZOS).save(buffer,format='JPEG',quality=96)
    raw=buffer.getvalue();sha=hashlib.sha256(raw).hexdigest()
    def modify(data):
        for name in list(data):
            if hashlib.sha256(data[name]).hexdigest()==oldsha:data[name]=raw  # producer fixture reuses identical image bytes
        for name in list(data):
            if name.endswith(('.json','.jsonl')):data[name]=data[name].replace(oldsha.encode(),sha.encode())
    archive=mutate_fixture(tmp_path/'synthetic-f2.alprsession',modify)
    session=MobileReviewSession.open(archive,sidecar_path=tmp_path/'f2.absent.json')
    assert session.bundle.validation.ok,session.bundle.validation.errors
    run.present(0,{},lambda:None);run.stop(clear=lambda:None)
    state,events,_=tl.read_run(run.directory/'run.json',m,run.manifest_sha);c=tl.marker_context(state,events,m,run.manifest_sha)
    frames,gaps=mobile._native_frames(session,m,run.manifest_sha,run.state['run_id'],run_context=c)
    f=frames[0];assert f['identity_status']=='PASS',f['geometry_reason']
    assert f['exposure_id']=='e0001' and f['marker_decoding']['decoded_tokens']==[token,token]
    assert f['mt_invocation_ids']==source.f6_evidence['frames'][f['frame_key']]['mt_invocation_ids']
    assert f['final_population_complete'] and f['finals'][0]['mz']['raw_prediction']=='A123'
    assert f['finals'][0]['mz']['source_bound'] and not session.sidecar_path.exists()
