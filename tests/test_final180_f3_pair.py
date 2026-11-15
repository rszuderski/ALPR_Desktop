"""Four bounded Desktop controls. Synthetic/host evidence is never A12 acceptance."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import ast,hashlib,io,json,math,os,shutil,time,zipfile
import cv2
import numpy as np
import pytest
from PIL import Image
from auto_annotation_tool.ranking import final180_markers as mk,final180_reference as ref,final180_timeline as tl,final180_mobile as mobile,photo_reference as photo
from auto_annotation_tool.gui import final180_tv as tv
from test_final180_final_blockers import app_fixture,close_app,plan,archive_image,finished,picture


@pytest.fixture(scope='module')
def evidence():
    value=os.environ.get('F6_F3_EVIDENCE')
    if not value:pytest.skip('Private prepared/evidence paths required for four delivery controls')
    out=Path(value).resolve()
    return NS(out=out,paths=photo.read_json(out/'paths.json'),delivery=photo.read_json(out/'delivery.json'))


def save(out,name,value):
    (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8')


def context(m,sha,rid):
    binding=mk.f3_binding(m,sha,rid)
    return dict(binding=binding,observed_tokens=[e['token'] for e in binding['entries']],technical_tokens=[])


def test_D1_qr_v1_h_all_slots_pair_faults_registry_and_historical_f2(evidence,tmp_path,monkeypatch):
    m,_,_=ref.load(evidence.delivery['final']);sha=evidence.delivery['final_manifest_sha256'];rid='b'*32
    c=context(m,sha,rid);entries=c['binding']['entries'];test=mk.f3_binding(m,sha,'c'*32,test_only=True)
    assert len(entries)==355 and len({e['pair_tag'] for e in entries})==355
    detector=cv2.QRCodeDetector();decoded_count=0;bound_fallbacks=0
    for e in entries+test['entries']:
        assert e['token'][0]!=e['token'][1] and all(len(v)==10 for v in e['token'])
        for value in e['token']:
            raster=mk.qr_f3(value);assert raster.shape==(29,29)
            assert all(np.all(a==255) for a in (raster[:4],raster[-4:],raster[:,:4],raster[:,-4:]))
            large=np.repeat(np.repeat(raster,8,0),8,1)
            found=detector.detectAndDecode(large)[0]
            if not found:
                # Same frozen-symbol-bounds route as the production rectifier;
                # localization failure alone is not an encoding/ECC failure.
                points=np.array([[[32,32],[199,32],[199,199],[32,199]]],dtype=np.float32)
                found=detector.decode(large,points)[0];bound_fallbacks+=1
            assert found==value
            decoded_count+=1
    for i,e in enumerate(entries):
        d=mk.qualify_f3(e['token'],m,sha,rid,c)
        assert d['pair_verified'] and d['phase']==e['phase']
        assert (d['status']=='PASS')==(e['phase']=='P')
        adjacent=entries[(i+1)%len(entries)]
        assert mk.qualify_f3([e['token'][0],adjacent['token'][1]],m,sha,rid,c)['status']=='N/A'
    pair=next(e['token'] for e in entries if e['phase']=='P')
    foreign=mk.payload(m,sha,'d'*32,'e0001','P')
    cases=[['',pair[1]],[pair[0],''],pair[::-1],[pair[0],pair[0]],foreign,
        [pair[0],'RZZZP'+pair[1][5:]],mk.payload(m,sha,rid,'e0001','T')]
    for value in cases:assert mk.qualify_f3(value,m,sha,rid,c)['status']=='N/A'
    assert mk.qualify_f3(pair,m,'e'*64,rid,c)['status']=='N/A'
    assert mk.qualify_f3(pair,m,sha,rid,dict(c,observed_tokens=[]))['status']=='N/A'
    assert mk.qualify_f3(pair,m,sha,rid,None)['status']=='N/A'
    ct=dict(c,technical_tokens=[test['entries'][0]['token']])
    assert mk.qualify_f3(test['entries'][0]['token'],m,sha,rid,ct)['technical_purpose']=='TEST_ONLY'

    prepared,pm=plan(tmp_path/'registry',version=mk.F3_LAYOUT);r=tv.PresentationRun(prepared,pm);r.stop()
    original=mk._short
    with monkeypatch.context() as patch:
        patch.setattr(mk,'_short',lambda value,n:r.state['qr_binding']['run_tag'] if n==4 else original(value,n))
        with pytest.raises(ValueError,match='F3_RUN_TAG_COLLISION'):tv.PresentationRun(prepared,pm)
    with monkeypatch.context() as patch:
        patch.setattr(mk,'_short',lambda value,n:'A'*n if n==5 else original(value,n))
        with pytest.raises(ValueError,match='F3_PAIR_TAG_COLLISION'):mk.f3_binding(pm,r.manifest_sha,'e'*32)
    newid='f'*32;collision=r.state['qr_binding']['entries'][0]['pair_tag']
    prefix='F3_PAIR|RUN|'+r.manifest_sha+'|'+newid+'|0|COUNTDOWN|T'
    with monkeypatch.context() as patch:
        patch.setattr(mk,'_short',lambda value,n:collision if value==prefix else original(value,n))
        bad=mk.f3_binding(pm,r.manifest_sha,newid)
        with pytest.raises(ValueError,match='F3_PAIR_TAG_COLLISION'):
            tl.check_run_tag(prepared,dict(manifest_sha256=r.manifest_sha,run_id=newid,qr_binding=bad))
    changed=photo.read_json(r.directory/'run.json');changed['qr_binding']['entries'][0]['token'][0]='LAAAAAAAAA'
    save(r.directory,'run.json',changed)
    with pytest.raises(ValueError,match='F3_RUN_INDEX_MISMATCH'):tl.read_run(r.directory/'run.json',pm,r.manifest_sha)

    historical=photo.read_json(evidence.paths['final_evaluation']);old=Path(evidence.paths['final']);om,_,_=ref.load(old)
    oldrun=next(p for p in old.glob('runs/run_*/run.json') if photo.read_json(p)['run_id']==historical['run_id'])
    state,events,_=tl.read_run(oldrun,om,photo.digest(old/'manifest.json'));oc=tl.marker_context(state,events,om,state['manifest_sha256'])
    frame=next(f for f in historical['frames'] if f['identity_status']=='PASS');record=frame['source_image']
    archive=Path(historical['source_archive_path']);assert photo.digest(archive)==historical['source_archive_sha256']
    with zipfile.ZipFile(archive) as z:raw=z.read(record['entry'])
    assert hashlib.sha256(raw).hexdigest()==record['sha256']
    with Image.open(io.BytesIO(raw)) as image:d=mk.decode(image,om,state['manifest_sha256'],state['run_id'],run_context=oc)
    assert d['status']=='PASS' and d['decoded_tokens']==frame['marker_decoding']['decoded_tokens']
    save(evidence.out,'D1.json',dict(status='PASS',run_slots=len(entries),test_only_slots=1,independent_qr_roundtrips=decoded_count,
        qr_version=1,ecc='H',module_grid=[21,21],format_copies_checked=2,version_bits='ABSENT_FOR_V1_USE_GRID_DIMENSIONS',
        bounded_symbol_fallbacks=bound_fallbacks,
        all_adjacent_mixed_pairs_rejected=True,tag_and_pair_collisions_rejected=True,
        historical_f2=dict(archive_sha256=historical['source_archive_sha256'],source_image_sha256=record['sha256'],frame_key=frame['frame_key'],status=d['status'])))


def test_D2_five_photos_same_layout_negative_pixels_and_historical_geometry(evidence):
    prepared=Path(evidence.delivery['technical']);m,mat,_=ref.load(prepared);sha=photo.digest(prepared/'manifest.json');rid='b'*32
    board=m['layout'];old=photo.read_json(Path(evidence.paths['technical'])/'manifest.json')['layout'];c=context(m,sha,rid)
    assert board['photo_rect']==old['photo_rect'] and board['aruco']==old['aruco']
    assert board['qr']['module_pixel_widths']==[7,8] and all(r[2]-r[0]==210 for r in board['qr_rects'])
    assert board['qr']['active_symbol_px']==152
    assert board['photo_rect'][0]-board['qr_rects'][0][2]==57
    calibration,identity=tv.calibration_frame(m,sha,technical_id='c'*32)
    neutral,_=mk.render(None,board,identity['token'])
    for rect in board['qr_rects']+[a['rect'] for a in board['aruco']]:assert calibration.crop(rect).tobytes()==neutral.crop(rect).tobytes()
    calibration.save(evidence.out/'HOST_F3_TEST_ONLY.png')
    for i,e in enumerate(m['exposures']):
        pair=mk.payload(m,sha,rid,e['exposure_id'],'P');image,fit=mk.render(mat.image(e['image_id']),board,pair)
        decoded=mk.decode(image,m,sha,rid,run_context=c)
        assert decoded['status']=='PASS' and decoded['decoded_tokens']==pair and sorted(decoded['detected_ids'])==[0,1,2,3]
        assert fit['rect']==e['actual_image_rect']
        if i==0:
            first=image;image.save(evidence.out/'HOST_F3_FIRST_P.png')
    for index in (0,1):
        missing=first.copy();missing.paste('white',tuple(board['qr_rects'][index]))
        decoded=mk.decode(missing,m,sha,rid,run_context=c)
        assert decoded['status']=='N/A' and sum(bool(v) for v in decoded['decoded_tokens'])==1
    a=board['aruco'][0]['rect'];missing=first.copy();missing.paste('white',tuple(a))
    assert mk.decode(missing,m,sha,rid,run_context=c)['reason']=='ARUCO_MISSING_OR_DUPLICATED'
    rect=board['qr_rects'][1];clone=first.copy();clone.paste(first.crop(board['qr_rects'][0]),tuple(rect[:2]))
    assert mk.decode(clone,m,sha,rid,run_context=c)['status']=='N/A'
    for identity in ('COUNTDOWN','PREROLL','POSTROLL'):
        pair=mk.payload(m,sha,rid,identity,'T');im,_=mk.render(None,board,pair);d=mk.decode(im,m,sha,rid,run_context=c)
        assert d['pair_verified'] and d['reason']=='TECHNICAL_TRANSITION_FRAME'
        if identity=='POSTROLL':im.save(evidence.out/'HOST_F3_POSTROLL.png')

    previous=photo.read_json(evidence.paths['historic_evaluation'])
    frames=[f for f in previous['frames'] if 'matrix_tv_to_camera' in f.get('marker_decoding',{})];assert len(frames)==14
    fm,_,_=ref.load(evidence.delivery['final']);bounds=lambda q:[float(q[:,0].min()),float(q[:,1].min()),float(q[:,0].max()),float(q[:,1].max())]
    rois=[]
    for f in frames:
        b=bounds(mk.project(mk.rect_points(board['photo_rect']),f['marker_decoding']['matrix_tv_to_camera']))
        rois.append([math.floor(b[0]),math.floor(b[1]),math.ceil(b[2]),math.ceil(b[3])])
    envelope=[min(r[0] for r in rois),min(r[1] for r in rois),max(r[2] for r in rois),max(r[3] for r in rois)]
    stress=[envelope[0]-6,envelope[1]-6,envelope[2]+6,envelope[3]+6];slack=float('inf')
    for f in frames:
        d=dict(f['marker_decoding'],status='PASS')
        assert mk.input_exclusion(d,board,stress)['status']=='PASS'
        for r in rois:
            assert mk.input_exclusion(d,board,[r[0]-6,r[1]-6,r[2]+6,r[3]+6])['status']=='PASS'
        for e in fm['exposures']:
            b=bounds(mk.project(mk.rect_points(e['actual_image_rect']),d['matrix_tv_to_camera']))
            slack=min(slack,b[0]-envelope[0]-6,b[1]-envelope[1]-6,envelope[2]-b[2]-6,envelope[3]-b[3]-6)
    assert slack>0
    save(evidence.out,'D2.json',dict(status='PASS_HOST_REPROJECTION',photographs_rendered=5,historical_homographies=14,
        photographs_contained=176,photo_rect=board['photo_rect'],qr_rects=board['qr_rects'],full_qr_pixels=210,
        module_pixel_widths=[7,8],mean_module_px=210/29,active_symbol_px=152,
        marker_intersections=0,stress_roi=stress,min_photo_slack_px=slack,photo_placement_unchanged=True,A12_TV='NOT_VERIFIED'))


def test_D3_native_tk_five_p_t_completed_import_one_archive_and_na(evidence,tmp_path,monkeypatch):
    prepared=tmp_path/'plans/prepared_private';shutil.copytree(evidence.delivery['technical'],prepared)
    clock=NS(value=1_000_000_000);now=lambda:clock.value;controller=tv.PresentationRun
    monkeypatch.setattr(tv,'time',NS(perf_counter_ns=now,monotonic=time.monotonic))
    monkeypatch.setattr(tv,'PresentationRun',lambda *a,**kw:controller(*a,**({'clock':now}|kw)))
    root,app,errors=app_fixture(tmp_path,monkeypatch);draws=[];draw=app.draw
    monkeypatch.setattr(app,'draw',lambda image:(draws.append(image.copy()),draw(image)))
    try:
        app.loaded(prepared);app.show_calibration();viewer=app.viewer
        assert not (prepared/'runs').exists()
        app.begin_countdown();assert app.mode=='COUNTDOWN' and app.viewer is viewer,app.status.get()
        clock.value=app.count_end
        for i in range(5):
            deadline=time.monotonic()+10
            while app.mode!='RUNNING' and time.monotonic()<deadline:app._tick();root.update();time.sleep(.01)
            assert app.mode=='RUNNING' and app.run.current==i,app.status.get()
            clock.value=app.run.deadline;app._tick()
            if i<4:clock.value=app.transition_end
        assert app.mode=='COMPLETED' and app._post_roll_hold
        m=app.manifest;r=app.run;state,events,_=tl.read_run(r.directory/'run.json',m,r.manifest_sha)
        ctx=tl.marker_context(state,events,m,r.manifest_sha,prepared=prepared)
        decoded=[mk.decode(im,m,r.manifest_sha,state['run_id'],run_context=ctx) for im in draws]
        phases=['P' if d['status']=='PASS' else 'T' for d in decoded]
        assert all(d['pair_verified'] and d['decoded_tokens'][0]!=d['decoded_tokens'][1] for d in decoded)
        assert phases==['T','T','T','P','T','P','T','P','T','P','T','P','T']
        assert all(row['status']=='COMPLETED' for row in tl.project(m,events)) and events[-1]['event']=='RUN_COMPLETED'
        hashes={p.name:photo.digest(p) for p in r.directory.iterdir() if p.is_file()}
        clock.value+=60_000_000_000;app._tick();app.stop_btn.invoke()
        assert app.mode=='COMPLETED' and app.viewer is None and 'Przerwany' not in app.status.get()
        assert hashes=={p.name:photo.digest(p) for p in r.directory.iterdir() if p.is_file()}
        # Native reader, real pixel decode and existing exporter, on an explicitly synthetic archive.
        archive=archive_image(tmp_path/'archive',draws[phases.index('P')]);match=mobile.locate_run(archive,[prepared.parent])
        assert match['status']=='PASS' and match['run_path']==str((r.directory/'run.json').resolve()),match
        result=mobile.analyze_import(archive,r.directory/'run.json',output=tmp_path/'results',run_match=match)
        report=photo.read_json(result/'evaluation.json')
        assert report['metrics_status']=='NOT_AVAILABLE_NO_GT' and report['summary']['system_full_read']['rate'] is None
        assert (result/'readability_preflight.json').is_file() and report['readability_preflight']['summary']['verified_pairs']>0
        assert report['readability_preflight']['a12_f3_acceptance']=='NOT_VERIFIED'
        legacy,lm=plan(tmp_path/'legacy',version=mk.F2_CLEAR_LAYOUT);lr=finished(legacy,lm)
        legacy_archive=archive_image(tmp_path/'legacy_archive',picture(lm,lr))
        legacy_match=mobile.locate_run(legacy_archive,[prepared.parent,legacy.parent])
        assert legacy_match['status']=='PASS' and legacy_match['run_proof']['run_id']==lr.state['run_id'],legacy_match
        f3_match=mobile.locate_run(archive,[prepared.parent,legacy.parent])
        assert f3_match['status']=='PASS' and f3_match['run_proof']['run_id']==r.state['run_id'],f3_match
        clone=tmp_path/'clone'/prepared.name;shutil.copytree(prepared,clone)
        ambiguous=mobile.locate_run(archive,[prepared.parent,clone.parent]);assert ambiguous['status']=='N/A'
        monkeypatch.setattr(tv,'DEFAULT_EXPERIMENT',prepared.parent);dialogs=[];panels=[]
        monkeypatch.setattr(tv.filedialog,'askopenfilename',lambda **kw:dialogs.append(kw['title']) or str(archive))
        monkeypatch.setattr(tv,'FinalResultsWindow',lambda *a,**kw:panels.append(a[1]))
        app.import_android();deadline=time.monotonic()+40
        while not panels and time.monotonic()<deadline:root.update();time.sleep(.02)
        assert panels and len(dialogs)==1 and not app.manual_run_btn.winfo_manager(),app.status.get()
        assert not errors
        save(evidence.out,'D3.json',dict(status='PASS_NATIVE_TK_HOST',phases=phases,completed=5,postroll_preserved=True,
            calibration_creates_run=False,late_close_preserves_completed=True,archive_dialogs=1,
            synthetic_import_result=str(result),ambiguous=ambiguous['reason'],no_gt_metrics=report['metrics_status'],A12_TV='NOT_VERIFIED'))
    finally:close_app(root,app)


def test_D4_new_preparations_all_original_bytes_and_unchanged_scoring(evidence):
    hashes=photo.read_json(evidence.out/'protection_before.json');allowed=set(evidence.paths['code']);changed=[]
    for name,stamp in hashes.items():
        if name in allowed:continue
        p=Path(name);s=p.stat()
        if s.st_size!=stamp['bytes'] or s.st_mtime_ns!=stamp['mtime_ns'] or photo.digest(p)!=stamp['sha256']:changed.append(name)
    assert not changed,changed
    final=Path(evidence.delivery['final']);m,material,gt=ref.load(final);old,_,_=ref.load(evidence.paths['final'])
    assert len(m['exposures'])==176 and m['dwell_seconds']==5 and m['policy']==old['policy']
    for name in ('material.json','reference.json','source_annotations.xml'):
        assert photo.digest(final/name)==photo.digest(Path(evidence.paths['final'])/name)
    for e,o in zip(m['exposures'],old['exposures']):
        for key in ('position','exposure_id','image_id','photo_sha256','source_role','actual_image_rect','photo_placement'):assert e[key]==o[key]
        assert photo.digest(material.verify(e['image_id']))==e['photo_sha256']
    pf=ref.reference_preflight(final);assert pf['complete_gt_plates']==209 and pf['counts']['GT_COMPLETE']==176
    tech=Path(evidence.delivery['technical']);tm,_,tg=ref.load(tech)
    assert len(tm['exposures'])==5 and tm['mode']=='TECHNICAL_NON_FINAL' and tg is None and not (tech/'reference.json').exists()
    assert not (tech/'runs').exists() and not (final/'runs').exists()
    for p in (final,tech):
        doc=photo.read_json(p/'manifest.json');index=photo.read_json(p/'qr_plan_index.json')
        assert index==mk.f3_plan_index(doc,photo.digest(p/'manifest.json')) and index['run_id'] is None
    # Policy and core selectors remain exactly the pre-change implementation.
    for filename,names in [('final180_markers.py',('geometry','input_exclusion','transform_gt','photo_placement','qr_f2','qualify_f2','f2_binding','layout_f2')),
            ('final180_mobile.py',('select_first','score_frame','_order'))]:
        before=ast.parse((evidence.out/'before'/filename).read_text('utf-8'))
        after=ast.parse((Path(mk.__file__).parent/filename).read_text('utf-8'))
        functions=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)}
        a,b=functions(before),functions(after)
        assert all(a[name]==b[name] for name in names)
    save(evidence.out,'D4.json',dict(status='PASS',protected_files=len(hashes)-len(allowed),changed_protected_files=changed,
        final_photographs=176,gt_plates=209,reference_sha256=photo.digest(final/'reference.json'),
        technical_photographs=5,technical_reference='NONE',production_runs_created=0,scoring_unchanged=True,
        new_preparations=evidence.delivery))
