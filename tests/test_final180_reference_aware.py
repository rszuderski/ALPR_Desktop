"""V3 presentation / reference / source evidence remain independent dimensions."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib,io,json,os,subprocess,sys,time
import pytest
from PIL import Image
from auto_annotation_tool.ranking import final180_reference as ref,final180_mobile as mobile,final180_markers as mk,final180_timeline as timeline,photo_reference as photo
from auto_annotation_tool.gui import final180_tv as tv
from test_final180_desktop import frozen,material,plate,candidate,scoring_state


def no_gt(tmp_path,n=3):
    source=tmp_path/'source';source.mkdir(parents=True)
    for i in range(n):Image.new('RGB',(100,80),(20+i,40,90)).save(source/f'{i}.png')
    return ref.freeze_photographs(source,tmp_path/'plans',screen=(1280,720),technical=True)


def native_f2_archive(tmp_path,manifest,run):
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    from test_final180_timeline import FIXTURES,mutate_fixture
    image,_=mk.render(None,manifest['layout'],mk.payload(manifest,run.manifest_sha,run.state['run_id'],'e0001','P'))
    s=MobileReviewSession.open(FIXTURES/'r0-positive.alprsession',sidecar_path=tmp_path/'fixture.absent.json')
    record=s.tv_bbox_evidence['rows'][0]['source_image'];old=record['sha256'];buf=io.BytesIO()
    image.resize((record['width_px'],record['height_px']),Image.Resampling.LANCZOS).save(buf,format='JPEG',quality=96)
    raw=buf.getvalue();sha=hashlib.sha256(raw).hexdigest()
    def replace(data):
        for k in data:
            if hashlib.sha256(data[k]).hexdigest()==old:data[k]=raw
            elif k.endswith(('.json','.jsonl')):data[k]=data[k].replace(old.encode(),sha.encode())
    return mutate_fixture(tmp_path/'native_f2.alprsession',replace)


def test_no_gt_is_no_reference_file_and_presentation_survives(tmp_path):
    d=no_gt(tmp_path);m,mat,gt=ref.load(d);p=ref.reference_preflight(d)
    assert gt is None and m['reference_mode']=='none' and m['reference_sha256'] is None
    assert not(d/'reference.json').exists() and not(d/'source_annotations.xml').exists()
    assert p['counts']['GT_MISSING']==3 and all(r['plate_count'] is None for r in p['coverage'])
    assert ref.display_ready(m)
    frame,identity=tv.calibration_frame(m,photo.digest(d/'manifest.json'));assert identity['phase']=='T'
    assert not(d/'runs').exists()
    run=tv.PresentationRun(d,m);run.present(0,{},lambda:None);run.stop(clear=lambda:None)
    reopened,events,_=timeline.read_run(run.directory/'run.json',m,run.manifest_sha)
    assert events and reopened['run_id']==run.state['run_id']
    later=ref.replan(d,7);assert ref.load(later)[2] is None and not(later/'reference.json').exists()


def test_native_partial_negative_missing_and_new_reference_version(tmp_path):
    d=no_gt(tmp_path);m,mat,_=ref.load(d);before={p:photo.digest(p) for p in d.rglob('*') if p.is_file()}
    external=tmp_path/'operator_reference.json';native=ref.working_reference(d,external);keys=mat.order
    native.empty(keys[0]);oid=native.add(keys[1],[[10,10],[70,10],[70,40],[10,40]])
    p=ref.reference_preflight(d,reference_path=external)
    assert p['mode']=='partial' and p['counts']==dict(GT_COMPLETE=1,GT_INCOMPLETE=1,GT_MISSING=1,GT_CONFLICT=0)
    assert p['coverage'][0]['plate_count']==0 and p['coverage'][0]['no_plates']
    assert p['coverage'][2]['plate_count'] is None
    native.edit(keys[1],oid,text='A123');p2=ref.reference_preflight(d,reference_path=external)
    assert p2['counts']['GT_COMPLETE']==2 and p2['reference_sha256']!=p['reference_sha256']
    assert all(photo.digest(path)==sha for path,sha in before.items())
    assert ref.reference_preflight(d)['mode']=='none'  # no implicit reference search


@pytest.mark.parametrize('fault',['sha','photo_id','duplicate','geometry'])
def test_bad_reference_is_conflict_not_negative(tmp_path,fault):
    d=no_gt(tmp_path);_,mat,_=ref.load(d);p=tmp_path/'reference.json';native=ref.working_reference(d,p)
    oid=native.add(mat.order[0],[[10,10],[70,10],[70,40],[10,40]]);native.edit(mat.order[0],oid,text='A123')
    data=photo.read_json(p);key=mat.order[0];review=data['reviews'][key];state=review['versions'][review['timeline'][review['cursor']]]['state']
    if fault=='sha':data['images'][key]['source_file_sha256']='0'*64
    elif fault=='photo_id':data['images'][key]['image_id']='foreign'
    elif fault=='duplicate':state['objects'].append(deepcopy(state['objects'][0]))
    else:state['objects'][0]['quad']=[[0,0]]*4
    p.write_text(json.dumps(data),'utf-8');gt=ref.reference_preflight(d,reference_path=p)
    assert gt['mode']=='invalid' and gt['counts']['GT_CONFLICT']==3
    assert all(x['plate_count'] is None for x in gt['coverage'])
    assert ref.display_ready(ref.load(d)[0])


def test_legacy_reference_accepted_without_new_approvals_and_corruption_detachable(tmp_path):
    old=frozen(tmp_path/'old',unreadable=(0,));baseline=photo.digest(old/'reference.json')
    upgraded=ref.relayout(old,(1280,720),reference_aware=True);p=ref.reference_preflight(upgraded)
    assert p['mode']=='complete' and p['counts']['GT_COMPLETE']==3 and p['full_text']==1 and p['unreadable']==1 and p['negative_images']==1
    assert photo.digest(upgraded/'reference.json')==baseline
    (upgraded/'reference.json').write_text('{}','utf-8')
    assert ref.reference_preflight(upgraded)['mode']=='invalid'
    detached=ref.relayout(upgraded,(1280,720),reference_aware=True,ignore_reference=True)
    assert ref.reference_preflight(detached)['mode']=='none' and not(detached/'reference.json').exists()
    assert (upgraded/'reference.json').read_text()=='{}' and photo.digest(old/'reference.json')==baseline


def test_raw_xml_does_not_invent_scene_completeness(tmp_path):
    d=ref.freeze(material(tmp_path/'raw'),tmp_path/'plans',layout_version=mk.F2_LAYOUT,reference_aware=True)
    p=ref.reference_preflight(d)
    assert p['mode']=='partial' and p['counts']['GT_INCOMPLETE']==2 and p['counts']['GT_COMPLETE']==1


def test_empty_xml_does_not_create_fake_reference_and_sources_are_combined(tmp_path):
    a=material(tmp_path/'input',n=2,negative=0,empty=(0,1))
    p=ref.audit_reference_preflight(a);assert p['mode']=='none'
    d=ref.freeze(a,tmp_path/'plans',layout_version=mk.F2_LAYOUT,reference_aware=True)
    assert ref.load(d)[2] is None and not(d/'reference.json').exists() and not(d/'source_annotations.xml').exists()
    extra=tmp_path/'extra';extra.mkdir();Image.new('RGB',(100,80),'red').save(extra/'same.png')
    d2=ref.freeze_photographs(a['positive_dir'],tmp_path/'plans',screen=(1280,720),additional_source=extra)
    assert len(ref.load(d2)[0]['exposures'])==3 and ref.reference_preflight(d2)['counts']['GT_MISSING']==3


def test_orphan_xml_and_conflicting_duplicate_bytes_are_not_complete(tmp_path):
    import xml.etree.ElementTree as ET,shutil
    a=material(tmp_path/'xml',n=2,negative=0);tree=ET.parse(a['xml_path'])
    ET.SubElement(tree.getroot(),'image',id='99',name='orphan.png',width='100',height='80');tree.write(a['xml_path'])
    audited=ref.audit(a['xml_path'],a['positive_dir'],expected_negative=0)
    assert ref.audit_reference_preflight(audited)['mode']=='invalid'
    source=tmp_path/'duplicate';source.mkdir();Image.new('RGB',(100,80),'gray').save(source/'a.png');shutil.copy2(source/'a.png',source/'b.png')
    d=ref.freeze_photographs(source,tmp_path/'dup_plans',screen=(1280,720));_,mat,_=ref.load(d);g=ref.working_reference(d,tmp_path/'manual.json')
    g.empty(mat.order[0]);oid=g.add(mat.order[1],[[10,10],[70,10],[70,40],[10,40]]);g.edit(mat.order[1],oid,text='TEST')
    assert 'SAME_BYTES_CONFLICTING_REFERENCE' in ref.reference_preflight(d,reference_path=g.path)['error']


def test_no_gt_native_import_still_assigns_frame_and_keeps_raw_mz(tmp_path):
    d=no_gt(tmp_path,1);m,_,_=ref.load(d);run=tv.PresentationRun(d,m);archive=native_f2_archive(tmp_path,m,run)
    run.present(0,{},lambda:None);run.stop(clear=lambda:None)
    before={p:photo.digest(p) for p in d.rglob('*') if p.is_file()}
    result=mobile.analyze_archive(d,archive,run.directory/'run.json')
    assert result['metrics_status']=='NOT_AVAILABLE_NO_GT'
    assert result['frames'][0]['identity_status']=='PASS' and result['frames'][0]['exposure_id']=='e0001'
    assert result['raw_mz'][0]['raw_prediction']=='A123' and result['raw_mz'][0]['source_bound']
    assert result['rows'][0]['plates']==[] and result['rows'][0]['tp'] is None and result['rows'][0]['gt_plate_count'] is None
    assert result['summary']['system_full_read']['rate'] is None and result['summary']['conditional_mz_cer']['value'] is None
    exported=mobile.export_report(result,tmp_path/'results')
    assert photo.read_json(exported/'evaluation.json')==json.loads(json.dumps(result)) and (exported/'raw_mz.csv').exists()
    assert all(photo.digest(p)==sha for p,sha in before.items())


def scored_report(tmp_path,*,missing=False):
    import xml.etree.ElementTree as ET,zipfile
    a=material(tmp_path/'source',n=2,negative=1);xml=Path(a['xml_path']);tree=ET.parse(xml)
    for node in tree.findall('.//attribute[@name="ground_truth_text"]'):node.text='TEST'
    tree.write(xml,encoding='utf-8',xml_declaration=True)
    a=ref.audit(xml,a['positive_dir'],a['negative_dir'],negative_declared=True,expected_negative=1)
    d=ref.relayout(ref.freeze(a,tmp_path/'plans',technical=True),(1280,720),reference_aware=True);m,mat,native_gt=ref.load(d);preflight=ref.reference_preflight(d)
    rows=[];frames=[]
    raw_mt=[];raw_mz=[];archive=tmp_path/'synthetic_view_evidence.zip'
    # Complete reference / synthetic GT geometry explicitly exercises existing scoring.
    with zipfile.ZipFile(archive,'w') as z:
        for i,e in enumerate(m['exposures']):
            state=native_gt.state(e['image_id']);objects=state['objects'];finals=[candidate('a',[20,20,80,45],'TESX')] if i==0 else []
            _,_,frame=scoring_state(objects,finals);frame.update(exposure_id=e['exposure_id'],frame_key=str(i),mt_invocation_ids=[f'call-{i}'])
            image=Image.new('RGB',(100,100),'black');image.paste(mat.image(e['image_id']),(0,0));buf=io.BytesIO();image.save(buf,format='PNG');raw=buf.getvalue();entry=f'frame{i}.png';z.writestr(entry,raw)
            frame['source_image']=dict(status='WRITTEN',entry=entry,sha256=hashlib.sha256(raw).hexdigest())
            row=mobile.score_frame(e,state,frame);row['presentation']=dict(status='COMPLETED');rows.append(row);frames.append(frame)
            raw_mt.append(dict(mt_invocation_id=f'call-{i}',frame_key=str(i),exposure_id=e['exposure_id']))
            if i==0:raw_mz.append(dict(mz_invocation_id='mz-0',mt_invocation_id='call-0',frame_key=str(i),exposure_id=e['exposure_id'],raw_prediction='TESX',source_bound=True))
    if missing:preflight['mode']='partial';preflight['coverage'][1].update(status='GT_MISSING',plate_count=None,reason='NO_REFERENCE');preflight['counts'].update(GT_COMPLETE=2,GT_MISSING=1)
    report=dict(schema='alpr.final.evaluation.v1',session_id='SYNTHETIC',run_id='b'*32,source_archive_path=str(archive),source_archive_sha256=photo.digest(archive),
        manifest_sha256=photo.digest(d/'manifest.json'),rows=rows,frames=frames,presentation_timeline=[dict(status='COMPLETED')]*3,
        summary=mobile.summarize(rows,frames,m),go='NO_GO',go_reasons=['SYNTHETIC_NOT_A12'],raw_mt=raw_mt,raw_mz=raw_mz)
    return d,mobile.reference_report(report,preflight,d)


def test_cer_exact_subset_and_no_automatic_false_positives(tmp_path):
    _,report=scored_report(tmp_path)
    assert report['rows'][0]['tp']==1 and report['rows'][0]['wrong_reads']==1
    assert report['summary']['conditional_mz_cer']['value']==.25
    assert report['rows'][1]['fn']==1 and report['rows'][2]['negative_correct_silence'] is True
    _,partial=scored_report(tmp_path/'partial',missing=True)
    assert partial['rows'][1]['fn'] is None and partial['rows'][1]['plates']==[]
    assert partial['summary']['detection']['fn']==0  # excluded GT must not survive in aggregate FN
    assert partial['summary']['system_full_read']['rate'] is None and partial['metrics_status']=='EVALUATED_SUBSET'


def test_authentic_profile_not_scenario_names():
    b=NS(validation=NS(ok=True),collection_session=dict(scenario_id='FINAL176',execution=dict(capture_profile='ordinary')),f6_evidence=dict(available=True))
    assert not tv.authentic_final_bundle(b)
    b.collection_session['execution']['capture_profile']='f6_final180_v1';assert tv.authentic_final_bundle(b)
    b.validation.ok=False;assert not tv.authentic_final_bundle(b)


def test_actual_frozen_176_reference_readonly():
    path=os.environ.get('ALPR_F6_PREPARED')
    if not path:pytest.skip('Private frozen FINAL reference not supplied')
    d=Path(path);before=photo.digest(d/'reference.json');p=ref.reference_preflight(d)
    assert p['mode']=='complete' and p['counts']['GT_COMPLETE']==176
    assert (p['complete_gt_plates'],p['full_text'],p['unreadable'],p['negative_images'])==(209,203,6,10)
    assert photo.digest(d/'reference.json')==before


def test_new_reference_new_analysis_keeps_original_run_and_exports(tmp_path):
    d=no_gt(tmp_path,1);m,mat,_=ref.load(d);run=tv.PresentationRun(d,m);archive=native_f2_archive(tmp_path,m,run)
    run.present(0,{},lambda:None);run.stop(clear=lambda:None)
    baseline={p:photo.digest(p) for p in d.rglob('*') if p.is_file()};first=mobile.analyze_import(archive,run.directory/'run.json',output=tmp_path/'analyses')
    first_sha=photo.digest(first/'evaluation.json');gt=ref.working_reference(d,tmp_path/'manual.json');gt.empty(mat.order[0])
    second=mobile.analyze_import(archive,run.directory/'run.json',output=tmp_path/'analyses',reference_path=gt.path)
    assert first!=second and photo.digest(first/'evaluation.json')==first_sha
    assert photo.read_json(first/'evaluation.json')['reference_preflight']['mode']=='none'
    assert photo.read_json(second/'evaluation.json')['reference_preflight']['mode']=='complete'
    assert all(photo.digest(p)==sha for p,sha in baseline.items())
    with pytest.raises(ValueError,match='GT_WORK_COPY'):ref.working_reference(d,d/'reference.json')


def test_native_tk_browser_import_report_no_gt_and_optional_existing_editor(tmp_path,monkeypatch):
    if os.environ.get('F6_V3_TK_ISOLATED')!='1':
        child=subprocess.run([sys.executable,'-m','pytest',str(Path(__file__).resolve())+'::test_native_tk_browser_import_report_no_gt_and_optional_existing_editor','-q'],
            env=dict(os.environ,F6_V3_TK_ISOLATED='1'),capture_output=True,text=True,timeout=120)
        assert child.returncode==0,child.stdout+child.stderr;return
    import tkinter as tk
    from auto_annotation_tool import config
    from auto_annotation_tool.session import SessionManager
    from auto_annotation_tool.gui import z4_mobile_report_browser as browser_module,z4_mobile_sample_review as manual_module
    from auto_annotation_tool.ranking.mobile_package_experiments import MobilePackageExperimentStore,read_mobile_report_bundle
    d=no_gt(tmp_path,1);m,_,_=ref.load(d);run=tv.PresentationRun(d,m);archive=native_f2_archive(tmp_path,m,run)
    run.present(0,{},lambda:None);run.stop(clear=lambda:None);before={p:photo.digest(p) for p in d.rglob('*') if p.is_file()}
    preferences=SessionManager(tmp_path/'preferences.json');preferences.set('final_presentation','prepared_path',str(d))
    monkeypatch.setattr(config,'SESSION',preferences);monkeypatch.setattr(browser_module,'SESSION',preferences)
    monkeypatch.setattr(browser_module,'MobilePackageExperimentStore',lambda:MobilePackageExperimentStore(tmp_path/'store'))
    # All real Tk controls and callbacks execute, but no window reaches the TV.
    monkeypatch.setattr(tk.Toplevel,'deiconify',lambda self:None);monkeypatch.setattr(tk.Toplevel,'lift',lambda self:None)
    monkeypatch.setattr(tk.Toplevel,'focus_force',lambda self:None)
    monkeypatch.setattr(tv.messagebox,'showerror',lambda *a,**k:pytest.fail(str(a)))
    monkeypatch.setattr(tv.filedialog,'askopenfilename',lambda **kwargs:str(run.directory/'run.json'))
    root=tk.Tk();root.withdraw();errors=[];root.report_callback_exception=lambda *args:errors.append(args)
    owner=NS(app=NS(root=root,palette={}));browser=browser_module.MobileReportBrowser(owner,parent=root);browser.window.withdraw()
    app=None
    try:
        # The actual report-browser entry invokes repository code, not plan/*.py.
        browser.open_final_presentation();app=browser._final_presentation;app.window.withdraw()
        assert app.prepared==d and 'GT: brak' in app.reference_info.get()
        assert 'GT nieprzypisane' in app.details.get('1.0','end') and not(d/'reference.json').exists()
        app._make_viewer=lambda monitor:pytest.fail('Import must not create a presentation window')
        bundle=read_mobile_report_bundle(archive);browser.current_report=bundle.report;browser.current_bundle=bundle;browser._sync_final_action()
        assert 'Wyniki pokazu' in browser.science_panel.review_action_button.cget('text')
        # Native CTA, common real analyzer, actual producer fixture + F2 pixels.
        browser.science_panel.review_action_button.invoke()
        deadline=time.monotonic()+45
        while not app.result_windows and time.monotonic()<deadline:root.update();time.sleep(.02)
        assert app.result_windows,app.status.get()
        panel=app.result_windows[-1];panel.window.withdraw();root.update()
        assert panel.report['metrics_status']=='NOT_AVAILABLE_NO_GT' and 'SKUTECZNOŚĆ: N/A' in panel.summary_text
        assert len(panel.tree.get_children())==1 and 'GT niedostępne' in panel.tree.item('e0001','values')[0]
        assert panel.report['frames'][0]['exposure_id']=='e0001';panel.tree.selection_set('e0001');panel.tree.event_generate('<<TreeviewSelect>>');root.update()
        assert 'diagnostyczny' in panel.selection_info.get().lower() and 'A123' in panel.details.get('1.0','end')
        assert app.viewer is None and all(photo.digest(p)==sha for p,sha in before.items())
        # The existing editor opens a new external native reference, not frozen GT.
        editor=app.edit_analysis_gt(panel.path,archive,run.directory/'run.json');editor.window.withdraw();root.update()
        assert not editor.session.path.is_relative_to(d) and editor.session.material.readonly is False
        editor.window.destroy();root.update()
        # F5A/ordinary reports retain their original manual route.
        calls=[];monkeypatch.setattr(manual_module,'open_mobile_sample_review',lambda *a,**k:calls.append(a))
        browser.current_report=bundle.report  # private empty-store refresh may clear selection during the async import
        browser.current_bundle=NS(validation=NS(ok=True),collection_session=dict(execution=dict(capture_profile='ordinary')),f6_evidence={})
        browser._sync_final_action();browser.open_sample_review();assert len(calls)==1
        assert not errors,errors
    finally:
        if app is not None:app._close()
        browser._closing=True
        for token in root.tk.call('after','info'):
            try:root.after_cancel(token)
            except tk.TclError:pass
        browser.window.destroy();root.destroy()


def test_native_tk_scored_result_selection_and_export_values(tmp_path,monkeypatch):
    if os.environ.get('F6_V3_RESULT_TK')!='1':
        child=subprocess.run([sys.executable,'-m','pytest',str(Path(__file__).resolve())+'::test_native_tk_scored_result_selection_and_export_values','-q'],env=dict(os.environ,F6_V3_RESULT_TK='1'),capture_output=True,text=True,timeout=60)
        assert child.returncode==0,child.stdout+child.stderr;return
    import tkinter as tk
    d,r=scored_report(tmp_path);path=mobile.export_report(r,tmp_path/'results');root=tk.Tk();root.withdraw()
    panel=tv.FinalResultsWindow(root,path);panel.window.withdraw()
    try:
        assert len(panel.tree.get_children())==3
        panel.tree.selection_set('e0001');panel.select();root.update()
        assert 'CER: 0.25' in panel.details.get('1.0','end')
        panel.filter.set('Pominięte GT');panel.populate();assert panel.tree.get_children()==('e0002',)
        assert 'CER: N/A' in panel.details.get('1.0','end') and 'None' not in panel.details.get('1.0','end')
        panel.filter.set('Negatywy');panel.populate();assert panel.tree.get_children()==('e0003',)
        saved=photo.read_json(path/'evaluation.json');assert saved['summary']['detection']==panel.report['summary']['detection']
        assert saved['summary']['conditional_mz_cer']['value']==.25
    finally:panel.window.destroy();root.destroy()
