"""P0 r147/r149: source-preserving export, approval and SHA variant identity."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,hashlib,json,os,shutil,time
os.environ.setdefault('MPLBACKEND','Agg')
import pytest
from windows_dpi import enable_windows_dpi_awareness
DPI_MODE=enable_windows_dpi_awareness()
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_ppm_seed_active_click import settled
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_mt_object_evaluation import evaluate_session,POLICY_SHA256
from auto_annotation_tool.ranking.mobile_mt_object_export import replay
from auto_annotation_tool.ranking.mobile_verification_snapshot import has_valid_ready_snapshot
from auto_annotation_tool.gui import z4_mobile_sample_review as module

SID='exp-1792699088870-cc33764a'
ARCHIVE_SHA='b3e3e7f6b3a08e6865babdd7fd06bfbcaa0839aa5131ded8cba1c8e7a2ee9d47'
OLD_SHA='a04aa171e4535ac1204b719f27755d55a1b1601cf2616d792eaebcc3b64b1f34'

def stamp(p):
    p=Path(p)
    if not p.exists():return None
    s=p.stat();return s.st_size,s.st_mtime_ns,hashlib.sha256(p.read_bytes()).hexdigest()

def wait(root,h,ed):
    native.pump(root,lambda:not h._saving and not ed.in_refresh and not ed.refresh_after,seconds=45)
    ready(root,h,ed)

@pytest.fixture(scope='module')
def baseline():
    p=os.environ.get('F4_REVISION_BASELINE')
    if not p or not Path(p).is_file():pytest.skip('Pinned r147/r149 real-source baseline required')
    return json.loads(Path(p).read_text(encoding='utf-8'))

@pytest.fixture
def actual(tmp_path,baseline,request):
    revision=getattr(request,'param',149)
    r=next(r for r in baseline['sessions'] if r['session_id']==SID)
    src=tmp_path/'data';src.mkdir()
    shutil.copyfile(r['private_source'],src/'source.alprsession')
    old=Path(__file__).parents[1]/'output/F4_EVAL2_20261101_001/private'/SID/'review.json'
    side=old if revision==147 else Path(r['private_sidecar'])
    shutil.copyfile(side,src/'review.json')
    s=MobileReviewSession.open(src/'source.alprsession',sidecar_path=src/'review.json')
    assert s.review.review_revision==revision
    return s

@pytest.fixture
def gui(tmp_path,monkeypatch,actual,request):
    import tkinter as tk
    from auto_annotation_tool.gui import mobile_canvas_editor as editor
    theme,width,height=getattr(request,'param',('light',1649,896))
    old=tk.Tk
    def make_root(*a,**k):
        root=old(*a,**k);root.tk.call('tk','scaling',120/72);return root
    monkeypatch.setattr(native.tk,'Tk',make_root)
    palette=editor.palette
    monkeypatch.setattr(editor,'palette',lambda w,b:palette(w,'#FFFFFF' if theme=='light' else '#17232D'))
    def namespace(**kwargs):
        if 'palette' in kwargs:
            kwargs['palette']=dict(panel='#F5F5F2',fg='#213547',muted='#5E7283',accent='#14628B') if theme=='light' else dict(panel='#17232D',fg='#EBF2F6',muted='#B5C4CE',accent='#77CBF0')
        return NS(**kwargs)
    monkeypatch.setattr(native,'SimpleNamespace',namespace)
    monkeypatch.setattr(native,'source_session',lambda _:actual)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    h.window.state('normal');h.window.geometry(f'{width}x{height}+10+10');h.window.deiconify();root.update()
    calls=[];store=NS(root_dir=tmp_path/'rank',apply_human_review=lambda s:calls.append('ranking') or False)
    monkeypatch.setattr(module,'MobilePackageExperimentStore',lambda *a,**k:store)
    h._test_ranking_calls=calls;wait(root,h,ed)
    try:yield root,h,ed,s
    finally:
        if ed.entry is not None:ed.escape()
        try:next(gen)
        except StopIteration:pass

def dialogs(monkeypatch,tmp_path,*,choose=True,confirm=True):
    out=tmp_path/'03_wyniki';out.mkdir(exist_ok=True);calls=[]
    monkeypatch.setattr(module.filedialog,'askdirectory',lambda **k:calls.append('directory') or (str(out) if choose else ''))
    monkeypatch.setattr(module.messagebox,'askyesno',lambda *a,**k:calls.append('confirm') or confirm)
    return out,calls

def do_export(gui,monkeypatch,tmp_path):
    root,h,ed,s=gui;out,calls=dialogs(monkeypatch,tmp_path)
    h.export();wait(root,h,ed)
    manifests=list(out.glob(SID+'/review_*/export_manifest.json'))
    assert manifests,'Export did not publish: '+h.status_var.get()
    return out,manifests[-1].parent,calls

def test_R01_R02_real_source_and_exact_r147_snapshot(actual):
    assert actual.session_id==SID and hashlib.sha256(actual.path.read_bytes()).hexdigest()==ARCHIVE_SHA
    p=Path(__file__).parents[1]/'output/F4_EVAL2_20261101_001/private'/SID/'review.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==OLD_SHA

def test_R03_R04_R05_recorded_actions_preserve_private_text(actual):
    events=[e for v in actual.review.invocation_annotations.values() for s in v.values() if isinstance(s,dict) for e in s.get('events',[])]
    recent=[(e['action'],e['review_revision_before']) for e in events if e.get('review_revision_before',0)>=147]
    assert recent==[('operator_unconfirm_associated_frames',147),('operator_confirm_associated_frames',148)]
    assert actual.review.review_status=='IN_PROGRESS'

def test_R07_R09_approve_copy_and_repeated_complete_are_idempotent(actual):
    from auto_annotation_tool.ranking.mobile_verification_policy import verification_progress
    before=copy.deepcopy(stacks.snapshot(actual,next(iter(actual.mt_invocations))))
    assert verification_progress(actual)['status']=='READY_TO_APPROVE'
    actual.complete();assert actual.review.review_revision==150 and actual.review.review_status=='COMPLETED'
    assert has_valid_ready_snapshot(actual)
    s=stamp(actual.sidecar_path);actual.complete();assert stamp(actual.sidecar_path)==s
    assert stacks.snapshot(actual,next(iter(actual.mt_invocations)))==before

def test_R08_incomplete_copy_is_not_auto_approved(actual):
    gid=next(iter(actual.mt_invocations));oid=core.scene(actual,gid)['objects'][0]['object_id']
    partial.toggle_stack(actual,gid,oid=oid)
    frozen=stamp(actual.sidecar_path)
    with pytest.raises(ValueError):actual.complete()
    assert stamp(actual.sidecar_path)==frozen and actual.review.review_status=='IN_PROGRESS'

@pytest.mark.parametrize('actual',[147,149],indirect=True)
def test_R10_R17_GUI_export_is_read_only(gui,monkeypatch,tmp_path):
    root,h,ed,s=gui;before=stamp(s.sidecar_path);state=copy.deepcopy(s.review.to_dict())
    out,target,calls=do_export(gui,monkeypatch,tmp_path)
    assert stamp(s.sidecar_path)==before and s.review.to_dict()==state
    assert not h._test_ranking_calls,'Export must not write the report index'
    data=replay(next(target.glob('mt_object_evaluation_v2_*')))
    assert data['review_revision']==state['review_revision'] and data['matching_policy_sha256']==POLICY_SHA256
    manifest=json.loads((target/'export_manifest.json').read_text(encoding='utf-8'))
    assert manifest['review_revision']==state['review_revision'] and manifest['review_status']==state['review_status']
    assert manifest['f4_variant']['label']=='DAY_INT8 · r01'
    assert manifest['f4_variant']['provenance']=='F4_REGISTRY_SHA'
    assert manifest['primary_evaluation']['methodology_version']=='F4-EVAL-2.0'
    assert not list(target.glob('mt_detection_evaluation_v1_*'))

def test_R11_two_GUI_exports_keep_old_version(gui,monkeypatch,tmp_path):
    root,h,ed,s=gui;before=stamp(s.sidecar_path)
    out,a,_=do_export(gui,monkeypatch,tmp_path);frozen={str(p.relative_to(a)):p.read_bytes() for p in a.rglob('*') if p.is_file()}
    _,b,_=do_export(gui,monkeypatch,tmp_path)
    assert a!=b and stamp(s.sidecar_path)==before
    assert frozen=={str(p.relative_to(a)):p.read_bytes() for p in a.rglob('*') if p.is_file()}

def test_R12_started_GTZ_cannot_be_saved_by_export(gui,monkeypatch,tmp_path):
    root,h,ed,s=gui;oid=core.scene(s,h.current_invocation)['objects'][0]['object_id']
    ed.start_entry(oid);ready(root,h,ed);ed.entry_var.set('PRIVATE_TEST');ed.entry_typed()
    frozen=stamp(s.sidecar_path);out,calls=dialogs(monkeypatch,tmp_path)
    h.export();native.pump(root)
    deadline=time.monotonic()+.9
    while time.monotonic()<deadline:root.update();time.sleep(.01)
    assert stamp(s.sidecar_path)==frozen and not calls
    assert ed.entry is not None and ed.entry_var.get()=='PRIVATE_TEST' and not list(out.iterdir())

def test_R13_started_exclusion_is_kept_without_export(gui,monkeypatch,tmp_path):
    from auto_annotation_tool.gui.mobile_canvas_clean_view import show_exclusion
    root,h,ed,s=gui;show_exclusion(ed);ed.clean_reason.set('PRIVATE_REASON');frozen=stamp(s.sidecar_path)
    out,calls=dialogs(monkeypatch,tmp_path);h.export();native.pump(root)
    assert stamp(s.sidecar_path)==frozen and not calls and not list(out.iterdir())
    assert ed.clean_reason.get()=='PRIVATE_REASON' and ed.clean_exclusion.winfo_manager()

@pytest.mark.parametrize('choose,confirm',[(False,True),(True,False)])
def test_R15_cancel_does_not_create_or_save(gui,monkeypatch,tmp_path,choose,confirm):
    root,h,ed,s=gui;before=stamp(s.sidecar_path)
    out,_=dialogs(monkeypatch,tmp_path,choose=choose,confirm=confirm);h.export();native.pump(root)
    assert not list(out.iterdir()) and stamp(s.sidecar_path)==before and not h._test_ranking_calls

def test_reviewer_same_value_is_noop(actual):
    before=stamp(actual.sidecar_path);actual.set_reviewer(actual.review.reviewer_id)
    assert stamp(actual.sidecar_path)==before

def test_R28_numbers_and_GT_geometry_unchanged(actual):
    r=evaluate_session(actual);o=r['ocr_summary']
    assert (r['tp'],r['fp'],r['fn'])==(16,4,2) and r['geometry_summary']['n_geometry']==16
    assert (o['n_exact'],o['n_reads'],o['n_text_errors'],o['n_gt_characters'])==(10,16,11,111)


@pytest.mark.parametrize('route',['mouse','focus'])
def test_R12_native_export_focus_and_press_keep_GTZ_draft(gui,monkeypatch,tmp_path,route):
    root,h,ed,s=gui;oid=core.scene(s,h.current_invocation)['objects'][0]['object_id']
    ed.start_entry(oid);ready(root,h,ed);ed.entry.focus_force();root.update()
    ed.entry_var.set('PRIVATE_TEST');ed.entry_typed();before=stamp(s.sidecar_path)
    out,calls=dialogs(monkeypatch,tmp_path)
    if route=='mouse':
        h.export_button.event_generate('<ButtonPress-1>',x=5,y=5)
        h.export_button.event_generate('<ButtonRelease-1>',x=5,y=5)
    else:
        h.export_button.focus_force();root.update();h.export_button.invoke()
    deadline=time.monotonic()+.9
    while time.monotonic()<deadline:root.update();time.sleep(.01)
    assert stamp(s.sidecar_path)==before and not calls and not list(out.iterdir())
    assert ed.entry is not None and ed.entry_var.get()=='PRIVATE_TEST'
    assert 'Zakończ lub anuluj edycję' in h.status_var.get()


def test_R14_explicit_GTZ_edit_has_own_revision_before_read_only_export(gui,monkeypatch,tmp_path):
    root,h,ed,s=gui;revision=s.review.review_revision
    before_events=copy.deepcopy(core.scene(s,h.current_invocation)['events'])
    oid=core.scene(s,h.current_invocation)['objects'][0]['object_id']
    ed.start_entry(oid);ready(root,h,ed);ed.entry_var.set('PRIVATE_TEST');ed.entry_enter();wait(root,h,ed)
    assert s.review.review_revision>revision and s.review.review_status=='IN_PROGRESS'
    assert core.scene(s,h.current_invocation)['events']!=before_events
    saved=stamp(s.sidecar_path);events=copy.deepcopy(s.review.invocation_annotations)
    do_export(gui,monkeypatch,tmp_path)
    assert stamp(s.sidecar_path)==saved and s.review.invocation_annotations==events


@pytest.mark.parametrize('draft',['draw','drag'])
def test_export_preserves_geometry_draft(gui,monkeypatch,tmp_path,draft):
    root,h,ed,s=gui;before=stamp(s.sidecar_path)
    if draft=='draw':ed.mode='draw';ed.points=[[.1,.1],[.3,.1]]
    else:ed.drag={'private_test':True}
    out,calls=dialogs(monkeypatch,tmp_path);h.export();root.update()
    assert stamp(s.sidecar_path)==before and not calls and not list(out.iterdir())
    assert ed.points if draft=='draw' else ed.drag
    ed.drag=None;ed.points=[];ed.mode='select'


def test_R16_external_change_during_final_manifest_blocks_publish(gui,monkeypatch,tmp_path):
    from auto_annotation_tool.gui import mobile_review_export_layout as layout
    root,h,ed,s=gui;out,calls=dialogs(monkeypatch,tmp_path)
    original=layout._plain_manifest;external={}
    def change_after_hashing(*a,**k):
        result=original(*a,**k);doc=json.loads(s.sidecar_path.read_text(encoding='utf-8'))
        doc['review_revision']+=1
        s.sidecar_path.write_text(json.dumps(doc,ensure_ascii=False),encoding='utf-8')
        external['state']=stamp(s.sidecar_path);return result
    monkeypatch.setattr(layout,'_plain_manifest',change_after_hashing)
    h.export();wait(root,h,ed)
    assert external and stamp(s.sidecar_path)==external['state']
    assert not list(out.rglob('export_manifest.json'))
    assert list(out.rglob('EKSPORT_NIEKOMPLETNY.txt')) and not h._test_ranking_calls
    assert 'REVIEW_REVISION_CHANGED' in h.status_var.get()


def test_R07_R17_private_approval_export_reload_preserves_annotations(gui,monkeypatch,tmp_path):
    root,h,ed,s=gui
    annotations={g:copy.deepcopy(stacks.snapshot(s,g)) for g in s.mt_invocations}
    s.complete();assert s.review.review_revision==150 and has_valid_ready_snapshot(s)
    before=stamp(s.sidecar_path);s.complete();assert stamp(s.sidecar_path)==before
    out,target,calls=do_export(gui,monkeypatch,tmp_path)
    fresh=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert fresh.review.review_status=='COMPLETED' and fresh.review.review_revision==150
    assert {g:stacks.snapshot(fresh,g) for g in fresh.mt_invocations}==annotations
    result=replay(next(target.glob('mt_object_evaluation_v2_*')))
    assert (result['tp'],result['fp'],result['fn'])==(16,4,2)
    assert (result['ocr_summary']['n_exact'],result['ocr_summary']['n_reads'],result['ocr_summary']['n_text_errors'],result['ocr_summary']['n_gt_characters'])==(10,16,11,111)
    assert stamp(s.sidecar_path)==before and not h._test_ranking_calls
    evidence=os.environ.get('F4_REVISION_OUTPUT')
    if evidence:
        index=1
        while (Path(evidence)/f'private/approved_r150_proof_{index:03d}').exists():index+=1
        dest=Path(evidence)/f'private/approved_r150_proof_{index:03d}';dest.mkdir()
        shutil.copytree(target,dest/'export')
        shutil.copyfile(s.sidecar_path,dest/'review.json')
        shutil.copytree(s.sidecar_path.parent/'ready_for_evaluation',dest/'ready_for_evaluation')


def resolve_variant(rows=None,**changes):
    from auto_annotation_tool.ranking.mobile_f4_variant import resolve,REGISTRY
    import csv
    if rows is None:rows=list(csv.DictReader(REGISTRY.open(encoding='utf-8-sig'),delimiter=';'))
    kw=dict(archive_sha=ARCHIVE_SHA,session_id=SID,source_session_ids=[SID,SID],source_scenarios=['PAIR_DAY_01'],
        declared_variant=None,collection_variant='DAY_INT8',source_repeats=[1,1],rows=rows,source_verified=True)
    kw.update(changes);return resolve(**kw)


def test_R19_R20_exact_registry_and_raw_missing_variant(actual):
    from auto_annotation_tool.ranking.mobile_f4_variant import for_session
    from auto_annotation_tool.ranking.mobile_scientific_review import session_conditions,session_conditions_text
    before=stamp(actual.sidecar_path);raw=copy.deepcopy(actual.bundle.experiment_session)
    r=for_session(actual);assert r['status']=='VERIFIED' and r['label']=='DAY_INT8 · r01'
    assert r['scenario']=='PAIR_DAY_01' and r['provenance']=='F4_REGISTRY_SHA'
    assert r['declared_android_variant'] is None and r['declared_collection_variant']=='DAY_INT8'
    info=session_conditions(actual);text=session_conditions_text(info)
    assert info['variant']=='brak danych' and 'Wariant F4 (rejestr SHA): DAY_INT8 · r01' in text
    assert 'Wariant zadeklarowany przez Androida: brak danych' in text and 'provenance=F4_REGISTRY_SHA' in text
    assert actual.bundle.experiment_session==raw and stamp(actual.sidecar_path)==before


@pytest.mark.parametrize('change,reason',[
    ({'archive_sha':'1'*64},'REGISTRY_SHA_NOT_FOUND'),
    ({'source_session_ids':[]},'SOURCE_SESSION_ID_UNPROVEN'),
    ({'source_session_ids':[SID,'other']},'SOURCE_SESSION_ID_UNPROVEN'),
    ({'source_verified':False},'SOURCE_SHA_UNVERIFIED'),
    ({'source_scenarios':[]},'SOURCE_SCENARIO_UNPROVEN'),
    ({'source_repeats':[]},'SOURCE_REPEAT_UNPROVEN'),
])
def test_R21_fail_closed_without_identity_proof(change,reason):
    r=resolve_variant(**change)
    assert r['status']=='UNCONFIRMED' and r['reason']==reason and r['analysis_variant'] is None


def test_R22_duplicate_registry_sha_is_unconfirmed():
    from auto_annotation_tool.ranking.mobile_f4_variant import REGISTRY
    import csv
    rows=list(csv.DictReader(REGISTRY.open(encoding='utf-8-sig'),delimiter=';'))
    rows.append(next(r for r in rows if r['archive_sha256']==ARCHIVE_SHA).copy())
    assert resolve_variant(rows=rows)['reason']=='REGISTRY_SHA_DUPLICATE'


@pytest.mark.parametrize('changes,reason',[
    ({'declared_variant':'DAY_FP32'},'SOURCE_VARIANT_CONFLICT'),
    ({'collection_variant':'NIGHT_INT8'},'SOURCE_VARIANT_CONFLICT'),
    ({'source_scenarios':['PAIR_NIGHT_01']},'SOURCE_SCENARIO_CONFLICT'),
    ({'source_repeats':[2]},'SOURCE_REPEAT_CONFLICT'),
    ({'source_repeats':[1.5]},'SOURCE_REPEAT_INVALID'),
    ({'source_repeats':[True]},'SOURCE_REPEAT_INVALID'),
])
def test_R23_conflicting_nonempty_source_is_not_silently_fixed(changes,reason):
    r=resolve_variant(**changes)
    assert r['status']=='CONFLICT' and r['reason']==reason and r['analysis_variant'] is None


def test_R24_all_twelve_real_registry_labels_do_not_infer_night_model(baseline):
    from auto_annotation_tool.ranking.mobile_f4_variant import for_session
    results=[]
    for row in baseline['sessions']:
        s=MobileReviewSession.open(row['private_source'],sidecar_path=row['private_sidecar'])
        r=for_session(s);assert r['status']=='VERIFIED',(s.session_id,r['reason'])
        if r['scenario']=='PAIR_NIGHT_01' and r['source_variant']=='NIGHT_F32':
            assert r['analysis_variant']=='DAY_FP32'
        results.append(r)
    assert len(results)==12 and any(r['source_variant']=='NIGHT_F32' for r in results)
    out=os.environ.get('F4_REVISION_OUTPUT')
    if out:(Path(out)/'variant_labels_12.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')


@pytest.mark.parametrize('registry_case',['real','unknown','duplicate','conflict'])
def test_R19_R21_R22_R23_analysis_presentation_fails_closed(actual,tmp_path,monkeypatch,registry_case):
    from auto_annotation_tool.gui import mobile_analysis as analysis
    import csv
    inp=analysis.make_input(actual.bundle.report,source_path=actual.path,results_root=tmp_path/'03_wyniki',review_root=tmp_path/'catalog')
    rows=list(csv.DictReader(analysis.REGISTRY.open(encoding='utf-8-sig'),delimiter=';'))
    if registry_case!='real':
        row=next(r for r in rows if r['archive_sha256']==ARCHIVE_SHA)
        if registry_case=='unknown':rows.remove(row)
        elif registry_case=='duplicate':rows.append(row.copy())
        else:row['scenario']='PAIR_NIGHT_01'
        registry=tmp_path/'registry.csv'
        with registry.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0].keys(),delimiter=';');writer.writeheader();writer.writerows(rows)
        monkeypatch.setattr(analysis,'REGISTRY',registry)
    s=analysis.analyze([inp]).sessions[0]
    assert s.source_status=='VERIFIED'
    if registry_case=='real':
        assert (s.metadata['variant'],s.metadata['repeat'],s.metadata['scenario'])==('DAY_INT8','r01','PAIR_DAY_01')
        assert s.metadata['variant_provenance']=='F4_REGISTRY_SHA' and s.metadata['declared_android_variant'] is None
    else:
        assert s.metadata['variant']==('KONFLIKT' if registry_case=='conflict' else 'NIEPOTWIERDZONE')
        assert s.metadata['variant_provenance'] is None


@pytest.mark.parametrize('actual,gui',[(147,('light',1649,896)),(149,('light',1649,896)),
    (149,('dark',1649,896)),(149,('light',1180,760)),(149,('dark',1180,760))],indirect=True)
def test_R06_R19_R30_native_status_conditions_history_and_results(gui,request):
    from PIL import ImageGrab
    from auto_annotation_tool.ranking.mobile_review_status import approval_caption
    root,h,ed,s=gui;theme,width,height=request.node.callspec.params['gui']
    frozen=stamp(s.sidecar_path);ready(root,h,ed);ed.update_report();root.update()
    inv=ed.inventory();assert inv['total']==18 and inv['approval_ready']==18 and inv['approval_remaining']==0
    caption=approval_caption(inv,s.review.review_status,s.review.review_revision)
    assert caption==('18/18 sprawdzone — zatwierdzona rewizja r147' if s.review.review_revision==147 else '18/18 sprawdzone — oczekuje na zatwierdzenie')
    assert ed.report_host.progress.cget('text')==caption and caption in h.science_dashboard.canvas_title.cget('text')
    assert caption in ed.contexts.fields['session'].cget('text')
    assert 'DAY_INT8 · r01' in h.conditions_view.get('1.0','end')
    assert 'F4_REGISTRY_SHA' in h.science_dashboard.model['conditions']
    tabs=ed.review_tabs;receipt=[];out=os.environ.get('F4_REVISION_OUTPUT')
    for index,name in [(1,'canvas_results'),(2,'history'),(3,'conditions')]:
        tabs.select(index);root.update();h.window.lift();root.update();time.sleep(.15)
        if index==2:
            cap=h.science_dashboard.conditions_caption
            button=next(w for w in cap.master.winfo_children() if w.winfo_class()=='Button')
            assert cap.winfo_rootx()+cap.winfo_width()<button.winfo_rootx()
            assert int(float(cap.cget('wraplength')))<=cap.winfo_width()
        if out:
            dest=Path(out)/'native';dest.mkdir(exist_ok=True)
            filename=f'r{s.review.review_revision}_{theme}_{width}x{height}_{name}.png'
            x,y=h.window.winfo_rootx(),h.window.winfo_rooty()
            image=ImageGrab.grab(bbox=(x,y,x+width,y+height),all_screens=True)
            assert image.size==(width,height);image.save(dest/filename)
            receipt.append(dict(png=filename,caption=caption,review_revision=s.review.review_revision,
                review_status=s.review.review_status,variant='DAY_INT8 · r01',provenance='F4_REGISTRY_SHA',
                tk_scaling=float(root.tk.call('tk','scaling')),dpi_mode=DPI_MODE,input_method='native_Tk_event_generate_and_own_widget_API_not_physical_operator_input'))
    if out:(Path(out)/'native'/f'r{s.review.review_revision}_{theme}_{width}x{height}.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    assert stamp(s.sidecar_path)==frozen


def test_export_keeps_unsaved_reviewer(gui,monkeypatch,tmp_path):
    root,h,ed,s=gui;before=stamp(s.sidecar_path);h.reviewer_var.set('PRIVATE_REVIEWER')
    out,calls=dialogs(monkeypatch,tmp_path);h.export();root.update()
    assert stamp(s.sidecar_path)==before and not calls and not list(out.iterdir())
    assert h.reviewer_var.get()=='PRIVATE_REVIEWER'
    h.reviewer_var.set(s.review.reviewer_id)


def test_R07_R09_R17_standard_approve_button_then_export_reload(gui,monkeypatch,tmp_path):
    root,h,ed,s=gui;before={g:copy.deepcopy(stacks.snapshot(s,g)) for g in s.mt_invocations}
    dialogs(monkeypatch,tmp_path)
    h.complete_button.invoke();wait(root,h,ed)
    assert s.review.review_revision==150 and s.review.review_status=='COMPLETED'
    assert has_valid_ready_snapshot(s)
    assert {g:stacks.snapshot(s,g) for g in s.mt_invocations}==before
    saved=stamp(s.sidecar_path);h.complete_button.invoke();wait(root,h,ed)
    assert stamp(s.sidecar_path)==saved
    assert h._test_ranking_calls==['ranking']  # explicit approval, only on the fake index
    h._test_ranking_calls.clear()
    _,target,_=do_export(gui,monkeypatch,tmp_path)
    assert not h._test_ranking_calls and stamp(s.sidecar_path)==saved
    fresh=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    data=replay(next(target.glob('mt_object_evaluation_v2_*')))
    assert fresh.review.review_revision==data['review_revision']==150 and fresh.review.review_status=='COMPLETED'
    assert {g:stacks.snapshot(fresh,g) for g in fresh.mt_invocations}==before
    assert (data['tp'],data['fp'],data['fn'])==(16,4,2)
    assert (data['ocr_summary']['n_exact'],data['ocr_summary']['n_reads'],data['ocr_summary']['n_text_errors'],data['ocr_summary']['n_gt_characters'])==(10,16,11,111)
    evidence=os.environ.get('F4_REVISION_OUTPUT')
    if evidence:
        index=1
        while (Path(evidence)/f'private/approved_gui_r150_{index:03d}').exists():index+=1
        dest=Path(evidence)/f'private/approved_gui_r150_{index:03d}';dest.mkdir()
        shutil.copytree(target,dest/'export');shutil.copyfile(s.sidecar_path,dest/'review.json')
        shutil.copytree(s.sidecar_path.parent/'ready_for_evaluation',dest/'ready_for_evaluation')
        (dest/'receipt.json').write_text(json.dumps(dict(route='native Tk complete_button.invoke; explicit confirmation; GUI export; reload',
            review_revision=150,review_status='COMPLETED',source_archive_sha256=ARCHIVE_SHA,
            sidecar_sha256=stamp(s.sidecar_path)[2],primary_policy_sha256=POLICY_SHA256),indent=2),encoding='utf-8')
