"""T01-T20: view-only GTR/GTZ and source-aligned rulers on real Windows Tk."""
from pathlib import Path
from types import SimpleNamespace
import copy,hashlib,json,shutil,time
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_mt_detection_evaluation as mt
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/Z4_GTR_GTZ_GUIDES_20261027_001'
SID='exp-1792699303016-c8b05efb';G=SID+'-a00000008';OID='GT:'+G

def private_case(path,*,keep_manual=False):
    path.mkdir(parents=True,exist_ok=True)
    row=next(r for r in json.loads((OUT/'baseline_f4.json').read_text(encoding='utf-8'))['sessions'] if r['session_id']==SID)
    archive=OUT/'read_only_case.alprsession'
    if not archive.exists():shutil.copyfile(row['path'],archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==row['archive']['sha256']
    sidecar=path/'private.review.json';sidecar.write_text(json.dumps(row['sidecar_payload'],ensure_ascii=False,indent=2),encoding='utf-8')
    s=MobileReviewSession.open(archive,sidecar_path=sidecar)
    # Explicit test setup only. Real rev213 has a separate new manual outline;
    # never drop it or merge it on the production sidecar.
    if not keep_manual:
        for obj in list(core.scene(s,G)['objects']):
            if not obj['parent_id']:core.remove_operator_geometry(s,G,obj['object_id'])
    return s

def reference_quad():
    old=json.loads((OUT/'case_provenance.json').read_text(encoding='utf-8'))
    return copy.deepcopy(next(p['gt_quad'] for p in old['plates'].values() if p['gt_quad']))

@pytest.fixture
def gui(tmp_path,monkeypatch,request):
    s=private_case(tmp_path/'data',keep_manual=getattr(request,'param',False))
    monkeypatch.setattr(native,'source_session',lambda _:s)
    original=native.tk.Tk
    def scaled(*a,**kw):
        root=original(*a,**kw);root.tk.call('tk','scaling',120/72);return root
    monkeypatch.setattr(native.tk,'Tk',scaled)
    mode=getattr(request,'param',None)
    if mode in ('light','dark'):
        from auto_annotation_tool.gui import mobile_canvas_editor as editor
        palette=editor.palette
        monkeypatch.setattr(editor,'palette',lambda widget,bg:palette(widget,'#FFFFFF' if mode=='light' else '#17232D'))
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,_=next(gen)
    try:
        from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
        iid=next(i for i in h.subject_tree.get_children('') if G in subject_groups(s,h._subject_ids[i]))
        h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
        native.pump(root,lambda:h.current_invocation==G and h._image_entry==core.scene(s,G)['image_entry'] and not h._saving)
        ready(root,h,ed);yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass

def choose(root,h,ed,oid=OID):
    l,t,r,b,_=ed.stack_badges[oid]
    stamp=int(time.monotonic()*1000)%1000000000
    h.canvas.event_generate('<ButtonPress-1>',x=round((l+r)/2),y=round(t+10),time=stamp)
    h.canvas.event_generate('<ButtonRelease-1>',x=round((l+r)/2),y=round(t+10),time=stamp+10)
    ready(root,h,ed);assert ed.selected==oid

def motion(root,h,ed,point):
    x,y=map(round,ed.to_canvas(point));h.canvas.event_generate('<Motion>',x=x,y=y);native.pump(root)
    return x,y

def draw(root,h,ed):
    native.key(h,'d')
    for p in reference_quad():native.click(h,ed,p)
    ready(root,h,ed)

def test_T06_D_cursor_has_exact_crosshair_and_orthogonal_guides(gui):
    root,h,ed,s=gui;choose(root,h,ed);native.key(h,'d');x,y=motion(root,h,ed,(590,620))
    assert h.canvas.cget('cursor')=='crosshair'
    lines=h.canvas.find_withtag('gtr_guide_core');assert len(lines)==2
    coordinates=[h.canvas.coords(i) for i in lines]
    assert any(q[1]==q[3]==y for q in coordinates)
    assert any(q[0]==q[2]==x for q in coordinates)

def test_T01_baseline_twelve_sha_and_revisions():
    b=json.loads((OUT/'baseline_f4.json').read_text(encoding='utf-8'))
    assert len(b['sessions'])==12 and b['disk_free_bytes']>2*1024**3 and b['index']==''
    assert b['approved_policy_candidate_sha256']==mt.POLICY['approval']['approved_candidate_sha256']

def test_T02_one_object_has_GTR_and_GTZ(gui):
    root,h,ed,s=gui;pid=stacks.snapshot(s,G)['links'][OID]['plate_id']
    choose(root,h,ed);draw(root,h,ed)
    v=stacks.snapshot(s,G);assert len(v['plates'])==1 and v['links'][OID]['plate_id']==pid
    labels=[h.canvas.itemcget(i,'text') for i in h.canvas.find_withtag('clean_label_text')]
    assert any('GTR 1' in t for t in labels) and any('GTZ: WU1092S' in t for t in labels)
    assert v['plates'][pid]['gt_text']=='WU1092S' and v['plates'][pid]['gt_quad']

def test_T03_text_without_outline_has_independent_states(gui):
    root,h,ed,s=gui;frozen=s.sidecar_path.read_bytes();choose(root,h,ed)
    text=ed.contexts.data['object']
    assert 'GTR: brak obrysu' in text and 'GTZ: WU1092S' in text and 'Nie wykonano odczytu' in text
    assert OID not in ed.badges and s.sidecar_path.read_bytes()==frozen

def test_T04_outline_without_text_edits_GTZ_inline(gui):
    root,h,ed,s=gui;choose(root,h,ed);draw(root,h,ed)
    core.set_text(s,G,OID,'');ready(root,h,ed)
    assert 'GTZ: nieuzupełniony' in ed.contexts.data['object']
    assert any('GTZ: wpisz numer' in h.canvas.itemcget(i,'text') for i in h.canvas.find_withtag('clean_label_text'))
    native.pump(root,lambda:not h._saving and not ed.in_refresh)
    ed.start_entry(OID);native.pump(root);assert ed.entry.winfo_ismapped()
    ed.entry_var.set('WU1092S');ed.entry.event_generate('<Return>');ready(root,h,ed)
    rev=s.review.review_revision;native.pump(root,lambda:not h._saving and not ed.in_refresh)
    ed.start_entry(OID);ed.entry_var.set('CANCEL');ed.escape();ready(root,h,ed)
    assert s.review.review_revision==rev and next(iter(stacks.snapshot(s,G)['plates'].values()))['gt_text']=='WU1092S'

def test_T05_labels_do_not_change_canonical_counts(gui):
    root,h,ed,s=gui;before=mt.evaluate_unit(s,G);frozen=s.sidecar_path.read_bytes()
    choose(root,h,ed);ed.sync();ed.overlay();native.pump(root)
    assert ed.contexts.counter_labels['gt'].cget('text')=='Tablice GT'
    assert mt.evaluate_unit(s,G)==before and s.sidecar_path.read_bytes()==frozen

@pytest.mark.parametrize('gui',['light','dark'],indirect=True)
def test_T07_T08_fixed_two_layer_colors_in_both_themes(gui):
    root,h,ed,s=gui;native.key(h,'d');motion(root,h,ed,(590,620))
    for tag,color,width in (('gtr_guide_core','#165D8C',1),('gtr_guide_halo','#EAF5FC',2)):
        items=h.canvas.find_withtag(tag);assert len(items)==2
        assert all(h.canvas.itemcget(i,'fill')==color and float(h.canvas.itemcget(i,'width'))==width for i in items)
        assert all(not h.canvas.itemcget(i,'dash') for i in items)

def test_T08_Entry_owns_D_space_Tab_without_rulers(gui):
    root,h,ed,s=gui;choose(root,h,ed);draw(root,h,ed)
    ed.start_entry(OID);native.pump(root);entry=ed.entry;entry.focus_force();entry.selection_clear();entry.icursor('end')
    rev=s.review.review_revision
    for key in ('d','space'):
        entry.event_generate('<KeyPress-'+key+'>');entry.event_generate('<KeyRelease-'+key+'>');native.pump(root)
    assert ed.mode=='edit' and 'D' in ed.entry_var.get() and not h.canvas.find_withtag('gtr_guides')
    assert s.review.review_revision==rev
    ed.entry_var.set('WU1092S');entry.event_generate('<Tab>');native.pump(root)
    assert ed.entry is None and ed.selected==OID and s.review.review_revision==rev

def test_T09_T13_four_points_one_write_existing_link_shown(gui):
    root,h,ed,s=gui;choose(root,h,ed);before=copy.deepcopy(s.review.to_dict());pid=stacks.snapshot(s,G)['links'][OID]['plate_id']
    native.key(h,'d');assert 'GTR 1 ← MT 1 (BBOX)' in ed.clean_caption.cget('text')
    for point in reference_quad()[:3]:motion(root,h,ed,point);native.click(h,ed,point)
    assert s.review.review_revision==before['review_revision']
    assert len(h.canvas.find_withtag('drawing_corner'))==3 and h.canvas.find_withtag('gtr_guides')
    native.click(h,ed,reference_quad()[3]);ready(root,h,ed)
    assert s.review.review_revision==before['review_revision']+1 and not h.canvas.find_withtag('gtr_guides')
    assert 'MT 1 (BBOX) ↔ GTR 1' in ed.clean_caption.cget('text')
    assert stacks.snapshot(s,G)['links'][OID]['plate_id']==pid and len(stacks.snapshot(s,G)['plates'])==1
    assert {k:v for k,v in s.review.to_dict()['invocation_annotations'].items() if k!=G}=={k:v for k,v in before['invocation_annotations'].items() if k!=G}

@pytest.mark.parametrize('n',[1,2,3])
def test_T10_escape_no_write_or_orphan_guides(gui,n):
    root,h,ed,s=gui;choose(root,h,ed);frozen=s.sidecar_path.read_bytes();native.key(h,'d')
    for p in reference_quad()[:n]:motion(root,h,ed,p);native.click(h,ed,p)
    native.key(h,'Escape');ready(root,h,ed)
    assert not h.canvas.find_withtag('gtr_guides') and not ed.points and s.sidecar_path.read_bytes()==frozen

def test_T11_multiple_zoom_pan_roundtrips_match_saved_corners(gui):
    root,h,ed,s=gui;choose(root,h,ed);native.key(h,'d');expected=[]
    for point,z in zip(reference_quad(),(.8,1.7,2.5,1.2)):
        h._view_zoom=z;h._view_center=(h._view_center[0]+3,h._view_center[1]-2);h._paint_image();native.pump(root)
        x,y=motion(root,h,ed,point);expected.append(ed.to_source(x,y))
        assert any(h.canvas.coords(i)[1]==h.canvas.coords(i)[3]==y for i in h.canvas.find_withtag('gtr_guide_core'))
        native.click(h,ed,point)
    ready(root,h,ed);pid=stacks.snapshot(s,G)['links'][OID]['plate_id']
    assert stacks.snapshot(s,G)['plates'][pid]['gt_quad']==core.ordered_quad(expected,ed._read()['image_size'],ed._read()['roi'])

def test_T12_clipping_to_photo_and_leave_cleanup(gui):
    root,h,ed,s=gui;native.key(h,'d');x,y=motion(root,h,ed,(590,620))
    from auto_annotation_tool.gui.mobile_canvas_guides import image_bounds
    l,t,r,b=image_bounds(ed)
    for i in h.canvas.find_withtag('gtr_guides'):
        x1,y1,x2,y2=h.canvas.coords(i);assert l<=x1<=r and l<=x2<=r and t<=y1<=b and t<=y2<=b
    h.canvas.event_generate('<Motion>',x=1,y=round(h.canvas.winfo_height()/2));native.pump(root)
    assert not h.canvas.find_withtag('gtr_guides')
    motion(root,h,ed,(590,620));h.canvas.event_generate('<Leave>');native.pump(root)
    assert not h.canvas.find_withtag('gtr_guides')

@pytest.mark.parametrize('gui',[True],indirect=True)
def test_T14_only_selected_bbox_is_drawing_context(gui):
    root,h,ed,s=gui;choose(root,h,ed);native.key(h,'d');motion(root,h,ed,(590,620))
    assert len(h.canvas.find_withtag('drawing_mt_bbox'))==1
    assert not h.canvas.find_withtag('stack_badge') and not h.canvas.find_withtag('operator_quad')
    assert OID in h.canvas.gettags(h.canvas.find_withtag('drawing_mt_bbox')[0])

def test_T15_overlapping_independent_D_does_not_link(gui):
    root,h,ed,s=gui;before=copy.deepcopy(stacks.snapshot(s,G));ed.escape();draw(root,h,ed)
    after=stacks.snapshot(s,G);assert len(after['plates'])==len(before['plates'])+1
    assert after['links'][OID]==before['links'][OID] and 'MT: niepowiązany' in ed.clean_caption.cget('text')

def test_T16_two_overlapping_physical_references_stay_distinct(tmp_path):
    from test_z4_mt_detection_evaluation import approved_session
    s=approved_session(tmp_path,n=2);g='mt-one';v=stacks.snapshot(s,g);ids=list(v['plates'])
    q=next(iter(v['plates'].values()))['gt_quad'];stacks.edit_gt(s,g,'GT:a1',quad=q)
    frozen=s.sidecar_path.read_bytes();v=stacks.snapshot(s,g)
    ed=SimpleNamespace(h=SimpleNamespace(session=s,current_invocation=g),cache_key=None,_read=lambda:core.scene(s,g),selected='GT:a1')
    rows=clean.records(ed)
    assert len(rows)==2 and len({r['pid'] for r in rows})==2
    assert set(stacks.snapshot(s,g)['plates'])==set(ids) and s.sidecar_path.read_bytes()==frozen

@pytest.mark.parametrize('gui',[True],indirect=True)
def test_T17_existing_manual_explicit_link_uses_same_GTZ(gui):
    from auto_annotation_tool.gui.mobile_canvas_stacks import context_menu
    root,h,ed,s=gui;choose(root,h,ed);parents=copy.deepcopy(ed._read()['parents'])
    menu=context_menu(ed,show=False)
    n=next(i for i in range(menu.index('end')+1) if menu.type(i)=='command' and menu.entrycget(i,'label').startswith('Powiąż z ręczną GTR'))
    menu.invoke(n);ready(root,h,ed)
    v=stacks.snapshot(s,G);assert len(v['plates'])==1 and next(iter(v['plates'].values()))['gt_text']=='WU1092S'
    assert 'MT 1 (BBOX) ↔ GTR 1' in ed.clean_caption.cget('text') and ed._read()['parents']==parents

def test_T17_fp_never_receives_an_implicit_reference(gui):
    root,h,ed,s=gui;choose(root,h,ed)
    ed._commit(lambda:stacks.toggle_unassociated(s,G,OID,partial_review=True),'test FP');ready(root,h,ed)
    draw(root,h,ed);v=stacks.snapshot(s,G)
    assert v['links'][OID]['status']=='UNASSOCIATED_NO_PLATE'
    assert '↔' not in ed.clean_caption.cget('text')

def test_T18_undo_redo_restart_preserves_one_GTR_GTZ(gui):
    root,h,ed,s=gui;choose(root,h,ed);draw(root,h,ed);after=copy.deepcopy(stacks.snapshot(s,G)['plates'])
    parents=copy.deepcopy(ed._read()['parents']);ed.undo();ready(root,h,ed)
    assert not next(iter(stacks.snapshot(s,G)['plates'].values()))['gt_quad']
    ed.undo(redo=True);ready(root,h,ed)
    re=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert stacks.snapshot(re,G)['plates']==after and core.scene(re,G)['parents']==parents

def test_T19_bbox_vs_pose_ocr_contract_unchanged(gui):
    root,h,ed,s=gui;source=copy.deepcopy(s.attempts[G]);choose(root,h,ed);draw(root,h,ed)
    result=mt.evaluate_unit(s,G);diag=stacks.diagnostics(s,[G])['rows'][0]
    assert (result['n_gt'],result['tp'],result['fp'],result['fn'])==(1,0,1,1)
    assert diag['polygon_iou_pct'] is diag['nmce_pct'] is diag['cer'] is None
    assert diag['mz_status']=='NOT_RUN' and diag['mt_quad'] is None and s.attempts[G]==source

def test_T20_all_originals_and_reports_preserved():
    b=json.loads((OUT/'baseline_f4.json').read_text(encoding='utf-8'))
    for n,v in b['protected_files'].items():
        p=Path(n);assert p.exists()==v['exists']
        if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
    assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in b['prior_reports'].items())

def test_motion_updates_only_line_tags_not_full_context(gui,monkeypatch):
    root,h,ed,s=gui;choose(root,h,ed);native.key(h,'d');motion(root,h,ed,(590,620))
    ids=h.canvas.find_withtag('gtr_guides');frozen=s.sidecar_path.read_bytes()
    def unexpected(*args,**kwargs):raise AssertionError('Full overlay/context work in drawing Motion')
    with monkeypatch.context() as patch:
        patch.setattr(ed,'request_overlay',unexpected);patch.setattr(clean,'context',unexpected)
        for i in range(50):motion(root,h,ed,(590+i/10,620))
    assert h.canvas.find_withtag('gtr_guides')==ids and s.sidecar_path.read_bytes()==frozen

def test_T17_unavailable_source_has_no_fictional_link(gui,tmp_path,monkeypatch):
    root,h,ed,s=gui;choose(root,h,ed);draw(root,h,ed)
    with monkeypatch.context() as patch:
        patch.setattr(s,'path',tmp_path/'missing.alprsession')
        text=clean.bar_caption(ed)
        assert '↔' not in text and 'brak wiarygodnej lokalizacji' in text

@pytest.mark.parametrize('dpi',[96,144])
def test_T11_additional_tk_scaling_keeps_click_center(gui,dpi):
    root,h,ed,s=gui;frozen=s.sidecar_path.read_bytes()
    root.tk.call('tk','scaling',dpi/72);h.window.geometry('1180x760+20+20');ready(root,h,ed)
    choose(root,h,ed);native.key(h,'d');x,y=motion(root,h,ed,(590,620))
    expected=ed.to_source(x,y)
    lines=[h.canvas.coords(i) for i in h.canvas.find_withtag('gtr_guide_core')]
    assert any(q[1]==q[3]==y for q in lines) and any(q[0]==q[2]==x for q in lines)
    native.click(h,ed,(590,620));assert ed.points[0]==pytest.approx(expected)
    ed.escape();ready(root,h,ed);assert s.sidecar_path.read_bytes()==frozen

def test_T12_actual_image_change_clears_view_only_guides(gui):
    root,h,ed,s=gui;frozen=s.sidecar_path.read_bytes()
    choose(root,h,ed);native.key(h,'d');motion(root,h,ed,(590,620))
    assert h.canvas.find_withtag('gtr_guides')
    native.key(h,'e')
    native.pump(root,lambda:h.current_invocation!=G and not h._saving and ed.skey==h.current_invocation)
    ready(root,h,ed)
    assert not h.canvas.find_withtag('gtr_guides') and ed.mode=='select' and not ed.points
    assert s.sidecar_path.read_bytes()==frozen
