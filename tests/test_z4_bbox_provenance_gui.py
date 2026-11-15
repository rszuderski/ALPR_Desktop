"""Source BBOX has one MT identity; it must never masquerade as operator GT."""
from pathlib import Path
import copy,json
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks
from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean,mobile_canvas_stacks as stack_ui

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/Z4_BBOX_PROVENANCE_20261027_001'
SID='exp-1792699303016-c8b05efb';G=SID+'-a00000008';OID='GT:'+G

@pytest.fixture
def bbox_gui(tmp_path,monkeypatch,request):
    current=getattr(request,'param','historical')=='current'
    source=(OUT/'baseline_f4.json' if current else ROOT/'output/Z4_GT_EDIT_RESPONSIVENESS_20261026_001/baseline_f4.json')
    row=next(r for r in json.loads(source.read_text(encoding='utf-8'))['sessions'] if r['session_id']==SID)
    sidecar=tmp_path/'private.review.json';sidecar.write_text(json.dumps(row['sidecar_payload'],ensure_ascii=False,indent=2),encoding='utf-8')
    s=MobileReviewSession.open(Path(row['path']),sidecar_path=sidecar)
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,_=next(gen)
    try:
        from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
        iid=next(i for i in h.subject_tree.get_children('') if G in subject_groups(s,h._subject_ids[i]))
        h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
        native.pump(root,lambda:h.current_invocation==G and h._image_entry==core.scene(s,G)['image_entry'] and not h._saving)
        ready(root,h,ed)
        yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass

def source_label(h):
    return [h.canvas.itemcget(item,'text') for item in h.canvas.find_withtag('stack_badge_text')
            if OID in h.canvas.gettags(item)]

def choose_source(root,h,ed):
    l,t,r,b,pid=ed.stack_badges[OID]
    x,y=round((l+r)/2),round((t+b)/2)
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y,time=10000)
    h.canvas.event_generate('<ButtonRelease-1>',x=x,y=y,time=10010)
    ready(root,h,ed);assert ed.selected==OID

def test_real_source_bbox_and_manual_gt_are_distinct_without_phantom_gt_field(bbox_gui):
    root,h,ed,s=bbox_gui;before=s.sidecar_path.read_bytes();scene=core.scene(s,G)
    manual=next(o for o in scene['objects'] if not o['parent_id'])
    bbox=h.canvas.find_withtag('model_location_bbox')
    assert len(bbox)==1 and h.canvas.itemcget(bbox[0],'outline')=='#58B8D5'
    assert source_label(h) and all('MT 1 · BBOX' in text and 'GT ' not in text for text in source_label(h))
    assert OID not in ed.badges
    assert not [i for i in h.canvas.find_withtag('gt_badge') if OID in h.canvas.gettags(i)]
    manual_label=[h.canvas.itemcget(i,'text') for i in h.canvas.find_withtag('stack_badge_text')
                  if manual['object_id'] in h.canvas.gettags(i)]
    assert manual_label and all('GTR 1' in text for text in manual_label)
    assert scene['parents'][0]['quad'] is None and manual['quad']
    assert s.sidecar_path.read_bytes()==before

def test_bbox_input_is_blocked_without_a_real_gt_outline(bbox_gui):
    root,h,ed,s=bbox_gui;before=s.sidecar_path.read_bytes();choose_source(root,h,ed)
    assert ed.contexts.titles['object'].cget('text')=='WYBRANE WSKAZANIE MT'
    assert 'BBOX' in ed.contexts.data['object'] and 'brak narożników' in ed.contexts.data['object']
    assert not ed.clean_gt_slot.winfo_ismapped()
    assert not ed._gt_policy(OID)['gt_editable']
    ed.start_entry(OID);native.pump(root)
    assert ed.entry is None and s.sidecar_path.read_bytes()==before

def test_ppm_changes_the_source_decision_but_keeps_manual_gt_and_bbox(bbox_gui):
    root,h,ed,s=bbox_gui;scene=core.scene(s,G);parents=copy.deepcopy(scene['parents'])
    manual=copy.deepcopy(next(o for o in scene['objects'] if not o['parent_id']))
    choose_source(root,h,ed)
    _,l,t,r,b=ed.stack_bbox_hit[0];native.click(h,ed,((l+r)/2,(t+b)/2),3);ready(root,h,ed)
    assert partial.progress(ed._read())['object_decisions'][OID]==partial.NOT_PLATE
    assert h.canvas.find_withtag('model_location_bbox') and len(h.canvas.find_withtag('fp_diagonal'))==2
    assert all('BBOX' in text and 'Błędna' in text for text in source_label(h))
    assert OID not in ed.badges and core.scene(s,G)['parents']==parents
    assert core._object(core.scene(s,G),manual['object_id'])['quad']==manual['quad']
    ed.undo();ready(root,h,ed)
    assert partial.progress(ed._read())['object_decisions'][OID]==partial.UNRESOLVED
    assert OID not in ed.badges

def test_space_does_not_turn_bbox_into_gt_or_renumber_the_drawn_gt(bbox_gui):
    root,h,ed,s=bbox_gui;choose_source(root,h,ed);parents=copy.deepcopy(ed._read()['parents'])
    native.key(h,'space');ready(root,h,ed)
    assert partial.progress(ed._read())['object_decisions'][OID]==partial.PLATE
    rows=clean.records(ed)
    source=next(r for r in rows if r['oid']==OID)
    drawn=next(r for r in rows if not r['candidate'])
    assert source['gt'] is None and source['n'] is None and drawn['n']==1
    assert all('MT 1 · BBOX' in text for text in source_label(h))
    assert OID not in ed.badges and not ed._gt_policy(OID)['gt_editable']
    assert ed._read()['parents']==parents

def test_explicit_existing_link_joins_the_gt_without_synthesizing_mt_corners(bbox_gui):
    root,h,ed,s=bbox_gui;choose_source(root,h,ed)
    scene=core.scene(s,G);parents=copy.deepcopy(scene['parents']);before=core.summary(s)
    manual=next(o for o in scene['objects'] if not o['parent_id']);quad=copy.deepcopy(manual['quad'])
    menu=stack_ui.context_menu(ed,show=False)
    index=next(i for i in range(menu.index('end')+1) if menu.type(i)=='command'
               and menu.entrycget(i,'label').startswith('Powiąż z ręczną GTR'))
    menu.invoke(index);ready(root,h,ed)
    v=stacks.snapshot(s,G)
    assert len(v['plates'])==1 and v['candidates'][0]['plate_id']
    assert v['candidates'][0]['mt_quad'] is None and ed._read()['parents']==parents
    assert next(iter(v['plates'].values()))['gt_quad']==quad
    assert len(h.canvas.find_withtag('model_location_bbox'))==1  # Source BBOX remains distinct from real GT.
    assert OID in ed.badges and ed._gt_policy(OID)['gt_editable']
    ed.start_entry(OID);assert ed.entry is not None
    ed.entry_var.set('WU1092S');ed.entry_enter();ready(root,h,ed)
    diagnostic=stacks.diagnostics(s,[G])['rows'][0]
    assert diagnostic['polygon_iou_pct'] is diagnostic['nmce_pct'] is diagnostic['cer'] is None
    after=core.summary(s)
    assert (after['geometry_n'],after['evaluated_reads'])==(before['geometry_n'],before['evaluated_reads'])

@pytest.mark.parametrize('bbox_gui',['current'],indirect=True)
def test_existing_bbox_only_text_is_preserved_without_a_fake_gt_badge(bbox_gui):
    root,h,ed,s=bbox_gui;before=s.sidecar_path.read_bytes();revision=s.review.review_revision
    v=stacks.snapshot(s,G);assert next(iter(v['plates'].values()))['gt_text']=='WU1092S'
    assert all('MT 1 · BBOX' in text for text in source_label(h))
    assert OID not in ed.badges
    choose_source(root,h,ed)
    assert 'GTZ: WU1092S' in ed.contexts.data['object']
    ed.start_entry(OID);native.pump(root)
    assert ed.entry is None and s.review.review_revision==revision and s.sidecar_path.read_bytes()==before

@pytest.mark.parametrize('bbox_gui',['current'],indirect=True)
def test_new_drawn_gt_and_existing_link_menu_use_the_same_number(bbox_gui):
    root,h,ed,s=bbox_gui
    baseline=json.loads((ROOT/'output/Z4_GT_EDIT_RESPONSIVENESS_20261026_001/baseline_f4.json').read_text(encoding='utf-8'))
    row=next(r for r in baseline['sessions'] if r['session_id']==SID)
    old_scene=row['sidecar_payload']['invocation_annotations'][G][core.KEY]
    quad=next(o['quad'] for o in old_scene['objects'] if not o['parent_id'])
    native.key(h,'d')
    for point in quad:native.click(h,ed,point)
    ready(root,h,ed)
    drawn=next(r for r in clean.records(ed) if not r['candidate'])
    assert drawn['n']==1
    choose_source(root,h,ed);menu=stack_ui.context_menu(ed,show=False)
    labels=[menu.entrycget(i,'label') for i in range(menu.index('end')+1) if menu.type(i)=='command']
    assert 'Powiąż z ręczną GTR 1' in labels
    assert not any(label.startswith('Powiąż z ręczną GTR 2') for label in labels)
