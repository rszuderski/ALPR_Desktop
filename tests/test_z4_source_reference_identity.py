"""A source layer never edits a linked GTR; BAD keeps source and references."""
import copy
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_gtr_reapproval_gui import gui
from test_z4_gtr_reapproval import G,OID,MANUAL,edit,pid
from test_z4_frame_stacks import ready
from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_geometry_layers as layers
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks

def frozen(s):return s.sidecar_path.read_bytes(),copy.deepcopy(s.review.to_dict())

@pytest.mark.parametrize('kind',['BBOX','POSE'])
def test_R_on_source_cannot_reset_linked_GTR(gui,kind):
    root,h,ed,s=gui;edit(s);ready(root,h,ed);assert nav.select_layer(ed,kind,OID);ready(root,h,ed)
    before=frozen(s);ed.remove_selected(NS(state=0));ready(root,h,ed)
    assert frozen(s)==before

def test_R_removes_GTR_geometry_and_undo_restores_same_reference(gui):
    root,h,ed,s=gui;q=edit(s);ready(root,h,ed);nav.select_layer(ed,'GTR',OID);ready(root,h,ed)
    ident=pid(s);ed.remove_selected(NS(state=0));ready(root,h,ed)
    assert stacks.snapshot(s,G)['plates'][ident]['gt_quad'] is None
    assert layers.model(ed)['gtr_count']==0 and layers.model(ed)['active'] in ('BBOX','POSE')
    ed.undo();ready(root,h,ed)
    assert pid(s)==ident and stacks.snapshot(s,G)['plates'][ident]['gt_quad']==q

@pytest.mark.parametrize('kind',['BBOX','POSE'])
@pytest.mark.parametrize('undo',['toggle','history'])
def test_BAD_roundtrip_preserves_all_GTR_and_original_source(gui,kind,undo):
    root,h,ed,s=gui;edit(s);manual=core.add_plate(s,G,MANUAL,partial_review=True);ready(root,h,ed)
    nav.select_layer(ed,kind,OID);ready(root,h,ed);ed.show_fp_var.set(True)
    before=copy.deepcopy(stacks.snapshot(s,G));parents=copy.deepcopy(core.scene(s,G)['parents'])
    def ppm():
        l,t,r,b=ed._context_badge['rect'];x,y=round((l+r)/2),round((t+b)/2)
        ed.c.event_generate('<ButtonPress-3>',x=x,y=y);ed.c.event_generate('<ButtonRelease-3>',x=x,y=y);ready(root,h,ed)
    ppm();bad=stacks.snapshot(s,G)
    assert bad['links'][OID]['status']=='UNASSOCIATED_NO_PLATE' and bad['plates']==before['plates']
    if undo=='toggle':ppm()
    else:ed.undo();ready(root,h,ed)
    after=stacks.snapshot(s,G)
    assert after['plates']==before['plates'] and after['links']==before['links']
    assert core.scene(s,G)['parents']==parents and layers.model(ed)['active']==kind

def test_unselected_manual_GTR_has_its_own_identity_label(gui):
    root,h,ed,s=gui;manual=core.add_plate(s,G,MANUAL,partial_review=True);ready(root,h,ed)
    nav.select_layer(ed,'POSE',OID);ready(root,h,ed);before=frozen(s)
    labels=[i for i in ed.c.find_withtag('reference_identity_text') if manual in ed.c.gettags(i)]
    assert len(labels)==1 and 'GTR 2' in ed.c.itemcget(labels[0],'text')
    assert frozen(s)==before
    l,t,r,b=ed.reference_identity_hits[manual]
    assert nav.hit(ed,(l+r)/2,(t+b)/2)==(manual,'layer:GTR',None)

def test_coincident_source_and_GTR_are_explained_without_changing_geometry(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'POSE',OID);ready(root,h,ed);before=frozen(s)
    text=ed.navigator.hint['text']
    assert 'Pose MT' in text and 'GTR 1' in text and 'te same' in text
    assert any('GTR 1' in ed.c.itemcget(i,'text') for i in ed.c.find_withtag('reference_identity_text'))
    assert frozen(s)==before

def test_reference_identity_labels_obey_REF_and_drawing_visibility(gui):
    root,h,ed,s=gui;core.add_plate(s,G,MANUAL,partial_review=True);ready(root,h,ed)
    nav.select_layer(ed,'POSE',OID);ready(root,h,ed);assert ed.c.find_withtag('reference_identity_text')
    ed.ref_visible.set(False);ed.request_overlay();ready(root,h,ed)
    assert not ed.c.find_withtag('reference_identity_text')
    ed.ref_visible.set(True);ed.escape();native.key(h,'d');ready(root,h,ed)
    assert ed.mode=='draw' and not ed.c.find_withtag('reference_identity_text')

def test_crop_resolves_selected_source_without_navigation_or_stale_cached_crop(tmp_path,monkeypatch):
    from test_z4_mt_geometry_layers import source_session
    s=source_session(tmp_path/'data',mz=True)
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        ready(root,h,ed);nav.select_layer(ed,'POSE',OID);ready(root,h,ed)
        selected=ed.selected;before=frozen(s);center=h._view_center;captures=[]
        h._crop_preview_row={'attempt_id':'other-attempt','image_entry':'wrong.jpg'}
        monkeypatch.setattr(h,'_multi_focus_attempt',lambda _:pytest.fail('Crop must not navigate the record tree'))
        monkeypatch.setattr(h,'_show_crop_preview',lambda:captures.append(copy.deepcopy(h._crop_preview_row)))
        assert ed.clean_menu.entrycget(ed.clean_mz_crop_index,'state')=='normal'
        ed.show_crop()
        assert len(captures)==1 and captures[0]['attempt_id']=='a0' and captures[0]['image_entry'] in s.entry_names
        assert ed.selected==selected and h._view_center==center and frozen(s)==before
    finally:
        try:next(gen)
        except StopIteration:pass

def test_crop_unavailable_for_NOT_RUN_and_manual_reference(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'POSE',OID);ready(root,h,ed)
    assert ed.clean_menu.entrycget(ed.clean_mz_crop_index,'state')=='disabled'
    manual=core.add_plate(s,G,MANUAL,partial_review=True);ready(root,h,ed);ed._choose_plate(manual);ready(root,h,ed)
    assert ed.clean_menu.entrycget(ed.clean_mz_crop_index,'state')=='disabled'

def test_pending_presence_does_not_claim_linked_GTR_is_missing(gui):
    from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
    root,h,ed,s=gui;partial.toggle_plate(s,G,OID);ready(root,h,ed)
    nav.select_layer(ed,'POSE',OID);ready(root,h,ed)
    value=layers.model(ed);before=frozen(s);text=nav.source_context(ed,value)[1]
    assert value['gtr_count']==1 and 'Brak powiązanej referencji GTR' not in text
    assert 'oczekuje na zatwierdzenie ramek' in text and frozen(s)==before
