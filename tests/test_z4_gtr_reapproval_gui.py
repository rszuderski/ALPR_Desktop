"""Native Canvas events keep per-GTR approval separate from MT presence."""
import copy
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_gtr_reapproval import verified, pending, pid, MANUAL, G, OID, edit
from test_z4_frame_stacks import ready
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_canvas_partial_review as partial
from auto_annotation_tool.gui import mobile_canvas_navigator as nav

@pytest.fixture
def gui(tmp_path,monkeypatch):
    s=verified(tmp_path/'data')
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:ready(root,h,ed);yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass

def view(h,ed):
    return ed.c.winfo_width(),ed.c.winfo_height(),h._view_zoom,h._view_pixel_scale,h._view_center,ed.to_canvas((150,140))

def approved(root,h,ed,s):
    ready(root,h,ed)
    assert core.scene(s,G)['checked'] and ed.checked_var.get() and not pending(s)
    if ed.selected:
        assert ed.clean_gtr_confirm_button.winfo_manager() and ed.clean_gtr_confirm_button['text']=='Cofnij weryfikację · Spacja'
    else:assert not ed.clean_gtr_confirm_button.winfo_manager()

@pytest.mark.parametrize('size',['1649x896','1180x760'])
@pytest.mark.parametrize('approve',['space','button'])
def test_actual_corner_release_loses_checked_and_explicit_approval_restores_it(gui,size,approve):
    root,h,ed,s=gui;h.window.state('normal');h.window.geometry(size+'+10+10');ready(root,h,ed)
    native.focus_canvas(h);nav.select_layer(ed,'GTR');ready(root,h,ed)
    approved(root,h,ed,s);baseline=view(h,ed);parents=copy.deepcopy(core.scene(s,G)['parents'])
    q=core.scene(s,G)['objects'][0]['quad'];x,y=map(round,ed.to_canvas(q[0]))
    ed.c.event_generate('<ButtonPress-1>',x=x,y=y);assert ed.drag
    ed.c.event_generate('<B1-Motion>',x=x+8,y=y+3)
    ed.c.event_generate('<ButtonRelease-1>',x=x+8,y=y+3);ready(root,h,ed)
    assert not core.scene(s,G)['checked'] and not ed.checked_var.get() and pending(s)==[pid(s)]
    assert partial.progress(core.scene(s,G))['object_decisions'][OID]==partial.PLATE
    assert ed.clean_gtr_confirm_button.winfo_manager() and 'Zatwierdź ramki' in ed.clean_gtr_confirm_button['text']
    assert view(h,ed)==baseline and core.scene(s,G)['parents']==parents
    if approve=='space':native.key(h,'space')
    else:ed.clean_gtr_confirm_button.invoke()
    approved(root,h,ed,s)
    assert partial.progress(core.scene(s,G))['object_decisions'][OID]==partial.PLATE and view(h,ed)==baseline

def test_four_clicks_create_pending_manual_GTR_and_Space_approves_it(gui):
    root,h,ed,s=gui;ed.escape();native.key(h,'d');ready(root,h,ed)
    before=s.sidecar_path.read_bytes()
    for point in MANUAL[:3]:native.click(h,ed,point)
    assert s.sidecar_path.read_bytes()==before and core.scene(s,G)['checked']
    native.click(h,ed,MANUAL[3]);ready(root,h,ed)
    assert len(pending(s))==1 and not ed.checked_var.get() and ed.selected!=OID
    assert ed.clean_gtr_confirm_button.winfo_manager()
    native.key(h,'space');approved(root,h,ed,s)
    assert partial.progress(core.scene(s,G))['plate_count']==2

def test_open_edit_or_cancel_unstored_draft_keeps_verification_and_bytes(gui):
    root,h,ed,s=gui;before=s.sidecar_path.read_bytes();native.focus_canvas(h)
    nav.select_layer(ed,'GTR');native.key(h,'d');ready(root,h,ed)
    assert ed.mode=='edit';ed.escape();ready(root,h,ed)
    native.key(h,'d');native.click(h,ed,MANUAL[0]);native.key(h,'Escape');ready(root,h,ed)
    approved(root,h,ed,s);assert s.sidecar_path.read_bytes()==before

@pytest.mark.parametrize('kind',['POSE','BBOX'])
def test_source_space_approves_its_associated_changed_GTR(gui,kind):
    root,h,ed,s=gui;edit(s);ed.cache_key=None;ed.sync();nav.select_layer(ed,kind);ready(root,h,ed)
    assert pending(s)==[pid(s)]
    native.key(h,'space');approved(root,h,ed,s)
    before=copy.deepcopy(core.scene(s,G)['objects']);native.key(h,'space');ready(root,h,ed)
    assert not ed.checked_var.get() and pending(s)==[pid(s)]
    native.key(h,'space');approved(root,h,ed,s)
    assert core.scene(s,G)['objects']==before
