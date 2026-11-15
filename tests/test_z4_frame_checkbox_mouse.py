"""Real Tk mouse/keyboard parity; only private review fixtures are changed."""
import copy
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_space_toggle_verification import gui, stable
from test_z4_gtr_reapproval import verified, G, OID, pid
from test_z4_frame_stacks import ready
from auto_annotation_tool.gui import mobile_canvas_navigator as nav
from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial,mobile_canvas_review as core

def click_checkbox(root,ed):
    ed.approve.update_idletasks()
    # Windows Tk invokes a Checkbutton on release only after pointer entry.
    ed.approve.event_generate('<Enter>',x=12,y=ed.approve.winfo_height()//2)
    ed.approve.event_generate('<ButtonPress-1>',x=12,y=ed.approve.winfo_height()//2)
    ed.approve.event_generate('<ButtonRelease-1>',x=12,y=ed.approve.winfo_height()//2)
    ed.approve.event_generate('<Leave>')
    root.update()

@pytest.mark.parametrize('kind',['BBOX','POSE','GTR'])
def test_mouse_checkbox_toggles_same_stack_as_Space_without_changing_geometry(gui,kind):
    root,h,ed,s=gui;nav.select_layer(ed,kind);ready(root,h,ed)
    before=stable(s);revision=s.review.review_revision
    assert ed.approve['state']=='normal' and ed.frame_checked_var.get()
    for index,expected in enumerate((False,True,False,True),1):
        if index%2:click_checkbox(root,ed)
        else:native.key(h,'space')
        ready(root,h,ed)
        assert partial.stack_is_verified(core.scene(s,G),oid=OID) is expected
        assert ed.frame_checked_var.get() is expected and ed.checked_var.get() is expected
        assert s.review.review_revision==revision+index and stable(s)==before

@pytest.fixture
def two_gui(tmp_path,monkeypatch):
    s=verified(tmp_path/'data',n=2)
    for oid in (OID,'GT:a1'):partial.confirm_stack(s,G,oid=oid)
    monkeypatch.setattr(native,'source_session',lambda _:s)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

def test_checkbox_shows_selected_stack_not_whole_image_and_keeps_other_plate(two_gui):
    root,h,ed,s=two_gui;nav.select_layer(ed,'BBOX',OID);ready(root,h,ed)
    other=copy.deepcopy(core.scene(s,G)['objects'][1]);revision=s.review.review_revision
    click_checkbox(root,ed);ready(root,h,ed)
    assert not ed.frame_checked_var.get() and not ed.checked_var.get()
    assert partial.stack_is_verified(core.scene(s,G),oid='GT:a1')
    nav.select_layer(ed,'BBOX','GT:a1');ready(root,h,ed)
    assert ed.frame_checked_var.get() and not ed.checked_var.get()
    assert core.scene(s,G)['objects'][1]==other and s.review.review_revision==revision+1
    nav.select_layer(ed,'GTR',OID);ready(root,h,ed);click_checkbox(root,ed);ready(root,h,ed)
    assert ed.frame_checked_var.get() and ed.checked_var.get()
    assert ed.contexts.fields['image'].winfo_manager() # Whole-image completion stays explicit.

def test_checkbox_without_target_or_during_draft_cannot_confirm_every_frame(gui):
    root,h,ed,s=gui;ed.escape();ready(root,h,ed);before=s.sidecar_path.read_bytes()
    assert ed.approve['state']=='disabled' and 'Wybierz' in ed.approve['text']
    click_checkbox(root,ed);ready(root,h,ed);assert s.sidecar_path.read_bytes()==before
    nav.select_layer(ed,'BBOX',OID);native.key(h,'d');ready(root,h,ed)
    if ed.mode!='draw':ed.escape();native.key(h,'d');ready(root,h,ed)
    assert ed.mode=='draw' and ed.approve['state']=='disabled'
    click_checkbox(root,ed);assert s.sidecar_path.read_bytes()==before

def test_failed_checkbox_save_restores_display_from_saved_state(gui,monkeypatch):
    root,h,ed,s=gui;nav.select_layer(ed,'GTR');ready(root,h,ed);before=s.sidecar_path.read_bytes()
    def fail(*a,**kw):raise OSError('Injected write failure on private test copy')
    monkeypatch.setattr(partial,'toggle_stack',fail)
    click_checkbox(root,ed);ready(root,h,ed)
    assert ed.frame_checked_var.get() and ed.checked_var.get() and s.sidecar_path.read_bytes()==before
    assert 'Nie zapisano' in h.status_var.get()

def test_space_on_focused_checkbox_does_not_double_toggle(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'GTR');ready(root,h,ed);revision=s.review.review_revision
    ed.approve.focus_force();native.pump(root,lambda:root.focus_get() is ed.approve)
    for _ in range(5):ed.approve.event_generate('<KeyPress-space>');root.update()
    ed.approve.event_generate('<KeyRelease-space>');ready(root,h,ed)
    assert s.review.review_revision==revision+1 and not ed.frame_checked_var.get()
    click_checkbox(root,ed);ready(root,h,ed)
    assert s.review.review_revision==revision+2 and ed.frame_checked_var.get()

def test_click_checkbox_during_save_is_ignored(gui):
    root,h,ed,s=gui;nav.select_layer(ed,'GTR');ready(root,h,ed);before=s.sidecar_path.read_bytes()
    h._saving=True
    try:
        ed.sync();assert ed.approve['state']=='disabled'
        click_checkbox(root,ed);assert s.sidecar_path.read_bytes()==before
    finally:h._saving=False;ready(root,h,ed)
    assert ed.approve['state']=='normal'
