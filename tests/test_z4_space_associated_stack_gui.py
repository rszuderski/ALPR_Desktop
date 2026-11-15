"""Any active layer toggles its stack without cycling layers."""
import copy
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_gtr_reapproval_gui import gui
from test_z4_gtr_reapproval import edit,pid,pending,G,OID
from test_z4_frame_stacks import ready
from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_geometry_layers as layers,mobile_canvas_approval_visuals as colors
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_canvas_partial_review as partial

@pytest.mark.parametrize('kind',['BBOX','POSE','GTR'])
def test_Space_approves_pending_GT_and_presence_from_any_layer_without_switching(gui,kind):
    root,h,ed,s=gui;edit(s);partial.toggle_plate(s,G,OID);ready(root,h,ed);assert nav.select_layer(ed,kind);ready(root,h,ed)
    parents=copy.deepcopy(core.scene(s,G)['parents']);q=copy.deepcopy(core.scene(s,G)['objects'][0]['quad']);revision=s.review.review_revision
    native.key(h,'space');ready(root,h,ed)
    assert not pending(s) and core.scene(s,G)['checked'] and ed.checked_var.get()
    assert partial.progress(core.scene(s,G))['object_decisions'][OID]==partial.PLATE
    assert layers.model(ed)['active']==kind and core.scene(s,G)['parents']==parents and core.scene(s,G)['objects'][0]['quad']==q
    assert s.review.review_revision==revision+1
    native.key(h,'space');ready(root,h,ed)
    assert not ed.checked_var.get() and pending(s) and s.review.review_revision==revision+2
    assert core.scene(s,G)['objects'][0]['quad']==q and core.scene(s,G)['parents']==parents
    native.key(h,'space');ready(root,h,ed);assert ed.checked_var.get() and s.review.review_revision==revision+3
    item=next(i for i in ed.c.find_withtag('operator_quad') if pid(s) in ed.c.gettags(i));assert ed.c.itemcget(item,'outline')==colors.GREEN

@pytest.mark.parametrize('kind',['BBOX','POSE','GTR'])
def test_button_uses_same_stack_approval_on_every_layer(gui,kind):
    root,h,ed,s=gui;edit(s);ready(root,h,ed);nav.select_layer(ed,kind);ready(root,h,ed)
    assert ed.clean_gtr_confirm_button.winfo_manager() and ed.clean_gtr_confirm_button['text']=='Zatwierdź ramki · Spacja'
    ed.clean_gtr_confirm_button.invoke();ready(root,h,ed);assert ed.checked_var.get() and not pending(s)

def test_hover_over_reserved_stack_after_Escape_confirms_without_reselecting(gui,monkeypatch):
    root,h,ed,s=gui;edit(s);ready(root,h,ed);nav.select_layer(ed,'GTR');ed.escape();ready(root,h,ed)
    q=core.scene(s,G)['objects'][0]['quad'];point=tuple(sum(p[i] for p in q)/4 for i in (0,1));xy=ed.to_canvas(point)
    monkeypatch.setattr(ed,'_canvas_hotkey_pointer',lambda event:xy)
    native.key(h,'space');ready(root,h,ed);assert not ed.selected and ed.checked_var.get() and not pending(s)

def test_no_target_draft_or_typing_cannot_confirm_a_stack(gui,monkeypatch):
    root,h,ed,s=gui;edit(s);ready(root,h,ed);ed.escape();ready(root,h,ed);before=s.sidecar_path.read_bytes()
    monkeypatch.setattr(ed,'_canvas_hotkey_pointer',lambda event:None)
    native.key(h,'space');ready(root,h,ed);assert s.sidecar_path.read_bytes()==before and pending(s)
    native.key(h,'d');ready(root,h,ed);native.key(h,'space');ready(root,h,ed)
    assert ed.mode=='draw' and s.sidecar_path.read_bytes()==before and pending(s)
    ed.escape();nav.select_layer(ed,'GTR',OID);ed.start_entry(OID);ready(root,h,ed)
    assert ed.confirm(NS(widget=ed.entry,keysym='space',state=0)) is None
    assert s.sidecar_path.read_bytes()==before and pending(s);ed.escape()
