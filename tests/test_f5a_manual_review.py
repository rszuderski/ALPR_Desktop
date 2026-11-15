"""Ordinary F5A Canvas, native Tk events, private sidecars only.

Real K02 is opt-in via ALPR_OBSERVATION_K02. Its test decisions are mechanics
fixtures, never accepted scientific GT. Synthetic text tests use no real plate.
"""
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
import test_mobile_p2b2_canvas_gui as native
from test_mobile_p2b2_canvas_review import source_session
from test_z4_frame_stacks import ready
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean
from auto_annotation_tool.gui import mobile_canvas_visibility as visibility
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking import mobile_candidate_stacks as stacks
from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking import mobile_f5a_assisted as a3
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def manual_gui(tmp_path, monkeypatch):
    s=source_session(tmp_path,n=1)
    s.bundle.report.experiment_index['series_id']=a3.SERIES
    monkeypatch.setattr(native,'source_session',lambda _:s)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)


def hidden(root,h,ed):
    ready(root,h,ed)
    assert visibility.gt_only(ed)
    assert not clean.source(ed)['candidates']
    assert not any(r.get('seed') or r['candidate'] for r in clean.records(ed))
    assert all(r['source']=='saved_reference' for r in visibility.contours(ed))
    assert not ed.source_hud.winfo_ismapped()
    assert 'Wyniki modeli ukryte' in ed.contexts.data['image']
    assert 'MZ:' not in ' '.join(ed.contexts.data.values())
    native.key(h,'Shift_L')
    assert not ed.shift_preview.active


def draw(h,ed,points):
    native.key(h,'d')
    assert ed.mode=='draw'
    for point in points:native.click(h,ed,point)
    rows=clean.records(ed)
    row=next(r for r in rows if r['plate'])
    assert row['plate']['geometry_state']=='operator_drawn'
    ed._choose_plate(row['oid']);ed.start_entry(row['oid'])
    assert ed.entry is not None,h.status_var.get()
    ed.entry.focus_force();native.pump(h.window,lambda:h.window.focus_get() is ed.entry)
    return row['pid']


def unreadable(root,ed,monkeypatch):
    from auto_annotation_tool.gui import mobile_canvas_editor
    monkeypatch.setattr(mobile_canvas_editor.simpledialog,'askstring',
        lambda *a,**kw:'Nieczytelny tekst na tej klatce')
    ed.entry.event_generate('<Button-3>');root.update()
    assert ed.entry is None


def test_manual_gt_text_autosave_enter_and_explicit_model_reveal(manual_gui,monkeypatch):
    root,h,ed,s=manual_gui;gid=h.current_invocation
    parents=copy.deepcopy(core.scene(s,gid)['parents'])
    hidden(root,h,ed)
    # Deliberately independent synthetic geometry, near the synthetic MT.
    pid=draw(h,ed,[(145,135),(305,135),(305,215),(145,215)])
    ed.entry.delete(0,'end');ed.entry.insert(0,'SYN123')
    ed.entry.event_generate('<KeyRelease-3>')
    native.pump(root,lambda:stacks.snapshot(s,gid)['plates'][pid]['gt_text']=='SYN123')
    ed.entry.delete(0,'end');ed.entry.insert(0,'SYN124')
    ed.entry.event_generate('<Return>');ready(root,h,ed)
    assert stacks.snapshot(s,gid)['plates'][pid]['gt_text']=='SYN124'
    hidden(root,h,ed)
    assert all(c['association_status']=='UNRESOLVED' for c in stacks.snapshot(s,gid)['candidates'])
    before=s.sidecar_path.read_bytes()
    ed.clean_menu.invoke(1);ready(root,h,ed)
    assert not visibility.gt_only(ed) and s.sidecar_path.read_bytes()==before
    assert clean.records(ed)[0]['candidate']['mz_prediction']=='WA01234'
    row=clean.records(ed)[0];ed._choose_plate(row['oid']);native.key(h,'space');ready(root,h,ed)
    assert partial.progress(core.scene(s,gid))['object_decisions'][row['oid']]==partial.PLATE
    assert core.summary(s)['evaluated_reads']==1
    assert core.scene(s,gid)['parents']==parents
    # Undo/redo stays on the existing ordinary history.
    ed.undo();ready(root,h,ed)
    assert partial.progress(core.scene(s,gid))['object_decisions'][row['oid']]==partial.UNRESOLVED
    ed.undo(redo=True);ready(root,h,ed)
    assert partial.progress(core.scene(s,gid))['object_decisions'][row['oid']]==partial.PLATE
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    # The synthetic report label was patched only in memory; compare saved
    # assessments, not that deliberately different source metadata.
    assert reopened.review.invocation_annotations==s.review.invocation_annotations
    assert reopened.review.review_revision==s.review.review_revision


def test_manual_gt_unreadable_and_frame_na_are_explicit(manual_gui,monkeypatch):
    root,h,ed,s=manual_gui;gid=h.current_invocation
    pid=draw(h,ed,[(650,350),(800,350),(800,420),(650,420)])
    unreadable(root,ed,monkeypatch);ready(root,h,ed)
    p=stacks.snapshot(s,gid)['plates'][pid]
    assert p['gt_text']=='' and p['text_exclusion_reason']=='Nieczytelny tekst na tej klatce'
    assert core.summary(s)['cer'] is None
    hidden(root,h,ed)
    ed.clean_menu.invoke(1);ready(root,h,ed)
    from auto_annotation_tool.gui.mobile_canvas_navigator import select_layer
    assert select_layer(ed,'POSE','GT:a0');ready(root,h,ed)
    native.click(h,ed,(225,175),button=3);ready(root,h,ed)
    assert partial.progress(core.scene(s,gid))['object_decisions']['GT:a0']==partial.NOT_PLATE
    counts=core.counts(core.scene(s,gid))
    assert counts['false_positive']==1 and counts['missed']==1 and counts['linked']==0
    saved=stacks.snapshot(s,gid)['plates'][pid]
    assert all(saved[k]==v for k,v in p.items())
    p=saved  # Existing FP action also records the distinct physical identity.
    clean.show_exclusion(ed);ed.clean_reason.set('')
    before=s.sidecar_path.read_bytes();assert not clean.save_exclusion(ed)
    assert s.sidecar_path.read_bytes()==before
    ed.clean_reason.set('Prywatny test: nie można ocenić klatki')
    assert clean.save_exclusion(ed);ready(root,h,ed)
    assert core.scene(s,gid)['exclusion_reason']=='Prywatny test: nie można ocenić klatki'
    assert stacks.snapshot(s,gid)['plates'][pid]==p


def test_k02_actual_source_draw_mt_na_resume_export_keeps_a3(tmp_path,monkeypatch):
    archive=os.environ.get('ALPR_OBSERVATION_K02');mapping=os.environ.get('ALPR_A3_A2')
    if not archive or not mapping:pytest.skip('Private K02 and read-only A2 fixtures required')
    original_sha=sha(archive);mapping_sha=sha(mapping)
    dest=tmp_path/'source.alprsession';shutil.copyfile(archive,dest)
    s=MobileReviewSession.open(dest,sidecar_path=tmp_path/'review.json')
    ref=a3.Reference(mapping);gt_sha=sha(ref.gt_path);ref.attach(s)
    exposure=next(e for e in ref.exposures.values() if e['session_id']==s.session_id)
    a3.decide_identity(s,exposure['exposure_id'],'CONFIRMED')  # private pre-existing layer
    a3_before=copy.deepcopy(s.review.provenance[a3.KEY])
    s=MobileReviewSession.open(dest,sidecar_path=s.sidecar_path)
    assert not getattr(s,'_f5a_reference',None)
    # A persisted registry must never force A3 on ordinary opening.
    monkeypatch.setattr(a3,'registered',lambda *a,**kw:pytest.fail('A3 registry read in ordinary mode'))
    monkeypatch.setattr(native,'source_session',lambda _:s)
    generator=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,s=next(generator)
    try:
        assert s.session_id=='exp-1794265037188-43dc06b8'
        assert not getattr(h,'_f5a_a3',None)
        assert len(s.mt_invocations)==44 and len(s.scene_frames)==10
        assert s.review.provenance[a3.KEY]==a3_before
        hidden(root,h,ed)
        from PIL import ImageGrab
        ImageGrab.grab(window=h.window.winfo_id()).save(tmp_path/'k02_hidden.png')
        gid=next(g for g,u in s.mt_invocations.items() if u.executed and not u.cancelled
            and not u.execution_failed and core.context(s,g) and any(r['mt_status']=='VALID_QUAD' for r in u.records))
        aid=s.mt_invocations[gid].records[0]['id']
        h._multi_focus_attempt(aid);h._select_record()
        native.pump(root,lambda:h.current_invocation==gid and h._image_entry==s.attempts[aid]['source_frame_entry'])
        assert h._image.size==(960,1280)
        assert core.scene(s,gid)['roi']==[291.0,477.0,532.0,590.0]
        source=copy.deepcopy(s.attempts);parents=copy.deepcopy(core.scene(s,gid)['parents'])
        hidden(root,h,ed)
        # Source pixels inspected separately. This is not accepted plate GT.
        pid=draw(h,ed,[(476,542),(494,540),(495,550),(477,552)])
        unreadable(root,ed,monkeypatch);ready(root,h,ed)
        hidden(root,h,ed)
        ed.clean_menu.invoke(1);ready(root,h,ed)
        row=next(r for r in clean.records(ed) if r['candidate'])
        assert row['candidate']['mz_prediction']==s.attempts[aid]['prediction']=='H'
        ed._choose_plate(row['oid']);native.key(h,'space');ready(root,h,ed)
        assert partial.progress(core.scene(s,gid))['object_decisions'][row['oid']]==partial.PLATE, (h.status_var.get(),ed.selected,ed.gt_focus_id,clean.selected(ed))
        assert core.summary(s)['evaluated_reads']==0 and core.summary(s)['cer'] is None
        assert s.review.provenance[a3.KEY]==a3_before
        ImageGrab.grab(window=h.window.winfo_id()).save(tmp_path/'k02_review.png')
        # Next actual MT has its own image binding and no copied polygon.
        next_gid=next(g for g,u in s.mt_invocations.items() if g!=gid and u.executed and not u.cancelled and core.context(s,g))
        next_aid=s.mt_invocations[next_gid].records[0]['id'];h._multi_focus_attempt(next_aid);h._select_record()
        native.pump(root,lambda:h.current_invocation==next_gid and h._image_entry==s.attempts[next_aid]['source_frame_entry'])
        hidden(root,h,ed);assert not stacks.snapshot(s,next_gid)['plates']
        assert s.attempts==source and core.scene(s,gid)['parents']==parents
        saved=s.review.to_dict();frozen=s.sidecar_path.read_bytes()
    finally:
        try:next(generator)
        except StopIteration:pass
    reopened=MobileReviewSession.open(dest,sidecar_path=s.sidecar_path)
    assert reopened.review.to_dict()==saved
    # Actually reopen the ordinary window, not just the JSON parser.
    monkeypatch.setattr(native,'source_session',lambda _:reopened)
    generator=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,reopened=next(generator)
    try:
        h._multi_focus_attempt(aid);h._select_record()
        native.pump(root,lambda:h.current_invocation==gid and h._image_entry==reopened.attempts[aid]['source_frame_entry'])
        hidden(root,h,ed)
        assert stacks.snapshot(reopened,gid)['plates'][pid]['text_exclusion_reason']=='Nieczytelny tekst na tej klatce'
        assert reopened.sidecar_path.read_bytes()==frozen
        output=tmp_path/'export';output.mkdir();paths=reopened.export(output)
        assert output/'canvas_review_plates.csv' in paths
        assert json.loads((output/'f5a_a3_operator_review.json').read_text(encoding='utf-8'))['layer']==a3_before
        assert reopened.sidecar_path.read_bytes()==frozen
        assert reopened.review.provenance[a3.KEY]==a3_before
    finally:
        try:next(generator)
        except StopIteration:pass
    assert sha(archive)==original_sha and sha(dest)==original_sha
    assert sha(mapping)==mapping_sha and sha(ref.gt_path)==gt_sha
