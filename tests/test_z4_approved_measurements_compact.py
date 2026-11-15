"""Approved review != complete measurement coverage; concise Polish UI."""
from pathlib import Path
import copy,json,shutil
import pytest
OUT=Path(__file__).resolve().parents[1]/'output/Z4_APPROVED_MEASUREMENTS_COMPACT_PANEL_PL_20261030_001'
F='exp-1792699303016-c8b05efb';SHA='60771b9570f15d321076c90c2bdc3e7e0e2bb8b0debda6af2a07559040c5e77c'
from auto_annotation_tool.gui.mobile_measurement_availability import measurement_limits,polish_model_caption
from auto_annotation_tool.gui.mobile_compare_review_gate import imported_verification,inspect_canvas
from auto_annotation_tool.gui.mobile_report_canvas_view import load_canvas_snapshot,canvas_overview
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking import mobile_canvas_review as core

def inputs(tmp):
    diag=json.loads((OUT/'diagnosis.json').read_text(encoding='utf-8'));source=Path(diag['browser_snapshot']['verification']['export_dir'])
    root=tmp/'results';dest=root/F/source.name;shutil.copytree(source,dest)
    review=tmp/'reviews';side=review/'human_reviews'/(SHA+'.review.json');side.parent.mkdir(parents=True)
    shutil.copyfile(OUT/'private'/(F+'.review.json'),side)
    return root,review,dest,side

def test_actual_export_is_approved_with_named_limits_and_quality_gate_unchanged(tmp_path):
    root,review,dest,side=inputs(tmp_path);before={p:p.read_bytes() for p in dest.rglob('*') if p.is_file()}
    snapshot=load_canvas_snapshot(F,SHA,root,review);gate=snapshot['verification'];view=canvas_overview(snapshot,session_id=F,source_sha=SHA)
    assert gate['state']=='APPROVED_LIMITED' and 'Niepełna' not in gate['label']
    assert imported_verification(F,SHA,root,review).approved and not imported_verification(F,SHA,root,review).ready
    assert 'ZATWIERDZONA' in view['status'] and '21/21' in view['progress']
    assert 'Obraz 07' in view['progress_subtitle'] and 'Obraz 08' in view['progress_subtitle']
    assert 'BBOX' in view['progress_subtitle'] and 'Nie wszystkie obrazy' not in view['progress_subtitle']
    assert not inspect_canvas(snapshot['summary'],'COMPLETED')[0]
    assert before=={p:p.read_bytes() for p in before}

def test_newer_live_revision_cannot_keep_old_approved_display(tmp_path):
    root,review,dest,side=inputs(tmp_path);data=json.loads(side.read_text(encoding='utf-8'));data['review_revision']+=1;data['review_status']='IN_PROGRESS'
    side.write_text(json.dumps(data),encoding='utf-8')
    gate=imported_verification(F,SHA,root,review)
    assert gate.state=='STALE' and not gate.approved

def test_review_status_mismatch_is_invalid_even_if_manifest_is_resealed(tmp_path):
    from test_z4_akt89f_results_identity import reseal
    root,review,dest,side=inputs(tmp_path);p=dest/'review.json';data=json.loads(p.read_text(encoding='utf-8'));data['review_status']='IN_PROGRESS'
    p.write_text(json.dumps(data),encoding='utf-8');reseal(dest)
    gate=imported_verification(F,SHA,root,review)
    assert gate.state=='INVALID' and not gate.approved

def test_caption_translation_is_display_only():
    source='MT-261009-1053-4A2D + MZ-DAY CLEAN INT8 — research ONLY, no production import approval'
    value=polish_model_caption(source)
    assert value=='MT-261009-1053-4A2D + MZ-DAY CLEAN INT8 — wyłącznie do badań; import produkcyjny niezatwierdzony'
    assert 'research ONLY' in source and polish_model_caption('MT-ABC + MZ-123')=='MT-ABC + MZ-123'

def test_availability_does_not_change_any_metrics():
    d=json.loads((OUT/'diagnosis.json').read_text(encoding='utf-8'));before=copy.deepcopy(d)
    rows=measurement_limits(d['summary'],d['mt'])
    assert [r['label'] for r in rows]==['Obraz 07','Obraz 08'] and d==before

def test_pending_reference_explanation_has_image_label_without_internal_ID():
    rows=measurement_limits(dict(scenes=[dict(mt_invocation_id='g',status='PENDING',
        issues=['Canvas g: Zatwierdź nową lub zmienioną GTR: PGT:0123456789abcdef'])]))
    assert rows==[dict(gid='g',label='Obraz 01',reasons=['obrys operatora czeka na zatwierdzenie'])]

@pytest.fixture
def gui(tmp_path,monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    side=tmp_path/'review.json';shutil.copyfile(OUT/'private'/(F+'.review.json'),side)
    s=MobileReviewSession.open(OUT/'private/day.alprsession',sidecar_path=side)
    monkeypatch.setattr(native,'source_session',lambda _:s);cls=native.module.MobileSampleReviewWindow
    def shown(*a,**kw):
        h=cls(*a,**kw);h.window.state('normal');h.window.deiconify();return h
    monkeypatch.setattr(native.module,'MobileSampleReviewWindow',shown)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

def test_native_results_identify_and_open_each_unmeasurable_image_without_write(gui):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    root,h,ed,s=gui;ready(root,h,ed);before=s.sidecar_path.read_bytes();panel=ed.report_host
    assert 'Weryfikacja zatwierdzona' in panel.notice['text'] and 'niekompletną oceną' not in panel.notice['text']
    assert '21/21' in panel.progress['text'] and 'W toku' not in panel.mt_counts['text']
    for number in (7,8):
        gid=F+f'-a{number:08d}';assert gid in panel.limit_buttons
        ed.review_tabs.select(panel);panel.limit_buttons[gid].invoke()
        native.pump(root,lambda:h.current_invocation==gid and not h._saving);ready(root,h,ed)
        assert ed.review_tabs.index(ed.review_tabs.select())==0 and s.sidecar_path.read_bytes()==before

def test_native_sidebar_is_compact_and_details_toggle_is_read_only(gui):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    root,h,ed,s=gui;ready(root,h,ed);p=ed.contexts;before=s.sidecar_path.read_bytes()
    assert not p.object_details.winfo_ismapped()
    compact=p.sections['object'].winfo_height();p.details_button.invoke();native.pump(root)
    assert p.object_details.winfo_ismapped() and p.sections['object'].winfo_height()>compact+100
    p.details_button.invoke();native.pump(root)
    assert not p.object_details.winfo_ismapped() and s.sidecar_path.read_bytes()==before
    assert '20' in p.counters['gt']['font'] and '10' in p.fields['object']['font']
