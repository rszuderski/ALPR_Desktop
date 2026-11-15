"""Material/GT/presentation adapters on synthetic images and native stores."""
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tkinter as tk

import pytest
from PIL import Image

from auto_annotation_tool.ranking import photo_material as material
from auto_annotation_tool.ranking import photo_reference as photo
from auto_annotation_tool.ranking import final180_reference as presentation
from auto_annotation_tool.ranking import final180_markers as markers
from auto_annotation_tool.gui import final180_tv as tv


@pytest.fixture
def selected(tmp_path, monkeypatch):
    source = tmp_path/'zdjęcia próbne'; source.mkdir()
    Image.new('RGB', (160, 100), '#996655').save(source/'ten sam.png')
    Image.new('RGB', (160, 100), '#112233').save(source/'bez tablic.png')
    nested = source/'podkatalog'; nested.mkdir()
    exif = Image.Exif(); exif[274] = 6
    Image.new('RGB', (160, 100), '#335511').save(nested/'ten sam.png', exif=exif)
    (source/'pomijany.txt').write_text('not an image', encoding='utf-8')
    monkeypatch.setenv('ALPR_MATERIALS_DIR', str(tmp_path/'sessions'))
    return material.DirectoryMaterial.select(source, recursive=True)


def quad(x=10):
    return [[x, 10], [x+45, 10], [x+45, 35], [x, 35]]


def test_directory_snapshot_exif_explicit_recursion_and_no_gt(selected):
    assert len(selected.order) == 3
    assert not selected.default_output.exists()
    reopened = material.DirectoryMaterial.select(selected.root, recursive=True)
    assert reopened.material_id == selected.material_id and reopened.order == selected.order
    shallow = material.DirectoryMaterial.select(selected.root, recursive=False)
    assert len(shallow.order) == 2 and shallow.material_id != selected.material_id
    oriented = next(row for row in selected.images.values() if row['exif_orientation'] == 6)
    assert (oriented['width'], oriented['height']) == (100, 160)
    assert selected.image(oriented['image_id']).size == (100, 160)
    assert selected.reference_status()['state'] == 'none'


def test_empty_directory_has_clear_rejection(tmp_path):
    folder = tmp_path/'empty'; folder.mkdir()
    with pytest.raises(ValueError, match='nie zawiera'):
        material.DirectoryMaterial.select(folder, store=tmp_path/'sessions')


def test_missing_image_is_listed_without_order_matching(selected,tmp_path):
    import shutil
    session=selected.reference(create=True);session.empty(selected.order[0])
    exported=tmp_path/'full_reference.json';material.export_reference(selected,exported)
    subset=tmp_path/'subset';subset.mkdir()
    for key in selected.order[:2]:
        relative=selected.images[key]['relative_path'];target=subset/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(selected.verify(key),target)
    small=material.DirectoryMaterial.select(subset,recursive=True)
    result=material.reference_compatibility(small,exported)
    assert len(result['mapping'])==2 and not result['compatible']
    assert any(r['kind']=='REFERENCE_IMAGE_NOT_MATCHED' for r in result['differences'])


def test_session_created_only_for_annotation_resume_multiple_plates_negative(selected):
    session = selected.reference(create=True)
    assert session.path.exists() and not session.data['reviews']
    keys = selected.order
    oid1 = session.add(keys[0], quad()); oid2 = session.add(keys[0], quad(75))
    session.edit(keys[0], oid1, text='TEST123')
    session.edit(keys[0], oid2, text='', text_state='UNREADABLE_TEXT', reason='Zasłonięty numer')
    session.empty(keys[1]); session.remember(keys[2])
    reopened = selected.reference(create=True)
    assert reopened.data == session.data
    assert len(reopened.state(keys[0])['objects']) == 2
    assert reopened.state(keys[1])['presence'] == 'NO_PLATES'
    assert reopened.state(keys[2])['presence'] == 'UNRESOLVED'
    assert selected.reference_status()['complete'] == 2


def test_show_without_gt_has_every_qr_no_session_and_preserves_sources(selected):
    before = {key:photo.digest(selected.verify(key)) for key in selected.order}
    prepared, bound = material.prepare_presentation(selected, screen=(1920,1080), dwell_seconds=2)
    doc, frozen, gt = presentation.load(prepared)
    assert gt is None and bound is None and not selected.default_output.exists()
    assert not (prepared/'reference.json').exists()
    assert doc['layout']['layout_version'] == markers.F3_LAYOUT
    assert len(doc['exposures']) == 3 and frozen.order == selected.order
    assert material.material_for_presentation(prepared).material_id == selected.material_id
    tokens = []
    run_id='0123456789abcdef0123456789abcdef'
    manifest_sha=photo.digest(prepared/'manifest.json')
    for exposure in doc['exposures']:
        token = markers.payload(doc, manifest_sha, run_id, exposure['exposure_id'], 'P')
        frame, placement = markers.render(frozen.image(exposure['image_id']), doc['layout'], token)
        assert frame.size == (1920,1080) and placement
        context=dict(binding=markers.f3_binding(doc,manifest_sha,run_id),observed_tokens=[token],technical_tokens=[])
        decoded=markers.decode(frame,doc,manifest_sha,run_id,run_context=context)
        assert decoded['status']=='PASS' and decoded['exposure_id']==exposure['exposure_id']
        assert markers.qualify_f3([token[0],''],doc,manifest_sha,run_id,context)['status']=='N/A'
        tokens.append(str(token))
    assert len(set(tokens)) == 3
    assert before == {key:photo.digest(selected.verify(key)) for key in selected.order}


def test_gt_added_after_show_is_native_override_without_mutating_old_plan(selected):
    prepared, _ = material.prepare_presentation(selected, screen=(1920,1080), dwell_seconds=2)
    original = {p:photo.digest(p) for p in prepared.rglob('*') if p.is_file()}
    session = selected.reference(create=True)
    oid = session.add(selected.order[0], quad()); session.edit(selected.order[0], oid, text='TEST123')
    session.empty(selected.order[1])
    bound = material.reference_for_presentation(selected, prepared)
    coverage = presentation.reference_preflight(prepared, reference_path=bound)
    assert coverage['mode'] == 'partial'
    assert coverage['counts']['GT_COMPLETE'] == 2 and coverage['counts']['GT_MISSING'] == 1
    assert presentation.reference_preflight(prepared)['mode'] == 'none'
    assert all(photo.digest(p) == sha for p,sha in original.items())
    mapped = presentation.load(prepared, reference_path=bound)[2]
    assert mapped.state(selected.order[0]) == session.state(selected.order[0])
    assert mapped.data['presentation_binding']['geometry'].endswith('NO_CAMERA_PROJECTION')


def test_reference_import_reports_byte_changes_new_missing_and_requires_decision(selected, tmp_path):
    session = selected.reference(create=True)
    session.empty(selected.order[0])
    source_gt = tmp_path/'original_gt.json'; material.export_reference(selected, source_gt)
    before = photo.digest(source_gt)
    first = selected.verify(selected.order[0])
    Image.new('RGB', (160,100), '#00ff00').save(first)
    added = selected.root/'nowe.png'; Image.new('RGB',(160,100),'#cc00cc').save(added)
    changed = material.DirectoryMaterial.select(selected.root, recursive=True)
    comparison = material.reference_compatibility(changed, source_gt)
    kinds = {row['kind'] for row in comparison['differences']}
    assert 'IMAGE_IDENTITY_OR_GEOMETRY_CONFLICT' in kinds and 'NEW_OR_AMBIGUOUS_IMAGE' in kinds
    with pytest.raises(ValueError, match='listę różnic'):
        material.import_reference(changed, source_gt)
    ref, report = material.import_reference(changed, source_gt, accept_partial=True, expected_sha=before)
    assert all(ref.state(k)['presence'] == 'UNRESOLVED' for k in changed.order)
    assert report['mapping'] and photo.digest(source_gt) == before
    with pytest.raises(ValueError, match='zmieniła się'):
        selected.verify(selected.order[0])


def test_import_preserves_assisted_origin_and_never_downgrades_it(selected, tmp_path):
    assisted_material = copy(selected); assisted_material.predictions_exposed = True
    assisted = photo.PhotoReference(assisted_material, tmp_path/'assisted.json')
    assisted.empty(selected.order[0])
    assert assisted.data['events'][-1]['predictions_exposed'] is True
    imported, _ = material.import_reference(selected, assisted.path)
    assert imported.predictions_exposed is True
    oid = imported.add(selected.order[1], quad())
    assert imported.data['events'][-1]['predictions_exposed'] is True
    review = imported.data['reviews'][selected.order[1]]
    assert review['versions'][review['timeline'][-1]]['predictions_exposed'] is True
    independent = photo.PhotoReference(selected, tmp_path/'independent.json')
    independent.empty(selected.order[2])
    imported2, _ = material.import_reference(selected, independent.path)
    assert imported2.predictions_exposed is True
    assert list((selected.directory/'reference_history').glob('*.json'))


def test_existing_canonical_gt_roundtrip_keeps_original_bytes(selected, tmp_path):
    session = selected.reference(create=True); session.empty(selected.order[0])
    exported = tmp_path/'export.json'; material.export_reference(selected, exported)
    before = exported.read_bytes()
    assert material.reference_compatibility(selected, exported)['compatible']
    imported, report = material.import_reference(selected, exported)
    assert imported.state(selected.order[0])['presence'] == 'NO_PLATES'
    assert len(report['mapping']) == 3 and exported.read_bytes() == before
    with pytest.raises(FileExistsError):
        material.export_reference(selected, exported)


def test_already_open_editor_keeps_future_corrections_assisted_after_results(selected, tmp_path):
    session=selected.reference(create=True)
    session.empty(selected.order[0])
    before=session.path.read_bytes()
    results=tmp_path/'results';results.mkdir()
    (results/'evaluation.json').write_text('{}',encoding='utf-8')
    material.note_predictions_exposed(selected,result_path=results)
    assert session.path.read_bytes()==before  # viewing alone never rewrites GT
    session.empty(selected.order[1])
    assert session.data['predictions_exposed'] is True
    assert session.data['annotation_protocol']==photo.ASSISTED_PROTOCOL
    versions=session.data['reviews'][selected.order[1]]
    assert versions['versions'][versions['timeline'][-1]]['predictions_exposed'] is True
    assert any(p.read_bytes()==before for p in session.path.parent.glob('reference.pre-context-*.json'))
    prepared,_=material.prepare_presentation(selected,screen=(1920,1080),dwell_seconds=2)
    bound=material.reference_for_presentation(selected,prepared)
    assert presentation.reference_preflight(prepared,reference_path=bound)['mode']=='partial'


def test_conflicting_gt_does_not_prevent_unannotated_presentation(selected):
    session=selected.reference(create=True);data=deepcopy(session.data)
    data['images'][selected.order[0]]['width']+=1
    session.path.write_text(json.dumps(data),encoding='utf-8')
    before=session.path.read_bytes()
    assert selected.reference_status()['state']=='invalid'
    prepared,bound=material.prepare_presentation(selected,screen=(1920,1080),dwell_seconds=2)
    assert presentation.display_ready(presentation.load(prepared)[0]) and bound is None
    assert session.path.read_bytes()==before


def test_native_mobile_analysis_later_gt_preserves_all_recorded_calls(selected, tmp_path):
    from test_final180_reference_aware import native_f2_archive
    from auto_annotation_tool.ranking import final180_mobile as mobile
    prepared,_=material.prepare_presentation(selected,screen=(1920,1080),dwell_seconds=2)
    doc,_,_=presentation.load(prepared)
    run=tv.PresentationRun(prepared,doc)
    archive=native_f2_archive(tmp_path,doc,run)
    run.present(0,{},lambda:None);run.stop(clear=lambda:None)
    originals={p:photo.digest(p) for p in prepared.rglob('*') if p.is_file()}
    archive_sha=photo.digest(archive)
    before=mobile.analyze_archive(prepared,archive,run.directory/'run.json')
    assert before['metrics_status']=='NOT_AVAILABLE_NO_GT'
    assert before['frames'][0]['identity_status']=='PASS'
    assert before['frames'][0]['exposure_id']=='e0001'
    reference=selected.reference(create=True)
    oid=reference.add(selected.order[0],quad());reference.edit(selected.order[0],oid,text='TEST123')
    bound=material.reference_for_presentation(selected,prepared)
    after=mobile.analyze_archive(prepared,archive,run.directory/'run.json',reference_path=bound)
    assert before['raw_mt']==after['raw_mt'] and before['raw_mz']==after['raw_mz']
    assert after['reference_preflight']['counts']['GT_MISSING']==2
    assert after['summary']['system_full_read']['rate'] is None
    exported=mobile.export_report(after,tmp_path/'analysis')
    assert (exported/'evaluation.json').is_file()
    assert photo.digest(archive)==archive_sha and all(photo.digest(p)==sha for p,sha in originals.items())


@pytest.mark.parametrize('payload',[b'PK\x03\x04broken',b'PKinvalid',b'not an archive',b''])
def test_broken_archive_stays_integrity_error(tmp_path,payload):
    from auto_annotation_tool.ranking.mobile_package_experiments import ReportBundleReader
    path=tmp_path/'broken.alprsession';path.write_bytes(payload)
    with pytest.raises(ValueError,match='integralno'):
        ReportBundleReader().read_many(path)


def test_packaged_viewer_has_no_private_script_dependency():
    core = tv.legacy_module(tv.LEGACY_DEFAULT)
    assert Path(core.__file__).parent == Path(tv.__file__).parent
    assert all(hasattr(core, name) for name in ('TvApp','choose_monitors','set_dpi_aware'))


def test_three_photo_show_native_callbacks_without_gt_or_external_viewer(selected,tmp_path,monkeypatch):
    import time
    from test_final180_final_blockers import app_fixture,close_app
    root,app,errors=app_fixture(tmp_path,monkeypatch)
    try:
        app.use_material(selected);app.dwell.set('0.1')
        app.begin_countdown()
        deadline=time.monotonic()+35
        while time.monotonic()<deadline and app.mode not in ('COMPLETED','ERROR'):
            root.update();time.sleep(.01)
        assert app.mode=='COMPLETED',app.status.get()
        assert app.run.next_index==3 and not selected.default_output.exists()
        assert not errors
        assert app.run.state['android_start']=='NOT_OBSERVED'
    finally:
        close_app(root,app)


def test_stop_during_preparation_never_starts_delayed_show(selected,tmp_path,monkeypatch):
    import threading,time
    from test_final180_final_blockers import app_fixture,close_app
    root,app,errors=app_fixture(tmp_path,monkeypatch)
    entered=threading.Event();released=threading.Event();finished=threading.Event()
    real_prepare=material.prepare_presentation
    def delayed(*args,**kwargs):
        entered.set()
        if not released.wait(10):raise RuntimeError('test did not release worker')
        try:return real_prepare(*args,**kwargs)
        finally:finished.set()
    monkeypatch.setattr(material,'prepare_presentation',delayed)
    try:
        app.use_material(selected);app.begin_countdown()
        assert entered.wait(5) and app.mode=='PREPARING'
        app.abort();released.set()
        deadline=time.monotonic()+15
        while time.monotonic()<deadline and not finished.is_set():root.update();time.sleep(.01)
        for _ in range(15):root.update();time.sleep(.02)
        assert finished.is_set() and app.run is None and app.viewer is None
        assert not list(selected.directory.glob('presentations/*/runs/*/run.json'))
        assert not errors
    finally:
        released.set();close_app(root,app)


def test_native_two_gt_entries_reuse_one_canvas_session_and_restore_selection(selected, monkeypatch):
    from auto_annotation_tool.gui.photo_material_ui import open_material_editor
    from auto_annotation_tool.gui.z4_mobile_report_browser import MobileReportBrowser
    from auto_annotation_tool.gui.mobile_canvas_editor import CanvasReviewEditor
    root = tk.Tk(); root.withdraw()
    monkeypatch.setattr(tv.monitor_dpi, 'final_monitors', lambda:[dict(name='TEST',x=0,y=0,w=1920,h=1080,verified=True,geometry_source='HOST_TEST')])
    args = SimpleNamespace(xml=None, positives=None, negatives=None, output=selected.directory/'plans',
        dwell=2, negatives_declared=False, unreadable_ids=[], unreadable_xml_sha=None)
    app = tv.make_app(tv.legacy_module(tv.LEGACY_DEFAULT), args, parent=root)
    app.window.withdraw()
    try:
        app.use_material(selected)
        assert app.recursive_photographs.get() is True
        assert not selected.default_output.exists()
        editor1 = app.edit_material_reference()
        editor1.window.withdraw(); root.update()
        assert isinstance(editor1._canvas_editor, CanvasReviewEditor)
        browser = SimpleNamespace(window=root, _final_presentation=app)
        editor2 = MobileReportBrowser.open_photo_reference(browser)
        assert editor2 is editor1
        editor1.session.empty(selected.order[0])
        editor1.select(selected.order[1]); root.update()
        reference_id = editor1.session.data['reference_id']
        editor1.close(); root.update()
        assert app.source_material.material_id == selected.material_id
        assert 'w opracowaniu' in app.reference_info.get()
        editor3 = open_material_editor(root, selected)
        assert editor3.session.data['reference_id'] == reference_id
        assert editor3.current_invocation == selected.order[1]
        editor3.close()
    finally:
        app._close(); root.destroy()
