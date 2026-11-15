"""F5A/21 original Android TV crop delivery; mutations/GT stay in private copies."""
from pathlib import Path
from _private_test_data import private_test_path
import copy
import hashlib
import json
import shutil
import zipfile
from dataclasses import replace

import pytest

from auto_annotation_tool.ranking.mobile_package_experiments import read_mobile_report_bundle
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking import mobile_tv_bbox_evidence as tv

PRODUCER = private_test_path('F5A_20_ANDROID_DOWODY_20261106')
PROOF = private_test_path('F5A_21_DESKTOP_TV_BBOX_DOWODY_20261107')
INVENTORY = {r['name']: r for r in json.loads((PRODUCER / 'fixture_inventory.json').read_text(encoding='utf-8'))}
REFERENCE = json.loads((PROOF / 'canonical_java_reference.json').read_text(encoding='utf-8'))


def original(name):
    filename = name if name.endswith('.alprsession') else name + '.alprsession'
    path = PRODUCER / 'fixtures' / filename
    assert path.is_file(), 'BLOCKED: original Android fixture missing'
    assert path.stat().st_size == INVENTORY[filename]['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == INVENTORY[filename]['sha256']
    return path


def session(tmp_path, name):
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'source.alprsession'
    shutil.copyfile(original(name), path)
    return MobileReviewSession.open(path, sidecar_path=folder / 'review.json')


@pytest.mark.parametrize('name', sorted(INVENTORY))
def test_original_android_native_import(name, tmp_path, request):
    path = original(name)
    bundle = read_mobile_report_bundle(path)
    expected = {'PASS': 'VALID', 'REJECT': 'INVALID'}.get(INVENTORY[name]['expected_android'], INVENTORY[name]['expected_android'])
    assert bundle.tv_bbox_evidence['status'] == expected, bundle.validation.errors
    assert bundle.validation.ok == (expected != 'INVALID')
    if expected == 'INVALID':
        assert any('TV BBOX evidence contract' in e for e in bundle.validation.errors)
    else:
        s = session(tmp_path, path.stem)
        assert s.tv_bbox_evidence == bundle.tv_bbox_evidence
        assert not s.review.scene_annotations and not s.review.invocation_annotations
    assert bundle.to_dict()['tv_bbox_evidence'] == bundle.tv_bbox_evidence
    request.node.user_properties.append(('fixture', dict(name=name, expected=expected,
        observed=bundle.tv_bbox_evidence['status'], sha256=INVENTORY[name]['sha256'], reasons=bundle.validation.errors,
        dependencies=len(bundle.tv_bbox_evidence.get('dependencies', {})))))


@pytest.mark.parametrize('reference', REFERENCE, ids=[r['name'] for r in REFERENCE])
def test_setup_json_canonical_matches_unchanged_java(reference):
    name = reference['name']
    if name.endswith('.alprsession'):
        with zipfile.ZipFile(original(name)) as archive:
            d = tv.loads(archive.read('session.json'))['tv_bbox_crop']
        assert tv.canonical(d) == reference['descriptor_canonical']
        assert tv.digest(d) == reference['descriptor_sha256'] == d['digest_sha256']
        c = d['calibration_snapshot']
        assert tv.canonical(c) == reference['calibration_canonical']
        assert tv.digest(c) == reference['calibration_sha256'] == c['digest_sha256']
    else:
        inputs = json.loads((PROOF / 'canonical_inputs.json').read_text(encoding='utf-8'))
        value = tv.loads(next(r['json_text'] for r in inputs if r['name'] == name))
        assert tv.canonical(value) == reference['canonical']
        assert tv.digest(value) == reference['sha256']


def mutation(tmp_path, edit):
    with zipfile.ZipFile(original('tv-valid-for-negatives')) as archive:
        payloads = {n: archive.read(n) for n in archive.namelist()}
    s, r, d = [json.loads(payloads[n]) for n in ('session.json', 'report.json', 'samples/schema.json')]
    rows = list(map(json.loads, payloads[tv.INDEX].splitlines()))
    edit(s, r, d, rows, payloads)
    for name, value in (('session.json', s), ('report.json', r), ('samples/schema.json', d)):
        payloads[name] = json.dumps(value, ensure_ascii=False).encode('utf-8')
    payloads[tv.INDEX] = ('\n'.join(json.dumps(row) for row in rows) + '\n').encode()
    manifest = json.loads(payloads['manifest.json'])
    manifest['entry_sha256'] = {n: hashlib.sha256(b).hexdigest() for n, b in payloads.items() if n != 'manifest.json'}
    payloads['manifest.json'] = json.dumps(manifest).encode()
    path = tmp_path / 'synthetic-contract-mutation.alprsession'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
    return path


def install_snapshot(s, r, snapshot, *, seal_calibration=True):
    if seal_calibration:
        snapshot['calibration_snapshot']['digest_sha256'] = tv.digest(snapshot['calibration_snapshot'])
    snapshot['calibration_digest_sha256'] = snapshot['calibration_snapshot']['digest_sha256']
    snapshot['digest_sha256'] = tv.digest(snapshot)
    s['tv_bbox_crop'] = copy.deepcopy(snapshot)
    s['execution']['tv_bbox_crop'] = copy.deepcopy(snapshot)
    r['research_collection']['tv_bbox_crop'] = copy.deepcopy(snapshot)
    r['research_collection']['execution']['tv_bbox_crop'] = copy.deepcopy(snapshot)


@pytest.mark.parametrize('which', ['descriptor', 'nested-calibration'])
def test_independent_digests_not_bypassed(tmp_path, which):
    def edit(s, r, d, rows, payloads):
        frozen = s['tv_bbox_crop']
        if which == 'descriptor':
            frozen['offset_to_camera_full']['x'] += 1
        else:
            frozen['calibration_snapshot']['distance_estimate']['reason'] = 'changed without resealing calibration'
            install_snapshot(s, r, frozen, seal_calibration=False)
    bundle = read_mobile_report_bundle(mutation(tmp_path, edit))
    assert bundle.tv_bbox_evidence['status'] == 'INVALID'
    assert ('Descriptor digest' if which == 'descriptor' else 'Calibration digest') in bundle.tv_bbox_evidence['reason']


@pytest.mark.parametrize('case,reason', [
    ('transform', 'TV transform'), ('homography', 'TV transform'), ('space', 'coordinate spaces'),
    ('bbox', 'BBOX floor/ceil'), ('bbox-type', 'Integer dimension'), ('offset', 'BBOX offset'),
    ('crop-size', 'Crop dimensions'), ('draft', 'Approved calibration'), ('winding', 'Convex'),
    ('duplicate-corner', 'Convex'), ('camera', 'camera_setup'), ('geometry', 'crop/rotation'),
    ('stale-geometry', 'Stale calibration'), ('autozoom', 'AZ OFF'), ('start', 'START freeze'),
])
def test_frozen_semantics_with_valid_outer_manifest_and_seals(tmp_path, case, reason):
    def edit(s, r, d, rows, payloads):
        frozen = s['tv_bbox_crop']
        if case == 'transform': frozen['processing_transform'] = 'PERSPECTIVE'
        elif case == 'homography': frozen['H_TV'] = 'APPLIED'
        elif case == 'space': frozen['source_coordinate_space'] = tv.LOCAL_SPACE
        elif case == 'bbox': frozen['tv_bbox_px']['right'] += 1
        elif case == 'bbox-type': frozen['tv_bbox_px']['left'] = 80.0
        elif case == 'offset': frozen['offset_to_camera_full']['x'] += 1
        elif case == 'crop-size': frozen['input_width_px'] += 1
        elif case == 'draft': frozen['calibration_snapshot']['status'] = 'DRAFT'
        elif case in ('winding', 'duplicate-corner'):
            corners = frozen['corners_norm']
            corners[:] = list(reversed(corners)) if case == 'winding' else [corners[0], corners[0], *corners[2:]]
            frozen['calibration_snapshot']['corners_norm'] = copy.deepcopy(corners)
        elif case == 'camera': frozen['camera_setup']['zoom_ratio'] = 2
        elif case == 'geometry': frozen['source_geometry']['crop_rect_px'][2] -= 1
        elif case == 'stale-geometry': frozen['calibration_snapshot']['source_geometry']['camera_id'] = 'foreign'
        elif case == 'autozoom':
            s['execution']['feature_flags']['autozoom'] = True
            r['research_collection']['execution']['feature_flags']['autozoom'] = True
        elif case == 'start':
            s['tv_bbox_crop_frozen_at_utc'] = '2000-01-01T00:00:00Z'
            r['research_collection']['tv_bbox_crop_frozen_at_utc'] = s['tv_bbox_crop_frozen_at_utc']
        install_snapshot(s, r, frozen)
    result = read_mobile_report_bundle(mutation(tmp_path, edit))
    assert result.tv_bbox_evidence['status'] == 'INVALID'
    assert reason in result.tv_bbox_evidence['reason'], result.validation.errors


@pytest.mark.parametrize('field,value', [('source_timestamp_nanos', 9007199254740994.0),
    ('source_timestamp_nanos', 9007199254740995), ('source_sequence', True), ('source_timestamp_nanos', 2**63)])
def test_exact_source_stamp_binding(tmp_path, field, value):
    def edit(s, r, d, rows, payloads): rows[0]['source_stamp'][field] = value
    assert read_mobile_report_bundle(mutation(tmp_path, edit)).tv_bbox_evidence['status'] == 'INVALID'


@pytest.mark.parametrize('field', ['input_scale', 'input_pad_x'])
def test_captured_inverse_letterbox_metadata_must_match_roi(tmp_path, field):
    def edit(s, r, d, rows, payloads):
        attempts = list(map(json.loads, payloads['samples/attempts.jsonl'].splitlines()))
        attempts[0][field] += .25
        payloads['samples/attempts.jsonl'] = ('\n'.join(json.dumps(v) for v in attempts)+'\n').encode()
    result = read_mobile_report_bundle(mutation(tmp_path, edit))
    assert result.tv_bbox_evidence['status'] == 'INVALID'
    assert 'Inverse-letterbox metadata conflict' in result.tv_bbox_evidence['reason']


@pytest.mark.parametrize('case', ['capability-number', 'capability-false', 'scene', 'captured', 'exact-image', 'summary', 'duplicate'])
def test_status_identity_and_pending_fail_closed(tmp_path, case):
    def edit(s, r, d, rows, payloads):
        if case == 'capability-number': d['capabilities'][tv.CAPABILITY] = 1
        elif case == 'capability-false': d['capabilities'][tv.CAPABILITY] = False
        elif case == 'scene': rows[0]['scene_event_id'] = 'foreign-scene'
        elif case == 'captured': rows[0]['crop_image']['status'] = 'CAPTURED'
        elif case == 'exact-image': rows[0]['crop_image']['exact_model_input'] = True
        elif case == 'duplicate': rows.append(copy.deepcopy(rows[0]))
        else:
            s['tv_bbox_input_evidence']['images_written'] += 1
            r['research_collection']['tv_bbox_input_evidence']['images_written'] += 1
    assert read_mobile_report_bundle(mutation(tmp_path, edit)).tv_bbox_evidence['status'] == 'INVALID'


@pytest.mark.parametrize('image', ['source_image', 'crop_image'])
def test_missing_known_dimensions_cannot_contradict_snapshot(tmp_path, image):
    def edit(s, r, d, rows, payloads):
        rows[0][image].update(status='MISSING', entry=None, sha256=None, missing_reason='synthetic_loss')
        rows[0][image]['width_px'] += 1
        for owner in (s, r['research_collection']):
            owner['tv_bbox_input_evidence'].update(images_written=1, images_missing=1, complete=False)
    bundle = read_mobile_report_bundle(mutation(tmp_path, edit))
    assert bundle.tv_bbox_evidence['status'] == 'INVALID'
    assert 'Missing image dimensions conflict' in bundle.tv_bbox_evidence['reason']


@pytest.mark.parametrize('w,h,bw,bh,crop,rotation,pw,ph,box', [
    (640, 400, 640, 400, [0, 0, 640, 400], 0, 800, 600, [80, 60, 560, 341]),
    (432, 656, 720, 480, [32, 24, 688, 456], 90, 1280, 720, [54, 98, 378, 558]),
    (400, 640, 640, 400, [0, 0, 640, 400], 90, 1024, 768, [50, 96, 350, 545]),
])
def test_source_crop_rotation_and_aspect_ratios(w, h, bw, bh, crop, rotation, pw, ph, box):
    with zipfile.ZipFile(original('tv-r0')) as archive:
        d = tv.loads(archive.read('session.json'))['tv_bbox_crop']
    g = d['source_geometry']
    scale = min(pw/w, ph/h)
    dx, dy = (pw-scale*w)/2, (ph-scale*h)/2
    g.update(source_image_width=w, source_image_height=h, buffer_width=bw, buffer_height=bh,
        crop_rect_px=crop, rotation_degrees=rotation, display_rotation_degrees=rotation,
        preview_width=pw, preview_height=ph, source_to_view_matrix=[scale, 0, dx, 0, scale, dy, 0, 0, 1],
        usable_view_rect=[dx, dy, dx+scale*w, dy+scale*h])
    d['calibration_snapshot']['source_geometry'] = copy.deepcopy(g)
    d['calibration_snapshot']['digest_sha256'] = tv.digest(d['calibration_snapshot'])
    d.update(source_width_px=w, source_height_px=h, tv_bbox_px=dict(zip(('left', 'top', 'right', 'bottom'), box)),
        input_width_px=box[2]-box[0], input_height_px=box[3]-box[1], offset_to_camera_full=dict(x=box[0], y=box[1]),
        calibration_digest_sha256=d['calibration_snapshot']['digest_sha256'])
    d['digest_sha256'] = tv.digest(d)
    assert tv.validate_descriptor(d)['tv_bbox_px'] == dict(zip(('left', 'top', 'right', 'bottom'), box))


def test_zero_roi_fallback_partial_and_no_auto_decisions(tmp_path):
    zero = session(tmp_path, 'tv-zero-roi')
    assert zero.tv_bbox_evidence['rows'][0]['attempt_links'] == []
    assert not zero.attempts and not zero.mt_invocations
    assert not zero.review.scene_annotations
    fallback = session(tmp_path, 'tv-r2-fallback-multiple')
    row = fallback.tv_bbox_evidence['rows'][0]
    assert len(row['attempt_links']) == len(fallback.attempts) == 4 and len(fallback.mt_invocations) == 3
    last = row['attempt_links'][-1]
    assert list(last['roi_local_xy'].values()) == [0, 0, 480, 281]
    assert fallback.tv_bbox_evidence['descriptor']['tv_bbox_px'] == dict(left=80, top=60, right=560, bottom=341)
    assert row['source_stamp']['source_timestamp_nanos'] == 9007199254740994
    missing = session(tmp_path, 'tv-source-unavailable')
    assert missing.tv_bbox_evidence['status'] == 'PARTIAL'
    assert all(missing.tv_bbox_evidence['rows'][0][k]['status'] == 'MISSING' for k in ('source_image', 'crop_image'))
    assert not missing.review.scene_annotations and not missing.review.invocation_annotations
    interrupted = session(tmp_path, 'tv-interrupted')
    assert interrupted.tv_bbox_evidence['summary']['pending_loss_unknown'] is True
    assert interrupted.tv_bbox_evidence['status'] == 'PARTIAL'


def test_native_comparison_keeps_input_modes_and_frozen_context_separate(tmp_path):
    from auto_annotation_tool.gui.mobile_import_comparison import record_from_import
    from auto_annotation_tool.gui.mobile_comparison_core import validate_selection
    from auto_annotation_tool.gui.mobile_comparison_scope import validate_comparison
    a, b = session(tmp_path, 'tv-r0'), session(tmp_path, 'legacy-full-frame')
    records = [record_from_import(s.bundle.report, s.bundle, include_review=False)[0] for s in (a, b)]
    assert [r.input_mode for r in records] == [tv.MODE, tv.FULL]
    for guard in (validate_selection, validate_comparison):
        with pytest.raises(ValueError, match='INPUT_MODE_COHORT_MISMATCH'): guard(records)
    base = records[0].input_context
    for field in ('descriptor_sha256', 'series_id', 'models'):
        changed = copy.deepcopy(base)
        if field == 'models': changed['models']['mt']['model_fingerprint'] = 'different'
        else: changed[field] = 'different'
        assert tv.cohort_conflict([base, changed]) == 'TV_BBOX_FROZEN_INPUT_COHORT_MISMATCH'
    r2 = session(tmp_path, 'tv-r2-fallback-multiple')
    assert tv.cohort_conflict([base, tv.input_context(r2.bundle.collection_session, r2.tv_bbox_evidence)]) is None


def test_native_analysis_mixed_input_modes_returns_na(tmp_path, monkeypatch):
    from auto_annotation_tool.ranking import mobile_package_experiments as package
    from auto_annotation_tool.gui.mobile_analysis import make_input, analyze
    # Freeze the legacy fallback import clock only; IDs and source bytes stay intact.
    monkeypatch.setattr(package, '_utc_now_iso', lambda: '2026-11-07T00:00:00Z')
    sources = [session(tmp_path, n) for n in ('tv-r0', 'legacy-full-frame')]
    inputs = [make_input(s.bundle.report, source_path=s.path, results_root=tmp_path/'03_wyniki', review_root=tmp_path/'reviews') for s in sources]
    result = analyze(inputs, question='DIAGNOSTIC')
    assert len(result.sessions) == 2
    for s in result.sessions:
        assert s.source_status == 'VERIFIED', s.source_reason
        assert all(m.status == 'NOT_COMPARABLE' and m.value is None and 'INPUT_MODE_COHORT_MISMATCH' in m.missing_reasons for m in s.metrics.values())


@pytest.mark.parametrize('name,status', [('tv-r0', 'VALID'), ('tv-r2-fallback-multiple', 'VALID'),
    ('tv-interrupted', 'PARTIAL'), ('legacy-full-frame', 'ABSENT')])
def test_T19_full_native_roundtrip(tmp_path, monkeypatch, name, status, request):
    import test_mobile_p2b2_canvas_gui as native
    from auto_annotation_tool.gui.mobile_comparison_core import load_session_export, InvalidExport
    s = session(tmp_path, name)
    s.set_review_mode('assisted')
    before = json.loads(json.dumps(s.review.to_dict()))
    reopened = MobileReviewSession.open(s.path, sidecar_path=s.sidecar_path)
    assert reopened.review.to_dict() == before
    monkeypatch.setattr(native, 'source_session', lambda _: reopened)
    generator = native.gui.__wrapped__(tmp_path, monkeypatch)
    root, h, ed, s = next(generator)
    try:
        h.window.title('F5A21 SYNTHETIC TV BBOX ACCEPTANCE - DEVICE_RUN=NOT_RUN')
        assert h._image_source_size == (640, 400)
        assert h._image_entry in {r['source_frame_entry'] for r in s.scene_frames.values()}
        if status != 'ABSENT':
            native.pump(root, lambda: bool(h.canvas.find_withtag('tv_bbox_domain')))
            assert tv.MODE in h.conditions_view.get('1.0', 'end')
            assert 'H_TV=NOT_APPLIED' in h.conditions_view.get('1.0', 'end')
        out = tmp_path/'03_wyniki'
        out.mkdir()
        errors = []
        monkeypatch.setattr(native.module.messagebox, 'showerror', lambda *a, **kw: errors.append(a))
        monkeypatch.setattr(native.module.filedialog, 'askdirectory', lambda **kw: str(out))
        monkeypatch.setattr(native.module.messagebox, 'askyesno', lambda *a, **kw: True)
        h.export()
        native.pump(root, lambda: not h._saving, seconds=90)
        assert not errors, errors
        manifests = list(out.rglob('export_manifest.json'))
        assert len(manifests) == 1, h.status_var.get()
        folder = manifests[0].parent
        loaded = load_session_export(folder)
        assert loaded.tv_bbox_evidence == s.tv_bbox_evidence
        assert loaded.tv_bbox_evidence['status'] == status
        final = MobileReviewSession.open(s.path, sidecar_path=folder/'review.json')
        assert final.review.to_dict() == before and final.tv_bbox_evidence == s.tv_bbox_evidence
        assert not final.review.scene_annotations and not final.review.invocation_annotations
        manifest = json.loads(manifests[0].read_text(encoding='utf-8'))
        for dep in manifest['files']:
            assert hashlib.sha256((folder/dep['path']).read_bytes()).hexdigest() == dep['sha256']
        assert hashlib.sha256(s.path.read_bytes()).hexdigest() == s.bundle.source_archive_sha256
        if status == 'ABSENT':
            assert not (folder/tv.EXPORT_FILE).exists() and not (folder/tv.NAMESPACE).exists()
            assert loaded.input_mode == tv.FULL
        else:
            assert loaded.input_mode == tv.MODE and loaded.input_context['offset_to_camera_full'] == dict(x=80, y=60)
        request.node.user_properties.append(('roundtrip', dict(fixture=name, status=status,
            source_sha256=s.bundle.source_archive_sha256, manifest_sha256=hashlib.sha256(manifests[0].read_bytes()).hexdigest(),
            dependencies=len(manifest['files']), tv_dependencies=len(s.tv_bbox_evidence.get('dependencies', {})),
            full_native_export=True, source_int64=[r['source_stamp']['source_timestamp_nanos'] for r in s.tv_bbox_evidence.get('rows', [])], DEVICE_RUN='NOT_RUN')))
        if name == 'tv-r0':
            # Correct outer hashes must not bypass the immutable payload contract.
            payload = folder/tv.NAMESPACE/s.tv_bbox_evidence['rows'][0]['crop_image']['entry']
            saved_payload, saved_manifest = payload.read_bytes(), manifests[0].read_bytes()
            payload.write_bytes(saved_payload[:-1]+bytes([saved_payload[-1] ^ 1]))
            for dep in manifest['files']:
                if dep['path'] == payload.relative_to(folder).as_posix(): dep['sha256'] = hashlib.sha256(payload.read_bytes()).hexdigest()
            manifests[0].write_text(json.dumps(manifest), encoding='utf-8')
            with pytest.raises(InvalidExport, match='TV dependency SHA'): load_session_export(folder)
            payload.write_bytes(saved_payload)
            manifests[0].write_bytes(saved_manifest)
            # Offset/ROI contradiction with ALL outer, dependency, and source-manifest hashes valid.
            changed = {}
            for entry in (tv.INDEX, 'samples/attempts.jsonl'):
                path = folder/tv.NAMESPACE/entry
                changed[path] = path.read_bytes()
                values = list(map(json.loads, changed[path].splitlines()))
                if entry == tv.INDEX: values[0]['attempt_links'][0]['roi_local_xy']['left'] += 1
                else: values[0]['tv_bbox_roi_local_xy']['left'] += 1
                path.write_text('\n'.join(json.dumps(v) for v in values)+'\n', encoding='utf-8')
            source_manifest = folder/tv.NAMESPACE/'manifest.json'
            descriptor = folder/tv.EXPORT_FILE
            changed[source_manifest], changed[descriptor] = source_manifest.read_bytes(), descriptor.read_bytes()
            sm, desc = json.loads(changed[source_manifest]), json.loads(changed[descriptor])
            for path in list(changed):
                if path not in (source_manifest, descriptor): sm['entry_sha256'][path.relative_to(folder/tv.NAMESPACE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            source_manifest.write_text(json.dumps(sm), encoding='utf-8')
            for path in changed:
                if path != descriptor:
                    desc['dependencies'][path.relative_to(folder).as_posix()] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            descriptor.write_text(json.dumps(desc), encoding='utf-8')
            mf = json.loads(saved_manifest)
            for dep in mf['files']:
                path = folder/dep['path']
                if path in changed: dep.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            manifests[0].write_text(json.dumps(mf), encoding='utf-8')
            with pytest.raises(InvalidExport, match='ROI translation'): load_session_export(folder)
            for path, data in changed.items(): path.write_bytes(data)
            manifests[0].write_bytes(saved_manifest)
            assert load_session_export(folder).tv_bbox_evidence == s.tv_bbox_evidence
    finally:
        try: next(generator)
        except StopIteration: pass


def test_existing_canvas_full_gt_and_read_only_crop(tmp_path, monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    from auto_annotation_tool.ranking import mobile_scene_frame_review as frames
    s = session(tmp_path, 'tv-zero-roi')
    s.set_review_mode('assisted')
    monkeypatch.setattr(native, 'source_session', lambda _: s)
    generator = native.gui.__wrapped__(tmp_path, monkeypatch)
    root, h, ed, s = next(generator)
    try:
        key = next(iter(s.scene_frames))
        assert h._image_source_size == (640, 400) and ed.available()
        assert not s.review.scene_annotations and not s.attempts
        assert any('Wycinek TV' in v for v in h.image_choice.cget('values'))
        native.key(h, 'space')
        assert frames.scene(s, key)['presence'] == 'NO_PLATES'
        ed.undo()
        native.key(h, 'd')
        for point in ((15, 15), (55, 15), (55, 40), (15, 40)): native.click(h, ed, point)
        ed.escape()
        objects = frames.scene(s, key)['objects']
        assert len(objects) == 1 and all(0 <= x < 80 and 0 <= y < 60 for x, y in objects[0]['quad'])
        assert not s.attempts and not s.mt_invocations
        original_review = copy.deepcopy(s.review.to_dict())
        h.image_choice.current(1)
        h._load_image()
        native.pump(root, lambda: h._image_entry == s.tv_bbox_evidence['rows'][0]['crop_image']['entry'] and h._image is not None)
        assert h._image_source_size == (480, 281) and not ed.available()
        assert s.review.to_dict() == original_review
        h.image_choice.current(0)
        h._load_image()
        native.pump(root, lambda: h._image_source_size == (640, 400) and ed.available())
        assert frames.scene(s, key)['objects'][0]['quad'] == objects[0]['quad']
    finally:
        try: next(generator)
        except StopIteration: pass


def test_context_only_stays_read_only_even_when_shared_image_matches_full_camera():
    from types import SimpleNamespace
    from auto_annotation_tool.gui.mobile_canvas_editor import CanvasReviewEditor
    messages = []
    row = dict(context_only=True)
    h = SimpleNamespace(_saving=False, _image=object(), _image_entry='same-image', _image_source_size=(640, 400),
        _image_rows=[('context', row)], image_choice=SimpleNamespace(current=lambda: 0))
    ed = SimpleNamespace(h=h, busy=False, _read=lambda: dict(image_entry='same-image', image_size=(640, 400)), message=messages.append)
    assert CanvasReviewEditor.available(ed) is False and 'Wycinek TV' in messages[-1]
    row['context_only'] = False
    assert CanvasReviewEditor.available(ed) is True
