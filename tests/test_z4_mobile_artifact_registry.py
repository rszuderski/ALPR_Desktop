
"""No GUI, no real-model conversion, and no mutation of archived packages."""
import hashlib
import json
from pathlib import Path
import zipfile

from auto_annotation_tool.gui import z4_mobile_artifact_registry as registry


def make_package(root, name, manifest):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as file:
        file.writestr('manifest.json', json.dumps(manifest))
    return path


def single_model(role='plate', precision='int8'):
    return {'schema': 'alpr.model.v1', 'role': role, 'model_id': 'TEST',
            'variants': [{'runtime': 'tflite', 'precision': precision}]}


def complete_package():
    return {'schema': 'alpr.package.v1', 'package_id': 'SET',
            'models': {'plate': {'variants': [{'runtime': 'tflite', 'precision': 'int8'}]},
                       'character': {'variants': [{'runtime': 'tflite', 'precision': 'fp32'}]}}}


def test_read_archive_single_full_and_invalid(tmp_path):
    root = tmp_path / 'models'
    mt = make_package(root, 'plate/one.alprmodel', single_model())
    mz = make_package(root, 'char/char.alprmodel', single_model('character','fp32'))
    both = make_package(root, 'alpr/complete.alprmodel', complete_package())
    bad = make_package(root, 'alpr/bad.alprmodel', {'schema': 'other'})
    rows, truncated = registry.list_managed_packages(root)
    assert not truncated and len(rows) == 4
    results = {r.path: r for r in rows}
    assert results[str(mt.resolve())].role == 'MT'
    assert results[str(mz.resolve())].role == 'MZ'
    assert results[str(both.resolve())].role == 'MT+MZ'
    assert results[str(both.resolve())].kind == 'PACKAGE'
    assert results[str(bad.resolve())].state == 'MANIFEST_ERROR'
    assert all(r.state != 'APPROVED' for r in rows)


def test_no_historical_events_inferred(tmp_path):
    root = tmp_path / 'managed'
    make_package(root,'plate/historical.alprmodel',single_model())
    rows, events, note = registry.library_snapshot(root, tmp_path)
    assert len(rows)==1 and events==[] and 'Starsze pliki' in note
    assert not registry.history_file(root).exists()


def test_event_roundtrip_and_artifact_sha(tmp_path):
    root=tmp_path/'managed'
    a=make_package(root,'plate/m.alprmodel',single_model())
    operation='Z4-20261014-TEST'
    registry.append_event(root,operation_id=operation,operation='SINGLE_MODEL_EXPORT',outcome='STARTED')
    event=registry.append_event(root,operation_id=operation,operation='SINGLE_MODEL_EXPORT',
                                outcome='SUCCEEDED',artifact_path=str(a),detail='F4')
    events,note=registry.load_history(root)
    assert not note and len(events)==2
    assert events[1]['artifact_sha256']==registry.sha256_file(a)
    assert event['schema']==registry.EVENT_SCHEMA
    rows,listed_events,_=registry.library_snapshot(root,tmp_path)
    assert len(rows)==1 and len(listed_events)==2


def test_failed_export_does_not_claim_model(tmp_path):
    root=tmp_path/'managed'
    registry.append_event(root,operation_id='Z4-20261014-ERROR',operation='COMPLETE_PACKAGE_EXPORT',
                          outcome='FAILED',detail='export failed')
    events,_=registry.load_history(root)
    assert len(events)==1 and events[0]['artifact_sha256'] is None


def test_unrelated_directories_and_truncation(tmp_path):
    root=tmp_path/'managed'
    make_package(root,'plate/1.alprmodel',single_model())
    make_package(root,'plate/2.alprmodel',single_model())
    make_package(tmp_path/'unrelated','anything.alprmodel',single_model())
    rows,truncated=registry.list_managed_packages(root,max_files=1)
    assert len(rows)==1 and truncated


def test_manifest_oversized_and_corrupt(tmp_path):
    root=tmp_path/'managed'
    root.mkdir()
    bad=root/'broken.alprmodel'
    bad.write_bytes(b'not-a-valid-zip')
    big=root/'big.alprmodel'
    with zipfile.ZipFile(big,'w',compression=zipfile.ZIP_DEFLATED) as arc:
        arc.writestr('manifest.json',b'A'*(registry.MANIFEST_CAP+1))
    rows,_=registry.list_managed_packages(root)
    assert len(rows)==2
    assert all(row.state=='MANIFEST_ERROR' for row in rows)


def test_research_candidate_does_not_claim_package(tmp_path,monkeypatch):
    repo=tmp_path
    path=repo/registry.RESEARCH_REL
    path.parent.mkdir(parents=True)
    path.write_bytes(b'research-artifact')
    row=registry.research_candidate(repo)
    assert row is not None and row.kind=='RESEARCH_TFLITE'
    assert row.state=='RESEARCH_SHA_MISMATCH'


def test_no_mutation_of_existing_packages(tmp_path):
    root=tmp_path/'managed'
    a=make_package(root,'plate/orig.alprmodel',single_model())
    before=hashlib.sha256(a.read_bytes()).hexdigest()
    registry.library_snapshot(root,tmp_path)
    assert hashlib.sha256(a.read_bytes()).hexdigest()==before
    assert not registry.history_file(root).exists()


def test_logged_export_sha_tamper_is_detected(tmp_path):
    root=tmp_path/'managed'
    a=make_package(root,'plate/m.alprmodel',single_model())
    registry.append_event(root,operation_id='Z4_TAMPER_0001',operation='SINGLE_MODEL_EXPORT',
                          outcome='SUCCEEDED',artifact_path=str(a))
    rows,events,_=registry.library_snapshot(root,tmp_path)
    assert rows[0].state=='REGISTERED_SHA_MATCH_NOT_VERIFIED'
    with a.open('ab') as stream:
        stream.write(b'changed-after-export')
    rows,events,_=registry.library_snapshot(root,tmp_path)
    assert rows[0].state=='REGISTERED_SHA_MISMATCH'
