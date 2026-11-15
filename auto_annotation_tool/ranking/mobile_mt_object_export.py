"""Atomic additive F4-EVAL-2.0 export and deterministic replay validation."""
from __future__ import annotations
from pathlib import Path
import csv
import hashlib
import io
import json
import shutil
import tempfile

from . import mobile_mt_object_evaluation as mt


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sidecar_stamp(session):
    path = session.sidecar_path
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, _sha(path)


def _review_digest(session):
    return hashlib.sha256(json.dumps(session.review.to_dict(), sort_keys=True,
        ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def capture_review_state(session):
    """The exported revision must be the saved review, including external edits."""
    session.verify_source(force=True)
    state=(session.review.review_revision, _review_digest(session), _sidecar_stamp(session))
    if session.sidecar_path.exists():
        disk=json.loads(session.sidecar_path.read_text(encoding='utf-8-sig'))
        disk.pop('derived_metrics',None)
        digest=hashlib.sha256(json.dumps(disk,sort_keys=True,ensure_ascii=False,
            separators=(',', ':')).encode('utf-8')).hexdigest()
        if digest!=state[1]:
            raise ValueError('STALE_LOADED_REVIEW: zapis oceny różni się od wczytanej rewizji; otwórz sesję ponownie.')
    _publish_guard(session,*state)
    return state


def _write_csv(path, rows, fields):
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False, sort_keys=True) if isinstance(row.get(k), (list, dict))
                else row.get(k) for k in fields})


def _publish_guard(session, revision, review_digest, sidecar_stamp):
    session.verify_source(force=True)
    if (session.review.review_revision != revision or _review_digest(session) != review_digest
            or _sidecar_stamp(session) != sidecar_stamp):
        raise ValueError('REVIEW_REVISION_CHANGED: ocena zmieniła się podczas eksportu; odśwież sesję.')


def export(session, directory):
    """Publish a complete new namespace, or leave no published partial result."""
    revision, review_digest, sidecar_stamp = capture_review_state(session)
    data = mt.evaluate_session(session)
    if data['status'] == 'STALE_REVIEW' or data['review_revision'] != revision:
        raise ValueError('REVIEW_REVISION_CHANGED')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.mt_object_v2_stage_', dir=root))
    try:
        def write(name, payload):
            with (stage/name).open('x', encoding='utf-8', newline='') as handle:
                handle.write(payload)
        write('mt_object_evaluation.json', json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True))
        write('policy.json', json.dumps(json.loads(mt._POLICY_JSON), ensure_ascii=False, indent=2, sort_keys=True))
        write('review_snapshot.json', json.dumps(session.review.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        fields = [*mt.IDENTITY_FIELDS, 'matching_policy_id', 'matching_policy_sha256', 'gt_mode',
            'status', 'n_qualified', 'n_total', 'n_final_mt', 'n_gt', 'tp', 'fp', 'fn',
            'precision', 'recall', 'f1', 'partial_fp_lower_bound', 'unavailable_reasons', 'review_tasks',
            'completion_eligibility','source_role_evidence']
        _write_csv(stage/'units.csv', data['units'], fields)
        pairs = [dict(unit_id=u['unit_id'], **p) for u in data['units'] for p in u['matched_pairs']]
        _write_csv(stage/'pairs.csv', pairs, ['unit_id', 'attempt_id', 'object_id', 'gt_id',
            'mt_detection_index', 'matched_by', 'operator_review_evidence', 'bbox_iou', 'bbox_reason',
            'polygon_iou_pct', 'nmce_pct', 'pose_reason', 'disagreement_codes', 'source_bbox', 'source_pose_quad', 'gt_quad'])
        ocr = data['ocr_summary']
        _write_csv(stage/'mz_quality_context.csv', ocr['rows'], list(ocr['rows'][0]) if ocr['rows'] else ['attempt_id', 'quality_state'])
        write('report.md', f"# F4-EVAL-2.0 · PRIMARY MT object · ASSISTED\n\n"
            f"Retrospektywna metoda przyjęta 2026-11-01; nie jest prerejestracją ani COCO mAP.\n\n"
            f"Polityka: `{mt.POLICY_ID}`; SHA-256: `{mt.POLICY_SHA256}`.\n\n"
            f"Sesja: {session.session_id}; rewizja {revision}. {data['status']}; "
            f"kwalifikowane {data['n_qualified']}/{data['n_total']} wywołań.\n\n"
            f"TP/FP/FN: {data['tp']}/{data['fp']}/{data['fn']}; n_FINAL={data['n_final_mt']}; n_GT={data['n_gt']}.\n\n"
            f"Precision {mt.format_rate(data['precision'])}; Recall {mt.format_rate(data['recall'])}; F1 {mt.format_rate(data['f1'])}.\n\n"
            f"Warunkowe OCR: {ocr['status']}, Exact {ocr['n_exact']}/{ocr['n_reads']}, "
            f"CER {ocr['n_text_errors']}/{ocr['n_gt_characters']} znaków GT.\n\n"
            "BBOX IoU, Polygon IoU i NMCE są pomocnicze; nieważny BBOX nie odrzuca decyzji obiektowej. "
            "Braki i zadania operatora są jawne w units.csv. Liczba wierszy OCR nie jest liczbą wywołań MZ. "
            "Czasy i liczniki źródłowe obejmują także wykonania po FP; brak pojedynczych czasów MZ to N/A. "
            "E2E F4 pozostaje N/A bez mapy plansz i reguły okna. Archiwalny BBOX v1 ma odrębny cel.\n")
        paths = sorted(p for p in stage.iterdir() if p.is_file())
        manifest = dict(schema=mt.SCHEMA, matching_policy_id=mt.POLICY_ID,
            matching_policy_sha256=mt.POLICY_SHA256, methodology_version='F4-EVAL-2.0', adopted_date='2026-11-01',
            chronology='RETROSPECTIVE_NOT_PREREGISTERED', gt_mode='ASSISTED',
            session_id=session.session_id, source_archive_sha256=data['source_archive_sha256'],
            review_revision=revision, review_snapshot_digest=review_digest,
            files={p.name: dict(sha256=_sha(p), size=p.stat().st_size) for p in paths},
            algorithm_sha256={p.name: _sha(p) for p in (Path(mt.__file__), Path(__file__),
                Path(__file__).with_name('mobile_object_mz_quality.py'))})
        if data.get('completion_eligibility'):
            manifest.update(completion_eligibility=data['completion_eligibility'],
                completion_eligibility_algorithm_sha256=_sha(Path(mt.__file__).with_name('mobile_frame_review_state.py')),
                source_review_sha256=sidecar_stamp[2] if sidecar_stamp else None,
                source_role_algorithm_sha256=_sha(Path(mt.__file__).with_name('mobile_mt_postdedup.py')),
                source_role_scope=dict(protocol='alpr.desktop.source_roles.v2',
                    unavailable_invocations=[u['mt_invocation_id'] for u in data['units']
                        if 'SOURCE_FINAL_ROLE_UNPROVEN' in u['unavailable_reasons']]),
                metric_coverage=dict(mt=[data['n_qualified'],data['n_total']],
                    geometry=[data['geometry_summary']['n_geometry'],data['geometry_summary']['n_pairs']],
                    ocr=[ocr['n_reads'],ocr['n_target']],
                    mz_executed_all=sum(row['executed'] for row in ocr['rows'])))
            with (stage/'report.md').open('a',encoding='utf-8') as handle:
                handle.write('\n\nEligibility od 2026-11-02: AUTO_FROM_FRAME_DECISIONS. ASSISTED; '
                    'operator_reference_scope; FN tylko w zakresie zapisanych GT operatora. '
                    'Nie jest to niezależny, wyczerpujący spis tablic. 0 FINAL/0 GT: NO_REFERENCE_EVIDENCE / N/A.\n')
            manifest['files']['report.md']=dict(sha256=_sha(stage/'report.md'),size=(stage/'report.md').stat().st_size)
        write('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        replay(stage)
        _publish_guard(session, revision, review_digest, sidecar_stamp)
        for index in range(1, 10000):
            dest = root/f'mt_object_evaluation_v2_{index:03d}'
            try:
                stage.rename(dest)
            except FileExistsError:
                continue
            except OSError:
                if dest.exists():
                    continue
                raise
            return sorted(p for p in dest.iterdir() if p.is_file())
        raise ValueError('Brak wolnego katalogu eksportu F4-EVAL-2.0')
    finally:
        if stage.exists():
            assert stage.resolve().is_relative_to(root.resolve())
            shutil.rmtree(stage)


def replay(directory):
    """Verify every artefact and re-run primary/geometry/OCR pure calculations."""
    root = Path(directory)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if (manifest['schema'] != mt.SCHEMA or manifest['matching_policy_sha256'] != mt.POLICY_SHA256
            or manifest['matching_policy_id'] != mt.POLICY_ID):
        raise ValueError('EXPORT_POLICY_MISMATCH')
    for name, expected in manifest['files'].items():
        path = root/name
        if path.parent.resolve() != root.resolve() or _sha(path) != expected['sha256'] or path.stat().st_size != expected['size']:
            raise ValueError('EXPORT_FILE_MISMATCH: ' + name)
    policy = json.loads((root/'policy.json').read_text(encoding='utf-8'))
    if not mt._approved(policy):
        raise ValueError('EXPORT_POLICY_MISMATCH')
    data = json.loads((root/'mt_object_evaluation.json').read_text(encoding='utf-8'))
    if data.get('completion_eligibility')!=manifest.get('completion_eligibility'):
        raise ValueError('EXPORT_COMPLETION_ELIGIBILITY_MISMATCH')
    recalculated = mt.aggregate_units(mt.score_unit_v2(u['evaluation_input'], policy=policy) for u in data['units'])
    def canonical(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    for key in recalculated:
        if canonical(recalculated[key]) != canonical(data[key]):
            raise ValueError('EXPORT_REPLAY_MISMATCH: ' + key)
    from .mobile_object_mz_quality import score_readout, summarize_readouts
    for row in data['ocr_summary']['rows']:
        if canonical(score_readout(row)) != canonical(row):
            raise ValueError('EXPORT_OCR_REPLAY_MISMATCH')
    for key, value in summarize_readouts(data['ocr_summary']['rows']).items():
        if canonical(value) != canonical(data['ocr_summary'][key]):
            raise ValueError('EXPORT_OCR_SUMMARY_REPLAY_MISMATCH: '+key)
    snapshot = json.loads((root/'review_snapshot.json').read_text(encoding='utf-8'))
    if hashlib.sha256(canonical(snapshot).encode('utf-8')).hexdigest() != manifest['review_snapshot_digest']:
        raise ValueError('EXPORT_REVIEW_SNAPSHOT_MISMATCH')
    if any(data[k] != manifest[k] for k in ('session_id', 'source_archive_sha256', 'review_revision')):
        raise ValueError('EXPORT_IDENTITY_MISMATCH')
    return data
