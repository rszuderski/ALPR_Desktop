"""Read-only native adapter for the Android F5A/16 tensor evidence contract.

No tensor conversion, rectification, model execution or scoring takes place here.
JSON integers stay Python integers, including SourceStamp values above 2**53.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import zipfile

from .mobile_scene_evidence import safe_path, stamp, integer, _unique_fields

SCHEMA = 'alpr.mz_input_evidence.v1'
CAPABILITY = 'mz_input_evidence_v1'
INDEX = 'samples/mz_input_evidence.jsonl'
EXPORT_SCHEMA = 'alpr.desktop_mz_input_evidence.v1'
EXPORT_FILE = 'mz_input_evidence.json'
NAMESPACE = 'mz_input_evidence'
DIAGNOSTIC_PURPOSES = {'SYNTHETIC_DIAGNOSTIC': 'SYNTHETIC-',
                       'DEVICE_DIAGNOSTIC': 'DEVICE-DIAGNOSTIC-MZ-'}
LIMITS = dict(tensor_bytes=8388608, memory_bytes=16777216,
              outstanding_captures=2, payloads=16, session_bytes=67108864)
STAMP = ('scene_generation', 'visual_epoch', 'camera_transform_generation',
         'source_sequence', 'source_timestamp_nanos', 'source_timestamp_domain')
MISSING_REASONS = {'TENSOR_SIZE_LIMIT', 'MEMORY_OR_QUEUE_LIMIT', 'SESSION_PAYLOAD_LIMIT',
                   'BUFFER_CONTRACT_UNAVAILABLE', 'CAPTURE_FAILED', 'CANCELLED_BEFORE_CALL',
                   'PAYLOAD_WRITE_FAILED', 'WRITE_NOT_RECORDED', 'PAYLOAD_INTEGRITY_FAILURE', 'MZ_NOT_RUN'}


def require(condition, reason):
    if not condition:
        raise ValueError('MZ_INPUT_EVIDENCE_INVALID: ' + reason)


def loads(data):
    def constant(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    return json.loads(data, object_pairs_hook=_unique_fields, parse_constant=constant)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def is_diagnostic_purpose(value):
    return isinstance(value, str) and value in DIAGNOSTIC_PURPOSES


def diagnostic_cohort_conflict(purposes):
    values = set(purposes)
    return len(values) > 1 and any(is_diagnostic_purpose(v) for v in values)


def absent(purpose=None):
    return dict(available=False, status='ABSENT', rows=[], dependencies={},
                measurement_purpose=purpose)


def read_extension(archive, descriptor, session, manifest, report):
    """Reject contradictions; explicit missing evidence remains usable PARTIAL data."""
    caps = descriptor.get('capabilities') or {}
    require(isinstance(caps, dict), 'Capability object')
    enabled = caps.get(CAPABILITY, False)
    require(type(enabled) is bool, 'Capability must be boolean')
    names = archive.namelist()
    require(len(names) == len(set(names)), 'Duplicate ZIP entry')
    claimed = INDEX in names or any(n.startswith('samples/mz_inputs/') for n in names)
    if not enabled:
        research = report.get('research_collection') or {}
        require(not is_diagnostic_purpose(session.get('measurement_purpose'))
                and not (isinstance(research, dict) and is_diagnostic_purpose(research.get('measurement_purpose'))),
                'Diagnostic purpose without capability')
        require(not claimed and not descriptor.get('mz_input_evidence_file')
                and not session.get('mz_input_evidence_enabled') and not session.get('mz_input_evidence'),
                'Undeclared MZ evidence')
        return absent(session.get('measurement_purpose'))
    require(descriptor.get('mz_input_evidence_file') == INDEX, 'Index path')
    require(archive.getinfo('manifest.json').file_size <= 32 * 1024 * 1024, 'Manifest size')
    require(loads(archive.read('manifest.json').decode('utf-8-sig')) == manifest, 'Manifest fields')
    deps = {}

    def read(name, maximum=32 * 1024 * 1024):
        safe_path(name)
        info = archive.getinfo(name)
        require(0 <= info.file_size <= maximum, 'Entry size: ' + name)
        data = archive.read(info)  # ZipFile checks CRC as the entry is consumed.
        digest = hashlib.sha256(data).hexdigest()
        require(len(data) == info.file_size and manifest.get('entry_sha256', {}).get(name) == digest,
                'Manifest SHA/size: ' + name)
        deps[name] = dict(sha256=digest, bytes=len(data))
        return data

    def rows(name):
        text = read(name).decode('utf-8')
        require(not text.startswith('\ufeff'), 'JSONL BOM')
        result = [loads(line) for line in text.splitlines() if line.strip()]
        require(all(isinstance(r, dict) for r in result), 'JSONL row object')
        return result

    # Reparse identity documents strictly, rather than trusting a preview parser.
    for name, expected in (('session.json', session), ('samples/schema.json', descriptor)):
        require(loads(read(name).decode('utf-8-sig')) == expected, 'Metadata conflict: ' + name)
    original_report = loads(read('report.json').decode('utf-8-sig'))
    research = original_report.get('research_collection') or {}
    require(isinstance(research, dict), 'Research collection object')
    require(research == report.get('research_collection'), 'Research metadata conflict')
    require(session.get('mz_input_evidence_enabled') is True
            and research.get('mz_input_evidence_enabled') is True, 'Frozen diagnostic setting')
    purpose = session.get('measurement_purpose')
    summary = session.get('mz_input_evidence')
    require(is_diagnostic_purpose(purpose) and isinstance(summary, dict)
            and purpose == research.get('measurement_purpose') == summary.get('purpose'), 'Diagnostic purpose')
    series = session.get('series_id')
    prefix = DIAGNOSTIC_PURPOSES[purpose]
    require(isinstance(series, str) and series.startswith(prefix) and len(series) > len(prefix)
            and series == series.strip() and research.get('series_id') == series, 'Diagnostic series')
    sid = session.get('session_id')
    require(isinstance(sid, str) and sid, 'Session identity')
    attempts = {}
    for attempt in rows('samples/attempts.jsonl'):
        aid = attempt.get('attempt_id')
        require(isinstance(aid, str) and aid.startswith(sid + '-a') and aid not in attempts,
                'Missing/duplicate attempt identity')
        safe_path(aid)
        require('/' not in aid and attempt.get('session_id') == sid, 'Attempt session')
        stamp(attempt)
        count = attempt.get('mz_input_call_count', 0)
        require(type(count) is int and 0 <= count <= 1000, 'Call count')
        attempts[aid] = attempt
    csv_rows = list(csv.DictReader(io.StringIO(read('samples/attempts.csv').decode('utf-8-sig'))))
    require(len(csv_rows) == len(attempts), 'Attempt CSV denominator')
    csv_ids = set()
    for row in csv_rows:
        aid = row.get('attempt_id')
        require(aid in attempts and aid not in csv_ids, 'Attempt CSV identity')
        csv_ids.add(aid)
        for key in ('session_id', 'mt_invocation_id', *STAMP):
            # Compare integer spelling, never pass a timestamp through float().
            require(row.get(key, '') == str(attempts[aid].get(key, '')), 'Attempt CSV identity: ' + key)

    records = rows(INDEX)
    keys = set()
    written = missing = not_run = total_bytes = context_missing = 0
    tensor_entries = set()
    for row in records:
        require(row.get('schema') == SCHEMA and row.get('session_id') == sid, 'Record schema/session')
        aid = row.get('attempt_id')
        require(aid in attempts, 'Unknown attempt')
        attempt = attempts[aid]
        require(type(row.get('stale_or_cancelled')) is bool, 'Cancellation type')
        for key in ('mt_invocation_id', 'stale_or_cancelled', 'cancel_reason', 'mt_status', 'mz_status', 'roi_policy', 'mt_detection_index'):
            require(key in row and row[key] == attempt.get(key), 'Attempt context: ' + key)
        require(all(isinstance(row[k], str) for k in ('mt_invocation_id', 'cancel_reason', 'mt_status', 'mz_status', 'roi_policy')),
                'Context identity types')
        if row['mt_detection_index'] is not None:
            integer(row['mt_detection_index'])
        source = row.get('source_stamp')
        stamp(source)
        require(set(source) == set(STAMP), 'SourceStamp fields')
        require(all(source[k] == attempt[k] for k in STAMP), 'SourceStamp mismatch')
        for key in ('left', 'top', 'right', 'bottom'):
            require(row['roi'][key] == attempt.get('roi_' + key), 'ROI mismatch')
            require(row['mt_bbox'][key] == attempt.get('plate_' + key), 'MT bbox mismatch')
        for key in ('status', 'corners_norm', 'source_width_px', 'source_height_px', 'coordinate_space', 'corner_order'):
            require(row['mt_pose'][key] == attempt.get('mt_pose_' + key), 'MT Pose mismatch')
        call = row.get('call_index')
        require('call_index' in row and (call is None or type(call) is int), 'Call index type')
        require((aid, call) not in keys, 'Duplicate evidence index')
        keys.add((aid, call))
        tensor = row['tensor']
        status = tensor.get('write_status')
        require(status in {'WRITTEN', 'MISSING'}, 'Tensor write status')
        calls = attempt.get('mz_input_call_count', 0)
        if call is None:
            require(calls == 0 and not attempt.get('mz_executed') and row.get('invocation_status') == 'NOT_RUN'
                    and status == 'MISSING' and (tensor.get('missing_reason') == 'MZ_NOT_RUN'
                    or (tensor.get('missing_reason') == 'CANCELLED_BEFORE_CALL' and row['stale_or_cancelled'])),
                    'NOT_RUN contradiction')
            not_run += 1
        else:
            require(0 <= call < calls and attempt.get('mz_executed') is True
                    and attempt.get('mt_executed') is True and bool(row['mt_invocation_id']), 'Unknown MZ/MT call')
            require(row.get('invocation_status') in {'ENTERED', 'RETURNED', 'THREW'}, 'Invocation status')
            if status == 'MISSING':
                missing += 1
        if status == 'MISSING':
            require(tensor.get('missing_reason') in MISSING_REASONS
                    and tensor.get('entry') is None and tensor.get('sha256') is None, 'Missing tensor claim')
        if status == 'WRITTEN' or 'shape' in tensor:
            shape = tensor['shape']
            require(isinstance(shape, list) and len(shape) == 4
                    and all(type(d) is int and 0 < d < 2**31 for d in shape) and shape[0] == 1, 'Tensor shape')
            dtype, layout = tensor.get('dtype'), tensor.get('layout')
            require(dtype in {'FLOAT32', 'INT8', 'UINT8'} and layout in {'NHWC', 'NCHW'}, 'Tensor dtype/layout')
            require(shape[3 if layout == 'NHWC' else 1] == 3 and tensor.get('color') in {'RGB', 'BGR'}, 'Tensor channels/color')
            size = math.prod(shape) * (4 if dtype == 'FLOAT32' else 1)
            require(type(tensor.get('byte_size')) is int and tensor['byte_size'] == size, 'Tensor shape byte size')
            require(tensor.get('byte_order') in {'LITTLE_ENDIAN', 'BIG_ENDIAN'}, 'Byte order')
            for k in ('original_position', 'original_limit', 'original_capacity', 'range_start', 'range_end'):
                integer(tensor[k])
            require(0 <= tensor['original_position'] <= tensor['original_limit'] <= tensor['original_capacity']
                    and tensor['range_start'] == 0 and tensor['range_end'] == tensor['original_limit'] == size,
                    'Buffer range')
            require(all(finite(tensor.get(k)) for k in ('normalization_scale', 'normalization_offset', 'quantization_scale')),
                    'Preprocessing values')
            qzero = tensor.get('quantization_zero_point')
            require(type(qzero) is int and -(2**31) <= qzero < 2**31, 'Quantization zero point')
            if dtype != 'FLOAT32':
                require(tensor['quantization_scale'] > 0 and
                        (-128 <= qzero <= 127 if dtype == 'INT8' else 0 <= qzero <= 255), 'Quantization contract')
        if status == 'WRITTEN':
            name = f'samples/mz_inputs/{aid}/{call}.bin'
            require(call is not None and tensor.get('entry') == name and tensor.get('missing_reason') is None,
                    'Tensor path/status')
            require(size <= LIMITS['tensor_bytes'] and sha(tensor.get('sha256')), 'Tensor byte budget/SHA')
            data = read(name, LIMITS['tensor_bytes'])
            require(len(data) == size and hashlib.sha256(data).hexdigest() == tensor['sha256'], 'Tensor SHA/length')
            tensor_entries.add(name)
            total_bytes += size
            written += 1
            for key in ('model_id', 'variant_id', 'model_fingerprint', 'backend'):
                require(isinstance(row['model'].get(key), str) and row['model'][key], 'Model identity: ' + key)
            model = row['model']
            require((sha(model.get('artifact_sha256')) and model.get('artifact_sha256_status') == 'OBSERVED_RUNTIME_METADATA')
                    or (model.get('artifact_sha256') is None and model.get('artifact_sha256_status') == 'NOT_AVAILABLE'),
                    'Model artifact provenance')
            frozen = ((session.get('execution') or {}).get('stages') or {}).get('mz')
            if frozen:
                require(all(model.get(k) == frozen.get(k) for k in ('model_id', 'variant_id')), 'Frozen MZ model conflict')
            for key in ('crop_width', 'crop_height'):
                require(type(row.get(key)) is int and row[key] > 0, 'Crop size')
            letterbox = row['letterbox']
            require(all(finite(letterbox.get(k)) for k in ('scale', 'pad_x', 'pad_y')) and letterbox['scale'] > 0
                    and letterbox.get('padding_rgb') == 114 and letterbox.get('interpolation') == 'bilinear', 'Letterbox')
            corners = row.get('recognition_corners_source_px')
            require(isinstance(corners, list) and len(corners) == 4
                    and all(isinstance(p, list) and len(p) == 2 and all(finite(v) for v in p) for p in corners)
                    and row.get('recognition_corner_order') == 'AS_PASSED_TO_PLATE_RECTIFIER', 'Recognition corners')
        if 'homography' in row or status == 'WRITTEN':
            require(row['homography'] == dict(status='NOT_AVAILABLE', reason='RECTIFIER_TRANSFORM_NOT_EXPOSED',
                                               H=None, H_inverse=None), 'Unproven homography')
        for key in ('capture_elapsed_nanos', 'captured_elapsed_nanos', 'queued_elapsed_nanos', 'queue_elapsed_nanos', 'write_elapsed_nanos'):
            if key in row:
                integer(row[key])
        for key, expected in (('source_image', attempt.get('source_frame_entry')),
                              ('rectified_crop', attempt.get('evidence_entry') if attempt.get('evidence_kind') == 'plate_crop' else None)):
            image = row[key]
            require(image.get('encoding') == 'JPEG' and image.get('exact_tensor_input') is False, 'JPEG is contextual')
            if image.get('status') == 'WRITTEN':
                require(bool(expected) and image.get('entry') == expected and sha(image.get('sha256'))
                        and image.get('missing_reason') is None, 'Image identity')
                data = read(expected, 20 * 1024 * 1024)
                require(hashlib.sha256(data).hexdigest() == image['sha256'], 'Context JPEG SHA')
                from PIL import Image
                with Image.open(io.BytesIO(data)) as decoded:
                    require(decoded.format == 'JPEG' and decoded.width * decoded.height <= 40_000_000, 'Context JPEG format/size')
                    if key == 'source_image':
                        require(decoded.size == (image.get('width_px'), image.get('height_px'))
                                == (attempt.get('source_frame_width_px'), attempt.get('source_frame_height_px'))
                                and image.get('coordinate_space') == 'mt_source_bitmap_xy'
                                and image.get('origin') == 'pipeline_bitmap_before_roi', 'Context source dimensions')
                    decoded.load()
            else:
                require(image.get('status') == 'MISSING' and image.get('missing_reason')
                        and image.get('entry') is None and image.get('sha256') is None, 'Context missing status')
                context_missing += 1
    for aid, attempt in attempts.items():
        calls = attempt.get('mz_input_call_count', 0)
        require(all((aid, i) in keys for i in range(calls)) if calls else (aid, None) in keys,
                'Incomplete call index')
    require({n for n in names if n.startswith('samples/mz_inputs/') and not n.endswith('/')} == tensor_entries,
            'Orphan tensor payload')
    summary = session['mz_input_evidence']
    require(summary == research.get('mz_input_evidence') and summary.get('schema') == SCHEMA
            and summary.get('purpose') == purpose and summary.get('limits') == LIMITS, 'Summary/protocol conflict')
    for key, expected in dict(expected_calls=written + missing, written=written, missing=missing, not_run_attempts=not_run).items():
        require(type(summary.get(key)) is int and summary[key] == expected, 'Summary count: ' + key)
    require(type(summary.get('complete')) is bool and type(summary.get('pending_loss_unknown')) is bool, 'Summary status type')
    require(not summary['complete'] or (written > 0 and missing == 0 and not summary['pending_loss_unknown']), 'False complete claim')
    require(written <= LIMITS['payloads'] and total_bytes <= LIMITS['session_bytes'], 'Session payload budget')
    costs = session.get('mz_input_diagnostic_costs')
    if costs is not None:
        require(costs == research.get('mz_input_diagnostic_costs')
                and costs.get('scope') == 'DIAGNOSTIC_ONLY_NOT_MODEL_INFERENCE', 'Diagnostic costs scope')
        for key in ('capture_total_elapsed_nanos', 'payload_write_total_elapsed_nanos', 'index_append_total_elapsed_nanos'):
            integer(costs[key])
    return dict(available=True, status='VALID' if summary['complete'] and not context_missing else 'PARTIAL',
                rows=records, summary=summary, dependencies=deps, measurement_purpose=purpose,
                tensor_payloads=sorted(tensor_entries), context_missing=context_missing, diagnostic_costs=costs)


def diagnostic_text(evidence):
    """Existing Canvas details pane: facts only, never operator decisions or scores."""
    text = [f"MZ input evidence: {evidence.get('status', 'ABSENT')}"]
    if evidence.get('measurement_purpose'):
        text.append('Zakres: ' + evidence['measurement_purpose'])
    if not evidence.get('available'):
        return '\n'.join(text + ['Dokładne wejście MZ: N/A (brak dowodu).'])
    text += ['Dokładne wejście MZ: bajty .bin; JPEG źródła i cropu: kontekst.',
             'H / H⁻¹: N/A — RECTIFIER_TRANSFORM_NOT_EXPOSED.']
    for row in evidence['rows']:
        tensor = row['tensor']
        text.append(f"{row['attempt_id']} / call_index={row['call_index']} / MT={row['mt_invocation_id']} / "
                    f"{row['invocation_status']} / {tensor['write_status']} / "
                    f"{tensor.get('entry') or tensor.get('missing_reason')} / "
                    f"shape={tensor.get('shape', 'N/A')} dtype={tensor.get('dtype', 'N/A')} "
                    f"layout={tensor.get('layout', 'N/A')} byte_order={tensor.get('byte_order', 'N/A')} / "
                    f"SourceStamp={json.dumps(row['source_stamp'], ensure_ascii=False)}")
    return '\n'.join(text)


def export(session, directory):
    """Called by the existing native review exporter, with original dependency bytes."""
    evidence = session.mz_input_evidence
    if not evidence.get('available'):
        # Older F4/F5A exports retain their original file set. Absence is also
        # explicit in the native reader result, without inventing an extension.
        return []
    directory = Path(directory)
    paths = []
    deps = {}
    if evidence.get('available'):
        with zipfile.ZipFile(session.path) as archive:
            original = archive.read('manifest.json')
            dependencies = dict(evidence['dependencies'])
            dependencies['manifest.json'] = dict(sha256=hashlib.sha256(original).hexdigest(), bytes=len(original))
            for name, expected in dependencies.items():
                data = archive.read(name)
                require(len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], 'Source changed: ' + name)
                target = directory / NAMESPACE / safe_path(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                paths.append(target)
                deps[target.relative_to(directory).as_posix()] = expected
    doc = dict(schema=EXPORT_SCHEMA, session_id=session.session_id,
               source_archive_sha256=session.review.source_archive_sha256, review_revision=session.review.review_revision,
               status=evidence['status'], measurement_purpose=evidence.get('measurement_purpose'), dependencies=deps)
    target = directory / EXPORT_FILE
    target.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
    return paths + [target]


def load_export(directory, manifest, source_session=None):
    """Native report re-import: verify dependency closure and replay the same validator."""
    directory = Path(directory)
    target = directory / EXPORT_FILE
    covered = {r['path']: r for r in manifest.get('files', [])}
    expected_enabled = (source_session or {}).get('mz_input_evidence_enabled') is True
    if not target.exists():
        require(EXPORT_FILE not in covered and not (directory / NAMESPACE).exists(), 'Missing exported MZ descriptor')
        require(not expected_enabled, 'Source declares MZ evidence; export descriptor missing')
        return absent((source_session or {}).get('measurement_purpose'))
    require(EXPORT_FILE in covered, 'Unmanifested MZ descriptor')
    doc = loads(target.read_text(encoding='utf-8'))
    require(doc.get('schema') == EXPORT_SCHEMA and all(doc.get(k) == manifest.get(k)
            for k in ('session_id', 'source_archive_sha256', 'review_revision')), 'Export MZ identity')
    dependencies = doc.get('dependencies')
    require(isinstance(dependencies, dict), 'Export dependencies')
    payloads = {}
    for name, expected in dependencies.items():
        safe_path(name)
        require(name.startswith(NAMESPACE + '/') and name in covered, 'Unmanifested MZ dependency')
        path = directory / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected['bytes']
                and 0 <= expected['bytes'] <= 32 * 1024 * 1024, 'Missing/size MZ dependency: ' + name)
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected['sha256'], 'Export dependency SHA: ' + name)
        payloads[name[len(NAMESPACE) + 1:]] = data
    actual = {p.relative_to(directory).as_posix() for p in (directory / NAMESPACE).rglob('*') if p.is_file()}
    require(actual == set(dependencies), 'Export dependency closure')
    if doc.get('status') == 'ABSENT':
        require(not payloads and not expected_enabled, 'ABSENT contradicts source evidence')
        return absent(doc.get('measurement_purpose'))

    class DependencyArchive:
        def namelist(self):
            return list(payloads)

        def getinfo(self, name):
            return SimpleNamespace(filename=name, file_size=len(payloads[name]))

        def read(self, name):
            return payloads[name if isinstance(name, str) else name.filename]

    archive = DependencyArchive()
    if source_session is not None:
        require(loads(payloads['session.json']) == source_session, 'Review/source session conflict')
    result = read_extension(archive, loads(payloads['samples/schema.json']), loads(payloads['session.json']),
                            loads(payloads['manifest.json']), loads(payloads['report.json']))
    require(result['status'] == doc['status'] and result['measurement_purpose'] == doc.get('measurement_purpose'), 'Export status conflict')
    require(set(payloads) == set(result['dependencies']) | {'manifest.json'}, 'Unreferenced export dependency')
    return result
