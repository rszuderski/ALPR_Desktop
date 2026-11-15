"""Read-only F4-EVAL-2.0 projection for the existing analysis result model."""
from pathlib import Path
from ..ranking import mobile_mt_object_evaluation as objects
from ..ranking import mobile_mt_detection_evaluation as bbox_v1
from ..ranking.mobile_human_review import MobileReviewSession


def quality(source, exported_review):
    from .mobile_analysis import METRICS, _metric, _missing
    s = source
    primary = ['mt_tp', 'mt_fp', 'mt_fn', 'mt_precision_pct', 'mt_recall_pct', 'mt_f1_pct']
    secondary = ['bbox_v1_tp', 'bbox_v1_fp', 'bbox_v1_fn', 'bbox_v1_precision_pct', 'bbox_v1_recall_pct', 'bbox_v1_f1_pct']
    other = ['pose_availability_pct', 'polygon_iou_pct', 'nmce_pct', 'bbox_iou_pct', 'exact_mz_pct', 'micro_cer_pct']
    side = s.input.review_root/'human_reviews'/(s.input.source_sha256+'.review.json')
    try:
        if s.quality_state in ('INVALID','UNREADABLE'):
            raise ValueError('QUALITY_EXPORT_INVALID: '+s.quality_reason)
        if side.is_file():
            review_path = side
        elif s.input.review_seen:
            raise ValueError('CURRENT_REVIEW_DISAPPEARED')
        elif exported_review is not None:
            review_path = exported_review.directory/'review.json'
        else:
            # An unreviewed source may be inspected, but no result or decision
            # is persisted. Its units remain N/A with concrete review tasks.
            review_path = side
        session = MobileReviewSession.open(Path(s.input.source_path), sidecar_path=review_path)
        mt = objects.evaluate_session(session)
        if (mt['status'] == 'STALE_REVIEW' or mt['matching_policy_sha256'] != objects.POLICY_SHA256
                or mt['session_id'] != s.input.session_id or mt['source_archive_sha256'] != s.input.source_sha256):
            raise ValueError('OBJECT_SOURCE_OR_POLICY_MISMATCH')
        old = bbox_v1.evaluate_session(session)
        # A new exported v2, if present in the selected verified revision, must
        # agree with a fresh source-bound calculation, not merely carry a label.
        if exported_review is not None:
            from ..ranking.mobile_mt_object_export import replay
            from .mobile_analysis import _digest
            namespaces = sorted(exported_review.directory.glob('mt_object_evaluation_v2_*/manifest.json'))
            if namespaces:
                previous=replay(namespaces[-1].parent)
                if any(previous.get(k)!=mt.get(k) for k in ('session_id','source_archive_sha256','review_revision')):
                    raise ValueError('EXPORTED_V2_RECOMPUTATION_MISMATCH')
                old_eligibility=(previous.get('completion_eligibility') or {}).get('sha256','legacy_checked_v1')
                new_eligibility=(mt.get('completion_eligibility') or {}).get('sha256','legacy_checked_v1')
                if old_eligibility==new_eligibility and _digest(previous)!=_digest(mt):
                    raise ValueError('EXPORTED_V2_RECOMPUTATION_MISMATCH')
                if old_eligibility!=new_eligibility:
                    s.metadata['historical_export_eligibility']=dict(previous=old_eligibility,current=new_eligibility,
                        n_qualified_before=previous['n_qualified'],n_qualified_after=mt['n_qualified'],n_total=mt['n_total'])
        s.metadata.update(review_revision=session.review.review_revision,
            quality_evidence=[str(review_path), s.input.source_path],
            matching_policy_id=objects.POLICY_ID, matching_policy_sha256=objects.POLICY_SHA256,
            quality_method='F4-EVAL-2.0', quality_method_adopted='2026-11-01',
            quality_chronology='RETROSPECTIVE_NOT_PREREGISTERED', gt_mode='ASSISTED',
            bbox_secondary_policy_id=bbox_v1.POLICY_ID, bbox_secondary_policy_sha256=bbox_v1.POLICY_SHA256,
            quality_scope=dict(n_qualified=mt['n_qualified'], n_total=mt['n_total']),
            completion_eligibility=mt.get('completion_eligibility'),
            mz_execution_counters=mt['ocr_summary']['execution_counters'],
            mz_quality_state_counts=mt['ocr_summary']['state_counts'])
        labels = {gid: f'Obraz {i:02d}' for i, gid in enumerate(session.mt_invocations, 1)}
        old_units = {u['unit_id']: u for u in old['units']}
        s.quality_units = [dict(unit_id=u['unit_id'], mt_invocation_id=u['mt_invocation_id'], image=u['image_entry'],
            label=labels[u['mt_invocation_id']], status=u['status'], tp=u['tp'], fp=u['fp'], fn=u['fn'],
            missing_reasons=u['unavailable_reasons'], review_tasks=u['review_tasks'],
            matching_policy_id=objects.POLICY_ID, matching_policy_sha256=objects.POLICY_SHA256,
            completion_eligibility=u.get('completion_eligibility'),
            matched_pairs=u['matched_pairs'], final_outcomes=u['final_outcomes'],
            bbox_v1={k: old_units[u['unit_id']].get(k) for k in ('status', 'tp', 'fp', 'fn', 'unavailable_reasons')}) for u in mt['units']]
        def counts(result, prefix, explanation):
            n, total = result['n_qualified'], result['n_total']
            reasons = [r for u in result['units'] if u['status'] != 'MEASURED' for r in u['unavailable_reasons']]
            for k in ('tp', 'fp', 'fn'):
                _metric(s, prefix+k, result[k], n, total, reasons=reasons, quality=True, explanation=explanation)
            for k, num, den in (('precision_pct', result['tp'], None if not n else result['tp']+result['fp']),
                    ('recall_pct', result['tp'], None if not n else result['tp']+result['fn']),
                    ('f1_pct', None if not n else 2*result['tp'], None if not n else 2*result['tp']+result['fp']+result['fn'])):
                _metric(s, prefix+k, 100*num/den if n and den else None, n, total, quality=True,
                    numerator=num, denominator=den, reasons=reasons or (() if den else ('ZERO_DENOMINATOR',)), explanation=explanation)
        counts(mt, 'mt_', 'F4-EVAL-2.0 · retrospektywnie 2026-11-01 · ASSISTED. Jeden FINAL, potwierdzona relacja MT ↔ fizyczna GTR, 1:1; bez progu BBOX. Eligibility 2026-11-02: AUTO_FROM_FRAME_DECISIONS; FN w zakresie zapisanych GT operatora.')
        counts(old, 'bbox_v1_', 'Pomocnicza archiwalna metoda BBOX v1: IoU ≥ 0,50, greedy 1:1; inny cel niż ocena obiektu MT v2.')
        geo, ocr = mt['geometry_summary'], mt['ocr_summary']
        partial = mt['n_qualified'] < mt['n_total']
        missing_units = ('UNQUALIFIED_MT_UNITS_OUTSIDE_GEOMETRY',) if partial else ()
        pairs, geometry = geo['n_pairs'], geo['n_geometry']
        _metric(s, 'pose_availability_pct', 100*geometry/pairs if pairs else None, pairs, pairs,
            numerator=geometry, denominator=pairs, quality=True, subset=partial,
            reasons=missing_units, explanation='Dostępne rzeczywiste Pose / potwierdzone pary TP. Zakres MT podano osobno.')
        for key, value in (('polygon_iou_pct', geo['polygon_iou_mean_pct']), ('nmce_pct', geo['nmce_mean_pct'])):
            _metric(s, key, value, geometry, pairs, quality=True, subset=partial,
                reasons=(*geo['unavailable_reasons'], *missing_units),
                explanation='Pomocniczy ciągły pomiar rzeczywistego source Pose ↔ GTR; bez rekonstrukcji narożników.')
        _metric(s, 'bbox_iou_pct', 100*geo['bbox_iou_mean'] if geo['n_bbox'] else None,
            geo['n_bbox'], pairs, quality=True, subset=partial,
            reasons=('SOURCE_BBOX_UNAVAILABLE_OR_DEGENERATE',) if geo['n_bbox'] < pairs else (),
            explanation='Pomocnicze ciągłe BBOX IoU dla par v2; zerowe BBOX są N/A tej miary.')
        for key, value, num, den in (('exact_mz_pct', 100*ocr['exact_rate'] if ocr['exact_rate'] is not None else None, ocr['n_exact'], ocr['n_reads']),
                ('micro_cer_pct', 100*ocr['micro_cer'] if ocr['micro_cer'] is not None else None, ocr['n_text_errors'], ocr['n_gt_characters'])):
            _metric(s, key, value, ocr['n_reads'], ocr['n_target'], quality=True,
                numerator=num, denominator=den, reasons=ocr['unavailable_reasons'],
                explanation='Warunkowe OCR: TP + potwierdzone GTZ + dowiedziony crop/attempt. Puste i błędne wykonane odczyty wliczone; FP i NOT_RUN poza mianownikiem. To nie E2E.')
    except (OSError, ValueError, KeyError, TypeError) as error:
        for key in (*primary, *secondary, *other):
            _missing(s, key, 'OBJECT_COUNTS_NOT_VERIFIED', explanation=str(error))
