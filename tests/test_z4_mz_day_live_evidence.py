#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests: MZ-DAY live TRAIN calibration and class-order, without conversion."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
MODULE = HERE / 'auto_annotation_tool/exporters/mz_raw_int8_live_evidence.py'
spec = importlib.util.spec_from_file_location('auto_annotation_tool.exporters.mz_raw_int8_live_evidence', MODULE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

ALPHABET = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
MODEL_SHA = helper.SOURCES['clean_raw_tflite'][1]
CHECKPOINT_SHA = helper.CHECKPOINT_SHA256


class LiveAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.good_path = self.root / 'Workspace' / 'E-MZ-DN-01' / 'raporty_do_pracy' / 'calibration_int8_character.json'
        self.good_path.parent.mkdir(parents=True)
        self.good_path.write_text('RESEARCH_MANIFEST', encoding='utf-8')
        self.assessor_calls = []
        self.scanner_calls = []

    def scan(self, *, paths=(), truncated=False, scope='MANAGED_REPORT_AND_CALIBRATION_ROOTS',
             authority='PREFILTER_ONLY_VALIDATE_BEFORE_BIND', reason='', **kwargs):
        return {
            'matches': [{'path': str(p)} for p in paths],
            'truncated': truncated,
            'search_scope': scope,
            'authority': authority,
            'directories_scanned': 5,
            'files_scanned': 2,
            'limit_reason': reason,
        }

    def scanner(self, root, **kwargs):
        self.scanner_calls.append((root, kwargs))
        return self.scan(paths=self.paths if hasattr(self, 'paths') else (),
                         truncated=getattr(self, 'truncated', False),
                         authority=getattr(self, 'authority', 'PREFILTER_ONLY_VALIDATE_BEFORE_BIND'))

    def assessor(self, root, *, calibration_manifest=None, verify_checkpoint_labels=False):
        self.assessor_calls.append((root, calibration_manifest, verify_checkpoint_labels))
        self.assertTrue(verify_checkpoint_labels)
        self.assertEqual(root, self.root)
        cal = ({'status': 'EXPLORATORY_ONLY_REVALIDATED', 'sha256': 'e'*64,
                'input_fingerprint': 'f'*64} if calibration_manifest is not None else {'status': 'NOT_REVALIDATED'})
        blockers = [
            'ANDROID_DUAL_OUTPUT_DECODER_NOT_APPROVED',
            'ALPRMODEL_EXPORT_NOT_APPROVED_BY_RESEARCH_CONTRACT',
            'FINAL_TEST_NOT_VERIFIED',
        ]
        if calibration_manifest is None:
            blockers.append('LIVE_TRAIN_CALIBRATION_MANIFEST_REQUIRED')
        return {
            'schema': 'alpr.f4.mz_raw_int8_packaging_readiness.v1',
            'status': 'BLOCKED_NOT_ANDROID_DEPLOYABLE',
            'research_artifact_integrity': 'PASS_PINNED_SHA_AND_LIVE_GRAPH_EVIDENCE',
            'model_sha256': MODEL_SHA,
            'checkpoint_sha256': CHECKPOINT_SHA,
            'class_order': {'status': 'CHECKPOINT_VERIFIED', 'ordered_labels': list(ALPHABET)},
            'calibration': cal,
            'blockers': blockers,
            'alprmodel_export_approved': False,
            'android_import_approved': False,
            'final_test': 'NOT_VERIFIED',
            'conversion_performed': False,
            'packaging_performed': False,
        }

    def audit(self, **kwargs):
        return helper.inspect_mz_day_live_evidence(self.root, scanner=self.scanner, assessor=self.assessor, **kwargs)

    def test_unique_prefilter_invokes_authoritative_calibration(self):
        self.paths = [self.good_path]
        r = self.audit()
        self.assertEqual(r['manifest_discovery']['selection'], 'UNIQUE_PREFILTER_LIVE_VALIDATION_REQUIRED')
        self.assertTrue(r['manifest_discovery']['unique_complete_auto_binding'])
        self.assertEqual(self.assessor_calls[0][1], self.good_path)
        self.assertEqual(r['calibration']['status'], 'EXPLORATORY_ONLY_REVALIDATED')
        self.assertNotIn('LIVE_TRAIN_CALIBRATION_MANIFEST_REQUIRED', r['remaining_blockers'])
        self.assertEqual(r['class_order']['ordered_labels'], ALPHABET)
        self.assertEqual(r['class_order']['order_sha256'], helper.ordered_label_digest(ALPHABET))
        self.assertEqual(self.scanner_calls[0][1]['checkpoint_sha256'], CHECKPOINT_SHA)
        self.assertEqual(self.scanner_calls[0][1]['role'], 'character')
        self.assertEqual(self.scanner_calls[0][1]['image_size'], 320)
        self.assertEqual(self.scanner_calls[0][1]['mode'], 'exploratory')

    def test_zero_match_retains_calibration_blocker(self):
        self.paths = []
        r = self.audit()
        self.assertEqual(r['manifest_discovery']['selection'], 'NONE_FOUND_IN_MANAGED_SCOPE')
        self.assertIn('LIVE_TRAIN_CALIBRATION_MANIFEST_REQUIRED', r['remaining_blockers'])
        self.assertTrue(r['class_order']['status'] == 'CHECKPOINT_VERIFIED')

    def test_multiple_matches_requires_explicit_user_choice(self):
        self.paths = [self.good_path, self.good_path.parent / 'another_calibration.json']
        r = self.audit()
        self.assertEqual(r['manifest_discovery']['selection'], 'AMBIGUOUS_MULTIPLE_MATCHES_NO_AUTOMATIC_BINDING')
        self.assertIn('CALIBRATION_MANIFEST_AMBIGUOUS_SELECT_EXPLICITLY', r['remaining_blockers'])
        self.assertIsNone(self.assessor_calls[0][1])

    def test_truncated_scan_never_auto_binds_even_one_candidate(self):
        self.paths = [self.good_path]
        self.truncated = True
        r = self.audit()
        self.assertEqual(r['manifest_discovery']['selection'], 'INCOMPLETE_SCAN_NO_AUTOMATIC_BINDING')
        self.assertIn('CALIBRATION_DISCOVERY_INCOMPLETE', r['remaining_blockers'])
        self.assertIsNone(self.assessor_calls[0][1])

    def test_explicit_manifest_must_still_be_live_validated(self):
        self.paths = [self.good_path, self.good_path.parent/'another.json']
        r = self.audit(calibration_manifest=self.good_path)
        self.assertEqual(r['manifest_discovery']['selection'], 'USER_EXPLICIT_LIVE_VALIDATION_REQUIRED')
        self.assertFalse(r['manifest_discovery']['unique_complete_auto_binding'])
        self.assertEqual(self.assessor_calls[0][1], self.good_path)

    def test_no_android_approval_or_final_test_promotion(self):
        self.paths = [self.good_path]
        before = self.good_path.read_bytes()
        r = self.audit()
        self.assertFalse(r['android_import_approved'])
        self.assertFalse(r['alprmodel_export_approved'])
        self.assertEqual(r['final_test'], 'NOT_VERIFIED')
        self.assertFalse(r['conversion_performed'])
        self.assertFalse(r['packaging_performed'])
        self.assertIn('ANDROID_DUAL_OUTPUT_DECODER_NOT_APPROVED', r['remaining_blockers'])
        self.assertIn('ALPRMODEL_EXPORT_NOT_APPROVED_BY_RESEARCH_CONTRACT', r['remaining_blockers'])
        self.assertEqual(self.good_path.read_bytes(), before)
        self.assertEqual(list(self.root.rglob('*.alprmodel')), [])

    def test_invalid_prefilter_authority_rejected(self):
        self.authority = 'APPROVED'
        with self.assertRaisesRegex(helper.LiveEvidenceAuditError, 'AUTHORITY_MISMATCH'):
            self.audit()
        self.assertEqual(self.assessor_calls, [])

    def test_live_assessor_cannot_claim_android_approval(self):
        self.paths = [self.good_path]
        parent_assessor = self.assessor
        def forged(*args, **kwargs):
            payload = parent_assessor(*args, **kwargs)
            payload['android_import_approved'] = True
            return payload
        with self.assertRaisesRegex(helper.LiveEvidenceAuditError, 'UNSAFE_FIELD'):
            helper.inspect_mz_day_live_evidence(self.root, scanner=self.scanner, assessor=forged)

    def test_missing_hard_blocker_rejected(self):
        parent_assessor = self.assessor
        def forged(*args, **kwargs):
            payload = parent_assessor(*args, **kwargs)
            payload['blockers'].remove('FINAL_TEST_NOT_VERIFIED')
            return payload
        with self.assertRaisesRegex(helper.LiveEvidenceAuditError, 'MISSING_HARD_BLOCKER'):
            helper.inspect_mz_day_live_evidence(self.root, scanner=self.scanner, assessor=forged)

    def test_bad_alphabet_order_is_not_certified(self):
        valid = {'class_order': {'status': 'CHECKPOINT_VERIFIED', 'ordered_labels': list(ALPHABET)}}
        self.assertEqual(helper.validate_checkpoint_class_order(valid)['count'], 36)
        invalid = {'class_order': {'status': 'CHECKPOINT_VERIFIED', 'ordered_labels': list(ALPHABET[:-1]) + ['9']}}
        with self.assertRaisesRegex(helper.LiveEvidenceAuditError, 'ALPHABET_NOT_EXACT'):
            helper.validate_checkpoint_class_order(invalid)

    def test_order_digest_changes_when_swapping_indices(self):
        other = list(ALPHABET)
        other[0], other[1] = other[1], other[0]
        self.assertNotEqual(helper.ordered_label_digest(ALPHABET), helper.ordered_label_digest(other))

    def test_empty_and_invalid_prefilter_are_blocked(self):
        for malformed in ({}, {'authority':'PREFILTER_ONLY_VALIDATE_BEFORE_BIND'}, None):
            with self.subTest(payload=malformed):
                with self.assertRaises(helper.LiveEvidenceAuditError):
                    helper.inspect_mz_day_live_evidence(self.root, scanner=lambda *a, v=malformed, **kw:v, assessor=self.assessor)

    def test_normal_scan_does_not_confuse_no_match_with_global_absence(self):
        self.paths = []
        r = self.audit()
        self.assertEqual(r['manifest_discovery']['scope'], 'MANAGED_REPORT_AND_CALIBRATION_ROOTS')
        self.assertEqual(r['manifest_discovery']['selection'], 'NONE_FOUND_IN_MANAGED_SCOPE')


if __name__ == '__main__':
    unittest.main()
