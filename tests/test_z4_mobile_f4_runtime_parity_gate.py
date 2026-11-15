#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4LIB022: altered decoder/output layout can never look like a clean F4 match.

Read-only synthetic manifest audit. No models, TFLite conversions, Android
runtime or final_test are started by these tests.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'tests/test_z4_mobile_f4_cross_precision.py'
spec = importlib.util.spec_from_file_location('_f4_cross_precision_fixtures_022', REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
compare = reference.compare
inventory = reference.inventory
model = reference.model
PINS = reference.PINS


class F4RuntimeParityGateTests(unittest.TestCase):
    def setUp(self):
        self.rows = [model(role, precision) for precision in ('fp32', 'int8')
                     for role in ('MT', 'MZ')]

    def audit(self):
        current = inventory.build_f4_inventory(self.rows, checkpoint_sha_by_role=PINS)
        return compare.compare_f4_cross_precision(self.rows, current)

    def test_same_manifest_metadata_is_not_device_approval(self):
        result = self.audit()
        self.assertEqual(result['status'], 'MANIFEST_CROSS_PRECISION_MATCH_RUNTIME_NOT_VERIFIED')
        self.assertFalse(result['graph_checked'])
        self.assertFalse(result['android_approved'])
        self.assertFalse(result['f4_experiment_performed'])

    def test_different_mz_decoder_cannot_be_clean_match(self):
        self.rows[-1].manifest['output']['decoder'] = 'ultralytics_detect_split_ltrb_scores_int8_v1'
        result = self.audit()
        mz = result['roles']['MZ']
        self.assertEqual(mz['role_state'], 'RUNTIME_PARITY_REVIEW_REQUIRED')
        self.assertEqual(result['status'], 'RUNTIME_PARITY_REVIEW_REQUIRED')
        self.assertEqual(mz['checked_pairs'][0]['state'], 'RUNTIME_PARITY_REVIEW_REQUIRED')
        self.assertIn('runtime_decoder_differs', mz['checked_pairs'][0]['decoder_review'])
        self.assertIn('ALPR Mobile', compare.describe_f4_cross_precision(result))
        self.assertFalse(result['android_approved'])

    def test_different_output_format_requires_review(self):
        self.rows[-1].manifest['output']['output_format'] = 'raw_split_ltrb_scores_v1'
        result = self.audit()
        self.assertEqual(result['status'], 'RUNTIME_PARITY_REVIEW_REQUIRED')
        self.assertIn('runtime_output_format_differs', result['roles']['MZ']['checked_pairs'][0]['decoder_review'])

    def test_different_tensor_layout_requires_review(self):
        self.rows[2].manifest['output']['tensor_layout'] = 'channels_last'
        self.assertEqual(self.audit()['roles']['MT']['role_state'], 'RUNTIME_PARITY_REVIEW_REQUIRED')

    def test_different_box_format_requires_review(self):
        self.rows[2].manifest['output']['box_format'] = 'ltrb'
        self.assertEqual(self.audit()['status'], 'RUNTIME_PARITY_REVIEW_REQUIRED')

    def test_unknown_decoder_metadata_has_priority_over_review(self):
        self.rows[-1].manifest['output']['decoder'] = 'different'
        del self.rows[-1].manifest['output']['tensor_layout']
        result = self.audit()
        self.assertEqual(result['status'], 'INCONCLUSIVE_METADATA')
        self.assertIn('runtime_tensor_layout_missing', result['roles']['MZ']['checked_pairs'][0]['unknown'])

    def test_semantic_mismatch_overrides_decoder_review(self):
        self.rows[-1].manifest['output']['decoder'] = 'different'
        self.rows[-1].manifest['output']['nms_required'] = False
        result = self.audit()
        self.assertEqual(result['status'], 'BLOCKED_METADATA_MISMATCH')
        self.assertIn('output_nms_required_differs', result['roles']['MZ']['checked_pairs'][0]['blockers'])

    def test_multiple_runtime_review_pairs_are_not_unique(self):
        self.rows[-1].manifest['output']['decoder'] = 'different'
        alt = copy.deepcopy(self.rows[-1]); alt.path='/test/MZ_int8_other.alprmodel'
        alt.name='MZ_int8_other.alprmodel'
        alt.manifest['output']['output_format'] = 'different'
        self.rows.append(alt)
        result = self.audit()
        self.assertEqual(result['roles']['MZ']['role_state'], 'AMBIGUOUS_SELECT_EXPLICITLY')
        self.assertEqual(result['status'], 'AMBIGUOUS_SELECT_EXPLICITLY')
        self.assertEqual(result['roles']['MZ']['total_combinations'], 2)

    def test_clean_candidate_plus_runtime_review_requires_explicit_choice(self):
        alt = copy.deepcopy(self.rows[-1]); alt.path='/test/MZ_int8_other.alprmodel'
        alt.name='MZ_int8_other.alprmodel'
        alt.manifest['output']['decoder'] = 'different'
        self.rows.append(alt)
        result = self.audit()
        self.assertEqual(result['roles']['MZ']['role_state'], 'AMBIGUOUS_SELECT_EXPLICITLY')
        self.assertEqual(result['status'], 'AMBIGUOUS_SELECT_EXPLICITLY')

    def test_both_roles_runtime_review_stays_unverified(self):
        self.rows[2].manifest['output']['decoder'] = 'mt_split'
        self.rows[3].manifest['output']['decoder'] = 'mz_split'
        result = self.audit()
        self.assertEqual(result['status'], 'RUNTIME_PARITY_REVIEW_REQUIRED')
        self.assertEqual([result['roles'][role]['role_state'] for role in ('MT','MZ')],
                         ['RUNTIME_PARITY_REVIEW_REQUIRED'] * 2)
        self.assertFalse(result['graph_checked'])

    def test_missing_int8_package_retains_absence_not_review(self):
        self.rows = [row for row in self.rows if not
                     (row.role == 'MZ' and row.manifest['variants'][0]['precision'] == 'int8')]
        result = self.audit()
        self.assertEqual(result['status'], 'BLOCKED_NO_BOTH_PRECISIONS')


if __name__ == '__main__':
    unittest.main()
