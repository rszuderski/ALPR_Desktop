#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only discovery gates and source contract for Z4 manifest UX."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
GUI = HERE / 'auto_annotation_tool/gui/z4_model_export.py'
HELPER = HERE / 'auto_annotation_tool/gui/z4_mobile_manifest_index.py'
spec = importlib.util.spec_from_file_location('z4_manifest_idx_test', HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SHA = 'f'*64


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reports = self.root / 'Workspace' / 'E-MZ-DN-01' / 'raporty_do_pracy'
        self.reports.mkdir(parents=True)

    def doc(self, **override):
        record = dict(schema='alpr.int8_calibration.exploratory.v2',
                      status='EXPLORATORY_ONLY', role='character', split='train',
                      checkpoint={'sha256': SHA},
                      preprocessing={'width': 320, 'height': 320},
                      config={'confidence_threshold': .25, 'iou_threshold': .45},
                      input_fingerprint='a'*64, samples=[{'source': 'train-1'}])
        record.update(override)
        return record

    def write(self, name, content, parent=None):
        target = (parent or self.reports) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(content), encoding='utf-8')
        return target

    def scan(self, **kwargs):
        return module.discover_matching_calibrations(
            self.root, role=kwargs.pop('role', 'character'),
            checkpoint_sha256=kwargs.pop('sha', SHA),
            image_size=kwargs.pop('size', 320),
            confidence_threshold=kwargs.pop('conf', .25),
            iou_threshold=kwargs.pop('iou', .45),
            mode=kwargs.pop('mode', 'exploratory'), **kwargs,
        )

    def test_unique_manifest_prefilter_only(self):
        candidate = self.write('calibration_character_EXPLORATORY_ONLY.json', self.doc())
        result = self.scan()
        self.assertEqual(len(result['matches']), 1)
        self.assertEqual(result['matches'][0]['path'], str(candidate.resolve()))
        self.assertEqual(result['authority'], 'PREFILTER_ONLY_VALIDATE_BEFORE_BIND')

    def test_wrong_checkpoint_not_auto_selected(self):
        self.write('calibration_mz.json', self.doc(checkpoint={'sha256': '1'*64}))
        self.assertEqual(self.scan()['matches'], [])

    def test_wrong_input_size_or_threshold_not_selected(self):
        self.write('calibration_mz.json', self.doc())
        self.assertEqual(self.scan(size=512)['matches'], [])
        self.assertEqual(self.scan(conf=.3)['matches'], [])
        self.assertEqual(self.scan(iou=.5)['matches'], [])

    def test_wrong_role_or_mode_not_selected(self):
        self.write('calibration_mz.json', self.doc(role='plate'))
        self.assertEqual(self.scan()['matches'], [])
        self.assertEqual(self.scan(mode='strict')['matches'], [])

    def test_empty_or_forged_fingerprint_rejected(self):
        self.write('calibration_mz.json', self.doc(input_fingerprint='NOT_SHA'))
        self.assertEqual(self.scan()['matches'], [])

    def test_corrupt_json_skipped(self):
        (self.reports/'calibration_broken.json').write_text('{', encoding='utf-8')
        self.assertEqual(self.scan()['matches'], [])

    def test_multiple_matches_not_unique(self):
        self.write('calibration_mz_1.json', self.doc())
        self.write('calibration_mz_2.json', self.doc())
        self.assertEqual(len(self.scan()['matches']), 2)

    def test_bounded_lookup_marks_incomplete(self):
        self.write('calibration_mz_1.json', self.doc())
        self.write('calibration_mz_2.json', self.doc())
        self.assertTrue(self.scan(max_matches=1)['truncated'])

    def test_avoid_image_tree_and_large_json(self):
        self.write('calibration_hidden.json', self.doc(), parent=self.root/'Workspace'/'images')
        (self.reports/'calibration_enormous.json').write_bytes(b' '*(module.MAX_JSON_BYTES+1))
        self.assertEqual(self.scan()['matches'], [])

    def test_unrelated_tree_cannot_exhaust_directory_limit(self):
        noisy = self.root / 'Workspace' / 'totally_unrelated'
        # A formerly counted 30-level JSON/diagnostics tree is not a managed
        # report location, even if it contains thousands of unrelated entries.
        for _ in range(30):
            noisy = noisy / 'bulkfiles'
            noisy.mkdir(parents=True)
        target = self.write('calibration_character_EXPLORATORY_ONLY.json', self.doc())
        result = self.scan(max_directories=10)
        self.assertFalse(result['truncated'], result)
        self.assertEqual([item['path'] for item in result['matches']], [str(target.resolve())])
        self.assertLessEqual(result['directories_scanned'], 10)

    def test_unrelated_files_do_not_consume_manifest_file_limit(self):
        for i in range(25):
            (self.reports / f'log_{i:03d}.csv').write_text('training log', encoding='utf-8')
        target = self.write('calibration_character_EXPLORATORY_ONLY.json', self.doc())
        result = self.scan(max_files=1)
        self.assertFalse(result['truncated'], result)
        self.assertEqual(result['files_scanned'], 1)
        self.assertEqual(result['matches'][0]['path'], str(target.resolve()))

    def test_report_subfolders_still_searched(self):
        nested = self.reports / 'archive' / 'v2'
        target = self.write('calibration_character_EXPLORATORY_ONLY.json', self.doc(), parent=nested)
        result = self.scan()
        self.assertFalse(result['truncated'], result)
        self.assertEqual(result['matches'][0]['path'], str(target.resolve()))

    def test_output_report_root_and_project_report_root_both_searched(self):
        one = self.write('calibration_1.json', self.doc(), parent=self.reports)
        two = self.write('calibration_2.json', self.doc(),
                         parent=self.root / 'output' / 'experiment_a' / 'reports')
        three = self.write('calibration_3.json', self.doc(),
                           parent=self.root / 'Workspace' / '9_projects' / 'myproject' / 'raporty')
        result = self.scan()
        self.assertFalse(result['truncated'], result)
        self.assertEqual(set(item['path'] for item in result['matches']),
                         {str(p.resolve()) for p in (one, two, three)})

    def test_directory_limit_still_blocks_incomplete_discovery(self):
        self.write('calibration_1.json', self.doc(), parent=self.reports)
        for i in range(8):
            (self.reports / f'archive_{i}').mkdir(parents=True)
        result = self.scan(max_directories=2)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['limit_reason'], 'directory_limit')

    def test_candidate_file_limit_still_blocks_incomplete_discovery(self):
        self.write('calibration_1.json', self.doc())
        self.write('calibration_2.json', self.doc())
        result = self.scan(max_files=1)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['limit_reason'], 'candidate_file_limit')

    def test_irrelevant_nested_calibration_filename_is_out_of_scope(self):
        target = self.write('calibration_fake.json', self.doc(),
                            parent=self.root / 'Workspace' / 'unrelated' / 'massive_data' / 'private')
        self.assertTrue(target.is_file())
        result = self.scan()
        self.assertFalse(result['truncated'])
        self.assertEqual(result['matches'], [])
        self.assertEqual(result['search_scope'], 'MANAGED_REPORT_AND_CALIBRATION_ROOTS')

    def test_training_dataset_tree_never_traversed(self):
        self.write('calibration_wrong.json', self.doc(),
                   parent=self.root / 'Workspace' / 'E-X' / '4_training_datasets' / 'dataset')
        self.assertEqual(self.scan()['matches'], [])

    def test_gui_has_async_validation_and_no_auto_export(self):
        gui = GUI.read_text(encoding='utf-8-sig')
        for token in ('discover_matching_calibrations',
                      'Znajdź zgodny manifest',
                      'exploratory_calibration.validate_manifest',
                      'executor_modal.after(0, done)',
                      'def _pending_executor_rows(',
                      'Wybierz plik ręcznie',
                      'calibration_fingerprint'):
            self.assertIn(token, gui)


if __name__ == '__main__':
    unittest.main()
