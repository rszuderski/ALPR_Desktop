#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression gates for Z4 optional reuse of a verified MT INT8 package."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

MODULE_PATH = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_mobile_package_reuse.py'
spec = importlib.util.spec_from_file_location('_reuse_verified_mt_testing', MODULE_PATH)
reuse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reuse)


class ReusedMtPackageGates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.best = self.root / 'best.pt'
        self.best.write_bytes(b'fixed-synthetic-checkpoint')
        self.package = self.root / 'mt.alprmodel'
        self.validator_calls = []

    def validator(self, path, *, require_precision=False):
        self.validator_calls.append((Path(path), require_precision))
        if not require_precision:
            raise AssertionError('Full precision gate was bypassed')

    def create_package(self, *, role='plate', checkpoint_sha=None, precision='int8',
                       variants=1, imgsz=512, evidence=True, calibration=True):
        manifest = {
            'schema': 'alpr.model.v1', 'role': role, 'task': 'pose',
            'model_id': 'MT-DAY-TEST', 'labels': ['plate'],
            'source': {'checkpoint_sha256': checkpoint_sha or reuse.file_sha256(self.best)},
            'input': {'width': imgsz, 'height': imgsz, 'data_type': 'INT8'},
            'output': {'class_count': 1, 'keypoint_count': 4,
                       'confidence_threshold': .25, 'iou_threshold': .45},
            'variants': [dict(id='tflite-int8', runtime='tflite', precision=precision,
                              graph_precision={'verified': True} if evidence else None)
                         for _ in range(variants)],
        }
        if calibration:
            manifest['calibration_evidence'] = {'file': 'calibration_manifest.json',
                                                'mode': 'exploratory', 'status': 'EXPLORATORY_ONLY'}
            manifest['research_calibration'] = {'final_test_status': 'NOT_VERIFIED'}
        with zipfile.ZipFile(self.package, 'w') as archive:
            archive.writestr('manifest.json', json.dumps(manifest))

    def verify(self, *, expected_image_size=512):
        return reuse.validate_existing_mt_package(
            self.package, self.best, expected_image_size=expected_image_size,
            validate_package=self.validator)

    def test_valid_strict_reuse_and_full_precision_callback(self):
        self.create_package()
        result = self.verify()
        self.assertEqual('EXPLORATORY_ONLY', result['calibration_status'])
        self.assertEqual('int8', result['precision'])
        self.assertEqual(.25, result['confidence_threshold'])
        self.assertEqual(.45, result['iou_threshold'])
        self.assertEqual(reuse.file_sha256(self.package), result['package_sha256'])
        self.assertEqual([(self.package.resolve(), True)], self.validator_calls)

    def test_reject_wrong_checkpoint(self):
        self.create_package(checkpoint_sha='f' * 64)
        with self.assertRaisesRegex(reuse.ReusedMtPackageError, 'SHA checkpointu'):
            self.verify()

    def test_reject_character_package(self):
        self.create_package(role='character')
        with self.assertRaisesRegex(reuse.ReusedMtPackageError, 'role=plate'):
            self.verify()

    def test_reject_fp32_or_extra_variant(self):
        self.create_package(precision='fp32')
        with self.assertRaisesRegex(reuse.ReusedMtPackageError, 'int8'):
            self.verify()
        self.create_package(variants=2)
        with self.assertRaisesRegex(reuse.ReusedMtPackageError, 'jeden wariant'):
            self.verify()

    def test_reject_absent_graph_proof_or_calibration(self):
        self.create_package(evidence=False)
        with self.assertRaisesRegex(reuse.ReusedMtPackageError, 'dowodu analizy'):
            self.verify()
        self.create_package(calibration=False)
        with self.assertRaisesRegex(reuse.ReusedMtPackageError, 'kalibracji'):
            self.verify()

    def test_reject_wrong_imgsz(self):
        self.create_package(imgsz=320)
        with self.assertRaisesRegex(reuse.ReusedMtPackageError, 'imgsz'):
            self.verify()

    def test_reject_missing_source_and_missing_validator(self):
        with self.assertRaisesRegex(reuse.ReusedMtPackageError, 'brak wskazanego pakietu'):
            self.verify()
        self.create_package()
        with self.assertRaisesRegex(reuse.ReusedMtPackageError, 'brak pełnej walidacji'):
            reuse.validate_existing_mt_package(self.package, self.best, expected_image_size=512,
                                               validate_package=None)

    def test_validator_failure_blocks(self):
        self.create_package()
        def fail(*args, **kwargs):
            raise RuntimeError('STATIC_GRAPH_MIXED')
        with self.assertRaisesRegex(reuse.ReusedMtPackageError, 'STATIC_GRAPH_MIXED'):
            reuse.validate_existing_mt_package(self.package, self.best, expected_image_size=512,
                                               validate_package=fail)

    def test_gui_routes_only_mt_to_existing_package(self):
        gui = (Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_model_export.py').read_text(encoding='utf-8-sig')
        assert 'plate_package=plate_package' in gui
        assert 'plate_request=plate_request' in gui
        assert 'character_request=_build_state_request(' in gui
        assert 'validate_existing_mt_package(' in gui
        assert 'MT: zapisano zweryfikowany pakiet do migawki SHA' in gui
        assert 'with tempfile.TemporaryDirectory(prefix="alpr_mt_source_snapshot_"' in gui
        assert 'reused_verified_mt' in gui


if __name__ == '__main__':
    unittest.main()
