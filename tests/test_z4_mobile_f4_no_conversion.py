#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed F4-only packaging gates. No YOLO/GPU or actual export is invoked."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'auto_annotation_tool/gui/z4_mobile_f4_no_conversion.py'
GUI = ROOT / 'auto_annotation_tool/gui/z4_model_export.py'
spec = importlib.util.spec_from_file_location('z4_f4_noconv', HELPER)
f4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f4)


class F4SelectionTests(unittest.TestCase):
    def select(self, **kwargs):
        args = dict(has_vehicle=False, use_mt=True, use_mz=True,
                    mt_precision='int8', mz_precision='int8')
        args.update(kwargs)
        return f4.require_f4_selection(**args)

    def test_pair_int8(self):
        self.assertEqual(self.select(), 'int8')

    def test_pair_fp32(self):
        self.assertEqual(self.select(mt_precision='fp32', mz_precision='fp32'), 'fp32')

    def test_missing_mt_fails_before_export(self):
        with self.assertRaisesRegex(f4.F4NoConversionError, 'MT wymaga'):
            self.select(use_mt=False)

    def test_missing_mz_fails_and_does_not_approve_raw_tflite(self):
        with self.assertRaisesRegex(f4.F4NoConversionError, 'Surowy MZ-DAY CLEAN'):
            self.select(use_mz=False)

    def test_mismatched_precision_blocked(self):
        with self.assertRaisesRegex(f4.F4NoConversionError, 'różne precyzje'):
            self.select(mz_precision='fp32')

    def test_vehicle_role_blocks_zero_conversion(self):
        with self.assertRaisesRegex(f4.F4NoConversionError, 'bez MP'):
            self.select(has_vehicle=True)

    def test_bad_precision_blocked(self):
        with self.assertRaisesRegex(f4.F4NoConversionError, 'jawnie'):
            self.select(mt_precision='mixed', mz_precision='mixed')


class F4BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mt = self.root / 'mt.alprmodel'
        self.mz = self.root / 'mz.alprmodel'
        self.mt.write_bytes(b'this is synthetic MT; full graph validation lives in caller')
        self.mz.write_bytes(b'this is synthetic MZ; full graph validation lives in caller')
        self.dest = self.root / 'alpr_result.alprmodel'
        self.request = SimpleNamespace(
            destination=self.dest,
            plate_package=self.mt,
            character_package=self.mz,
            vehicle_package=None, plate_request=None, character_request=None,
            vehicle_request=None,
            metadata={
                'reused_verified_mt': self.proof(self.mt, 'int8'),
                'reused_verified_mz': self.proof(self.mz, 'int8'),
            },
        )

    def proof(self, path: Path, precision: str):
        return {
            'path': str(path),
            'package_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'precision': precision,
            'execution_precision_android': 'UNVERIFIED',
        }

    def test_int8_pair_accepted_as_technical_bundle_only(self):
        result = f4.require_f4_no_conversion_request(self.request)
        self.assertEqual(result['precision'], 'int8')
        self.assertEqual(result['MT']['sha256'], self.proof(self.mt, 'int8')['package_sha256'])
        self.assertEqual(result['android_execution'], 'NOT_VERIFIED')
        self.assertFalse(self.dest.exists())

    def test_fp32_pair_accepted(self):
        self.request.metadata['reused_verified_mt']['precision'] = 'fp32'
        self.request.metadata['reused_verified_mz']['precision'] = 'fp32'
        self.assertEqual(f4.require_f4_no_conversion_request(self.request)['precision'], 'fp32')

    def test_any_child_conversion_request_is_blocked(self):
        for field in ('plate_request', 'character_request', 'vehicle_request'):
            setattr(self.request, field, object())
            with self.subTest(field=field), self.assertRaisesRegex(f4.F4NoConversionError, 'nowy eksport'):
                f4.require_f4_no_conversion_request(self.request)
            setattr(self.request, field, None)

    def test_vehicle_package_blocked(self):
        self.request.vehicle_package = self.mt
        with self.assertRaisesRegex(f4.F4NoConversionError, 'MP'):
            f4.require_f4_no_conversion_request(self.request)

    def test_mutated_source_sha_rejected(self):
        self.mz.write_bytes(b'changed while preparing package')
        with self.assertRaisesRegex(f4.F4NoConversionError, 'zmieniono paczkę MZ'):
            f4.require_f4_no_conversion_request(self.request)

    def test_missing_or_wrong_evidence_rejected(self):
        self.request.metadata['reused_verified_mt'] = {}
        with self.assertRaisesRegex(f4.F4NoConversionError, 'inną paczkę'):
            f4.require_f4_no_conversion_request(self.request)

    def test_other_source_identity_rejected(self):
        self.request.metadata['reused_verified_mt']['path'] = str(self.mz)
        with self.assertRaisesRegex(f4.F4NoConversionError, 'inną paczkę'):
            f4.require_f4_no_conversion_request(self.request)

    def test_source_cannot_be_output(self):
        self.request.destination = self.mt
        with self.assertRaisesRegex(f4.F4NoConversionError, 'nadpisanie modelu'):
            f4.require_f4_no_conversion_request(self.request)

    def test_mz_raw_tflite_is_not_package(self):
        self.request.character_package = self.root / 'mz.tflite'
        with self.assertRaisesRegex(f4.F4NoConversionError, 'MZ wymaga gotowej paczki'):
            f4.require_f4_no_conversion_request(self.request)

    def test_missing_role_file_rejected(self):
        self.mt.unlink()
        with self.assertRaisesRegex(f4.F4NoConversionError, 'MT nie jest'):
            f4.require_f4_no_conversion_request(self.request)

    def test_mixed_precision_evidence_rejected(self):
        self.request.metadata['reused_verified_mz']['precision'] = 'fp32'
        with self.assertRaisesRegex(f4.F4NoConversionError, 'różne precyzje'):
            f4.require_f4_no_conversion_request(self.request)

    def test_bad_output_extension_rejected(self):
        self.request.destination = self.root / 'bad.zip'
        with self.assertRaisesRegex(f4.F4NoConversionError, 'wynik'):
            f4.require_f4_no_conversion_request(self.request)

    def test_source_symlink_rejected(self):
        link = self.root / 'link.alprmodel'
        try:
            link.symlink_to(self.mt)
        except (OSError, NotImplementedError):
            self.skipTest('No symlink privileges in this environment')
        self.request.plate_package = link
        with self.assertRaisesRegex(f4.F4NoConversionError, 'zwykłym'):
            f4.require_f4_no_conversion_request(self.request)


class F4UiGateTests(unittest.TestCase):
    def test_gui_has_opt_in_to_fresh_conversion_and_preflight_gate(self):
        text = GUI.read_text(encoding='utf-8-sig')
        for token in ('Przygotuj nowy MT z wybranego best.pt',
                      'Wykorzystaj gotowy MT z biblioteki',
                      'Przygotuj nowy MZ z wybranego best.pt',
                      'Wykorzystaj gotowy MZ z biblioteki',
                      'f4_no_conversion = tk.BooleanVar(master=executor_modal, value=False)',
                      'f4_no_conversion.trace_add("write", _mark_executor_dirty)',
                      'require_f4_selection(',
                      'require_f4_no_conversion_request(bundle_request)',
                      'require_f4_no_conversion_request(request, require_fresh_destination=True)',
                      '"f4_zero_conversion_only": bool(f4_no_conversion.get())',
                      'Wykonanie ALPR Mobile nadal wymaga osobnego testu na telefonie.'):
            self.assertIn(token, text)
        self.assertIn('conversion_plan = build_conversion_plan(', text)


if __name__ == '__main__':
    unittest.main()
