#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F4: no inadvertent overwrite of historical packages in the selected output.

No exporter/graph execution; tests use synthetic files, checkpoint hashes and
an immutable request facade. Final destination is checked again before worker.
"""
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
spec = importlib.util.spec_from_file_location('z4_f4_no_clobber_testing', HELPER)
f4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f4)


class F4OutputSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mt = self.root / 'mt.alprmodel'
        self.mz = self.root / 'mz.alprmodel'
        self.mt.write_bytes(b'MT frozen bytes')
        self.mz.write_bytes(b'MZ frozen bytes')
        self.destination = self.root / 'f4_result.alprmodel'
        self.request = SimpleNamespace(
            destination=self.destination,
            plate_package=self.mt,
            character_package=self.mz,
            plate_request=None, character_request=None, vehicle_request=None,
            vehicle_package=None,
            metadata={
                'reused_verified_mt': self.proof(self.mt),
                'reused_verified_mz': self.proof(self.mz),
                'f4_zero_conversion_only': True,
            }
        )

    @staticmethod
    def proof(p):
        return {'path': str(p),
                'package_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                'precision': 'int8'}

    def checked(self, final=True):
        return f4.require_f4_no_conversion_request(
            self.request, require_fresh_destination=final)

    def test_new_path_allowed_and_never_written_by_gate(self):
        before = [self.mt.read_bytes(), self.mz.read_bytes()]
        outcome = self.checked()
        self.assertEqual(outcome['precision'], 'int8')
        self.assertFalse(self.destination.exists())
        self.assertEqual(before, [self.mt.read_bytes(), self.mz.read_bytes()])

    def test_existing_unrelated_alpr_archive_is_not_overwritten(self):
        self.destination.write_bytes(b'HISTORICAL F4 RUN - IMMUTABLE')
        with self.assertRaisesRegex(f4.F4NoConversionError, 'docelowy plik już istnieje'):
            self.checked()
        self.assertEqual(self.destination.read_bytes(), b'HISTORICAL F4 RUN - IMMUTABLE')

    def test_existing_destination_of_zero_length_also_blocked(self):
        self.destination.touch()
        with self.assertRaisesRegex(f4.F4NoConversionError, 'Wybierz NOWĄ nazwę'):
            self.checked()
        self.assertEqual(self.destination.stat().st_size, 0)

    def test_default_preflight_does_not_false_block_existing_suggestion(self):
        self.destination.write_bytes(b'older-result')
        result = self.checked(final=False)
        self.assertEqual(result['precision'], 'int8')
        self.assertEqual(self.destination.read_bytes(), b'older-result')

    def test_race_between_preview_and_selected_output_rechecked(self):
        self.assertEqual(self.checked(final=False)['precision'], 'int8')
        self.destination.write_bytes(b'created-while-user-selected-save-location')
        with self.assertRaises(f4.F4NoConversionError):
            self.checked(final=True)
        self.assertEqual(self.destination.read_bytes(), b'created-while-user-selected-save-location')

    def test_existing_source_remains_rejected_independent_of_new_switch(self):
        self.request.destination = self.mt
        for final in (False, True):
            with self.subTest(final=final), self.assertRaises(f4.F4NoConversionError):
                self.checked(final=final)
        self.assertEqual(self.mt.read_bytes(), b'MT frozen bytes')

    def test_destination_directory_or_symlink_is_blocked(self):
        self.destination.mkdir()
        with self.assertRaisesRegex(f4.F4NoConversionError, 'docelowy plik już istnieje'):
            self.checked()
        self.destination.rmdir()
        self.destination.symlink_to(self.root / 'not-yet-created.alprmodel')
        with self.assertRaisesRegex(f4.F4NoConversionError, 'docelowy plik już istnieje'):
            self.checked()

    def test_output_does_not_approve_android(self):
        self.assertEqual(self.checked()['android_execution'], 'NOT_VERIFIED')

    def test_gui_checks_final_destination_before_thread_and_in_worker(self):
        content = GUI.read_text(encoding='utf-8-sig')
        form = content.split('        def export_executor_package() -> None:', 1)[1]
        selected = form.split('            set_running(True)', 1)[0]
        self.assertIn('require_f4_no_conversion_request(request, require_fresh_destination=True)', selected)
        worker = form.split('            def worker() -> None:', 1)[1]
        self.assertIn('require_f4_no_conversion_request(request, require_fresh_destination=True)', worker)
        self.assertLess(form.find('require_f4_no_conversion_request(request, require_fresh_destination=True)'),
                        form.find('set_running(True)'))
        # The earlier preview call must still use legacy/default non-final behavior.
        preview = content.split('        def _build_executor_package_request(', 1)[1].split('        def _pending_executor_rows(', 1)[0]
        self.assertIn('require_f4_no_conversion_request(bundle_request)', preview)


if __name__ == '__main__':
    unittest.main()
