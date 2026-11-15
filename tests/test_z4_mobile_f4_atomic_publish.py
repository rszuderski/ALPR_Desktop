#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F4 no-clobber publication: fake validated ZIP, no model conversion or GPU."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest import mock
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'auto_annotation_tool/gui/z4_mobile_f4_atomic_publish.py'
GUI = ROOT / 'auto_annotation_tool/gui/z4_model_export.py'
spec = importlib.util.spec_from_file_location('f4_atomic_publish_test', HELPER)
f4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f4)


@dataclass(frozen=True)
class Request:
    destination: Path
    metadata: dict
    plate_package: Path | None = None
    character_package: Path | None = None
    plate_request: object | None = None
    character_request: object | None = None
    vehicle_request: object | None = None
    vehicle_package: Path | None = None


class F4AtomicPublishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.final = self.root / 'F4_20261014_INT8.alprmodel'
        self.mt = self.root / 'mt.alprmodel'
        self.mz = self.root / 'mz.alprmodel'
        self.mt.write_bytes(b'MT frozen synthetic bytes')
        self.mz.write_bytes(b'MZ frozen synthetic bytes')
        self.request = Request(
            self.final,
            {'f4_zero_conversion_only': True, 'selected': ['MT', 'MZ']},
            plate_package=self.mt, character_package=self.mz,
        )
        self.calls = []
        self.validation_calls = []
        self.validated_hashes = []

    def fake_final_validator(self, package):
        """Validate the rewritten final ZIP in place, not the exporter stage."""
        package = Path(package)
        self.assertTrue(package.is_file())
        self.assertFalse(self.final.exists())
        with zipfile.ZipFile(package) as archive:
            self.assertEqual(archive.testzip(), None)
            manifest = json.loads(archive.read('manifest.json'))
            self.assertEqual(manifest['schema'], 'alpr.package.v1')
            self.assertEqual(
                manifest['export_reproducibility']['package']['destination']['path'],
                str(self.final.resolve()),
            )
        self.validation_calls.append(package)
        self.validated_hashes.append(hashlib.sha256(package.read_bytes()).hexdigest())


    def fake_export(self, request, *, race=None, wrong_destination=False, corrupt=False):
        self.calls.append(request.destination)
        if race is not None:
            race()
        if corrupt:
            request.destination.write_bytes(b'not-zip')
            return request.destination
        root = {
            'schema': 'alpr.package.v1',
            'export_reproducibility': {
                'package': {'destination': {'path': str(request.destination if not wrong_destination
                                                    else self.root / 'another.alprmodel'),
                                            'name': request.destination.name,
                                            'exists': False, 'is_file': False, 'is_dir': False}},
            },
            'models': {'plate': {'role': 'plate'}, 'character': {'role': 'character'}},
            'metadata': {'research': 'not Android approved'},
        }
        with zipfile.ZipFile(request.destination, 'w', compression=zipfile.ZIP_DEFLATED) as out:
            out.writestr('manifest.json', json.dumps(root))
            out.writestr('models/plate/model.alprmodel', self.mt.read_bytes())
            out.writestr('models/character/model.alprmodel', self.mz.read_bytes())
            out.writestr('models/plate/manifest.json', '{"role":"plate"}')
        return request.destination

    def test_atomic_publish_updates_root_provenance_and_preserves_children(self):
        result = f4.export_f4_atomic_no_overwrite(self.request, validate_final=self.fake_final_validator, export_fn=self.fake_export)
        self.assertEqual(result, self.final)
        self.assertEqual(len(self.calls), 1)
        self.assertNotEqual(self.calls[0], self.final)
        with zipfile.ZipFile(self.final) as zip_file:
            root = json.loads(zip_file.read('manifest.json'))
            self.assertEqual(root['export_reproducibility']['package']['destination']['path'], str(self.final.resolve()))
            self.assertEqual(root['export_reproducibility']['package']['destination']['name'], self.final.name)
            self.assertEqual(zip_file.read('models/plate/model.alprmodel'), self.mt.read_bytes())
            self.assertEqual(zip_file.read('models/character/model.alprmodel'), self.mz.read_bytes())
            self.assertEqual(zip_file.testzip(), None)
        self.assertEqual(self.mt.read_bytes(), b'MT frozen synthetic bytes')
        self.assertEqual(self.mz.read_bytes(), b'MZ frozen synthetic bytes')
        self.assertEqual(list(self.root.glob('.*.f4-*')), [])
        # A second export cannot replace a previously published archive.
        old_sha = hashlib.sha256(self.final.read_bytes()).hexdigest()
        with self.assertRaisesRegex(f4.F4AtomicPublishError, 'już istnieje'):
            f4.export_f4_atomic_no_overwrite(self.request, validate_final=self.fake_final_validator, export_fn=self.fake_export)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(hashlib.sha256(self.final.read_bytes()).hexdigest(), old_sha)

    def test_existing_destination_never_overwritten(self):
        self.final.write_bytes(b'previous F4 result')
        with self.assertRaisesRegex(f4.F4AtomicPublishError, 'już istnieje'):
            f4.export_f4_atomic_no_overwrite(self.request, validate_final=self.fake_final_validator, export_fn=self.fake_export)
        self.assertEqual(self.final.read_bytes(), b'previous F4 result')
        self.assertEqual(self.calls, [])

    def test_race_mid_export_preserves_newly_created_destination(self):
        def racer():
            self.final.write_bytes(b'other process result')
        with self.assertRaisesRegex(f4.F4AtomicPublishError, 'powstał podczas pakowania'):
            f4.export_f4_atomic_no_overwrite(
                self.request, validate_final=self.fake_final_validator, export_fn=lambda request: self.fake_export(request, race=racer))
        self.assertEqual(self.final.read_bytes(), b'other process result')
        self.assertEqual(list(self.root.glob('.*.f4-*')), [])

    def test_race_on_atomic_link_is_fail_closed(self):
        def racer(_src, destination):
            Path(destination).write_bytes(b'other process at link point')
            raise FileExistsError('occupied')
        with mock.patch.object(f4.os, 'link', side_effect=racer):
            with self.assertRaisesRegex(f4.F4AtomicPublishError, 'równolegle'):
                f4.export_f4_atomic_no_overwrite(self.request, validate_final=self.fake_final_validator, export_fn=self.fake_export)
        self.assertEqual(self.final.read_bytes(), b'other process at link point')
        self.assertEqual(list(self.root.glob('.*.f4-*')), [])

    def test_link_not_supported_does_not_fallback_to_replace(self):
        with mock.patch.object(f4.os, 'link', side_effect=OSError('hard links unsupported')):
            with self.assertRaisesRegex(f4.F4AtomicPublishError, 'atomowy zapis'):
                f4.export_f4_atomic_no_overwrite(self.request, validate_final=self.fake_final_validator, export_fn=self.fake_export)
        self.assertFalse(self.final.exists())
        self.assertEqual(list(self.root.glob('.*.f4-*')), [])

    def test_bad_root_reproducibility_is_rejected(self):
        with self.assertRaisesRegex(f4.F4AtomicPublishError, 'proweniencji'):
            f4.export_f4_atomic_no_overwrite(
                self.request, validate_final=self.fake_final_validator, export_fn=lambda request: self.fake_export(request, wrong_destination=True))
        self.assertFalse(self.final.exists())

    def test_bad_zip_rejected(self):
        with self.assertRaises(zipfile.BadZipFile):
            f4.export_f4_atomic_no_overwrite(
                self.request, validate_final=self.fake_final_validator, export_fn=lambda request: self.fake_export(request, corrupt=True))
        self.assertFalse(self.final.exists())
        self.assertEqual(list(self.root.glob('.*.f4-*')), [])

    def test_normal_export_and_fresh_conversion_not_allowed_in_atomic_f4(self):
        for altered in (replace(self.request, metadata={'f4_zero_conversion_only': False}),
                        replace(self.request, plate_request=object()),
                        replace(self.request, vehicle_package=self.mt)):
            with self.subTest(altered=altered), self.assertRaises(f4.F4AtomicPublishError):
                f4.export_f4_atomic_no_overwrite(altered, validate_final=self.fake_final_validator, export_fn=self.fake_export)
        self.assertEqual(self.calls, [])

    def test_symlink_destination_refused_even_when_broken(self):
        try:
            self.final.symlink_to(self.root / 'missing.alprmodel')
        except (OSError, NotImplementedError):
            self.skipTest('No symlink rights')
        with self.assertRaisesRegex(f4.F4AtomicPublishError, 'już istnieje'):
            f4.export_f4_atomic_no_overwrite(self.request, validate_final=self.fake_final_validator, export_fn=self.fake_export)
        self.assertEqual(self.calls, [])

    def test_wrong_export_result_path_refused(self):
        def wrong_result(request):
            self.fake_export(request)
            return self.mt
        with self.assertRaisesRegex(f4.F4AtomicPublishError, 'pliku pośredniego'):
            f4.export_f4_atomic_no_overwrite(self.request, validate_final=self.fake_final_validator, export_fn=wrong_result)
        self.assertFalse(self.final.exists())

    def test_exporter_failure_cleans_only_stage(self):
        def boom(request):
            raise RuntimeError('mock exporter error')
        with self.assertRaisesRegex(RuntimeError, 'mock exporter error'):
            f4.export_f4_atomic_no_overwrite(self.request, validate_final=self.fake_final_validator, export_fn=boom)
        self.assertFalse(self.final.exists())
        self.assertTrue(self.mt.is_file())
        self.assertTrue(self.mz.is_file())
        self.assertEqual(list(self.root.glob('.*.f4-*')), [])

    def test_validation_of_final_zip_runs_after_rewrite_before_publish(self):
        self.assertEqual(self.validation_calls, [])
        result = f4.export_f4_atomic_no_overwrite(
            self.request, validate_final=self.fake_final_validator, export_fn=self.fake_export,
        )
        self.assertEqual(result, self.final)
        self.assertEqual(len(self.validation_calls), 1)
        self.assertNotEqual(self.validation_calls[0], self.calls[0])
        self.assertEqual(hashlib.sha256(self.final.read_bytes()).hexdigest(),
                         self.validated_hashes[0])

    def test_missing_authoritative_validator_fails_before_export(self):
        with self.assertRaisesRegex(f4.F4AtomicPublishError, 'walidacji'):
            f4.export_f4_atomic_no_overwrite(
                self.request, export_fn=self.fake_export, validate_final=None,
            )
        self.assertEqual(self.calls, [])
        self.assertFalse(self.final.exists())

    def test_validator_exception_never_publishes_or_overwrites(self):
        def reject(_package):
            raise ValueError('SPLIT_OUTPUT_NOT_APPROVED')
        with self.assertRaisesRegex(f4.F4AtomicPublishError, 'SPLIT_OUTPUT_NOT_APPROVED'):
            f4.export_f4_atomic_no_overwrite(
                self.request, validate_final=reject, export_fn=self.fake_export,
            )
        self.assertFalse(self.final.exists())
        self.assertEqual(list(self.root.glob('.*.f4-*')), [])
        self.assertEqual(self.mt.read_bytes(), b'MT frozen synthetic bytes')
        self.assertEqual(self.mz.read_bytes(), b'MZ frozen synthetic bytes')

    def test_validator_explicit_false_blocks_publish(self):
        with self.assertRaisesRegex(f4.F4AtomicPublishError, 'walidator odrzucił'):
            f4.export_f4_atomic_no_overwrite(
                self.request, validate_final=lambda package: False, export_fn=self.fake_export,
            )
        self.assertFalse(self.final.exists())

    def test_validator_modifying_zip_during_check_blocks_publish(self):
        def mutate(package):
            with zipfile.ZipFile(package, 'a') as out:
                out.writestr('post_validation_injection.txt', 'changed')
        with self.assertRaisesRegex(f4.F4AtomicPublishError, 'zmieniło się podczas walidacji'):
            f4.export_f4_atomic_no_overwrite(
                self.request, validate_final=mutate, export_fn=self.fake_export,
            )
        self.assertFalse(self.final.exists())
        self.assertEqual(list(self.root.glob('.*.f4-*')), [])

    def test_final_validator_used_by_gui_with_require_precision(self):
        code = GUI.read_text(encoding='utf-8-sig')
        self.assertIn('validate_final=lambda candidate: package_exporter.validate_package(', code)
        self.assertIn('candidate, require_precision=True,', code)

    def test_gui_applies_no_clobber_only_to_f4(self):
        code = GUI.read_text(encoding='utf-8-sig')
        worker = code.split('            def worker() -> None:', 1)[-1]
        self.assertIn('def export_without_f4_overwrite(active_request):', worker)
        self.assertIn('export_f4_atomic_no_overwrite(', worker)
        self.assertIn('if bool((active_request.metadata or {}).get("f4_zero_conversion_only")):', worker)
        self.assertIn('return export_with_pinned_mz(active_request)', worker)
        self.assertIn('package_path = export_without_f4_overwrite(pinned_request)', worker)
        self.assertIn('package_path = export_without_f4_overwrite(request)', worker)
        self.assertIn('require_f4_no_conversion_request(request, require_fresh_destination=True)', worker)


if __name__ == '__main__':
    unittest.main()
