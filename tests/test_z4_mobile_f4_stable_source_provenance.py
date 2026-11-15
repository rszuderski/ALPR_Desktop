#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4LIB021: final F4 ZIP must reference stable, SHA-pinned originals, not deleted temp snapshots."""
from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'auto_annotation_tool/gui/z4_mobile_f4_atomic_publish.py'
spec = importlib.util.spec_from_file_location('f4_pinned_origins_test', MODULE)
f4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f4)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Request:
    destination: Path
    metadata: dict
    plate_package: Path
    character_package: Path
    plate_request: object | None = None
    character_request: object | None = None
    vehicle_request: object | None = None
    vehicle_package: Path | None = None


class F4OriginalSourceProofs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.original = {
            'plate': self.folder / 'MT_20261012_F4_INT8.alprmodel',
            'character': self.folder / 'MZ_DAY_SIGNED_INT8.alprmodel',
        }
        self.snapshots = {
            'plate': self.folder / 'temporary_mt_snapshot.alprmodel',
            'character': self.folder / 'temporary_mz_snapshot.alprmodel',
        }
        for role in self.original:
            self.original[role].write_bytes(f'validated-{role}-F4-original-synthetic'.encode())
            self.snapshots[role].write_bytes(self.original[role].read_bytes())
        self.proofs = {
            f'reused_verified_{label}': {
                'path': str(self.original[role].resolve()),
                'package_sha256': digest(self.original[role]),
                'checkpoint_sha256': 'a'*64,
                'precision': 'int8',
            }
            for role,label in (('plate','mt'),('character','mz'))
        }
        self.final = self.folder / 'F4_PAIR_INT8.alprmodel'
        self.request = Request(self.final, {
            'f4_zero_conversion_only': True,
            **self.proofs,
        },self.snapshots['plate'],self.snapshots['character'])
        self.recorded = []
        self.manifest_mutator = None

    def fake_export(self, request: Request):
        roles = ('plate','character')
        manifest = {
            'schema': 'alpr.package.v1',
            'metadata': copy.deepcopy(request.metadata),
            'models': {
                role: {
                    'role': role,
                    'package_file': f'models/{role}/model.alprmodel',
                    'sha256': {f'models/{role}/model.alprmodel':digest(self.snapshots[role])},
                }
                for role in roles
            },
            'export_reproducibility': {
                'package': {'destination': {'path': str(request.destination.resolve()),
                                            'name':request.destination.name,
                                            'exists':False,'is_file':False,'is_dir':False}},
                'models': {
                    role: {
                        'source_kind':'existing_package',
                        'source_package': {
                            'path':str(self.snapshots[role].resolve()),
                            'name':self.snapshots[role].name,
                            'sha256':digest(self.snapshots[role]),
                            'exists':True,'is_file':True,'is_dir':False,
                        },
                    }
                    for role in roles
                },
            },
        }
        if self.manifest_mutator:
            self.manifest_mutator(manifest)
        with zipfile.ZipFile(request.destination,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('manifest.json',json.dumps(manifest))
            for role in roles:
                archive.writestr(f'models/{role}/model.alprmodel',self.snapshots[role].read_bytes())
        return request.destination

    def validate(self,path):
        with zipfile.ZipFile(path,'r') as archive:
            root=json.loads(archive.read('manifest.json'))
            self.recorded.append(root)
            self.assertEqual(root['schema'],'alpr.package.v1')
            self.assertEqual(archive.testzip(),None)

    def run_export(self,validator=None):
        return f4.export_f4_atomic_no_overwrite(
            self.request, export_fn=self.fake_export,
            validate_final=validator or self.validate,
        )

    def test_rewrites_both_ephemeral_snapshots_into_sha_pinned_originals(self):
        result=self.run_export()
        self.assertEqual(result,self.final)
        with zipfile.ZipFile(result,'r') as archive:
            root=json.loads(archive.read('manifest.json'))
            src=root['export_reproducibility']['models']
            for role in ('plate','character'):
                entry=src[role]['source_package']
                self.assertEqual(entry['path'],str(self.original[role].resolve()))
                self.assertEqual(entry['name'],self.original[role].name)
                self.assertEqual(entry['sha256'],digest(self.original[role]))
                self.assertTrue(entry['exists'] and entry['is_file'])
                self.assertEqual(archive.read(f'models/{role}/model.alprmodel'),self.original[role].read_bytes())
        self.assertEqual(self.recorded[0],root)
        self.assertFalse(self.original['plate'].is_symlink())
        self.assertEqual(len(list(self.folder.glob('.*.f4-*'))),0)

    def test_wrong_original_digest_blocked_without_publish(self):
        self.original['plate'].write_bytes(b'changed after pin')
        with self.assertRaisesRegex(f4.F4AtomicPublishError,'zmieniła się'):
            self.run_export()
        self.assertFalse(self.final.exists())

    def test_embedded_model_differs_from_original_blocked(self):
        self.snapshots['character'].write_bytes(b'substituted MZ')
        with self.assertRaisesRegex(f4.F4AtomicPublishError,'SHA zagnieżdżonej paczki'):
            self.run_export()
        self.assertFalse(self.final.exists())

    def test_root_hash_is_required_for_original_source(self):
        def bad(root):
            root['models']['character']['sha256']['models/character/model.alprmodel']='f'*64
        self.manifest_mutator=bad
        with self.assertRaisesRegex(f4.F4AtomicPublishError,'SHA zagnieżdżonej paczki'):
            self.run_export()
        self.assertFalse(self.final.exists())

    def test_missing_one_proof_is_blocked(self):
        partial=dict(self.request.metadata)
        del partial['reused_verified_mz']
        self.request=replace(self.request,metadata=partial)
        with self.assertRaisesRegex(f4.F4AtomicPublishError,'brak jednego'):
            self.run_export()
        self.assertFalse(self.final.exists())

    def test_root_metadata_cannot_disagree_with_operator_pin(self):
        def mutate(root):
            root['metadata']['reused_verified_mt']['package_sha256']='0'*64
        self.manifest_mutator=mutate
        with self.assertRaisesRegex(f4.F4AtomicPublishError,'SHA przypiętego źródła'):
            self.run_export()
        self.assertFalse(self.final.exists())

    def test_temporary_source_proof_must_match_nested_model(self):
        def mutate(root):
            root['export_reproducibility']['models']['plate']['source_package']['sha256']='f'*64
        self.manifest_mutator=mutate
        with self.assertRaisesRegex(f4.F4AtomicPublishError,'dowód migawki'):
            self.run_export()
        self.assertFalse(self.final.exists())

    def test_missing_exporter_source_provenance_is_rejected(self):
        def mutate(root):
            del root['export_reproducibility']['models']['plate']['source_package']
        self.manifest_mutator=mutate
        with self.assertRaisesRegex(f4.F4AtomicPublishError,'dowód migawki'):
            self.run_export()
        self.assertFalse(self.final.exists())

    def test_original_changed_while_full_zip_is_validated_blocks_publish(self):
        def mutate_original_after_validation(path):
            self.validate(path)
            self.original['character'].write_bytes(b'changed after zip validation')
        with self.assertRaisesRegex(f4.F4AtomicPublishError,'podczas walidacji'):
            self.run_export(validator=mutate_original_after_validation)
        self.assertFalse(self.final.exists())
        self.assertTrue(self.snapshots['character'].is_file())

    def test_source_proofs_still_do_not_claim_android_approval(self):
        self.run_export()
        with zipfile.ZipFile(self.final) as archive:
            manifest=json.loads(archive.read('manifest.json'))
            self.assertNotIn('android_import_approved',manifest)
            self.assertNotIn('f4_experiment_performed',manifest)


if __name__=='__main__':
    unittest.main()
