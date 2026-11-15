"""Versioned photograph materials; native PhotoReference remains the GT store.

Material selection never creates GT or mobile observations. Presentation plans
and imported references are explicit copies, with source identities retained.
"""
from copy import copy, deepcopy
from pathlib import Path
import json
import os
import uuid

from PIL import Image, ImageOps
from . import photo_reference as photo
from . import final180_reference as presentation
from ..gt_pack import _atomic_write_json

SCHEMA = 'alpr.photo_material.v1'
BINDING_SCHEMA = 'alpr.photo_material_binding.v1'
IDENTITY_FIELDS = ('source_file_sha256', 'width', 'height', 'exif_orientation', 'coordinate_space')


def storage_root():
    explicit = os.environ.get('ALPR_MATERIALS_DIR', '').strip()
    from alpr_runtime import user_data_root
    return Path(explicit).expanduser().absolute() if explicit else user_data_root() / 'materials'


class DirectoryMaterial(photo.Material):
    """The existing read-only image adapter over an explicit directory snapshot."""
    readonly = False
    annotation_protocol = photo.PROTOCOL
    predictions_exposed = False

    def __init__(self, directory, *, source=None):
        self.directory = Path(directory).resolve(strict=True)
        self.sort_path = self.directory / 'material.json'
        self.tv_path = self.sort_path
        self.data = photo.read_json(self.sort_path)
        if self.data.get('schema') != SCHEMA:
            raise ValueError('Nieobsługiwany manifest materiału.')
        identity = {k: self.data[k] for k in ('schema', 'recursive', 'images', 'order', 'presentation')}
        if self.data.get('material_id') != 'material-' + presentation.fingerprint(identity):
            raise ValueError('Niezgodna suma manifestu materiału.')
        self.material_id = self.data['material_id']
        location = source if source is not None else photo.read_json(self.directory / 'location.json')['source_root']
        self.root = Path(location).resolve(strict=True)
        self.images = deepcopy(self.data['images'])
        self.order = list(self.data['order'])
        if not self.order or len(set(self.order)) != len(self.order) or set(self.order) != set(self.images):
            raise ValueError('Nieprawidłowa lista fotografii materiału.')
        self.independence = dict(status='NOT_VERIFIED', scope='SHA_IDENTIFICATION_NOT_VISUAL_TRAINING_DISJOINTNESS')
        self.proof = {str(self.sort_path): dict(sha256=photo.digest(self.sort_path), stamp=photo.stamp(self.sort_path))}
        self._verified = {}
        for key in self.order:
            row = self.images[key]
            relative = Path(row['relative_path'])
            if relative.is_absolute() or '..' in relative.parts or ':' in str(relative) or '\\' in row['relative_path']:
                raise ValueError('Nieprawidłowa ścieżka fotografii.')
            self.verify(key)

    @property
    def default_output(self):
        return self.directory / 'reference.json'

    @classmethod
    def select(cls, source, *, recursive=False, store=None):
        if not str(source).strip():
            raise ValueError('Wybierz katalog fotografii.')
        root = Path(source).resolve(strict=True)
        if not root.is_dir() or root.is_symlink() or root.is_junction():
            raise ValueError('Wybierz zwykły katalog fotografii.')
        store = Path(store or storage_root()).resolve()
        if store.is_relative_to(root):
            raise ValueError('Wybierz katalog zdjęć poza katalogiem danych aplikacji.')
        images = {}
        for path in presentation.files(root, recursive=recursive):
            before = photo.stamp(path)
            sha = photo.digest(path)
            relative = path.relative_to(root).as_posix()
            key = 'img-sha256-' + sha + '-' + presentation.fingerprint(relative)[:12]
            with Image.open(path) as original:
                raw = list(original.size)
                orientation = int(original.getexif().get(274, 1))
                image = ImageOps.exif_transpose(original)
                image.load()
                width, height = image.size
            if photo.stamp(path) != before:
                raise ValueError('Fotografia zmieniła się podczas odczytu: ' + relative)
            images[key] = dict(image_id=key, relative_path=relative, aliases=[relative],
                source_file_sha256=sha, size_bytes=before[0], width=width, height=height,
                raw_size=raw, exif_orientation=orientation, coordinate_space='exif_transposed_source_pixels')
        if not images:
            raise ValueError('Katalog nie zawiera obsługiwanych fotografii. Sprawdź opcję podkatalogów.')
        data = dict(schema=SCHEMA, recursive=bool(recursive), images=images, order=list(images),
            presentation=dict(scale='FIT', orientation='EXIF_TRANSPOSE', coordinates='exif_transposed_source_pixels'))
        data['material_id'] = 'material-' + presentation.fingerprint(data)
        directory = store / data['material_id']
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / 'material.json'
        if manifest.exists():
            if photo.read_json(manifest) != data:
                raise ValueError('Istniejący manifest materiału jest niezgodny; nie został nadpisany.')
        else:
            _atomic_write_json(manifest, data)
        material = cls(directory, source=root)
        _atomic_write_json(directory / 'location.json', dict(source_root=str(root)))
        _atomic_write_json(store / 'recent.json', dict(material_id=material.material_id, directory=str(directory)))
        return material

    def reference(self, *, create=False):
        material = copy(self)
        if self.default_output.exists():
            data = photo.read_json(self.default_output)
            # Provenance comes from persisted evidence, never a UI label.
            material.annotation_protocol = data['annotation_protocol']
            material.predictions_exposed = data['predictions_exposed']
        elif not create:
            return None
        ref = photo.PhotoReference(material)
        if create and not ref.path.exists():
            changed = deepcopy(ref.data)
            changed['material_binding'] = self.binding()
            ref._save(changed, self.order[0], 'START_REFERENCE_SESSION')
        return ref

    def binding(self):
        return dict(schema=BINDING_SCHEMA, material_id=self.material_id,
            manifest_sha256=photo.digest(self.sort_path), manifest=deepcopy(self.data))

    def annotation_context(self):
        path = self.directory / 'prediction_exposure.json'
        if not path.exists():
            return dict(predictions_exposed=False)
        data = photo.read_json(path)
        if data.get('material_id') != self.material_id or data.get('predictions_exposed') is not True:
            raise ValueError('Niezgodny zapis pochodzenia oznaczeń materiału.')
        return data

    def reference_status(self):
        try:
            for key in self.order:
                self.verify(key)
            ref = self.reference()
            if ref is None:
                return dict(state='none', label='brak', complete=0, total=len(self.order))
            ready = sum(photo.complete(ref.state(key)) for key in self.order)
            return dict(state='complete' if ready == len(self.order) else 'partial',
                label='gotowe' if ready == len(self.order) else 'w opracowaniu', complete=ready,
                total=len(self.order), reference_id=ref.data['reference_id'], revision=ref.data['revision'],
                predictions_exposed=ref.predictions_exposed)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return dict(state='invalid', label='niezgodne', complete=0, total=len(self.order), error=str(exc))


def recent_material(*, store=None):
    root = Path(store or storage_root())
    pointer = root / 'recent.json'
    if not pointer.is_file():
        return None
    data = photo.read_json(pointer)
    directory = root / data['material_id']
    if directory.parent.resolve() != root.resolve():
        raise ValueError('Nieprawidłowy identyfikator zapisanej sesji.')
    return DirectoryMaterial(directory)


def note_predictions_exposed(material, *, result_path):
    """Persist actual viewing of a source-bound model report; never erase it."""
    _atomic_write_json(material.directory/'prediction_exposure.json', dict(
        schema='alpr.photo_prediction_exposure.v1', material_id=material.material_id,
        predictions_exposed=True, result_path=str(Path(result_path).resolve()),
        result_sha256=photo.digest(Path(result_path)/'evaluation.json'), at=presentation.utc()))


def _read_reference(path):
    path = Path(path).resolve(strict=True)
    before = photo.digest(path)
    data = photo.read_json(path)
    if data.get('schema') not in (photo.SCHEMA, photo.SCHEMA_V2, photo.SCHEMA_V3):
        raise ValueError('Wybierz plik danych referencyjnych PhotoReference.')
    if type(data.get('predictions_exposed')) is not bool or not data.get('annotation_protocol'):
        raise ValueError('Referencja nie ma wiarygodnie zapisanego pochodzenia.')
    if data['annotation_protocol'] not in (photo.PROTOCOL, photo.ASSISTED_PROTOCOL, presentation.REFERENCE_PROTOCOL):
        raise ValueError('Nieobsługiwany protokół pochodzenia referencji.')
    # Reuse native validation on the imported document, without fabricating a
    # filesystem source or claiming independent verification of its old manifests.
    from types import SimpleNamespace
    validator = object.__new__(photo.PhotoReference)
    validator.annotation_protocol = data['annotation_protocol']
    validator.predictions_exposed = data['predictions_exposed']
    validator.material = SimpleNamespace(images=data['images'], independence=data['dataset_independence'],
        proof={name:dict(sha256=sha) for name, sha in data['source_manifests'].items()})
    validator._validate(data)
    if photo.digest(path) != before:
        raise ValueError('Referencja zmieniła się podczas odczytu.')
    return data, before


def reference_compatibility(material, path):
    data, sha = _read_reference(path)
    mapping = {}
    differences = []
    source_images = data['images']
    used = set()
    for key in material.order:
        row = material.images[key]
        matches = [old for old, item in source_images.items()
            if item.get('source_relative_path', item.get('relative_path')) == row['relative_path']]
        if not matches and key in source_images:
            matches = [key]
        if len(matches) != 1:
            differences.append(dict(kind='NEW_OR_AMBIGUOUS_IMAGE', path=row['relative_path']))
            continue
        old = matches[0]
        fields = [f for f in IDENTITY_FIELDS if source_images[old].get(f) != row.get(f)]
        if fields:
            differences.append(dict(kind='IMAGE_IDENTITY_OR_GEOMETRY_CONFLICT', path=row['relative_path'], fields=fields))
            continue
        material.verify(key)
        mapping[old] = key
        used.add(old)
    for old, row in source_images.items():
        if old not in used:
            differences.append(dict(kind='REFERENCE_IMAGE_NOT_MATCHED', path=row.get('source_relative_path', row['relative_path'])))
    return dict(schema='alpr.photo_reference_compatibility.v1', source_sha256=sha,
        material_id=material.material_id, mapping=mapping, differences=differences,
        compatible=not differences, predictions_exposed=data['predictions_exposed'])


def import_reference(material, path, *, accept_partial=False, expected_sha=None):
    report = reference_compatibility(material, path)
    if expected_sha is not None and report['source_sha256'] != expected_sha:
        raise ValueError('Plik GT zmienił się po porównaniu. Porównaj ponownie.')
    if report['differences'] and not accept_partial:
        raise ValueError('GT niezgodne: najpierw przejrzyj listę różnic i wybierz zgodny podzbiór.')
    if not report['mapping']:
        raise ValueError('Brak fotografii o zgodnej tożsamości i geometrii. Nie podłączono GT.')
    data, sha = _read_reference(path)
    if sha != report['source_sha256']:
        raise ValueError('Plik GT zmienił się po porównaniu.')
    previous = material.reference()
    exposed = bool(data['predictions_exposed'] or (previous is not None and previous.predictions_exposed))
    adapted = copy(material)
    adapted.annotation_protocol = data['annotation_protocol']
    adapted.predictions_exposed = exposed
    ref = object.__new__(photo.PhotoReference)
    ref.material = adapted
    ref.annotation_protocol = adapted.annotation_protocol
    ref.predictions_exposed = exposed
    ref.path = material.default_output
    ref._disk_sha = photo.digest(ref.path) if ref.path.exists() else None
    result = deepcopy(data)
    result.update(reference_id=str(uuid.uuid4()), revision=0, annotation_protocol=adapted.annotation_protocol,
        predictions_exposed=exposed, dataset_independence=adapted.independence,
        source_manifests={Path(p).name:v['sha256'] for p,v in adapted.proof.items()},
        images=deepcopy(adapted.images), reviews={}, events=[], last_image_id=adapted.order[0],
        material_binding=material.binding(), imported_from=dict(reference_id=data['reference_id'],
            revision=data['revision'], sha256=sha, annotation_protocol=data['annotation_protocol'],
            predictions_exposed=data['predictions_exposed'], source_manifests=data['source_manifests']),
        import_comparison=report)
    for old, key in report['mapping'].items():
        if old in data['reviews']:
            result['reviews'][key] = deepcopy(data['reviews'][old])
        if old == data['last_image_id']:
            result['last_image_id'] = key
    ref._validate(result)
    ref.data = deepcopy(result)
    if previous is not None:
        history = material.directory / 'reference_history'
        history.mkdir(exist_ok=True)
        backup = history / (ref._disk_sha + '.json')
        if not backup.exists():
            with backup.open('xb') as stream:
                stream.write(ref.path.read_bytes())
    ref._save(result, result['last_image_id'], 'IMPORT_REFERENCE_COPY')
    return ref, report


def export_reference(material, target):
    ref = material.reference()
    if ref is None:
        raise ValueError('Najpierw rozpocznij oznaczanie fotografii.')
    target = Path(target).resolve()
    if target.is_relative_to(material.root) or target.is_relative_to(material.directory):
        raise ValueError('Eksport zapisz poza materiałem i wewnętrzną sesją GT.')
    payload = ref.path.read_bytes()
    with target.open('xb') as stream:
        stream.write(payload)
    return photo.digest(target)


def reference_for_presentation(material, prepared):
    """Copy verified source GT into the existing frozen-image coordinate contract."""
    source = material.reference()
    if source is None:
        return None
    prepared = Path(prepared).resolve()
    doc, frozen, _ = presentation.load(prepared, strict_reference=False)
    if doc.get('material_binding', {}).get('material_id') != material.material_id:
        raise ValueError('Pokaz nie jest powiązany z tym materiałem.')
    if set(frozen.images) != set(material.images):
        raise ValueError('Zestaw pokazu ma inne fotografie.')
    for key in material.order:
        material.verify(key)
        if any(frozen.images[key].get(f) != material.images[key].get(f) for f in IDENTITY_FIELDS):
            raise ValueError('Geometria fotografii w pokazie nie odpowiada GT.')
    adapted = copy(frozen)
    adapted.predictions_exposed = source.predictions_exposed
    adapted.annotation_protocol = source.annotation_protocol
    data = deepcopy(source.data)
    data.update(images=deepcopy(frozen.images),
        source_manifests={Path(p).name:v['sha256'] for p,v in frozen.proof.items()},
        dataset_independence=frozen.independence,
        presentation_binding=dict(schema=BINDING_SCHEMA, material_id=material.material_id,
            manifest_sha256=photo.digest(prepared/'manifest.json'), source_reference_sha256=photo.digest(source.path),
            source_reference_id=source.data['reference_id'], source_revision=source.data['revision'],
            geometry='IDENTICAL_EXIF_TRANSPOSED_SOURCE_PIXELS_NO_CAMERA_PROJECTION'))
    folder = material.directory / 'presentation_references'
    folder.mkdir(exist_ok=True)
    target = folder / (presentation.fingerprint(data) + '.json')
    validator = object.__new__(photo.PhotoReference)
    validator.material = adapted
    validator.annotation_protocol = adapted.annotation_protocol
    validator.predictions_exposed = adapted.predictions_exposed
    validator._validate(data)
    if target.exists():
        if photo.read_json(target) != data:
            raise ValueError('Zapis powiązania GT został zmieniony.')
    else:
        _atomic_write_json(target, data)
    return target


def prepare_presentation(material, *, screen, dwell_seconds, root=None):
    from . import final180_markers as markers
    for key in material.order:
        material.verify(key)
    prepared = presentation.freeze_photographs(material.root, root or material.directory/'presentations',
        screen=screen, dwell_seconds=dwell_seconds, layout_version=markers.F3_LAYOUT, source_material=material)
    doc = photo.read_json(prepared/'manifest.json')
    doc['material_binding'] = material.binding()
    presentation._write_plan(prepared, doc)
    # A reference is bound explicitly by its immutable snapshot. Frozen runs can
    # later use a new snapshot without changing the original plan or Android data.
    state = material.reference_status()
    bound = reference_for_presentation(material, prepared) if state['state'] != 'invalid' else None
    _atomic_write_json(material.directory/'last_presentation.json', dict(prepared_path=str(prepared),
        manifest_sha256=photo.digest(prepared/'manifest.json'), reference_path=str(bound) if bound else None,
        reference_status=state))
    return prepared, bound


def material_for_presentation(prepared, *, store=None):
    doc = photo.read_json(Path(prepared)/'manifest.json')
    binding = doc.get('material_binding') or {}
    identifier = binding.get('material_id', '')
    if not identifier:
        return None
    root = Path(store or storage_root()).resolve()
    directory = root / identifier
    if directory.parent != root:
        raise ValueError('Nieprawidłowe powiązanie materiału z pokazem.')
    if not directory.is_dir():
        return None
    material = DirectoryMaterial(directory)
    if photo.digest(material.sort_path) != binding.get('manifest_sha256'):
        raise ValueError('Pokaz wskazuje inną wersję materiału.')
    return material
