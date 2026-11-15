"""Z4LIB010: full verification is hash-pinned and never credited to another row."""
from pathlib import Path
from types import SimpleNamespace
import queue

import pytest

from auto_annotation_tool.gui.z4_mobile_artifact_library import (
    MobileArtifactLibrary, verify_unchanged_mobile_package,
)
from auto_annotation_tool.gui.z4_mobile_artifact_registry import sha256_file


def test_desktop_verification_model_pins_sha_without_android_approval(tmp_path):
    package = tmp_path / 'model.alprmodel'
    package.write_bytes(b'a synthetic package fixture')
    seen = []
    def validate(path, *, require_precision):
        seen.append((path, require_precision))
        assert path.read_bytes() == b'a synthetic package fixture'
    proof = verify_unchanged_mobile_package(package, kind='MODEL', model_validator=validate, bundle_validator=None)
    assert seen == [(package, True)]
    assert proof['sha256'] == sha256_file(package)
    assert proof['android_verified'] is False
    assert proof['final_test_verified'] is False


def test_desktop_verification_bundle_uses_full_bundle_validator(tmp_path):
    package = tmp_path / 'pair.alprmodel'
    package.write_bytes(b'bundle')
    def fail_model(*args, **kwargs):
        pytest.fail('Wrong validator for complete bundle')
    calls = []
    def validate_bundle(path, *, require_precision):
        calls.append((path, require_precision))
    proof = verify_unchanged_mobile_package(package, kind='PACKAGE', model_validator=fail_model,
                                             bundle_validator=validate_bundle)
    assert proof['kind'] == 'PACKAGE' and calls == [(package, True)]


def test_rename_to_raw_tflite_and_unsupported_kind_are_blocked(tmp_path):
    raw = tmp_path / 'MZ_CLEAN.tflite'
    raw.write_bytes(b'research')
    with pytest.raises(ValueError, match=r'\.alprmodel'):
        verify_unchanged_mobile_package(raw, kind='MODEL', model_validator=lambda *a, **kw: None,
                                        bundle_validator=None)
    renamed = tmp_path / 'faux.alprmodel'
    renamed.write_bytes(b'not a trusted type')
    with pytest.raises(ValueError, match='rodzaj'):
        verify_unchanged_mobile_package(renamed, kind='RESEARCH_TFLITE', model_validator=lambda *a, **kw: None,
                                        bundle_validator=None)


def test_sha_change_during_full_validator_invalidates_result(tmp_path):
    package = tmp_path / 'model.alprmodel'
    package.write_bytes(b'initial')
    def mutate(path, *, require_precision):
        with path.open('ab') as stream:
            stream.write(b' unexpectedly changed')
    with pytest.raises(ValueError, match='SHA-256 paczki zmienił'):
        verify_unchanged_mobile_package(package, kind='MODEL', model_validator=mutate,
                                        bundle_validator=None)


def test_full_validator_failure_never_reports_success(tmp_path):
    package = tmp_path / 'model.alprmodel'
    package.write_bytes(b'initial')
    def fail(_path, *, require_precision):
        raise RuntimeError('GRAPH_PRECISION_MISMATCH')
    with pytest.raises(RuntimeError, match='GRAPH_PRECISION_MISMATCH'):
        verify_unchanged_mobile_package(package, kind='MODEL', model_validator=fail,
                                        bundle_validator=None)


class Variable:
    def __init__(self, value=''):
        self.value = value
    def set(self, value):
        self.value = value
    def get(self):
        return self.value


class Tree:
    def __init__(self, selected):
        self.selected = selected
    def selection(self):
        return (self.selected,) if self.selected else ()


class Window:
    def __init__(self):
        self.calls = []
    def after(self, millis, cb):
        self.calls.append((millis, cb))


def stub_verifier(*, selected='one', active_tab='packages', ticket_id=1):
    obj = object.__new__(MobileArtifactLibrary)
    obj.by_iid = {'one': SimpleNamespace(path='C:/mt-one.alprmodel', kind='MODEL'),
                  'two': SimpleNamespace(path='C:/mt-two.alprmodel', kind='MODEL')}
    obj.art_tree = Tree(selected)
    obj.events_tab = 'events'
    obj.packages_tab = 'packages'
    obj.notebook = SimpleNamespace(select=lambda: active_tab)
    obj.busy = True
    obj.refresh_button = SimpleNamespace(configure=lambda **kwargs: None)
    obj.window = Window()
    obj.status = Variable()
    obj._verification_active = {'token': ticket_id, 'path': 'C:/mt-one.alprmodel', 'kind': 'MODEL'}
    obj._verified_selection_path = None
    obj.pipe = queue.Queue()
    obj.detail_seen = []
    obj.show_text = obj.detail_seen.append
    return obj


def result(*, success=True, token=1):
    return {'token': token, 'path': 'C:/mt-one.alprmodel', 'kind': 'MODEL',
            'ok': success, 'sha256': 'd'*64, 'detail': 'GRAPH_ERROR' if not success else 'OK'}


def test_correct_selected_row_displays_pinned_sha_only_for_desktop():
    obj = stub_verifier()
    obj.pipe.put(('verified', result()))
    obj.poll()
    assert obj.busy is False
    assert 'Kontrola na komputerze zaliczona' in obj.status.get()
    assert len(obj.detail_seen) == 1
    assert 'd'*64 in obj.detail_seen[0]
    assert 'Nie potwierdzono jeszcze uruchomienia w ALPR Mobile ani jakości' in obj.detail_seen[0]
    assert obj._verified_selection_path == 'C:/mt-one.alprmodel'


def test_stale_result_when_selection_changes_is_not_assigned_to_second_row():
    obj = stub_verifier(selected='two')
    obj.pipe.put(('verified', result()))
    obj.poll()
    assert obj.busy is False
    assert obj.detail_seen == []
    assert obj._verified_selection_path is None
    assert 'innego zaznaczenia' in obj.status.get()


def test_stale_result_when_tab_changes_is_not_assigned_to_history():
    obj = stub_verifier(selected='one', active_tab='events')
    obj.pipe.put(('verified', result()))
    obj.poll()
    assert obj.detail_seen == []
    assert 'innego zaznaczenia' in obj.status.get()


def test_stale_token_cannot_overwrite_new_session():
    obj = stub_verifier(ticket_id=2)
    obj.pipe.put(('verified', result(token=1)))
    obj.poll()
    assert obj.detail_seen == []
    assert obj._verified_selection_path is None
    assert obj.busy is True
    assert obj._verification_active['token'] == 2


def test_failed_selected_row_remains_unverified():
    obj = stub_verifier()
    obj.pipe.put(('verified', result(success=False)))
    obj.poll()
    assert len(obj.detail_seen) == 1  # Operator must see the actual failure reason.
    assert 'NIEZALICZONA' in obj.detail_seen[0] and 'GRAPH_ERROR' in obj.detail_seen[0]
    assert obj._verified_selection_path is None
    assert obj._verification_error['detail'] == 'GRAPH_ERROR'
    assert 'niezaliczona' in obj.status.get()


def test_selection_change_clears_stale_pass_indication():
    obj = stub_verifier(selected='two')
    obj._verified_selection_path = 'C:/mt-one.alprmodel'
    obj._active_tab_source = lambda: 'artifact'
    obj.by_iid['two'] = SimpleNamespace(
        path='C:/mt-two.alprmodel', kind='MODEL', state='DISCOVERED_NOT_VERIFIED',
        error='', size_bytes=100, schema='alpr.model.v1', role='MT', model_id='M2',
        variants='TFLITE/INT8', manifest={}, child_metadata={},
    )
    obj.artifact_selected()
    assert obj._verified_selection_path is None
    assert 'Wybrano inną paczkę' in obj.status.get()
