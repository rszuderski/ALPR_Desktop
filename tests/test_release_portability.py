"""Portable release contracts, exercised without private data or installation."""
import importlib
import json
from pathlib import Path
import sys
import zipfile
import pytest

from auto_annotation_tool.ranking.mobile_package_experiments import read_mobile_report_bundles, ReportBundleReader

def report():
    return {'schema': 'alpr.mobile_benchmark_report.v1', 'report_id': 'portable-synthetic'}

@pytest.mark.parametrize('content', [lambda: json.dumps(report()).encode(), lambda: b'\xef\xbb\xbf  '+json.dumps([report()]).encode(), lambda: b' '*5000+json.dumps(report()).encode()])
def test_legacy_json_with_session_extension(tmp_path, content):
    path=tmp_path/'legacy.alprsession';path.write_bytes(content())
    assert read_mobile_report_bundles(path)[0].report.report_id=='portable-synthetic'

def test_real_zip_session(tmp_path):
    path=tmp_path/'valid.alprsession'
    with zipfile.ZipFile(path,'w') as z:z.writestr('report.json',json.dumps(report()))
    assert read_mobile_report_bundles(path)[0].report.report_id=='portable-synthetic'

@pytest.mark.parametrize('name,payload', [('broken.alprsession', b'PK\x03\x04{not-json'),('renamed.json', b'PK\x05\x06\0'),('unknown.alprsession', b'unknown bytes'),('empty.alprsession', b'')])
def test_binary_and_unknown_are_rejected_before_json(tmp_path, monkeypatch, name, payload):
    path=tmp_path/name;path.write_bytes(payload)
    def forbidden(*args):pytest.fail('binary entered JSON parser')
    monkeypatch.setattr(ReportBundleReader,'_read_json_bundles',forbidden)
    with pytest.raises(ValueError,match='integralności'):read_mobile_report_bundles(path)

def test_json_size_cap_is_preserved(tmp_path):
    path=tmp_path/'large.alprsession';path.write_bytes(b'{'+b' '*128)
    with pytest.raises(ValueError,match='JSON jest za duży'):ReportBundleReader(max_text_bytes=64).read_many(path)

def test_paths_do_not_depend_on_cwd_or_old_profile(tmp_path,monkeypatch):
    import alpr_runtime as runtime
    monkeypatch.setenv('ALPR_DATA_DIR',str(tmp_path/'private'))
    monkeypatch.delenv('ALPR_WORKSPACE_DIR',raising=False)
    before=runtime.workspace_root()
    other=tmp_path/'unrelated';other.mkdir();monkeypatch.chdir(other)
    assert runtime.workspace_root()==before==tmp_path/'private/Workspace'
    from auto_annotation_tool.session import SessionManager
    session=SessionManager();session.set('ui','theme','portable');session.save_session()
    assert session.session_file==tmp_path/'private/session.json'
    assert SessionManager().get('ui','theme')=='portable'

def test_frozen_bootstrap_never_installs_or_restarts(monkeypatch):
    import dependency_bootstrap as boot
    monkeypatch.setattr(sys,'frozen',True,raising=False)
    spec=boot.DEPENDENCY_SPECS[1]
    monkeypatch.setattr(boot,'probe_runtime_dependencies',lambda *a:[boot.DependencyIssue(spec,'missing')])
    monkeypatch.setattr(boot,'_show_tk_dependency_assistant',lambda *a:pytest.fail('interactive installer used'))
    with pytest.raises(RuntimeError,match='Niekompletna'):boot.ensure_runtime_dependencies()
    with pytest.raises(RuntimeError):boot._run_pip_install(['numpy'],print)
    with pytest.raises(RuntimeError):boot._restart_current_process(['main.py'])

def test_f6_uses_internal_code(monkeypatch):
    from auto_annotation_tool.gui import final180_tv as tv
    module=tv.legacy_module(tv.LEGACY_DEFAULT)
    assert module.__name__=='auto_annotation_tool.gui.f5a_tv_core'
    assert Path(module.__file__).parent==Path(tv.__file__).parent
    with pytest.raises(ValueError):tv.legacy_module(Path('unavailable-original')/'old-viewer.py')

def test_training_child_command_and_error_dialog(tmp_path,monkeypatch):
    import alpr_runtime as runtime
    monkeypatch.setattr(sys,'frozen',True,raising=False)
    assert runtime.python_module_command('auto_annotation_tool.training.training_worker','job.json')[1]=='--alpr-worker'
    monkeypatch.setenv('ALPR_DATA_DIR',str(tmp_path))
    from tkinter import messagebox
    dialogs=[];monkeypatch.setattr(messagebox,'showerror',lambda *a,**k:dialogs.append(a))
    runtime.report_startup_error(RuntimeError('controlled failure'))
    assert dialogs and 'controlled failure' in (tmp_path/'logs/startup_error.log').read_text()

def test_converter_asset_is_private_verified_and_cwd_independent(tmp_path,monkeypatch):
    import alpr_runtime as runtime
    monkeypatch.setenv('ALPR_DATA_DIR',str(tmp_path/'user'))
    monkeypatch.chdir(tmp_path)
    root=runtime.initialize_runtime()
    assert Path.cwd()==root/'runtime'
    asset=Path.cwd()/'calibration_image_sample_data_20x128x128x3_float32.npy'
    import hashlib
    assert hashlib.sha256(asset.read_bytes()).hexdigest()=='7760bb2bc187ab76c0a9249033ccfe2bf98423f075d6251d966a5deedc8a1379'
    assert not (tmp_path/asset.name).exists()
