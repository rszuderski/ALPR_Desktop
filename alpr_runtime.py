"""Filesystem and child-process policy shared by source and frozen Desktop."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import hashlib

APP_VERSION = '1.0.0-dyplom'

def is_frozen() -> bool:
    return bool(getattr(sys, 'frozen', False))

def resource_root() -> Path:
    return Path(__file__).resolve().parent

def user_data_root() -> Path:
    explicit = os.environ.get('ALPR_DATA_DIR', '').strip()
    if explicit:
        return Path(explicit).expanduser().absolute()
    local = os.environ.get('LOCALAPPDATA', '').strip()
    return (Path(local) if local else Path.home() / 'AppData' / 'Local') / 'ALPR_Desktop'

def workspace_root() -> Path:
    explicit = os.environ.get('ALPR_WORKSPACE_DIR', '').strip()
    if explicit:
        return Path(explicit).expanduser().absolute()
    settings = user_data_root() / 'paths.json'
    try:
        selected = str(json.loads(settings.read_text(encoding='utf-8')).get('workspace', '')).strip()
        if selected and Path(selected).is_absolute():
            return Path(selected)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return user_data_root() / 'Workspace'

def initialize_runtime() -> Path:
    root = user_data_root()
    root.mkdir(parents=True, exist_ok=True)
    # Some converter dependencies use cwd for their own temporary/diagnostic
    # files. Give them a private writable directory, never the installation.
    work = root / 'runtime'
    work.mkdir(parents=True, exist_ok=True)
    prepare_conversion_asset(work)
    os.chdir(work)
    for key, folder in (('MPLCONFIGDIR', 'matplotlib'), ('YOLO_CONFIG_DIR', 'ultralytics'), ('TORCH_HOME', 'torch')):
        os.environ.setdefault(key, str(root / 'cache' / folder))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    if is_frozen():
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ.setdefault('YOLO_OFFLINE', 'true')
        os.environ.setdefault('MPLBACKEND', 'TkAgg')
        if sys.stdout is None or sys.stderr is None:
            logs = root / 'logs'
            logs.mkdir(parents=True, exist_ok=True)
            stream = (logs / f'runtime_{os.getpid()}.log').open('a', encoding='utf-8', buffering=1)
            if sys.stdout is None:
                sys.stdout = stream
            if sys.stderr is None:
                sys.stderr = stream
        import ctypes
        logs = root / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        (logs / f'runtime_{os.getpid()}.json').write_text(json.dumps({
            'executable': sys.executable, 'frozen': True, 'version': APP_VERSION,
            'cwd': str(Path.cwd()), 'data_root': str(root),
            'administrator': bool(ctypes.windll.shell32.IsUserAnAdmin()) if sys.platform=='win32' else False,
        }, indent=2), encoding='utf-8')
    _install_test_isolation()
    return root

def prepare_conversion_asset(work: Path) -> None:
    name = 'calibration_image_sample_data_20x128x128x3_float32.npy'
    expected = '7760bb2bc187ab76c0a9249033ccfe2bf98423f075d6251d966a5deedc8a1379'
    source = resource_root() / 'auto_annotation_tool' / 'resources' / 'conversion' / name
    if not source.is_file():
        if is_frozen():
            raise RuntimeError('Niekompletna instalacja: brak zasobu konwertera LiteRT.')
        return
    destination = work / name
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
        return
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError('Uszkodzony zasób konwertera LiteRT: SHA256 nie zgadza się z wydaniem.')
    temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_bytes(payload)
    os.replace(temporary, destination)

_test_guard_installed = False

def _install_test_isolation() -> None:
    """Opt-in test policy inherited by training children; inactive in normal use."""
    global _test_guard_installed
    if _test_guard_installed or os.environ.get('ALPR_TEST_OFFLINE') != '1':
        return
    roots = [os.path.normcase(os.path.abspath(p)) for p in json.loads(os.environ.get('ALPR_TEST_DENY_ROOTS', '[]'))]
    def guard(event, args):
        if event == 'socket.connect':
            raise PermissionError('Offline acceptance: network unavailable')
        if event in {'open', 'os.listdir', 'os.scandir'} and args and isinstance(args[0], (str, bytes)):
            path = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
            if any(path == root or path.startswith(root + os.sep) for root in roots):
                raise PermissionError('Acceptance: original code/environment unavailable')
    sys.addaudithook(guard)
    _test_guard_installed = True

def python_module_command(module: str, *arguments: str) -> list[str]:
    launcher = [sys.executable] if is_frozen() else [sys.executable, str(resource_root() / 'main.py')]
    return [*launcher, '--alpr-worker', module, *map(str, arguments)]

def run_worker_if_requested() -> bool:
    if len(sys.argv) < 3 or sys.argv[1] != '--alpr-worker':
        return False
    module = sys.argv[2]
    allowed = {'auto_annotation_tool.training.training_worker'}
    if module not in allowed:
        raise ValueError('Nieobsługiwany moduł procesu Desktop: ' + module)
    import runpy
    sys.argv = [module, *sys.argv[3:]]
    runpy.run_module(module, run_name='__main__', alter_sys=True)
    return True

def report_startup_error(exc: BaseException) -> None:
    import traceback
    detail = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log = user_data_root() / 'logs' / 'startup_error.log'
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(detail, encoding='utf-8')
        suffix = f'\n\nSzczegóły zapisano w:\n{log}'
    except OSError:
        suffix = '\n\nNie udało się zapisać dziennika błędu.'
    if sys.stderr is not None:
        print(detail, file=sys.stderr)
    try:
        from tkinter import messagebox
        messagebox.showerror('ALPR Desktop — błąd uruchomienia', str(exc) + suffix)
    except Exception:
        if sys.platform == 'win32':
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, str(exc) + suffix, 'ALPR Desktop', 0x10)
