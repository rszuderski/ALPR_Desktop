"""Startup ordering and safe native-DPI fallback without any OS setting changes."""
from types import SimpleNamespace
import ctypes,importlib,sys
import pytest
import windows_dpi as dpi


@pytest.mark.parametrize('platform',['linux','darwin'])
def test_non_windows_does_not_call_windows_api(monkeypatch,platform):
    monkeypatch.setattr(dpi.sys,'platform',platform)
    monkeypatch.setattr(ctypes,'windll',None,raising=False)
    assert dpi.enable_windows_dpi_awareness()=='not_windows'


@pytest.mark.parametrize('result,expected',[(0,'system'),(-2147024891,'already_configured'),(0x80070005,'already_configured')])
def test_success_and_existing_manifest_never_call_legacy_api(monkeypatch,result,expected):
    calls=[]
    def modern(mode):calls.append(mode);return result
    monkeypatch.setattr(dpi.sys,'platform','win32')
    monkeypatch.setattr(ctypes,'windll',SimpleNamespace(shcore=SimpleNamespace(SetProcessDpiAwareness=modern)),raising=False)
    assert dpi.enable_windows_dpi_awareness()==expected and calls==[1]


@pytest.mark.parametrize('modern',[None,'raises','rejected'])
def test_legacy_fallback_only_if_modern_api_unavailable(monkeypatch,modern):
    calls=[]
    def legacy():calls.append('legacy');return 1
    api=SimpleNamespace(user32=SimpleNamespace(SetProcessDPIAware=legacy))
    if modern is not None:
        def configure(mode):
            if modern=='raises':raise OSError('API unavailable')
            return -2147024809
        api.shcore=SimpleNamespace(SetProcessDpiAwareness=configure)
    monkeypatch.setattr(dpi.sys,'platform','win32');monkeypatch.setattr(ctypes,'windll',api,raising=False)
    assert dpi.enable_windows_dpi_awareness()=='system_legacy' and calls==['legacy']


def test_unavailable_api_does_not_break_application_start(monkeypatch):
    monkeypatch.setattr(dpi.sys,'platform','win32')
    monkeypatch.setattr(ctypes,'windll',SimpleNamespace(),raising=False)
    assert dpi.enable_windows_dpi_awareness()=='unavailable'


def test_main_configures_dpi_before_bootstrap_can_create_tk(monkeypatch):
    events=[]
    monkeypatch.setattr(dpi,'enable_windows_dpi_awareness',lambda:events.append('dpi'))
    def probe(**kwargs):events.append('bootstrap');return False
    monkeypatch.setitem(sys.modules,'dependency_bootstrap',SimpleNamespace(ensure_runtime_dependencies=probe,get_last_probe_terminal_entries=lambda:[]))
    importlib.import_module('main').main()
    assert events==['dpi','bootstrap']
