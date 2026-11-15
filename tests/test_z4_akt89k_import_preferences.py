"""Local preferences only; importer writes synthetic catalogs under tmp_path."""
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from auto_annotation_tool.session import SessionManager
from auto_annotation_tool.gui import mobile_import_preferences as prefs
from auto_annotation_tool.gui import z4_mobile_report_browser as module
from auto_annotation_tool.ranking import MobilePackageExperimentStore
from test_mobile_report_full_rows import _report_payload


@pytest.fixture
def settings(tmp_path,monkeypatch):
    settings=SessionManager(tmp_path/'user/session.json')
    settings.set('ui','theme','unrelated-theme')
    monkeypatch.setattr(prefs,'SESSION',settings)
    monkeypatch.setattr(module,'CONFIG',SimpleNamespace(DIR_7_RANKINGS=tmp_path/'rank',DIR_7_RANKINGS_MOBILE_PACKAGES=tmp_path/'rank/mobile'))
    monkeypatch.setattr(module.messagebox,'showerror',lambda *a,**k:None)
    return settings


def report_file(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(_report_payload(path.stem))
    if path.suffix=='.json':path.write_text(data,encoding='utf-8')
    else:
        with zipfile.ZipFile(path,'w') as z:z.writestr('report.json',data)
    return path


def browser_for_import(tmp_path,monkeypatch,paths):
    b=object.__new__(module.MobileReportBrowser)
    b.window=None;b._import_in_progress=False;b._import_cancel_requested=False
    b.store=MobilePackageExperimentStore(tmp_path/'catalog')
    b.bundle_by_report_id={};b.messages=[]
    b._ask_mobile_report_paths=lambda initial:tuple(paths)
    b._set_import_controls=lambda running:setattr(b,'_import_in_progress',running)
    b._set_status=lambda text,tone='info':b.messages.append((text,tone))
    b._set_loading=lambda *a,**k:None;b._set_loading_progress=lambda *a,**k:None
    b._refresh_report_list=lambda *a:None;b._show_report=lambda *a:None;b._show_empty=lambda:None
    b._run_on_ui=lambda callback:callback()
    monkeypatch.setattr(module.threading,'Thread',lambda target,**kwargs:SimpleNamespace(start=target))
    return b


def test_first_use_enabled_opening_does_not_save_and_preferences_survive_reload(tmp_path,settings,monkeypatch):
    default=tmp_path/'default';default.mkdir()
    imported=report_file(tmp_path/'DAY_INT8/PAIR_DAY_01/r01/report.json')
    assert prefs.remember_enabled() and prefs.initial_directory(default)==default
    assert not settings.session_file.exists()
    prefs.remember_successful_import([imported])
    reloaded=SessionManager(settings.session_file);monkeypatch.setattr(prefs,'SESSION',reloaded)
    assert prefs.remember_enabled() and prefs.initial_directory(default)==imported.parent
    assert reloaded.get('ui','theme')=='unrelated-theme'
    assert reloaded.get(prefs.SECTION,prefs.FOLDER)==str(imported.parent)


@pytest.mark.parametrize('suffix',['.json','.zip','.alprsession'])
def test_record_folder_only_after_actual_catalog_save(tmp_path,settings,monkeypatch,suffix):
    source=report_file(tmp_path/'source'/('report'+suffix))
    b=browser_for_import(tmp_path,monkeypatch,[source])
    original_save=b.store.save;seen=[]
    def save():
        assert not settings.get(prefs.SECTION,prefs.FOLDER)
        original_save();seen.append(True)
    b.store.save=save;b.import_report()
    assert seen==[True] and b.store.file_path.is_file() and len(b.store.reports)==1
    assert settings.get(prefs.SECTION,prefs.FOLDER)==str(source.parent)
    assert b.messages[-1][1]=='success'


@pytest.mark.parametrize('case',['cancel_picker','invalid','save_error','add_error','cancel_worker'])
def test_unsuccessful_or_cancelled_import_keeps_last_success(tmp_path,settings,monkeypatch,case):
    old=report_file(tmp_path/'old/report.json');prefs.remember_successful_import([old])
    frozen=settings.session_file.read_bytes()
    source=report_file(tmp_path/'new/report.json')
    if case=='invalid':source.write_text('{invalid',encoding='utf-8')
    b=browser_for_import(tmp_path,monkeypatch,[] if case=='cancel_picker' else [source])
    def fail(*a,**k):raise OSError('synthetic write error')
    if case=='save_error':b.store.save=fail
    if case=='add_error':b.store.add_report=fail
    if case=='cancel_worker':
        original=module.read_mobile_report_bundles
        def read(path):
            yield from original(path)
            b._import_cancel_requested=True
        monkeypatch.setattr(module,'read_mobile_report_bundles',read)
    b.import_report()
    assert settings.session_file.read_bytes()==frozen
    assert settings.get(prefs.SECTION,prefs.FOLDER)==str(old.parent)


def test_partial_batch_remembers_last_success_not_last_failed_file(tmp_path,settings,monkeypatch):
    first=report_file(tmp_path/'one/good.json');second=report_file(tmp_path/'two/also_good.zip')
    failed=tmp_path/'three/bad.json';failed.parent.mkdir();failed.write_text('not json')
    b=browser_for_import(tmp_path,monkeypatch,[first,second,failed]);b.import_report()
    assert len(b.store.reports)==2
    assert settings.get(prefs.SECTION,prefs.FOLDER)==str(second.parent)
    assert b.messages[-1][1]=='warning'


def test_disabling_deletes_path_and_success_while_off_does_not_recreate_it(tmp_path,settings,monkeypatch):
    old=report_file(tmp_path/'old/old.json');prefs.remember_successful_import([old])
    prefs.set_remember_enabled(False)
    reloaded=SessionManager(settings.session_file);monkeypatch.setattr(prefs,'SESSION',reloaded)
    assert not prefs.remember_enabled() and prefs.FOLDER not in reloaded.data[prefs.SECTION]
    default=tmp_path/'default';default.mkdir();assert prefs.initial_directory(default)==default
    new=report_file(tmp_path/'new/new.json');b=browser_for_import(tmp_path,monkeypatch,[new]);b.import_report()
    assert len(b.store.reports)==1 and prefs.FOLDER not in reloaded.data[prefs.SECTION]
    prefs.set_remember_enabled(True)
    assert prefs.initial_directory(default)==default and prefs.FOLDER not in reloaded.data[prefs.SECTION]
    b.import_report();assert reloaded.get(prefs.SECTION,prefs.FOLDER)==str(new.parent)


@pytest.mark.parametrize('case',['missing','file','denied'])
def test_unavailable_remembered_folder_uses_default_without_updating_preference(tmp_path,settings,monkeypatch,case):
    default=tmp_path/'default';default.mkdir()
    saved=tmp_path/'unavailable'
    if case=='file':saved.write_text('not a directory')
    if case=='denied':saved.mkdir()
    settings.set(prefs.SECTION,prefs.FOLDER,str(saved));settings.save_session()
    frozen=settings.session_file.read_bytes()
    if case=='denied':
        original=prefs.os.scandir
        def scan(path):
            if Path(path)==saved:raise PermissionError('synthetic access denial')
            return original(path)
        monkeypatch.setattr(prefs.os,'scandir',scan)
    assert prefs.initial_directory(default)==default
    assert settings.session_file.read_bytes()==frozen


def test_preference_write_failure_does_not_turn_success_into_import_error(tmp_path,settings,monkeypatch):
    source=report_file(tmp_path/'source/report.json');b=browser_for_import(tmp_path,monkeypatch,[source])
    def fail():raise PermissionError('synthetic preference access denial')
    monkeypatch.setattr(settings,'save_session',fail)
    b.import_report()
    assert b.store.file_path.is_file() and b.messages[-1][1]=='success'


def widgets(widget):
    yield widget
    for child in widget.winfo_children():yield from widgets(child)


def picker_host(root):
    b=object.__new__(module.MobileReportBrowser)
    b.window=root;b.palette={};b.bg='#252526';b.panel='#252526';b.fg='#f3f3f3';b.muted='#c7c7c7'
    b.border='#526378';b.accent='#4f8de3';b.error='#e74c3c';b.warning='#f1c40f';b.success='#2ecc71'
    return b


def run_picker(default,callback,size='760x500'):
    import tkinter as tk
    root=tk.Tk();root.geometry(size+'+0+0');root.withdraw()
    errors=[];root.report_callback_exception=lambda *a:errors.append(a)
    b=picker_host(root)
    b.store=MobilePackageExperimentStore(default.parent/'private-picker-store')
    def drive():
        d=next(w for w in root.winfo_children() if isinstance(w,tk.Toplevel))
        try:
            d.geometry(size+'+0+0');d.update()
            callback(d)
        finally:
            if d.winfo_exists():d.destroy()
    root.after(100,drive)
    try:
        result=b._ask_mobile_report_paths(default)
        assert not errors,errors
        return result
    finally:root.destroy()


@pytest.mark.parametrize('size',['760x500','1180x760'])
def test_native_checkbox_layout_navigation_cancel_and_toggle_persistence(tmp_path,settings,monkeypatch,size):
    import tkinter as tk
    default=tmp_path/'default';default.mkdir()
    source=report_file(tmp_path/'remembered/good.json');prefs.remember_successful_import([source])
    other=tmp_path/'browsed';other.mkdir();(other/'visible.zip').touch()
    frozen=settings.session_file.read_bytes()
    def browse(d):
        entry=next(w for w in widgets(d) if isinstance(w,tk.Entry))
        check=next(w for w in widgets(d) if isinstance(w,tk.Checkbutton))
        listing=next(w for w in widgets(d) if isinstance(w,tk.Listbox))
        assert entry.get()==str(source.parent) and bool(d.getvar(check.cget('variable')))
        assert 'good.json' in '\n'.join(listing.get(0,'end'))
        assert entry.winfo_rooty()+entry.winfo_height()<=check.winfo_rooty()
        assert check.winfo_rootx()+check.winfo_width()<=d.winfo_rootx()+d.winfo_width()
        buttons={w.cget('text'):w for w in widgets(d) if w.winfo_class()=='TButton' and w.winfo_ismapped()}
        assert set(buttons)=={'Raporty','Pobrane','Pulpit','Folder wyżej','Odśwież','Anuluj','Importuj zaznaczone',
                             'Sprawdź folder z podfolderami','Wybierz folder…'}
        assert all(w.winfo_rootx()+w.winfo_width()<=d.winfo_rootx()+d.winfo_width() for w in buttons.values())
        assert listing.cget('selectmode')=='extended'
        buttons['Raporty'].invoke();assert entry.get()==str(default)
        entry.delete(0,'end');entry.insert(0,str(other));entry.focus_force();d.update();entry.event_generate('<Return>');d.update()
        assert 'visible.zip' in '\n'.join(listing.get(0,'end'))
        buttons['Anuluj'].invoke()
    assert run_picker(default,browse,size)==()
    assert settings.session_file.read_bytes()==frozen
    def disable(d):
        check=next(w for w in widgets(d) if isinstance(w,tk.Checkbutton));check.invoke()
        next(w for w in widgets(d) if w.winfo_class()=='TButton' and w.cget('text')=='Anuluj').invoke()
    assert run_picker(default,disable,size)==()
    monkeypatch.setattr(prefs,'SESSION',SessionManager(settings.session_file))
    def restored(d):
        entry=next(w for w in widgets(d) if isinstance(w,tk.Entry))
        check=next(w for w in widgets(d) if isinstance(w,tk.Checkbutton))
        assert entry.get()==str(default) and not bool(d.getvar(check.cget('variable')))
        check.invoke()  # Re-enable, then cancel: state survives, folder stays absent.
    run_picker(default,restored,size)
    final=SessionManager(settings.session_file)
    assert final.get(prefs.SECTION,prefs.ENABLED) is True and prefs.FOLDER not in final.data[prefs.SECTION]


def test_native_selection_without_completed_import_does_not_save_folder(tmp_path,settings):
    import tkinter as tk
    default=tmp_path/'default';default.mkdir()
    selected=report_file(default/'report.json')
    def accept(d):
        listing=next(w for w in widgets(d) if isinstance(w,tk.Listbox))
        i=next(i for i,row in enumerate(listing.get(0,'end')) if 'report.json' in row)
        listing.selection_set(i)
        next(w for w in widgets(d) if w.winfo_class()=='TButton' and w.cget('text')=='Importuj zaznaczone').invoke()
    assert run_picker(default,accept)==(selected,)
    assert not settings.session_file.exists() and not settings.get(prefs.SECTION,prefs.FOLDER)


@pytest.mark.parametrize('case',['missing','access_lost_after_initial_check'])
def test_native_unavailable_start_falls_back_and_does_not_rewrite_history(tmp_path,settings,monkeypatch,case):
    import tkinter as tk
    default=tmp_path/'default';default.mkdir();report_file(default/'fallback.json')
    saved=tmp_path/'saved'
    if case=='access_lost_after_initial_check':
        saved.mkdir();original=Path.iterdir
        def read(path):
            if path==saved:raise PermissionError('synthetic access lost between checking and listing')
            return original(path)
        monkeypatch.setattr(Path,'iterdir',read)
    settings.set(prefs.SECTION,prefs.FOLDER,str(saved));settings.save_session()
    frozen=settings.session_file.read_bytes()
    def verify(d):
        entry=next(w for w in widgets(d) if isinstance(w,tk.Entry))
        listing=next(w for w in widgets(d) if isinstance(w,tk.Listbox))
        assert entry.get()==str(default)
        assert 'fallback.json' in '\n'.join(listing.get(0,'end'))
    assert run_picker(default,verify)==()
    assert settings.session_file.read_bytes()==frozen
