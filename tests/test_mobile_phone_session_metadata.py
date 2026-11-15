"""Phone metadata is source text, independent of Desktop reviews and metrics."""
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import zipfile

import pytest

from auto_annotation_tool.ranking.mobile_package_experiments import (
    MobileBenchmarkReport, MobilePackageExperimentStore, read_mobile_report_bundle,
)
from auto_annotation_tool.gui.mobile_report_operator_metadata import (
    phone_session_metadata, phone_name_caption, phone_note_caption,
)
from auto_annotation_tool.gui.z4_mobile_report_browser_ux import matches_report_filter
from auto_annotation_tool.gui import z4_mobile_report_browser as gui


ABSENT = object()


def report(note=ABSENT, *, key='one'):
    raw = {'experiment': {'series_id': 'SERIA', 'scenario_id': 'SCENARIUSZ', 'session': {'id': key}}}
    if note is not ABSENT:
        raw['experiment']['notes'] = note
        raw['experiment']['session']['notes'] = note
    return MobileBenchmarkReport(report_id=key, package_id='package', variant_id='R0',
        source_archive_sha256='a'*64, source_path='not-an-operator-name.alprsession',
        measured_at='2026-11-09T08:00:00Z', raw=raw,
        experiment_index=dict(series_id='SERIA', scenario_id='SCENARIUSZ', experiment_session_id=key))


def bundle(r, payload=None):
    return NS(report=r, source_archive_sha256=r.source_archive_sha256,
        validation=NS(ok=True), report_payload=r.raw if payload is None else payload)


@pytest.mark.parametrize('value,state', [(ABSENT,'MISSING'), ('','EMPTY'), (' \n\t','EMPTY'),
    ('BLOK_B · 10x30 s · KOREKTA_K02','PRESENT'), (None,'ERROR'), (123,'ERROR'),
    ({'_preview_text':'part', '_omitted_chars':100},'TRUNCATED')])
def test_note_states_are_explicit(value,state):
    data=phone_session_metadata(report(value))
    assert data.note.state==state
    if state=='PRESENT':assert data.note.value==value
    assert data.name.state=='MISSING'


def test_missing_source_field_and_saved_index_are_distinct():
    r=report()
    assert phone_note_caption(phone_session_metadata(r))=='Brak notatki w zapisanym raporcie.'
    assert phone_note_caption(phone_session_metadata(r,bundle(r)))=='Brak notatki w archiwum'


def test_multiline_unicode_whitespace_and_long_note_are_lossless(tmp_path):
    text='  Pierwszy wiersz\nDrugi: Łódź\r\n' + 'pełna treść '*400 + '\nKONIEC_UNIKALNY  '
    r=report(text);before=copy.deepcopy(r.to_dict())
    compact=gui._compact_mobile_report_for_store(r)
    assert phone_session_metadata(compact).note.value==text
    store=MobilePackageExperimentStore(tmp_path/'store');store.add_report(compact)
    loaded=MobilePackageExperimentStore(tmp_path/'store').reports[0]
    assert phone_session_metadata(loaded).note.value==text
    assert matches_report_filter(loaded,'ŁÓDŹ koniec_unikalny')
    assert r.to_dict()==before


def test_notes_not_confused_with_later_desktop_review():
    r=report('TELEFON_ORIGINAL')
    r.raw['desktop_review']={'notes':'DESKTOP_LATER'}
    r.raw['human_review']={'comment':'DESKTOP_LATER'}
    assert phone_session_metadata(r).note.value=='TELEFON_ORIGINAL'
    assert not matches_report_filter(r,'DESKTOP_LATER')
    assert matches_report_filter(r,'TELEFON_ORIGINAL SERIA R0')
    assert matches_report_filter(r,'one package')


def test_no_invented_name_from_identifiers_filename_or_unknown_alias():
    r=report('phone note')
    r.raw.update(session_name='UNPROVEN_ALIAS',name='UNPROVEN_ALIAS',title='UNPROVEN_ALIAS')
    data=phone_session_metadata(r)
    assert data.name.state=='MISSING' and data.name.value==''
    assert phone_name_caption(data)=='Brak zapisanej nazwy sesji'
    assert not matches_report_filter(r,'UNPROVEN_ALIAS')


def test_one_original_note_field_is_sufficient_but_conflicts_fail_closed():
    r=report('one');del r.raw['experiment']['session']['notes']
    assert phone_session_metadata(r).note.value=='one'
    r.raw['experiment']['session']['notes']='two'
    data=phone_session_metadata(r)
    assert data.note.state=='ERROR' and not data.note.value
    assert 'Sprzeczne' in phone_note_caption(data)


@pytest.mark.parametrize('tamper',['sha','id','validation','missing_sha'])
def test_wrong_bundle_never_substitutes_other_sessions_note(tamper):
    r=report('one');b=bundle(r,report('OTHER').raw)
    if tamper=='sha':b.source_archive_sha256='b'*64
    if tamper=='id':b.report=replace(r,report_id='other')
    if tamper=='validation':b.validation.ok=False
    if tamper=='missing_sha':r=replace(r,source_archive_sha256='');b.source_archive_sha256=''
    assert phone_session_metadata(r,b).note.state=='ERROR'
    assert not matches_report_filter(r,'OTHER',b)


def test_legacy_truncated_index_uses_verified_bundle_without_mutation():
    r=report({'_preview_text':'short','_omitted_chars':1500});before=copy.deepcopy(r.to_dict())
    b=bundle(r,report('short\n'+'full '*500+'TAIL_SEARCH').raw)
    assert phone_session_metadata(r).note.state=='TRUNCATED'
    assert phone_session_metadata(r,b).note.state=='PRESENT'
    assert matches_report_filter(r,'TAIL_SEARCH',b)
    assert r.to_dict()==before


def test_real_archives_and_83_record_private_copy(tmp_path):
    evidence=os.environ.get('ALPR_PHONE_METADATA_AUDIT')
    if not evidence:pytest.skip('Read-only source audit fixture not supplied')
    base=Path(evidence)
    source_rows=json.loads((base/'source_audit_initial.json').read_text())
    for row in source_rows:
        path=Path(row['path']);before=path.stat();original_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        actual=read_mobile_report_bundle(path)
        assert actual.validation.ok,actual.validation.errors
        with zipfile.ZipFile(path) as archive:
            payload=json.loads(archive.read('report.json'))
        value=phone_session_metadata(actual.report,actual)
        assert value.note.value==payload['experiment']['notes']==payload['experiment']['session']['notes']
        assert value.name.state=='MISSING'
        assert original_sha==row['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.stat().st_mtime_ns==before.st_mtime_ns
    data=json.loads((base/'production_store_before.json').read_text(encoding='utf-8-sig'))
    assert len(data['reports'])==83
    root=tmp_path/'private_store';root.mkdir()
    store_path=root/MobilePackageExperimentStore.FILE_NAME
    store_path.write_text(json.dumps(data),encoding='utf-8')
    store=MobilePackageExperimentStore(root)
    before=[copy.deepcopy(r.to_dict()) for r in store.reports]
    assert len(before)==83
    for r in store.reports:
        phone_session_metadata(r);matches_report_filter(r,'BLOK_B')
    assert [r.to_dict() for r in store.reports]==before
    store.save()
    assert [r.to_dict() for r in MobilePackageExperimentStore(root).reports]==before


def test_native_browser_note_selection_search_and_refresh_do_not_write(tmp_path,monkeypatch):
    import gc,time,tkinter as tk
    from windows_dpi import enable_windows_dpi_awareness
    enable_windows_dpi_awareness()
    from auto_annotation_tool.session import SessionManager
    gc.collect()
    root=tk.Tk();root.withdraw()
    errors=[];root.report_callback_exception=lambda *args:errors.append(args)
    store=MobilePackageExperimentStore(tmp_path/'private_store')
    first=report('  Pierwszy\n'+'pełna notatka '*300+'KONIEC_TEKSTU  ',key='one')
    second=report('DRUGA_NOTATKA',key='two')
    second=replace(second,source_archive_sha256='b'*64)
    store.reports=[first,second];store.save();original=store.file_path.read_bytes()
    monkeypatch.setattr(gui,'MobilePackageExperimentStore',lambda:store)
    monkeypatch.setattr(gui,'CONFIG',NS(DIR_7_RANKINGS=tmp_path/'rank',DIR_7_RANKINGS_MOBILE_PACKAGES=store.root_dir))
    monkeypatch.setattr(gui,'SESSION',SessionManager(tmp_path/'session.json'))
    owner=NS(app=NS(root=root,palette={}),frame=root)
    browser=gui.MobileReportBrowser(owner,parent=root)
    def pump():
        end=time.monotonic()+.25
        while time.monotonic()<end:root.update();time.sleep(.01)
    try:
        browser.window.geometry('1280x800+0+0');pump()
        browser._show_report(first);pump()
        assert browser.phone_note_text.get('1.0','end-1c')==first.raw['experiment']['notes']
        assert browser.phone_note_text.cget('wrap')=='word' and browser.phone_note_text.cget('state')=='disabled'
        assert browser.hero_title_var.get()=='Brak zapisanej nazwy sesji'
        assert 'Seria: SERIA' in browser.hero_meta_var.get() and 'package' not in browser.hero_meta_var.get()
        assert tuple(browser.report_table.tree.cget('displaycolumns'))==('Wybor','Nazwa','Wariant','Powt.','Weryfikacja','Data')
        assert browser.phone_note_text.winfo_ismapped()
        assert browser.phone_note_text.winfo_rootx()+browser.phone_note_text.winfo_width() <= browser.window.winfo_rootx()+browser.window.winfo_width()
        if os.environ.get('ALPR_PHONE_METADATA_AUDIT'):
            from PIL import ImageGrab
            w=browser.window
            ImageGrab.grab(bbox=(w.winfo_rootx(),w.winfo_rooty(),w.winfo_rootx()+w.winfo_width(),w.winfo_rooty()+w.winfo_height())).save(
                Path(os.environ['ALPR_PHONE_METADATA_AUDIT'])/'browser_notes_synthetic.png')
        browser.search_var.set('KONIEC_TEKSTU');pump()
        assert len(browser.report_table.tree.get_children())==1
        browser.search_var.set('');pump();browser._show_report(second);pump()
        assert browser.phone_note_text.get('1.0','end-1c')=='DRUGA_NOTATKA'
        browser._refresh_report_list(auto_select=False);pump()
        assert browser.current_report.report_id=='two'
        assert browser.phone_note_text.get('1.0','end-1c')=='DRUGA_NOTATKA'
        browser._show_empty();assert browser.phone_note_text.get('1.0','end-1c')==''
        assert store.file_path.read_bytes()==original and not errors,errors
    finally:
        browser._request_close();root.update();root.destroy();gc.collect()


def test_filter_uses_only_saved_text_and_does_not_read_archive(monkeypatch):
    r=report('UNIKALNY SZUKANY TEKST')
    def forbidden(*a,**k):raise AssertionError('Filtering must not read files')
    monkeypatch.setattr(Path,'open',forbidden)
    for _ in range(10):assert matches_report_filter(r,'szukany unikalny')


@pytest.mark.parametrize('saved_note',[ABSENT,{'_preview_text':'short','_omitted_chars':100}])
def test_recovery_reuses_existing_attempt_guard_and_cache_without_saving(monkeypatch,saved_note):
    from auto_annotation_tool.gui.mobile_report_source_recovery import SourceLinkResult
    r=report(saved_note);original=copy.deepcopy(r.to_dict());full=bundle(r,report('RESTORED_FULL_NOTE').raw)
    full.trace_rows=();full.entries=()
    b=object.__new__(gui.MobileReportBrowser)
    b._closing=False;b._source_attempts=set();b._source_sequence=0;b._source_load_tokens={}
    b._source_failures={};b.bundle_by_report_id={};b.current_report=None;b._ordered_reports=[r]
    b._run_on_ui=lambda fn:fn();calls=[];refreshes=[]
    b._apply_report_filter=lambda **kw:refreshes.append(kw)
    monkeypatch.setattr(gui,'restore_source',lambda report,path:calls.append((report,path)) or SourceLinkResult('verified','ok',full,path))
    monkeypatch.setattr(gui.threading,'Thread',lambda target,**kw:NS(start=target))
    b._begin_source_recovery(r,'same.alprsession');b._begin_source_recovery(r,'same.alprsession')
    assert len(calls)==1 and len(refreshes)==1
    assert matches_report_filter(r,'RESTORED_FULL_NOTE',b._matching_cached_bundle(r))
    assert r.to_dict()==original  # production record stays untouched


def test_late_metadata_callback_does_not_replace_selected_sessions_text():
    from auto_annotation_tool.gui.mobile_report_source_recovery import SourceLinkResult
    first=report('FIRST');second=replace(report('SECOND',key='two'),source_archive_sha256='b'*64)
    b=object.__new__(gui.MobileReportBrowser)
    b._source_load_tokens={'one':1};b.current_report=second;b._source_failures={};b.bundle_by_report_id={}
    calls=[];b._show_report=lambda *a:calls.append(a)
    b._finish_source_recovery(first,SourceLinkResult('verified','ok',bundle(first),'old.alprsession'),1,manual=False)
    assert not calls and b.current_report is second
