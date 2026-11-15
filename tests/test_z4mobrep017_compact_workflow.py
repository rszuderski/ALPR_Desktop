"""Z4MOBREP017: operator-first layout, accessibility, canonical science unchanged.

Only presentation is tested: the app has one primary review action, sources
remain available on demand, and all scientific tables remain reachable.
"""
from __future__ import annotations

import ast
from pathlib import Path
import tkinter as tk

import pytest

from auto_annotation_tool.gui.mobile_report_science_dashboard import ScientificDashboard, build_overview

ROOT = Path(__file__).resolve().parents[1]
BROWSER = ROOT / 'auto_annotation_tool/gui/z4_mobile_report_browser.py'
DASHBOARD = ROOT / 'auto_annotation_tool/gui/mobile_report_science_dashboard.py'


def test_browser_has_compact_page_hierarchy_and_no_old_global_toolbar():
    source = BROWSER.read_text(encoding='utf-8')
    assert 'Z4MOBREP017' in source
    assert 'Raporty ALPR' in source
    assert 'width=332, minsize=292' in source
    assert 'root.grid_rowconfigure(1, weight=1)' in source
    assert 'self.progress = ttk.Progressbar(status_bar' in source
    assert 'text="Zamknij", command=self._request_close' not in source
    assert 'text="Weryfikuj próbki"' not in source
    assert 'text="Przegląd", command=' not in source
    assert 'self.btn_import_reports.grid(row=4, column=0' in source


def test_navigation_and_source_evidence_are_preserved():
    s = BROWSER.read_text(encoding='utf-8')
    for name in ('Importuj raport z telefonu', 'Podłącz archiwum',
                 'Źródło sprawdzone', 'Szczegóły', 'Porównanie', 'Próbki',
                 '_show_validation_details', '_begin_source_recovery',
                 '_finish_source_recovery', '_draw_latency_chart',
                 'read_mobile_sample_image', 'self.technical_notebook'):
        assert name in s
    assert s.count('self.notebook.add(') == 4
    assert s.count('self.technical_notebook.add(') == 7
    assert 'self.btn_link_archive.grid_remove()' in s
    assert 'self.btn_link_archive.grid()' in s
    assert "self.btn_report_warnings = ttk.Button(detail_tools" in s


def test_single_boxed_review_cta_and_auxiliary_refresh_as_flat_control():
    s = DASHBOARD.read_text(encoding='utf-8')
    assert s.count('text="Oceń zdjęcia"') == 1
    assert s.count('text="Odśwież"') == 1
    assert 'refresh_link = tk.Button' in s
    assert 'relief=tk.FLAT' in s
    assert 'text="Pełne dane i metodologia"' not in s
    # The photo review remains available in the sample tab as a text-like link.
    assert 'text="Przejdź do oceny zdjęć →"' in BROWSER.read_text(encoding='utf-8')


def test_canonical_science_methods_still_used_without_new_formulas():
    source = DASHBOARD.read_text('utf-8')
    assert 'MobileReviewSession(bundle)' in source
    assert 'make_scientific_summary(session, session.statistics()' in source
    assert 'def build_overview(' in source
    assert 'science_from_verified_bundle(bundle)' in BROWSER.read_text(encoding='utf-8')
    assert 'read_mobile_report_bundles(report_path)' in BROWSER.read_text(encoding='utf-8')


def test_python_files_parse_and_main_browser_class_present():
    for path in (BROWSER, DASHBOARD):
        ast.parse(path.read_text(encoding='utf-8'))
    source = BROWSER.read_text('utf-8')
    assert 'class MobileReportBrowser:' in source
    assert 'def open_sample_review(self)' in source
    assert 'def _set_loading(self' in source


def test_science_no_source_never_fake_zero_result():
    report = type('R', (), {'report_id':'x','source_archive_sha256':'f'*64,'latency':{},'raw':{}})()
    view = build_overview(report)
    assert all(card['value'] == 'Brak oceny' for card in view['cards'])
    assert view['quality_verified'] is False


def _palette():
    return {'panel': '#eef2e9', 'fg': '#182818', 'muted': '#58685b',
            'accent': '#3a843f', 'border': '#afc7ae'}


def _create_window(width: int):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f'No Tk display: {exc}')
    root.geometry(f'{width}x710+0+0')
    return root


def _collect_buttons(widget):
    result=[]
    for child in widget.winfo_children():
        if isinstance(child, tk.Button) or child.winfo_class() == 'TButton':
            result.append(child)
        result.extend(_collect_buttons(child))
    return result


def test_dashboard_resizes_four_two_one_without_losing_widgets():
    root = _create_window(1250)
    try:
        view = ScientificDashboard(root,palette=_palette(),on_review=lambda:None,
                                   on_refresh=lambda:None,on_details=lambda:None)
        view.pack(fill='both',expand=True)
        root.update_idletasks();root.update()
        assert view._card_columns == 4
        assert len(view._card_frames)==4
        root.geometry('730x710+0+0')
        root.update_idletasks();root.update()
        assert view._card_columns == 2
        root.geometry('450x710+0+0')
        root.update_idletasks();root.update()
        assert view._card_columns == 1
    finally:
        root.destroy()


def test_review_and_refresh_keyboard_accessible_buttons_invoke_callbacks():
    calls=[]
    root=_create_window(1060)
    try:
        view=ScientificDashboard(root,palette=_palette(),
             on_review=lambda:calls.append('review'),
             on_refresh=lambda:calls.append('refresh'),on_details=lambda:calls.append('details'))
        view.pack(fill='both',expand=True)
        root.update()
        buttons={b.cget('text'):b for b in _collect_buttons(view)}
        assert set(buttons)=={'Oceń zdjęcia','Odśwież'}
        buttons['Oceń zdjęcia'].invoke()
        buttons['Odśwież'].invoke()
        assert calls==['review','refresh']
    finally:
        root.destroy()


def test_scientific_cards_and_times_remain_read_only_after_render():
    root=_create_window(1160)
    try:
        view=ScientificDashboard(root,palette=_palette(),
             on_review=lambda:None,on_refresh=lambda:None,on_details=lambda:None)
        view.pack(fill='both',expand=True)
        d={'status':'W trakcie', 'progress':'1 z 7', 'progress_subtitle':'1 z 8',
           'progress_ratio':1/7,
           'cards':tuple({'value':v,'note':'GT zachowane'} for v in ('0 z 1','Brak oceny','Brak oceny','Brak oceny')),
           'performance': {'mt':{'n':8,'median':'0,76 s','p95':'0,87 s'},
                           'mz':{'n':6,'median':'1,02 s','p95':'1,39 s'},
                           'pipeline':{'n':8,'median':'2,40 s','p95':'3,65 s'}},
           'frames':'8 z 15 klatek', 'warnings':(), 'source':'Źródło sprawdzone'}
        view.render(d)
        root.update()
        assert view.card_vars[0][0].get() == '0 z 1'
        assert view.card_vars[1][0].get() == 'Brak oceny'
        assert '2,40 s' in view.speed_vars['pipeline'].get()
        assert view._card_columns == 4
        assert d['cards'][0]['value']=='0 z 1'
    finally:
        root.destroy()
