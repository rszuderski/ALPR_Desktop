"""Z4MOBREV030 — a plain-language number-verification form.

Execute real shipped methods in an isolated GUI harness. Never open/modify a
real Android archive, reviewer sidecar, or model.
"""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
import pytest
from test_z4mobrev025_mz_lifecycle_ui import window

PATH=Path(__file__).resolve().parents[1]/"auto_annotation_tool/gui/z4_mobile_sample_review.py"
SOURCE=PATH.read_text(encoding="utf-8")


def active():
    h=window()
    h.session.mt_invocations['inv'].cancelled=False
    h.session.attempts['a']['mz_status']='READ'
    h.gt_var.set('');h.note_var.set('')
    h.session.review.subjects['s']={'ground_truth':'','evaluable':None,'note':''}
    h._change=lambda work,**kwargs:work()
    return h


def test_single_plate_uses_one_human_readable_checkbox_not_two_radios():
    assert 'Nie mogę odczytać numeru ze zdjęcia' in SOURCE
    assert 'Jaki numer widzisz na zdjęciu?' in SOURCE
    for disallowed in ('Numer czytelny — weryfikuj odczyt MZ',
                       'Numer nieczytelny — wyłącz OCR z weryfikacji',
                       'Weryfikuj odczyt numeru',
                       'Wyłącz odczyt numeru z weryfikacji'):
        assert disallowed not in SOURCE
    assert 'ttk.Checkbutton(' in SOURCE
    assert 'self.subject_unreadable_button.pack(' in SOURCE


def test_blank_gt_does_not_create_an_exclusion():
    h=active()
    h._sync_subject_eval_choice();h._sync_mz_panel_state()
    assert h.subject_unreadable_var.get() is False
    assert h.gt_entry.options['state']=='normal'
    assert h.subject_eval_choice.get()=='assess'
    assert not h.mz_note_caption.visible
    assert h.session.calls==[]


def test_check_unreadable_creates_explicit_exclusion_without_gt():
    h=active()
    h.subject_unreadable_var.set(True)
    h._on_subject_unreadable_toggle()
    assert h.subject_eval_choice.get()=='exclude'
    assert h.session.calls[-1][1]['evaluable'] is False
    assert h.session.calls[-1][1]['ground_truth']==''
    assert h.gt_entry.options['state']=='disabled'
    assert h.mz_note_caption.visible and h.note_entry.visible
    assert 'Wykrycie tablicy' in h.mz_exclusion_caption.options['text']


def test_uncheck_unreadable_without_gt_returns_to_pending():
    h=active()
    h.session.review.subjects['s']={'ground_truth':'','evaluable':False,'note':''}
    h.subject_unreadable_var.set(False)
    h._on_subject_unreadable_toggle()
    assert h.subject_eval_choice.get()=='assess'
    assert h.session.calls[-1][1]['evaluable'] is None
    assert h.gt_entry.options['state']=='normal'
    assert not h.mz_note_caption.visible


def test_known_gt_cannot_be_silently_excluded(monkeypatch):
    h=active()
    h.gt_var.set('DA76')
    h.subject_unreadable_var.set(True)
    calls=[]
    monkeypatch.setitem(h._on_subject_unreadable_toggle.__globals__, 'messagebox',
                        SimpleNamespace(showwarning=lambda *a,**kw:calls.append((a,kw))))
    h._on_subject_unreadable_toggle()
    assert h.subject_unreadable_var.get() is False
    assert len(calls)==1
    assert h.session.calls==[]
    assert h.gt_var.get()=='DA76'


def test_known_gt_automatically_enters_evaluation():
    h=active()
    h.gt_var.set('da76')
    h._persist_subject_fields('s','da76','')
    assert h.session.calls[-1][1]['evaluable'] is True
    assert h.session.calls[-1][1]['ground_truth']=='da76'


def test_older_exclusion_sidecar_restores_checkbox_without_dropping_data():
    h=active()
    h.session.review.subjects['s']={
        'ground_truth':'', 'evaluable':False,'note':'tablica rozmazana'}
    h._sync_subject_eval_choice();h._sync_mz_panel_state()
    assert h.subject_unreadable_var.get() is True
    assert h.gt_entry.options['state']=='disabled'
    assert h.note_entry.visible
    assert h.session.review.subjects['s']['note']=='tablica rozmazana'


def test_cancelled_case_blocks_checkbox_and_cannot_mutate_sidecar():
    h=active();h.session.mt_invocations['inv'].cancelled=True
    h.subject_unreadable_var.set(True)
    h._on_subject_unreadable_toggle()
    h._sync_mz_panel_state()
    assert h.subject_unreadable_button.options['state']=='disabled'
    assert h.session.calls==[]


def test_tk_details_are_real_widgets_and_do_not_break_layout():
    try: root=tk.Tk()
    except tk.TclError: pytest.skip('Requires display server')
    try:
        root.geometry('400x300+0+0')
        panel=tk.Frame(root);panel.pack(fill='both',expand=True)
        h=active()
        h.gt_entry=ttk.Entry(panel);h.gt_entry.grid(row=0,column=0,sticky='ew')
        h.subject_unreadable_button=ttk.Checkbutton(panel,text='Nie mogę odczytać numeru ze zdjęcia')
        h.subject_unreadable_button.grid(row=1,column=0,sticky='ew')
        h.mz_exclusion_caption=ttk.Label(panel,text='');h.mz_exclusion_caption.grid(row=2,column=0,sticky='ew')
        h.mz_note_caption=ttk.Label(panel,text='Powód');h.mz_note_caption.grid(row=3,column=0,sticky='ew')
        h.note_entry=ttk.Entry(panel);h.note_entry.grid(row=4,column=0,sticky='ew')
        h._sync_mz_readability_details()
        root.update()
        assert not h.mz_note_caption.winfo_ismapped()
        h.subject_unreadable_var.set(True)
        h._sync_mz_readability_details()
        root.update()
        assert h.mz_note_caption.winfo_ismapped()
        assert h.note_entry.winfo_ismapped()
        assert 'disabled' in h.gt_entry.state()
    finally:root.destroy()
