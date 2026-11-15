"""Z4MOBREV004: kontrola Tk i kontraktu obrazow bez otwierania prawdziwych sesji."""
from __future__ import annotations
import tkinter as tk
import pytest
from types import SimpleNamespace
from auto_annotation_tool.gui.z4_mobile_sample_review import MobileSampleReviewWindow, _review_image_sources


@pytest.fixture(scope="module")
def tk_root():
    """Keep one Tcl/Tk interpreter for this entire native GUI test module.

    Reinitializing Tk after the first root was destroyed can fail on
    some Windows Tcl installations with `invalid command name tcl_findLibrary`.
    Windows/widgets are still destroyed independently after each test.
    """
    root = tk.Tk()
    root.withdraw()
    try:
        yield root
    finally:
        root.destroy()


def test_evidence_labels():
    attempt = {'id':'a1', 'mt_input_evidence_entry':'e/a.jpg', 'evidence_entry':'e/source.jpg'}
    sample = {'attempt_id':'a1', 'image_entry':'e/crop.jpg'}
    group = SimpleNamespace(evidence_entries={'e/a.jpg','e/b.jpg'})
    available = {'e/a.jpg','e/b.jpg','e/source.jpg','e/crop.jpg'}
    rows = _review_image_sources('sample', sample, attempt, group, [sample], available)
    labels = [label for label, _ in rows]
    assert labels[0] == 'Wejście MT — bieżąca próba', labels
    assert 'Wejście MT — bieżąca próba' in labels
    assert any('dodatkowy dowód' in label for label in labels), labels
    assert sum(label.startswith('Wycięta tablica') for label in labels) == 1, labels
    assert all('MT 1' not in label and 'MT 2' not in label for label in labels)
    assert len(rows) == len({row.get('image_entry') or row.get('evidence_entry') for _,row in rows})
    assert _review_image_sources('sample', sample, attempt, group, [sample], set()) == []


def test_sidebar_wheel_and_fixed_forms(tk_root):
    root = tk_root
    window = tk.Toplevel(root)
    window.geometry('1100x680+10+10')
    ui = MobileSampleReviewWindow.__new__(MobileSampleReviewWindow)
    ui.window = window
    ui.browser = SimpleNamespace(error='#bb4400', warning='#996600')
    ui.bg, ui.fg, ui.muted = '#f2f2eb', '#171b17', '#333333'
    ui.actions = []
    ui._image = None
    ui._photo = None
    ui._zoom_to_mt_pending = False
    try:
        ui._build()
        root.update()
        scroll = ui.review_sidebar_scroll
        # Przewijamy od poczatku takze nad Entry i Label wewnatrz panelu.
        scroll.yview_moveto(0)
        root.update()
        assert scroll.yview()[1] < 1, scroll.yview()
        ui.gt_entry.event_generate('<MouseWheel>', delta=-120)
        root.update()
        assert scroll.yview()[0] > 0, ('wheeling over entry', scroll.yview())
        old = scroll.yview()[0]
        ui.invocation_label.event_generate('<MouseWheel>', delta=-120)
        root.update()
        assert scroll.yview()[0] >= old, ('wheeling over label', scroll.yview())
        # Kolko nad podgladem nie przewija panelu z decyzjami.
        old = scroll.yview()[0]
        ui.canvas.event_generate('<MouseWheel>', delta=-120, x=120, y=120)
        root.update()
        assert scroll.yview()[0] == old, 'viewer wheel should not scroll sidebar'
        assert ui.invocation_count_radios and ui.detection_radios
        assert not hasattr(ui, 'accept_button')  # GT auto-save, no approval CTA
        assert ui.pose_edit_button.winfo_exists() and ui.pose_gt_unavailable_button.winfo_exists()
        assert not hasattr(ui, "pose_edit_checkbox")
        assert not ui.record_detail_host.winfo_ismapped()
        ui._toggle_record_details()
        root.update()
        assert ui.record_detail_host.winfo_ismapped()
        ui._toggle_record_details()
        root.update()
        assert not ui.record_detail_host.winfo_ismapped()
    finally:
        window.destroy()

def test_same_mz_result_when_switching_mt_record(tk_root):
    from PIL import Image
    root = tk_root
    window = tk.Toplevel(root)
    window.geometry('1150x720+15+15')
    ui = MobileSampleReviewWindow.__new__(MobileSampleReviewWindow)
    ui.window = window
    ui.browser = SimpleNamespace(error='#bb4400', warning='#996600')
    ui.bg, ui.fg, ui.muted = '#f2f2eb', '#171b17', '#333333'
    ui.actions = []
    ui._image = None
    ui._photo = None
    ui._view_zoom = 1.0
    ui._view_center = None
    ui._image_token = 0
    ui._image_source_size = None
    ui._image_entry = ''
    ui._zoom_to_mt_pending = False
    ui._saving = False
    ui._rendering = False
    ui._autosave_id = None
    ui._closed = False
    ui.current_record = None
    ui.current_invocation = ''
    ui.subject_key = 's1'
    ui._subject_stats = {'s1': {'evaluable_reads':1, 'mt_no_detection_invocations':0}}
    ui._build()
    # Bez watkow i zapisu – logika renderowania na kontraktowych danych.
    ui._submit = lambda work, done, saving=False: done(work())
    attempt = {'id':'a1', 'mt_status':'VALID_QUAD', 'mz_status':'NOT_RUN',
               'mt_input_evidence_entry':'e/a.jpg', 'evidence_entry':'e/a.jpg',
               'mt_detection_index':'0'}
    sample = {'id':'c1', 'attempt_id':'a1', 'image_entry':'e/crop.jpg',
              'mz_status':'READ', 'prediction':'ABC1234'}
    group = SimpleNamespace(executed=True, cancelled=False, execution_failed=False,
                            detection_count=1, execution_errors=[],
                            evidence_entries={'e/a.jpg'}, records=[attempt])
    sidecar = SimpleNamespace(subjects={}, attempt_annotations={}, sample_annotations={},
                             review_mode='assisted', review_mode_locked=False)
    class FakeSession:
        subjects = {'s1': {'legacy_identity':False, 'attempts':['a1'], 'samples':['c1']}}
        attempts = {'a1':attempt}
        samples = {'c1':sample}
        mt_invocations = {'mt-1':group}
        invocation_by_attempt = {'a1':'mt-1'}
        entry_names = {'e/a.jpg', 'e/crop.jpg'}
        attempts_available = True
        review = sidecar
        @staticmethod
        def subject_draft(key): return ''
        @staticmethod
        def predictions_visible(key): return True
        @staticmethod
        def cancelled(row): return False
        @staticmethod
        def invocation_annotation(key): return {}
        @staticmethod
        def image(row): return Image.new('RGB',(640,480),'white')
    ui.session = FakeSession()
    try:
        ui._render_subject()
        root.update()
        assert ui.current_record == ('sample','c1'), ui.current_record
        assert ui.prediction_label.cget('text') == 'ABC1234'
        assert ui.image_choice.get() == 'Wejście MT — bieżąca próba'
        iid = next(iid for iid,record in ui._record_ids.items() if record == ('attempt','a1'))
        ui.record_tree.selection_set(iid)
        root.update()
        assert ui.current_record == ('attempt','a1'), ui.current_record
        assert ui.prediction_label.cget('text') == 'ABC1234'
        assert 'powiązanego cropa' in ui.result_context.cget('text')
        assert 'Wejście MT — bieżąca próba' in ui.image_choice.cget('values')
    finally:
        window.destroy()


if __name__ == '__main__':
    test_evidence_labels()
    # The standalone probe also uses one interpreter for both widget checks.
    root = tk.Tk()
    root.withdraw()
    try:
        test_sidebar_wheel_and_fixed_forms(root)
        test_same_mz_result_when_switching_mt_record(root)
    finally:
        root.destroy()
    print('Z4MOBREV004_WIDGET_PROBE_PASS')
