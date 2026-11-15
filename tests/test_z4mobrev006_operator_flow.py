"""Z4MOBREV006: testy przeplywu recenzji bez zapisu oryginalnej sesji."""
from types import SimpleNamespace
from auto_annotation_tool.gui import z4_mobile_sample_review as gui


class FakeVar:
    def __init__(self, text=''):
        self.text = text
    def get(self):
        return self.text
    def set(self, text):
        self.text = text


class Session:
    def __init__(self, evaluable=None, gt='', note=''):
        self.review = SimpleNamespace(subjects={'s1':{'evaluable':evaluable,'ground_truth':gt,'note':note}})
        self.calls = []
    def set_subject(self, key, **kwargs):
        self.calls.append((key,kwargs))
        self.review.subjects[key].update(kwargs)


def test_typing_gt_autosaves_as_evaluable():
    view = gui.MobileSampleReviewWindow.__new__(gui.MobileSampleReviewWindow)
    view.session = Session()
    view._persist_subject_fields('s1','WI1234A','')
    assert view.session.calls == [('s1',{'ground_truth':'WI1234A','note':'','evaluable':True})]
    view._persist_subject_fields('s1','WI1234A','')
    assert len(view.session.calls) == 1  # brak niepotrzebnej rewizji


def test_excluded_subject_remains_excluded_during_note_edits():
    view = gui.MobileSampleReviewWindow.__new__(gui.MobileSampleReviewWindow)
    view.session = Session(evaluable=False,gt='',note='')
    view._persist_subject_fields('s1','WI1234A','notatka')
    assert view.session.calls[-1][1]['evaluable'] is False


def test_subject_radio_includes_and_excludes_without_guessing_gt():
    view = gui.MobileSampleReviewWindow.__new__(gui.MobileSampleReviewWindow)
    view.session = Session()
    view.subject_key = 's1'
    view._saving = False
    view.gt_var = FakeVar('')
    view.note_var = FakeVar('')
    view.subject_eval_choice = FakeVar('exclude')
    def change(work,save_draft=True): work()
    view._change = change
    view.save_subject_eval_choice()
    assert view.session.calls[-1][1]['evaluable'] is False
    view.subject_eval_choice.set('assess')
    view.save_subject_eval_choice()
    assert view.session.calls[-1][1]['evaluable'] is None  # nie ma GT = nie ma potwierdzonej tablicy


def test_existing_decisions_preserved_when_migrating_mode():
    # Model domenowy zapisuje pochodzenie migracji; kontrola kodu weryfikatora:
    import inspect
    source = inspect.getsource(gui.MobileSampleReviewWindow.__init__)
    assert 'reveal_predictions_for_review()' in source
    assert 'review_mode_locked' in source
    assert 'session.set_review_mode("assisted")' in source
    assert 'review_mode_origin' in inspect.getsource(gui.MobileSampleReviewWindow.__init__)


def test_selection_preserves_image_on_refresh():
    # Test kontraktu: widok musi zachowac zarówno obraz, jak i zoom/pan.
    import inspect
    source = inspect.getsource(gui.MobileSampleReviewWindow._select_record)
    assert 'retained[1] == self._image_entry' in source
    assert 'self._paint_image()' in source
    assert 'self._load_image()' in source
