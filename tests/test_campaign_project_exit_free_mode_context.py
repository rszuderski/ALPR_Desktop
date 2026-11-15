from types import SimpleNamespace

from auto_annotation_tool.gui import campaign_project_browser as browser


class AnnotationTab:
    def __init__(self):
        self.calls = []

    def clear_campaign_context(self, *, restore_free_mode_preview=True):
        self.calls.append(bool(restore_free_mode_preview))


class CharacterTab:
    def __init__(self):
        self.calls = 0

    def clear_campaign_context(self):
        self.calls += 1


class TrainingTab:
    def __init__(self):
        self.calls = 0

    def clear_campaign_context(self):
        self.calls += 1


class Host:
    def __init__(self):
        self.annotation = AnnotationTab()
        self.characters = CharacterTab()
        self.training = TrainingTab()
        self.app = SimpleNamespace(
            tabs={
                "annotation": self.annotation,
                "characters": self.characters,
                "training": self.training,
            }
        )
        self._reset_campaign_graph_runtime_state = lambda: None
        self._hide_project_loading_overlay = lambda: None


def test_lightweight_clear_does_not_leave_z2_or_z3_stale():
    host = Host()

    browser._clear_project_contexts(
        host,
        restore_free_mode_preview=False,
        hide_loading_overlay=False,
        lightweight_tab_clear=True,
    )

    assert host.annotation.calls == [False]
    assert host.characters.calls == 1
    assert host.training.calls == 1
    assert not hasattr(host.annotation, "_campaign_lightweight_context_detached")
    assert not hasattr(host.characters, "_campaign_lightweight_context_detached")


def test_regular_clear_still_forwards_restore_choice_to_z2():
    host = Host()

    browser._clear_project_contexts(
        host,
        restore_free_mode_preview=True,
        hide_loading_overlay=False,
        lightweight_tab_clear=False,
    )

    assert host.annotation.calls == [True]
    assert host.characters.calls == 1
    assert host.training.calls == 1
