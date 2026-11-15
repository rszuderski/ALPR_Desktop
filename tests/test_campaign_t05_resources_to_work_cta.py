from pathlib import Path

from auto_annotation_tool.gui import campaign_dashboard_ui as dashboard


def test_resources_pending_review_offers_existing_work_modal():
    source = Path(dashboard.__file__).read_text(encoding="utf-8-sig")

    assert '"Przejdź do pracy T05"' in source
    assert "_open_actions(edge_key)" in source
    assert "Zaimportowane anotacje ({pending_import_review_count})" in source
    assert "czekają na sprawdzenie." in source


def test_resources_to_work_does_not_jump_directly_to_pz2():
    source = Path(dashboard.__file__).read_text(encoding="utf-8-sig")
    pos = source.index('text="Przejdź do pracy T05"')
    local = source[max(0, pos - 1400): pos + 2200]

    assert "_open_actions(edge_key)" in local
    assert "execute_campaign_graph_action" not in local
    assert 'action_text = "Wcześniejsze anotacje…"' in source
