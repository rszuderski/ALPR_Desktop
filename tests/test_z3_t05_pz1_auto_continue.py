from types import SimpleNamespace
from unittest.mock import Mock, patch
import inspect

from auto_annotation_tool.gui import z3_extraction_tab_ui
from auto_annotation_tool.gui import z3_campaign_flow


def make_host():
    return SimpleNamespace(
        go_to_substep_2=Mock(),
        _return_to_t05_work_after_step3_pz1=Mock(),
        _campaign_pz2_sync_loading=False,
    )


def test_campaign_pz1_success_continues_directly_to_pz2():
    host = make_host()
    campaign = Mock()

    with patch.object(z3_extraction_tab_ui, "CAMPAIGN", campaign):
        result = z3_extraction_tab_ui._continue_campaign_pz1_to_pz2(host)

    assert result is True
    campaign.set_current_step.assert_called_once_with(3)
    campaign.set_step3_stage1_done.assert_called_once_with(True)
    campaign.set_step3_stage2_done.assert_called_once_with(False)
    campaign.set_step3_pending.assert_called_once_with()
    host.go_to_substep_2.assert_called_once_with(force=True)
    host._return_to_t05_work_after_step3_pz1.assert_not_called()


def test_campaign_pz1_direct_continue_falls_back_to_gate_on_exception():
    host = make_host()
    host.go_to_substep_2.side_effect = RuntimeError("simulated")
    host._campaign_pz2_sync_loading = True
    campaign = Mock()

    with patch.object(z3_extraction_tab_ui, "CAMPAIGN", campaign):
        result = z3_extraction_tab_ui._continue_campaign_pz1_to_pz2(host)

    assert result is False
    assert host._campaign_pz2_sync_loading is False
    host._return_to_t05_work_after_step3_pz1.assert_called_once_with()


def test_run_extraction_no_longer_roundtrips_success_through_t05():
    source = inspect.getsource(z3_extraction_tab_ui.run_extraction)

    assert source.count("_continue_campaign_pz1_to_pz2") >= 2
    assert "_commit_preview_and_open_pz2" in source
    assert "_commit_preview_and_return_t05" not in source
    assert (
        "Wyodrębnianie tablic zakończone. Otwieram PZ2 do pracy nad znakami."
        in source
    )


def test_auto_extract_copy_describes_automatic_pz2_transition():
    source = inspect.getsource(
        z3_campaign_flow.auto_progress_campaign_step3_entry
    )

    assert "Po wyodrębnieniu automatycznie otworzę PZ2." in source
    assert "Po wyodrębnieniu wrócisz do pracy bramki T05" not in source


def test_manual_return_helper_still_exists_as_separate_escape_path():
    assert callable(
        z3_campaign_flow.return_to_t05_work_after_step3_pz1
    )
