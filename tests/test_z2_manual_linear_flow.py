from types import SimpleNamespace

from auto_annotation_tool.gui.z2_free_mode_flow import build_z2_cta_state_free_mode


class DummyVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def make_host(*, input_ready=False, template_ready=False):
    select_input = object()
    start_annotation = object()
    go_next = object()
    return SimpleNamespace(
        is_processing=False,
        _dataset_export_completed=False,
        _manual_review_export_ready=False,
        _manual_review_active=False,
        _manual_template_ready_for_review=template_ready,
        gt_capture_enabled_var=DummyVar(True),
        _get_manual_entry_mode=lambda: "new",
        _manual_vehicle_assist_enabled=lambda: False,
        _is_workflow_step_complete=lambda step: bool(input_ready),
        _select_input_dir=select_input,
        _start_annotation=start_annotation,
        _go_to_next_workflow_step=go_next,
        _get_plate_dataset_export_approval_state=lambda: {"ok": False},
    )


def cta(host, *, step, input_ready=False, run_exists=False):
    return build_z2_cta_state_free_mode(
        host,
        route="manual",
        current_step=step,
        free_mode_screen="workflow",
        show_workflow_steps=True,
        has_existing_run=run_exists,
        input_dir_ready=input_ready,
        auto_setup_pending=False,
        auto_vehicle_choice="skip",
        manual_setup=True,
        manual_run_already_created=run_exists,
    )


def test_new_manual_entry_uses_choose_images_not_dalej():
    host = make_host()
    state = cta(host, step="manual_entry")

    assert state.show_start_controls is True
    assert state.start_enabled is True
    assert state.start_text == "Wybierz obrazy"
    assert state.start_command is host._select_input_dir
    assert state.next_enabled is False
    assert state.next_text == ""


def test_after_images_primary_action_is_start_annotation_without_dalej():
    host = make_host(input_ready=True)
    state = cta(host, step="manual_start", input_ready=True)

    assert state.show_start_controls is True
    assert state.start_enabled is True
    assert state.start_text == "Rozpocznij anotację"
    assert state.start_command is host._start_annotation
    assert state.next_enabled is False
    assert state.next_text == ""


def test_back_from_correction_returns_with_primary_resume_action():
    host = make_host(input_ready=True, template_ready=True)
    state = cta(
        host,
        step="manual_start",
        input_ready=True,
        run_exists=True,
    )

    assert state.start_enabled is True
    assert state.start_text == "Wróć do korekty"
    assert state.start_command is host._go_to_next_workflow_step
    assert state.next_enabled is False
    assert state.next_text == ""


def test_source_contract_auto_enters_manual_review_after_creation():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    process = (
        root / "auto_annotation_tool/gui/z2_annotation_process.py"
    ).read_text(encoding="utf-8-sig")

    assert "hold_manual_template_on_start = False" in process
    assert 'self.free_mode_screen_var.set("manual_review")' in process
    assert 'self._set_workflow_step("manual_start", refresh_detection_ui=True)' in process
    assert "Kliknij Dalej w karcie" not in process
