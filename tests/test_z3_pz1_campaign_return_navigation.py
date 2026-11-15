from types import SimpleNamespace

from auto_annotation_tool.gui import z3_shared_ui
from auto_annotation_tool.gui.z3_view_models import Step3ExtractWorkflowViewModel


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value


class _Widget:
    def __init__(self, manager=""):
        self.manager = manager
        self.options = {}

    def winfo_manager(self):
        return self.manager

    def config(self, **kwargs):
        self.options.update(kwargs)

    configure = config

    def cget(self, key):
        return self.options.get(key, "")

    def grid(self, *args, **kwargs):
        self.manager = "grid"

    def grid_remove(self):
        self.manager = ""

    def pack(self, *args, **kwargs):
        self.manager = "pack"

    def pack_forget(self):
        self.manager = ""


def test_campaign_pz1_view_model_keeps_return_navigation(monkeypatch):
    monkeypatch.setattr(z3_shared_ui, "is_step3_campaign_runtime", lambda host: True)
    monkeypatch.setattr(
        z3_shared_ui.CAMPAIGN,
        "get_step3_status",
        lambda: "pending",
    )

    host = SimpleNamespace(
        _get_extract_workflow_step=lambda lightweight=True: "start",
        _get_extract_entry_mode=lambda: "continue",
        _extract_last_source_binding_result={"ok": True},
        _is_extract_preview_ready_fast=lambda: True,
        _get_preferred_z2_source_candidate=lambda: None,
        annotation_run_dir_var=_Var(r"C:\run_z2"),
        preview_dir_var=_Var(r"C:\run_pz1"),
    )

    vm = z3_shared_ui.build_step3_extract_workflow_view_model(host)

    assert vm.linear_mode is True
    assert vm.show_tab_nav is True
    assert vm.show_back_nav is True
    assert vm.show_detect_nav is False


def test_campaign_pz1_return_button_is_visible_and_enabled():
    nav_row = _Widget("pack")
    prev_btn = _Widget()
    next_btn = _Widget()
    main_nav = _Widget()
    back_btn = _Widget()
    detect_frame = _Widget("grid")

    host = SimpleNamespace(
        extract_step_nav_row=nav_row,
        extract_step_back_btn=prev_btn,
        extract_step_next_btn=next_btn,
        extract_main_nav_panel=main_nav,
        btn_back_to_wizard_step3=back_btn,
        btn_to_detect_frame=detect_frame,
        _set_button_emphasis=lambda *args, **kwargs: None,
        _refresh_campaign_step3_navigation_visibility=lambda: None,
    )

    vm = Step3ExtractWorkflowViewModel(
        route="continue",
        current_step="start",
        linear_mode=True,
        show_step_nav=False,
        prev_enabled=False,
        next_enabled=False,
        next_visible=False,
        show_tab_nav=True,
        show_back_nav=True,
        show_detect_nav=False,
        clear_detect_emphasis=False,
    )

    z3_shared_ui.refresh_extract_step_nav_buttons(host, vm)

    assert main_nav.winfo_manager() == "grid"
    assert back_btn.winfo_manager() == "grid"
    assert back_btn.options.get("state") == "normal"
    assert back_btn.options.get("text") == "Wróć do pracy T05"
    assert detect_frame.winfo_manager() == ""
