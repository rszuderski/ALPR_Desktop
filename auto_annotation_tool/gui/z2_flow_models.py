from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class _CompatModel:
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return tuple(getattr(self, "__dataclass_fields__", {}).keys())

    def values(self):
        return tuple(getattr(self, key) for key in self.keys())

    def items(self):
        return tuple((key, getattr(self, key)) for key in self.keys())

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)


@dataclass(slots=True)
class Z2WorkflowBaseContext(_CompatModel):
    actual_route: str = ""
    route: str = ""
    campaign_context: bool = False
    auto_setup_pending: bool = False
    manual_entry_mode: str = ""
    auto_vehicle_choice: str = ""
    plate_model_selected: bool = False
    input_dir_value: str = ""
    input_dir_ready: bool = False
    vehicle_assist_enabled: bool = False
    has_existing_run: bool = False
    manual_setup: bool = False
    manual_continue: bool = False
    manual_import: bool = False
    manual_run_already_created: bool = False
    current_step: str = ""
    manual_review_active: bool = False
    manual_review_from_auto: bool = False
    auto_completed: bool = False
    campaign_reused_manual_count: int = 0


@dataclass(slots=True)
class Z2LeftPanelCopyContext(_CompatModel):
    actual_route: str = ""
    route: str = ""
    campaign_context: bool = False
    auto_setup_pending: bool = False
    manual_entry_mode: str = ""
    manual_setup: bool = False
    manual_import: bool = False
    vehicle_assist_enabled: bool = False
    auto_vehicle_choice: str = ""
    has_existing_run: bool = False
    manual_run_already_created: bool = False
    plate_model_selected: bool = False
    campaign_iteration_target: str = ""
    campaign_stage: int = 0
    campaign_iteration_num: int = 0
    current_step: str = ""
    current_index: int = 0
    total_steps: int = 0
    has_manual_history: bool = False
    auto_completed: bool = False
    manual_review_active: bool = False
    campaign_reused_manual_count: int = 0
    campaign_char_repair_mode: bool = False
    campaign_manual_skip_count: int = 0


@dataclass(slots=True)
class Z2CampaignRuntimeState(_CompatModel):
    route: str = ""
    campaign_stage: int = 0
    campaign_iteration_target: str = ""
    campaign_iteration_num: int = 1
    campaign_char_repair_mode: bool = False
    campaign_plate_step4_repair_mode: bool = False
    available_primary_action_ids: list[str] | None = None


@dataclass(slots=True)
class Z2FreeModeRuntimeState(_CompatModel):
    free_mode_screen: str = ""
    available_primary_action_ids: list[str] | None = None


@dataclass(slots=True)
class Z2LayoutState(_CompatModel):
    show_route_choice: bool = False
    show_export_followup: bool = False
    show_auto_followup: bool = False
    show_manual_review_followup: bool = False
    show_stage_export_cta: bool = False
    compact_export_followup: bool = False
    show_workflow_steps: bool = False
    compact_single_route_layout: bool = False
    compact_left_column_layout: bool = False
    show_campaign_context_header: bool = False
    show_nav_panel: bool = False
    show_right_panel: bool = False


@dataclass(slots=True)
class Z2CtaState(_CompatModel):
    show_start_controls: bool = False
    show_nav_controls: bool = False
    start_enabled: bool = False
    start_command: Callable[..., Any] | None = None
    start_text: str = "Wybierz tor"
    back_enabled: bool = False
    back_text: str = "Wstecz"
    next_enabled: bool = False
    next_text: str = "Dalej"
    suppress_duplicate_start_cta: bool = False


@dataclass(slots=True)
class Z2CopyPayload(_CompatModel):
    badge_text: str = "Wybierz tor pracy"
    badge_tone: str = "muted"
    route_text: str = "Na starcie widzisz tylko dwa kafle: autoanotacja albo anotacja reczna."
    route_tone: str = "muted"
    action_text: str = ""
    auto_choice_hint: str = ""
    manual_hint: str = ""
    manual_hint_tone: str = "muted"
    manual_template_hint: str = ""
    manual_template_tone: str = "muted"
    manual_vehicle_hint: str = ""
    manual_vehicle_tone: str = "muted"
    followup_title: str = "Co dalej po autoanotacji"
    followup_text: str = "Po zakonczeniu runu anotacji Z2 odblokujesz tutaj przejscie do korekty recznej."
    export_title: str = "Split i eksport datasetu"
    export_text: str = "Split i eksport sa osobnym krokiem na gotowym runie anotacji Z2."
    workflow_conf_title: str = "Ustaw confidence"
    workflow_conf_hint: str = ""
    workflow_vehicle_title: str = "Model pojazdow (YOLO Box)"
    workflow_vehicle_hint: str = ""
    workflow_input_title: str = "Folder obraz\u00f3w"
    workflow_input_hint: str = "Najpierw wybierz folder z obrazami, na ktorych ma pracowac aktualny tor Z2."
    workflow_start_title: str = "Uruchom proces Z2"
    workflow_start_intro: str = ""
    manual_entry_title: str = "Wybierz sposob wejscia do pracy recznej"
    manual_history_title: str = "Historia runow Z2"
    auto_plate_model_title: str = "Wskaz model tablic (YOLO Pose)"
    auto_plate_model_hint_text: str = "Ten model jest wymagany, aby uruchomic autoanotacje tablic."
    auto_plate_model_hint_tone: str = "muted"
    auto_vehicle_choice_title: str = "Dodaj opcjonalne boxowanie pojazdow"
    run_title: str = "Kreator Z2"
    run_intro_text: str = ""
