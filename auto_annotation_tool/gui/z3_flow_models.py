from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class _CompatModel:
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)


@dataclass(slots=True)
class Z3WorkflowBaseContext(_CompatModel):
    in_campaign: bool = False
    linear_mode: bool = False
    current_substep: int = 1
    active_subtab: str = ""
    iteration_target: str = ""
    step3_status: str = ""
    extract_route: str = ""
    extract_current_step: str = "entry"
    preview_dir_value: str = ""
    preview_ready: bool = False
    preview_plate_count: int = 0
    has_preview_run: bool = False
    pz3_selected_path: str = ""
    pz3_dataset_mode: str = "perfect"
    existing_dataset_path: str = ""
    has_export_outputs: bool = False
    finish_ready: bool = False


@dataclass(slots=True)
class Z3CampaignRuntimeState(_CompatModel):
    target_substep: int = 1
    entry_mode: str = ""
    workflow_step: str = "entry"
    annotation_run_dir: str = ""
    xml_path: str = ""
    images_dir: str = ""
    using_preferred_source: bool = False
    should_show_splash: bool = False
    should_hide_splash: bool = False
    splash_title: str = ""
    splash_body: str = ""
    splash_tone: str = "info"
    splash_show_progress: bool = False
    splash_show_return: bool = False
    available_primary_action_ids: list[str] | None = None


@dataclass(slots=True)
class Z3FreeModeRuntimeState(_CompatModel):
    target_substep: int = 1
    selected_path: str = ""
    dataset_source_mode: str = "perfect"
    available_primary_action_ids: list[str] | None = None


@dataclass(slots=True)
class Z3LayoutState(_CompatModel):
    show_extract_tab: bool = True
    show_detect_tab: bool = True
    show_dataset_tab: bool = True
    show_detect_back_to_extract: bool = True
    show_detect_to_dataset: bool = True
    show_dataset_back_to_detect: bool = True
    show_status_panel: bool = False
    show_finish_card: bool = False
    pin_preview_status: bool = False
    show_preview_source_panel: bool = True
    show_dataset_source_cards: bool = True


@dataclass(slots=True)
class Z3CtaState(_CompatModel):
    extract_start_enabled: bool = False
    extract_start_label: str = "Uruchom wyodrebnianie"
    extract_start_command: Callable[..., Any] | None = None
    detect_back_enabled: bool = True
    detect_back_label: str = "Wstecz"
    detect_back_command: Callable[..., Any] | None = None
    detect_next_enabled: bool = False
    detect_next_label: str = "Dalej"
    detect_next_command: Callable[..., Any] | None = None
    dataset_back_enabled: bool = True
    dataset_back_label: str = "Wstecz do PZ2"
    dataset_back_command: Callable[..., Any] | None = None
    finish_enabled: bool = False
    finish_visible: bool = False
    finish_label: str = "Wróć do grafu"
    finish_command_id: str = ""


@dataclass(slots=True)
class Z3PreviewState(_CompatModel):
    ready: bool = False
    preview_dir: str = ""
    metadata_path: str = ""
    images_dir: str = ""
    plate_count: int = 0
    active_plate_id: str = ""
    preview_box_mode: str = "AUTO"
    preview_sort_mode: str = "DEFAULT"
    source_line: str = ""
    status_text: str = ""
    import_focus_active: bool = False


@dataclass(slots=True)
class Z3DatasetGoalState(_CompatModel):
    selected_plate_count: int = 0
    selected_char_count: int = 0
    split_enabled: bool = False
    ready_dataset_path: str = ""
    ready_train: int = 0
    ready_val: int = 0
    ready_test: int = 0
    dataset_valid: bool = False
    review_pack_ready: bool = False


@dataclass(slots=True)
class Z3CopyPayload(_CompatModel):
    extract_entry_title: str = "Wejscie do Z3"
    extract_entry_intro: str = ""
    extract_source_title: str = "Zrodla wejscia"
    extract_source_intro: str = ""
    extract_source_hint: str = ""
    extract_start_title: str = "Uruchom wyodrebnianie"
    extract_start_intro: str = ""
    extract_run_hint: str = ""
    dataset_title: str = "Budowa datasetu"
    dataset_intro: str = ""
    dataset_status_title: str = "Podsumowanie"
    dataset_status_intro: str = ""
    finish_hint: str = ""
    finish_hint_tone: str = "muted"
