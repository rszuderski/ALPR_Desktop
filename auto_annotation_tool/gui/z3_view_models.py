#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lekkie modele widoku dla E3/Z3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .z2_view_models import Step2CtaViewModel


@dataclass(frozen=True)
class Step3ViewModel:
    stage_key: str = "step3"
    iteration_target: str = ""
    current_step: int = 0
    step3_status: str = "pending"
    state: str = "locked"
    title: str = "E3"
    summary: str = ""
    details: str = ""
    body_mode: str = ""
    body_visible: bool = False
    primary_cta: Step2CtaViewModel | None = None
    secondary_cta: Step2CtaViewModel | None = None


@dataclass(frozen=True)
class Step3EntryFlowViewModel:
    mode: str = ""
    message: str = ""
    target_substep: int = 0
    should_start_extraction: bool = False
    should_show_splash: bool = False
    should_hide_splash: bool = False
    splash_title: str = ""
    splash_body: str = ""
    splash_tone: str = "info"
    splash_show_progress: bool = False
    splash_show_return: bool = False


@dataclass(frozen=True)
class Step3CampaignNavigationViewModel:
    in_campaign: bool = False
    splash_visible: bool = False
    show_detect_back_to_extract: bool = True
    show_detect_return_to_graph: bool = False
    show_detect_to_dataset: bool = True
    show_dataset_back_to_detect: bool = True


@dataclass(frozen=True)
class Step3FinishActionViewModel:
    visible: bool = False
    label: str = ""
    command_id: str = ""
    enabled: bool = False
    emphasize: bool = False
    hint: str = ""
    hint_tone: str = "muted"
    back_to_wizard_enabled: bool = False


@dataclass(frozen=True)
class Step3Pz3PathSelectionViewModel:
    selected_path: str = ""
    show_dataset_section: bool = False
    show_cvat_section: bool = False
    show_status_section: bool = False
    dataset_card_selected: bool = False
    cvat_card_selected: bool = False
    dataset_badge_text: str = "DATASET"
    dataset_title_text: str = "Dataset znaków"
    dataset_desc_text: str = "Główna ścieżka PZ3: materiał z PZ2, zakres tablic perfect i utworzenie źródłowego datasetu znaków."
    cvat_badge_text: str = "OPCJA"
    cvat_title_text: str = "Korekta w CVAT"
    cvat_desc_text: str = "Obieg korekty poza aplikacją: wyślij cropy tablic do CVAT, popraw boxy znaków i wczytaj XML z powrotem w PZ3."


@dataclass(frozen=True)
class Step3Pz3DatasetModeViewModel:
    in_campaign: bool = False
    mode: str = "perfect"
    dataset_source_title: str = ""
    dataset_source_intro: str = ""
    source_preview_text: str = ""
    source_preview_tone: str = "muted"
    source_pool_text: str = ""
    source_pool_tone: str = "muted"
    source_next_text: str = ""
    source_next_tone: str = "muted"
    show_dataset_source_cards: bool = True
    perfect_selected: bool = True
    existing_selected: bool = False
    perfect_badge_text: str = "PZ2"
    perfect_title_text: str = "Materiał z PZ2"
    perfect_desc_text: str = "Źródłem są wyodrębnione tablice z PZ2 oznaczone jako perfect."
    existing_badge_text: str = "Z4"
    existing_title_text: str = "Warianty w Z4"
    existing_desc_text: str = "Gotowe datasety i ich warianty wybierzesz w Z4."
    show_existing_dataset_panel: bool = False
    cvat_option2_title: str = ""
    cvat_option2_tone: str = "muted"
    cvat_option2_desc: str = ""
    show_gold_filters: bool = True
    split_title: str = ""
    split_label: str = ""
    primary_export_label: str = ""
    primary_export_command_id: str = ""
    primary_export_enabled: bool = True
    primary_export_columnspan: int = 1
    show_classifier_export: bool = True
    classifier_export_enabled: bool = True
    action_hint: str = ""
    action_hint_tone: str = "muted"
    existing_dataset_status: str = ""
    existing_dataset_status_tone: str = "muted"


@dataclass(frozen=True)
class Step3Pz3StatusRowViewModel:
    label: str = ""
    text: str = ""
    tone: str = "muted"


@dataclass(frozen=True)
class Step3Pz3StatusPanelViewModel:
    title: str = "Podsumowanie"
    show_section: bool = False
    run_row: Step3Pz3StatusRowViewModel = field(default_factory=Step3Pz3StatusRowViewModel)
    dataset_row: Step3Pz3StatusRowViewModel = field(default_factory=Step3Pz3StatusRowViewModel)
    export_row: Step3Pz3StatusRowViewModel = field(default_factory=Step3Pz3StatusRowViewModel)
    readiness_row: Step3Pz3StatusRowViewModel = field(default_factory=Step3Pz3StatusRowViewModel)
    import_row: Step3Pz3StatusRowViewModel = field(default_factory=Step3Pz3StatusRowViewModel)
    finish_action: Step3FinishActionViewModel = field(default_factory=Step3FinishActionViewModel)


@dataclass(frozen=True)
class Step3ExtractStepCardViewModel:
    key: str = ""
    state: str = "pending"
    title: str = ""
    description: str = ""


@dataclass(frozen=True)
class Step3ExtractWorkflowViewModel:
    route: str = ""
    current_step: str = "entry"
    linear_mode: bool = False
    show_entry: bool = True
    show_source: bool = False
    show_start: bool = False
    source_title: str = "Źródła wejscia"
    start_title: str = "Uruchom wyodrebnianie"
    source_intro: str = ""
    source_hint: str = ""
    run_hint: str = ""
    run_hint_tone: str = "muted"
    start_hint: str = ""
    start_hint_tone: str = "muted"
    show_step_nav: bool = False
    prev_enabled: bool = False
    next_enabled: bool = False
    next_visible: bool = True
    show_tab_nav: bool = True
    show_back_nav: bool = False
    show_detect_nav: bool = True
    clear_detect_emphasis: bool = True
    step_cards: list[Step3ExtractStepCardViewModel] = field(default_factory=list)
