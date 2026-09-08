from __future__ import annotations

import json
import datetime
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..campaign_manager import CAMPAIGN
from ..config import CONFIG, logger
from ..training import TrainingHistory, YOLOPoseTrainer, TrainingStatus
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)
from .dataset_display import build_dataset_display_ref

if TYPE_CHECKING:
    from .tab_training import TrainingTab


def _char_dataset_gate_display_id() -> str:
    return "T05"


def _training_finish_gate_display_id() -> str:
    return "T06"


def _dataset_summary_id(path_like, *, target_hint: str, counts: dict | None = None) -> str:
    raw = str(path_like or "").strip()
    if not raw:
        return ""
    try:
        return build_dataset_display_ref(raw, target_hint=target_hint, counts=counts).id
    except Exception:
        try:
            return Path(raw).name
        except Exception:
            return raw


def _safe_step4_int(value, default: int = 0) -> int:
    try:
        number = int(value or default or 0)
    except Exception:
        number = int(default or 0)
    return number


def _current_step4_iteration() -> int:
    try:
        return int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        return 1


def _step4_record_declared_iteration(record: dict | None) -> int:
    data = dict(record or {})
    for field in ("dataset_iteration", "created_iteration", "iteration"):
        value = _safe_step4_int(data.get(field))
        if value > 0:
            return value
    return 0


def _step4_dataset_record_counts(record: dict | None) -> dict:
    data = dict(record or {})
    train = max(0, _safe_step4_int(data.get("train_images")))
    val = max(0, _safe_step4_int(data.get("val_images")))
    test = max(0, _safe_step4_int(data.get("test_images")))
    total = max(0, _safe_step4_int(data.get("total_images")))
    if total <= 0:
        total = train + val + test
    return {"train": train, "val": val, "test": test, "total": total}


def _current_iteration_step4_dataset_record(target: str | None) -> dict:
    normalized_target = str(target or "").strip().lower()
    if normalized_target not in {"char", "plate"}:
        normalized_target = "char"
    iteration_num = _current_step4_iteration()

    record = {}
    try:
        iteration_state = dict(CAMPAIGN.get_iteration_state(iteration_num=iteration_num) or {})
        record = dict(iteration_state.get("step4_dataset") or {})
    except Exception:
        record = {}
    if not record:
        try:
            bundle = dict(CAMPAIGN.get_iteration_artifact_bundle(iteration_num=iteration_num) or {})
            bundle_record = dict(bundle.get("step4_dataset") or {})
            if _step4_record_declared_iteration(bundle_record) == iteration_num:
                record = bundle_record
        except Exception:
            record = {}
    if not record:
        return {}

    record.setdefault("iteration", iteration_num)
    record_target = str(record.get("target") or "").strip().lower()
    if record_target and record_target != normalized_target:
        return {}
    counts = _step4_dataset_record_counts(record)
    if int(counts.get("total", 0) or 0) <= 0:
        return {}
    return dict(record)


def _resolve_step4_dataset_record_root(record: dict | None) -> Path | None:
    data = dict(record or {})
    candidates: list[Path] = []
    dataset_path = str(data.get("dataset_path") or "").strip()
    yaml_path = str(data.get("yaml_path") or "").strip()
    if dataset_path:
        try:
            candidate = Path(dataset_path)
            candidates.append(candidate.parent if candidate.is_file() and candidate.name.lower() == "data.yaml" else candidate)
        except Exception:
            pass
    if yaml_path:
        try:
            candidates.append(Path(yaml_path).parent)
        except Exception:
            pass
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_dir():
                return candidate
        except Exception:
            continue
    return candidates[0] if candidates else None


def _current_iteration_pz3_source_contract() -> dict:
    iteration_num = _current_step4_iteration()
    try:
        iteration_state = dict(CAMPAIGN.get_iteration_state(iteration_num=iteration_num) or {})
    except Exception:
        iteration_state = {}
    contracts = iteration_state.get("t06_contracts")
    contracts = dict(contracts) if isinstance(contracts, dict) else {}
    contract = contracts.get("pz3_char_dataset")
    contract = dict(contract) if isinstance(contract, dict) else {}
    if not bool(contract.get("fulfilled")):
        return {}
    product = str(contract.get("product") or "").strip().lower()
    if product and product != "char_yolo_dataset":
        return {}
    source_iteration = (
        _safe_step4_int(contract.get("source_iteration"))
        or _safe_step4_int(contract.get("created_iteration"))
        or _safe_step4_int(contract.get("iteration"))
    )
    if source_iteration > 0 and source_iteration != iteration_num:
        return {}
    dataset_path = str(contract.get("dataset_path") or contract.get("gold_dataset_path") or "").strip()
    if not dataset_path:
        return {}
    try:
        root = Path(dataset_path)
        if root.is_file() and root.name.lower() == "data.yaml":
            root = root.parent
        if not root.exists() or not root.is_dir():
            return {}
    except Exception:
        return {}
    contract["dataset_path"] = str(root)
    return contract


def _apply_campaign_project_base_model_selection(host: "TrainingTab", target: str | None) -> bool:
    normalized_target = str(target or "").strip().lower()
    if normalized_target not in {"plate", "char"}:
        return False

    try:
        model_info = dict(
            CAMPAIGN.get_effective_project_model(
                normalized_target,
                before_iteration=int(CAMPAIGN.get_current_iteration_num() or 1),
            )
            or {}
        )
    except Exception:
        model_info = {}

    model_path = str(model_info.get("path") or "").strip()
    if not model_path:
        return False

    try:
        path = Path(model_path)
    except Exception:
        return False
    if not path.exists() or not path.is_file():
        return False

    host.base_model_var.set(host._get_custom_base_model_label())
    host.base_custom_var.set(str(path))
    return True


def build_step4_dataset_workflow_view_model(
    host: "TrainingTab",
) -> Step4DatasetWorkflowViewModel:
    campaign_active = bool(CAMPAIGN.get_active_project_name())
    locked_target = host._get_locked_campaign_training_target()
    route_locked = campaign_active and locked_target in ("char", "plate")
    mode = getattr(host, "_step4_dataset_mode", "char")
    route_selected = bool(getattr(host, "_step4_route_selected", False))

    creator_summary = ""
    split_intro = (
        "PZ1 tworzy wariant treningowy train / val / test. "
        "PZ2 wybiera ten wariant z listy i uruchamia na nim trening."
    )
    split_summary = ""
    split_action_label = "Utwórz wariant treningowy"
    show_split_toggle = False
    split_toggle_label = "Popraw split"
    show_split_details = True
    char_ready_dataset = False
    char_ready_train = 0
    char_ready_val = 0
    char_ready_test = 0
    char_ready_path = ""

    if campaign_active and mode == "plate":
        plate_readiness = {}
        try:
            plate_readiness = dict(host.get_campaign_step4_readiness(iteration_target="plate") or {})
        except Exception:
            plate_readiness = {}
        approved_images = int(plate_readiness.get("project_approved_images", 0) or 0)
        approved_plates = int(plate_readiness.get("project_approved_plates", 0) or 0)
        ready_dataset_path = str(plate_readiness.get("ready_dataset") or "").strip()
        ready_train = int(plate_readiness.get("train_images", 0) or 0)
        ready_val = int(plate_readiness.get("val_images", 0) or 0)
        ready_test = int(plate_readiness.get("test_images", 0) or 0)
        xml_value = host._shorten_training_text(host._format_workspace_relative_path(host.cvat_xml_var.get()), 96)
        images_value = host._shorten_training_text(host._format_workspace_relative_path(host.cvat_images_var.get()), 96)
        creator_summary = (
            "Źródło datasetu tablic jest już podpięte z projektu.\n"
            f"Zatwierdzone: {approved_images} obraz(y), {approved_plates} tablic(e).\n"
            f"Plik anotacji: {xml_value}\n"
            f"Obrazy wejściowe: {images_value}"
        )
        if ready_dataset_path and (ready_train > 0 or ready_val > 0 or ready_test > 0):
            creator_summary += (
                f"\nAktualny dataset treningowy: train={ready_train}, val={ready_val}, test={ready_test}"
            )

    if campaign_active and mode == "char":
        char_dataset_gate_id = _char_dataset_gate_display_id()
        readiness = {}
        try:
            readiness = dict(host.get_campaign_step4_readiness(iteration_target="char") or {})
        except Exception:
            readiness = {}

        char_ready_path = str(readiness.get("ready_dataset") or "").strip()
        char_ready_train = int(readiness.get("train_images", 0) or 0)
        char_ready_val = int(readiness.get("val_images", 0) or 0)
        char_ready_test = int(readiness.get("test_images", 0) or 0)
        char_ready_dataset = bool(char_ready_path and char_ready_train > 0 and char_ready_val > 0)

        if char_ready_dataset:
            split_counts = {
                "train": char_ready_train,
                "val": char_ready_val,
                "test": char_ready_test,
                "total": char_ready_train + char_ready_val + char_ready_test,
            }
            split_ready_value = _dataset_summary_id(
                char_ready_path,
                target_hint="char",
                counts=split_counts,
            )
            split_intro = (
                "Wariant treningowy znaków jest już gotowy. PZ1 nie eksportuje źródłowego datasetu znaków; "
                "pokazuje split, który PZ2 może wykorzystać do treningu."
            )
            split_summary = (
                f"Źródłowy dataset znaków pochodzi z {char_dataset_gate_id}/PZ3.\n"
                f"Gotowy wariant: {split_ready_value}\n"
                f"Split: train={char_ready_train}, val={char_ready_val}, test={char_ready_test}\n"
                "Nie musisz przygotowywać wariantu ponownie. Traktuj tę sekcję jako narzędzie awaryjne, "
                "jeśli chcesz świadomie przebudować strukturę train / val / test."
            )
            show_split_toggle = True
            show_split_details = bool(getattr(host, "_step4_char_split_details_visible", False))
            split_toggle_label = "Ukryj opcje splitu" if show_split_details else "Popraw split"
            split_action_label = "Przebuduj wariant treningowy znaków"
        else:
            split_src_raw = str(host.split_src_var.get() or "").strip()
            split_src_value = _dataset_summary_id(split_src_raw, target_hint="char") if split_src_raw else "brak wskazanego datasetu"
            split_intro = (
                f"PZ1 przygotowuje wariant treningowy znaków ze źródła utworzonego w {char_dataset_gate_id}/PZ3. "
                "To jest split pod trening, a nie ponowny eksport datasetu znaków."
            )
            split_summary = (
                f"Źródło {char_dataset_gate_id}/PZ3: {split_src_value}\n"
                "Po utworzeniu wariantu PZ2 będzie mogło uruchomić trening modelu znaków."
            )
            split_action_label = "Utwórz wariant treningowy znaków"

    if campaign_active and not route_selected:
        return Step4DatasetWorkflowViewModel(
            mode=mode,
            in_campaign=True,
            route_selected=False,
            route_locked=route_locked,
            show_route_panel=not route_locked,
            show_waiting_panel=True,
            title="Wybierz tor po lewej stronie",
            description=(
                "W trybie projektu najpierw wybierz tor tablic albo tor znaków. "
                "Dopiero wtedy odblokuje się panel budowy datasetu."
            ),
            next_label="Wybierz tor",
        )

    if mode == "plate":
        if campaign_active:
            description = (
                "Tor tablic prowadzi do treningu modelu YOLO Pose. Jeśli masz gotowy dataset z eksportu Z2, "
                "możesz przejść dalej do PZ2. Kreator poniżej buduje nowy wariant datasetu "
                "z pliku anotacji XML i zgodnego folderu obrazów."
            )
        else:
            description = (
                "PZ1 przygotowuje wariant treningowy tablic. PZ2 wybierze ten wariant do treningu modelu."
            )
        if route_locked:
            campaign_plate_stats = {}
            try:
                campaign_plate_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
            except Exception:
                campaign_plate_stats = {}
            approved_images = int(campaign_plate_stats.get("images", 0) or 0)
            approved_plates = int(campaign_plate_stats.get("plates", 0) or 0)
            finish_gate_id = _training_finish_gate_display_id()
            description = (
                f"Bramka {finish_gate_id} korzysta z zatwierdzonych tablic projektu. "
                f"Materiał: {approved_images} obraz(y), {approved_plates} tablic(e). "
                "Utwórz wariant treningowy train/val/test, aby przejść do treningu modelu tablic."
            )
    else:
        if campaign_active:
            description = (
                "Wybierz ten tor, jeśli chcesz przygotować i podzielić dataset znaków "
                "i trenować model znaków na tablicach."
            )
        else:
            description = (
                "PZ1 przygotowuje wariant treningowy znaków. PZ2 wybierze ten wariant do treningu modelu."
            )
        if route_locked:
            if char_ready_dataset:
                description = (
                    "Ten etap ma już gotowy wariant treningowy znaków z poprawnym splitem. "
                    f"Aktualny stan: train={char_ready_train}, val={char_ready_val}, test={char_ready_test}. "
                    "Możesz przejść dalej do treningu, a ponowne przygotowanie wariantu traktować tylko jako opcjonalne narzędzie."
                )
            else:
                description = (
                    f"Ten etap korzysta ze źródłowego datasetu znaków z {_char_dataset_gate_display_id()}/PZ3. "
                    "W PZ1 przygotowujesz tylko wariant train/val/test; PZ2 uruchamia trening."
                )

    if route_locked:
        description = str(description or "").strip() + " Tor jest stały dla tej iteracji."

    title = (
        ("Tor tablic (YOLO Pose)" if mode == "plate" else "Tor znaków (YOLO Detect)")
        if campaign_active
        else ("Dataset tablic (YOLO Pose)" if mode == "plate" else "Dataset znaków (YOLO Detect)")
    )

    next_label = "Dalej do treningu"
    if campaign_active:
        next_target_label = "tablic" if mode == "plate" else "znak\u00f3w"
        next_label = f"Przejd\u017a do treningu modelu {next_target_label}"

    return Step4DatasetWorkflowViewModel(
        mode=mode,
        in_campaign=campaign_active,
        route_selected=route_selected,
        route_locked=route_locked,
        show_route_panel=not route_locked,
        show_waiting_panel=False,
        title=title,
        description=description,
        show_creator_section=(mode == "plate"),
        show_split_section=(mode == "char"),
        next_label=next_label,
        creator_summary=creator_summary,
        split_intro=split_intro,
        split_summary=split_summary,
        split_action_label=split_action_label,
        show_split_toggle=show_split_toggle,
        split_toggle_label=split_toggle_label,
        show_split_details=show_split_details,
    )


def build_step4_training_inputs_view_model(
    host: "TrainingTab",
) -> Step4TrainingInputsViewModel:
    campaign_active = bool(CAMPAIGN.get_active_project_name())
    current_base_key = str(getattr(host, "base_model_var", None).get() or "").strip()
    current_custom_value = str(getattr(host, "base_custom_var", None).get() or "").strip()
    show_project_custom_row = bool(host._is_custom_base_model_key(current_base_key))
    return Step4TrainingInputsViewModel(
        in_campaign=campaign_active,
        show_dataset_section=True,
        show_scope_hint=(not campaign_active),
        show_pose_warning=(not campaign_active),
        base_combo_state="readonly",
        show_custom_model=((not campaign_active) or show_project_custom_row),
        custom_entry_state=("readonly" if show_project_custom_row else "disabled"),
        show_custom_pick_button=show_project_custom_row,
    )


def build_step4_campaign_navigation_view_model(
    host: "TrainingTab",
) -> Step4CampaignNavigationViewModel:
    campaign_active = bool(CAMPAIGN.get_active_project_name())
    train_unlocked = bool(getattr(host, "_step4_train_unlocked", False))
    route_selected = bool(getattr(host, "_step4_route_selected", False))
    dataset_tab_visible = bool(getattr(host, "_step4_dataset_tab_visible", False))
    finish_ready = bool(getattr(host, "_step4_campaign_finish_ready", False))
    dataset_vm = build_step4_dataset_workflow_view_model(host)
    finish_gate_id = _training_finish_gate_display_id()
    train_back_label = (
        f"Wróć do bramki {finish_gate_id} i zakończ decyzję"
        if campaign_active and finish_ready
        else (f"Wróć do bramki {finish_gate_id} w grafie" if campaign_active else "Wstecz do PZ1")
    )

    return Step4CampaignNavigationViewModel(
        in_campaign=campaign_active,
        dataset_tab_enabled=(not campaign_active) or dataset_tab_visible,
        train_tab_enabled=(not campaign_active) or train_unlocked,
        next_enabled=(not campaign_active) or (route_selected and train_unlocked),
        next_label=str(dataset_vm.next_label or "Dalej do treningu"),
        show_dataset_back=campaign_active,
        dataset_back_label=(f"Wróć do bramki {finish_gate_id} w grafie" if campaign_active else ""),
        show_train_nav=True,
        show_train_back=True,
        train_back_label=train_back_label,
        show_complete_project=False,
        force_dataset_tab_selection=campaign_active and dataset_tab_visible and not train_unlocked,
    )


def set_campaign_training_target(host: "TrainingTab", target: str):
    target = (target or "char").strip().lower()
    if target not in ("char", "plate"):
        target = "char"

    host._campaign_training_target = target
    host._refresh_training_metric_reference()

    try:
        label = "znaków" if target == "char" else "tablic"
        host._append_train_log(f"[TARGET] Ustawiono kampanijny target treningu: model {label}.")
    except Exception:
        pass


def get_campaign_training_target(host: "TrainingTab") -> str:
    target = getattr(host, "_campaign_training_target", "char")
    return target if target in ("char", "plate") else "char"


def set_campaign_context(host: "TrainingTab", runs_dir=None, datasets_dir=None):
    """
    Przełącza TrainingTab na katalogi aktywnego projektu
    i odtwarza stan z4 dla bieżącego projektu.
    """
    campaign_target = CAMPAIGN.get_iteration_target()
    if campaign_target not in ("char", "plate"):
        campaign_target = "char"
        require_route_selection = True
    else:
        require_route_selection = False

    if datasets_dir is not None:
        host._campaign_datasets_dir = str(Path(datasets_dir))

    project_datasets_dir = Path(host._campaign_datasets_dir) if host._campaign_datasets_dir else None

    if runs_dir is None:
        host._reset_step4_transient_ui(
            datasets_dir=project_datasets_dir,
            target=campaign_target,
            require_route_selection=require_route_selection
        )
        try:
            host._restore_step4_campaign_project_state()
        except Exception:
            pass
        return

    new_runs_dir = Path(runs_dir)

    if getattr(host.trainer, "is_training", False):
        logger.warning("Nie można zmienić kontekstu projektu podczas aktywnego treningu.")
        return

    if host._campaign_runs_dir == str(new_runs_dir):
        host._reset_step4_transient_ui(
            datasets_dir=project_datasets_dir,
            target=campaign_target,
            require_route_selection=require_route_selection
        )
        try:
            host._restore_step4_campaign_project_state()
        except Exception:
            pass
        return

    host._campaign_runs_dir = str(new_runs_dir)
    new_runs_dir.mkdir(parents=True, exist_ok=True)

    host.history = TrainingHistory(history_dir=new_runs_dir)
    host.trainer = YOLOPoseTrainer(history=host.history)
    host._bind_trainer_callbacks()

    host._reset_step4_transient_ui(
        datasets_dir=project_datasets_dir,
        target=campaign_target,
        clear_builder_inputs=True,
        require_route_selection=require_route_selection
    )

    try:
        host._restore_step4_campaign_project_state()
    except Exception:
        pass

    logger.info(f"TrainingTab przełączony na projektowy katalog runów: {new_runs_dir}")


def clear_campaign_context(host: "TrainingTab"):
    """
    Czyści projektowy kontekst treningu i wraca do globalnych katalogów Workspace.
    Dodatkowo czyści kampanijne logi / wyniki widoczne w UI.
    """
    free_mode_target = CONFIG.normalize_task_target(
        getattr(host, "_step4_dataset_mode", getattr(host, "_campaign_training_target", "char"))
    )

    host._campaign_runs_dir = None
    host._campaign_datasets_dir = None

    host._rebind_free_mode_training_storage(target=free_mode_target, reload_history=False)

    host._reset_step4_transient_ui(
        target=free_mode_target,
        clear_builder_inputs=True,
        require_route_selection=False
    )

    try:
        host._update_step4_notebook_mode()
    except Exception:
        pass

    try:
        host._refresh_free_training_route_ui()
    except Exception:
        pass

    try:
        host._refresh_step4_dataset_mode_ui()
    except Exception:
        pass

    try:
        host._refresh_step4_campaign_navigation_ui()
    except Exception:
        pass

    try:
        host._update_training_dataset_hint()
    except Exception:
        pass


def open_campaign_step4_entry(
    host: "TrainingTab",
    *,
    iteration_target: str | None = None,
    preferred_subtab: str | None = None,
) -> dict:
    if not CAMPAIGN.get_active_project_name() or int(CAMPAIGN.get_current_step() or 0) < 4:
        return {"ok": False, "reason": "campaign_inactive"}

    target = str(iteration_target or CAMPAIGN.get_iteration_target() or "").strip().lower()
    if target not in {"plate", "char"}:
        target = "char"
    preferred_subtab = str(preferred_subtab or "").strip().lower()
    if preferred_subtab not in {"dataset", "train"}:
        preferred_subtab = ""

    perf_started = time.perf_counter()
    perf_last = perf_started
    perf_phases: list[str] = []

    def _perf_mark(name: str) -> None:
        nonlocal perf_last
        now = time.perf_counter()
        perf_phases.append(f"{name}={int((now - perf_last) * 1000)}ms")
        perf_last = now

    def _perf_log(reason: str = "ok") -> None:
        total_ms = int((time.perf_counter() - perf_started) * 1000)
        if total_ms < 300:
            return
        try:
            logger.info(
                f"[Z4/PZ1 PERF] open_campaign_step4_entry total={total_ms}ms "
                f"target={target} preferred={preferred_subtab or '-'} reason={reason} "
                f"train_unlocked={int(bool(getattr(host, '_step4_train_unlocked', False)))} "
                f"phases=[{', '.join(perf_phases) or 'no_slow_phase'}]"
            )
        except Exception:
            pass

    datasets_dir = CAMPAIGN.get_dir("datasets")
    runs_dir = CAMPAIGN.get_dir("runs")
    if datasets_dir is None:
        logger.error("Brak katalogu datasets dla aktywnego projektu.")
        return {"ok": False, "reason": "missing_datasets_dir"}

    try:
        host.set_campaign_training_target(target)
    except Exception:
        pass

    runs_dir_text = str(Path(runs_dir)) if runs_dir is not None else None
    datasets_dir_text = str(Path(datasets_dir))
    context_matches = bool(
        runs_dir_text
        and str(getattr(host, "_campaign_runs_dir", "") or "") == runs_dir_text
        and str(getattr(host, "_campaign_datasets_dir", "") or "") == datasets_dir_text
    )
    if context_matches:
        host._campaign_datasets_dir = datasets_dir_text
    else:
        host.set_campaign_context(
            runs_dir=runs_dir_text,
            datasets_dir=datasets_dir_text,
        )
    host._step4_dataset_mode = target
    host._step4_route_selected = True
    host._step4_dataset_tab_visible = True
    _perf_mark("context_reuse" if context_matches else "context_restore")

    datasets_dir = Path(datasets_dir)
    readiness = host.get_campaign_step4_readiness(iteration_target=target)
    _perf_mark("readiness")
    readiness_reason = str(readiness.get("reason") or "").strip().lower()
    ready_dataset_text = str(readiness.get("ready_dataset") or "").strip()
    source_dataset_text = str(readiness.get("source_dataset") or "").strip()
    latest_source = None
    if target == "char" and source_dataset_text:
        try:
            source_dataset_path = Path(source_dataset_text)
            if source_dataset_path.exists() and source_dataset_path.is_dir():
                latest_source = source_dataset_path
        except Exception:
            latest_source = None

    if latest_source is None:
        source_candidates = []
        try:
            for path in host._find_dataset_source_candidates(datasets_dir):
                inferred = host._infer_dataset_target(str(path))
                if target == "char":
                    if inferred != "char":
                        continue
                elif inferred not in {"plate", None}:
                    continue
                source_candidates.append(path)
        except Exception:
            source_candidates = []
        latest_source = max(source_candidates, key=lambda p: p.stat().st_mtime) if source_candidates else None
        _perf_mark("source_scan")
    else:
        _perf_mark("source_fast")

    if target == "char" and readiness_reason == "source_dataset_ready_for_split" and preferred_subtab != "train":
        preferred_subtab = "dataset"
        if source_dataset_text:
            try:
                source_dataset_path = Path(source_dataset_text)
                if source_dataset_path.exists() and source_dataset_path.is_dir():
                    latest_source = source_dataset_path
            except Exception:
                pass
        ready_dataset_text = ""

    ready_train = int(readiness.get("train_images", 0) or 0)
    ready_val = int(readiness.get("val_images", 0) or 0)
    has_training_ready_dataset = bool(
        readiness.get("ok", False)
        and ready_dataset_text
        and readiness_reason != "source_dataset_ready_for_split"
        and ready_train > 0
        and ready_val > 0
    )
    allow_dataset_source_entry = bool(
        target == "char"
        and preferred_subtab == "dataset"
        and latest_source is not None
        and readiness_reason in {"invalid_char_dataset", "missing_char_dataset", "source_dataset_ready_for_split"}
    )

    if (
        not readiness.get("ok", False)
        and readiness_reason != "stale_plate_dataset"
        and not allow_dataset_source_entry
    ):
        _perf_log(readiness_reason or "not_ready")
        return readiness

    host._step4_train_unlocked = bool(has_training_ready_dataset)
    if has_training_ready_dataset:
        try:
            host.dataset_var.set(ready_dataset_text)
        except Exception:
            pass
    elif target == "char" and preferred_subtab == "dataset":
        try:
            host.dataset_var.set("")
        except Exception:
            pass

    if target == "char" and latest_source is not None:
        host.split_src_var.set(str(latest_source))
        host.split_out_var.set(str(latest_source.parent / f"{latest_source.name}_Split_[DATA_I_CZAS]"))
    elif target == "char":
        host.split_src_var.set("")
        host.split_out_var.set(str(datasets_dir / "[BRAK_DATASETU_ZRODLOWEGO]"))

    host._step4_suppress_base_model_state_save = True
    try:
        if not _apply_campaign_project_base_model_selection(host, target):
            restored_saved_selection = host._apply_saved_step4_training_model_selection(target)
            if not restored_saved_selection:
                if target == "plate":
                    host.base_model_var.set(host._get_preferred_plate_pose_base_model())
                    host.base_custom_var.set("")
                else:
                    host.base_model_var.set(host._get_default_base_model_for_mode("char"))
                    host.base_custom_var.set("")
    finally:
        host._step4_suppress_base_model_state_save = False
    _perf_mark("model_selection")

    try:
        if preferred_subtab == "train":
            host._ensure_step4_train_tab_built()
            host.main_nb.select(host.tab_train)
            host._select_step4_analysis_tab(host.hist_tab)
        elif preferred_subtab == "dataset":
            host.main_nb.select(host.tab_dataset)
        elif bool(getattr(host, "_step4_train_unlocked", False)):
            host._ensure_step4_train_tab_built()
            host.main_nb.select(host.tab_train)
            host._select_step4_analysis_tab(host.hist_tab)
        else:
            host.main_nb.select(host.tab_dataset)
    except Exception:
        pass
    _perf_mark("tab_select")

    try:
        selected_tab = str(host.main_nb.select())
    except Exception:
        selected_tab = ""
    if bool(getattr(host, "_step4_train_tab_built", False)) and selected_tab == str(getattr(host, "tab_train", "")):
        try:
            host._refresh_training_start_state()
        except Exception:
            pass
        try:
            host._schedule_step4_deferred_model_refresh()
        except Exception:
            pass
        _perf_mark("train_ui")

    try:
        host._refresh_step4_campaign_navigation_ui()
    except Exception:
        pass
    _perf_mark("navigation")
    _perf_log("ok")

    dataset_hint = str(host.dataset_var.get() or "").strip()
    return {
        "ok": True,
        "iteration_target": target,
        "latest_source": str(latest_source or ""),
        "dataset_hint": dataset_hint,
        "train_unlocked": bool(getattr(host, "_step4_train_unlocked", False)),
    }


def restore_step4_campaign_project_state(host: "TrainingTab"):
    """
    Odtwarza stan z4 dla aktywnego projektu.
    """
    perf_started = time.perf_counter()
    perf_last = perf_started
    perf_phases: list[str] = []

    def _perf_mark(name: str) -> None:
        nonlocal perf_last
        now = time.perf_counter()
        perf_phases.append(f"{name}={int((now - perf_last) * 1000)}ms")
        perf_last = now

    def _perf_log(reason: str = "ok") -> None:
        total_ms = int((time.perf_counter() - perf_started) * 1000)
        if total_ms < 300:
            return
        try:
            logger.info(
                f"[Z4/PZ1 PERF] restore_step4_campaign_project_state total={total_ms}ms "
                f"reason={reason} target={str(getattr(host, '_step4_dataset_mode', '') or '-')} "
                f"train_unlocked={int(bool(getattr(host, '_step4_train_unlocked', False)))} "
                f"phases=[{', '.join(perf_phases) or 'no_slow_phase'}]"
            )
        except Exception:
            pass

    datasets_dir = Path(host._campaign_datasets_dir) if host._campaign_datasets_dir else None

    try:
        host.split_src_var.set("")
    except Exception:
        pass

    try:
        if datasets_dir is not None:
            host.split_out_var.set(str(datasets_dir / "[NAZWA_ZRODLA]_Split_[DATA_I_CZAS]"))
        else:
            host.split_out_var.set(str(host._get_datasets_base_dir() / "[NAZWA_ZRODLA]_Split_[DATA_I_CZAS]"))
    except Exception:
        pass

    try:
        host.dataset_var.set("")
    except Exception:
        pass

    try:
        if datasets_dir is not None:
            host.ds_out_var.set(str(datasets_dir / "Plates_CVAT_[DATA_I_CZAS]"))
        else:
            host.ds_out_var.set(str(host._get_datasets_base_dir() / "Plates_CVAT_[DATA_I_CZAS]"))
    except Exception:
        pass

    campaign_target = CAMPAIGN.get_iteration_target()
    remembered_target = campaign_target if campaign_target in ("char", "plate") else host.get_campaign_training_target()
    if remembered_target not in ("char", "plate"):
        remembered_target = "char"

    latest_source = None
    ready_dataset = None
    ready_target = remembered_target
    latest_xml = ""
    latest_xml_images = ""
    current_iter_images = ""

    if remembered_target == "char":
        fast_dataset_record = _current_iteration_step4_dataset_record("char")
        fast_dataset_root = _resolve_step4_dataset_record_root(fast_dataset_record)
        fast_counts = _step4_dataset_record_counts(fast_dataset_record)
        if (
            fast_dataset_root is not None
            and int(fast_counts.get("train", 0) or 0) > 0
            and int(fast_counts.get("val", 0) or 0) > 0
        ):
            ready_dataset = fast_dataset_root
            ready_target = "char"
        else:
            pz3_source = _current_iteration_pz3_source_contract()
            pz3_source_path = str(pz3_source.get("dataset_path") or "").strip()
            if pz3_source_path:
                try:
                    pz3_root = Path(pz3_source_path)
                    if pz3_root.exists() and pz3_root.is_dir():
                        latest_source = pz3_root
                except Exception:
                    latest_source = None

    if datasets_dir is not None and datasets_dir.exists():
        if latest_source is None:
            source_candidates = host._find_dataset_source_candidates(datasets_dir)
            if source_candidates:
                latest_source = max(source_candidates, key=lambda p: p.stat().st_mtime)

        if remembered_target == "plate":
            plate_dataset_info = host._resolve_campaign_plate_ready_dataset(datasets_dir)
            if not bool(plate_dataset_info.get("stale")):
                ready_dataset = plate_dataset_info.get("path")
                if ready_dataset is not None:
                    ready_target = "plate"
        elif ready_dataset is None:
            ready_candidates = host._find_ready_dataset_candidates(datasets_dir)

            preferred = [rec for rec in ready_candidates if rec[1] == remembered_target]
            if preferred:
                ready_dataset, ready_target, _ = max(preferred, key=lambda rec: rec[2])
            elif ready_candidates:
                ready_dataset, ready_target, _ = max(ready_candidates, key=lambda rec: rec[2])
    _perf_mark("dataset_scan")

    if remembered_target != "plate":
        try:
            auto_dir = CAMPAIGN.get_dir("auto_ann")
            if auto_dir is not None:
                xml_files = list(Path(auto_dir).rglob("annotations.xml"))
                if xml_files:
                    latest_xml_path = max(xml_files, key=lambda p: p.stat().st_mtime)
                    latest_xml = str(latest_xml_path)
                    manifest_path = latest_xml_path.parent / "run_manifest.json"
                    if manifest_path.exists():
                        try:
                            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        except Exception:
                            manifest = {}
                        manifest_input = str(manifest.get("input_dir") or "").strip() if isinstance(manifest, dict) else ""
                        if manifest_input and Path(manifest_input).exists():
                            latest_xml_images = manifest_input
        except Exception:
            latest_xml = ""
            latest_xml_images = ""
    _perf_mark("xml_scan")

    try:
        raw_dir = CAMPAIGN.get_dir("raw")
        iter_num = CAMPAIGN.get_current_iteration_num()
        iter_source_dir = CAMPAIGN.get_iteration_image_source_dir(iter_num) or CAMPAIGN.get_iteration_raw_dir(iter_num)
        if iter_source_dir is not None and Path(iter_source_dir).exists():
            current_iter_images = str(Path(iter_source_dir))
        elif raw_dir is not None:
            iter_dir = Path(raw_dir) / f"Iteracja_{iter_num:03d}"
            if iter_dir.exists():
                current_iter_images = str(iter_dir)
    except Exception:
        current_iter_images = ""

    if remembered_target == "plate":
        approved_source = host._get_campaign_plate_builder_source()
        approved_xml = str(approved_source.get("xml_path") or "").strip()
        approved_images_dir = str(approved_source.get("images_dir") or "").strip()
        if approved_xml and approved_images_dir:
            latest_xml = approved_xml
            latest_xml_images = approved_images_dir

    if latest_source is not None:
        try:
            host.split_src_var.set(str(latest_source))
            split_base = latest_source.parent if latest_source.parent != datasets_dir else Path(host._campaign_datasets_dir)
            host.split_out_var.set(str(split_base / f"{latest_source.name}_Split_[DATA_I_CZAS]"))
        except Exception:
            pass

    try:
        host.cvat_xml_var.set(latest_xml)
    except Exception:
        pass

    try:
        host.cvat_images_var.set(latest_xml_images or current_iter_images)
    except Exception:
        pass

    active_target = campaign_target if campaign_target in ("char", "plate") else ready_target
    host._step4_dataset_mode = active_target
    host.set_campaign_training_target(active_target)
    host._step4_route_selected = bool(campaign_target in ("char", "plate"))
    host._step4_train_unlocked = False
    host._step4_campaign_finish_ready = False

    finish_state = {}
    try:
        finish_state = host.get_campaign_step4_finish_state(iteration_target=active_target) or {}
    except Exception:
        try:
            finish_state = CAMPAIGN.get_step4_finish_state() or {}
        except Exception:
            finish_state = {}
    try:
        current_campaign_step = int(CAMPAIGN.get_current_step() or 0)
    except Exception:
        current_campaign_step = 0

    finish_ready = bool(finish_state.get("ready", False))
    finish_run_id = str(finish_state.get("run_id", "") or "").strip()
    finish_target = str(finish_state.get("target", "") or "").strip().lower()
    finish_iteration = int(finish_state.get("iteration", 0) or 0)
    if finish_target not in ("char", "plate"):
        finish_target = active_target

    step4_already_closed = current_campaign_step > 4
    if step4_already_closed:
        if finish_ready or finish_run_id:
            try:
                CAMPAIGN.set_step4_finish_state(False)
            except Exception:
                pass
        finish_ready = False
        finish_run_id = ""
        finish_target = active_target
        finish_iteration = 0
    elif not finish_ready or not finish_run_id:
        # Do not reopen T06 from training history alone. A completed run is a
        # candidate; the user must explicitly choose the project model.
        finish_ready = False
        finish_run_id = ""
        finish_target = active_target
        finish_iteration = 0

    if finish_ready and finish_target == active_target and finish_iteration == int(CAMPAIGN.get_current_iteration_num() or 0):
        host._step4_campaign_finish_ready = True
        host._step4_train_unlocked = True
        host.current_run_id = finish_run_id or host.current_run_id

    if ready_dataset is not None:
        try:
            host.dataset_var.set(str(ready_dataset))
        except Exception:
            pass

        host._step4_train_unlocked = True

        try:
            host._append_step4_builder_log(
                f"[KAMPANIA] Odtworzono gotowy dataset projektu: {ready_dataset.name}. "
                "Przejście do treningu zostało odblokowane, a ponowne przygotowanie datasetu pozostaje opcjonalne."
            )
        except Exception:
            pass
    else:
        try:
            host._append_step4_builder_log(
                "[KAMPANIA] Brak gotowego datasetu treningowego dla tego projektu. "
                "Pozostaję w pz1."
            )
        except Exception:
            pass

    try:
        host._refresh_step4_dataset_mode_ui()
    except Exception:
        pass
    _perf_mark("dataset_ui")

    should_refresh_train_ui_now = bool(getattr(host, "_step4_train_tab_built", False))
    if should_refresh_train_ui_now:
        try:
            host._refresh_base_model_choices()
        except Exception:
            pass
        _perf_mark("base_models")

    try:
        host._step4_suppress_base_model_state_save = True
        if not _apply_campaign_project_base_model_selection(host, active_target):
            if not host._apply_saved_step4_training_model_selection(active_target):
                if active_target == "plate":
                    host.base_model_var.set(host._get_preferred_plate_pose_base_model())
                    host.base_custom_var.set("")
                else:
                    host.base_model_var.set(host._get_default_base_model_for_mode("char"))
                    host.base_custom_var.set("")
    except Exception:
        pass
    finally:
        host._step4_suppress_base_model_state_save = False
    _perf_mark("model_selection")

    if should_refresh_train_ui_now:
        try:
            host._refresh_training_start_state()
        except Exception:
            pass
        try:
            host._schedule_step4_deferred_model_refresh()
        except Exception:
            pass
        _perf_mark("train_ui")

    try:
        host._refresh_step4_campaign_navigation_ui()
    except Exception:
        pass
    _perf_mark("navigation")

    try:
        campaign_active = bool(CAMPAIGN.get_active_project_name())
        if campaign_active and bool(getattr(host, "_step4_train_unlocked", False)):
            host._ensure_step4_train_tab_built()
            target_tab = host.tab_train
        else:
            target_tab = host.tab_dataset if campaign_active else host.tab_train
        host.main_nb.select(target_tab)
    except Exception:
        pass
    _perf_mark("tab_select")

    try:
        if bool(CAMPAIGN.get_active_project_name()) and bool(getattr(host, "_step4_train_unlocked", False)):
            host._select_step4_analysis_tab(host.hist_tab)
    except Exception:
        pass
    _perf_mark("analysis_select")

    try:
        if host._step4_route_selected:
            host._clear_step4_guidance()
        else:
            host._guide_step4_route_selection()
    except Exception:
        pass
    _perf_mark("guidance")
    _perf_log("ok")


def return_to_campaign_from_step4(host: "TrainingTab"):
    if not CAMPAIGN.get_active_project_name():
        return

    try:
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
        iteration_state = dict(CAMPAIGN.get_iteration_state(iteration_num=current_iteration) or {})
        session = dict(iteration_state.get("step4_work_session") or {})
        if bool(session.get("active")) and str(session.get("work_area", "") or "").strip().lower() == "z4":
            now = datetime.datetime.now().isoformat(timespec="seconds")
            session.update(
                {
                    "active": False,
                    "state": "returned_to_graph",
                    "returned_at": now,
                    "updated_at": now,
                }
            )
            CAMPAIGN.upsert_iteration_state(
                iteration_num=current_iteration,
                updates={"step4_work_session": session},
            )
    except Exception as exc:
        logger.debug(f"Nie udało się domknąć znacznika przerwanej pracy Z4: {exc}")

    try:
        campaign_tab = host.app.tabs.get("campaign")
        if campaign_tab:
            try:
                campaign_tab.request_wizard_stage_focus(
                    step_num=min(max(int(CAMPAIGN.get_current_step() or 4), 1), 4)
                )
            except Exception:
                pass
            campaign_tab._refresh_dashboard()
    except Exception:
        pass

    try:
        host.app.update_campaign_tab_access()
    except Exception:
        pass

    try:
        host.app.update_status(
            "Wracam do grafu kampanii.",
            "info"
        )
    except Exception:
        pass

    try:
        host.app.open_controlled_tab("campaign")
    except Exception:
        pass


def finish_campaign_step4(host: "TrainingTab"):
    if not CAMPAIGN.get_active_project_name():
        return False
    target = str(CAMPAIGN.get_iteration_target() or host.get_campaign_training_target() or "").strip().lower()

    if not host._step4_campaign_finish_ready:
        finish_state = {}
        try:
            target = str(CAMPAIGN.get_iteration_target() or host.get_campaign_training_target() or "").strip().lower()
            finish_state = dict(host.get_campaign_step4_finish_state(iteration_target=target) or {})
        except Exception:
            try:
                finish_state = dict(CAMPAIGN.get_step4_finish_state() or {})
            except Exception:
                finish_state = {}
        try:
            current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
        except Exception:
            current_iteration = 0
        finish_ready = bool(finish_state.get("ready", False))
        finish_iteration = int(finish_state.get("iteration", 0) or 0)
        if not (finish_ready and finish_iteration == current_iteration):
            return False
        host._step4_campaign_finish_ready = True
        run_id = str(finish_state.get("run_id", "") or "").strip()
        if run_id:
            try:
                host.current_run_id = run_id
            except Exception:
                pass

    campaign_tab = None
    try:
        campaign_tab = host.app.tabs.get("campaign") if getattr(host.app, "tabs", None) else None
    except Exception:
        campaign_tab = None

    next_mode = None
    if campaign_tab is not None and hasattr(campaign_tab, "_ask_iteration_advance_mode"):
        try:
            next_mode = campaign_tab._ask_iteration_advance_mode(completed_with_training=True)
        except Exception:
            next_mode = None
        if str(next_mode or "").strip().lower() not in {"reuse_input", "new_input"}:
            try:
                host.app.update_status(
                    f"Pozostajesz w {'E4T' if target == 'plate' else 'E4Z' if target == 'char' else 'E4T/E4Z'}. Iteracja nie została domknięta, więc zatwierdzenie pozostaje dostępne.",
                    "info",
                )
            except Exception:
                pass
            try:
                campaign_tab.request_wizard_stage_focus(step_num=4)
                campaign_tab._refresh_dashboard()
            except Exception:
                pass
            try:
                host._refresh_step4_campaign_navigation_ui()
            except Exception:
                pass
            return True

    try:
        CAMPAIGN.set_current_step(5)
    except Exception:
        return False

    host._step4_campaign_finish_ready = False
    try:
        CAMPAIGN.set_step4_finish_state(False)
    except Exception:
        pass

    try:
        host._append_train_log(
            "[KAMPANIA] Użytkownik domknął etap iteracji. "
            "Cykl iteracji został domknięty i można rozpocząć kolejną iterację."
        )
    except Exception:
        pass

    if campaign_tab is not None and str(next_mode or "").strip().lower() in {"reuse_input", "new_input"}:
        # No intermediate dataset/graph rebuild: the iteration worker will
        # publish the new E1 once its input is prepared.
        if host.app._get_selected_tab_key() != "campaign":
            host.app.open_controlled_tab("campaign")
        campaign_tab._start_iteration_advance(str(next_mode).strip().lower())
        return True

    try:
        host.main_nb.select(host.tab_dataset)
    except Exception:
        pass

    try:
        if campaign_tab:
            try:
                campaign_tab.request_wizard_stage_focus(step_num=4)
            except Exception:
                pass
            campaign_tab._refresh_dashboard()
    except Exception:
        pass

    try:
        host._refresh_step4_campaign_navigation_ui()
    except Exception:
        pass

    try:
        host.app.update_status(
            "Etap iteracji został zamknięty. W kampanii możesz rozpocząć nową iterację od tego samego zestawu zdjęć (E2) albo od nowego zestawu zdjęć (E1). Aktywne modele projektu pozostały zachowane.",
            "info"
        )
    except Exception:
        pass

    try:
        host.app.open_controlled_tab("campaign")
    except Exception:
        pass

    if campaign_tab is not None and str(next_mode or "").strip().lower() in {"reuse_input", "new_input"}:
        try:
            campaign_tab._start_iteration_advance(str(next_mode or "").strip().lower())
        except Exception:
            pass

    return True


def complete_campaign_project(host: "TrainingTab"):
    if not CAMPAIGN.get_active_project_name():
        return

    if not host._step4_campaign_finish_ready:
        return

    should_finish = host.app.themed_confirm(
        "Zakonczenie projektu",
        "Czy oznaczyc ten projekt jako zakończony?\n\n"
        "Projekt pozostanie dostępny do przegladu, ale dashboard nie będzie już prowadzil do nowej iteracji, "
        "dopoki ręcznie go nie wznowisz.",
        parent=host.frame,
        confirm_label="Zakończ projekt",
        tone="info"
    )
    if not should_finish:
        return

    if not CAMPAIGN.complete_project():
        return

    host._step4_campaign_finish_ready = False
    try:
        CAMPAIGN.set_step4_finish_state(False)
    except Exception:
        pass

    try:
        host._append_train_log(
            "[KAMPANIA] Użytkownik oznaczyl projekt jako zakończony. "
            "Nowa iteracja nie zostanie już proponowana, dopoki projekt nie zostanie wznowiony."
        )
    except Exception:
        pass

    try:
        host.main_nb.select(host.tab_dataset)
    except Exception:
        pass

    try:
        campaign_tab = host.app.tabs.get("campaign")
        if campaign_tab:
            campaign_tab._refresh_dashboard()
    except Exception:
        pass

    try:
        host._refresh_step4_campaign_navigation_ui()
    except Exception:
        pass

    try:
        project_name = CAMPAIGN.get_active_project_name()
        if project_name:
            host.app.update_status(
                f"Projekt '{project_name}' został oznaczony jako zakończony.",
                "info"
            )
    except Exception:
        pass

    try:
        host.app.update_campaign_tab_access()
    except Exception:
        pass

    try:
        host.app.open_controlled_tab("campaign")
    except Exception:
        pass


def promote_trained_model_to_campaign_if_needed(host: "TrainingTab"):
    target = (host._pending_campaign_model_type or "").strip().lower()
    if target not in ("char", "plate"):
        return False

    if not host.current_run_id:
        return False

    best_model = host._find_best_weights_for_run(host.current_run_id)
    if best_model is None or not best_model.exists():
        return False
    try:
        label = "znakow" if target == "char" else "tablic"
        host._append_train_log(
            f"[MODEL] Trening utworzyl kandydata modelu {label}: {best_model}. "
            f"Wynik bramki {_training_finish_gate_display_id()} nie zostal podpięty automatycznie."
        )
    except Exception:
        pass
    try:
        campaign_tab = host.app.tabs.get("campaign")
        if campaign_tab:
            campaign_tab._refresh_dashboard()
    except Exception:
        pass
    return False


def complete_campaign_step4_if_needed(host: "TrainingTab", target: str) -> bool:
    target = str(target or "").strip().lower()
    if target not in ("char", "plate"):
        return False

    if not CAMPAIGN.get_active_project_name():
        return False

    try:
        label = "znaków" if target == "char" else "tablic"
        host._append_train_log(
            f"[KAMPANIA] Model {label} został wypromowany do projektu. "
            "Możesz teraz domknąć etap iteracji albo oznaczyć cały projekt jako zakończony."
        )
    except Exception:
        pass

    try:
        campaign_tab = host.app.tabs.get("campaign")
        if campaign_tab:
            campaign_tab._refresh_dashboard()
    except Exception:
        pass

    try:
        host.app.update_status(
            f"Zakończono wybór wyniku toru '{target}'. Model został zapisany jako wynik bramki {_training_finish_gate_display_id()}.",
            "info"
        )
    except Exception:
        pass

    return True


def show_training_completion_summary(host: "TrainingTab", *, promoted: bool, can_finish_step4: bool):
    run_id = str(host.current_run_id or "").strip()
    if not run_id:
        return

    if str(getattr(host, "_last_training_completion_summary_run_id", "") or "").strip() == run_id:
        return

    run = None
    try:
        run = host.history.get_run(run_id)
    except Exception:
        run = None

    target = host.get_campaign_training_target()
    if target not in ("char", "plate"):
        target = str(getattr(host, "_pending_campaign_model_type", "") or "").strip().lower()

    if target == "char":
        target_label = "znaków"
    elif target == "plate":
        target_label = "tablic"
    else:
        target_label = "projektu"

    status_value = str(getattr(run, "status", "") or "").strip().lower() if run is not None else ""
    success = bool(status_value == TrainingStatus.COMPLETED.value)
    title = "Podsumowanie treningu"
    tone = "success" if success else "warning"

    if status_value == TrainingStatus.FAILED.value:
        title = "Trening zakończony błędem"
        tone = "error"
        try:
            host.app.themed_error(
                title,
                host._build_training_failure_message(run),
                parent=host.frame,
            )
            host._last_training_completion_summary_run_id = run_id
        except Exception:
            pass
        return

    lines = []
    if run is not None:
        lines.append(f"Run: {run.name}")
        lines.append(f"Preset / plik startowy YOLO: {run.base_model}")
        lines.append(f"Dataset: {run.dataset_path}")
        if float(getattr(run, "best_map50_95", 0.0) or 0.0) > 0.0:
            lines.append(f"Najlepsze mAP50-95: {float(run.best_map50_95):.3f}")

    if can_finish_step4:
        if promoted:
            lines.append("")
            lines.append(f"Model {target_label} został podpięty jako wynik bramki {_training_finish_gate_display_id()}.")
        else:
            lines.append("")
            lines.append(f"Ukończony model {target_label} jest kandydatem. Wynik bramki {_training_finish_gate_display_id()} nie został jeszcze podpięty.")
        if not promoted:
            lines.append(
                f"Dalej: wybierz jawnie wynik bramki {_training_finish_gate_display_id()} w historii treningow albo w rankingu. "
                f"Dopiero po takim wyborze bramka {_training_finish_gate_display_id()} bedzie gotowa do zatwierdzenia."
            )
        if promoted and str(target or "").strip().lower() == "char":
            lines.append(
                f"Dalej: wróć do grafu i użyj pola Zatwierdź na bramce {_training_finish_gate_display_id()}. "
                "To utrwali wynik treningu jako finał tej iteracji. Jeśli chcesz, możesz jeszcze wrócić do PZ1 i poprawić split datasetu znaków."
            )
        elif promoted:
            lines.append(
                f"Dalej: wróć do grafu i użyj pola Zatwierdź na bramce {_training_finish_gate_display_id()}. "
                "To utrwali wynik treningu jako finał tej iteracji. W Z4 możesz jeszcze przebudować dataset tablic w PZ1, jeśli chcesz."
            )
    else:
        lines.append("")
        if success:
            lines.append("Trening zakończył się poprawnie.")
        else:
            lines.append("Trening został zatrzymany albo zakończył się błędem.")
        lines.append("Dalej: możesz zmienić ustawienia i uruchomić kolejny trening.")

    try:
        host.app.themed_info(
            title,
            "\n".join(lines).strip(),
            parent=host.frame,
            tone=tone,
        )
        host._last_training_completion_summary_run_id = run_id
    except Exception:
        pass


def poll_training_completion(host: "TrainingTab"):
    """
    Lekki polling końca treningu:
    - czeka aż trainer.is_training spadnie do False
    - jeśli powstał best.pt, promuje model do projektu
    - w trybie kampanijnym odblokowuje ręczne zakończenie kroku 4
    """
    try:
        is_training = bool(getattr(host.trainer, "is_training", False))

        if is_training:
            host._training_completion_poll_job = host.frame.after(3000, host._poll_training_completion)
            return

        host._training_completion_poll_job = None
        promoted = promote_trained_model_to_campaign_if_needed(host)

        try:
            host._load_history()
        except Exception:
            pass

        run_status = ""
        try:
            if host.current_run_id:
                current_run = host.history.get_run(str(host.current_run_id))
                run_status = str(getattr(current_run, "status", "") or "").strip().lower()
        except Exception:
            run_status = ""

        try:
            host._remember_campaign_training_run_in_registry(
                run_id=str(host.current_run_id or "").strip(),
                status=run_status,
                target=host.get_campaign_training_target(),
            )
        except Exception:
            pass

        campaign_active = bool(CAMPAIGN.get_active_project_name())
        can_finish_step4 = campaign_active and bool(host.current_run_id)
        run_failed = bool(run_status == TrainingStatus.FAILED.value)

        if run_status == TrainingStatus.PAUSED.value:
            host._step4_campaign_finish_ready = False
            try:
                if CAMPAIGN.get_active_project_name():
                    CAMPAIGN.set_step4_finish_state(False)
            except Exception:
                pass
            host._set_training_ui_idle_state(
                "Trening wstrzymany. Ten run możesz wznowić później.",
                "#d35400"
            )
            try:
                host._append_train_log(
                    "[KAMPANIA] Run został wstrzymany. Etap 4 nie został jeszcze domknięty."
                )
            except Exception:
                pass
            try:
                host._refresh_step4_campaign_navigation_ui()
            except Exception:
                pass
            try:
                show_training_completion_summary(host, promoted=False, can_finish_step4=False)
            except Exception:
                pass
            return

        if run_failed:
            host._step4_campaign_finish_ready = False
            try:
                if CAMPAIGN.get_active_project_name():
                    CAMPAIGN.set_step4_finish_state(False)
            except Exception:
                pass
            host._set_training_ui_idle_state(
                "Trening zakończył się błędem. Bieżący węzeł treningowy nie może zostać jeszcze domknięty.",
                "#c0392b"
            )
            try:
                host._append_train_log(
                    "[KAMPANIA] Run zakończył się błędem. Najpierw popraw ustawienia albo dataset i uruchom trening ponownie."
                )
            except Exception:
                pass
            try:
                host._refresh_step4_campaign_navigation_ui()
            except Exception:
                pass
            try:
                show_training_completion_summary(host, promoted=False, can_finish_step4=False)
            except Exception:
                pass
            return

        finish_available = bool(
            can_finish_step4
            and promoted
            and run_status == TrainingStatus.COMPLETED.value
        )

        if finish_available:
            host._step4_campaign_finish_ready = True
            finish_target = host.get_campaign_training_target()
            if finish_target not in ("char", "plate"):
                finish_target = (host._pending_campaign_model_type or "").strip().lower()
            try:
                CAMPAIGN.set_step4_finish_state(
                    True,
                    run_id=str(host.current_run_id or ""),
                    target=finish_target,
                    iteration_num=int(CAMPAIGN.get_current_iteration_num() or 0),
                    selection_confirmed=True,
                    model_path=str(host._find_best_weights_for_run(host.current_run_id) or ""),
                )
            except Exception:
                pass
            host._set_training_ui_idle_state(
                f"Trening zakończony. Wynik jest wybrany: wróć do grafu i zatwierdź bramkę {_training_finish_gate_display_id()}, aby utrwalić finał iteracji.",
                "#1e8449"
            )
        else:
            host._step4_campaign_finish_ready = False
            try:
                if CAMPAIGN.get_active_project_name():
                    CAMPAIGN.set_step4_finish_state(False)
            except Exception:
                pass
            if run_status == TrainingStatus.COMPLETED.value:
                try:
                    host._preferred_campaign_training_result_run_id = str(host.current_run_id or "").strip()
                except Exception:
                    pass
                host._set_training_ui_idle_state(
                    (
                        "Trening zakończony. Run jest kandydatem: wybierz go jako wynik bramki "
                        f"{_training_finish_gate_display_id()} albo uruchom kolejny trening."
                    ),
                    "#2c3e50",
                )
            else:
                host._set_training_ui_idle_state("Trening zakończony lub zatrzymany.", "#2c3e50")

        try:
            host._refresh_campaign_training_result_selector()
        except Exception:
            pass

        try:
            host._refresh_step4_campaign_navigation_ui()
        except Exception:
            pass

        try:
            show_training_completion_summary(
                host,
                promoted=promoted,
                can_finish_step4=bool(can_finish_step4 and run_status == TrainingStatus.COMPLETED.value),
            )
        except Exception:
            pass

        if finish_available:
            try:
                host._guide_step4_finish_action()
            except Exception:
                pass

    except Exception as exc:
        logger.error(f"Błąd pollingu końca treningu: {exc}")
        host._training_completion_poll_job = None
        host._pending_campaign_model_type = None
        host._step4_campaign_finish_ready = False
        try:
            if CAMPAIGN.get_active_project_name():
                CAMPAIGN.set_step4_finish_state(False)
        except Exception:
            pass
        host._set_training_ui_idle_state("Błąd monitorowania końca treningu.", "#c0392b")

        try:
            host._refresh_step4_campaign_navigation_ui()
        except Exception:
            pass
