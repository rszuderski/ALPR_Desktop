"""Lightweight sanity checks for the campaign state-machine graph.

Run with:

    python -m auto_annotation_tool.campaign_graph_sanity

The check intentionally avoids Tkinter and project loading. It verifies the
static graph contract that the GUI relies on.
"""

from __future__ import annotations

from .campaign_iteration_paths import STEP1_ITERATION_PATHS
from .campaign_resource_state import CampaignResourceSnapshot
from .campaign_transition_evaluator import CampaignTransitionEvalContext, is_transition_ready
from .campaign_transition_graph import CAMPAIGN_TRANSITION_GRAPH
from .campaign_transition_resource_report import build_transition_resource_report
from .campaign_transition_specs import (
    TRANSITION_SPECS,
    get_transition_action_specs_for_edge,
    get_transition_specs_for_edge,
)


def _fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _edge_keys() -> tuple[str, ...]:
    return tuple(edge.key for edge in CAMPAIGN_TRANSITION_GRAPH.edges)


def _edge(edge_key: str):
    edge = CAMPAIGN_TRANSITION_GRAPH.get_edge(edge_key)
    _require(edge is not None, f"missing graph edge {edge_key}")
    return edge


def _snapshot(
    key: str,
    *,
    counter: int = 0,
    source: str = "project",
    tone: str = "success",
    meta: dict | None = None,
) -> CampaignResourceSnapshot:
    return CampaignResourceSnapshot(
        key=key,
        canonical_key=key,
        code=key.upper(),
        label=key,
        requirement="required",
        source=source,
        validation="OK",
        tone=tone,
        counter_text=str(counter) if counter else "",
        meta=dict(meta or {}),
    )


def _assert_path_cycle(path_key: str, edge_sequence: tuple[str, ...], *, iterations: int = 3) -> None:
    _require(iterations >= 1, "path cycle simulation needs at least one iteration")
    for iteration in range(1, iterations + 1):
        current_stage = "E1"
        for edge_key in edge_sequence:
            edge = _edge(edge_key)
            _require(
                edge.source == current_stage,
                f"path {path_key} iteration {iteration} expected {current_stage} before {edge_key}, got {edge.source}",
            )
            specs = get_transition_specs_for_edge(edge_key)
            _require(specs, f"path {path_key} edge {edge_key} has no transition specs")
            for spec in specs:
                if spec.path_key:
                    _require(
                        spec.path_key == path_key,
                        f"path {path_key} edge {edge_key} exposes incompatible spec {spec.key}:{spec.path_key}",
                    )
            current_stage = edge.target
        _require(current_stage == "E1", f"path {path_key} iteration {iteration} does not return to E1")


def main() -> None:
    graph_edges = _edge_keys()
    graph_edge_set = set(graph_edges)

    _require("e1_to_e2" in graph_edge_set, "missing merged E1->E2 edge")
    _require("e1_to_e2_plate_training" not in graph_edge_set, "legacy plate T01 edge is still active")
    _require("e1_to_e2_char_from_images" not in graph_edge_set, "legacy char T02 edge is still active")

    e1_to_e2_specs = get_transition_specs_for_edge("e1_to_e2")
    _require(len(e1_to_e2_specs) == 1, "merged E1->E2 must expose exactly one transition spec")
    merged_spec = e1_to_e2_specs[0]
    _require(merged_spec.badge_id == "T01", "merged E1->E2 must be visible as T01")
    _require(merged_spec.path_key == "", "merged T01 must choose the path through actions, not a fixed path")

    actions = {action.key: action for action in merged_spec.actions}
    _require("select_plate_training" in actions, "T01 missing plate-training path action")
    _require("select_char_from_images" in actions, "T01 missing char-from-images path action")
    _require(
        actions["select_plate_training"].payload.get("path_key") == "plate_training",
        "T01 plate action points to the wrong path",
    )
    _require(
        actions["select_char_from_images"].payload.get("path_key") == "char_from_images",
        "T01 char action points to the wrong path",
    )

    for legacy_edge in ("e1_to_e2_plate_training", "e1_to_e2_char_from_images"):
        legacy_specs = get_transition_specs_for_edge(legacy_edge)
        _require(legacy_specs == e1_to_e2_specs, f"{legacy_edge} no longer aliases to merged T01")

    def _report_by_key(specs, snapshots, *, selected_path: str = ""):
        report = build_transition_resource_report(specs, snapshots, selected_path=selected_path)
        return {row.key: row for row in report.rows}

    t01_empty_report = _report_by_key(e1_to_e2_specs, {}, selected_path="")
    _require(
        t01_empty_report.get("images") is not None and t01_empty_report["images"].requirement == "required",
        "T01 resources must keep images required before a path is selected",
    )
    t01_unselected_ready_report = build_transition_resource_report(
        e1_to_e2_specs,
        {"images": _snapshot("images", counter=12)},
        selected_path="",
    )
    _require(
        t01_unselected_ready_report.compact_status() == "OK",
        "T01 resources should display OK with images even before a work-path choice",
    )
    t01_plate_report = _report_by_key(
        e1_to_e2_specs,
        {"images": _snapshot("images", counter=12)},
        selected_path="plate_training",
    )
    _require(
        t01_plate_report["images"].requirement == "route_required_plate" and t01_plate_report["images"].present,
        "T01 plate resources must show images as the fulfilled route requirement",
    )

    badge_ids = {spec.badge_id for spec in TRANSITION_SPECS}
    _require({"T01", "T02", "T03", "T04", "T05", "T06"}.issubset(badge_ids), "active badges must expose T01-T06")
    _require("T07" not in badge_ids, "T07 is still present as an active badge")

    for spec in TRANSITION_SPECS:
        if spec.edge_key == "e4_to_e1":
            continue
        _require(spec.edge_key in graph_edge_set, f"transition spec {spec.key} points to missing edge {spec.edge_key}")

    for path_key, path_def in STEP1_ITERATION_PATHS.items():
        edge_sequence = tuple(path_def.get("edge_sequence") or ())
        _require(edge_sequence, f"path {path_key} has no edge sequence")
        missing_edges = [edge for edge in edge_sequence if edge not in graph_edge_set]
        _require(not missing_edges, f"path {path_key} references missing graph edges: {missing_edges}")
        for edge_key in edge_sequence:
            specs = get_transition_specs_for_edge(edge_key)
            _require(specs, f"path {path_key} edge {edge_key} has no transition spec")
        _assert_path_cycle(path_key, edge_sequence, iterations=3)

    for training_return_edge in ("e4t_to_e1", "e4z_to_e1"):
        specs = get_transition_specs_for_edge(training_return_edge)
        _require(len(specs) == 1, f"{training_return_edge} must resolve to one T06 spec")
        _require(specs[0].badge_id == "T06", f"{training_return_edge} must resolve to T06")

    for edge_key in ("e2_to_e3", "e2_to_e4"):
        first_actions = {action.key for action in get_transition_action_specs_for_edge(edge_key, iteration_num=1)}
        later_actions = {action.key for action in get_transition_action_specs_for_edge(edge_key, iteration_num=2)}
        _require("open_z2_first" in first_actions, f"{edge_key} missing first-iteration Z2 action")
        _require("open_z2_later" not in first_actions, f"{edge_key} exposes later Z2 action in iteration 1")
        _require("open_z2_first" not in later_actions, f"{edge_key} exposes first Z2 action after iteration 1")
        _require("open_z2_later" in later_actions, f"{edge_key} missing later-iteration Z2 action")

    empty_ctx = CampaignTransitionEvalContext(selected_path="", image_count=0, resource_snapshots={})
    _require(not is_transition_ready(merged_spec, empty_ctx), "T01 must not be ready without images")

    image_ctx = CampaignTransitionEvalContext(
        selected_path="",
        image_count=12,
        resource_snapshots={"images": _snapshot("images", counter=12)},
    )
    _require(not is_transition_ready(merged_spec, image_ctx), "T01 must wait for an explicit work-path choice")

    inferred_path_ctx = CampaignTransitionEvalContext(
        selected_path="plate_training",
        image_count=12,
        resource_snapshots={"images": _snapshot("images", counter=12)},
    )
    _require(
        not is_transition_ready(merged_spec, inferred_path_ctx),
        "T01 must not treat an inferred/default path as a work-path choice",
    )

    image_path_ctx = CampaignTransitionEvalContext(
        selected_path="plate_training",
        explicit_selected_path="plate_training",
        image_count=12,
        resource_snapshots={"images": _snapshot("images", counter=12)},
    )
    _require(is_transition_ready(merged_spec, image_path_ctx), "T01 plate path should be ready with images and an explicit work-path choice")

    stale_image_path_ctx = CampaignTransitionEvalContext(
        selected_path="plate_training",
        explicit_selected_path="plate_training",
        image_count=12,
        resource_snapshots={
            "images": _snapshot(
                "images",
                counter=12,
                meta={
                    "contract_kind": "O",
                    "contract_ready": False,
                    "stale": True,
                },
            )
        },
    )
    _require(
        not is_transition_ready(merged_spec, stale_image_path_ctx),
        "T01 must not be ready when the O resource contract is stale",
    )

    plate_path_without_images_ctx = CampaignTransitionEvalContext(
        selected_path="plate_training",
        explicit_selected_path="plate_training",
        image_count=0,
        plate_material_count=20,
        material_ready=True,
        resource_snapshots={"plate_run": _snapshot("plate_run", counter=20, tone="success")},
    )
    _require(
        not is_transition_ready(merged_spec, plate_path_without_images_ctx),
        "T01 plate path must require a new image pool even when plate material exists",
    )

    char_image_path_ctx = CampaignTransitionEvalContext(
        selected_path="char_from_images",
        explicit_selected_path="char_from_images",
        image_count=12,
        resource_snapshots={"images": _snapshot("images", counter=12)},
    )
    _require(
        is_transition_ready(merged_spec, char_image_path_ctx),
        "T01 char path should be ready with images for preparing plate annotations",
    )

    char_existing_plate_ctx = CampaignTransitionEvalContext(
        selected_path="char_from_images",
        explicit_selected_path="char_from_images",
        image_count=0,
        plate_material_count=20,
        material_ready=True,
        resource_snapshots={"plate_run": _snapshot("plate_run", counter=20, tone="success")},
    )
    _require(
        is_transition_ready(merged_spec, char_existing_plate_ctx),
        "T01 char path should be ready with existing plate material even when images are exhausted",
    )

    t02_specs = get_transition_specs_for_edge("e1_to_e3")
    _require(len(t02_specs) == 1 and t02_specs[0].badge_id == "T02", "E1->E3 must resolve to T02")
    t02_spec = t02_specs[0]
    no_plate_ctx = CampaignTransitionEvalContext(
        selected_path="char_from_ready_plates",
        plate_material_count=0,
        material_ready=False,
        resource_snapshots={},
    )
    _require(not is_transition_ready(t02_spec, no_plate_ctx), "T02 must not be ready without a plate source")
    t02_empty_report = _report_by_key(t02_specs, {}, selected_path="char_from_ready_plates")
    _require(
        t02_empty_report.get("plate_run") is not None and t02_empty_report["plate_run"].requirement == "required",
        "T02 resources must require AT/plate source",
    )

    wrong_path_ctx = CampaignTransitionEvalContext(
        selected_path="plate_training",
        plate_material_count=20,
        material_ready=True,
        resource_snapshots={"plate_run": _snapshot("plate_run", counter=20)},
    )
    _require(not is_transition_ready(t02_spec, wrong_path_ctx), "T02 must not be ready on the plate-training path")

    plate_ctx = CampaignTransitionEvalContext(
        selected_path="char_from_ready_plates",
        plate_material_count=20,
        material_ready=True,
        resource_snapshots={"plate_run": _snapshot("plate_run", counter=20)},
    )
    _require(is_transition_ready(t02_spec, plate_ctx), "T02 should be ready with an existing plate source")

    positive_contract_ctx = CampaignTransitionEvalContext(
        selected_path="char_from_ready_plates",
        plate_material_count=0,
        material_ready=False,
        resource_snapshots={
            "plate_run": _snapshot(
                "plate_run",
                counter=0,
                tone="warning",
                meta={
                    "contract_kind": "O->AT",
                    "contract_ready": True,
                },
            )
        },
    )
    _require(
        is_transition_ready(t02_spec, positive_contract_ctx),
        "T02 should accept a positive non-enforced AT contract",
    )
    t03_positive_report = _report_by_key(
        t02_specs,
        {
            "plate_run": _snapshot(
                "plate_run",
                tone="warning",
                meta={
                    "contract_kind": "O->AT",
                    "contract_ready": True,
                },
            )
        },
        selected_path="char_from_ready_plates",
    )
    _require(
        t03_positive_report["plate_run"].present,
        "T02 resource report should show a positive AT contract as present",
    )

    negative_contract_fallback_ctx = CampaignTransitionEvalContext(
        selected_path="char_from_ready_plates",
        plate_material_count=20,
        material_ready=True,
        resource_snapshots={
            "plate_run": _snapshot(
                "plate_run",
                counter=20,
                tone="success",
                meta={
                    "contract_kind": "O->AT",
                    "contract_ready": False,
                },
            )
        },
    )
    _require(
        is_transition_ready(t02_spec, negative_contract_fallback_ctx),
        "T02 must not let a negative non-enforced contract override legacy-ready material",
    )

    draft_plate_ctx = CampaignTransitionEvalContext(
        selected_path="char_from_ready_plates",
        plate_material_count=0,
        material_ready=False,
        resource_snapshots={"plate_run": _snapshot("plate_run", counter=20, tone="warning")},
    )
    _require(not is_transition_ready(t02_spec, draft_plate_ctx), "T02 must not be ready with draft plate annotations")

    stale_plate_ctx = CampaignTransitionEvalContext(
        selected_path="char_from_ready_plates",
        plate_material_count=20,
        material_ready=True,
        resource_snapshots={
            "plate_run": _snapshot(
                "plate_run",
                counter=20,
                tone="success",
                meta={
                    "contract_kind": "O->AT",
                    "contract_ready": False,
                    "requires_rematch": True,
                },
            )
        },
    )
    _require(
        not is_transition_ready(t02_spec, stale_plate_ctx),
        "T02 must not be ready when AT requires rematching to the current O resource",
    )

    t03_spec = get_transition_specs_for_edge("e2_to_e3")[0]
    t04_spec = get_transition_specs_for_edge("e2_to_e4")[0]
    approved_ctx = CampaignTransitionEvalContext(
        current_step=2,
        selected_path="char_from_images",
        resource_snapshots={"approved_plates": _snapshot("approved_plates", counter=12)},
    )
    _require(is_transition_ready(t03_spec, approved_ctx), "T03 should be ready with approved plates")
    _require(not is_transition_ready(t04_spec, approved_ctx), "T04 must not be ready on the char-from-images path")

    plate_training_ctx = CampaignTransitionEvalContext(
        current_step=2,
        selected_path="plate_training",
        resource_snapshots={"approved_plates": _snapshot("approved_plates", counter=12)},
    )
    _require(is_transition_ready(t04_spec, plate_training_ctx), "T04 should be ready with approved plates")
    _require(not is_transition_ready(t03_spec, plate_training_ctx), "T03 must not be ready on the plate-training path")

    t05_spec = get_transition_specs_for_edge("e3_to_e4")[0]
    dataset_ctx = CampaignTransitionEvalContext(
        current_step=3,
        selected_path="char_from_images",
        resource_snapshots={"char_dataset": _snapshot("char_dataset", counter=1, tone="success")},
    )
    _require(is_transition_ready(t05_spec, dataset_ctx), "T05 should be ready with an exported char dataset")

    t06_spec = get_transition_specs_for_edge("e4t_to_e1")[0]
    training_ctx = CampaignTransitionEvalContext(
        current_step=4,
        selected_path="plate_training",
        resource_snapshots={"training_result": _snapshot("training_result", counter=1, tone="success")},
    )
    _require(is_transition_ready(t06_spec, training_ctx), "T06 should be ready with a training result")

    pending_training_ctx = CampaignTransitionEvalContext(
        current_step=4,
        selected_path="plate_training",
        resource_snapshots={"training_result": _snapshot("training_result", source="completed run", tone="warning")},
    )
    _require(
        not is_transition_ready(t06_spec, pending_training_ctx),
        "T06 must not be ready when a training run exists but was not selected for the gate",
    )

    print(
        "OK campaign graph sanity: "
        f"{len(graph_edges)} edges, {len(TRANSITION_SPECS)} specs, {len(STEP1_ITERATION_PATHS)} paths"
    )


if __name__ == "__main__":
    main()
