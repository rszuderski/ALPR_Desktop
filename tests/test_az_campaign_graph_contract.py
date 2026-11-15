from pathlib import Path
import ast

from auto_annotation_tool.campaign_resource_state import CampaignResourceSnapshot
from auto_annotation_tool.campaign_transition_evaluator import (
    CampaignTransitionEvalContext,
    is_transition_ready,
)
from auto_annotation_tool.campaign_transition_resource_report import (
    build_transition_resource_report,
)
from auto_annotation_tool.campaign_transition_specs import (
    get_transition_specs_for_edge,
)


def _snapshot(
    key: str,
    *,
    source: str,
    tone: str,
    counter: str = "",
    contract_ready: bool,
    review_required: bool = False,
) -> CampaignResourceSnapshot:
    return CampaignResourceSnapshot(
        key=key,
        canonical_key=key,
        code=key.upper(),
        label=key,
        source=source,
        tone=tone,
        counter_text=counter,
        meta={
            "contract_ready": bool(contract_ready),
            "contract_enforced": True,
            "review_required": bool(review_required),
        },
    )


def _ready_images() -> CampaignResourceSnapshot:
    return _snapshot(
        "images",
        source="project images",
        tone="success",
        counter="12",
        contract_ready=True,
    )


def _ready_plate_run() -> CampaignResourceSnapshot:
    return _snapshot(
        "plate_run",
        source="approved AT",
        tone="success",
        counter="12 obrazów / 18 tablic",
        contract_ready=True,
    )


def _pending_az() -> CampaignResourceSnapshot:
    return _snapshot(
        "char_run",
        source="Registry projektu",
        tone="warning",
        counter="8/12",
        contract_ready=False,
        review_required=True,
    )


def _ready_az() -> CampaignResourceSnapshot:
    return _snapshot(
        "char_run",
        source="Registry projektu",
        tone="success",
        counter="12/12",
        contract_ready=True,
    )


def test_t01_readiness_depends_on_images_or_at_not_pending_az():
    spec = get_transition_specs_for_edge("e1_to_e2")[0]

    with_images = CampaignTransitionEvalContext(
        selected_path="char_from_images",
        explicit_selected_path="char_from_images",
        current_step=1,
        image_count=12,
        resource_snapshots={
            "images": _ready_images(),
            "char_run": _pending_az(),
        },
    )
    assert is_transition_ready(spec, with_images)

    az_only = CampaignTransitionEvalContext(
        selected_path="char_from_images",
        explicit_selected_path="char_from_images",
        current_step=1,
        resource_snapshots={"char_run": _ready_az()},
    )
    assert not is_transition_ready(spec, az_only)


def test_t02_readiness_depends_on_at_not_az():
    spec = get_transition_specs_for_edge("e1_to_e3")[0]

    with_at_and_pending_az = CampaignTransitionEvalContext(
        selected_path="char_from_ready_plates",
        explicit_selected_path="char_from_ready_plates",
        current_step=1,
        resource_snapshots={
            "plate_run": _ready_plate_run(),
            "char_run": _pending_az(),
        },
    )
    assert is_transition_ready(spec, with_at_and_pending_az)

    ready_az_without_at = CampaignTransitionEvalContext(
        selected_path="char_from_ready_plates",
        explicit_selected_path="char_from_ready_plates",
        current_step=1,
        resource_snapshots={"char_run": _ready_az()},
    )
    assert not is_transition_ready(spec, ready_az_without_at)


def test_pending_az_is_optional_and_does_not_block_t02_resource_report():
    spec = get_transition_specs_for_edge("e1_to_e3")[0]
    report = build_transition_resource_report(
        (spec,),
        {
            "plate_run": _ready_plate_run(),
            "char_run": _pending_az(),
        },
        selected_path="char_from_ready_plates",
    )

    rows = {row.canonical_key: row for row in report.rows}
    assert rows["plate_run"].required
    assert rows["plate_run"].present
    assert not rows["char_run"].required
    assert not rows["char_run"].blocking_missing
    assert report.required_ready
    assert report.compact_status() == "OK"


def test_az_import_browser_cannot_approve_gate_or_choose_route():
    path = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "campaign_dashboard_ui.py"
    )
    source = path.read_text(encoding="utf-8-sig")
    module = ast.parse(source)

    node = next(
        item
        for item in ast.walk(module)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == "_open_az_project_import_browser"
    )
    block = ast.get_source_segment(source, node) or ""

    assert "import_project_az_bindings(" in block
    assert "approve_step" not in block
    assert "set_iteration_path" not in block
    assert "set_current_step" not in block


def test_char_run_graph_fallback_is_registry_based_not_planned_placeholder():
    path = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "campaign_dashboard_ui.py"
    )
    source = path.read_text(encoding="utf-8-sig")

    assert "import anotacji znaków jest zasobem planowanym" not in source
    assert "stan AZ jest odczytywany z registry projektu" in source
