import ast
import copy
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "auto_annotation_tool" / "gui" / "z3_campaign_flow.py"


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value


def _helper():
    source = FLOW.read_text(encoding="utf-8-sig")
    module = ast.parse(source)
    outer = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "open_campaign_step3_entry"
    )
    helper = next(
        node
        for node in outer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_can_continue_forced_pz3_without_reextract"
    )

    factory = ast.parse(
        "def factory(force_dataset_entry, host):\n"
        "    pass\n"
    ).body[0]
    factory.body = [copy.deepcopy(helper)] + ast.parse(
        "def _ret():\n"
        "    return _can_continue_forced_pz3_without_reextract\n"
    ).body[0].body

    ns = {}
    exec(
        compile(
            ast.Module(body=[factory], type_ignores=[]),
            str(FLOW),
            "exec",
        ),
        ns,
    )
    return ns["factory"]


def _host(*, contract_ready=True, preview_ready=True):
    return SimpleNamespace(
        can_restore_step3_substep=lambda substep: False,
        _campaign_step3_pz2_current_contract_ready=lambda: contract_ready,
        preview_dir_var=_Var(r"C:\preview_current"),
        _get_saved_step3_preview_dir=lambda require_plates=True: r"C:\preview_saved",
        _is_usable_step3_preview_dir=lambda *args, **kwargs: preview_ready,
    )


def test_explicit_pz3_accepts_current_pz2_checkpoint_even_on_source_mismatch():
    helper = _helper()(True, _host(contract_ready=True, preview_ready=True))

    assert helper({
        "reason": "preview_source_mismatch",
        "preview_dir": r"C:\preview_current",
    }) is True


def test_explicit_pz3_does_not_bypass_missing_current_pz2_contract():
    helper = _helper()(True, _host(contract_ready=False, preview_ready=True))

    assert helper({
        "reason": "preview_source_mismatch",
        "preview_dir": r"C:\preview_current",
    }) is False


def test_explicit_pz3_does_not_bypass_broken_preview():
    helper = _helper()(True, _host(contract_ready=True, preview_ready=False))

    assert helper({
        "reason": "preview_source_mismatch",
        "preview_dir": r"C:\preview_current",
    }) is False


def test_non_explicit_entry_keeps_existing_reextract_behavior():
    helper = _helper()(False, _host(contract_ready=True, preview_ready=True))

    assert helper({
        "reason": "preview_source_mismatch",
        "preview_dir": r"C:\preview_current",
    }) is False


def test_source_no_longer_hard_blocks_mismatch_before_checkpoint_validation():
    source = FLOW.read_text(encoding="utf-8-sig")
    start = source.index("def _can_continue_forced_pz3_without_reextract")
    end = source.index("def _refresh_forced_pz3_dataset_surface", start)
    body = source[start:end]

    assert 'refresh_state.get("reason") in {' not in body
    assert "_campaign_step3_pz2_current_contract_ready" in body
