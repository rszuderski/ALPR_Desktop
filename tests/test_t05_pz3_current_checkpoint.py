from types import SimpleNamespace

from auto_annotation_tool.gui import z3_campaign_flow as flow


class Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class NB:
    def __init__(self):
        self.selected = "PZ1"

    def select(self, value=None):
        if value is not None:
            self.selected = str(value)
        return self.selected


def test_direct_t05_pz3_uses_current_checkpoint_without_approved_source(monkeypatch, tmp_path):
    preview = tmp_path / "preview"
    preview.mkdir()
    (preview / "metadata.json").write_text("{}", encoding="utf-8")
    (preview / "images").mkdir()

    nb = NB()
    host = SimpleNamespace(
        preview_dir_var=Var(str(preview)),
        _campaign_step3_pz2_current_contract_ready=lambda: True,
        _get_saved_step3_preview_dir=lambda require_plates=True: str(preview),
        _is_usable_step3_preview_dir=lambda *args, **kwargs: True,
        _step3_linear_mode=True,
        _campaign_force_pz2_entry=False,
        _campaign_force_pz3_entry=False,
        _campaign_force_detect_entry=False,
        _campaign_graph_entry_context={},
        _campaign_step3_hold_pz2_after_reextract=False,
        main_nb=nb,
        tab_dataset="PZ3",
        go_to_substep_3=lambda force=False: nb.select("PZ3"),
    )

    monkeypatch.setattr(flow, "_ensure_campaign_pz2_preview_loaded", lambda *a, **k: True)
    monkeypatch.setattr(flow.CAMPAIGN, "get_step3_preview_dir", lambda: str(preview))
    monkeypatch.setattr(flow.CAMPAIGN, "set_step3_preview_dir", lambda value: None)
    monkeypatch.setattr(flow.CAMPAIGN, "set_step3_stage1_done", lambda value: None)
    monkeypatch.setattr(flow.CAMPAIGN, "set_step3_stage2_done", lambda value: None)
    monkeypatch.setattr(flow.CAMPAIGN, "set_step3_substep", lambda value: None)

    result = flow._try_direct_campaign_pz3_checkpoint_entry(
        host,
        {
            "graph_edge_key": "e3_to_e4",
            "graph_gate_id": "T05",
            "target_substep": "pz3",
            "force_pz3": "1",
        },
    )

    assert result["ok"] is True
    assert result["direct_pz3_checkpoint"] is True
    assert host._campaign_force_pz3_entry is True
    assert host._campaign_force_pz2_entry is False
    assert host._campaign_force_detect_entry is False
    assert host.main_nb.select() == "PZ3"


def test_direct_pz3_does_not_bypass_missing_current_pz2_contract(tmp_path):
    preview = tmp_path / "preview"
    preview.mkdir()

    host = SimpleNamespace(
        preview_dir_var=Var(str(preview)),
        _campaign_step3_pz2_current_contract_ready=lambda: False,
    )

    result = flow._try_direct_campaign_pz3_checkpoint_entry(
        host,
        {"target_substep": "pz3", "force_pz3": "1"},
    )
    assert result is None


def test_direct_pz3_does_not_intercept_non_pz3_entry():
    host = SimpleNamespace()
    assert flow._try_direct_campaign_pz3_checkpoint_entry(
        host,
        {"target_substep": "detect", "force_pz2": "1"},
    ) is None


def test_navigation_primes_force_pz3_before_dataset_selection():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (
        root / "auto_annotation_tool" / "gui" / "campaign_navigation.py"
    ).read_text(encoding="utf-8-sig")

    force_at = source.index("tab_char._campaign_force_pz3_entry = True")
    select_at = source.index("_select_subtab(tab_char.tab_dataset)", force_at)
    open_at = source.index("open_campaign_step3_entry", force_at)

    assert force_at < select_at < open_at


def test_open_campaign_entry_fast_path_precedes_approved_source_rebuild():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (
        root / "auto_annotation_tool" / "gui" / "z3_campaign_flow.py"
    ).read_text(encoding="utf-8-sig")

    open_at = source.index("def open_campaign_step3_entry(")
    fast_at = source.index("_try_direct_campaign_pz3_checkpoint_entry(", open_at)
    approved_at = source.index("source_for_campaign(CAMPAIGN)", open_at)

    assert open_at < fast_at < approved_at
