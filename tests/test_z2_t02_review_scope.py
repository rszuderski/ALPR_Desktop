from pathlib import Path
from types import SimpleNamespace

from auto_annotation_tool.gui import z2_layout_ui_runtime
from auto_annotation_tool.gui import z2_gt_review


ROOT = Path(__file__).resolve().parents[1]


def _ann(name: str):
    return SimpleNamespace(filename=name)


def _host(scope_names):
    return SimpleNamespace(
        _campaign_graph_entry_context={"z2_work_mode": "t02_at_review"},
        current_annotation_run_dir=Path("run"),
        last_staging_run_dir=None,
        _load_annotation_run_manifest=lambda _run: {
            "input_scope_filenames": list(scope_names)
        },
        _is_free_mode_session_context=lambda: True,
        _get_preview_metric_filter_thresholds=lambda: (0.0, 0.0),
    )


def test_t02_scope_helper_reads_manifest_input_scope_names():
    host = _host(["BI360FE_001.jpg", "nested/BI8000E_001.jpg"])

    assert z2_layout_ui_runtime._get_t02_review_scope_filenames(host) == {
        "bi360fe_001.jpg",
        "bi8000e_001.jpg",
    }


def test_t02_filter_keeps_only_imported_at_scope(monkeypatch):
    monkeypatch.setattr(z2_gt_review, "missing_gt_filter_active", lambda _host: False)
    host = _host(["A.jpg", "B.jpg"])
    entries = [
        (0, _ann("A.jpg")),
        (1, _ann("B.jpg")),
        (2, _ann("C.jpg")),
    ]

    result = z2_layout_ui_runtime._filter_preview_list_entries(host, entries)

    assert [ann.filename for _idx, ann in result] == ["A.jpg", "B.jpg"]


def test_t02_summary_uses_review_scope_as_denominator():
    source = (
        ROOT / "auto_annotation_tool" / "gui" / "z2_preview_workflow.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def _refresh_preview_list_summary")
    end = source.index("def _open_preview_metric_filter_modal", start)
    segment = source[start:end]

    assert "_get_t02_review_scope_filenames()" in segment
    assert "summary_total_images = len(t02_scope_filenames)" in segment
    assert 'f"Zdjęcia: {visible_count}/{summary_total_images}"' in segment


def test_t02_commit_cta_says_save_not_select():
    source = (
        ROOT / "auto_annotation_tool" / "gui" / "z2_miniflow_runtime.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def _prompt_t02_at_review_commit_choice")
    end = source.index("def _return_to_campaign_t02_at_review", start)
    segment = source[start:end]

    assert 'text="Zapisz kontrolę T02"' in segment
    assert 'text="Zapisz kontrolę i wybierz T02"' not in segment
