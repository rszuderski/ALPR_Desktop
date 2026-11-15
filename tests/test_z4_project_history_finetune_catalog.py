from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")

def test_free_history_catalog_includes_projects():
    source = read("auto_annotation_tool/gui/z4_history_runtime.py")
    assert "CONFIG.DIR_9_PROJECTS" in source
    assert 'project_root / "5_training_runs"' in source

def test_history_rows_have_project_scope():
    source = read("auto_annotation_tool/gui/z4_training_runtime.py")
    assert 'f"project@{project_scope}"' in source
    assert 'source_target in {"plate", "char", "vehicle"}' in source

def test_external_completed_run_can_be_finetune_parent_in_free_mode():
    source = read("auto_annotation_tool/gui/z4_training_runtime.py")
    assert "active_target = str(self._get_selected_training_target()" in source
    assert "if CAMPAIGN.get_active_project_name():" in source

def test_parent_history_dir_is_persisted_for_resolution():
    source = read("auto_annotation_tool/gui/z4_training_metrics.py")
    assert "_step4_fine_tune_parent_history_dir" in source
    assert "external_history = TrainingHistory(" in source
