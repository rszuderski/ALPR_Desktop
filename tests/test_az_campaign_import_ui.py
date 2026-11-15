from pathlib import Path


def _source() -> str:
    path = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "campaign_dashboard_ui.py"
    )
    return path.read_text(encoding="utf-8-sig")


def test_char_run_primary_is_no_longer_hard_disabled():
    source = _source()

    # Sprawdzamy konkretnie dawną blokadę primary_enabled, a nie dowolne
    # wystąpienie row_key != "char_run" w innych, poprawnych gałęziach.
    old_primary_guard = (
        'primary_enabled = (\n'
        '                    spec_enabled\n'
        '                    and row_key != "char_run"\n'
        '                    and _button_enabled(old_buttons.get(row_key))\n'
        '                    and not annotation_dependency_blocked\n'
        '                )'
    )
    assert old_primary_guard not in source
    assert 'if row_key == "char_run":' in source
    assert 'az_target_crop_count = int(az_meta.get("crop_count", 0) or 0)' in source


def test_char_run_import_uses_az006_backend():
    source = _source()

    assert "def _open_az_project_import_browser()" in source
    assert "list_project_az_import_sources(" in source
    assert "import_project_az_bindings(" in source
    assert 'elif row_key == "char_run":' in source
    assert "_open_az_project_import_browser()" in source


def test_char_run_dependency_is_crop_based_not_image_based():
    source = _source()

    assert (
        "Najpierw przygotuj wyodrębnione tablice PZ1 dla projektu."
        in source
    )
    assert (
        "AZ można importować wyłącznie dla zgodnych logical crop_id."
        in source
    )


def test_az_import_browser_requires_explicit_importable_candidate():
    source = _source()

    assert "candidate.can_import" in source
    assert "state=tk.DISABLED" in source
    assert "imported_pending_review" in source
