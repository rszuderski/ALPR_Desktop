from pathlib import Path

from auto_annotation_tool.gui.z4_dataset_builder import (
    _format_step4_mz_completion_label,
)


def test_source_only_status_copy_is_operator_friendly():
    assert (
        _format_step4_mz_completion_label(
            "SOURCE_ONLY_READY_WITH_REPRESENTATION_WARNING",
            {"Q": 1},
            [],
        )
        == "Niepełne — wariant dopuszczony"
    )
    assert _format_step4_mz_completion_label("SOURCE_ONLY_READY", {}, []) == "Wystarczające"


def test_result_modal_copy_hides_manifest_language_for_source_only_variant():
    path = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z4_dataset_builder.py"
    )
    source = path.read_text(encoding="utf-8-sig")

    start = source.index("def _handle_step4_dataset_success_result")
    end = source.index("def _handle_step4_dataset_failure_result", start)
    segment = source[start:end]

    assert '"Pokrycie znaków"' in segment
    assert '"Syntetyczne uzupełnienie", "wyłączone (0 obrazów)"' in segment
    assert '"Reprezentacja MZ"' not in segment
    assert '"Miarka reprezentacji"' not in segment
    assert "Status zapisany w manifeście" not in segment
    assert 'title="Wariant treningowy gotowy"' in segment
    assert 'primary_text="Przejdź do treningu"' in segment
    assert "Niepełne pokrycie znaków jest ostrzeżeniem jakościowym i nie blokuje treningu." in segment
