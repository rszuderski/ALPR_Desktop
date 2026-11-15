from auto_annotation_tool.gui.z3_shared_ui import _parse_pz3_operation_summary


def test_pz3_export_footer_guides_user_through_local_cta_gate_and_z4():
    message = (
        "Proces zakończył się sukcesem: utworzono źródłowy dataset YOLO znaków.\n"
        "Wyeksportowane tablice: 10\n"
        "Wyeksportowane znaki: 70\n"
        "Dalej: 1. Kliknij „Przekaż dataset i wróć do grafu”.\n"
        "2. Na grafie w bramce T05 kliknij „Zatwierdź bramkę”.\n"
        "3. Po zatwierdzeniu T05 projekt przejdzie do E4Z; wybierz „Otwórz Z4”, "
        "aby przygotować wariant treningowy i split."
    )

    _lead, _rows, footer = _parse_pz3_operation_summary(message)

    assert footer.splitlines() == [
        "1. Kliknij „Przekaż dataset i wróć do grafu”.",
        "2. Na grafie w bramce T05 kliknij „Zatwierdź bramkę”.",
        "3. Po zatwierdzeniu T05 projekt przejdzie do E4Z; wybierz „Otwórz Z4”, aby przygotować wariant treningowy i split.",
    ]


def test_export_copy_does_not_jump_directly_from_pz3_to_z4():
    from pathlib import Path
    from auto_annotation_tool.gui import z3_goldpack_ui

    source = Path(z3_goldpack_ui.__file__).read_text(encoding="utf-8-sig")

    assert "Przekaż dataset i wróć do grafu" in source
    assert "Zatwierdź bramkę" in source
    assert "Otwórz Z4" in source
    assert "Dalej: przejdź do Z4" not in source
