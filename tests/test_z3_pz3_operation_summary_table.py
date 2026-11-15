from auto_annotation_tool.gui.z3_shared_ui import (
    _parse_pz3_operation_summary,
    _pz3_summary_row_tone,
)


def test_parse_pz3_export_summary_into_table_and_footer():
    message = (
        "Proces zakończył się sukcesem: utworzono źródłowy dataset YOLO znaków.\n"
        "Wyeksportowane tablice: 10\n"
        "Wyeksportowane znaki: 70\n"
        "Układ tablic: 1R*: 10\n"
        "Strategie: OCR exact, Manual / inne perfect\n"
        "Źródła: Auto z runu, Ręczne poprawki lokalne\n"
        "Split: źródło bez splitu (images/labels)\n"
        "Ścieżka: C:\\Example\\dataset\n"
        "Dalej: przejdź do Z4, aby przygotować wariant treningowy i split."
    )

    lead, rows, footer = _parse_pz3_operation_summary(message)

    assert lead.startswith("Proces zakończył się sukcesem")
    assert ("Wyeksportowane tablice", "10") in rows
    assert ("Wyeksportowane znaki", "70") in rows
    assert ("Układ tablic", "1R*: 10") in rows
    assert ("Ścieżka", r"C:\Example\dataset") in rows
    assert footer == "przejdź do Z4, aby przygotować wariant treningowy i split."


def test_pz3_summary_row_tones_are_semantic():
    assert _pz3_summary_row_tone("Wyeksportowane tablice") == "success"
    assert _pz3_summary_row_tone("Wyeksportowane znaki") == "success"
    assert _pz3_summary_row_tone("Strategie") == "accent"
    assert _pz3_summary_row_tone("Źródła") == "accent"
    assert _pz3_summary_row_tone("Split") == "accent"
    assert _pz3_summary_row_tone("Ścieżka") == "muted"
