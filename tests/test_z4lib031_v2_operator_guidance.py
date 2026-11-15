"""Z4LIB031_V2: real guidance on purpose/action/consequence/next step.

Static regression, no Tk, no export, no models or Android runtime.
"""
import ast
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui'
GUI = R / 'z4_model_export.py'


def _code(file):
    value = (R / file).read_text(encoding='utf-8-sig')
    ast.parse(value)
    return value


def test_center_tells_user_what_will_happen():
    s = _code('z4_model_export.py')
    assert 'Wybrane modele treningowe' in s
    assert 'otworzy ich ustawienia — nie uruchamia eksportu' in s


def test_export_explains_source_and_why_paths_may_be_empty():
    s = _code('z4_model_export.py')
    for p in ('1. Źródło modelu MT', '1. Źródło modelu MZ',
              'Przygotuj nowy MT z wybranego best.pt', 'Wykorzystaj gotowy MZ z biblioteki',
              'Najpierw sprawdzimy ustawienia', 'Gotowość do eksportu · źródła, ustawienia i zależności'):
        assert p in s


def test_inventory_explains_missing_pair_without_empty_selectors():
    s = _code('z4_model_export.py')
    for p in ('Po co ta kontrola?', 'Brakuje co najmniej jednej paczki',
              'przygotuj brakujące modele', 'role_labels[role].grid_remove()',
              'validate_f4_pair_choice('):
        assert p in s


def test_report_explains_status_and_next_step():
    i = _code('z4_mobile_f4_inventory.py')
    c = _code('z4_mobile_f4_comparability.py')
    assert 'Co sprawdzamy: opisy paczek' in i
    assert 'Co dalej: przed pomiarami' in i
    assert 'Czy gotowe paczki FP32 i INT8' in c
    assert 'Następny krok: sprawdź' in c


def test_internal_f4_ids_and_guards_stay():
    s = _code('z4_model_export.py')
    for p in ('def show_f4_inventory()', 'f4_zero_conversion_only',
              'require_f4_no_conversion_request(bundle_request)',
              'require_f4_no_conversion_request(request, require_fresh_destination=True)',
              'export_f4_atomic_no_overwrite(', 'model_card_columns = 1  # One wide scrollable card per role tab.'):
        assert p in s
    assert 'view.title("F4' not in s
    assert '"F4: pakiety' not in s


def test_public_messages_no_f4_prefix():
    for name in ('z4_mobile_f4_pair_selection.py', 'z4_mobile_f4_no_conversion.py',
                 'z4_mobile_f4_atomic_publish.py'):
        s = _code(name)
        assert "'F4:" not in s
        assert "'F4 bez konwersji:" not in s
