"""Z4LIB030: F4 inventory has no empty package selectors when a pair is missing.

No Tk display is used: we execute only the scoped UI callback with fake widgets.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "auto_annotation_tool/gui/z4_model_export.py"


class FakeVar:
    def __init__(self, value=""):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


class FakeWidget:
    def __init__(self):
        self.visible = True
        self.state = None
        self.values = None
    def grid(self, **kwargs):
        self.visible = True
    def grid_remove(self):
        self.visible = False
    def configure(self, **kwargs):
        self.state = kwargs.get('state', self.state)
        self.values = kwargs.get('values', self.values)


def callback(report, fmt="FP32"):
    tree = ast.parse(GUI.read_text(encoding="utf-8-sig"))
    definitions = [node for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == "_populate_pair_choices"]
    assert len(definitions) == 1
    func = definitions[0]
    env = {
        "inventory_context": {"report": report},
        "fmt_var": FakeVar(fmt),
        "choice_var": {role: FakeVar() for role in ("MT", "MZ")},
        "dropdowns": {role: FakeWidget() for role in ("MT", "MZ")},
        "role_labels": {role: FakeWidget() for role in ("MT", "MZ")},
        "options_by_role": {role: {} for role in ("MT", "MZ")},
        "selection_notice": FakeVar(),
        "choose_button": FakeWidget(),
        "Path": Path,
        "_sync_pair_action": lambda: None,
    }
    module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
    exec(compile(module, str(GUI), "exec"), env, env)
    env["_populate_pair_choices"]()
    return env


def test_fp32_missing_models_hides_disabled_empty_fields():
    report = {"pairs": {"fp32": {"state": "BLOCKED_NO_MATCHING_PACKAGE",
                               "candidates": {"MT": [], "MZ": []}}}}
    ui = callback(report)
    assert all(not ui["dropdowns"][r].visible for r in ("MT", "MZ"))
    assert all(not ui["role_labels"][r].visible for r in ("MT", "MZ"))
    assert not ui["choose_button"].visible
    assert "FP32" in ui["selection_notice"].get()
    assert "przygotuj brakujące modele" in ui["selection_notice"].get()


def test_int8_partial_inventory_shows_count_not_empty_selectors():
    report = {"pairs": {"int8": {"state": "BLOCKED_NO_MATCHING_PACKAGE",
                               "candidates": {"MT": [{"path": "C:/mt.alprmodel"}], "MZ": []}}}}
    ui = callback(report, "INT8")
    assert "MT — 1 kandydat(ów)" in ui["selection_notice"].get()
    assert "MZ — brak zgodnej paczki" in ui["selection_notice"].get()
    assert not ui["dropdowns"]["MT"].visible
    assert not ui["dropdowns"]["MZ"].visible
    assert not ui["choose_button"].visible


def test_complete_pair_restores_both_selectors_and_pair_button():
    report = {"pairs": {"fp32": {"state": "MANIFEST_CANDIDATES_REQUIRE_FULL_VALIDATION",
                               "candidates": {"MT": [{"path": "C:/mt.alprmodel"}],
                                              "MZ": [{"path": "C:/mz.alprmodel"}]}}}}
    ui = callback(report)
    assert all(ui["dropdowns"][r].visible for r in ("MT", "MZ"))
    assert all(ui["role_labels"][r].visible for r in ("MT", "MZ"))
    assert ui["choose_button"].visible
    assert ui["dropdowns"]["MT"].state == "readonly"
    assert len(ui["options_by_role"]["MT"]) == 1
    assert len(ui["options_by_role"]["MZ"]) == 1
    assert "walidacja grafu i SHA" in ui["selection_notice"].get()


def test_truncated_scan_is_not_treated_as_available():
    report = {"scan_truncated": True, "pairs": {"fp32": {
        "state": "MANIFEST_CANDIDATES_REQUIRE_FULL_VALIDATION",
        "candidates": {"MT": [{"path": "C:/mt.alprmodel"}],
                       "MZ": [{"path": "C:/mz.alprmodel"}]}}}}
    ui = callback(report)
    assert not ui["choose_button"].visible
    assert all(not ui["dropdowns"][r].visible for r in ("MT", "MZ"))
    assert "Skan biblioteki niekompletny" in ui["selection_notice"].get()


def test_approval_and_export_guards_remain_unmodified():
    text = GUI.read_text(encoding="utf-8-sig")
    assert "validate_f4_pair_choice(" in text
    assert "require_f4_no_conversion_request(bundle_request)" in text
    assert "require_f4_no_conversion_request(request, require_fresh_destination=True)" in text
    assert "export_f4_atomic_no_overwrite(" in text
    assert "model_card_columns = 1  # One wide scrollable card per role tab." in text
