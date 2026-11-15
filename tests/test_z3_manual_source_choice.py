from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import z3_extraction_tab_ui as extract_ui


class Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def test_normalize_manual_source_kind_accepts_only_two_routes():
    host = SimpleNamespace(extract_manual_source_kind_var=Value("crops"))
    assert extract_ui.normalize_manual_source_kind(host) == "crops"
    assert extract_ui.normalize_manual_source_kind(host, "xml") == "xml"
    assert extract_ui.normalize_manual_source_kind(host, "other") == ""


def test_select_crops_hides_xml_binding_semantics():
    host = SimpleNamespace(
        extract_manual_source_kind_var=Value(""),
        _extract_last_source_binding_result={"ok": True},
        _set_source_binding_status=Mock(),
        _refresh_extract_workflow_ui=Mock(),
        _refresh_source_binding_status=Mock(),
    )

    extract_ui.select_manual_source_kind(host, "crops")

    assert host.extract_manual_source_kind_var.get() == "crops"
    assert host._extract_last_source_binding_result == {"ok": False}
    host._set_source_binding_status.assert_called_once_with("", "warning")
    host._refresh_source_binding_status.assert_not_called()
    host._refresh_extract_workflow_ui.assert_called_once()


def test_select_xml_revalidates_only_xml_route():
    host = SimpleNamespace(
        extract_manual_source_kind_var=Value(""),
        _extract_last_source_binding_result={"ok": False},
        _set_source_binding_status=Mock(),
        _refresh_extract_workflow_ui=Mock(),
        _refresh_source_binding_status=Mock(return_value={"ok": False}),
    )

    extract_ui.select_manual_source_kind(host, "xml")

    assert host.extract_manual_source_kind_var.get() == "xml"
    host._refresh_source_binding_status.assert_called_once_with(
        allow_autofind=False
    )
    host._set_source_binding_status.assert_not_called()
    host._refresh_extract_workflow_ui.assert_called_once()
