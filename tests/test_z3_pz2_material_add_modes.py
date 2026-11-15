
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import z3_pz2_material_append as material


ROOT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def test_add_material_modal_exposes_two_complementary_append_routes():
    source = _read("auto_annotation_tool/gui/z3_pz2_material_append.py")
    assert "Gotowe tablice + AZ" in source
    assert "Obrazy + AT" in source
    assert "_open_pz2_add_az_package(host, context=context)" in source
    assert "_open_pz2_add_images_at(host, context=context)" in source


def test_images_at_route_uses_existing_e1_importer_and_existing_z2_review(tmp_path):
    source = tmp_path / "donor"
    images = source / "images"
    images.mkdir(parents=True)
    xml_path = source / "annotations.xml"
    xml_path.write_text("<annotations/>", encoding="utf-8")
    (images / "one.jpg").write_bytes(b"x")

    imported_run = tmp_path / "project" / "_staging" / "imported_at"
    imported_run.mkdir(parents=True)
    (imported_run / "annotations.xml").write_text(
        "<annotations/>",
        encoding="utf-8",
    )

    campaign_tab = SimpleNamespace(
        _import_project_start_plate_run=Mock(return_value=True),
    )
    annotation_tab = SimpleNamespace(
        open_existing_run_for_campaign_review=Mock(return_value=True),
    )
    app = SimpleNamespace(
        tabs={
            "campaign": campaign_tab,
            "annotation": annotation_tab,
        },
        open_controlled_tab=Mock(),
        update_status=Mock(),
    )
    host = SimpleNamespace(
        app=app,
        frame=None,
        _flush_scheduled_preview_metadata_save=Mock(),
    )
    context = {
        "project_name": "Target",
        "project_root": tmp_path / "project",
        "preview_dir": tmp_path / "preview",
    }

    previous_master = tmp_path / "old_master"

    with (
        patch.object(
            material.filedialog,
            "askopenfilename",
            return_value=str(xml_path),
        ),
        patch.object(
            material.CAMPAIGN,
            "get_master_pool_dir",
            return_value=previous_master,
        ),
        patch.object(
            material.CAMPAIGN,
            "set_master_pool_dir",
            return_value=True,
        ) as set_pool,
        patch.object(
            material.CAMPAIGN,
            "get_project_start_plate_source",
            return_value={
                "source_run_path": str(imported_run),
                "source_xml_path": str(imported_run / "annotations.xml"),
                "source_input_path": str(images),
            },
        ),
        patch.object(
            material.CAMPAIGN,
            "get_active_project_name",
            return_value="Target",
        ),
    ):
        assert material._open_pz2_add_images_at(
            host,
            context=context,
        )

    campaign_tab._import_project_start_plate_run.assert_called_once()
    kwargs = campaign_tab._import_project_start_plate_run.call_args.kwargs
    assert kwargs["selected_xml_path"] == xml_path
    assert kwargs["refresh_dashboard_after_import"] is False
    assert kwargs["confirm_import"] is True

    assert set_pool.call_args_list[0].args[0] == images
    assert set_pool.call_args_list[-1].args[0] == previous_master

    annotation_tab.open_existing_run_for_campaign_review.assert_called_once_with(
        imported_run,
        iteration_target="char",
        manual_template=False,
        defer_ui_restore=False,
    )
    assert (
        annotation_tab._campaign_graph_entry_context["graph_gate_id"]
        == "T05"
    )
    assert (
        annotation_tab._campaign_graph_entry_context["z2_work_mode"]
        == "pz2_append_images_at_review"
    )
    app.open_controlled_tab.assert_called_once_with("annotation")


def test_images_at_route_restores_master_pool_when_import_fails(tmp_path):
    source = tmp_path / "donor"
    images = source / "images"
    images.mkdir(parents=True)
    xml_path = source / "annotations.xml"
    xml_path.write_text("<annotations/>", encoding="utf-8")
    (images / "one.jpg").write_bytes(b"x")

    campaign_tab = SimpleNamespace(
        _import_project_start_plate_run=Mock(return_value=False),
    )
    app = SimpleNamespace(
        tabs={"campaign": campaign_tab},
    )
    host = SimpleNamespace(
        app=app,
        frame=None,
        _flush_scheduled_preview_metadata_save=Mock(),
    )
    context = {
        "project_name": "Target",
        "project_root": tmp_path / "project",
        "preview_dir": tmp_path / "preview",
    }
    previous_master = tmp_path / "old_master"

    with (
        patch.object(
            material.filedialog,
            "askopenfilename",
            return_value=str(xml_path),
        ),
        patch.object(
            material.CAMPAIGN,
            "get_master_pool_dir",
            return_value=previous_master,
        ),
        patch.object(
            material.CAMPAIGN,
            "set_master_pool_dir",
            return_value=True,
        ) as set_pool,
    ):
        assert not material._open_pz2_add_images_at(
            host,
            context=context,
        )

    assert set_pool.call_args_list[0].args[0] == images
    assert set_pool.call_args_list[-1].args[0] == previous_master


def test_az_direct_append_contract_is_still_present():
    source = _read("auto_annotation_tool/gui/z3_pz2_material_append.py")
    assert "target_artifact_dir=images_dir" in source
    assert "materialize_az_append_to_preview(" in source
    assert "imported_pending_review" in _read(
        "auto_annotation_tool/registry/az_package_transport.py"
    )
