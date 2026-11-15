from types import SimpleNamespace
import inspect

from auto_annotation_tool.gui import z2_overlay_dock_ui
from auto_annotation_tool.gui import z2_preview_editor
from auto_annotation_tool.gui import z2_run_io_runtime
from auto_annotation_tool.gui import z2_session_runtime


def _host(*, approved=False):
    approved_names = {"a.jpg"} if approved else set()
    ann = SimpleNamespace(filename="a.jpg", detections=[object()])
    host = SimpleNamespace(
        current_annotations=[ann],
        _preview_approved_filenames=approved_names,
        _campaign_pending_approved_filenames=set(approved_names),
        _preview_approval_version=0,
        _preview_counter_version=0,
        _current_preview_plate_count_cache=None,
    )
    host._get_plate_detections = lambda item: list(item.detections)
    host._get_preview_approved_filenames = lambda: set(approved_names)

    def is_approved(item, approved_names=None):
        lookup = (
            set(approved_names)
            if approved_names is not None
            else set(host._preview_approved_filenames)
        )
        return str(item.filename).lower() in {
            str(name).lower() for name in lookup
        }

    host._preview_annotation_is_explicitly_approved = is_approved
    return host, ann


def test_plate_count_cache_recomputes_after_counter_version_change():
    host, ann = _host(approved=False)

    first = z2_run_io_runtime._get_current_preview_plate_count_state(host)
    assert first["total_plates"] == 1
    assert first["approved_plates"] == 0

    ann.detections.append(object())
    host._preview_counter_version += 1

    second = z2_run_io_runtime._get_current_preview_plate_count_state(host)
    assert second["total_plates"] == 2
    assert second["approved_plates"] == 0
    assert second["counter_version"] == 1


def test_approved_plate_count_recomputes_after_geometry_version_change():
    host, ann = _host(approved=True)

    first = z2_run_io_runtime._get_current_preview_plate_count_state(host)
    assert first["approved_images"] == 1
    assert first["approved_plates"] == 1

    ann.detections.append(object())
    host._preview_counter_version += 1

    second = z2_run_io_runtime._get_current_preview_plate_count_state(host)
    assert second["approved_images"] == 1
    assert second["approved_plates"] == 2


def test_runtime_cache_invalidation_bumps_counter_version():
    host = SimpleNamespace(_preview_counter_version=7)

    z2_session_runtime._invalidate_preview_runtime_caches(host)

    assert host._preview_counter_version == 8
    assert host._current_preview_plate_count_cache is None


def test_new_polygon_fast_cache_path_updates_approved_plates():
    source = inspect.getsource(
        z2_preview_editor._commit_new_preview_polygon
    )

    assert "_preview_counter_version" in source
    assert 'count_cache["approved_plates"]' in source
    assert "_preview_annotation_is_explicitly_approved" in source


def test_gate_counter_copy_is_explicit_about_ok_and_gt():
    text = z2_overlay_dock_ui._format_preview_gate_have_text(
        {
            "approved_images": 1,
            "approved_plates": 2,
            "gt_plates": 2,
            "gt_total_plates": 2,
            "missing_gt": 0,
        },
        "plates",
    )

    assert text == "TABLICE [OK]\n2\nGT 2/2"


def test_gate_counter_without_approval_does_not_claim_raw_frames():
    text = z2_overlay_dock_ui._format_preview_gate_have_text(
        {
            "approved_images": 0,
            "approved_plates": 0,
            "gt_plates": 0,
            "gt_total_plates": 0,
            "missing_gt": 0,
        },
        "plates",
    )

    assert text == "TABLICE [OK]\n0"


def test_fullscreen_gate_cache_key_depends_on_counter_version_and_gt():
    source = inspect.getsource(
        z2_overlay_dock_ui.render_preview_overlay_dock
    )

    assert "_preview_counter_version" in source
    assert '"gt_plates"' in source
    assert '"gt_total_plates"' in source
    assert '"missing_gt"' in source
    assert '"gt_ready"' in source
