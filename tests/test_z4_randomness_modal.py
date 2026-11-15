from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import tkinter as tk
import time

import pytest

from auto_annotation_tool.gui.z4_augmentation_modal import Step4AugmentationModal
from auto_annotation_tool.gui.z4_randomness_modal import RandomnessModal
from auto_annotation_tool.training.dataset_augmentation import AugmentationProfile, default_manual_randomness_config
from auto_annotation_tool.training.augmentation_ranges import SPEC_BY_KEY, RANGE_SCHEMA


@pytest.fixture
def host():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    p = AugmentationProfile(task_target="plate", rain_strength=.6, rain_edge_mist_strength=.5,
                            night_light_strength=0, dark_relief_strength=0, tyndall_strength=0).normalized()
    owner = SimpleNamespace(window=root, _current_sample=None, _manual_randomness=deepcopy(p.manual_randomness),
                            randomness_mode_var=tk.StringVar(root, "Realistyczny"),
                            _refresh_preview=Mock(), _randomness_mode_to_label=lambda mode: "Ręczny")
    owner._profile_from_vars = lambda: replace(p, manual_randomness=owner._manual_randomness)
    yield owner
    dialog = getattr(owner, "_manual_randomness_dialog", None)
    if dialog and not dialog.closed:
        dialog.close()
    root.destroy()


def test_single_instance_and_cancel_leaves_owner_unchanged(host):
    dialog = RandomnessModal(host, show=False)
    original = deepcopy(host._manual_randomness)
    dialog._use_range(SPEC_BY_KEY["rain_strength"])
    assert dialog.draft.changed
    dialog.focus = Mock()
    assert Step4AugmentationModal._open_manual_randomness_modal(host) is dialog
    dialog.focus.assert_called_once()
    dialog.close()
    assert host._manual_randomness == original
    assert host.randomness_mode_var.get() == "Realistyczny"
    host._refresh_preview.assert_not_called()


def test_toggle_zero_can_be_reenabled_and_apply_persists(host):
    dialog = RandomnessModal(host, show=False)
    spec = SPEC_BY_KEY["rain_strength"]
    row = dialog.rows[spec.key]
    assert row["entry"].instate(["disabled"])
    row["toggle"].invoke()
    assert row["entry"].instate(["!disabled"])
    row["text"].set("0,10")
    assert dialog._commit_row(spec)
    assert "0,50" in row["summary"].cget("text") and "0,70" in row["summary"].cget("text")
    dialog._slide(spec, "0")
    assert row["entry"].instate(["disabled"])
    row["toggle"].invoke()
    assert row["entry"].instate(["!disabled"])
    row["text"].set("0,12")
    dialog.apply()
    assert host._manual_randomness["schema"] == RANGE_SCHEMA
    assert host._manual_randomness["overrides"]["rain_strength"]["delta"] == .12
    host._refresh_preview.assert_called_once_with(redraw_only=False)
    again = RandomnessModal(host, show=False)
    assert again.draft.state(spec) == ("range", .12)


def test_invalid_entry_blocks_apply_and_navigation(host):
    dialog = RandomnessModal(host, show=False)
    spec = SPEC_BY_KEY["rain_strength"]
    dialog._use_range(spec)
    dialog.rows[spec.key]["text"].set("abc")
    dialog.apply()
    assert not dialog.closed
    assert "Gęstość" in dialog.status.cget("text")
    dialog.nav.selection_clear(0, "end")
    dialog.nav.selection_set(dialog.groups.index("wind"))
    dialog._select_group()
    assert dialog.active_group == "weather"
    dialog.rows[spec.key]["text"].set("0,15")
    dialog.apply()
    assert dialog.closed


def test_inactive_effect_and_mt_categories(host):
    dialog = RandomnessModal(host, show=False)
    assert "wind" in dialog.groups and "material" not in dialog.groups and "dirt" not in dialog.groups
    dialog.advanced.set(True)
    dialog._render_group()
    row = dialog.rows["rain_lens_strength"]
    assert row["toggle"].instate(["disabled"])
    assert "wyłączony" in row["summary"].cget("text")
    assert dialog.window.bind("<Escape>")
    assert dialog.window.bind("<MouseWheel>")


def test_legacy_row_requires_explicit_range_change(host):
    config = default_manual_randomness_config("plate")
    config["groups"]["weather"].update(enabled=True, amount=50)
    config["groups"]["weather"]["fields"]["rain_strength"].update(enabled=True, amount=50)
    host._manual_randomness = config
    dialog = RandomnessModal(host, show=False)
    row = dialog.rows["rain_strength"]
    assert "25%" in row["summary"].cget("text")
    assert row["entry"].instate(["disabled"])
    dialog.apply()
    assert host._manual_randomness == config
    again = RandomnessModal(host, show=False)
    again.rows["rain_strength"]["legacy_button"].invoke()
    assert again.rows["rain_strength"]["entry"].instate(["!disabled"])


def test_pending_preview_is_cancelled_on_close_and_reroll_changes_only_seeds(host):
    dialog = RandomnessModal(host, show=False)
    before = dialog.draft.export()
    seeds = dialog.preview_seeds
    dialog.reroll()
    assert dialog.preview_seeds != seeds and dialog.draft.export() == before
    dialog.close()
    assert dialog._preview_after is None and dialog._poll_after is None
    assert dialog._cancel_event.is_set()
    host.window.update()


def test_stale_worker_result_cannot_replace_current_preview(host):
    dialog = RandomnessModal(host, show=False)
    dialog._display_previews = Mock()
    dialog._busy = True
    dialog._results.put((dialog._revision-1, [{}, {}, {}, {}], ""))
    dialog._poll_preview()
    assert not dialog._busy
    dialog._display_previews.assert_not_called()


def test_new_edit_discards_queued_preview_before_debounce(host):
    dialog = RandomnessModal(host, show=False)
    dialog._pending = (dialog._revision, None, None, ())
    dialog._schedule_preview()
    assert dialog._pending is None
    assert dialog._cancel_event.is_set()


def test_group_bulk_actions_and_mz_geometry(host):
    base = host._profile_from_vars()
    host._profile_from_vars = lambda: replace(base, task_target="char", dirt_flow_points=10, dark_relief_strength=.3, night_light_strength=.5)
    dialog = RandomnessModal(host, show=False)
    assert "geometry" not in dialog.groups and "material" in dialog.groups
    dialog.active_group = "illumination"
    dialog._render_group()
    dialog._set_group(True)
    assert dialog.draft.overrides["relief_bounce_depth"]["enabled"]
    dialog._set_group(False)
    assert not dialog.draft.overrides["relief_bounce_depth"]["enabled"]


def test_real_tk_layout_scroll_and_grab_recovery(host):
    from dataclasses import asdict
    import numpy as np
    host.window.deiconify()
    host.window.update()
    host.window.grab_set()
    dialog = RandomnessModal(host)
    payload = {"augmented_rgb": np.full((120, 240, 3), 100, dtype=np.uint8), "sample_profile": asdict(dialog.base)}
    dialog._display_previews([payload] * 4)
    for width, height in ((1040, 688), (800, 600)):
        dialog.window.geometry(f"{width}x{height}+20+20")
        dialog.window.update()
        assert dialog.window.grab_current() is dialog.window
        assert dialog.canvas.winfo_width() > 300
        assert dialog.canvas.winfo_height() > 100
        for widget in (dialog.apply_button, dialog.preview_labels[-1]):
            right = widget.winfo_rootx() - dialog.window.winfo_rootx() + widget.winfo_width()
            bottom = widget.winfo_rooty() - dialog.window.winfo_rooty() + widget.winfo_height()
            assert right <= dialog.window.winfo_width()
            assert bottom <= dialog.window.winfo_height()
    dialog.advanced.set(True)
    dialog._render_group()
    dialog.window.update()
    before = dialog.canvas.yview()
    dialog.canvas.event_generate("<MouseWheel>", delta=-120)
    dialog.window.update()
    assert dialog.canvas.yview()[0] > before[0]
    dialog.close()
    assert host.window.grab_current() is host.window


def test_background_renderer_updates_tk_thumbnails(host, tmp_path, monkeypatch):
    import numpy as np
    from auto_annotation_tool.training import dataset_augmentation as aug
    if not aug.is_albumentations_available():
        pytest.skip("Albumentations unavailable")
    monkeypatch.setattr(aug, "is_cuda_available", lambda: False)
    path = tmp_path / "sample.png"
    image = np.full((48, 96, 3), 80, dtype=np.uint8)
    image[24:] = 200
    aug.cv2.imwrite(str(path), image)
    host._current_sample = path
    dialog = RandomnessModal(host, show=False)
    dialog._use_range(SPEC_BY_KEY["rain_strength"])
    deadline = time.monotonic()+15
    while not dialog._preview_profiles and time.monotonic() < deadline:
        host.window.update()
        time.sleep(.01)
    assert len(dialog._preview_profiles) == 4, dialog.preview_status.cget("text")
    assert len(dialog._images) == 4
    assert all(label.cget("image") for label in dialog.preview_labels)
    assert len({p["rain_strength"] for p in dialog._preview_profiles[1:]}) == 3


def test_large_preview_comparison_expand_and_restore_preserve_draft(host):
    from dataclasses import asdict
    import numpy as np
    dialog = RandomnessModal(host)
    dialog.window.geometry("1040x740+20+20")
    dialog.window.update()
    payload = {"augmented_rgb": np.full((720, 1280, 3), 100, dtype=np.uint8), "sample_profile": asdict(dialog.base)}
    dialog._display_previews([payload] * 4)
    dialog.window.update()
    dialog._redraw_preview_images()
    original = dialog.draft.export()
    seeds = dialog.preview_seeds
    assert dialog.preview_view.get() == "Próbka 1"
    assert dialog._images[1].height() > 180
    assert dialog._images[1].width() > 320
    dialog.preview_view.set("Porównanie 2×2")
    dialog._layout_preview_cells()
    dialog.window.update()
    assert all(cell.winfo_ismapped() for cell in dialog.preview_cells)
    assert dialog.preview_cells[2].grid_info()["row"] == 1
    dialog._focus_preview(2)
    dialog.window.update()
    dialog._redraw_preview_images()
    assert dialog._preview_expanded
    assert not dialog.settings_frame.winfo_ismapped()
    assert dialog._images[2].height() > 300
    assert dialog.apply_button.winfo_ismapped()
    dialog.window.geometry("800x600+20+20")
    dialog.window.update()
    dialog._redraw_preview_images()
    assert dialog._images[2].width() <= dialog.preview_labels[2].winfo_width()
    assert dialog._images[2].height() <= dialog.preview_labels[2].winfo_height()
    dialog._on_escape()
    dialog.window.update()
    assert not dialog.closed and dialog.settings_frame.winfo_ismapped()
    assert dialog.canvas.winfo_width() > 300 and dialog.canvas.winfo_height() > 100
    assert dialog.draft.export() == original and dialog.preview_seeds == seeds
