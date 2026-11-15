import json
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_annotation_tool.gui.z3_detection_algorithms import get_yolo_runtime_settings
from auto_annotation_tool.gui.z3_detection_pipeline_ui import (
    compile_detection_pipeline_blocks,
    open_detection_pipeline_advanced_modal,
    refresh_detection_pipeline_builder_property_panel,
)
from auto_annotation_tool.gui.z3_threshold_percent import create_percent_spinbox, format_percent
from auto_annotation_tool.gui.z3_detection_controls_ui import on_yolo_option_var_write
from auto_annotation_tool.gui.z3_session_state import load_local_session, save_local_setting


@pytest.fixture
def root():
    window = tk.Tk()
    window.withdraw()
    yield window
    window.destroy()


def enter(spin, value):
    spin.delete(0, tk.END)
    spin.insert(0, value)
    spin.tk.call(spin.cget("command"))


@pytest.mark.parametrize("original,expected", [(.11, "11"), (.55, "55"), (1.2, "120"),
                                                (2.954225352112676, "295.423"), (.00001, "0.001")])
def test_existing_threshold_is_displayed_as_percent_without_changing_precision(root, original, expected):
    source = tk.DoubleVar(root, value=original)
    save = Mock()
    spin = create_percent_spinbox(root, variable=source, from_=0.00001, to=4.5, increment=.05, command=save)
    assert spin.get() == expected
    spin.tk.call(spin.cget("command"))
    assert source.get() == original
    save.assert_called_once_with()


def test_percent_input_and_external_presets_synchronize_all_editors(root):
    source = tk.DoubleVar(root, value=.11)
    first = create_percent_spinbox(root, variable=source, from_=.01, to=.99, increment=.05)
    second = create_percent_spinbox(root, variable=source, from_=.01, to=.99, increment=.05)
    assert float(first.cget("from")) == 1
    assert float(first.cget("to")) == 99
    assert float(first.cget("increment")) == 5
    enter(first, "11,5")
    assert source.get() == pytest.approx(.115)
    assert first.get() == second.get() == "11.5"
    enter(first, "12,5%")
    assert source.get() == pytest.approx(.125)
    assert first.get() == second.get() == "12.5"
    source.set(.45)
    assert first.get() == second.get() == "45"


@pytest.mark.parametrize("text,expected", [("0", .00001), ("1000", 1.), ("nan", .25),
                                           ("inf", .25), ("-", .25), ("", .25)])
def test_invalid_and_out_of_range_entry_never_corrupts_runtime_threshold(root, text, expected):
    source = tk.DoubleVar(root, value=.25)
    spin = create_percent_spinbox(root, variable=source, from_=.00001, to=1., increment=.05)
    enter(spin, text)
    assert source.get() == expected
    assert spin.get() == format_percent(expected)


def test_spinbox_arrow_increments_percentage_points_and_saves_normalized_value(root):
    source = tk.DoubleVar(root, value=.11)
    spin = create_percent_spinbox(root, variable=source, from_=.01, to=.99, increment=.05)
    spin.event_generate("<<Increment>>")
    assert source.get() == pytest.approx(.16)
    assert spin.get() == "16"


@pytest.mark.parametrize("value,lower,upper,text,expected", [(240, 100, 255, "100", 255),
                                                            (255, 100, 255, "50", 128),
                                                            (0, -20, 20, "-5", -13)])
def test_image_brightness_thresholds_round_back_to_integer_levels(root, value, lower, upper, text, expected):
    source = tk.IntVar(root, value=value)
    spin = create_percent_spinbox(root, variable=source, from_=lower, to=upper, increment=1, reference=255.)
    enter(spin, text)
    assert source.get() == expected
    assert spin.get() == format_percent(expected, 255.)


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def host(root, tmp_path, block="yolo_box"):
    values = {"yolo_box_conf_var": .00001, "yolo_symbol_conf_var": .25,
              "yolo_conf_var": .00001, "yolo_iou_var": .11, "yolo_overlap_var": .05,
              "yolo_seq_center_y_var": .2722397476340694, "yolo_seq_min_h_ratio_var": .55,
              "yolo_seq_max_h_ratio_var": 2.954225352112676, "yolo_seq_max_w_ratio_var": 1.199288256227758,
              "yolo_seq_soft_overlap_var": .1417624521072797, "yolo_seq_hard_overlap_var": .16766467065868262,
              "ocr_conf_var": .25, "ocr_min_height_ratio_var": .58, "prep_angle_var": 0., "prep_clahe_var": 2.}
    owner = SimpleNamespace(frame=root, app=SimpleNamespace(palette={}),
                            _detection_pipeline_property_body=ttk.Frame(root),
                            _detection_pipeline_state={"selected_index": 0})
    for name, value in values.items():
        setattr(owner, name, tk.DoubleVar(root, value=value))
    for name, value in {"prep_height_var": 64, "prep_padding_var": 10, "prep_clip_var": 240,
                        "prep_denoise_var": 3, "prep_block_var": 11, "prep_c_var": 2,
                        "prep_erode_var": 0, "hybrid_rescue_max_chars_var": 0}.items():
        setattr(owner, name, tk.IntVar(root, value=value))
    for name in ("yolo_agnostic_nms_var", "detect_protect_manual_var", "detect_protect_perfect_var",
                 "detect_refine_perfect_yolo_var", "detect_refiner_continuity_guard_var", "hybrid_yolo_box_backend_var",
                 "do_clahe_var", "prep_use_bin_var"):
        setattr(owner, name, tk.BooleanVar(root, value=False))
    owner.interpolation_var = tk.StringVar(root, value="linear")
    owner._get_detection_pipeline_builder_blocks = lambda: [block]
    owner._get_detection_pipeline_block_meta = lambda key: {"badge": key, "title": key, "desc": ""}
    owner._compile_detection_pipeline_blocks = compile_detection_pipeline_blocks
    owner._get_effective_yolo_model_path = lambda: ""
    owner._pick_detection_pipeline_yolo_model = Mock()
    owner._show_detection_pipeline_model_details = Mock()
    owner._get_hybrid_rescue_max_chars = lambda: 0
    owner._open_filter_lab = Mock()
    owner._open_ocr_ranking_modal = Mock()
    owner._refresh_detection_refiner_guard_label = Mock()
    owner._refresh_detection_pipeline_builder_property_panel = Mock()
    owner._save_local_setting = Mock()

    def save():
        snapshot = {name: getattr(owner, name).get() for name in values}
        (tmp_path / "settings.json").write_text(json.dumps(snapshot), encoding="utf-8")
    owner._on_yolo_option_var_write = save
    owner._force_save_all = save
    return owner, values


@pytest.mark.parametrize("block,number", [("yolo_box", 9), ("yolo_symbol", 9), ("ocr_symbol", 2)])
def test_real_builder_controls_display_percent_and_save_runtime_units(root, tmp_path, block, number):
    owner, values = host(root, tmp_path, block)
    refresh_detection_pipeline_builder_property_panel(owner)
    spins = [w for w in descendants(owner._detection_pipeline_property_body) if isinstance(w, ttk.Spinbox)]
    assert len(spins) == number
    assert all(hasattr(spin, "_percent_binding") for spin in spins)
    labels = [w.cget("text") for w in descendants(owner._detection_pipeline_property_body) if isinstance(w, ttk.Label)]
    assert sum("[%]" in label for label in labels) == number
    assert all(getattr(owner, name).get() == value for name, value in values.items())
    field = next(spin for spin in spins if spin._percent_binding.variable is (owner.ocr_conf_var if block == "ocr_symbol" else owner.yolo_iou_var))
    enter(field, "11,5")
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert saved["ocr_conf_var" if block == "ocr_symbol" else "yolo_iou_var"] == pytest.approx(.115)
    if block != "ocr_symbol":
        assert get_yolo_runtime_settings(owner)["iou"] == pytest.approx(.115)
    owner._detection_pipeline_property_body.destroy()
    assert all(not getattr(owner, name).trace_info() for name in values)


def test_real_session_writer_and_reload_keep_normalized_pipeline_values(root, tmp_path):
    owner, _values = host(root, tmp_path)
    owner.local_session = {}
    owner.session_file = tmp_path / "char_tab_session.json"
    owner._save_local_setting = lambda key, value: save_local_setting(owner, key, value)
    owner._use_hybrid_yolo_box_backend = lambda: True
    owner._get_detection_method_key = lambda: "YOLO_BOX"
    owner._apply_yolo_option_check_style = Mock()
    owner._on_yolo_option_var_write = lambda: on_yolo_option_var_write(owner)
    refresh_detection_pipeline_builder_property_panel(owner)
    spins = [w for w in descendants(owner._detection_pipeline_property_body) if isinstance(w, ttk.Spinbox)]
    iou = next(w for w in spins if w._percent_binding.variable is owner.yolo_iou_var)
    width = next(w for w in spins if w._percent_binding.variable is owner.yolo_seq_max_w_ratio_var)
    enter(iou, "11,5%")
    enter(width, "120")
    reloaded = load_local_session(owner)
    assert reloaded["char_yolo_iou"] == pytest.approx(.115)
    assert reloaded["char_yolo_seq_max_w_ratio"] == 1.2
    assert reloaded["char_yolo_box_conf"] == .00001
    assert reloaded["char_yolo_seq_center_y"] == .2722397476340694
    owner.yolo_iou_var.set(reloaded["char_yolo_iou"])
    assert iou.get() == "11.5"


@pytest.mark.parametrize("block,percent_count", [("yolo_box", 10), ("yolo_symbol", 10), ("ocr_symbol", 3)])
def test_real_advanced_dialog_uses_percent_without_rescaling_pixel_dimensions(root, tmp_path, block, percent_count):
    owner, values = host(root, tmp_path, block)
    open_detection_pipeline_advanced_modal(owner, block)
    win = owner._detection_pipeline_advanced_modal
    spins = [w for w in descendants(win) if isinstance(w, ttk.Spinbox)]
    percent_spins = [w for w in spins if hasattr(w, "_percent_binding")]
    assert len(percent_spins) == percent_count
    assert all(getattr(owner, name).get() == value for name, value in values.items())
    if block == "ocr_symbol":
        clip = next(w for w in percent_spins if w._percent_binding.variable is owner.prep_clip_var)
        enter(clip, "100")
        assert owner.prep_clip_var.get() == 255
        native_height = next(w for w in spins if str(w.cget("textvariable")) == str(owner.prep_height_var))
        assert native_height.get() == "64"
    win.destroy()
    assert all(not getattr(owner, name).trace_info() for name in values)
