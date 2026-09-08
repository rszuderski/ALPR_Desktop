#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modal configuration for train-only dataset augmentation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import random
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from ..config import CONFIG, PIL_AVAILABLE, cv2, np
from ..training import (
    AugmentationProfile,
    DEFAULT_BASE_LIGHT_STRENGTH,
    DEFAULT_RELIEF_PROFILE_CURVE,
    DEFAULT_RELIEF_PROFILE_PRESET,
    build_contour_profile_wireframe_segments,
    default_manual_randomness_config,
    get_augmentation_presets_dir,
    get_albumentations_status,
    load_augmentation_preset,
    manual_randomness_group_specs,
    normalize_augmentation_randomness_mode,
    normalize_relief_profile_curve,
    normalize_relief_profile_preset,
    normalize_manual_randomness_config,
    preview_augmentation_image,
    relief_profile_curve_for_preset,
    relief_profile_preset_label,
    relief_profile_presets,
    save_augmentation_preset,
)

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageTk

try:
    import psutil
except Exception:
    psutil = None


class Step4AugmentationModal:
    """Single modal with settings and side-by-side visual preview."""

    def __init__(
        self,
        master,
        *,
        target: str,
        profile: AugmentationProfile,
        sample_images: list[Path] | None = None,
        sample_pool_limit: int | None = None,
        install_callback=None,
    ):
        self.master = master
        self.target = CONFIG.normalize_task_target(target)
        self.initial_profile = (profile or AugmentationProfile()).normalized()
        self.sample_images = []
        seen_sample_paths: set[str] = set()
        for path in list(sample_images or []):
            candidate = Path(path)
            if not candidate.exists():
                continue
            try:
                key = str(candidate.resolve())
            except Exception:
                key = str(candidate)
            if key in seen_sample_paths:
                continue
            seen_sample_paths.add(key)
            self.sample_images.append(candidate)
        try:
            self.sample_pool_limit = max(1, int(sample_pool_limit or 0))
        except Exception:
            self.sample_pool_limit = 0
        if self.sample_pool_limit <= 0:
            self.sample_pool_limit = max(1, len(self.sample_images)) if self.sample_images else 10000
        self.install_callback = install_callback
        self.result: AugmentationProfile | None = None
        self._photo_refs: list = []
        self._current_sample: Path | None = self.sample_images[0] if self.sample_images else None
        try:
            ui_seed = random.SystemRandom().randint(1, 2_147_483_647)
        except Exception:
            ui_seed = id(self) % 2_147_483_647 or 1
        self._ui_random = random.Random(ui_seed)
        self._sample_shuffle_bag: list[Path] = []
        self._sample_shuffle_scope_key = ""
        self._process_handle = None
        self._memory_snapshot: dict = {}

        self.window = tk.Toplevel(master)
        self.window.title("Syntetyczne zwiększanie datasetu")
        self._configure_modal_window(master)
        self.window.geometry("980x720")
        self.window.minsize(860, 620)
        self.window.protocol("WM_DELETE_WINDOW", self._cancel)
        self.window.bind("<Escape>", self._on_escape_key, add="+")
        self.window.bind("<Tab>", self._on_tab_key, add="+")
        self.window.bind("<Return>", self._on_enter_key, add="+")
        self.window.bind("<KP_Enter>", self._on_enter_key, add="+")
        self.window.bind("<space>", self._on_space_key, add="+")
        self.window.bind("<Key-space>", self._on_space_key, add="+")
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.window.bind(sequence, self._on_preview_mousewheel, add="+")
        for sequence in (
            "<Left>",
            "<Right>",
            "<Up>",
            "<Down>",
            "<Shift-Left>",
            "<Shift-Right>",
            "<Shift-Up>",
            "<Shift-Down>",
            "<Control-Left>",
            "<Control-Right>",
            "<Control-Up>",
            "<Control-Down>",
            "<Control-Shift-Left>",
            "<Control-Shift-Right>",
            "<Control-Shift-Up>",
            "<Control-Shift-Down>",
            "<Prior>",
            "<Next>",
            "<Shift-Prior>",
            "<Shift-Next>",
            "<Control-Prior>",
            "<Control-Next>",
            "<Control-Shift-Prior>",
            "<Control-Shift-Next>",
        ):
            self.window.bind(sequence, self._on_scene_keyboard_nudge, add="+")

        initial_sample = max(1, int(self.initial_profile.sample_size or 32))
        initial_sample = min(initial_sample, self.sample_pool_limit)
        self.sample_var = tk.IntVar(value=initial_sample)
        self.extra_var = tk.IntVar(value=int(self.initial_profile.extra_count or 0))
        self._last_valid_sample_size = int(initial_sample)
        self._last_valid_extra_count = max(0, int(self.initial_profile.extra_count or 0))
        self.extra_count_title_var = tk.StringVar()
        self.randomness_mode_var = tk.StringVar(
            value=self._randomness_mode_to_label(getattr(self.initial_profile, "randomness_mode", "realistic"))
        )
        self.scene_summary_var = tk.StringVar(value="")
        self.rotation_var = tk.DoubleVar(value=float(self.initial_profile.rotation_limit or 0.0))
        self.brightness_var = tk.DoubleVar(value=float(self.initial_profile.brightness_limit or 0.0))
        self.contrast_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "contrast_limit", 0.0) or 0.0))
        initial_saturation = getattr(self.initial_profile, "saturation_limit", 1.0)
        if initial_saturation is None:
            initial_saturation = 1.0
        self.saturation_var = tk.DoubleVar(value=float(initial_saturation))
        self.noise_var = tk.DoubleVar(value=float(self.initial_profile.noise_strength or 0.0))
        self.noise_grain_var = tk.IntVar(value=int(self.initial_profile.noise_grain_size or 1))
        self.rain_var = tk.DoubleVar(value=float(self.initial_profile.rain_strength or 0.0))
        initial_drop_size = float(getattr(self.initial_profile, "rain_drop_size", 0.07) or 0.07)
        self.rain_drop_size_var = tk.DoubleVar(value=initial_drop_size)
        self.rain_drop_size_min_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "rain_drop_size_min", initial_drop_size) if getattr(self.initial_profile, "rain_drop_size_min", None) is not None else initial_drop_size))
        self.rain_drop_size_max_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "rain_drop_size_max", initial_drop_size) if getattr(self.initial_profile, "rain_drop_size_max", None) is not None else initial_drop_size))
        self.rain_vector_field_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "rain_vector_field_strength", 0.0) or 0.0))
        self.rain_vortex_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "rain_vortex_strength", 0.0) or 0.0))
        self.rain_alpha_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "rain_alpha", 0.22) if getattr(self.initial_profile, "rain_alpha", None) is not None else 0.22))
        self.rain_lens_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "rain_lens_strength", 0.0) or 0.0))
        self.rain_edge_mist_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "rain_edge_mist_strength", 0.0) or 0.0))
        self.rain_edge_mist_radius_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "rain_edge_mist_radius", 0.45) if getattr(self.initial_profile, "rain_edge_mist_radius", None) is not None else 0.45))
        self.rain_edge_debug_var = tk.BooleanVar(value=False)
        self.tyndall_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "tyndall_strength", 0.55) if getattr(self.initial_profile, "tyndall_strength", None) is not None else 0.55))
        self.wet_reflection_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "wet_reflection_strength", 0.0) or 0.0))
        self.vehicle_speed_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "vehicle_speed", 0.0) or 0.0))
        self.night_var = tk.DoubleVar(value=float(self.initial_profile.night_strength or 0.0))
        self.night_luma_min_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "night_luma_min", 0.56) if getattr(self.initial_profile, "night_luma_min", None) is not None else 0.56))
        self.night_luma_max_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "night_luma_max", 0.92) if getattr(self.initial_profile, "night_luma_max", None) is not None else 0.92))
        initial_base_light = float(
            getattr(
                self.initial_profile,
                "night_light_strength",
                DEFAULT_BASE_LIGHT_STRENGTH,
            )
            if getattr(self.initial_profile, "night_light_strength", None) is not None
            else DEFAULT_BASE_LIGHT_STRENGTH
        )
        if initial_base_light <= 0.001:
            initial_base_light = DEFAULT_BASE_LIGHT_STRENGTH
        self.night_light_var = tk.DoubleVar(value=initial_base_light)
        self.night_bloom_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "night_bloom_strength", 0.0) or 0.0))
        self.night_iso_noise_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "night_iso_noise_strength", 0.0) or 0.0))
        self.night_warmth_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "night_light_warmth", 0.35) if getattr(self.initial_profile, "night_light_warmth", None) is not None else 0.35))

        def _initial_float(name: str, default: float) -> float:
            try:
                value = getattr(self.initial_profile, name, default)
                if value is None:
                    value = default
                value = float(value)
                return value if math.isfinite(value) else float(default)
            except Exception:
                return float(default)

        self.scene_plate_width_var = tk.DoubleVar(value=1.0)
        self.scene_plate_height_var = tk.DoubleVar(value=0.24)
        self.scene_camera_x_var = tk.DoubleVar(value=0.0)
        self.scene_camera_y_var = tk.DoubleVar(value=0.0)
        self.scene_camera_z_var = tk.DoubleVar(value=_initial_float("scene_camera_z", 1.65))
        self.scene_camera_target_x_var = tk.DoubleVar(value=0.0)
        self.scene_camera_target_y_var = tk.DoubleVar(value=0.0)
        self.scene_camera_target_z_var = tk.DoubleVar(value=0.0)
        self.scene_view_yaw_var = tk.DoubleVar(value=0.0)
        self.scene_view_pitch_var = tk.DoubleVar(value=89.0)
        self.scene_view_roll_var = tk.DoubleVar(value=0.0)
        self.scene_plate_texture_var = tk.BooleanVar(value=bool(getattr(self.initial_profile, "scene_plate_texture_enabled", True)))

        def _default_plate_world(norm_x: float, norm_y: float, scale: float = 1.0) -> tuple[float, float]:
            plate_w = max(0.25, min(3.0, float(self.scene_plate_width_var.get() or 1.0)))
            plate_h = max(0.08, min(1.2, float(self.scene_plate_height_var.get() or 0.24)))
            return (float(norm_x) - 0.5) * plate_w * scale, (0.5 - float(norm_y)) * plate_h * scale

        def _world_or_default(name: str, default: float) -> float:
            value = _initial_float(name, -999.0)
            return float(default) if value <= -998.0 else value

        def _profile_norm_or_default(name: str, default: float) -> float:
            value = _initial_float(name, -1.0)
            return value if 0.0 <= value <= 1.0 else float(default)

        self.traffic_headlight_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_strength", 0.0) or 0.0))
        self.traffic_headlight_count_var = tk.IntVar(value=int(float(getattr(self.initial_profile, "traffic_headlight_count", 1) if getattr(self.initial_profile, "traffic_headlight_count", None) is not None else 1)))
        self.traffic_headlight_1_warmth_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_1_warmth", self.night_warmth_var.get()) if getattr(self.initial_profile, "traffic_headlight_1_warmth", None) is not None and float(getattr(self.initial_profile, "traffic_headlight_1_warmth", -1.0)) >= 0 else self.night_warmth_var.get()))
        self.traffic_headlight_1_r_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_1_r", 1.0) if getattr(self.initial_profile, "traffic_headlight_1_r", None) is not None and float(getattr(self.initial_profile, "traffic_headlight_1_r", -1.0)) >= 0 else 1.0))
        self.traffic_headlight_1_g_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_1_g", 0.88) if getattr(self.initial_profile, "traffic_headlight_1_g", None) is not None and float(getattr(self.initial_profile, "traffic_headlight_1_g", -1.0)) >= 0 else 0.88))
        self.traffic_headlight_1_b_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_1_b", 0.54) if getattr(self.initial_profile, "traffic_headlight_1_b", None) is not None and float(getattr(self.initial_profile, "traffic_headlight_1_b", -1.0)) >= 0 else 0.54))
        self.traffic_headlight_1_cone_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_1_cone", 0.45) if getattr(self.initial_profile, "traffic_headlight_1_cone", None) is not None else 0.45))
        self.traffic_headlight_1_source_radius_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_1_source_radius", 0.08) if getattr(self.initial_profile, "traffic_headlight_1_source_radius", None) is not None else 0.08))
        self.traffic_headlight_source_x_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_source_x", -1.0) if getattr(self.initial_profile, "traffic_headlight_source_x", None) is not None else -1.0))
        self.traffic_headlight_source_y_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_source_y", -1.0) if getattr(self.initial_profile, "traffic_headlight_source_y", None) is not None else -1.0))
        self.traffic_headlight_target_x_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_target_x", -1.0) if getattr(self.initial_profile, "traffic_headlight_target_x", None) is not None else -1.0))
        self.traffic_headlight_target_y_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_target_y", -1.0) if getattr(self.initial_profile, "traffic_headlight_target_y", None) is not None else -1.0))
        r1_source_world = _default_plate_world(
            _profile_norm_or_default("traffic_headlight_source_x", 0.14),
            _profile_norm_or_default("traffic_headlight_source_y", 0.91),
            1.55,
        )
        r1_target_world = _default_plate_world(
            _profile_norm_or_default("traffic_headlight_target_x", 0.42),
            _profile_norm_or_default("traffic_headlight_target_y", 0.44),
            1.0,
        )
        self.traffic_headlight_source_world_x_var = tk.DoubleVar(value=_world_or_default("traffic_headlight_source_world_x", r1_source_world[0]))
        self.traffic_headlight_source_world_y_var = tk.DoubleVar(value=_world_or_default("traffic_headlight_source_world_y", r1_source_world[1]))
        self.traffic_headlight_source_world_z_var = tk.DoubleVar(value=_initial_float("traffic_headlight_source_world_z", 0.95))
        self.traffic_headlight_target_world_x_var = tk.DoubleVar(value=_world_or_default("traffic_headlight_target_world_x", r1_target_world[0]))
        self.traffic_headlight_target_world_y_var = tk.DoubleVar(value=_world_or_default("traffic_headlight_target_world_y", r1_target_world[1]))
        self.traffic_headlight_target_world_z_var = tk.DoubleVar(value=0.0)
        self.traffic_headlight_2_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_2_strength", 0.0) or 0.0))
        self.traffic_headlight_2_warmth_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_2_warmth", 0.35) if getattr(self.initial_profile, "traffic_headlight_2_warmth", None) is not None else 0.35))
        self.traffic_headlight_2_r_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_2_r", 1.0) if getattr(self.initial_profile, "traffic_headlight_2_r", None) is not None and float(getattr(self.initial_profile, "traffic_headlight_2_r", -1.0)) >= 0 else 1.0))
        self.traffic_headlight_2_g_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_2_g", 0.88) if getattr(self.initial_profile, "traffic_headlight_2_g", None) is not None and float(getattr(self.initial_profile, "traffic_headlight_2_g", -1.0)) >= 0 else 0.88))
        self.traffic_headlight_2_b_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_2_b", 0.54) if getattr(self.initial_profile, "traffic_headlight_2_b", None) is not None and float(getattr(self.initial_profile, "traffic_headlight_2_b", -1.0)) >= 0 else 0.54))
        self.traffic_headlight_2_cone_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_2_cone", 0.45) if getattr(self.initial_profile, "traffic_headlight_2_cone", None) is not None else 0.45))
        self.traffic_headlight_2_source_radius_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_2_source_radius", 0.08) if getattr(self.initial_profile, "traffic_headlight_2_source_radius", None) is not None else 0.08))
        self.traffic_headlight_2_source_x_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_2_source_x", -1.0) if getattr(self.initial_profile, "traffic_headlight_2_source_x", None) is not None else -1.0))
        self.traffic_headlight_2_source_y_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_2_source_y", -1.0) if getattr(self.initial_profile, "traffic_headlight_2_source_y", None) is not None else -1.0))
        self.traffic_headlight_2_target_x_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_2_target_x", -1.0) if getattr(self.initial_profile, "traffic_headlight_2_target_x", None) is not None else -1.0))
        self.traffic_headlight_2_target_y_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_2_target_y", -1.0) if getattr(self.initial_profile, "traffic_headlight_2_target_y", None) is not None else -1.0))
        r2_source_world = _default_plate_world(
            _profile_norm_or_default("traffic_headlight_2_source_x", 0.86),
            _profile_norm_or_default("traffic_headlight_2_source_y", 0.90),
            1.55,
        )
        r2_target_world = _default_plate_world(
            _profile_norm_or_default("traffic_headlight_2_target_x", 0.58),
            _profile_norm_or_default("traffic_headlight_2_target_y", 0.48),
            1.0,
        )
        self.traffic_headlight_2_source_world_x_var = tk.DoubleVar(value=_world_or_default("traffic_headlight_2_source_world_x", r2_source_world[0]))
        self.traffic_headlight_2_source_world_y_var = tk.DoubleVar(value=_world_or_default("traffic_headlight_2_source_world_y", r2_source_world[1]))
        self.traffic_headlight_2_source_world_z_var = tk.DoubleVar(value=_initial_float("traffic_headlight_2_source_world_z", 0.95))
        self.traffic_headlight_2_target_world_x_var = tk.DoubleVar(value=_world_or_default("traffic_headlight_2_target_world_x", r2_target_world[0]))
        self.traffic_headlight_2_target_world_y_var = tk.DoubleVar(value=_world_or_default("traffic_headlight_2_target_world_y", r2_target_world[1]))
        self.traffic_headlight_2_target_world_z_var = tk.DoubleVar(value=0.0)
        self.traffic_headlight_3_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_3_strength", 0.0) or 0.0))
        self.traffic_headlight_3_warmth_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_3_warmth", 0.35) if getattr(self.initial_profile, "traffic_headlight_3_warmth", None) is not None else 0.35))
        self.traffic_headlight_3_r_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_3_r", 1.0) if getattr(self.initial_profile, "traffic_headlight_3_r", None) is not None and float(getattr(self.initial_profile, "traffic_headlight_3_r", -1.0)) >= 0 else 1.0))
        self.traffic_headlight_3_g_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_3_g", 0.88) if getattr(self.initial_profile, "traffic_headlight_3_g", None) is not None and float(getattr(self.initial_profile, "traffic_headlight_3_g", -1.0)) >= 0 else 0.88))
        self.traffic_headlight_3_b_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_3_b", 0.54) if getattr(self.initial_profile, "traffic_headlight_3_b", None) is not None and float(getattr(self.initial_profile, "traffic_headlight_3_b", -1.0)) >= 0 else 0.54))
        self.traffic_headlight_3_cone_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_3_cone", 0.45) if getattr(self.initial_profile, "traffic_headlight_3_cone", None) is not None else 0.45))
        self.traffic_headlight_3_source_radius_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_3_source_radius", 0.08) if getattr(self.initial_profile, "traffic_headlight_3_source_radius", None) is not None else 0.08))
        self.traffic_headlight_3_source_x_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_3_source_x", -1.0) if getattr(self.initial_profile, "traffic_headlight_3_source_x", None) is not None else -1.0))
        self.traffic_headlight_3_source_y_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_3_source_y", -1.0) if getattr(self.initial_profile, "traffic_headlight_3_source_y", None) is not None else -1.0))
        self.traffic_headlight_3_target_x_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_3_target_x", -1.0) if getattr(self.initial_profile, "traffic_headlight_3_target_x", None) is not None else -1.0))
        self.traffic_headlight_3_target_y_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "traffic_headlight_3_target_y", -1.0) if getattr(self.initial_profile, "traffic_headlight_3_target_y", None) is not None else -1.0))
        r3_source_world = _default_plate_world(
            _profile_norm_or_default("traffic_headlight_3_source_x", 0.50),
            _profile_norm_or_default("traffic_headlight_3_source_y", 0.98),
            1.55,
        )
        r3_target_world = _default_plate_world(
            _profile_norm_or_default("traffic_headlight_3_target_x", 0.50),
            _profile_norm_or_default("traffic_headlight_3_target_y", 0.36),
            1.0,
        )
        self.traffic_headlight_3_source_world_x_var = tk.DoubleVar(value=_world_or_default("traffic_headlight_3_source_world_x", r3_source_world[0]))
        self.traffic_headlight_3_source_world_y_var = tk.DoubleVar(value=_world_or_default("traffic_headlight_3_source_world_y", r3_source_world[1]))
        self.traffic_headlight_3_source_world_z_var = tk.DoubleVar(value=_initial_float("traffic_headlight_3_source_world_z", 0.95))
        self.traffic_headlight_3_target_world_x_var = tk.DoubleVar(value=_world_or_default("traffic_headlight_3_target_world_x", r3_target_world[0]))
        self.traffic_headlight_3_target_world_y_var = tk.DoubleVar(value=_world_or_default("traffic_headlight_3_target_world_y", r3_target_world[1]))
        self.traffic_headlight_3_target_world_z_var = tk.DoubleVar(value=0.0)
        self.traffic_headlight_1_enabled_var = tk.BooleanVar(value=float(self.traffic_headlight_var.get() or 0.0) > 0.001)
        self.traffic_headlight_2_enabled_var = tk.BooleanVar(value=float(self.traffic_headlight_2_var.get() or 0.0) > 0.001)
        self.traffic_headlight_3_enabled_var = tk.BooleanVar(value=float(self.traffic_headlight_3_var.get() or 0.0) > 0.001)
        self.wet_mud_gloss_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "wet_mud_gloss_strength", 0.0) or 0.0))
        self.water_film_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "water_film_strength", 0.0) or 0.0))
        self.water_film_unevenness_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "water_film_unevenness", 0.35) if getattr(self.initial_profile, "water_film_unevenness", None) is not None else 0.35))
        self.water_film_lens_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "water_film_lens_strength", 0.0) or 0.0))
        self.water_film_contour_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "water_film_contour_response", 0.55) if getattr(self.initial_profile, "water_film_contour_response", None) is not None else 0.55))
        self.water_film_gloss_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "water_film_gloss_strength", 0.45) if getattr(self.initial_profile, "water_film_gloss_strength", None) is not None else 0.45))
        self.flare_var = tk.DoubleVar(value=float(self.initial_profile.flare_strength or 0.0))
        self.overexposure_var = tk.DoubleVar(value=float(self.initial_profile.overexposure_strength or 0.0))
        self.dirt_streak_var = tk.DoubleVar(value=float(self.initial_profile.dirt_streak_strength or 0.0))
        legacy_flow_strength = float(self.initial_profile.dirt_flow_strength or 0.0)
        self.dirt_flow_points_var = tk.IntVar(value=int(self.initial_profile.dirt_flow_points or (24 if legacy_flow_strength > 0 else 0)))
        self.dirt_flow_mass_min_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "dirt_flow_mass_min", 0.18) or 0.18))
        self.dirt_flow_mass_max_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "dirt_flow_mass_max", 1.0) or 1.0))
        self.dirt_flow_splash_scale_var = tk.DoubleVar(value=float(self.initial_profile.dirt_flow_splash_scale))
        self.dirt_flow_trail_length_var = tk.DoubleVar(value=float(self.initial_profile.dirt_flow_trail_length))
        self.dirt_flow_humidity_var = tk.DoubleVar(value=float(self.initial_profile.dirt_flow_humidity))
        initial_stickiness = float(getattr(self.initial_profile, "dirt_flow_stickiness", 0.45) if getattr(self.initial_profile, "dirt_flow_stickiness", None) is not None else 0.45)
        self.dirt_flow_stickiness_var = tk.DoubleVar(value=initial_stickiness)
        self.dirt_flow_stickiness_min_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "dirt_flow_stickiness_min", initial_stickiness) if getattr(self.initial_profile, "dirt_flow_stickiness_min", None) is not None else initial_stickiness))
        self.dirt_flow_stickiness_max_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "dirt_flow_stickiness_max", initial_stickiness) if getattr(self.initial_profile, "dirt_flow_stickiness_max", None) is not None else initial_stickiness))
        self.dirt_flow_air_angle_var = tk.DoubleVar(value=float(self.initial_profile.dirt_flow_air_angle or 0.0))
        self.dirt_flow_wind_strength_var = tk.DoubleVar(value=float(self.initial_profile.dirt_flow_wind_strength))
        self.dirt_flow_gravity_angle_var = tk.DoubleVar(value=float(self.initial_profile.dirt_flow_gravity_angle))
        self.dirt_flow_gravity_strength_var = tk.DoubleVar(value=float(self.initial_profile.dirt_flow_gravity_strength))
        self.dirt_flow_opacity_min_var = tk.DoubleVar(value=float(self.initial_profile.dirt_flow_opacity_min))
        self.dirt_flow_opacity_max_var = tk.DoubleVar(value=float(self.initial_profile.dirt_flow_opacity_max))
        self.dirt_flow_stop_on_contour_var = tk.BooleanVar(value=bool(getattr(self.initial_profile, "dirt_flow_stop_on_dark_contour", False)))
        initial_relief_strength = float(getattr(self.initial_profile, "dark_relief_strength", 0.30) or 0.0)
        if initial_relief_strength <= 0.001 and self.target != "plate":
            initial_relief_strength = 0.30
        self.dark_relief_var = tk.DoubleVar(value=initial_relief_strength)
        self.contour_detection_sensitivity_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "contour_detection_sensitivity", 0.55) if getattr(self.initial_profile, "contour_detection_sensitivity", None) is not None else 0.55))
        self.dark_relief_light_angle_var = tk.DoubleVar(value=float(self.initial_profile.dark_relief_light_angle or 135.0))
        self.relief_bounce_depth_var = tk.IntVar(value=int(float(getattr(self.initial_profile, "relief_bounce_depth", 1) if getattr(self.initial_profile, "relief_bounce_depth", None) is not None else 1)))
        self.relief_bounce_strength_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "relief_bounce_strength", 0.28) if getattr(self.initial_profile, "relief_bounce_strength", None) is not None else 0.28))
        self.relief_profile_preset_var = tk.StringVar(value=normalize_relief_profile_preset(getattr(self.initial_profile, "relief_profile_preset", DEFAULT_RELIEF_PROFILE_PRESET)))
        self._relief_profile_curve = normalize_relief_profile_curve(getattr(self.initial_profile, "relief_profile_curve", DEFAULT_RELIEF_PROFILE_CURVE))
        self._relief_profile_drag_index: int | None = None
        self.light_normal_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "light_normal_strength", 0.0) or 0.0))
        self.overhang_shadow_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "overhang_shadow_strength", 0.0) or 0.0))
        self.overhang_shadow_depth_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "overhang_shadow_depth", 0.60) if getattr(self.initial_profile, "overhang_shadow_depth", None) is not None else 0.60))
        self.overhang_shadow_skew_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "overhang_shadow_skew", 0.0) or 0.0))
        self.plate_reflect_gradient_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "plate_reflect_gradient_strength", 0.0) or 0.0))
        self.plate_reflect_glare_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "plate_reflect_glare_strength", 0.0) or 0.0))
        self.plate_reflect_curve_var = tk.DoubleVar(value=float(getattr(self.initial_profile, "plate_reflect_curve_strength", 0.0) or 0.0))
        initial_blur = float(getattr(self.initial_profile, "blur_strength", 0.0) or 0.0)
        if initial_blur <= 0.001 and bool(getattr(self.initial_profile, "blur_enabled", False)):
            initial_blur = 0.18
        self.blur_strength_var = tk.DoubleVar(value=initial_blur)
        self.blur_var = tk.BooleanVar(value=initial_blur > 0.001)
        default_class = self.initial_profile.class_name or ("plate" if self.target == "plate" else "")
        self.class_var = tk.StringVar(value=default_class)
        self._preview_after_id = None
        self._raw_preview_request_id = 0
        self._raw_preview_worker_running = False
        self._raw_preview_pending = False
        self._vector_tool: str | None = None
        self._vector_drag_start: tuple[int, int] | None = None
        self._vector_drag_end: tuple[int, int] | None = None
        self._vector_positions: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
        self._effect_seed = int(getattr(self.initial_profile, "seed", 42) or 42)
        self._active_toolbox = "geometry"
        self._effect_inspector_visible = True
        self._toolbox_buttons: dict[str, tk.Widget] = {}
        self._inspector_anchor_widgets: dict[str, tk.Widget] = {}
        self._inspector_field_hints: dict[str, str] = {}
        self._inspector_tooltip_window = None
        self._inspector_tooltip_after_id = None
        self._pending_inspector_scroll_anchor: str | None = None
        self._toolbox_panel_pos: tuple[int, int] | None = None
        self._toolbox_panel_drag: dict | None = None
        self._toolbox_panel_collapsed = False
        self._toolbox_panel_visible = False
        self._toolbox_panel_scroll = 0.0
        self._toolbox_panel_max_scroll = 0.0
        self._range_sync_guard = False
        self._canvas_overlay_regions: list[dict] = []
        self._canvas_slider_drag: dict | None = None
        self._wind_panel_drag: dict | None = None
        self._wind_strength_editor: tk.Entry | None = None
        self._wind_strength_editor_var: tk.StringVar | None = None
        self._headlight_drag: dict | None = None
        self._headlight_radius_drag: dict | None = None
        self._headlight_config_index: int | None = None
        self._headlight_panel_pos: tuple[int, int] | None = None
        self._headlight_panel_anchor_index: int | None = None
        self._headlight_panel_drag: dict | None = None
        self._scene_projection_drag: dict | None = None
        self._scene_view_drag: dict | None = None
        self._scene_pan_drag: dict | None = None
        self._scene_projection_active: str = "xy"
        self._scene_viewport_mode: str = "xy"
        self._scene_viewport_visible: bool = True
        self._scene_view_rotation_matrix: tuple[tuple[float, float, float], ...] | None = None
        self._scene_view_rotation_key: tuple[float, float, float] | None = None
        self._scene_view_pan_x = 0.0
        self._scene_view_pan_y = 0.0
        self._scene_projection_expanded: bool = False
        self._scene_plate_texture_cache_key: tuple | None = None
        self._scene_plate_texture_photo_ref = None
        self._scene_plate_texture_quad_cache_key: tuple | None = None
        self._scene_plate_texture_quad_photo_ref = None
        self._scene_contour_wireframe_cache_key: tuple | None = None
        self._scene_contour_wireframe_cache: list[dict] = []
        self._active_headlight_index: int | None = 1
        self._headlight_visibility = {1: True, 2: False, 3: False}
        self._headlight_visibility_vars: dict[int, tk.BooleanVar] = {}
        self._headlight_config_hover_index: int | None = None
        self._overlay_redraw_after_id = None
        self._preview_refresh_suspended = False
        self._preview_refresh_dirty = False
        self._inspector_scale_drag_active = False
        self._preview_zoom = self._default_preview_zoom()
        self._scene_view_zoom = 1.0
        self._preview_fullscreen = False
        self._preview_pan_x = 0.0
        self._preview_pan_y = 0.0
        self._preview_pan_drag: tuple[int, int, float, float] | None = None
        self._preview_canvas_mappings: dict[int, dict[str, float]] = {}
        self._fullscreen_show_original = False
        self._tab_toggle_bindtag = f"AugmentationPreviewTabToggle{id(self)}"
        self._preview_fullscreen_grid_snapshot: dict[str, dict] = {}
        self._manual_randomness = normalize_manual_randomness_config(
            getattr(self.initial_profile, "manual_randomness", {}),
            self.target,
        )
        self._refresh_extra_count_title()

        self._bind_range_guards()
        self._build()
        self._bind_fullscreen_preview_tab_toggle()
        self._bind_realtime_preview()
        self._refresh_dependency_status()
        self._center_window_on_master()
        self._refresh_preview()

    def _configure_modal_window(self, master):
        """Keep the tab blocked while preserving a normal maximizable window."""
        windowing_system = ""
        try:
            windowing_system = str(self.window.tk.call("tk", "windowingsystem") or "")
        except Exception:
            windowing_system = ""

        # On Windows transient dialog windows often lose the normal maximize button.
        if windowing_system != "win32":
            try:
                self.window.transient(master)
            except Exception:
                pass

        try:
            self.window.resizable(True, True)
        except Exception:
            pass
        try:
            self.window.grab_set()
        except Exception:
            pass
        try:
            self.window.lift(master)
        except Exception:
            try:
                self.window.lift()
            except Exception:
                pass

    def _center_window_on_master(self):
        try:
            self.window.update_idletasks()
            width = max(1, int(self.window.winfo_width() or self.window.winfo_reqwidth() or 980))
            height = max(1, int(self.window.winfo_height() or self.window.winfo_reqheight() or 720))
            master = self.master
            master.update_idletasks()
            root_x = int(master.winfo_rootx())
            root_y = int(master.winfo_rooty())
            root_w = int(master.winfo_width() or master.winfo_screenwidth())
            root_h = int(master.winfo_height() or master.winfo_screenheight())
            if root_w <= 1 or root_h <= 1:
                raise ValueError("master geometry not ready")
            x = root_x + max(0, (root_w - width) // 2)
            y = root_y + max(0, (root_h - height) // 2)
        except Exception:
            try:
                width = max(1, int(self.window.winfo_width() or self.window.winfo_reqwidth() or 980))
                height = max(1, int(self.window.winfo_height() or self.window.winfo_reqheight() or 720))
                screen_w = int(self.window.winfo_screenwidth())
                screen_h = int(self.window.winfo_screenheight())
                x = max(0, (screen_w - width) // 2)
                y = max(0, (screen_h - height) // 2)
            except Exception:
                return
        try:
            self.window.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            pass

    def show(self) -> AugmentationProfile | None:
        self.window.wait_window()
        return self.result

    def _bind_fullscreen_preview_tab_toggle(self):
        sequences = ("<Tab>", "<ISO_Left_Tab>", "<Shift-Tab>")
        tag = str(getattr(self, "_tab_toggle_bindtag", "") or "")
        if not tag:
            return

        try:
            for sequence in sequences:
                self.window.bind_class(tag, sequence, self._on_tab_key)
        except Exception:
            pass

        skipped_classes = {"Entry", "TEntry", "Text", "Spinbox", "TSpinbox"}

        def bind_tree(widget):
            try:
                widget_class = str(widget.winfo_class() or "")
                if widget_class not in skipped_classes:
                    tags = list(widget.bindtags())
                    if tag not in tags:
                        insert_at = 1 if tags else 0
                        tags.insert(insert_at, tag)
                        widget.bindtags(tuple(tags))
            except Exception:
                pass
            try:
                for child in widget.winfo_children():
                    bind_tree(child)
            except Exception:
                pass

        try:
            self.window.bind("<ISO_Left_Tab>", self._on_tab_key)
            self.window.bind("<Shift-Tab>", self._on_tab_key)
        except Exception:
            pass
        try:
            bind_tree(self.window)
        except Exception:
            pass
        canvas = getattr(self, "augmented_canvas", None)
        if canvas is not None:
            try:
                canvas.configure(takefocus=True)
            except Exception:
                pass

    def _build(self):
        self._build_compact_layout()
        return

        outer = ttk.Frame(self.window, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            outer,
            text="Syntetyczne zwiększanie datasetu",
            font=("Segoe UI", 14, "bold"),
        )
        title.pack(anchor=tk.W)

        intro = ttk.Label(
            outer,
            text=(
                "Syntetyczne powiększanie zbioru bazuje na efekcie bazowym, "
                "który dostrajasz w edytorze efektów. Program zapisze dodatkowe obrazy "
                "do splitu train; val i test pozostają bez zmian, żeby ocena modelu była uczciwa. "
                "Warianty nie będą kopią 1:1 ustawień z podglądu: dla każdego nowego zdjęcia "
                "zostanie dodany kontrolowany rozrzut parametrów liczony od cech bazowych."
            ),
            justify=tk.LEFT,
            wraplength=920,
        )
        intro.pack(anchor=tk.W, pady=(4, 10))

        dep_row = ttk.Frame(outer)
        dep_row.pack(fill=tk.X, pady=(0, 10))
        self.dep_status_lbl = ttk.Label(dep_row, text="Albumentations: sprawdzam...")
        self.dep_status_lbl.pack(side=tk.LEFT)
        self.dep_install_btn = ttk.Button(dep_row, text="Doinstaluj", command=self._install_dependency)
        self.dep_install_btn.pack(side=tk.LEFT, padx=(10, 0))

        dataset_settings = ttk.LabelFrame(outer, text=" Parametry datasetu ", padding=10)
        dataset_settings.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(dataset_settings, text="Losowa próbka obrazów źródłowych").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=3)
        ttk.Spinbox(dataset_settings, from_=1, to=self.sample_pool_limit, textvariable=self.sample_var, width=8).grid(row=0, column=1, sticky=tk.W, pady=3)
        ttk.Label(dataset_settings, text="Liczba dodatkowych obrazów train").grid(row=0, column=2, sticky=tk.W, padx=(18, 6), pady=3)
        ttk.Spinbox(dataset_settings, from_=0, to=100000, textvariable=self.extra_var, width=8).grid(row=0, column=3, sticky=tk.W, pady=3)
        if self.target == "plate":
            ttk.Label(dataset_settings, text="Nazwa klasy YOLO").grid(row=2, column=0, sticky=tk.W, padx=(0, 6), pady=(8, 0))
            ttk.Entry(dataset_settings, textvariable=self.class_var, width=18).grid(row=2, column=1, sticky=tk.W, pady=(8, 0))
            ttk.Label(dataset_settings, text="Np. plate albo pl.").grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=(18, 0), pady=(8, 0))
        else:
            ttk.Label(
                dataset_settings,
                text="Klasy znaków pozostają zgodne z data.yaml wybranego datasetu.",
            ).grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(8, 0))
        dataset_settings.columnconfigure(3, weight=1)

        image_settings = ttk.LabelFrame(outer, text=" Parametry obrazu ", padding=10)
        image_settings.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            image_settings,
            text="Te parametry zmieniają wygląd syntetycznych obrazów, ale nie liczbę pozycji w datasecie.",
            justify=tk.LEFT,
            wraplength=880,
        ).grid(row=0, column=0, columnspan=6, sticky=tk.EW, pady=(0, 8))
        ttk.Label(image_settings, text="Obrót").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=3)
        ttk.Spinbox(image_settings, from_=-15, to=15, increment=1, textvariable=self.rotation_var, width=6).grid(row=1, column=1, sticky=tk.W, pady=3)
        ttk.Label(image_settings, text="Jasność").grid(row=1, column=2, sticky=tk.W, padx=(18, 6), pady=3)
        ttk.Spinbox(image_settings, from_=0.0, to=0.25, increment=0.01, textvariable=self.brightness_var, width=8).grid(row=1, column=3, sticky=tk.W, pady=3)
        ttk.Label(image_settings, text="Siła szumu").grid(row=1, column=4, sticky=tk.W, padx=(18, 6), pady=3)
        ttk.Spinbox(image_settings, from_=0.0, to=0.08, increment=0.005, textvariable=self.noise_var, width=8).grid(row=1, column=5, sticky=tk.W, pady=3)
        ttk.Label(image_settings, text="Kontrast").grid(row=2, column=0, sticky=tk.W, padx=(0, 6), pady=3)
        ttk.Spinbox(image_settings, from_=0.0, to=0.25, increment=0.01, textvariable=self.contrast_var, width=8).grid(row=2, column=1, sticky=tk.W, pady=3)
        ttk.Label(image_settings, text="Nasycenie").grid(row=2, column=2, sticky=tk.W, padx=(18, 6), pady=3)
        ttk.Spinbox(image_settings, from_=0.0, to=3.0, increment=0.01, textvariable=self.saturation_var, width=8).grid(row=2, column=3, sticky=tk.W, pady=3)
        ttk.Label(image_settings, text="Rozmycie").grid(row=2, column=4, sticky=tk.W, padx=(18, 6), pady=3)
        ttk.Spinbox(image_settings, from_=0.0, to=1.0, increment=0.05, textvariable=self.blur_strength_var, width=8).grid(row=2, column=5, sticky=tk.W, pady=3)
        ttk.Label(image_settings, text="Ziarno/deszcz").grid(row=3, column=0, sticky=tk.W, padx=(0, 6), pady=3)
        ttk.Spinbox(image_settings, from_=1, to=12, increment=1, textvariable=self.noise_grain_var, width=6).grid(row=3, column=1, sticky=tk.W, pady=3)
        ttk.Label(image_settings, text="Gęstość deszczu").grid(row=3, column=2, sticky=tk.W, padx=(18, 6), pady=3)
        ttk.Spinbox(image_settings, from_=0.0, to=1.0, increment=0.05, textvariable=self.rain_var, width=8).grid(row=3, column=3, sticky=tk.W, pady=3)
        ttk.Label(image_settings, text="Efekt nocy").grid(row=3, column=4, sticky=tk.W, padx=(18, 6), pady=3)
        ttk.Spinbox(image_settings, from_=0.0, to=1.0, increment=0.05, textvariable=self.night_var, width=8).grid(row=3, column=5, sticky=tk.W, pady=3)
        ttk.Label(image_settings, text="Przebłyski").grid(row=4, column=0, sticky=tk.W, padx=(0, 6), pady=3)
        ttk.Spinbox(image_settings, from_=0.0, to=1.0, increment=0.05, textvariable=self.flare_var, width=8).grid(row=4, column=1, sticky=tk.W, pady=3)
        ttk.Label(image_settings, text="Prześwietlenie").grid(row=4, column=2, sticky=tk.W, padx=(18, 6), pady=3)
        ttk.Spinbox(image_settings, from_=0.0, to=1.0, increment=0.05, textvariable=self.overexposure_var, width=8).grid(row=4, column=3, sticky=tk.W, pady=3)
        ttk.Label(image_settings, text="Brudne zacieki").grid(row=5, column=0, sticky=tk.W, padx=(0, 6), pady=3)
        ttk.Spinbox(image_settings, from_=0.0, to=1.0, increment=0.05, textvariable=self.dirt_streak_var, width=8).grid(row=5, column=1, sticky=tk.W, pady=3)
        image_settings.columnconfigure(5, weight=1)

        preview_shell = ttk.LabelFrame(outer, text=" Podgląd ", padding=10)
        preview_shell.pack(fill=tk.BOTH, expand=True)
        preview_toolbar = ttk.Frame(preview_shell)
        preview_toolbar.pack(fill=tk.X, pady=(0, 8))
        self.preview_status_lbl = ttk.Label(preview_toolbar, text="")
        self.preview_status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(preview_toolbar, text="Losuj obraz", command=self._pick_random_sample).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(preview_toolbar, text="Odśwież podgląd", command=self._refresh_preview).pack(side=tk.RIGHT)

        canvases = ttk.Frame(preview_shell)
        canvases.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(canvases)
        right = ttk.Frame(canvases)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        ttk.Label(left, text="Oryginał").pack(anchor=tk.W)
        ttk.Label(right, text="Po augmentacji").pack(anchor=tk.W)
        self.original_canvas = tk.Canvas(left, bg="#111111", height=280, highlightthickness=0)
        self.augmented_canvas = tk.Canvas(right, bg="#111111", height=280, highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        self.augmented_canvas.pack(fill=tk.BOTH, expand=True)
        self.original_canvas.bind("<Configure>", lambda _event: self._refresh_preview(redraw_only=True), add="+")
        self.augmented_canvas.bind("<Configure>", lambda _event: self._refresh_preview(redraw_only=True), add="+")
        self._bind_preview_zoom_canvas(self.original_canvas)
        self._bind_preview_zoom_canvas(self.augmented_canvas)

        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(footer, text="Anuluj", command=self._cancel).pack(side=tk.RIGHT)
        ttk.Button(footer, text="Zapisz profil i wróć", command=self._accept).pack(side=tk.RIGHT, padx=(0, 8))

    def _build_compact_layout(self):
        self.window.geometry("1160x720")
        self.window.minsize(960, 620)

        outer = ttk.Frame(self.window, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        header = ttk.Frame(outer)
        self.header_frame = header
        header.grid(row=0, column=0, sticky=tk.EW, pady=(0, 8))
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Edytor efekt\u00f3w augmentacji",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, sticky=tk.W)

        dep_row = ttk.Frame(header)
        dep_row.grid(row=0, column=1, sticky=tk.E)
        self.dep_status_lbl = ttk.Label(dep_row, text="Albumentations: sprawdzam...")
        self.dep_status_lbl.pack(side=tk.LEFT)
        self.dep_install_btn = ttk.Button(dep_row, text="Doinstaluj", command=self._install_dependency)
        self.dep_install_btn.pack(side=tk.LEFT, padx=(8, 0))

        effect_intro_label = ttk.Label(
            header,
            text="Dostrój efekt bazowy. Liczbę generowanych obrazów ustawiasz w PZ1.",
            justify=tk.LEFT,
            wraplength=980,
        )
        effect_intro_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))

        body = ttk.Frame(outer)
        body.grid(row=1, column=0, sticky=tk.NSEW)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        controls_host = ttk.Frame(body, width=340)
        self.controls_host = controls_host
        controls_host.grid(row=0, column=0, sticky=tk.NS, padx=(0, 10))
        controls_host.grid_propagate(False)
        controls_host.columnconfigure(0, weight=1)
        controls_host.rowconfigure(0, weight=1)

        controls_canvas = tk.Canvas(controls_host, highlightthickness=0, bd=0)
        controls_scrollbar = ttk.Scrollbar(controls_host, orient=tk.VERTICAL, command=controls_canvas.yview)
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_canvas.grid(row=0, column=0, sticky=tk.NSEW)
        controls_scrollbar.grid(row=0, column=1, sticky=tk.NS)

        controls = ttk.Frame(controls_canvas)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor=tk.NW)
        controls.columnconfigure(0, weight=1)

        def refresh_controls_scroll(_event=None):
            try:
                controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))
            except Exception:
                pass

        def resize_controls_window(event):
            try:
                controls_canvas.itemconfigure(controls_window, width=max(1, int(event.width)))
            except Exception:
                pass

        def scroll_controls(event):
            try:
                if getattr(event, "num", None) == 4:
                    units = -3
                elif getattr(event, "num", None) == 5:
                    units = 3
                else:
                    delta = int(getattr(event, "delta", 0) or 0)
                    units = -1 * int(delta / 120) if delta else 0
                if units:
                    controls_canvas.yview_scroll(units, "units")
                    return "break"
            except Exception:
                return None
            return None

        def bind_controls_scroll_tree(widget):
            try:
                for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    widget.bind(sequence, scroll_controls, add="+")
                for child in widget.winfo_children():
                    bind_controls_scroll_tree(child)
            except Exception:
                pass

        controls.bind("<Configure>", refresh_controls_scroll, add="+")
        controls_canvas.bind("<Configure>", resize_controls_window, add="+")
        self.window.after_idle(lambda: bind_controls_scroll_tree(controls))

        dataset_settings = ttk.LabelFrame(controls, text=" Parametry datasetu ", padding=8)
        dataset_settings.grid(row=0, column=0, sticky=tk.EW, pady=(0, 8))
        dataset_settings.columnconfigure(1, weight=1)
        ttk.Label(
            dataset_settings,
            textvariable=self.extra_count_title_var,
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        ttk.Label(dataset_settings, text="Liczba nowych zdjęć").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(dataset_settings, from_=0, to=100000, textvariable=self.extra_var, width=8).grid(row=1, column=1, sticky=tk.EW, pady=2)
        ttk.Label(dataset_settings, text="dopisz do train").grid(row=1, column=2, sticky=tk.W, padx=(6, 0), pady=2)
        ttk.Label(dataset_settings, text="Baza losowania").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(dataset_settings, from_=1, to=self.sample_pool_limit, textvariable=self.sample_var, width=8).grid(row=2, column=1, sticky=tk.EW, pady=2)
        ttk.Label(dataset_settings, text=f"max {self.sample_pool_limit}").grid(row=2, column=2, sticky=tk.W, padx=(6, 0), pady=2)
        if self.target == "plate":
            ttk.Label(dataset_settings, text="Klasa YOLO").grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
            ttk.Entry(dataset_settings, textvariable=self.class_var, width=14).grid(row=3, column=1, sticky=tk.EW, pady=(6, 0))
        else:
            ttk.Label(
                dataset_settings,
                text="Klasy znaków zgodne z data.yaml.",
                wraplength=260,
            ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))

        dataset_settings.grid_remove()

        image_settings = ttk.LabelFrame(controls, text=" Obraz i deszcz ", padding=6)
        image_settings.grid(row=1, column=0, sticky=tk.EW)
        image_settings.columnconfigure(1, weight=1)
        image_settings.columnconfigure(3, weight=1)

        def spin_cell(row, col, label, variable, from_, to, increment, width=6):
            ttk.Label(image_settings, text=label).grid(row=row, column=col, sticky=tk.W, pady=1, padx=(0, 4))
            ttk.Spinbox(
                image_settings,
                from_=from_,
                to=to,
                increment=increment,
                textvariable=variable,
                width=width,
            ).grid(row=row, column=col + 1, sticky=tk.EW, pady=1, padx=(0, 8 if col == 0 else 0))

        spin_cell(0, 0, "Obrót", self.rotation_var, -15, 15, 1)
        spin_cell(0, 2, "Jasność", self.brightness_var, 0.0, 0.25, 0.01)
        spin_cell(1, 0, "Kontrast", self.contrast_var, 0.0, 0.25, 0.01)
        spin_cell(1, 2, "Nasycenie", self.saturation_var, 0.0, 3.0, 0.01)
        spin_cell(2, 0, "Rozmycie", self.blur_strength_var, 0.0, 1.0, 0.05)
        spin_cell(2, 2, "Szum", self.noise_var, 0.0, 0.08, 0.005)
        spin_cell(3, 0, "Ziarno", self.noise_grain_var, 1, 12, 1)
        spin_cell(3, 2, "Gęstość deszczu", self.rain_var, 0.0, 1.0, 0.05)
        spin_cell(4, 0, "Kropla", self.rain_drop_size_var, 0.0, 1.0, 0.05)

        dirt_flow_settings = ttk.LabelFrame(controls, text=" Błoto i światło ", padding=6)
        dirt_flow_settings.grid(row=2, column=0, sticky=tk.EW, pady=(8, 0))
        dirt_flow_settings.columnconfigure(1, weight=1)
        dirt_flow_settings.columnconfigure(3, weight=1)

        def compact_spin(row, col, label, variable, from_, to, increment, width=6):
            ttk.Label(dirt_flow_settings, text=label).grid(row=row, column=col, sticky=tk.W, pady=1, padx=(0, 4))
            ttk.Spinbox(
                dirt_flow_settings,
                from_=from_,
                to=to,
                increment=increment,
                textvariable=variable,
                width=width,
            ).grid(row=row, column=col + 1, sticky=tk.EW, pady=1, padx=(0, 8 if col == 0 else 0))

        compact_spin(0, 0, "Grudki", self.dirt_flow_points_var, 0, 2000, 1, 6)
        compact_spin(0, 2, "Rozm. plam", self.dirt_flow_splash_scale_var, 0.0, 1.0, 0.05)
        compact_spin(1, 0, "Dł. smug", self.dirt_flow_trail_length_var, 0.0, 1.0, 0.05)
        compact_spin(1, 2, "Wilgoć", self.dirt_flow_humidity_var, 0.0, 1.0, 0.05)
        compact_spin(2, 0, "Krycie min", self.dirt_flow_opacity_min_var, 0.0, 1.0, 0.05)
        compact_spin(2, 2, "Krycie max", self.dirt_flow_opacity_max_var, 0.0, 1.0, 0.05)
        compact_spin(3, 0, "Prędkość/uderzenie", self.vehicle_speed_var, 0.0, 1.0, 0.05)
        compact_spin(3, 2, "Wypukłość konturów", self.dark_relief_var, 0.0, 3.0, 0.05)
        preview_shell = ttk.LabelFrame(body, text=" Podgląd ", padding=8)
        try:
            image_settings.grid_remove()
            dirt_flow_settings.grid_remove()
        except Exception:
            pass

        domain_info = ttk.LabelFrame(controls, text=" Domena augmentacji ", padding=8)
        domain_info.grid(row=1, column=0, sticky=tk.EW, pady=(8, 0))
        domain_info.columnconfigure(0, weight=1)
        ttk.Label(
            domain_info,
            text=self._domain_info_text(),
            justify=tk.LEFT,
            wraplength=270,
        ).grid(row=0, column=0, sticky=tk.EW)
        controls_host.grid_remove()
        self._effects_controls_panel_hidden = True

        preview_shell.grid(row=0, column=0, sticky=tk.NSEW)
        preview_shell.columnconfigure(0, weight=1)
        preview_shell.rowconfigure(1, weight=1)
        self._build_effect_inspector(body)

        preview_toolbar = ttk.Frame(preview_shell)
        self.preview_toolbar = preview_toolbar
        preview_toolbar.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
        preview_toolbar.columnconfigure(0, weight=1)
        self.preview_status_lbl = ttk.Label(preview_toolbar, text="")
        self.preview_status_lbl.grid(row=0, column=0, sticky=tk.EW)
        randomness_box = ttk.Frame(preview_toolbar)
        randomness_box.grid(row=0, column=1, sticky=tk.E, padx=(12, 0))
        ttk.Label(randomness_box, text="Losowość serii").pack(side=tk.LEFT, padx=(0, 6))
        self.randomness_combo = ttk.Combobox(
            randomness_box,
            textvariable=self.randomness_mode_var,
            values=[label for _mode, label in self._randomness_label_pairs()],
            state="readonly",
            width=14,
        )
        self.randomness_combo.pack(side=tk.LEFT)
        ttk.Button(
            randomness_box,
            text="Odchylki",
            command=self._open_manual_randomness_modal,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            randomness_box,
            text="Wczytaj preset",
            command=self._load_augmentation_profile_from_file,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            randomness_box,
            text="Zapisz preset",
            command=self._save_augmentation_profile_to_preset,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            randomness_box,
            text="Ustawienia",
            command=self._show_effect_inspector,
        ).pack(side=tk.LEFT, padx=(8, 0))

        canvases = ttk.Frame(preview_shell)
        self.canvases_frame = canvases
        canvases.grid(row=1, column=0, sticky=tk.NSEW)
        canvases.columnconfigure(0, weight=1)
        canvases.rowconfigure(0, weight=1)

        right = ttk.Frame(canvases)
        self.left_preview_frame = None
        self.right_preview_frame = right
        right.grid(row=0, column=0, sticky=tk.NSEW)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.original_canvas = None
        self.augmented_canvas = tk.Canvas(right, bg="#111111", height=320, highlightthickness=0)
        self.augmented_canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self.augmented_canvas.bind("<Configure>", lambda _event: self._refresh_preview(redraw_only=True), add="+")
        self._bind_preview_zoom_canvas(self.augmented_canvas)
        self._bind_vector_canvas(self.augmented_canvas)

        footer = ttk.Frame(outer)
        self.footer_frame = footer
        footer.grid(row=2, column=0, sticky=tk.EW, pady=(8, 0))
        ttk.Button(footer, text="Anuluj", command=self._cancel).pack(side=tk.RIGHT)
        ttk.Button(footer, text="Zastosuj", command=self._accept).pack(side=tk.RIGHT, padx=(0, 8))

    def _randomness_label_pairs(self) -> tuple[tuple[str, str], ...]:
        return (
            ("fixed", "Stały"),
            ("soft", "Łagodny"),
            ("realistic", "Realistyczny"),
            ("wide", "Agresywny"),
            ("manual", "Reczny"),
        )

    def _randomness_mode_to_label(self, mode: object) -> str:
        normalized = normalize_augmentation_randomness_mode(mode)
        for value, label in self._randomness_label_pairs():
            if value == normalized:
                return label
        return "Realistyczny"

    def _randomness_label_to_mode(self, label: object) -> str:
        normalized_label = str(label or "").strip().lower()
        for value, item_label in self._randomness_label_pairs():
            if normalized_label == item_label.lower():
                return value
        return normalize_augmentation_randomness_mode(label)

    def _selected_randomness_mode(self) -> str:
        try:
            return normalize_augmentation_randomness_mode(
                self._randomness_label_to_mode(self.randomness_mode_var.get())
            )
        except Exception:
            return "realistic"

    def _manual_randomness_config_for_profile(self) -> dict:
        return normalize_manual_randomness_config(
            getattr(self, "_manual_randomness", {}),
            self.target,
        )

    def _center_child_window(self, window: tk.Toplevel, width: int, height: int) -> None:
        try:
            self.window.update_idletasks()
            window.update_idletasks()
            root_x = int(self.window.winfo_rootx())
            root_y = int(self.window.winfo_rooty())
            root_w = int(self.window.winfo_width() or width)
            root_h = int(self.window.winfo_height() or height)
            x = root_x + max(0, (root_w - width) // 2)
            y = root_y + max(0, (root_h - height) // 2)
            window.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            try:
                screen_w = int(window.winfo_screenwidth())
                screen_h = int(window.winfo_screenheight())
                window.geometry(f"{width}x{height}+{max(0, (screen_w - width) // 2)}+{max(0, (screen_h - height) // 2)}")
            except Exception:
                pass

    def _open_manual_randomness_modal(self):
        config = self._manual_randomness_config_for_profile()
        window = tk.Toplevel(self.window)
        window.title("Reczne odchylki augmentacji")
        window.transient(self.window)
        window.resizable(True, True)
        self._center_child_window(window, 760, 640)

        shell = ttk.Frame(window, padding=12)
        shell.grid(row=0, column=0, sticky=tk.NSEW)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        intro = ttk.Label(
            shell,
            text=(
                "Ustaw, ktore grupy efektow i ktore cechy moga losowo odchylac sie "
                "od profilu bazowego. 0% blokuje odchylke, 100% daje pelny zakres presetu."
            ),
            wraplength=700,
            justify=tk.LEFT,
        )
        intro.grid(row=0, column=0, sticky=tk.EW, pady=(0, 10))

        canvas = tk.Canvas(shell, highlightthickness=0)
        scrollbar = ttk.Scrollbar(shell, orient=tk.VERTICAL, command=canvas.yview)
        body = ttk.Frame(canvas)
        body_id = canvas.create_window((0, 0), window=body, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=1, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=1, column=1, sticky=tk.NS)
        body.columnconfigure(0, weight=1)

        def _sync_scrollregion(_event=None):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
                canvas.itemconfigure(body_id, width=max(1, int(canvas.winfo_width())))
            except Exception:
                pass

        body.bind("<Configure>", _sync_scrollregion, add="+")
        canvas.bind("<Configure>", _sync_scrollregion, add="+")

        variables: dict[str, dict] = {}

        def _make_percent_row(parent, row: int, label: str, enabled_var, amount_var, *, disabled: bool = False):
            state = tk.DISABLED if disabled else tk.NORMAL
            ttk.Checkbutton(parent, variable=enabled_var, state=state).grid(row=row, column=0, sticky=tk.W, padx=(0, 6), pady=2)
            ttk.Label(parent, text=label).grid(row=row, column=1, sticky=tk.W, padx=(0, 8), pady=2)
            value_lbl = ttk.Label(parent, text=f"{float(amount_var.get() or 0.0):.0f}%", width=5, anchor=tk.E)
            slider = ttk.Scale(parent, from_=0, to=100, variable=amount_var, orient=tk.HORIZONTAL, state=state)
            slider.grid(row=row, column=2, sticky=tk.EW, padx=(0, 8), pady=2)
            value_lbl.grid(row=row, column=3, sticky=tk.E, pady=2)

            def _refresh_label(*_args):
                try:
                    value_lbl.configure(text=f"{float(amount_var.get() or 0.0):.0f}%")
                except Exception:
                    pass

            amount_var.trace_add("write", _refresh_label)

        for group_index, spec in enumerate(manual_randomness_group_specs(self.target)):
            group_key = str(spec.get("key") or "")
            group_state = (config.get("groups") or {}).get(group_key, {})
            frame = ttk.LabelFrame(body, text=str(spec.get("label") or group_key), padding=10)
            frame.grid(row=group_index, column=0, sticky=tk.EW, pady=(0, 10))
            frame.columnconfigure(2, weight=1)
            applicable = bool(spec.get("applicable", True))
            enabled_var = tk.BooleanVar(value=bool(group_state.get("enabled")) and applicable)
            amount_var = tk.DoubleVar(value=float(group_state.get("amount", 100.0) if applicable else 0.0))
            variables[group_key] = {"enabled": enabled_var, "amount": amount_var, "fields": {}}
            _make_percent_row(
                frame,
                0,
                "Cala grupa",
                enabled_var,
                amount_var,
                disabled=not applicable,
            )
            note = str(spec.get("note") or "")
            if note:
                ttk.Label(frame, text=note, wraplength=660, justify=tk.LEFT).grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=(0, 6))
            start_row = 2
            raw_fields = group_state.get("fields", {}) if isinstance(group_state.get("fields"), dict) else {}
            for field_index, field_spec in enumerate(spec.get("fields", [])):
                field_key = str(field_spec.get("key") or "")
                field_state = raw_fields.get(field_key, {}) if isinstance(raw_fields, dict) else {}
                field_enabled = tk.BooleanVar(value=bool(field_state.get("enabled", True)) and applicable)
                field_amount = tk.DoubleVar(value=float(field_state.get("amount", 100.0) if applicable else 0.0))
                variables[group_key]["fields"][field_key] = {"enabled": field_enabled, "amount": field_amount}
                _make_percent_row(
                    frame,
                    start_row + field_index,
                    str(field_spec.get("label") or field_key),
                    field_enabled,
                    field_amount,
                    disabled=not applicable,
                )

        footer = ttk.Frame(shell)
        footer.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(10, 0))
        footer.columnconfigure(0, weight=1)

        def _apply():
            groups = {}
            for group_key, group_vars in variables.items():
                field_values = {}
                for field_key, field_vars in group_vars["fields"].items():
                    field_values[field_key] = {
                        "enabled": bool(field_vars["enabled"].get()),
                        "amount": float(field_vars["amount"].get() or 0.0),
                    }
                groups[group_key] = {
                    "enabled": bool(group_vars["enabled"].get()),
                    "amount": float(group_vars["amount"].get() or 0.0),
                    "fields": field_values,
                }
            self._manual_randomness = normalize_manual_randomness_config(
                {"target": self.target, "groups": groups},
                self.target,
            )
            self.randomness_mode_var.set(self._randomness_mode_to_label("manual"))
            self._refresh_preview(redraw_only=True)
            window.destroy()

        ttk.Button(footer, text="Anuluj", command=window.destroy).pack(side=tk.RIGHT)
        ttk.Button(footer, text="Zastosuj odchylki", command=_apply).pack(side=tk.RIGHT, padx=(0, 8))

    def _profile_var_bindings(self) -> tuple[tuple[str, object], ...]:
        return (
            ("sample_size", self.sample_var),
            ("extra_count", self.extra_var),
            ("rotation_limit", self.rotation_var),
            ("brightness_limit", self.brightness_var),
            ("contrast_limit", self.contrast_var),
            ("saturation_limit", self.saturation_var),
            ("noise_strength", self.noise_var),
            ("noise_grain_size", self.noise_grain_var),
            ("rain_strength", self.rain_var),
            ("rain_drop_size", self.rain_drop_size_var),
            ("rain_drop_size_min", self.rain_drop_size_min_var),
            ("rain_drop_size_max", self.rain_drop_size_max_var),
            ("rain_vector_field_strength", self.rain_vector_field_var),
            ("rain_vortex_strength", self.rain_vortex_var),
            ("rain_alpha", self.rain_alpha_var),
            ("rain_lens_strength", self.rain_lens_var),
            ("rain_edge_mist_strength", self.rain_edge_mist_var),
            ("rain_edge_mist_radius", self.rain_edge_mist_radius_var),
            ("tyndall_strength", self.tyndall_var),
            ("wet_reflection_strength", self.wet_reflection_var),
            ("vehicle_speed", self.vehicle_speed_var),
            ("night_strength", self.night_var),
            ("night_luma_min", self.night_luma_min_var),
            ("night_luma_max", self.night_luma_max_var),
            ("night_light_strength", self.night_light_var),
            ("night_bloom_strength", self.night_bloom_var),
            ("night_iso_noise_strength", self.night_iso_noise_var),
            ("night_light_warmth", self.night_warmth_var),
            ("scene_camera_z", self.scene_camera_z_var),
            ("scene_view_yaw", self.scene_view_yaw_var),
            ("scene_view_pitch", self.scene_view_pitch_var),
            ("scene_view_roll", self.scene_view_roll_var),
            ("scene_plate_texture_enabled", self.scene_plate_texture_var),
            ("traffic_headlight_strength", self.traffic_headlight_var),
            ("traffic_headlight_count", self.traffic_headlight_count_var),
            ("traffic_headlight_1_warmth", self.traffic_headlight_1_warmth_var),
            ("traffic_headlight_1_r", self.traffic_headlight_1_r_var),
            ("traffic_headlight_1_g", self.traffic_headlight_1_g_var),
            ("traffic_headlight_1_b", self.traffic_headlight_1_b_var),
            ("traffic_headlight_1_cone", self.traffic_headlight_1_cone_var),
            ("traffic_headlight_1_source_radius", self.traffic_headlight_1_source_radius_var),
            ("traffic_headlight_source_x", self.traffic_headlight_source_x_var),
            ("traffic_headlight_source_y", self.traffic_headlight_source_y_var),
            ("traffic_headlight_target_x", self.traffic_headlight_target_x_var),
            ("traffic_headlight_target_y", self.traffic_headlight_target_y_var),
            ("traffic_headlight_source_world_x", self.traffic_headlight_source_world_x_var),
            ("traffic_headlight_source_world_y", self.traffic_headlight_source_world_y_var),
            ("traffic_headlight_source_world_z", self.traffic_headlight_source_world_z_var),
            ("traffic_headlight_target_world_x", self.traffic_headlight_target_world_x_var),
            ("traffic_headlight_target_world_y", self.traffic_headlight_target_world_y_var),
            ("traffic_headlight_2_strength", self.traffic_headlight_2_var),
            ("traffic_headlight_2_warmth", self.traffic_headlight_2_warmth_var),
            ("traffic_headlight_2_r", self.traffic_headlight_2_r_var),
            ("traffic_headlight_2_g", self.traffic_headlight_2_g_var),
            ("traffic_headlight_2_b", self.traffic_headlight_2_b_var),
            ("traffic_headlight_2_cone", self.traffic_headlight_2_cone_var),
            ("traffic_headlight_2_source_radius", self.traffic_headlight_2_source_radius_var),
            ("traffic_headlight_2_source_x", self.traffic_headlight_2_source_x_var),
            ("traffic_headlight_2_source_y", self.traffic_headlight_2_source_y_var),
            ("traffic_headlight_2_target_x", self.traffic_headlight_2_target_x_var),
            ("traffic_headlight_2_target_y", self.traffic_headlight_2_target_y_var),
            ("traffic_headlight_2_source_world_x", self.traffic_headlight_2_source_world_x_var),
            ("traffic_headlight_2_source_world_y", self.traffic_headlight_2_source_world_y_var),
            ("traffic_headlight_2_source_world_z", self.traffic_headlight_2_source_world_z_var),
            ("traffic_headlight_2_target_world_x", self.traffic_headlight_2_target_world_x_var),
            ("traffic_headlight_2_target_world_y", self.traffic_headlight_2_target_world_y_var),
            ("traffic_headlight_3_strength", self.traffic_headlight_3_var),
            ("traffic_headlight_3_warmth", self.traffic_headlight_3_warmth_var),
            ("traffic_headlight_3_r", self.traffic_headlight_3_r_var),
            ("traffic_headlight_3_g", self.traffic_headlight_3_g_var),
            ("traffic_headlight_3_b", self.traffic_headlight_3_b_var),
            ("traffic_headlight_3_cone", self.traffic_headlight_3_cone_var),
            ("traffic_headlight_3_source_radius", self.traffic_headlight_3_source_radius_var),
            ("traffic_headlight_3_source_x", self.traffic_headlight_3_source_x_var),
            ("traffic_headlight_3_source_y", self.traffic_headlight_3_source_y_var),
            ("traffic_headlight_3_target_x", self.traffic_headlight_3_target_x_var),
            ("traffic_headlight_3_target_y", self.traffic_headlight_3_target_y_var),
            ("traffic_headlight_3_source_world_x", self.traffic_headlight_3_source_world_x_var),
            ("traffic_headlight_3_source_world_y", self.traffic_headlight_3_source_world_y_var),
            ("traffic_headlight_3_source_world_z", self.traffic_headlight_3_source_world_z_var),
            ("traffic_headlight_3_target_world_x", self.traffic_headlight_3_target_world_x_var),
            ("traffic_headlight_3_target_world_y", self.traffic_headlight_3_target_world_y_var),
            ("wet_mud_gloss_strength", self.wet_mud_gloss_var),
            ("water_film_strength", self.water_film_var),
            ("water_film_unevenness", self.water_film_unevenness_var),
            ("water_film_lens_strength", self.water_film_lens_var),
            ("water_film_contour_response", self.water_film_contour_var),
            ("water_film_gloss_strength", self.water_film_gloss_var),
            ("flare_strength", self.flare_var),
            ("overexposure_strength", self.overexposure_var),
            ("dirt_streak_strength", self.dirt_streak_var),
            ("dirt_flow_points", self.dirt_flow_points_var),
            ("dirt_flow_mass_min", self.dirt_flow_mass_min_var),
            ("dirt_flow_mass_max", self.dirt_flow_mass_max_var),
            ("dirt_flow_splash_scale", self.dirt_flow_splash_scale_var),
            ("dirt_flow_trail_length", self.dirt_flow_trail_length_var),
            ("dirt_flow_humidity", self.dirt_flow_humidity_var),
            ("dirt_flow_stickiness_min", self.dirt_flow_stickiness_min_var),
            ("dirt_flow_stickiness_max", self.dirt_flow_stickiness_max_var),
            ("dirt_flow_air_angle", self.dirt_flow_air_angle_var),
            ("dirt_flow_wind_strength", self.dirt_flow_wind_strength_var),
            ("dirt_flow_opacity_min", self.dirt_flow_opacity_min_var),
            ("dirt_flow_opacity_max", self.dirt_flow_opacity_max_var),
            ("dirt_flow_stop_on_dark_contour", self.dirt_flow_stop_on_contour_var),
            ("contour_detection_sensitivity", self.contour_detection_sensitivity_var),
            ("dark_relief_strength", self.dark_relief_var),
            ("dark_relief_light_angle", self.dark_relief_light_angle_var),
            ("relief_profile_preset", self.relief_profile_preset_var),
            ("relief_bounce_depth", self.relief_bounce_depth_var),
            ("relief_bounce_strength", self.relief_bounce_strength_var),
            ("overhang_shadow_strength", self.overhang_shadow_var),
            ("overhang_shadow_depth", self.overhang_shadow_depth_var),
            ("overhang_shadow_skew", self.overhang_shadow_skew_var),
            ("plate_reflect_gradient_strength", self.plate_reflect_gradient_var),
            ("plate_reflect_glare_strength", self.plate_reflect_glare_var),
            ("plate_reflect_curve_strength", self.plate_reflect_curve_var),
            ("blur_strength", self.blur_strength_var),
            ("class_name", self.class_var),
        )

    def _apply_profile_to_vars(self, profile: AugmentationProfile) -> None:
        profile = (profile or AugmentationProfile(task_target=self.target)).normalized()
        self._preview_refresh_suspended = True
        try:
            for attr, var in self._profile_var_bindings():
                if not hasattr(profile, attr):
                    continue
                value = getattr(profile, attr)
                try:
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(value))
                    elif isinstance(var, tk.IntVar):
                        var.set(int(float(value or 0)))
                    elif isinstance(var, tk.DoubleVar):
                        var.set(float(value or 0.0))
                    else:
                        var.set(str(value or ""))
                except Exception:
                    pass
            self.blur_var.set(float(getattr(profile, "blur_strength", 0.0) or 0.0) > 0.001 or bool(getattr(profile, "blur_enabled", False)))
            self.traffic_headlight_1_enabled_var.set(float(getattr(profile, "traffic_headlight_strength", 0.0) or 0.0) > 0.001)
            self.traffic_headlight_2_enabled_var.set(float(getattr(profile, "traffic_headlight_2_strength", 0.0) or 0.0) > 0.001)
            self.traffic_headlight_3_enabled_var.set(float(getattr(profile, "traffic_headlight_3_strength", 0.0) or 0.0) > 0.001)
            self.relief_profile_preset_var.set(normalize_relief_profile_preset(getattr(profile, "relief_profile_preset", DEFAULT_RELIEF_PROFILE_PRESET)))
            self._relief_profile_curve = normalize_relief_profile_curve(getattr(profile, "relief_profile_curve", DEFAULT_RELIEF_PROFILE_CURVE))
            self._invalidate_contour_wireframe_cache()
            self._effect_seed = int(getattr(profile, "seed", 42) or 42)
            self._manual_randomness = normalize_manual_randomness_config(
                getattr(profile, "manual_randomness", {}),
                self.target,
            )
            self.randomness_mode_var.set(self._randomness_mode_to_label(getattr(profile, "randomness_mode", "realistic")))
        finally:
            self._preview_refresh_suspended = False
        self._refresh_extra_count_title()
        self._refresh_scene_summary()
        self._refresh_preview(redraw_only=True)

    def _augmentation_preset_initial_dir(self) -> Path:
        try:
            path = get_augmentation_presets_dir(self.target)
        except Exception:
            path = Path(CONFIG.WORKSPACE_DIR) / "8_presets" / "augmentation" / self.target
        path = Path(path)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            return path.resolve()
        except Exception:
            return path.absolute()

    def _load_augmentation_profile_from_file(self):
        initial_dir = self._augmentation_preset_initial_dir()
        path = filedialog.askopenfilename(
            parent=self.window,
            title="Wczytaj preset augmentacji",
            initialdir=str(initial_dir),
            initialfile="",
            filetypes=(
                ("JSON", "*.json"),
                ("Wszystkie pliki", "*.*"),
            ),
        )
        if not path:
            return
        try:
            profile = load_augmentation_preset(Path(path), target=self.target)
        except Exception as exc:
            messagebox.showerror("Nie mozna wczytac presetu", f"Plik nie zawiera poprawnego presetu augmentacji:\n{exc}", parent=self.window)
            return
        self._apply_profile_to_vars(profile)
        messagebox.showinfo("Preset wczytany", "Ustawienia augmentacji zostaly wczytane do modala.", parent=self.window)

    def _save_augmentation_profile_to_preset(self):
        default_name = f"aug_{self.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        name = simpledialog.askstring(
            "Zapisz preset augmentacji",
            "Nazwa presetu:",
            initialvalue=default_name,
            parent=self.window,
        )
        if not name:
            return
        try:
            path = save_augmentation_preset(
                self._profile_from_vars(),
                name,
                target=self.target,
            )
        except Exception as exc:
            messagebox.showerror("Nie mozna zapisac presetu", f"Nie udalo sie zapisac presetu augmentacji:\n{exc}", parent=self.window)
            return
        messagebox.showinfo("Preset zapisany", f"Preset zapisano w:\n{path}", parent=self.window)

    def _domain_intro_text(self) -> str:
        if self.target != "plate":
            return "Profil: tablica/znaki. Priorytet: deszcz, błoto, relief, lokalne światło."
        if self.target == "plate":
            return "Profil: zdjęcia pojazdu. Priorytet: scena, deszcz, światło."
        return "Profil: tablica/znaki. Priorytet: błoto, relief, lokalne światło."

    def _domain_info_text(self) -> str:
        if self.target != "plate":
            return "Wycięte tablice i znaki. Efekty lokalne oraz ten sam model deszczu co w torze tablic."
        if self.target == "plate":
            return "Całe zdjęcia pojazdu. Efekty sceny bez lokalnego błota tablicy."
        return "Wycięte tablice i znaki. Efekty lokalne bez deszczu sceny."

    def _domain_toolboxes(self) -> list[tuple[str, str]]:
        if self.target != "plate":
            return [
                ("geometry", "Geometria"),
                ("color", "Kolor"),
                ("weather", "Pogoda"),
                ("material", "Materiał"),
                ("relief", "Kontury"),
                ("dirt", "Błoto"),
                ("illumination", "Światło"),
                ("sensor", "Kamera"),
            ]
        return [
            ("geometry", "Geometria"),
            ("color", "Kolor"),
            ("weather", "Pogoda"),
            ("illumination", "Światło"),
            ("sensor", "Kamera"),
        ]

    def _normalize_toolbox_key(self, key: object) -> str:
        value = str(key or "").strip().lower()
        aliases = {
            "light": "illumination",
            "swiatlo": "illumination",
            "światło": "illumination",
            "kontury": "relief",
            "relief": "relief",
            "zabrudzenia": "dirt",
            "brud": "dirt",
            "bloto": "dirt",
            "błoto": "dirt",
            "kamera": "sensor",
            "optyka": "sensor",
        }
        return aliases.get(value, value)

    def _build_effect_inspector(self, parent):
        inspector = ttk.LabelFrame(parent, text=" Inspektor efektu ", padding=10)
        self.effect_inspector_shell = inspector
        inspector.grid(row=0, column=1, sticky=tk.NSEW, padx=(10, 0))
        inspector.grid_propagate(False)
        try:
            inspector.configure(width=340)
        except Exception:
            pass
        inspector.columnconfigure(0, weight=1)
        inspector.rowconfigure(4, weight=1)

        header = ttk.Frame(inspector)
        header.grid(row=0, column=0, sticky=tk.EW)
        header.columnconfigure(0, weight=1)
        self.effect_inspector_title_var = tk.StringVar(value=self._active_toolbox_title())
        ttk.Label(
            header,
            textvariable=self.effect_inspector_title_var,
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(header, text="×", width=3, command=self._hide_effect_inspector).grid(row=0, column=1, sticky=tk.E)
        ttk.Label(
            inspector,
            text="Wybierz efekt, a jego ustawienia pojawią się tutaj. Podgląd pozostaje odsłonięty.",
            justify=tk.LEFT,
            wraplength=300,
        ).grid(row=1, column=0, sticky=tk.EW, pady=(4, 10))

        self.toolbox_bar = ttk.Frame(inspector)
        self.toolbox_bar.grid(row=2, column=0, sticky=tk.EW, pady=(0, 10))
        ttk.Separator(inspector, orient=tk.HORIZONTAL).grid(row=3, column=0, sticky=tk.EW, pady=(0, 8))

        content_host = ttk.Frame(inspector)
        content_host.grid(row=4, column=0, sticky=tk.NSEW)
        content_host.columnconfigure(0, weight=1)
        content_host.rowconfigure(0, weight=1)
        content_canvas = tk.Canvas(content_host, highlightthickness=0, bd=0)
        content_scrollbar = ttk.Scrollbar(content_host, orient=tk.VERTICAL, command=content_canvas.yview)
        content_canvas.configure(yscrollcommand=content_scrollbar.set)
        content_canvas.grid(row=0, column=0, sticky=tk.NSEW)
        content_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.toolbox_body_canvas = content_canvas
        self.toolbox_body = ttk.Frame(content_canvas)
        self.toolbox_body.columnconfigure(0, weight=1)
        self.toolbox_body_window = content_canvas.create_window((0, 0), window=self.toolbox_body, anchor=tk.NW)

        def refresh_scroll(_event=None):
            try:
                content_canvas.configure(scrollregion=content_canvas.bbox("all"))
            except Exception:
                pass

        def resize_body(event):
            try:
                content_canvas.itemconfigure(self.toolbox_body_window, width=max(1, int(event.width)))
            except Exception:
                pass

        def scroll_inspector(event):
            try:
                if getattr(event, "num", None) == 4:
                    units = -3
                elif getattr(event, "num", None) == 5:
                    units = 3
                else:
                    delta = int(getattr(event, "delta", 0) or 0)
                    units = -1 * int(delta / 120) if delta else 0
                if units:
                    content_canvas.yview_scroll(units, "units")
                    return "break"
            except Exception:
                return None
            return None

        def bind_scroll_tree(widget):
            try:
                for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    widget.bind(sequence, scroll_inspector, add="+")
                for child in widget.winfo_children():
                    bind_scroll_tree(child)
            except Exception:
                pass

        self._effect_inspector_scroll_handler = scroll_inspector
        self.toolbox_body.bind("<Configure>", refresh_scroll, add="+")
        content_canvas.bind("<Configure>", resize_body, add="+")
        try:
            self.window.after_idle(self._bind_effect_inspector_scroll_tree)
        except Exception:
            pass
        self._build_toolbox_bar()
        self._render_active_toolbox()

    def _show_effect_inspector(self, *, refresh: bool = True):
        self._effect_inspector_visible = True
        if bool(getattr(self, "_preview_fullscreen", False)):
            if refresh:
                self._draw_vector_overlay()
            return
        shell = getattr(self, "effect_inspector_shell", None)
        if shell is not None:
            try:
                shell.grid()
            except Exception:
                pass
        if refresh:
            self._render_active_toolbox()
            self._refresh_preview(redraw_only=True)

    def _hide_effect_inspector(self):
        self._effect_inspector_visible = False
        shell = getattr(self, "effect_inspector_shell", None)
        if shell is not None:
            try:
                shell.grid_remove()
            except Exception:
                pass
        self._refresh_preview(redraw_only=True)

    def _refresh_effect_inspector_title(self):
        try:
            self.effect_inspector_title_var.set(self._active_toolbox_title())
        except Exception:
            pass

    def _bind_effect_inspector_scroll_tree(self, widget=None):
        handler = getattr(self, "_effect_inspector_scroll_handler", None)
        if handler is None:
            return
        if widget is None:
            widget = getattr(self, "effect_inspector_shell", None)
        if widget is None:
            return
        try:
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(sequence, handler, add="+")
            for child in widget.winfo_children():
                self._bind_effect_inspector_scroll_tree(child)
        except Exception:
            pass

    def _build_toolbox_bar(self):
        self._toolbox_buttons = {}
        for child in self.toolbox_bar.winfo_children():
            child.destroy()
        for column in (0, 1, 2):
            try:
                self.toolbox_bar.columnconfigure(column, weight=1, uniform="effect_tabs")
            except Exception:
                pass
        for index, (key, label) in enumerate(self._domain_toolboxes()):
            button = tk.Canvas(
                self.toolbox_bar,
                bd=1,
                relief=tk.FLAT,
                width=104,
                height=40,
                highlightthickness=1,
                cursor="hand2",
            )
            button.grid(row=index // 3, column=index % 3, sticky=tk.EW, padx=2, pady=2)
            button.bind("<Button-1>", lambda _event, selected=key: self._select_toolbox(selected), add="+")
            button.bind("<Enter>", lambda _event, selected=key: self._set_toolbox_tab_hover(selected, True), add="+")
            button.bind("<Leave>", lambda _event, selected=key: self._set_toolbox_tab_hover(selected, False), add="+")
            button.bind("<Configure>", lambda _event, selected=key: self._draw_toolbox_tab(selected), add="+")
            self._toolbox_buttons[key] = button
        self._refresh_toolbox_tab_labels()

    def _toolbox_tab_colors(self, key: str, *, hover: bool = False) -> dict:
        active = self._normalize_toolbox_key(key) == self._normalize_toolbox_key(self._active_toolbox)
        if active:
            return {
                "bg": "#155f3b",
                "fg": "#f3fff7",
                "highlightbackground": "#1fb96a",
            }
        if hover:
            return {
                "bg": "#e7f5ec",
                "fg": "#123322",
                "highlightbackground": "#66c98a",
            }
        return {
            "bg": "#f4f8f5",
            "fg": "#22322a",
            "highlightbackground": "#b7c7bd",
        }

    def _draw_toolbox_icon(self, canvas, key: str, cx: int, cy: int, color: str, muted: str):
        key = self._normalize_toolbox_key(key)
        try:
            if key == "geometry":
                canvas.create_rectangle(cx - 10, cy - 7, cx + 9, cy + 7, outline=color, width=2)
                for px, py in ((cx - 10, cy - 7), (cx + 9, cy - 7), (cx + 9, cy + 7), (cx - 10, cy + 7)):
                    canvas.create_oval(px - 2, py - 2, px + 2, py + 2, fill=color, outline="")
                canvas.create_line(cx - 13, cy + 11, cx + 13, cy + 11, fill=muted, width=1, arrow=tk.LAST)
                return
            if key == "color":
                for offset, fill in ((-8, "#e55353"), (0, "#42c56f"), (8, "#4a8df0")):
                    canvas.create_rectangle(cx + offset - 3, cy - 9, cx + offset + 3, cy + 9, fill=fill, outline="")
                canvas.create_rectangle(cx - 13, cy - 11, cx + 13, cy + 11, outline=color, width=1)
                return
            if key == "weather":
                canvas.create_arc(cx - 12, cy - 9, cx + 3, cy + 7, start=40, extent=230, outline=color, width=2, style=tk.ARC)
                canvas.create_arc(cx - 2, cy - 9, cx + 13, cy + 7, start=-90, extent=230, outline=color, width=2, style=tk.ARC)
                for offset in (-8, 0, 8):
                    canvas.create_line(cx + offset - 2, cy + 8, cx + offset - 6, cy + 15, fill=muted, width=2)
                return
            if key == "material":
                canvas.create_rectangle(cx - 12, cy - 8, cx + 12, cy + 8, outline=color, width=2)
                canvas.create_line(cx - 9, cy + 1, cx - 4, cy - 2, cx + 1, cy + 2, cx + 7, cy - 1, fill=muted, width=2, smooth=True)
                canvas.create_line(cx - 10, cy + 12, cx + 10, cy + 12, fill=color, width=1)
                return
            if key == "relief":
                canvas.create_line(cx - 11, cy + 8, cx - 5, cy - 8, cx + 2, cy + 8, cx + 8, cy - 8, fill=color, width=2, smooth=True)
                for offset in (-7, -2, 3, 8):
                    canvas.create_line(cx + offset, cy - 6, cx + offset - 3, cy + 7, fill=muted, width=1)
                return
            if key == "dirt":
                for dx, dy, radius in ((-7, -4, 4), (1, 1, 6), (9, -5, 3), (-2, 9, 3)):
                    canvas.create_oval(cx + dx - radius, cy + dy - radius, cx + dx + radius, cy + dy + radius, fill=color if radius >= 4 else muted, outline="")
                canvas.create_line(cx - 9, cy + 10, cx + 10, cy + 14, fill=muted, width=2, smooth=True)
                return
            if key == "illumination":
                canvas.create_oval(cx - 13, cy - 5, cx - 5, cy + 5, fill=color, outline="")
                canvas.create_polygon(cx - 3, cy - 8, cx + 14, cy - 15, cx + 14, cy + 15, cx - 3, cy + 8, fill="", outline=muted, width=2)
                canvas.create_line(cx - 3, cy, cx + 12, cy, fill=color, width=1, arrow=tk.LAST)
                return
            if key == "sensor":
                canvas.create_rectangle(cx - 13, cy - 8, cx + 12, cy + 9, outline=color, width=2)
                canvas.create_oval(cx - 6, cy - 6, cx + 7, cy + 7, outline=muted, width=2)
                canvas.create_oval(cx - 1, cy - 1, cx + 2, cy + 2, fill=color, outline="")
                canvas.create_rectangle(cx - 8, cy - 12, cx + 2, cy - 8, fill=color, outline="")
                return
            canvas.create_oval(cx - 10, cy - 10, cx + 10, cy + 10, outline=color, width=2)
        except Exception:
            pass

    def _draw_toolbox_tab(self, key: str, *, hover: bool = False):
        button = getattr(self, "_toolbox_buttons", {}).get(key)
        if button is None:
            return
        colors = self._toolbox_tab_colors(key, hover=hover)
        label = next((text for value, text in self._domain_toolboxes() if value == key), key)
        try:
            width = max(88, int(button.winfo_width() or button.cget("width") or 88))
        except Exception:
            width = 100
        try:
            height = max(36, int(button.winfo_height() or button.cget("height") or 40))
        except Exception:
            height = 40
        icon_color = "#f3fff7" if self._normalize_toolbox_key(key) == self._normalize_toolbox_key(self._active_toolbox) else "#155f3b"
        muted = "#9ff0bc" if self._normalize_toolbox_key(key) == self._normalize_toolbox_key(self._active_toolbox) else "#5c8f70"
        if hover and self._normalize_toolbox_key(key) != self._normalize_toolbox_key(self._active_toolbox):
            icon_color = "#0f6a3d"
            muted = "#2f9e60"
        try:
            button.configure(bg=colors["bg"], highlightbackground=colors["highlightbackground"])
            button.delete("all")
            button.create_rectangle(0, 0, width - 1, height - 1, fill=colors["bg"], outline=colors["highlightbackground"], width=1)
            self._draw_toolbox_icon(button, key, 18, height // 2, icon_color, muted)
            button.create_text(
                max(50, width // 2 + 13),
                height // 2,
                text=label,
                fill=colors["fg"],
                font=("Segoe UI", 8, "bold"),
                anchor=tk.CENTER,
                width=max(42, width - 44),
            )
        except Exception:
            pass

    def _refresh_toolbox_tab_labels(self):
        for key, button in getattr(self, "_toolbox_buttons", {}).items():
            self._draw_toolbox_tab(key)

    def _set_toolbox_tab_hover(self, key: str, hover: bool):
        button = getattr(self, "_toolbox_buttons", {}).get(key)
        if button is None or key == self._active_toolbox:
            return
        self._draw_toolbox_tab(key, hover=hover)

    def _select_toolbox(self, key: str):
        key = self._normalize_toolbox_key(key)
        previous_key = self._normalize_toolbox_key(getattr(self, "_active_toolbox", ""))
        inspector_was_hidden = not bool(getattr(self, "_effect_inspector_visible", True))
        if bool(getattr(self, "_preview_fullscreen", False)):
            self._toolbox_panel_visible = True
            self._toolbox_panel_collapsed = False
            self._scene_projection_expanded = False
            if key != previous_key:
                self._toolbox_panel_scroll = 0.0
        scene_visibility_changed = False
        desired_scene_visible = True
        if bool(getattr(self, "_scene_viewport_visible", False)) != desired_scene_visible:
            self._scene_viewport_visible = desired_scene_visible
            scene_visibility_changed = True
        if key == self._active_toolbox and not inspector_was_hidden:
            self._pending_inspector_scroll_anchor = "__top__"
            self._refresh_toolbox_tab_labels()
            self._refresh_effect_inspector_title()
            self._render_active_toolbox()
            if bool(getattr(self, "_preview_fullscreen", False)):
                self._draw_vector_overlay()
            elif scene_visibility_changed:
                self._refresh_preview(redraw_only=True)
            return
        self._active_toolbox = key
        self._pending_inspector_scroll_anchor = "__top__"
        if inspector_was_hidden:
            self._show_effect_inspector(refresh=False)
        self._refresh_toolbox_tab_labels()
        self._refresh_effect_inspector_title()
        self._render_active_toolbox()
        self._refresh_preview(redraw_only=True)

    def _render_active_toolbox(self):
        body = getattr(self, "toolbox_body", None)
        if body is None:
            return
        for child in body.winfo_children():
            child.destroy()
        self._render_effect_inspector_body(body)
        try:
            self.window.after_idle(self._bind_effect_inspector_scroll_tree)
        except Exception:
            pass
        try:
            self.window.after_idle(self._bind_fullscreen_preview_tab_toggle)
        except Exception:
            pass
        return
        shell = ttk.LabelFrame(body, text=f" {self._active_toolbox_title()} ", padding=4)
        shell.grid(row=0, column=0, sticky=tk.EW)
        for column in (1, 3, 5):
            shell.columnconfigure(column, weight=1)

        fields = self._toolbox_fields(self._active_toolbox)
        row = 0
        col = 0
        for field in fields:
            if field.get("type") == "range_slider" and col != 0:
                col = 0
                row += 1
            if field.get("type") == "check":
                ttk.Checkbutton(shell, text=str(field["label"]), variable=field["var"]).grid(
                    row=row,
                    column=col,
                    columnspan=2,
                    sticky=tk.W,
                    padx=(0, 12),
                    pady=2,
                )
            elif field.get("type") == "range":
                self._add_toolbox_range(
                    shell,
                    row,
                    col,
                    str(field["label"]),
                    field["min_var"],
                    field["max_var"],
                    field["from"],
                    field["to"],
                    field["step"],
                    field.get("width", 6),
                )
            elif field.get("type") == "range_slider":
                self._add_toolbox_range_slider(
                    shell,
                    row,
                    0,
                    str(field["label"]),
                    field["min_var"],
                    field["max_var"],
                    field["from"],
                    field["to"],
                    field["step"],
                )
                row += 1
                col = 0
                continue
            elif field.get("type") == "slider":
                self._add_toolbox_slider(
                    shell,
                    row,
                    col,
                    str(field["label"]),
                    field["var"],
                    field["from"],
                    field["to"],
                    field["step"],
                )
            else:
                self._add_toolbox_spin(
                    shell,
                    row,
                    col,
                    str(field["label"]),
                    field["var"],
                    field["from"],
                    field["to"],
                    field["step"],
                    field.get("width", 7),
                )
            col += 2
            if col >= 6:
                col = 0
                row += 1
        if fields:
            return
        ttk.Label(shell, text="Brak ustawień dla tej domeny.").grid(row=0, column=0, sticky=tk.W)

    def _render_effect_inspector_body(self, body):
        self._refresh_effect_inspector_title()
        self._inspector_anchor_widgets = {}
        active_toolbox = self._normalize_toolbox_key(self._active_toolbox)
        shell = ttk.Frame(body, padding=(2, 0, 8, 8))
        shell.grid(row=0, column=0, sticky=tk.EW)
        shell.columnconfigure(0, weight=1)
        fields = self._toolbox_fields(active_toolbox)
        self._inspector_field_hints = {
            str(field.get("label", "")): str(field.get("hint", "")).strip()
            for field in fields
            if str(field.get("hint", "")).strip()
        }
        if active_toolbox == "illumination":
            self._inspector_field_hints.update(
                {
                    "Promień źródła": "Rozmiar tarczy reflektora. Daje wyraźny, miękki efekt tylko wtedy, gdy źródło jest blisko powierzchni tablicy; przy większym Z wpływ promienia szybko zanika.",
                }
            )
        row = 0
        for field in fields:
            ftype = field.get("type")
            if ftype == "section":
                self._add_inspector_section_header(
                    shell,
                    row,
                    str(field["label"]),
                    str(field.get("text", "")),
                    bool(field.get("priority", False)),
                )
            elif ftype == "check":
                self._add_inspector_check(shell, row, str(field["label"]), field["var"])
            elif ftype == "range":
                self._add_inspector_range(shell, row, str(field["label"]), field["min_var"], field["max_var"], field["from"], field["to"], field["step"], field.get("width", 8))
            elif ftype == "range_slider":
                self._add_inspector_range_slider(shell, row, str(field["label"]), field["min_var"], field["max_var"], field["from"], field["to"], field["step"])
            elif ftype == "profile_curve":
                self._add_inspector_relief_profile_editor(shell, row, str(field["label"]))
            elif ftype == "slider":
                self._add_inspector_slider(shell, row, str(field["label"]), field["var"], field["from"], field["to"], field["step"])
            elif ftype == "xyz":
                self._add_inspector_xyz(shell, row, str(field["label"]), field["vars"], field.get("ranges"), field.get("step", 0.01))
            elif ftype == "xy_fixed_z":
                self._add_inspector_xy_fixed_z(shell, row, str(field["label"]), field["vars"], field.get("ranges"), field.get("step", 0.01), field.get("fixed_z", 0.0))
            else:
                self._add_inspector_spin(shell, row, str(field["label"]), field["var"], field["from"], field["to"], field["step"], field.get("width", 8))
            row += 1
        if active_toolbox == "illumination":
            row = self._render_headlight_inspector_section(shell, row)
        self._schedule_pending_inspector_scroll()
        if fields or active_toolbox == "illumination":
            return
        ttk.Label(shell, text="Brak ustawień dla tej domeny.").grid(row=row, column=0, sticky=tk.W)

    def _schedule_pending_inspector_scroll(self):
        anchor = getattr(self, "_pending_inspector_scroll_anchor", None)
        if not anchor:
            return
        try:
            self.window.after_idle(lambda expected=anchor: self._apply_pending_inspector_scroll(expected))
        except Exception:
            self._apply_pending_inspector_scroll(anchor)

    def _apply_pending_inspector_scroll(self, expected: str):
        if getattr(self, "_pending_inspector_scroll_anchor", None) != expected:
            return
        self._pending_inspector_scroll_anchor = None
        canvas = getattr(self, "toolbox_body_canvas", None)
        if canvas is None:
            return
        if expected == "__top__":
            try:
                canvas.yview_moveto(0)
            except Exception:
                pass
            return
        widget = getattr(self, "_inspector_anchor_widgets", {}).get(expected)
        body = getattr(self, "toolbox_body", None)
        if widget is None or body is None:
            return
        try:
            self.window.update_idletasks()
        except Exception:
            pass
        try:
            y = 0
            current = widget
            while current is not None and current is not body:
                y += int(current.winfo_y() or 0)
                current = current.master
            bbox = canvas.bbox("all") or (0, 0, 1, 1)
            total_height = max(1, int(bbox[3] - bbox[1]))
            fraction = max(0.0, min(1.0, (float(y) - 6.0) / float(total_height)))
            canvas.yview_moveto(fraction)
        except Exception:
            pass

    def _add_inspector_section_header(self, parent, row: int, label: str, text: str = "", priority: bool = False):
        bg = "#e8f7ed" if priority else "#f2f6ef"
        outline = "#35a866" if priority else "#86b982"
        fg = "#0f5b31" if priority else "#2d5a35"
        frame = tk.Frame(
            parent,
            bg=bg,
            highlightthickness=1,
            highlightbackground=outline,
            padx=9,
            pady=7,
        )
        frame.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=(12 if row else 2, 6))
        frame.columnconfigure(0, weight=1)
        tk.Label(
            frame,
            text=label,
            bg=bg,
            fg=fg,
            font=("Segoe UI", 9, "bold"),
            anchor=tk.W,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky=tk.EW)
        if text:
            tk.Label(
                frame,
                text=text,
                bg=bg,
                fg="#31443a",
                font=("Segoe UI", 8),
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=290,
            ).grid(row=1, column=0, sticky=tk.EW, pady=(3, 0))
        return frame

    def _tk_color_to_hex(self, color: object, fallback: str = "#252526") -> str:
        raw = str(color or "").strip() or fallback
        try:
            r, g, b = self.window.winfo_rgb(raw)
            return f"#{r // 256:02x}{g // 256:02x}{b // 256:02x}"
        except Exception:
            return fallback

    def _blend_hex_colors(self, first: str, second: str, ratio: float) -> str:
        ratio = max(0.0, min(1.0, float(ratio)))
        first = self._tk_color_to_hex(first)
        second = self._tk_color_to_hex(second)
        try:
            a = tuple(int(first[index:index + 2], 16) for index in (1, 3, 5))
            b = tuple(int(second[index:index + 2], 16) for index in (1, 3, 5))
            mixed = tuple(int(round(a[i] * ratio + b[i] * (1.0 - ratio))) for i in range(3))
            return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"
        except Exception:
            return first

    def _inspector_setting_colors(self, parent) -> dict[str, str]:
        style = ttk.Style(self.window)
        style_bg = style.lookup("TFrame", "background") or style.lookup("TLabel", "background") or "#252526"
        style_fg = style.lookup("TLabel", "foreground") or "#f3f3f3"
        try:
            parent_bg = parent.cget("background")
        except Exception:
            parent_bg = style_bg
        panel = self._tk_color_to_hex(parent_bg or style_bg, "#252526")
        fg = self._tk_color_to_hex(style_fg, "#f3f3f3")
        accent = "#4ec980"
        return {
            "border": self._blend_hex_colors(accent, panel, 0.54),
            "body": self._blend_hex_colors(panel, accent, 0.90),
            "header": self._blend_hex_colors(panel, accent, 0.78),
            "fg": fg,
            "accent": self._blend_hex_colors(accent, fg, 0.82),
        }

    def _inspector_hint_for_label(self, label: str) -> str:
        try:
            return str(getattr(self, "_inspector_field_hints", {}).get(str(label), "")).strip()
        except Exception:
            return ""

    def _cancel_inspector_tooltip_after(self) -> None:
        after_id = getattr(self, "_inspector_tooltip_after_id", None)
        if not after_id:
            return
        self._inspector_tooltip_after_id = None
        try:
            self.window.after_cancel(after_id)
        except Exception:
            pass

    def _hide_inspector_tooltip(self) -> None:
        self._cancel_inspector_tooltip_after()
        tooltip = getattr(self, "_inspector_tooltip_window", None)
        self._inspector_tooltip_window = None
        if tooltip is None:
            return
        try:
            tooltip.destroy()
        except Exception:
            pass

    def _show_inspector_tooltip(self, widget, text: str) -> None:
        text = str(text or "").strip()
        if not text:
            return
        self._hide_inspector_tooltip()
        try:
            tooltip = tk.Toplevel(self.window)
            tooltip.wm_overrideredirect(True)
            tooltip.configure(bg="#4ec980")
            tooltip.attributes("-topmost", True)
            body = tk.Message(
                tooltip,
                text=text,
                width=285,
                bg="#0f1a16",
                fg="#eafff2",
                font=("Segoe UI", 9),
                padx=10,
                pady=8,
                justify=tk.LEFT,
            )
            body.pack(padx=1, pady=1)
            try:
                x = int(widget.winfo_rootx() + widget.winfo_width() + 10)
                y = int(widget.winfo_rooty() - 2)
                screen_w = int(widget.winfo_screenwidth() or 0)
                screen_h = int(widget.winfo_screenheight() or 0)
                if screen_w and x + 315 > screen_w:
                    x = max(8, int(widget.winfo_rootx() - 315))
                if screen_h and y + 120 > screen_h:
                    y = max(8, screen_h - 130)
                tooltip.geometry(f"+{x}+{y}")
            except Exception:
                pass
            self._inspector_tooltip_window = tooltip
        except Exception:
            self._inspector_tooltip_window = None

    def _schedule_inspector_tooltip(self, widget, text: str) -> None:
        self._cancel_inspector_tooltip_after()
        try:
            self._inspector_tooltip_after_id = self.window.after(
                260,
                lambda target=widget, value=text: self._show_inspector_tooltip(target, value),
            )
        except Exception:
            self._show_inspector_tooltip(widget, text)

    def _bind_inspector_hint(self, widget, text: str) -> None:
        text = str(text or "").strip()
        if not text:
            return
        try:
            widget.bind("<Enter>", lambda _event, target=widget, value=text: self._schedule_inspector_tooltip(target, value), add="+")
            widget.bind("<Leave>", lambda _event: self._hide_inspector_tooltip(), add="+")
            widget.bind("<ButtonPress-1>", lambda _event, target=widget, value=text: self._show_inspector_tooltip(target, value), add="+")
        except Exception:
            pass

    def _add_inspector_row(self, parent, row: int, label: str):
        colors = self._inspector_setting_colors(parent)
        outer = tk.Frame(
            parent,
            bg=colors["border"],
            padx=1,
            pady=1,
            bd=0,
            highlightthickness=0,
        )
        outer.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=(7, 11))
        outer.columnconfigure(0, weight=1)
        frame = tk.Frame(outer, bg=colors["body"], padx=10, pady=9, bd=0, highlightthickness=0)
        frame.grid(row=0, column=0, sticky=tk.EW)
        frame.columnconfigure(0, weight=1)
        header = tk.Frame(frame, bg=colors["header"], bd=0, highlightthickness=0)
        header.grid(row=0, column=0, columnspan=3, sticky=tk.EW, pady=(0, 8))
        header.columnconfigure(0, weight=1)
        tk.Label(
            header,
            text=label,
            bg=colors["header"],
            fg=colors["accent"],
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=4,
            anchor=tk.W,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky=tk.EW)
        hint = self._inspector_hint_for_label(label)
        if hint:
            hint_badge = tk.Label(
                header,
                text="?",
                bg=self._blend_hex_colors(colors["header"], "#4ec980", 0.45),
                fg="#071218",
                font=("Segoe UI", 8, "bold"),
                width=2,
                cursor="hand2",
                padx=2,
                pady=1,
            )
            hint_badge.grid(row=0, column=1, sticky=tk.E, padx=(6, 8), pady=3)
            self._bind_inspector_hint(hint_badge, hint)
        return frame

    def _add_inspector_spin(self, parent, row: int, label: str, variable, from_, to, increment, width=8):
        frame = self._add_inspector_row(parent, row, label)
        ttk.Spinbox(
            frame,
            from_=from_,
            to=to,
            increment=increment,
            textvariable=variable,
            width=width,
        ).grid(row=1, column=0, sticky=tk.EW)

    def _add_inspector_xyz(self, parent, row: int, label: str, variables, ranges=None, increment=0.01):
        frame = self._add_inspector_row(parent, row, label)
        ranges = ranges or ((-3.0, 3.0), (-3.0, 3.0), (-1.0, 8.0))
        axes = ("X", "Y", "Z")
        for col in range(3):
            frame.columnconfigure(col, weight=1)
            cell = ttk.Frame(frame)
            cell.grid(row=1, column=col, sticky=tk.EW, padx=(0 if col == 0 else 6, 0))
            cell.columnconfigure(1, weight=1)
            ttk.Label(cell, text=axes[col], width=2, anchor=tk.W).grid(row=0, column=0, sticky=tk.W, padx=(0, 3))
            try:
                from_, to = ranges[col]
            except Exception:
                from_, to = (-3.0, 3.0)
            ttk.Spinbox(
                cell,
                from_=from_,
                to=to,
                increment=increment,
                textvariable=variables[col],
                width=7,
            ).grid(row=0, column=1, sticky=tk.EW)

    def _add_inspector_xy_fixed_z(self, parent, row: int, label: str, variables, ranges=None, increment=0.01, fixed_z: float = 0.0):
        frame = self._add_inspector_row(parent, row, label)
        ranges = ranges or ((-3.0, 3.0), (-3.0, 3.0))
        try:
            if len(variables) > 2:
                variables[2].set(float(fixed_z))
        except Exception:
            pass
        axes = ("X", "Y")
        for col in range(2):
            frame.columnconfigure(col, weight=1)
            cell = ttk.Frame(frame)
            cell.grid(row=1, column=col, sticky=tk.EW, padx=(0 if col == 0 else 6, 0))
            cell.columnconfigure(1, weight=1)
            ttk.Label(cell, text=axes[col], width=2, anchor=tk.W).grid(row=0, column=0, sticky=tk.W, padx=(0, 3))
            try:
                from_, to = ranges[col]
            except Exception:
                from_, to = (-3.0, 3.0)
            ttk.Spinbox(
                cell,
                from_=from_,
                to=to,
                increment=increment,
                textvariable=variables[col],
                width=7,
            ).grid(row=0, column=1, sticky=tk.EW)
        frame.columnconfigure(2, weight=0)
        ttk.Label(
            frame,
            text=f"Z = {self._format_scene_float(fixed_z, 0.0)}",
            anchor=tk.CENTER,
        ).grid(row=1, column=2, sticky=tk.EW, padx=(8, 0))

    def _add_inspector_check(self, parent, row: int, label: str, variable):
        frame = self._add_inspector_row(parent, row, label)
        ttk.Checkbutton(frame, text="Włącz", variable=variable).grid(row=1, column=0, sticky=tk.W)

    def _add_inspector_range(self, parent, row: int, label: str, min_var, max_var, from_, to, increment, width=8):
        self._add_inspector_range_slider(parent, row, label, min_var, max_var, from_, to, increment)

    def _add_inspector_slider(self, parent, row: int, label: str, variable, from_, to, increment):
        frame = self._add_inspector_row(parent, row, label)
        frame.columnconfigure(0, weight=1)
        value_lbl = ttk.Label(frame, text=self._format_toolbox_value(variable.get(), increment), width=7, anchor=tk.E)
        scale = ttk.Scale(
            frame,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=lambda value, lbl=value_lbl, step=increment: lbl.configure(text=self._format_toolbox_value(value, step)),
        )
        scale.grid(row=1, column=0, sticky=tk.EW)
        value_lbl.grid(row=1, column=1, sticky=tk.E, padx=(8, 0))
        self._bind_inspector_scale_click(scale, variable, value_lbl, from_, to, increment)

    def _current_relief_profile_curve(self):
        preset = normalize_relief_profile_preset(self.relief_profile_preset_var.get())
        return relief_profile_curve_for_preset(preset, getattr(self, "_relief_profile_curve", DEFAULT_RELIEF_PROFILE_CURVE))

    def _set_relief_profile_curve(self, curve, *, preset: str = "custom", refresh_delay_ms: int = 90, live: bool = False) -> None:
        preset = normalize_relief_profile_preset(preset)
        self.relief_profile_preset_var.set(preset)
        self._relief_profile_curve = relief_profile_curve_for_preset(preset, curve)
        if live:
            return
        self._invalidate_contour_wireframe_cache()
        self._draw_vector_overlay()
        self._schedule_preview_refresh(delay_ms=refresh_delay_ms)

    def _add_inspector_relief_profile_editor(self, parent, row: int, label: str):
        frame = self._add_inspector_row(parent, row, label)
        frame.columnconfigure(0, weight=1)
        colors = self._inspector_setting_colors(frame)

        top = tk.Frame(frame, bg=colors["body"], bd=0, highlightthickness=0)
        top.grid(row=1, column=0, sticky=tk.EW, pady=(0, 7))
        top.columnconfigure(0, weight=1)
        status_var = tk.StringVar(value=f"Aktywny: {relief_profile_preset_label(self.relief_profile_preset_var.get())}")
        tk.Label(
            top,
            textvariable=status_var,
            bg=colors["body"],
            fg=colors["fg"],
            font=("Segoe UI", 8, "bold"),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky=tk.W)
        tk.Label(
            top,
            text="3 sklejone krzywe: lewy spad, grzbiet, prawy spad. Przeciągane punkty działają symetrycznie.",
            bg=colors["body"],
            fg=self._blend_hex_colors(colors["fg"], colors["body"], 0.72),
            font=("Segoe UI", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=292,
        ).grid(row=1, column=0, sticky=tk.EW, pady=(2, 0))

        preset_row = tk.Frame(frame, bg=colors["body"], bd=0, highlightthickness=0)
        preset_row.grid(row=2, column=0, sticky=tk.EW, pady=(0, 7))
        preset_columns = 3
        for column in range(preset_columns):
            preset_row.columnconfigure(column, weight=1, uniform="relief_profile_preset")
        for index, preset_spec in enumerate(relief_profile_presets()):
            key = str(preset_spec.get("key") or DEFAULT_RELIEF_PROFILE_PRESET)
            text = str(preset_spec.get("label") or key)
            btn = tk.Button(
                preset_row,
                text=text,
                command=lambda item=preset_spec: apply_preset(item),
                bg=self._blend_hex_colors(colors["body"], "#4ec980", 0.78),
                fg=colors["fg"],
                activebackground=self._blend_hex_colors(colors["body"], "#4ec980", 0.58),
                activeforeground=colors["fg"],
                relief=tk.FLAT,
                bd=0,
                padx=5,
                pady=4,
                cursor="hand2",
                font=("Segoe UI", 8, "bold"),
                wraplength=78,
            )
            grid_row, grid_column = divmod(index, preset_columns)
            btn.grid(
                row=grid_row,
                column=grid_column,
                sticky=tk.EW,
                padx=(0 if grid_column == 0 else 4, 0),
                pady=(0 if grid_row == 0 else 4, 0),
            )

        canvas = tk.Canvas(
            frame,
            height=112,
            highlightthickness=1,
            highlightbackground=self._blend_hex_colors(colors["border"], "#4ec980", 0.55),
            bd=0,
            bg=self._blend_hex_colors(colors["body"], "#10161a", 0.75),
            cursor="hand2",
        )
        canvas.grid(row=3, column=0, sticky=tk.EW)
        drag_state = {"index": None}

        def bezier_point(points, t: float) -> tuple[float, float]:
            t = max(0.0, min(1.0, float(t)))
            inv = 1.0 - t
            x = (
                inv * inv * inv * points[0][0]
                + 3.0 * inv * inv * t * points[1][0]
                + 3.0 * inv * t * t * points[2][0]
                + t * t * t * points[3][0]
            )
            y = (
                inv * inv * inv * points[0][1]
                + 3.0 * inv * inv * t * points[1][1]
                + 3.0 * inv * t * t * points[2][1]
                + t * t * t * points[3][1]
            )
            return x, y

        def segments_from_curve(curve):
            return (
                [curve[0], curve[1], curve[2], curve[3]],
                [curve[3], curve[4], curve[5], curve[6]],
                [curve[6], curve[7], curve[8], curve[9]],
            )

        def canvas_metrics():
            width = max(220, int(canvas.winfo_width() or 300))
            height = max(92, int(canvas.winfo_height() or 112))
            return 18.0, 14.0, float(width - 18), float(height - 20)

        def point_to_canvas(point) -> tuple[float, float]:
            left, top_y, right, bottom = canvas_metrics()
            return left + float(point[0]) * (right - left), bottom - float(point[1]) * (bottom - top_y)

        def canvas_to_point(handle_index: int, x: float, y: float):
            left, top_y, right, bottom = canvas_metrics()
            x01 = max(0.02, min(0.98, (float(x) - left) / max(1.0, right - left)))
            y01 = max(0.0, min(1.0, (bottom - float(y)) / max(1.0, bottom - top_y)))
            current = list(normalize_relief_profile_curve(getattr(self, "_relief_profile_curve", DEFAULT_RELIEF_PROFILE_CURVE)))
            mirror_pairs = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
            left_index = mirror_pairs.get(handle_index, handle_index) if handle_index > 4 else handle_index
            if handle_index > 4:
                x01 = 1.0 - x01
            if left_index == 1:
                lower, upper = 0.006, min(0.24, float(current[2][0]) - 0.006)
            elif left_index == 2:
                lower, upper = float(current[1][0]) + 0.006, min(0.38, float(current[3][0]) - 0.006)
            elif left_index == 3:
                lower, upper = max(0.040, float(current[2][0]) + 0.006), min(0.48, float(current[4][0]) - 0.006)
            else:
                lower, upper = max(0.090, float(current[3][0]) + 0.006), 0.50
            if lower > upper:
                lower, upper = upper, lower
            return max(lower, min(upper, x01)), y01

        def apply_symmetric_handle(handle_index: int, point) -> tuple:
            current = list(normalize_relief_profile_curve(getattr(self, "_relief_profile_curve", DEFAULT_RELIEF_PROFILE_CURVE)))
            mirror_pairs = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
            left_index = mirror_pairs.get(handle_index, handle_index) if handle_index > 4 else handle_index
            x01, y01 = point
            current[left_index] = (x01, y01)
            mirror_index = mirror_pairs.get(left_index)
            if mirror_index is not None:
                current[mirror_index] = (1.0 - x01, y01)
            return normalize_relief_profile_curve(tuple(current))

        def redraw() -> None:
            try:
                canvas.delete("all")
                left, top_y, right, bottom = canvas_metrics()
                width = max(220, int(canvas.winfo_width() or 300))
                height = max(92, int(canvas.winfo_height() or 112))
                curve = self._current_relief_profile_curve()
                status_var.set(f"Aktywny: {relief_profile_preset_label(self.relief_profile_preset_var.get())}")
                canvas.create_rectangle(0, 0, width, height, fill=self._blend_hex_colors(colors["body"], "#10161a", 0.75), outline="")
                for frac in (0.25, 0.50, 0.75):
                    x = left + (right - left) * frac
                    y = bottom - (bottom - top_y) * frac
                    canvas.create_line(x, top_y, x, bottom, fill="#29423e", dash=(2, 4))
                    canvas.create_line(left, y, right, y, fill="#29423e", dash=(2, 4))
                canvas.create_line(left, bottom, right, bottom, fill="#6b7f7a", width=1)
                canvas.create_line(left, top_y, left, bottom, fill="#6b7f7a", width=1)
                canvas_points = [point_to_canvas(point) for point in curve]
                segment_colors = ("#f5a742", "#ffe2a9", "#f5a742")
                for segment_index, segment in enumerate(segments_from_curve(curve)):
                    segment_points = []
                    for sample in range(34):
                        bx, by = bezier_point(segment, sample / 33.0)
                        segment_points.extend(point_to_canvas((bx, by)))
                    if len(segment_points) >= 4:
                        canvas.create_line(
                            *segment_points,
                            fill=segment_colors[min(segment_index, len(segment_colors) - 1)],
                            width=2,
                            smooth=True,
                        )
                for chain in ((0, 1, 2, 3), (3, 4, 5, 6), (6, 7, 8, 9)):
                    pts = []
                    for idx in chain:
                        pts.extend(canvas_points[idx])
                    canvas.create_line(*pts, fill="#5d7f75", dash=(3, 3), width=1)
                label_specs = (
                    (0.09, "spad"),
                    (0.50, "grzbiet"),
                    (0.91, "spad"),
                )
                for x_frac, text in label_specs:
                    canvas.create_text(
                        left + (right - left) * x_frac,
                        top_y + 10,
                        text=text,
                        fill="#9fe7cb",
                        font=("Segoe UI", 7, "bold"),
                        anchor=tk.CENTER,
                    )
                for idx, point in enumerate(canvas_points):
                    editable = idx in {1, 2, 3, 4, 5, 6, 7, 8}
                    radius = 5 if editable else 4
                    if idx in {3, 6}:
                        fill, outline = "#55c5ff", "#c3edff"
                    elif idx in {4, 5}:
                        fill, outline = "#4ec980", "#c9ffdc"
                    elif editable:
                        fill, outline = "#f5a742", "#ffe2a9"
                    else:
                        fill, outline = "#81908d", "#aab8b4"
                    canvas.create_oval(
                        point[0] - radius,
                        point[1] - radius,
                        point[0] + radius,
                        point[1] + radius,
                        fill=fill,
                        outline=outline,
                        width=1,
                        tags=(f"profile_handle_{idx}",),
                    )
                canvas.create_text(
                    left,
                    bottom + 10,
                    text="krawędź",
                    fill="#9ab0aa",
                    font=("Segoe UI", 7),
                    anchor=tk.W,
                )
                canvas.create_text(
                    (left + right) / 2.0,
                    top_y - 4,
                    text="wysokość tuszu",
                    fill="#c5efe0",
                    font=("Segoe UI", 7, "bold"),
                    anchor=tk.S,
                )
                canvas.create_text(
                    right,
                    bottom + 10,
                    text="krawędź",
                    fill="#9ab0aa",
                    font=("Segoe UI", 7),
                    anchor=tk.E,
                )
            except Exception:
                pass

        def nearest_handle(event) -> int | None:
            try:
                points = [point_to_canvas(point) for point in normalize_relief_profile_curve(getattr(self, "_relief_profile_curve", DEFAULT_RELIEF_PROFILE_CURVE))]
                candidates = []
                for idx in (1, 2, 3, 4, 5, 6, 7, 8):
                    px, py = points[idx]
                    distance = math.hypot(float(event.x) - px, float(event.y) - py)
                    candidates.append((distance, idx))
                distance, idx = min(candidates, key=lambda item: item[0])
                return idx if distance <= 16.0 else None
            except Exception:
                return None

        def set_handle(handle_index: int, event, *, refresh_delay_ms: int) -> None:
            point = canvas_to_point(handle_index, event.x, event.y)
            curve = apply_symmetric_handle(handle_index, point)
            self._set_relief_profile_curve(
                curve,
                preset="custom",
                refresh_delay_ms=refresh_delay_ms,
                live=bool(drag_state.get("index") is not None and refresh_delay_ms >= 120),
            )
            redraw()

        def on_press(event):
            handle = nearest_handle(event)
            if handle is None:
                return "break"
            self._begin_canvas_live_edit()
            try:
                canvas.grab_set()
            except Exception:
                pass
            drag_state["index"] = handle
            self._relief_profile_drag_index = handle
            return "break"

        def on_drag(event):
            handle = drag_state.get("index")
            if handle in (1, 2, 3, 4, 5, 6, 7, 8):
                set_handle(int(handle), event, refresh_delay_ms=180)
            return "break"

        def on_release(event):
            handle = drag_state.get("index")
            if handle in (1, 2, 3, 4, 5, 6, 7, 8):
                point = canvas_to_point(int(handle), event.x, event.y)
                curve = apply_symmetric_handle(int(handle), point)
                self._set_relief_profile_curve(curve, preset="custom", refresh_delay_ms=120, live=False)
                redraw()
            drag_state["index"] = None
            self._relief_profile_drag_index = None
            try:
                canvas.grab_release()
            except Exception:
                pass
            self._end_canvas_live_edit(delay_ms=120)
            return "break"

        def apply_preset(preset_spec: dict) -> None:
            key = str(preset_spec.get("key") or DEFAULT_RELIEF_PROFILE_PRESET)
            curve = preset_spec.get("curve") or DEFAULT_RELIEF_PROFILE_CURVE
            self._set_relief_profile_curve(curve, preset=key, refresh_delay_ms=60)
            redraw()

        canvas.bind("<Configure>", lambda _event: redraw(), add="+")
        canvas.bind("<Button-1>", on_press, add="+")
        canvas.bind("<B1-Motion>", on_drag, add="+")
        canvas.bind("<ButtonRelease-1>", on_release, add="+")
        redraw()
        return frame

    def _add_inspector_color_slider(self, parent, row: int, label: str, variable, from_, to, increment, color: str):
        frame = self._add_inspector_row(parent, row, label)
        frame.columnconfigure(1, weight=1)
        swatch = tk.Frame(frame, width=14, height=14, bg=color, highlightthickness=1, highlightbackground="#4b5563")
        swatch.grid(row=1, column=0, sticky=tk.W, padx=(0, 8))
        swatch.grid_propagate(False)
        value_lbl = ttk.Label(frame, text=self._format_toolbox_value(variable.get(), increment), width=7, anchor=tk.E)
        scale = tk.Scale(
            frame,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=variable,
            resolution=increment,
            showvalue=False,
            troughcolor=color,
            activebackground=color,
            highlightthickness=0,
            bd=0,
            sliderlength=16,
            command=lambda value, lbl=value_lbl, step=increment: lbl.configure(text=self._format_toolbox_value(value, step)),
        )
        scale.grid(row=1, column=1, sticky=tk.EW)
        value_lbl.grid(row=1, column=2, sticky=tk.E, padx=(8, 0))
        self._bind_inspector_scale_click(scale, variable, value_lbl, from_, to, increment)

    def _add_inspector_range_slider(self, parent, row: int, label: str, min_var, max_var, from_, to, increment):
        frame = self._add_inspector_row(parent, row, label)
        frame.columnconfigure(0, weight=1)
        colors = self._inspector_setting_colors(frame)
        value_row = tk.Frame(frame, bg=colors["body"], bd=0, highlightthickness=0)
        value_row.grid(row=1, column=0, sticky=tk.EW)
        value_row.columnconfigure(1, weight=1)
        min_label = tk.Label(
            value_row,
            bg=colors["body"],
            fg="#41c973",
            font=("Segoe UI", 8, "bold"),
            anchor=tk.W,
        )
        max_label = tk.Label(
            value_row,
            bg=colors["body"],
            fg="#4aa8ff",
            font=("Segoe UI", 8, "bold"),
            anchor=tk.E,
        )
        min_label.grid(row=0, column=0, sticky=tk.W)
        max_label.grid(row=0, column=2, sticky=tk.E)

        canvas = tk.Canvas(
            frame,
            height=34,
            highlightthickness=0,
            bd=0,
            bg=colors["body"],
            cursor="sb_h_double_arrow",
        )
        canvas.grid(row=2, column=0, sticky=tk.EW, pady=(3, 0))
        drag_state = {"handle": None}

        try:
            start = float(from_)
            end = float(to)
            step = float(increment or 0.0)
        except Exception:
            start, end, step = 0.0, 1.0, 0.01
        span = max(0.000001, end - start)

        def _format_pair_value(value: object) -> str:
            return self._format_toolbox_value(value, step or 0.01)

        def _read_values() -> tuple[float, float]:
            self._coerce_range_pair(min_var, max_var, start, end, "max")
            try:
                low_value = max(start, min(end, float(min_var.get())))
                high_value = max(start, min(end, float(max_var.get())))
            except Exception:
                low_value, high_value = start, end
            if low_value > high_value:
                low_value, high_value = high_value, low_value
            return low_value, high_value

        def _snap_value(value: float) -> float | int:
            value = max(start, min(end, float(value)))
            if step > 0:
                value = round(value / step) * step
            if step >= 1:
                return int(round(value))
            return value

        def _value_to_x(value: float, left: float, right: float) -> float:
            ratio = max(0.0, min(1.0, (float(value) - start) / span))
            return left + (right - left) * ratio

        def _x_to_value(x: float, left: float, right: float):
            ratio = max(0.0, min(1.0, (float(x) - left) / max(1.0, right - left)))
            return _snap_value(start + span * ratio)

        def refresh() -> None:
            low_value, high_value = _read_values()
            try:
                min_label.configure(text=f"od {_format_pair_value(low_value)}")
                max_label.configure(text=f"do {_format_pair_value(high_value)}")
            except Exception:
                pass
            try:
                canvas.delete("range_slider")
                width_px = max(120, int(canvas.winfo_width() or 280))
                left = 15.0
                right = float(width_px - 15)
                track_y = 17.0
                min_x = _value_to_x(low_value, left, right)
                max_x = _value_to_x(high_value, left, right)
                canvas.create_line(left, track_y, right, track_y, fill="#53615c", width=4, tags=("range_slider",))
                canvas.create_line(min_x, track_y, max_x, track_y, fill="#78d79a", width=6, tags=("range_slider",))
                canvas.create_oval(min_x - 7, track_y - 7, min_x + 7, track_y + 7, fill="#41c973", outline="#0d1f18", width=1, tags=("range_slider",))
                canvas.create_rectangle(max_x - 7, track_y - 7, max_x + 7, track_y + 7, fill="#4aa8ff", outline="#0d1f18", width=1, tags=("range_slider",))
            except Exception:
                pass

        def set_from_event(event, forced_handle: str | None = None, *, schedule: bool = True):
            try:
                width_px = max(120, int(canvas.winfo_width() or 280))
                left = 15.0
                right = float(width_px - 15)
                low_value, high_value = _read_values()
                min_x = _value_to_x(low_value, left, right)
                max_x = _value_to_x(high_value, left, right)
                handle = forced_handle or drag_state.get("handle")
                if handle not in ("min", "max"):
                    click_x = float(getattr(event, "x", 0.0) or 0.0)
                    handle = "min" if abs(click_x - min_x) <= abs(click_x - max_x) else "max"
                    drag_state["handle"] = handle
                value = _x_to_value(float(getattr(event, "x", 0.0) or 0.0), left, right)
                if handle == "min":
                    min_var.set(min(float(value), high_value))
                    changed = "min"
                else:
                    max_var.set(max(float(value), low_value))
                    changed = "max"
                self._coerce_range_pair(min_var, max_var, start, end, changed)
                refresh()
                if schedule:
                    self._schedule_preview_refresh()
                return "break"
            except Exception:
                return None

        def begin_live_edit() -> None:
            if bool(getattr(self, "_inspector_scale_drag_active", False)):
                return
            self._inspector_scale_drag_active = True
            self._begin_canvas_live_edit()
            try:
                canvas.grab_set()
            except Exception:
                pass

        def press(event):
            begin_live_edit()
            return set_from_event(event, schedule=False)

        def drag(event):
            begin_live_edit()
            return set_from_event(event, schedule=False)

        def release(event=None):
            try:
                if event is not None:
                    set_from_event(event, schedule=False)
            finally:
                drag_state["handle"] = None
                try:
                    canvas.grab_release()
                except Exception:
                    pass
                self._inspector_scale_drag_active = False
                self._end_canvas_live_edit(delay_ms=120)
            return "break"

        try:
            canvas.bind("<Configure>", lambda _event: refresh(), add="+")
            canvas.bind("<Button-1>", press, add="+")
            canvas.bind("<B1-Motion>", drag, add="+")
            canvas.bind("<ButtonRelease-1>", release, add="+")
        except Exception:
            pass
        refresh()

    def _bind_inspector_scale_click(self, scale, variable, value_label, from_, to, increment):
        """Make a trough click jump to the clicked X, not to Tk's page ends."""
        try:
            start = float(from_)
            end = float(to)
            step = float(increment or 0.0)
        except Exception:
            return

        def set_from_event(event):
            try:
                width = max(1, int(scale.winfo_width() or 1))
                ratio = max(0.0, min(1.0, float(getattr(event, "x", 0) or 0) / float(max(1, width - 1))))
                value = start + (end - start) * ratio
                if step > 0:
                    value = round(value / step) * step
                low, high = (start, end) if start <= end else (end, start)
                value = max(low, min(high, value))
                variable.set(value)
                try:
                    value_label.configure(text=self._format_toolbox_value(value, step or 0.01))
                except Exception:
                    pass
                return "break"
            except Exception:
                return None

        def begin_live_edit() -> None:
            if bool(getattr(self, "_inspector_scale_drag_active", False)):
                return
            self._inspector_scale_drag_active = True
            self._begin_canvas_live_edit()
            try:
                scale.grab_set()
            except Exception:
                pass

        def on_press(event):
            begin_live_edit()
            return set_from_event(event)

        def on_drag(event):
            begin_live_edit()
            return set_from_event(event)

        def on_release(event):
            try:
                set_from_event(event)
            finally:
                try:
                    scale.grab_release()
                except Exception:
                    pass
                self._inspector_scale_drag_active = False
                self._end_canvas_live_edit(delay_ms=120)
            return "break"

        try:
            scale.bind("<Button-1>", on_press, add=False)
            scale.bind("<B1-Motion>", on_drag, add=False)
            scale.bind("<ButtonRelease-1>", on_release, add=False)
        except Exception:
            pass

    def _format_scene_float(self, value: object, default: float = 0.0) -> str:
        try:
            value = float(value)
            if not math.isfinite(value):
                value = float(default)
        except Exception:
            value = float(default)
        return f"{value:.2f}"

    def _format_scene_xyz_from_vars(self, x_var, y_var, z_var, defaults: tuple[float, float, float]) -> str:
        return (
            f"{self._format_scene_float(self._read_float_var(x_var, defaults[0]), defaults[0])}, "
            f"{self._format_scene_float(self._read_float_var(y_var, defaults[1]), defaults[1])}, "
            f"{self._format_scene_float(self._read_float_var(z_var, defaults[2]), defaults[2])}"
        )

    def _headlight_world_var_triplet(self, index: int, handle: str):
        handle = "target" if str(handle or "").lower() == "target" else "source"
        if index == 1:
            return (
                (self.traffic_headlight_target_world_x_var, self.traffic_headlight_target_world_y_var, self.traffic_headlight_target_world_z_var)
                if handle == "target"
                else (self.traffic_headlight_source_world_x_var, self.traffic_headlight_source_world_y_var, self.traffic_headlight_source_world_z_var)
            )
        if index == 2:
            return (
                (self.traffic_headlight_2_target_world_x_var, self.traffic_headlight_2_target_world_y_var, self.traffic_headlight_2_target_world_z_var)
                if handle == "target"
                else (self.traffic_headlight_2_source_world_x_var, self.traffic_headlight_2_source_world_y_var, self.traffic_headlight_2_source_world_z_var)
            )
        return (
            (self.traffic_headlight_3_target_world_x_var, self.traffic_headlight_3_target_world_y_var, self.traffic_headlight_3_target_world_z_var)
            if handle == "target"
            else (self.traffic_headlight_3_source_world_x_var, self.traffic_headlight_3_source_world_y_var, self.traffic_headlight_3_source_world_z_var)
        )

    def _headlight_strength_var(self, index: int):
        if index == 1:
            return self.traffic_headlight_var
        if index == 2:
            return self.traffic_headlight_2_var
        return self.traffic_headlight_3_var

    def _enforce_scene_invariants(self) -> None:
        fixed_values = (
            (self.scene_plate_width_var, 1.0),
            (self.scene_plate_height_var, 0.24),
            (self.scene_camera_x_var, 0.0),
            (self.scene_camera_y_var, 0.0),
            (self.scene_camera_target_x_var, 0.0),
            (self.scene_camera_target_y_var, 0.0),
            (self.scene_camera_target_z_var, 0.0),
            (self.traffic_headlight_target_world_z_var, 0.0),
            (self.traffic_headlight_2_target_world_z_var, 0.0),
            (self.traffic_headlight_3_target_world_z_var, 0.0),
        )
        for variable, value in fixed_values:
            try:
                if abs(self._read_float_var(variable, value) - value) > 1e-9:
                    variable.set(value)
            except Exception:
                pass
        for variable, minimum, maximum in (
            (self.scene_camera_z_var, 0.35, 4.0),
            (self.scene_view_yaw_var, -180.0, 180.0),
            (self.scene_view_pitch_var, -89.0, 89.0),
            (self.scene_view_roll_var, -180.0, 180.0),
        ):
            try:
                current = self._read_float_var(variable, 0.0)
                clamped = max(float(minimum), min(float(maximum), current))
                if abs(current - clamped) > 1e-9:
                    variable.set(clamped)
            except Exception:
                pass
        for index in (1, 2, 3):
            try:
                target_x_var, target_y_var, _target_z_var = self._headlight_world_var_triplet(index, "target")
                for variable, minimum, maximum in (
                    (target_x_var, -0.5, 0.5),
                    (target_y_var, -0.12, 0.12),
                ):
                    current = self._read_float_var(variable, 0.0)
                    clamped = max(float(minimum), min(float(maximum), current))
                    if abs(current - clamped) > 1e-9:
                        variable.set(clamped)
            except Exception:
                pass

    def _camera_axis_strength_from_distance(self, camera_z: float, target_x: float = 0.0, target_y: float = 0.0) -> float:
        try:
            camera_z = max(0.35, min(4.0, float(camera_z)))
        except Exception:
            camera_z = 1.65
        try:
            target_offset = math.hypot(float(target_x), float(target_y))
            target_distance = math.sqrt(camera_z * camera_z + target_offset * target_offset)
            target_cos = camera_z / max(0.001, target_distance)
        except Exception:
            target_cos = 1.0
        return max(0.0, min(1.0, (1.0 / (1.0 + camera_z * 0.42)) * (0.74 + 0.26 * target_cos)))

    def _scene_summary_text(self) -> str:
        try:
            self._enforce_scene_invariants()
            camera_z = self._format_scene_float(self._read_float_var(self.scene_camera_z_var, 1.65), 1.65)
            camera = f"Kamera: frontalna, cel w centrum, Z={camera_z}"
            lines = [camera]
            for index in (1, 2, 3):
                strength_var = self._headlight_strength_var(index)
                strength = self._read_float_var(strength_var, 0.0)
                state = "ON" if self._headlight_effect_enabled(index) and strength > 0.001 else "off"
                visible = "widok" if self._headlight_cones_are_visible(index) else "ukryty"
                source = self._format_scene_xyz_from_vars(*self._headlight_world_var_triplet(index, "source"), (0.0, 0.0, 0.95))
                tx_var, ty_var, _tz_var = self._headlight_world_var_triplet(index, "target")
                target = (
                    f"{self._format_scene_float(self._read_float_var(tx_var, 0.0), 0.0)}, "
                    f"{self._format_scene_float(self._read_float_var(ty_var, 0.0), 0.0)}, 0.00"
                )
                lines.append(f"R{index}: {state} {strength:.2f}, {visible} | {source} -> cel {target}")
            return "\n".join(lines)
        except Exception:
            return "Scena XYZ: brak danych."

    def _refresh_scene_summary(self) -> None:
        var = getattr(self, "scene_summary_var", None)
        if var is None:
            return
        try:
            text = self._scene_summary_text()
            if var.get() != text:
                var.set(text)
        except Exception:
            pass

    def _add_inspector_scene_summary(self, parent, row: int):
        self._refresh_scene_summary()
        colors = self._inspector_setting_colors(parent)
        outer = tk.Frame(parent, bg=colors["border"], padx=1, pady=1, bd=0, highlightthickness=0)
        outer.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=(0, 10))
        outer.columnconfigure(0, weight=1)
        frame = tk.Frame(outer, bg=colors["body"], padx=10, pady=9, bd=0, highlightthickness=0)
        frame.grid(row=0, column=0, sticky=tk.EW)
        frame.columnconfigure(0, weight=1)
        tk.Label(
            frame,
            text="Stan sceny XYZ",
            bg=colors["header"],
            fg=colors["accent"],
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=4,
            anchor=tk.W,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
        tk.Label(
            frame,
            textvariable=self.scene_summary_var,
            bg=colors["body"],
            fg=colors["fg"],
            font=("Consolas", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=290,
        ).grid(row=1, column=0, sticky=tk.EW)

    def _current_headlight_inspector_index(self) -> int:
        selected = getattr(self, "_headlight_config_index", None) or getattr(self, "_active_headlight_index", None) or 1
        try:
            selected = int(selected)
        except Exception:
            selected = 1
        if selected not in (1, 2, 3):
            selected = 1
        return selected

    def _select_headlight_for_inspector(self, index: int):
        self._active_toolbox = "illumination"
        try:
            index = int(index or 1)
        except Exception:
            index = 1
        if index not in (1, 2, 3):
            index = 1
        self._select_headlight(index, reveal=True)
        self._headlight_config_index = index
        self._headlight_panel_pos = None
        self._headlight_panel_anchor_index = None
        self._pending_inspector_scroll_anchor = "headlights"
        self._refresh_toolbox_tab_labels()
        self._refresh_effect_inspector_title()
        self._render_active_toolbox()
        self._draw_vector_overlay()

    def _render_headlight_inspector_section(self, parent, row: int) -> int:
        selected = self._current_headlight_inspector_index()
        section = ttk.LabelFrame(parent, text=" Reflektory R1/R2/R3 ", padding=8)
        section.grid(row=row, column=0, sticky=tk.EW, pady=(10, 0))
        self._inspector_anchor_widgets["headlights"] = section
        for column in (0, 1, 2):
            section.columnconfigure(column, weight=1)
        for index in (1, 2, 3):
            ttk.Button(
                section,
                text=("> " if index == selected else "") + f"R{index}",
                command=lambda value=index: self._select_headlight_for_inspector(value),
            ).grid(row=0, column=index - 1, sticky=tk.EW, padx=3, pady=(0, 8))
        ttk.Label(
            section,
            text=f"Aktywny reflektor: R{selected}",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(0, 6))
        ttk.Label(
            section,
            text="Kierunek R1/R2/R3 steruje też cieniem wypukłości konturów i odbiciami.",
            justify=tk.LEFT,
            wraplength=285,
        ).grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=(0, 6))
        visibility_vars = getattr(self, "_headlight_visibility_vars", {})
        visibility_var = visibility_vars.get(selected)
        if visibility_var is None:
            visibility_var = tk.BooleanVar(value=self._headlight_cones_are_visible(selected))
            visibility_vars[selected] = visibility_var
            self._headlight_visibility_vars = visibility_vars
        else:
            try:
                visibility_var.set(self._headlight_cones_are_visible(selected))
            except Exception:
                pass
        ttk.Checkbutton(
            section,
            text="Pokaż reprezentację na podglądzie",
            variable=visibility_var,
            command=lambda idx=selected: self._toggle_headlight_cone_visibility(idx) or self._render_active_toolbox() or self._draw_vector_overlay(),
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        ttk.Checkbutton(
            section,
            text="Włącz efekt reflektora",
            variable=self._headlight_enabled_var(selected),
            command=lambda idx=selected: self._on_headlight_effect_checkbox_changed(idx),
        ).grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        strength_var, red_var, green_var, blue_var, cone_var, source_radius_var = self._headlight_control_vars(selected)
        controls = (
            ("Natężenie", strength_var, 0.0, 1.0, 0.01, None),
            ("Kolor R", red_var, 0.0, 1.0, 0.01, "#f05252"),
            ("Kolor G", green_var, 0.0, 1.0, 0.01, "#22c55e"),
            ("Kolor B", blue_var, 0.0, 1.0, 0.01, "#3b82f6"),
            ("Promień stożka", cone_var, 0.02, 2.5, 0.01, None),
            ("Promień źródła", source_radius_var, 0.0, 2.5, 0.01, None),
        )
        subrow = 5
        for label, variable, from_, to, step, color in controls:
            if color:
                self._add_inspector_color_slider(section, subrow, label, variable, from_, to, step, color)
            else:
                self._add_inspector_slider(section, subrow, label, variable, from_, to, step)
            subrow += 1
        self._add_inspector_section_header(
            section,
            subrow,
            "Pozycja w swiecie",
            "Zrodlo to polozenie reflektora. Cel jest punktem na powierzchni tablicy. Strzalki przesuwaja cel XY, Shift+strzalki zrodlo XY, PageUp/PageDown zrodlo Z.",
            priority=True,
        )
        subrow += 1
        self._add_inspector_xyz(
            section,
            subrow,
            "Zrodlo reflektora",
            self._headlight_world_var_triplet(selected, "source"),
            ((-3.0, 3.0), (-3.0, 3.0), (0.0, 8.0)),
            0.01,
        )
        subrow += 1
        self._add_inspector_xy_fixed_z(
            section,
            subrow,
            "Cel na tablicy",
            self._headlight_world_var_triplet(selected, "target"),
            ((-0.5, 0.5), (-0.12, 0.12)),
            0.01,
            0.0,
        )
        return row + 1

    def _scene_projection_ranges(self, index: int) -> dict[str, tuple[float, float]]:
        try:
            source_x, source_y, source_z = (
                self._read_float_var(var, default)
                for var, default in zip(self._headlight_world_var_triplet(index, "source"), (0.0, 0.0, 0.95))
            )
            target_x, target_y, _target_z = (
                self._read_float_var(var, default)
                for var, default in zip(self._headlight_world_var_triplet(index, "target"), (0.0, 0.0, 0.0))
            )
            camera_z = self._read_float_var(self.scene_camera_z_var, 1.65)
            camera_target_x = self._read_float_var(self.scene_camera_target_x_var, 0.0)
            camera_target_y = self._read_float_var(self.scene_camera_target_y_var, 0.0)
        except Exception:
            source_x, source_y, source_z = 0.0, 0.0, 0.95
            target_x, target_y = 0.0, 0.0
            camera_z, camera_target_x, camera_target_y = 1.65, 0.0, 0.0
        x_extent = max(0.75, abs(source_x) * 1.18, abs(target_x) * 1.18, abs(camera_target_x) * 1.18)
        y_extent = max(0.34, abs(source_y) * 1.18, abs(target_y) * 1.18, abs(camera_target_y) * 1.18)
        z_max = max(1.25, source_z * 1.16, camera_z * 1.06)
        return {
            "x": (-x_extent, x_extent),
            "y": (-y_extent, y_extent),
            "z": (0.0, z_max),
        }

    def _projection_to_canvas(
        self,
        x_value: float,
        y_value: float,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        width: int,
        height: int,
        pad: int = 14,
    ) -> tuple[float, float]:
        xmin, xmax = x_range
        ymin, ymax = y_range
        x = pad + (float(x_value) - xmin) / max(0.0001, xmax - xmin) * max(1.0, width - pad * 2)
        y = height - pad - (float(y_value) - ymin) / max(0.0001, ymax - ymin) * max(1.0, height - pad * 2)
        return x, y

    def _projection_from_canvas(
        self,
        x: float,
        y: float,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        width: int,
        height: int,
        pad: int = 14,
    ) -> tuple[float, float]:
        xmin, xmax = x_range
        ymin, ymax = y_range
        x_norm = (float(x) - pad) / max(1.0, width - pad * 2)
        y_norm = (height - pad - float(y)) / max(1.0, height - pad * 2)
        x_value = xmin + max(0.0, min(1.0, x_norm)) * (xmax - xmin)
        y_value = ymin + max(0.0, min(1.0, y_norm)) * (ymax - ymin)
        return x_value, y_value

    def _scene_view_modes(self) -> tuple[tuple[str, str], ...]:
        return (
            ("xy", "XY"),
            ("xz", "XZ"),
            ("yz", "YZ"),
        )

    def _active_scene_viewport_mode(self) -> str:
        mode = str(getattr(self, "_scene_viewport_mode", "xy") or "xy").lower()
        valid = {key for key, _label in self._scene_view_modes()}
        if mode not in valid:
            mode = "xy"
            self._scene_viewport_mode = mode
        return mode

    def _scene_viewport_rects(
        self, width: int, height: int
    ) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
        rect = (18.0, 60.0, float(width - 18), float(height - 18))
        return rect, (rect[0], rect[1] + 30.0, rect[2], rect[3])

    def _scene_view_preset_angles(self, mode: str) -> tuple[float, float, float]:
        mode = str(mode or "xy").lower()
        if mode == "xz":
            return 0.0, 0.0, 0.0
        if mode == "yz":
            return 90.0, 0.0, 0.0
        return 0.0, 89.0, 0.0

    def _apply_scene_view_preset(self, mode: str) -> None:
        yaw, pitch, roll = self._scene_view_preset_angles(mode)
        self.scene_view_yaw_var.set(yaw)
        self.scene_view_pitch_var.set(pitch)
        self.scene_view_roll_var.set(roll)
        self._set_scene_view_rotation_matrix(self._scene_view_matrix_from_angles(yaw, pitch, roll))

    def _scene_view_mode_from_projection(self, projection: str) -> str | None:
        mode = str(projection or "").lower()
        valid = {key for key, _label in self._scene_view_modes()}
        return mode if mode in valid else None

    def _scene_viewport_ranges(self) -> dict[str, tuple[float, float]]:
        try:
            camera_z = self._read_float_var(self.scene_camera_z_var, 1.65)
            camera_target_x = self._read_float_var(self.scene_camera_target_x_var, 0.0)
            camera_target_y = self._read_float_var(self.scene_camera_target_y_var, 0.0)
        except Exception:
            camera_z, camera_target_x, camera_target_y = 1.65, 0.0, 0.0
        xs = [-0.65, 0.65, camera_target_x]
        ys = [-0.24, 0.24, camera_target_y]
        zs = [0.0, max(1.0, camera_z)]
        for index in (1, 2, 3):
            for handle, defaults in (("source", (0.0, 0.0, 0.95)), ("target", (0.0, 0.0, 0.0))):
                try:
                    x_var, y_var, z_var = self._headlight_world_var_triplet(index, handle)
                    xs.append(self._read_float_var(x_var, defaults[0]))
                    ys.append(self._read_float_var(y_var, defaults[1]))
                    zs.append(self._read_float_var(z_var, defaults[2]))
                except Exception:
                    pass
        x_extent = max(0.75, max(abs(value) for value in xs) * 1.18)
        y_extent = max(0.34, max(abs(value) for value in ys) * 1.18)
        z_max = max(1.25, max(zs) * 1.08)
        return {"x": (-x_extent, x_extent), "y": (-y_extent, y_extent), "z": (0.0, z_max)}

    def _project_scene_point_for_mode(self, point: tuple[float, float, float], projection: str) -> tuple[float, float]:
        projection = str(projection or "xy").lower()
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        if projection == "xz":
            return x, z
        if projection == "yz":
            return y, z
        return x, y

    def _should_show_contour_wireframe(self) -> bool:
        try:
            if bool(self.rain_edge_debug_var.get()):
                return True
        except Exception:
            pass
        # The contour/profile wireframe is diagnostic. Merely opening the
        # relief inspector must not override the checkbox state.
        return bool(getattr(self, "_relief_profile_drag_index", None) is not None)

    def _draw_projected_contour_wireframe(
        self,
        canvas: tk.Canvas,
        contour_wire_segments: list[dict],
        project_point,
        *,
        tags: tuple[str, ...],
        main: bool = True,
    ) -> None:
        if not contour_wire_segments:
            return

        def contour_wire_color(kind: str) -> str:
            kind = str(kind or "").lower()
            if kind == "base":
                return "#24545a"
            if kind == "strut":
                return "#f5a742"
            if kind == "edge":
                return "#2f8188"
            if kind == "rib":
                return "#5ff2dd"
            if kind == "rail":
                return "#8df0c4"
            return "#62f1dc"

        draw_order = {"base": 0, "edge": 1, "rail": 2, "rib": 3, "strut": 4, "ridge": 5}
        for segment in sorted(contour_wire_segments, key=lambda item: draw_order.get(str(item.get("kind", "wire")), 2)):
            try:
                kind = str(segment.get("kind", "wire") or "wire")
                sx, sy = project_point(segment["start"])
                ex, ey = project_point(segment["end"])
                if math.hypot(float(ex) - float(sx), float(ey) - float(sy)) < 0.45:
                    continue
                canvas.create_line(
                    sx,
                    sy,
                    ex,
                    ey,
                    fill=contour_wire_color(kind),
                    width=1,
                    dash=(2, 4) if kind == "base" else None,
                    tags=tags,
                )
            except Exception:
                continue

    def _draw_scene_viewport_mode_icons(
        self,
        canvas: tk.Canvas,
        width: int,
        height: int,
        start_x: int | None = None,
        start_y: int = 10,
        max_right: int | None = None,
    ) -> float:
        active = self._active_scene_viewport_mode()
        modes = self._scene_view_modes()
        icon_w = 42
        icon_h = 38
        gap = 4
        texture_slots = 1
        slot_count = len(modes) + texture_slots
        if max_right is not None and start_x is not None:
            available = max(0, int(max_right) - int(start_x))
            compact_w = int((available - (slot_count - 1) * gap) / max(1, slot_count))
            icon_w = max(30, min(icon_w, compact_w))
        total_w = slot_count * icon_w + (slot_count - 1) * gap
        if start_x is None:
            x = max(10, min(max(10, width - total_w - 10), 250))
        else:
            x = max(10, int(start_x))
            if max_right is not None and x + total_w > int(max_right):
                x = max(10, int(max_right) - total_w)
        y = int(start_y)
        canvas.create_rectangle(
            x - 6,
            y,
            x + total_w + 6,
            y + icon_h,
            fill="#0b1218",
            outline="#26333a",
            width=1,
            tags=("aug_overlay", "scene_view_icons"),
        )
        texture_enabled = bool(self.scene_plate_texture_var.get())
        canvas.create_rectangle(
            x,
            y + 2,
            x + icon_w,
            y + icon_h - 2,
            fill=("#183124" if texture_enabled else "#171b20"),
            outline=("#7fd8a8" if texture_enabled else "#596269"),
            width=2 if texture_enabled else 1,
            tags=("aug_overlay", "scene_view_icons"),
        )
        canvas.create_text(
            x + icon_w / 2,
            y + 14,
            text="IMG",
            anchor=tk.CENTER,
            fill=("#bfffd6" if texture_enabled else "#cbd5dc"),
            font=("Segoe UI", 8, "bold"),
            tags=("aug_overlay", "scene_view_icons"),
        )
        canvas.create_text(
            x + icon_w / 2,
            y + 27,
            text=("ON" if texture_enabled else "OFF"),
            anchor=tk.CENTER,
            fill=("#8df0b7" if texture_enabled else "#9aa7ad"),
            font=("Segoe UI", 6, "bold"),
            tags=("aug_overlay", "scene_view_icons"),
        )
        self._register_canvas_overlay_region(
            "scene_plate_texture_toggle",
            (x, y, x + icon_w, y + icon_h),
        )
        x += icon_w + gap
        for mode, label in modes:
            is_active = mode == active
            fill = "#183124" if is_active else "#121a20"
            outline = "#7fd8a8" if is_active else "#596269"
            text_fill = "#bfffd6" if is_active else "#d7dde1"
            canvas.create_rectangle(
                x,
                y + 2,
                x + icon_w,
                y + icon_h - 2,
                fill=fill,
                outline=outline,
                width=2 if is_active else 1,
                tags=("aug_overlay", "scene_view_icons"),
            )
            canvas.create_text(
                x + icon_w / 2,
                y + icon_h / 2,
                text=label,
                anchor=tk.CENTER,
                fill=text_fill,
                font=("Segoe UI", 8, "bold"),
                tags=("aug_overlay", "scene_view_icons"),
            )
            self._register_canvas_overlay_region(
                "scene_viewport_mode",
                (x, y, x + icon_w, y + icon_h),
                mode=mode,
            )
            x += icon_w + gap
        return x + 6

    def _draw_scene_flat_viewport_body(
        self,
        canvas: tk.Canvas,
        index: int,
        projection: str,
        rect: tuple[float, float, float, float],
    ) -> None:
        projection = str(projection or "xy").lower()
        if projection not in {"xy", "xz", "yz"}:
            projection = "xy"
        x1, y1, x2, y2 = [float(value) for value in rect]
        plot_w = max(1.0, x2 - x1)
        plot_h = max(1.0, y2 - y1)
        pad = 34
        ranges = self._scene_viewport_ranges()
        if projection == "xz":
            x_range, y_range = ranges["x"], ranges["z"]
        elif projection == "yz":
            x_range, y_range = ranges["y"], ranges["z"]
        else:
            x_range, y_range = ranges["x"], ranges["y"]
        axis_x, axis_y = self._scene_projection_axis_labels(projection)

        def to_abs(point: tuple[float, float, float]) -> tuple[float, float]:
            px, py = self._project_scene_point_for_mode(point, projection)
            cx, cy = self._projection_to_canvas(px, py, x_range, y_range, int(plot_w), int(plot_h), pad=pad)
            return x1 + cx, y1 + cy

        def to_abs_2d(point: tuple[float, float]) -> tuple[float, float]:
            cx, cy = self._projection_to_canvas(point[0], point[1], x_range, y_range, int(plot_w), int(plot_h), pad=pad)
            return x1 + cx, y1 + cy

        viewport_scale = min(
            max(1.0, plot_w - 2 * pad) / max(0.001, float(x_range[1]) - float(x_range[0])),
            max(1.0, plot_h - 2 * pad) / max(0.001, float(y_range[1]) - float(y_range[0])),
        )

        zero_x, zero_y = to_abs_2d((0.0, 0.0))
        canvas.create_line(x1 + pad, zero_y, x2 - pad, zero_y, fill="#53616a", width=1, arrow=tk.LAST, tags=("aug_overlay", "scene_viewport"))
        canvas.create_line(zero_x, y2 - pad, zero_x, y1 + pad, fill="#53616a", width=1, arrow=tk.LAST, tags=("aug_overlay", "scene_viewport"))
        canvas.create_text(x2 - pad + 6, zero_y, text=axis_x, anchor=tk.W, fill="#d7dde1", font=("Segoe UI", 10, "bold"), tags=("aug_overlay", "scene_viewport"))
        canvas.create_text(zero_x + 6, y1 + pad, text=axis_y, anchor=tk.W, fill="#d7dde1", font=("Segoe UI", 10, "bold"), tags=("aug_overlay", "scene_viewport"))

        plate_color = "#42d77d"
        texture_enabled = bool(self.scene_plate_texture_var.get())

        def draw_texture_box(left: float, top: float, right: float, bottom: float, *, center_line: bool = False) -> None:
            box_left = min(float(left), float(right))
            box_right = max(float(left), float(right))
            box_top = min(float(top), float(bottom))
            box_bottom = max(float(top), float(bottom))
            if box_right - box_left < 8:
                cx = (box_left + box_right) / 2.0
                box_left = cx - 4
                box_right = cx + 4
            if box_bottom - box_top < 8:
                cy = (box_top + box_bottom) / 2.0
                box_top = cy - 4
                box_bottom = cy + 4
            if texture_enabled:
                photo = self._scene_plate_texture_photo(box_right - box_left, box_bottom - box_top)
                if photo is not None:
                    canvas.create_image(
                        (box_left + box_right) / 2.0,
                        (box_top + box_bottom) / 2.0,
                        image=photo,
                        anchor=tk.CENTER,
                        tags=("aug_overlay", "scene_viewport"),
                    )
                else:
                    canvas.create_rectangle(box_left, box_top, box_right, box_bottom, fill="#1b2d23", outline="", tags=("aug_overlay", "scene_viewport"))
                    stripe_x = box_left - (box_bottom - box_top)
                    while stripe_x < box_right:
                        canvas.create_line(
                            stripe_x,
                            box_bottom,
                            stripe_x + (box_bottom - box_top),
                            box_top,
                            fill="#31513c",
                            width=1,
                            tags=("aug_overlay", "scene_viewport"),
                        )
                        stripe_x += 10
            canvas.create_rectangle(
                box_left,
                box_top,
                box_right,
                box_bottom,
                outline=plate_color,
                width=2,
                tags=("aug_overlay", "scene_viewport"),
            )
            if center_line:
                cy = (box_top + box_bottom) / 2.0
                canvas.create_line(box_left, cy, box_right, cy, fill=plate_color, width=1, dash=(3, 3), tags=("aug_overlay", "scene_viewport"))

        if projection == "xy":
            p_left, p_top = to_abs((-0.5, 0.12, 0.0))
            p_right, p_bottom = to_abs((0.5, -0.12, 0.0))
            draw_texture_box(p_left, p_top, p_right, p_bottom)
        elif projection == "xz":
            p_left, base_y = to_abs((-0.5, 0.0, 0.0))
            p_right, _base_y = to_abs((0.5, 0.0, 0.0))
            band_h = max(18.0, min(54.0, plot_h * 0.10))
            draw_texture_box(p_left, base_y - band_h / 2.0, p_right, base_y + band_h / 2.0, center_line=True)
        else:
            p_left, base_y = to_abs((0.0, -0.12, 0.0))
            p_right, _base_y = to_abs((0.0, 0.12, 0.0))
            band_h = max(18.0, min(54.0, plot_h * 0.10))
            draw_texture_box(p_left, base_y - band_h / 2.0, p_right, base_y + band_h / 2.0, center_line=True)

        contour_wire_segments = self._scene_contour_wireframe_world_segments()
        if contour_wire_segments:
            self._draw_projected_contour_wireframe(
                canvas,
                contour_wire_segments,
                to_abs,
                tags=("aug_overlay", "scene_viewport", "contour_profile_wire"),
                main=True,
            )

    def _draw_scene_viewport(self, canvas: tk.Canvas, width: int, height: int) -> None:
        mode = self._active_scene_viewport_mode()
        body_mode = mode
        canvas.create_rectangle(0, 0, width, height, fill="#071014", outline="", tags=("aug_overlay", "scene_viewport"))
        rect, body_rect = self._scene_viewport_rects(width, height)
        self._register_canvas_overlay_region("scene_viewport_panel", rect, projection=body_mode)
        canvas.create_rectangle(rect[0], rect[1], rect[2], rect[3], fill="#0b1218", outline="#33444d", width=1, tags=("aug_overlay", "scene_viewport"))
        index = self._current_headlight_inspector_index()
        title = f"Preset sceny {mode.upper()}"
        canvas.create_text(
            rect[0] + 10,
            rect[1] + 15,
            text=title,
            anchor=tk.W,
            fill="#bfffd6",
            font=("Segoe UI", 10, "bold"),
            tags=("aug_overlay", "scene_viewport"),
        )
        self._draw_scene_3d_panel_body(canvas, index, body_rect, True)
        self._draw_scene_axis_hint(canvas, body_rect, mode)

    def _scene_axis_gizmo_rect(self, rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(value) for value in rect]
        box_w = 118.0
        box_h = 78.0
        inspector_visible = bool(getattr(self, "_effect_inspector_visible", True))
        anchors = {
            True: (x2 - box_w - 18.0, y2 - box_h - 16.0),
            False: (x2 - box_w - 30.0, y2 - box_h - 16.0),
        }
        left, top = anchors[inspector_visible]
        left = max(x1 + 12.0, min(left, x2 - box_w - 8.0))
        top = max(y1 + 12.0, min(top, y2 - box_h - 8.0))
        return left, top, left + box_w, top + box_h

    def _draw_scene_axis_hint(self, canvas: tk.Canvas, rect: tuple[float, float, float, float], mode: str) -> None:
        try:
            mode = str(mode or "xy").lower()
            box_left, box_top, box_right, box_bottom = self._scene_axis_gizmo_rect(rect)
            ox = box_left + 42.0
            oy = box_bottom - 26.0
            canvas.create_rectangle(
                box_left,
                box_top,
                box_right,
                box_bottom,
                fill="#071014",
                outline="#3f5562",
                width=1,
                tags=("aug_overlay", "scene_viewport", "scene_axis_hint"),
            )
            canvas.create_text(
                box_left + 8,
                box_top + 9,
                text=f"orientacja {mode.upper()}",
                anchor=tk.W,
                fill="#d7dde1",
                font=("Segoe UI", 7, "bold"),
                tags=("aug_overlay", "scene_viewport", "scene_axis_hint"),
            )
            origin_rot = self._scene_view_rotate_point((0.0, 0.0, 0.0))
            axes = (
                ("X", (0.42, 0.0, 0.0), "#ff6b6b", "#ffb0b0"),
                ("Y", (0.0, 0.42, 0.0), "#58d68d", "#a7f0c2"),
                ("Z", (0.0, 0.0, 0.42), "#5cc8ff", "#b7eaff"),
            )
            axis_vectors: list[tuple[str, float, float, float, str, str]] = []
            max_screen_length = 0.001
            for label, endpoint, color, text_color in axes:
                rotated = self._scene_view_rotate_point(endpoint)
                dx = float(rotated[0]) - float(origin_rot[0])
                dy = -(float(rotated[2]) - float(origin_rot[2]))
                depth = float(rotated[1]) - float(origin_rot[1])
                screen_length = math.hypot(dx, dy)
                max_screen_length = max(max_screen_length, screen_length)
                axis_vectors.append((label, dx, dy, depth, color, text_color))

            draw_order = sorted(axis_vectors, key=lambda item: item[3])
            for label, dx, dy, depth, color, text_color in draw_order:
                screen_len = math.hypot(dx, dy)
                if screen_len < max_screen_length * 0.18:
                    radius = 9.0
                    fill = "#123044" if depth >= 0 else "#10161a"
                    canvas.create_oval(
                        ox - radius,
                        oy - radius,
                        ox + radius,
                        oy + radius,
                        fill=fill,
                        outline=color,
                        width=2,
                        tags=("aug_overlay", "scene_viewport", "scene_axis_hint"),
                    )
                    dot_radius = 2.6 if depth >= 0 else 5.5
                    canvas.create_oval(
                        ox - dot_radius,
                        oy - dot_radius,
                        ox + dot_radius,
                        oy + dot_radius,
                        fill=color if depth >= 0 else "",
                        outline=color,
                        width=1,
                        tags=("aug_overlay", "scene_viewport", "scene_axis_hint"),
                    )
                    canvas.create_text(
                        ox + 13,
                        oy - 1,
                        text=label,
                        anchor=tk.W,
                        fill=text_color,
                        font=("Segoe UI", 8, "bold"),
                        tags=("aug_overlay", "scene_viewport", "scene_axis_hint"),
                    )
                    continue
                length = 34.0 * (0.72 + 0.28 * min(1.0, screen_len / max_screen_length))
                ux = dx / screen_len
                uy = dy / screen_len
                end_x = ox + ux * length
                end_y = oy + uy * length
                width_px = 3 if depth >= 0 else 2
                canvas.create_line(
                    ox,
                    oy,
                    end_x,
                    end_y,
                    fill=color,
                    width=width_px,
                    arrow=tk.LAST,
                    arrowshape=(9, 10, 4),
                    tags=("aug_overlay", "scene_viewport", "scene_axis_hint"),
                )
                canvas.create_text(
                    end_x + ux * 5,
                    end_y + uy * 5,
                    text=label,
                    anchor=tk.CENTER,
                    fill=text_color,
                    font=("Segoe UI", 8, "bold"),
                    tags=("aug_overlay", "scene_viewport", "scene_axis_hint"),
                )
            canvas.create_oval(
                ox - 3,
                oy - 3,
                ox + 3,
                oy + 3,
                fill="#f6d365",
                outline="#071014",
                width=1,
                tags=("aug_overlay", "scene_viewport", "scene_axis_hint"),
            )
        except Exception:
            pass

    def _add_scene_projection_panel(self, parent, row: int, index: int) -> None:
        colors = self._inspector_setting_colors(parent)
        outer = tk.Frame(parent, bg=colors["border"], padx=1, pady=1, bd=0, highlightthickness=0)
        outer.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=(8, 0))
        outer.columnconfigure(0, weight=1)
        frame = tk.Frame(outer, bg=colors["body"], padx=9, pady=8, bd=0, highlightthickness=0)
        frame.grid(row=0, column=0, sticky=tk.EW)
        for column in range(3):
            frame.columnconfigure(column, weight=1)
        tk.Label(
            frame,
            text="Rzuty robocze R{}".format(index),
            bg=colors["header"],
            fg=colors["accent"],
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=4,
            anchor=tk.W,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.EW, pady=(0, 6))
        tk.Label(
            frame,
            text="S = źródło reflektora, T = punkt celowania na tablicy. Przeciągnij marker w rzucie.",
            bg=colors["body"],
            fg=colors["fg"],
            font=("Segoe UI", 8),
            justify=tk.LEFT,
            wraplength=285,
        ).grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=(0, 7))
        live_canvases = {}
        for key, canvas in getattr(self, "_scene_projection_canvases", {}).items():
            try:
                if bool(canvas.winfo_exists()):
                    live_canvases[key] = canvas
            except Exception:
                continue
        self._scene_projection_canvases = live_canvases
        self._scene_projection_specs = {
            key: spec for key, spec in getattr(self, "_scene_projection_specs", {}).items() if key in live_canvases
        }
        for column, projection in enumerate(("xy", "xz", "yz")):
            canvas = tk.Canvas(
                frame,
                width=88,
                height=92,
                bg=colors["body"],
                highlightthickness=1,
                highlightbackground=colors["border"],
                bd=0,
                cursor="hand2",
            )
            canvas.grid(row=2, column=column, sticky=tk.EW, padx=(0 if column == 0 else 5, 0))
            self._scene_projection_canvases[id(canvas)] = canvas
            self._scene_projection_specs[id(canvas)] = {"index": int(index), "projection": projection}
            canvas.bind("<Button-1>", self._on_scene_projection_press, add="+")
            canvas.bind("<B1-Motion>", self._on_scene_projection_drag, add="+")
            canvas.bind("<ButtonRelease-1>", self._on_scene_projection_release, add="+")
            self._draw_scene_projection_canvas(canvas, int(index), projection)

    def _draw_scene_projection_canvas(self, canvas: tk.Canvas, index: int, projection: str) -> None:
        try:
            self._enforce_scene_invariants()
            width = max(1, int(canvas.winfo_width() or canvas.cget("width") or 88))
            height = max(1, int(canvas.winfo_height() or canvas.cget("height") or 92))
            canvas.delete("all")
            try:
                bg = canvas.cget("background")
            except Exception:
                bg = "#141a16"
            line = self._blend_hex_colors("#6ee787", bg, 0.50)
            muted = self._blend_hex_colors("#94a3b8", bg, 0.58)
            plate = "#42d77d"
            source_color = "#ff9f1c"
            target_color = "#35d477"
            ranges = self._scene_projection_ranges(index)
            source_vars = self._headlight_world_var_triplet(index, "source")
            target_vars = self._headlight_world_var_triplet(index, "target")
            sx = self._read_float_var(source_vars[0], 0.0)
            sy = self._read_float_var(source_vars[1], 0.0)
            sz = self._read_float_var(source_vars[2], 0.95)
            tx = self._read_float_var(target_vars[0], 0.0)
            ty = self._read_float_var(target_vars[1], 0.0)
            if projection == "xy":
                title, x_range, y_range = "XY", ranges["x"], ranges["y"]
                source = (sx, sy)
                target = (tx, ty)
                plate_left, plate_top = self._projection_to_canvas(-0.5, 0.12, x_range, y_range, width, height)
                plate_right, plate_bottom = self._projection_to_canvas(0.5, -0.12, x_range, y_range, width, height)
                canvas.create_rectangle(plate_left, plate_top, plate_right, plate_bottom, outline=plate, width=1, dash=(3, 2))
            elif projection == "xz":
                title, x_range, y_range = "XZ", ranges["x"], ranges["z"]
                source = (sx, sz)
                target = (tx, 0.0)
                left, base_y = self._projection_to_canvas(-0.5, 0.0, x_range, y_range, width, height)
                right, _base_y = self._projection_to_canvas(0.5, 0.0, x_range, y_range, width, height)
                canvas.create_line(left, base_y, right, base_y, fill=plate, width=2)
            else:
                title, x_range, y_range = "YZ", ranges["y"], ranges["z"]
                source = (sy, sz)
                target = (ty, 0.0)
                left, base_y = self._projection_to_canvas(-0.12, 0.0, x_range, y_range, width, height)
                right, _base_y = self._projection_to_canvas(0.12, 0.0, x_range, y_range, width, height)
                canvas.create_line(left, base_y, right, base_y, fill=plate, width=2)
            zero_x, zero_y = self._projection_to_canvas(0.0, 0.0, x_range, y_range, width, height)
            canvas.create_line(12, zero_y, width - 10, zero_y, fill=muted, width=1)
            canvas.create_line(zero_x, 14, zero_x, height - 12, fill=muted, width=1)
            sxp, syp = self._projection_to_canvas(source[0], source[1], x_range, y_range, width, height)
            txp, typ = self._projection_to_canvas(target[0], target[1], x_range, y_range, width, height)
            canvas.create_line(sxp, syp, txp, typ, fill=line, width=2, arrow=tk.LAST)
            for x, y, color, label in (
                (txp, typ, target_color, "T"),
                (sxp, syp, source_color, "S"),
            ):
                radius = 5
                canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="#111827", width=1)
                canvas.create_text(x + 7, y - 7, text=label, anchor=tk.W, fill=color, font=("Segoe UI", 7, "bold"))
            canvas.create_text(6, 6, text=title, anchor=tk.NW, fill=line, font=("Segoe UI", 8, "bold"))
        except Exception:
            pass

    def _redraw_scene_projection_canvases(self) -> None:
        for key, canvas in list(getattr(self, "_scene_projection_canvases", {}).items()):
            try:
                if not bool(canvas.winfo_exists()):
                    self._scene_projection_canvases.pop(key, None)
                    self._scene_projection_specs.pop(key, None)
                    continue
                spec = self._scene_projection_specs.get(key, {})
                self._draw_scene_projection_canvas(canvas, int(spec.get("index", 1) or 1), str(spec.get("projection", "xy")))
            except Exception:
                pass

    def _scene_projection_handle_from_event(self, canvas: tk.Canvas, spec: dict, event) -> str:
        projection = str(spec.get("projection", "xy"))
        index = int(spec.get("index", 1) or 1)
        ranges = self._scene_projection_ranges(index)
        source_vars = self._headlight_world_var_triplet(index, "source")
        target_vars = self._headlight_world_var_triplet(index, "target")
        sx = self._read_float_var(source_vars[0], 0.0)
        sy = self._read_float_var(source_vars[1], 0.0)
        sz = self._read_float_var(source_vars[2], 0.95)
        tx = self._read_float_var(target_vars[0], 0.0)
        ty = self._read_float_var(target_vars[1], 0.0)
        width = max(1, int(canvas.winfo_width() or canvas.cget("width") or 88))
        height = max(1, int(canvas.winfo_height() or canvas.cget("height") or 92))
        if projection == "xy":
            x_range, y_range = ranges["x"], ranges["y"]
            source = self._projection_to_canvas(sx, sy, x_range, y_range, width, height)
            target = self._projection_to_canvas(tx, ty, x_range, y_range, width, height)
        elif projection == "xz":
            x_range, y_range = ranges["x"], ranges["z"]
            source = self._projection_to_canvas(sx, sz, x_range, y_range, width, height)
            target = self._projection_to_canvas(tx, 0.0, x_range, y_range, width, height)
        else:
            x_range, y_range = ranges["y"], ranges["z"]
            source = self._projection_to_canvas(sy, sz, x_range, y_range, width, height)
            target = self._projection_to_canvas(ty, 0.0, x_range, y_range, width, height)
        ex = float(getattr(event, "x", 0) or 0)
        ey = float(getattr(event, "y", 0) or 0)
        source_distance = math.hypot(ex - source[0], ey - source[1])
        target_distance = math.hypot(ex - target[0], ey - target[1])
        if source_distance <= 13 or target_distance <= 13:
            return "source" if source_distance <= target_distance else "target"
        if projection == "xy":
            value_x, value_y = self._projection_from_canvas(ex, ey, x_range, y_range, width, height)
            if -0.5 <= value_x <= 0.5 and -0.12 <= value_y <= 0.12:
                return "target"
        return "source"

    def _set_scene_projection_point(self, spec: dict, handle: str, event) -> None:
        projection = str(spec.get("projection", "xy"))
        index = int(spec.get("index", 1) or 1)
        canvas = spec.get("canvas")
        if canvas is None:
            return
        width = max(1, int(canvas.winfo_width() or canvas.cget("width") or 88))
        height = max(1, int(canvas.winfo_height() or canvas.cget("height") or 92))
        ranges = self._scene_projection_ranges(index)
        source_vars = self._headlight_world_var_triplet(index, "source")
        target_vars = self._headlight_world_var_triplet(index, "target")
        x = float(getattr(event, "x", 0) or 0)
        y = float(getattr(event, "y", 0) or 0)
        if projection == "xy":
            value_x, value_y = self._projection_from_canvas(x, y, ranges["x"], ranges["y"], width, height)
            if handle == "target":
                target_vars[0].set(max(-0.5, min(0.5, value_x)))
                target_vars[1].set(max(-0.12, min(0.12, value_y)))
            else:
                source_vars[0].set(max(-3.0, min(3.0, value_x)))
                source_vars[1].set(max(-3.0, min(3.0, value_y)))
        elif projection == "xz":
            value_x, value_z = self._projection_from_canvas(x, y, ranges["x"], ranges["z"], width, height)
            if handle == "target":
                target_vars[0].set(max(-0.5, min(0.5, value_x)))
            else:
                source_vars[0].set(max(-3.0, min(3.0, value_x)))
                source_vars[2].set(max(0.0, min(8.0, value_z)))
        else:
            value_y, value_z = self._projection_from_canvas(x, y, ranges["y"], ranges["z"], width, height)
            if handle == "target":
                target_vars[1].set(max(-0.12, min(0.12, value_y)))
            else:
                source_vars[1].set(max(-3.0, min(3.0, value_y)))
                source_vars[2].set(max(0.0, min(8.0, value_z)))
        self._enforce_scene_invariants()
        self._refresh_scene_summary()
        self._redraw_scene_projection_canvases()
        self._draw_vector_overlay()

    def _on_scene_projection_press(self, event):
        canvas = getattr(event, "widget", None)
        spec = dict(getattr(self, "_scene_projection_specs", {}).get(id(canvas), {}))
        if canvas is None or not spec:
            return
        spec["canvas"] = canvas
        handle = self._scene_projection_handle_from_event(canvas, spec, event)
        spec["handle"] = handle
        self._scene_projection_drag = spec
        self._begin_canvas_live_edit()
        self._set_scene_projection_point(spec, handle, event)
        return "break"

    def _on_scene_projection_drag(self, event):
        spec = getattr(self, "_scene_projection_drag", None)
        if not isinstance(spec, dict):
            return
        handle = str(spec.get("handle", "source"))
        self._set_scene_projection_point(spec, handle, event)
        return "break"

    def _on_scene_projection_release(self, event):
        spec = getattr(self, "_scene_projection_drag", None)
        if isinstance(spec, dict):
            handle = str(spec.get("handle", "source"))
            self._set_scene_projection_point(spec, handle, event)
        self._scene_projection_drag = None
        self._end_canvas_live_edit()
        return "break"

    def _scene_projection_axis_labels(self, projection: str) -> tuple[str, str]:
        projection = str(projection or "xy").lower()
        if projection == "3d":
            return "X", "Z"
        if projection == "xz":
            return "X", "Z"
        if projection == "yz":
            return "Y", "Z"
        return "X", "Y"

    def _scene_projection_overlay_layout(self, width: int, height: int) -> list[dict]:
        active = str(getattr(self, "_scene_projection_active", "3d") or "3d").lower()
        if active not in {"3d", "xy", "xz", "yz"}:
            active = "3d"
            self._scene_projection_active = active
        projections = ("3d", "xy", "xz", "yz")
        gap = 8
        if bool(getattr(self, "_scene_projection_expanded", False)):
            main_w = min(max(260, int(width * 0.36)), max(220, width - 24), 420)
            main_h = min(max(178, int(height * 0.34)), max(156, height - 168), 300)
            x = max(12, width - main_w - 12)
            y = 56
            items = [{"projection": active, "rect": (x, y, x + main_w, y + main_h), "main": True}]
            mini_w = max(64, min(96, int((main_w - gap * 3) / 4)))
            mini_h = max(62, min(86, int(mini_w * 0.72)))
            mini_y = y + main_h + gap
            if mini_y + mini_h > height - 10:
                mini_y = max(56, y - mini_h - gap)
            mini_x = x
            for offset, projection in enumerate(projections):
                rect = (mini_x + offset * (mini_w + gap), mini_y, mini_x + offset * (mini_w + gap) + mini_w, mini_y + mini_h)
                items.append({"projection": projection, "rect": rect, "main": False})
            return items
        panel_w = max(92, min(118, int((width - 36) / 2)))
        panel_h = max(72, min(92, int(height * 0.18)))
        block_w = panel_w * 2 + gap
        block_h = panel_h * 2 + gap
        x = max(12, width - block_w - 12)
        y = 58 if block_h + 70 <= height else max(48, height - block_h - 12)
        items = []
        for idx, projection in enumerate(projections):
            col = idx % 2
            row = idx // 2
            x1 = x + col * (panel_w + gap)
            y1 = y + row * (panel_h + gap)
            items.append({"projection": projection, "rect": (x1, y1, x1 + panel_w, y1 + panel_h), "main": False})
        return items

    def _scene_projection_values(self, index: int, projection: str) -> dict:
        ranges = self._scene_projection_ranges(index)
        source_vars = self._headlight_world_var_triplet(index, "source")
        target_vars = self._headlight_world_var_triplet(index, "target")
        sx = self._read_float_var(source_vars[0], 0.0)
        sy = self._read_float_var(source_vars[1], 0.0)
        sz = self._read_float_var(source_vars[2], 0.95)
        tx = self._read_float_var(target_vars[0], 0.0)
        ty = self._read_float_var(target_vars[1], 0.0)
        camera_z = self._read_float_var(self.scene_camera_z_var, 1.65)
        camera_target_x = self._read_float_var(self.scene_camera_target_x_var, 0.0)
        camera_target_y = self._read_float_var(self.scene_camera_target_y_var, 0.0)
        projection = str(projection or "xy").lower()
        if projection == "xz":
            return {
                "x_range": ranges["x"],
                "y_range": ranges["z"],
                "source": (sx, sz),
                "target": (tx, 0.0),
                "plate": ("line", -0.5, 0.5, 0.0),
            }
        if projection == "yz":
            return {
                "x_range": ranges["y"],
                "y_range": ranges["z"],
                "source": (sy, sz),
                "target": (ty, 0.0),
                "plate": ("line", -0.12, 0.12, 0.0),
            }
        return {
            "x_range": ranges["x"],
            "y_range": ranges["y"],
            "source": (sx, sy),
            "target": (tx, ty),
            "plate": ("rect", -0.5, -0.12, 0.5, 0.12),
        }

    def _draw_scene_projection_overlay(self, canvas: tk.Canvas, width: int, height: int) -> None:
        try:
            index = self._current_headlight_inspector_index()
            for item in self._scene_projection_overlay_layout(width, height):
                self._draw_scene_projection_overlay_panel(
                    canvas,
                    index,
                    str(item.get("projection", "xy")),
                    tuple(item.get("rect", (0, 0, 1, 1))),
                    bool(item.get("main", False)),
                )
        except Exception:
            pass

    def _draw_scene_projection_overlay_panel(
        self,
        canvas: tk.Canvas,
        index: int,
        projection: str,
        rect: tuple[float, float, float, float],
        main: bool,
    ) -> None:
        projection = str(projection or "xy").lower()
        if projection not in {"3d", "xy", "xz", "yz"}:
            projection = "xy"
        active = projection == str(getattr(self, "_scene_projection_active", "3d") or "3d").lower()
        x1, y1, x2, y2 = [float(value) for value in rect]
        panel_w = max(1.0, x2 - x1)
        panel_h = max(1.0, y2 - y1)
        outline = "#76e0a3" if active else "#596269"
        header_fill = "#15352a" if active else "#172027"
        title_fill = "#c8f8dc" if active else "#d7dde1"
        fill = "#0b1218"
        header_h = 24 if main else 20
        canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=(2 if active else 1), tags=("aug_overlay", "scene_projection"))
        self._register_canvas_overlay_region("scene_projection_panel", (x1, y1, x2, y2), projection=projection, index=index)
        canvas.create_rectangle(x1, y1, x2, y1 + header_h, fill=header_fill, outline="", tags=("aug_overlay", "scene_projection"))
        title = f"R{index} {projection.upper()}"
        canvas.create_text(
            x1 + 8,
            y1 + header_h / 2,
            text=title,
            anchor=tk.W,
            fill=title_fill,
            font=("Segoe UI", 8 if not main else 10, "bold"),
            tags=("aug_overlay", "scene_projection"),
        )
        expanded = bool(getattr(self, "_scene_projection_expanded", False))
        button_text = "MIN" if active and expanded else "MAX"
        button_w = 34 if main else 30
        bx1 = x2 - button_w - 5
        by1 = y1 + 4
        bx2 = x2 - 5
        by2 = y1 + header_h - 4
        canvas.create_rectangle(bx1, by1, bx2, by2, fill="#10161a", outline="#7fd8a8", width=1, tags=("aug_overlay", "scene_projection"))
        canvas.create_text(
            (bx1 + bx2) / 2,
            (by1 + by2) / 2,
            text=button_text,
            anchor=tk.CENTER,
            fill="#bfffd6",
            font=("Segoe UI", 6 if not main else 7, "bold"),
            tags=("aug_overlay", "scene_projection"),
        )
        self._register_canvas_overlay_region(
            "scene_projection_toggle",
            (bx1, by1, bx2, by2),
            projection=projection,
            index=index,
        )

        plot_x1 = x1 + (12 if main else 9)
        plot_y1 = y1 + header_h + (12 if main else 8)
        plot_x2 = x2 - (12 if main else 8)
        plot_y2 = y2 - (14 if main else 9)
        if projection == "3d":
            tex_w = 42 if main else 34
            tex_x2 = max(x1 + 42, bx1 - 5)
            tex_x1 = max(x1 + 6, tex_x2 - tex_w)
            tex_y1 = by1
            tex_y2 = by2
            texture_enabled = bool(self.scene_plate_texture_var.get())
            canvas.create_rectangle(
                tex_x1,
                tex_y1,
                tex_x2,
                tex_y2,
                fill=("#183124" if texture_enabled else "#171b20"),
                outline=("#7fd8a8" if texture_enabled else "#596269"),
                width=1,
                tags=("aug_overlay", "scene_projection"),
            )
            canvas.create_text(
                (tex_x1 + tex_x2) / 2,
                (tex_y1 + tex_y2) / 2,
                text=("IMG" if texture_enabled else "SKEL"),
                anchor=tk.CENTER,
                fill=("#bfffd6" if texture_enabled else "#cbd5dc"),
                font=("Segoe UI", 6 if not main else 7, "bold"),
                tags=("aug_overlay", "scene_projection"),
            )
            self._register_canvas_overlay_region(
                "scene_plate_texture_toggle",
                (tex_x1, tex_y1, tex_x2, tex_y2),
                projection=projection,
                index=index,
            )
            self._draw_scene_3d_panel_body(
                canvas,
                index,
                (plot_x1, plot_y1, plot_x2, plot_y2),
                main,
            )
            return
        plot_w = max(1.0, plot_x2 - plot_x1)
        plot_h = max(1.0, plot_y2 - plot_y1)
        pad = 14 if main else 9
        values = self._scene_projection_values(index, projection)
        x_range = values["x_range"]
        y_range = values["y_range"]
        axis_x, axis_y = self._scene_projection_axis_labels(projection)

        def to_abs(point: tuple[float, float]) -> tuple[float, float]:
            px, py = self._projection_to_canvas(point[0], point[1], x_range, y_range, int(plot_w), int(plot_h), pad=pad)
            return plot_x1 + px, plot_y1 + py

        zero_x, zero_y = to_abs((0.0, 0.0))
        muted = "#6e7c86"
        axis_color = "#9fb1bc"
        canvas.create_line(plot_x1 + pad, zero_y, plot_x2 - pad, zero_y, fill=muted, width=1, arrow=tk.LAST, tags=("aug_overlay", "scene_projection"))
        canvas.create_line(zero_x, plot_y2 - pad, zero_x, plot_y1 + pad, fill=muted, width=1, arrow=tk.LAST, tags=("aug_overlay", "scene_projection"))
        canvas.create_text(plot_x2 - pad + 4, zero_y, text=axis_x, anchor=tk.W, fill=axis_color, font=("Segoe UI", 7 if not main else 9, "bold"), tags=("aug_overlay", "scene_projection"))
        canvas.create_text(zero_x + 4, plot_y1 + pad, text=axis_y, anchor=tk.W, fill=axis_color, font=("Segoe UI", 7 if not main else 9, "bold"), tags=("aug_overlay", "scene_projection"))

        plate_color = "#42d77d"
        plate_spec = values.get("plate")
        if plate_spec and plate_spec[0] == "rect":
            left, top = to_abs((plate_spec[1], plate_spec[4]))
            right, bottom = to_abs((plate_spec[3], plate_spec[2]))
            canvas.create_rectangle(left, top, right, bottom, outline=plate_color, width=1, dash=(4, 3), tags=("aug_overlay", "scene_projection"))
        elif plate_spec and plate_spec[0] == "line":
            left, base_y = to_abs((plate_spec[1], plate_spec[3]))
            right, _base_y = to_abs((plate_spec[2], plate_spec[3]))
            canvas.create_line(left, base_y, right, base_y, fill=plate_color, width=2, tags=("aug_overlay", "scene_projection"))
            if main:
                canvas.create_text(right + 5, base_y, text="tablica", anchor=tk.W, fill=plate_color, font=("Segoe UI", 7), tags=("aug_overlay", "scene_projection"))

        contour_wire_segments = self._scene_contour_wireframe_world_segments()
        if contour_wire_segments:
            def project_wire_point(point: tuple[float, float, float]) -> tuple[float, float]:
                px, py = self._project_scene_point_for_mode(point, projection)
                return to_abs((px, py))

            self._draw_projected_contour_wireframe(
                canvas,
                contour_wire_segments,
                project_wire_point,
                tags=("aug_overlay", "scene_projection", "contour_profile_wire"),
                main=main,
            )

        if main:
            canvas.create_text(
                x1 + 10,
                y2 - 8,
                text="C kamera",
                anchor=tk.SW,
                fill="#9fb1bc",
                font=("Segoe UI", 7),
                tags=("aug_overlay", "scene_projection"),
            )

    def _scene_view_angles(self) -> tuple[float, float, float]:
        return (
            self._read_float_var(self.scene_view_yaw_var, 0.0),
            self._read_float_var(self.scene_view_pitch_var, 89.0),
            self._read_float_var(self.scene_view_roll_var, 0.0),
        )

    def _scene_matrix_multiply(
        self,
        left: tuple[tuple[float, float, float], ...],
        right: tuple[tuple[float, float, float], ...],
    ) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            tuple(
                float(left[row][0]) * float(right[0][col])
                + float(left[row][1]) * float(right[1][col])
                + float(left[row][2]) * float(right[2][col])
                for col in range(3)
            )
            for row in range(3)
        )

    def _scene_axis_rotation_matrix(self, axis: str, angle_deg: float) -> tuple[tuple[float, float, float], ...]:
        radians = math.radians(float(angle_deg))
        c = math.cos(radians)
        s = math.sin(radians)
        axis = str(axis or "").lower()
        if axis == "x":
            return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
        if axis == "y":
            return ((c, 0.0, -s), (0.0, 1.0, 0.0), (s, 0.0, c))
        return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))

    def _scene_view_matrix_from_angles(self, yaw_deg: float, pitch_deg: float, roll_deg: float) -> tuple[tuple[float, float, float], ...]:
        # These axes match the screen projection used below: X is horizontal,
        # Z is vertical on screen, and Y is depth.
        yaw_matrix = self._scene_axis_rotation_matrix("z", yaw_deg)
        pitch_matrix = self._scene_axis_rotation_matrix("x", pitch_deg)
        roll_matrix = self._scene_axis_rotation_matrix("y", roll_deg)
        return self._scene_matrix_multiply(roll_matrix, self._scene_matrix_multiply(pitch_matrix, yaw_matrix))

    def _set_scene_view_rotation_matrix(self, matrix: tuple[tuple[float, float, float], ...]) -> None:
        self._scene_view_rotation_matrix = matrix
        self._scene_view_rotation_key = self._scene_view_angles()

    def _scene_view_matrix(self) -> tuple[tuple[float, float, float], ...]:
        matrix = getattr(self, "_scene_view_rotation_matrix", None)
        current_key = self._scene_view_angles()
        matrix_key = getattr(self, "_scene_view_rotation_key", None)
        if isinstance(matrix, tuple) and len(matrix) == 3 and (matrix_key is None or matrix_key == current_key):
            return matrix
        matrix = self._scene_view_matrix_from_angles(*current_key)
        self._set_scene_view_rotation_matrix(matrix)
        return matrix

    def _scene_view_rotate_point_with_matrix(
        self,
        point: tuple[float, float, float],
        matrix: tuple[tuple[float, float, float], ...],
    ) -> tuple[float, float, float]:
        x, y, z = float(point[0]), float(point[1]), float(point[2]) - 0.42
        return (
            float(matrix[0][0]) * x + float(matrix[0][1]) * y + float(matrix[0][2]) * z,
            float(matrix[1][0]) * x + float(matrix[1][1]) * y + float(matrix[1][2]) * z,
            float(matrix[2][0]) * x + float(matrix[2][1]) * y + float(matrix[2][2]) * z,
        )

    def _scene_view_rotate_point(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        return self._scene_view_rotate_point_with_matrix(point, self._scene_view_matrix())

    def _scene_view_screen_components(self, point: tuple[float, float, float]) -> tuple[float, float]:
        rx, ry, rz = self._scene_view_rotate_point(point)
        return rx, -rz

    def _sync_headlight_normalized_from_world(self, index: int, handle: str) -> None:
        handle = "target" if str(handle or "").lower() == "target" else "source"
        try:
            world_x_var, world_y_var = self._headlight_world_xy_vars(index, handle)
            norm_x, norm_y = self._headlight_world_xy_to_norm(
                self._read_float_var(world_x_var, 0.0),
                self._read_float_var(world_y_var, 0.0),
                source=handle != "target",
            )
            sx_var, sy_var, tx_var, ty_var, _strength_var, _warmth_var = self._headlight_var_group(index)
            if handle == "target":
                tx_var.set(norm_x)
                ty_var.set(norm_y)
            else:
                sx_var.set(norm_x)
                sy_var.set(norm_y)
        except Exception:
            pass

    def _scene_plate_texture_source_image(self):
        if not PIL_AVAILABLE:
            return None
        try:
            payload = getattr(self, "_last_preview_payload", {}) if hasattr(self, "_last_preview_payload") else {}
            if not isinstance(payload, dict):
                payload = {}
            show_original = bool(getattr(self, "_fullscreen_show_original", False))
            source_name = "original_rgb" if show_original else "augmented_rgb"
            source = payload.get(source_name)
            if source is None:
                fallback_name = "augmented_rgb" if show_original else "original_rgb"
                source = payload.get(fallback_name)
                source_name = fallback_name
            if source is not None:
                source_identity = id(source)
                return Image.fromarray(source).convert("RGB"), (
                    "payload",
                    source_name,
                    source_identity,
                    "plain",
                )
            sample = getattr(self, "_current_sample", None)
            if sample is None:
                return None
            path = Path(sample)
            image = Image.open(path).convert("RGB")
            return image, ("file", str(path), "plain")
        except Exception:
            return None

    def _scene_plate_original_source_image(self):
        if not PIL_AVAILABLE:
            return None
        try:
            sample = getattr(self, "_current_sample", None)
            if sample is not None:
                path = Path(sample)
                if path.exists():
                    stat = path.stat()
                    image = Image.open(path).convert("RGB")
                    return image, ("original_file", str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
            payload = getattr(self, "_last_preview_payload", {}) if hasattr(self, "_last_preview_payload") else {}
            if isinstance(payload, dict):
                source = payload.get("original_rgb")
                if source is not None:
                    array = np.asarray(source) if np is not None else source
                    shape = tuple(getattr(array, "shape", ()) or ())
                    checksum = 0
                    try:
                        if np is not None and shape:
                            checksum = int(np.asarray(array, dtype="uint8").reshape(-1)[:: max(1, int(np.asarray(array).size // 128) or 1)].sum())
                    except Exception:
                        checksum = 0
                    return Image.fromarray(source).convert("RGB"), ("original_payload", shape, checksum)
        except Exception:
            return None
        return None

    def _scene_contour_wireframe_world_segments(self) -> list[dict]:
        try:
            if not self._should_show_contour_wireframe():
                return []
            if np is None:
                return []
            relief_strength = max(0.0, min(3.0, self._read_float_var(self.dark_relief_var, 0.30)))
            image = None
            source_key = None
            sample = getattr(self, "_current_sample", None)
            if sample is not None:
                try:
                    path = Path(sample)
                    if path.exists():
                        stat = path.stat()
                        source_key = ("original_file", str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
                except Exception:
                    source_key = None
            if source_key is None:
                source = self._scene_plate_original_source_image()
                if source is None:
                    return []
                image, source_key = source
            relief_preset = normalize_relief_profile_preset(self.relief_profile_preset_var.get())
            relief_curve = relief_profile_curve_for_preset(relief_preset, getattr(self, "_relief_profile_curve", DEFAULT_RELIEF_PROFILE_CURVE))
            contour_sensitivity = max(0.0, min(1.0, self._read_float_var(self.contour_detection_sensitivity_var, 0.55)))
            plate_curve_strength = max(0.0, min(2.0, self._read_float_var(self.plate_reflect_curve_var, 0.0)))
            cache_key = ("contour_wireframe_v6", source_key, relief_preset, relief_curve, round(contour_sensitivity, 3), round(plate_curve_strength, 3))
            if (
                cache_key == getattr(self, "_scene_contour_wireframe_cache_key", None)
                and isinstance(getattr(self, "_scene_contour_wireframe_cache", None), list)
            ):
                normalized_segments = self._scene_contour_wireframe_cache
            else:
                if image is None:
                    try:
                        image = Image.open(Path(sample)).convert("RGB")
                    except Exception:
                        return []
                geometry_profile = AugmentationProfile(
                    task_target=self.target,
                    dark_relief_strength=0.30,
                    contour_detection_sensitivity=contour_sensitivity,
                    relief_profile_preset=relief_preset,
                    relief_profile_curve=relief_curve,
                ).normalized()
                normalized_segments = build_contour_profile_wireframe_segments(np.asarray(image), geometry_profile)
                self._scene_contour_wireframe_cache_key = cache_key
                self._scene_contour_wireframe_cache = normalized_segments
            if not normalized_segments:
                return []
            plate_w = 1.0
            plate_h = 0.24
            # This is a diagnostic mesh scale, not the physical relief height
            # used by augmentation. Side projections need visible Z lift so the
            # selected profile reads as a raised "snake" instead of a flat line.
            z_scale = max(0.060, min(0.260, 0.075 + relief_strength * 0.060))
            curve_gain = min(1.0, plate_curve_strength / 2.0)
            curve_amplitude = plate_w * (0.025 + 0.085 * curve_gain) * curve_gain if curve_gain > 0.001 else 0.0

            def to_world(point) -> tuple[float, float, float]:
                x01, y01, z01 = point
                world_x = (float(x01) - 0.5) * plate_w
                x_norm = max(-1.0, min(1.0, world_x / max(0.0001, plate_w * 0.5)))
                base_z = curve_amplitude * max(0.0, 1.0 - x_norm * x_norm)
                return (
                    world_x,
                    (0.5 - float(y01)) * plate_h,
                    base_z + max(0.0, min(1.0, float(z01))) * z_scale,
                )

            world_segments: list[dict] = []
            for segment in normalized_segments:
                try:
                    world_segments.append(
                        {
                            "start": to_world(segment.get("start", (0.0, 0.0, 0.0))),
                            "end": to_world(segment.get("end", (0.0, 0.0, 0.0))),
                            "kind": str(segment.get("kind", "wire") or "wire"),
                        }
                    )
                except Exception:
                    continue
            return world_segments
        except Exception:
            return []

    def _scene_plate_texture_photo(self, width: float, height: float):
        if not bool(self.scene_plate_texture_var.get()) or not PIL_AVAILABLE:
            return None
        try:
            max_w = max(32, int(width))
            max_h = max(18, int(height))
            source = self._scene_plate_texture_source_image()
            if source is None:
                return None
            image, source_key = source
            key = (source_key, max_w, max_h)
            if key == getattr(self, "_scene_plate_texture_cache_key", None) and getattr(self, "_scene_plate_texture_photo_ref", None) is not None:
                return self._scene_plate_texture_photo_ref
            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                resample = Image.LANCZOS
            image = image.copy()
            image.thumbnail((max_w, max_h), resample)
            if image.width < max_w or image.height < max_h:
                canvas_image = Image.new("RGB", (max_w, max_h), "#0b1218")
                canvas_image.paste(image, ((max_w - image.width) // 2, (max_h - image.height) // 2))
                image = canvas_image
            photo = ImageTk.PhotoImage(image)
            self._scene_plate_texture_cache_key = key
            self._scene_plate_texture_photo_ref = photo
            self._photo_refs.append(photo)
            self._photo_refs = self._photo_refs[-12:]
            return photo
        except Exception:
            return None

    def _solve_perspective_coefficients(
        self,
        source_points: list[tuple[float, float]],
        target_points: list[tuple[float, float]],
    ) -> tuple[float, ...] | None:
        if len(source_points) != 4 or len(target_points) != 4:
            return None
        matrix: list[list[float]] = []
        values: list[float] = []
        for (x, y), (u, v) in zip(target_points, source_points):
            x = float(x)
            y = float(y)
            u = float(u)
            v = float(v)
            matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
            matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
            values.extend([u, v])
        size = 8
        for col in range(size):
            pivot = max(range(col, size), key=lambda row: abs(matrix[row][col]))
            if abs(matrix[pivot][col]) < 1e-9:
                return None
            if pivot != col:
                matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
                values[col], values[pivot] = values[pivot], values[col]
            pivot_value = matrix[col][col]
            for idx in range(col, size):
                matrix[col][idx] /= pivot_value
            values[col] /= pivot_value
            for row in range(size):
                if row == col:
                    continue
                factor = matrix[row][col]
                if abs(factor) < 1e-12:
                    continue
                for idx in range(col, size):
                    matrix[row][idx] -= factor * matrix[col][idx]
                values[row] -= factor * values[col]
        return tuple(values)

    def _scene_plate_texture_quad_photo(self, plate_xy: list[tuple[float, float]]):
        if not bool(self.scene_plate_texture_var.get()) or not PIL_AVAILABLE or len(plate_xy) != 4:
            return None
        try:
            min_x = math.floor(min(point[0] for point in plate_xy))
            max_x = math.ceil(max(point[0] for point in plate_xy))
            min_y = math.floor(min(point[1] for point in plate_xy))
            max_y = math.ceil(max(point[1] for point in plate_xy))
            width = max(2, int(max_x - min_x))
            height = max(2, int(max_y - min_y))
            if width < 4 or height < 4:
                return None
            source = self._scene_plate_texture_source_image()
            if source is None:
                return None
            image, source_key = source
            try:
                resample = Image.Resampling.BICUBIC
            except Exception:
                resample = Image.BICUBIC
            source_w = max(32, min(1400, int(max(width, 32))))
            aspect = max(0.05, float(image.height) / max(1.0, float(image.width)))
            source_h = max(18, min(700, int(round(source_w * aspect))))
            image = image.copy().resize((source_w, source_h), resample)
            local_quad = [(float(x) - float(min_x), float(y) - float(min_y)) for x, y in plate_xy]
            rounded_quad = tuple((round(x, 2), round(y, 2)) for x, y in local_quad)
            key = (source_key, "flip_x_pi", width, height, rounded_quad)
            if key == getattr(self, "_scene_plate_texture_quad_cache_key", None) and getattr(self, "_scene_plate_texture_quad_photo_ref", None) is not None:
                return self._scene_plate_texture_quad_photo_ref, min_x, min_y, width, height
            source_points = [
                (0.0, float(source_h - 1)),
                (float(source_w - 1), float(source_h - 1)),
                (float(source_w - 1), 0.0),
                (0.0, 0.0),
            ]
            coeffs = self._solve_perspective_coefficients(source_points, local_quad)
            if coeffs is None:
                return None
            transform_group = getattr(Image, "Transform", None)
            transform = getattr(transform_group, "PERSPECTIVE", None) if transform_group is not None else None
            if transform is None:
                transform = getattr(Image, "PERSPECTIVE")
            warped = image.transform((width, height), transform, coeffs, resample=resample).convert("RGBA")
            mask = Image.new("L", (width, height), 0)
            ImageDraw.Draw(mask).polygon(local_quad, fill=255)
            warped.putalpha(mask)
            photo = ImageTk.PhotoImage(warped)
            self._scene_plate_texture_quad_cache_key = key
            self._scene_plate_texture_quad_photo_ref = photo
            self._photo_refs.append(photo)
            self._photo_refs = self._photo_refs[-12:]
            return photo, min_x, min_y, width, height
        except Exception:
            return None

    def _draw_scene_3d_panel_body(
        self,
        canvas: tk.Canvas,
        index: int,
        rect: tuple[float, float, float, float],
        main: bool,
    ) -> None:
        try:
            self._enforce_scene_invariants()
            x1, y1, x2, y2 = [float(value) for value in rect]
            plot_w = max(1.0, x2 - x1)
            plot_h = max(1.0, y2 - y1)
            camera_z = self._read_float_var(self.scene_camera_z_var, 1.65)
            camera_target_x = self._read_float_var(self.scene_camera_target_x_var, 0.0)
            camera_target_y = self._read_float_var(self.scene_camera_target_y_var, 0.0)
            plate = [(-0.5, -0.12, 0.0), (0.5, -0.12, 0.0), (0.5, 0.12, 0.0), (-0.5, 0.12, 0.0)]
            camera = (0.0, 0.0, camera_z)
            camera_target = (camera_target_x, camera_target_y, 0.0)
            contour_wire_segments = self._scene_contour_wireframe_world_segments()
            all_points = plate + [camera, camera_target]
            all_points.extend(self._headlight_scene_world_points(visible_only=True))
            for segment in contour_wire_segments:
                try:
                    all_points.extend([segment["start"], segment["end"]])
                except Exception:
                    continue
            yaw, pitch, _roll = self._scene_view_angles()
            scale_matrix = self._scene_view_matrix_from_angles(yaw, pitch, 0.0)
            scale_rotated = [
                self._scene_view_rotate_point_with_matrix(point, scale_matrix)
                for point in all_points
            ]
            # Roll is a screen-plane rotation. It must not change the auto-fit
            # scale, otherwise the plate appears to zoom in/out while rolling.
            scale_values = [(item[0], -item[2]) for item in scale_rotated]
            min_x = min(value[0] for value in scale_values)
            max_x = max(value[0] for value in scale_values)
            min_y = min(value[1] for value in scale_values)
            max_y = max(value[1] for value in scale_values)
            rotated = [self._scene_view_rotate_point(point) for point in all_points]
            scale = min(plot_w / max(0.15, max_x - min_x + 0.18), plot_h / max(0.15, max_y - min_y + 0.18))
            try:
                zoom_factor = float(getattr(self, "_scene_view_zoom", 1.0) or 1.0)
            except Exception:
                zoom_factor = 1.0
            scale *= max(0.20, min(8.0, zoom_factor))
            center_x = (x1 + x2) / 2.0 + float(getattr(self, "_scene_view_pan_x", 0.0) or 0.0)
            center_y = (y1 + y2) / 2.0 + float(getattr(self, "_scene_view_pan_y", 0.0) or 0.0)
            mid_x = (min_x + max_x) / 2.0
            mid_y = (min_y + max_y) / 2.0
            depth_values = [float(item[1]) for item in rotated]
            depth_min = min(depth_values)
            depth_span = max(0.0001, max(depth_values) - depth_min)

            def project(point: tuple[float, float, float]) -> tuple[float, float, float]:
                rx, ry, rz = self._scene_view_rotate_point(point)
                sx = center_x + (rx - mid_x) * scale
                sy = center_y + ((-rz) - mid_y) * scale
                return sx, sy, ry

            def depth_color(start, end, fill: str) -> str:
                try:
                    _sx, _sy, sd = project(start)
                    _ex, _ey, ed = project(end)
                    avg_depth = (float(sd) + float(ed)) / 2.0
                    close = 1.0 - max(0.0, min(1.0, (avg_depth - depth_min) / depth_span))
                    return self._blend_hex_colors(fill, "#071014", 0.36 + close * 0.54)
                except Exception:
                    return fill

            def draw_line(start, end, fill, width=1, dash=None, arrow=None, depth_shade: bool = True):
                sx, sy, _sd = project(start)
                ex, ey, _ed = project(end)
                line_fill = depth_color(start, end, fill) if depth_shade else fill
                canvas.create_line(sx, sy, ex, ey, fill=line_fill, width=width, dash=dash, arrow=arrow, tags=("aug_overlay", "scene_projection", "scene_3d"))
                return (sx, sy), (ex, ey)

            def draw_scene_polygon(points, *, fill: str, outline: str = "", width: int = 1, stipple: str | None = None):
                coords = []
                for point in points:
                    px, py, _depth = project(point)
                    coords.extend((px, py))
                options = {
                    "fill": fill,
                    "outline": outline,
                    "width": width,
                    "tags": ("aug_overlay", "scene_projection", "scene_3d"),
                }
                if stipple:
                    options["stipple"] = stipple
                canvas.create_polygon(*coords, **options)

            plate_points = [project(point) for point in plate]
            plate_xy = [(point[0], point[1]) for point in plate_points]
            flat_plate = [coord for point in plate_xy for coord in point]
            texture_enabled = bool(self.scene_plate_texture_var.get())
            if texture_enabled:
                quad_texture = self._scene_plate_texture_quad_photo(plate_xy)
                if quad_texture is not None:
                    photo, min_px, min_py, tex_w, tex_h = quad_texture
                    canvas.create_image(
                        min_px + tex_w / 2.0,
                        min_py + tex_h / 2.0,
                        image=photo,
                        anchor=tk.CENTER,
                        tags=("aug_overlay", "scene_projection", "scene_3d"),
                    )
                else:
                    canvas.create_polygon(*flat_plate, fill="#1b2d23", outline="", tags=("aug_overlay", "scene_projection", "scene_3d"))
            else:
                canvas.create_polygon(*flat_plate, fill="#0d1814", outline="", tags=("aug_overlay", "scene_projection", "scene_3d"))
            closed_plate = flat_plate + [flat_plate[0], flat_plate[1]]
            canvas.create_line(*closed_plate, fill="#42d77d", width=(2 if main else 1), tags=("aug_overlay", "scene_projection", "scene_3d"))
            if contour_wire_segments:
                def contour_wire_color(kind: str) -> str:
                    kind = str(kind or "").lower()
                    if kind == "base":
                        return "#24545a"
                    if kind == "strut":
                        return "#6ee0b8"
                    if kind == "ridge":
                        return "#ffd36a"
                    if kind == "edge":
                        return "#2f8188"
                    if kind == "rib":
                        return "#52e3d0"
                    return "#62f1dc"

                draw_order = {"base": 0, "edge": 1, "rail": 2, "rib": 3, "strut": 4, "ridge": 5}
                for segment in sorted(contour_wire_segments, key=lambda item: draw_order.get(str(item.get("kind", "wire")), 2)):
                    try:
                        kind = str(segment.get("kind", "wire") or "wire")
                        draw_line(
                            segment["start"],
                            segment["end"],
                            contour_wire_color(kind),
                            width=1,
                            dash=(2, 4) if kind == "base" else None,
                            arrow=None,
                            depth_shade=False,
                        )
                    except Exception:
                        continue
            axes = (
                ((0.0, 0.0, 0.0), (0.70, 0.0, 0.0), "X", "#ff6b6b"),
                ((0.0, 0.0, 0.0), (0.0, 0.34, 0.0), "Y", "#58d68d"),
                ((0.0, 0.0, 0.0), (0.0, 0.0, 0.70), "Z", "#5cc8ff"),
            )

            def draw_axis(start, end, label: str, color: str) -> None:
                start_xy, end_xy = draw_line(start, end, color, width=2 if main else 1, arrow=None, depth_shade=False)
                dx = float(end_xy[0]) - float(start_xy[0])
                dy = float(end_xy[1]) - float(start_xy[1])
                length = math.hypot(dx, dy)
                if length < (7.0 if main else 4.5):
                    radius = 8 if main else 5
                    canvas.create_oval(
                        start_xy[0] - radius,
                        start_xy[1] - radius,
                        start_xy[0] + radius,
                        start_xy[1] + radius,
                        fill="#071218",
                        outline=color,
                        width=2 if main else 1,
                        tags=("aug_overlay", "scene_projection", "scene_3d"),
                    )
                    dot_radius = 2.8 if main else 2.0
                    canvas.create_oval(
                        start_xy[0] - dot_radius,
                        start_xy[1] - dot_radius,
                        start_xy[0] + dot_radius,
                        start_xy[1] + dot_radius,
                        fill=color,
                        outline="",
                        tags=("aug_overlay", "scene_projection", "scene_3d"),
                    )
                    canvas.create_text(
                        start_xy[0] + radius + 4,
                        start_xy[1] - 4,
                        text=label,
                        anchor=tk.W,
                        fill=color,
                        font=("Segoe UI", 8 if main else 7, "bold"),
                        tags=("aug_overlay", "scene_projection", "scene_3d"),
                    )
                    return
                length = max(0.0001, length)
                ux, uy = dx / length, dy / length
                px, py = -uy, ux
                arrow_len = 10 if main else 8
                arrow_w = 4 if main else 3
                base_x = float(end_xy[0]) - ux * arrow_len
                base_y = float(end_xy[1]) - uy * arrow_len
                canvas.create_polygon(
                    float(end_xy[0]),
                    float(end_xy[1]),
                    base_x + px * arrow_w,
                    base_y + py * arrow_w,
                    base_x - px * arrow_w,
                    base_y - py * arrow_w,
                    fill=color,
                    outline="",
                    tags=("aug_overlay", "scene_projection", "scene_3d"),
                )
                canvas.create_text(end_xy[0] + 4, end_xy[1] - 4, text=label, anchor=tk.W, fill=color, font=("Segoe UI", 8 if main else 7, "bold"), tags=("aug_overlay", "scene_projection", "scene_3d"))

            for start, end, label, color in axes:
                draw_axis(start, end, label, color)

            yaw, pitch, roll = self._scene_view_angles()
            status = f"Euler Y{yaw:.0f} P{pitch:.0f} R{roll:.0f}"
            canvas.create_text(x1 + 8, y2 - 12, text=status, anchor=tk.SW, fill="#d7dde1", font=("Segoe UI", 7 if not main else 8, "bold"), tags=("aug_overlay", "scene_projection", "scene_3d"))
            if main:
                canvas.create_text(
                    x1 + 8,
                    y2 - 1,
                    text="Ctrl+LPM: yaw/pitch | Ctrl+Shift+LPM: roll",
                    anchor=tk.SW,
                    fill="#9fb1bc",
                    font=("Segoe UI", 7),
                    tags=("aug_overlay", "scene_projection", "scene_3d"),
                )
        except Exception:
            pass

    def _set_scene_projection_overlay_point(self, spec: dict, event) -> None:
        projection = str(spec.get("projection", "xy") or "xy").lower()
        index = int(spec.get("index", 1) or 1)
        handle = str(spec.get("handle", "source") or "source").lower()
        plot_x1, plot_y1, plot_x2, plot_y2 = [float(value) for value in spec.get("plot_rect", spec.get("rect", (0, 0, 1, 1)))]
        plot_w = max(1.0, plot_x2 - plot_x1)
        plot_h = max(1.0, plot_y2 - plot_y1)
        pad = int(spec.get("pad", 14 if plot_w >= 170 or plot_h >= 140 else 9) or 9)
        x = float(getattr(event, "x", 0) or 0) - plot_x1
        y = float(getattr(event, "y", 0) or 0) - plot_y1
        ranges = self._scene_projection_ranges(index)
        source_vars = self._headlight_world_var_triplet(index, "source")
        target_vars = self._headlight_world_var_triplet(index, "target")

        def set_if_changed(variable, value: float) -> None:
            try:
                value = float(value)
                if abs(self._read_float_var(variable, -9999.0) - value) > 0.0008:
                    variable.set(value)
            except Exception:
                try:
                    variable.set(value)
                except Exception:
                    pass

        if projection == "xy":
            value_x, value_y = self._projection_from_canvas(x, y, ranges["x"], ranges["y"], int(plot_w), int(plot_h), pad=pad)
            if handle == "target":
                set_if_changed(target_vars[0], max(-0.5, min(0.5, value_x)))
                set_if_changed(target_vars[1], max(-0.12, min(0.12, value_y)))
                set_if_changed(target_vars[2], 0.0)
            elif handle == "camera":
                pass
            else:
                set_if_changed(source_vars[0], max(-3.0, min(3.0, value_x)))
                set_if_changed(source_vars[1], max(-3.0, min(3.0, value_y)))
        elif projection == "xz":
            value_x, value_z = self._projection_from_canvas(x, y, ranges["x"], ranges["z"], int(plot_w), int(plot_h), pad=pad)
            if handle == "target":
                set_if_changed(target_vars[0], max(-0.5, min(0.5, value_x)))
                set_if_changed(target_vars[2], 0.0)
            elif handle == "camera":
                set_if_changed(self.scene_camera_z_var, max(0.35, min(4.0, value_z)))
            else:
                set_if_changed(source_vars[0], max(-3.0, min(3.0, value_x)))
                set_if_changed(source_vars[2], max(0.0, min(8.0, value_z)))
        else:
            value_y, value_z = self._projection_from_canvas(x, y, ranges["y"], ranges["z"], int(plot_w), int(plot_h), pad=pad)
            if handle == "target":
                set_if_changed(target_vars[1], max(-0.12, min(0.12, value_y)))
                set_if_changed(target_vars[2], 0.0)
            elif handle == "camera":
                set_if_changed(self.scene_camera_z_var, max(0.35, min(4.0, value_z)))
            else:
                set_if_changed(source_vars[1], max(-3.0, min(3.0, value_y)))
                set_if_changed(source_vars[2], max(0.0, min(8.0, value_z)))
        self._enforce_scene_invariants()
        self._refresh_scene_summary()

    def _active_toolbox_title(self) -> str:
        active_key = self._normalize_toolbox_key(self._active_toolbox)
        title = next((label for key, label in self._domain_toolboxes() if key == active_key), "Ustawienia")
        if active_key == "illumination":
            title = f"{title} - R{self._current_headlight_inspector_index()}"
        return title

    def _toolbox_fields(self, key: str) -> list[dict]:
        key = self._normalize_toolbox_key(key)
        if key == "geometry":
            if self.target != "plate":
                return [
                    {
                        "type": "section",
                        "label": "Geometria chroniona",
                        "text": "W torze znaków nie zmieniamy geometrii tablicy po rektyfikacji. Zmieniaj światło, materiał, pogodę i kamerę.",
                        "priority": True,
                    },
                ]
            return [
                {"label": "Obrót", "var": self.rotation_var, "from": -15, "to": 15, "step": 1},
            ]
        if key == "color":
            return [
                {"label": "Jasność", "type": "slider", "var": self.brightness_var, "from": 0.0, "to": 0.25, "step": 0.01},
                {"label": "Kontrast", "type": "slider", "var": self.contrast_var, "from": 0.0, "to": 0.25, "step": 0.01},
                {"label": "Nasycenie", "type": "slider", "var": self.saturation_var, "from": 0.0, "to": 3.0, "step": 0.01},
            ]
        if key == "weather":
            return self._toolbox_fields("rain")
        if key == "sensor":
            return [
                {"label": "Rozmycie", "type": "slider", "var": self.blur_strength_var, "from": 0.0, "to": 1.0, "step": 0.01},
                {"label": "Szum", "type": "slider", "var": self.noise_var, "from": 0.0, "to": 0.08, "step": 0.005},
                {"label": "Ziarno", "var": self.noise_grain_var, "from": 1, "to": 12, "step": 1},
                {"label": "Szum ISO", "type": "slider", "var": self.night_iso_noise_var, "from": 0.0, "to": 1.0, "step": 0.01},
                {"label": "Poświata", "type": "slider", "var": self.night_bloom_var, "from": 0.0, "to": 1.0, "step": 0.01},
            ]
        if key == "material":
            if self.target == "plate":
                return [
                    {
                        "type": "section",
                        "label": "Powierzchnia obrazu",
                        "text": "W torze tablic materiał lokalnej tablicy nie jest zmieniany osobno. Użyj pogody, światła i kamery.",
                        "priority": True,
                    },
                ]
            return [
                {
                    "type": "section",
                    "label": "Film wodny i odbicia tablicy",
                    "text": "Parametry fizycznej powierzchni tablicy: mokra tafla, połysk, soczewkowanie i odbicia materiałowe.",
                    "priority": True,
                },
                {"label": "Grubość filmu", "type": "slider", "var": self.water_film_var, "from": 0.0, "to": 1.0, "step": 0.01},
                {"label": "Nierówność tafli", "type": "slider", "var": self.water_film_unevenness_var, "from": 0.0, "to": 1.0, "step": 0.01},
                {"label": "Soczewkowanie filmu", "type": "slider", "var": self.water_film_lens_var, "from": 0.0, "to": 1.0, "step": 0.01},
                {"label": "Reakcja na kontury", "type": "slider", "var": self.water_film_contour_var, "from": 0.0, "to": 1.0, "step": 0.01},
                {"label": "Połysk filmu", "type": "slider", "var": self.water_film_gloss_var, "from": 0.0, "to": 1.0, "step": 0.01},
                {
                    "type": "section",
                    "label": "Odbicia powierzchni",
                    "text": "Odbicia wynikające z materiału tablicy, bez zmiany anotacji i bez symulowania zabrudzeń.",
                },
                {
                    "label": "Kierunkowe cieniowanie",
                    "type": "slider",
                    "var": self.plate_reflect_gradient_var,
                    "from": 0.0,
                    "to": 2.0,
                    "step": 0.02,
                    "hint": "Łagodny gradient światło/cień zgodny z kierunkiem reflektora. Strona bliżej źródła robi się jaśniejsza, przeciwna ciemniejsza. Połysk reguluje osobny suwak odblasku.",
                },
                {"label": "Odblask tablicy", "type": "slider", "var": self.plate_reflect_glare_var, "from": 0.0, "to": 2.0, "step": 0.02},
                {
                    "label": "Wygięcie poziome powierzchni",
                    "type": "slider",
                    "var": self.plate_reflect_curve_var,
                    "from": 0.0,
                    "to": 2.0,
                    "step": 0.02,
                    "hint": "Subtelna symulacja tablicy dokręconej do zaoblonego miejsca na karoserii: odbicie i perspektywa zmieniają się po osi poziomej.",
                },
            ]
        if key == "relief":
            if self.target == "plate":
                return [
                    {
                        "type": "section",
                        "label": "Kontury znaków",
                        "text": "Relief konturów działa na wyciętych tablicach w torze znaków.",
                        "priority": True,
                    },
                ]
            return [
                {
                    "type": "section",
                    "label": "Relief i kontury znaków",
                    "text": "Geometria tuszu: wypukłość znaków, cień lokalny i diagnostyka profilu.",
                    "priority": True,
                },
                {
                    "label": "Pokaż kontury i profil",
                    "var": self.rain_edge_debug_var,
                    "type": "check",
                    "hint": "Tryb diagnostyczny: pokazuje wykryte kontury oraz aproksymowany profil wypukłości używany przez cień, reflektory, film wodny i smugi. Nie jest zapisywany do datasetu.",
                },
                {
                    "label": "Czułość wykrywania konturów",
                    "type": "slider",
                    "var": self.contour_detection_sensitivity_var,
                    "from": 0.0,
                    "to": 1.0,
                    "step": 0.01,
                    "hint": "Wyższa wartość pomaga złapać słabe, przepalone lub rozmyte znaki. Niższa ogranicza fałszywe kontury z tła tablicy.",
                },
                {"label": "Wypukłość konturów", "type": "slider", "var": self.dark_relief_var, "from": 0.0, "to": 3.0, "step": 0.02},
                {
                    "label": "Profil przekroju",
                    "type": "profile_curve",
                    "hint": "Krzywa opisuje przekrój wypukłego tuszu od lewej krawędzi znaku do prawej. Przeciągnij dwa środkowe punkty, aby zmienić kształt profilu.",
                },
            ]
        if key == "dirt":
            if self.target == "plate":
                return [
                    {
                        "type": "section",
                        "label": "Zabrudzenia tablicy",
                        "text": "Błoto i smugi działają na wyciętych tablicach w torze znaków.",
                        "priority": True,
                    },
                ]
            return [
                {
                    "type": "section",
                    "label": "Błoto: grudki i smugi",
                    "text": "Fizyczne zabrudzenia na tablicy. Grudki mają masę i krycie, a smugi rozciągają się zgodnie z ruchem i wiatrem.",
                    "priority": True,
                },
                {"label": "Liczba grudek błota", "var": self.dirt_flow_points_var, "from": 0, "to": 2000, "step": 1, "width": 7},
                {"label": "Zakres masy grudek", "type": "range", "min_var": self.dirt_flow_mass_min_var, "max_var": self.dirt_flow_mass_max_var, "from": 0.05, "to": 2.8, "step": 0.05},
                {"label": "Smugi błota", "type": "slider", "var": self.dirt_flow_trail_length_var, "from": 0.0, "to": 1.0, "step": 0.05},
                {"label": "Wilgoć błota", "type": "slider", "var": self.dirt_flow_humidity_var, "from": 0.0, "to": 1.0, "step": 0.05},
                {"label": "Zakres lepkości", "type": "range_slider", "min_var": self.dirt_flow_stickiness_min_var, "max_var": self.dirt_flow_stickiness_max_var, "from": 0.0, "to": 1.0, "step": 0.05},
                {"label": "Zakres krycia", "type": "range", "min_var": self.dirt_flow_opacity_min_var, "max_var": self.dirt_flow_opacity_max_var, "from": 0.0, "to": 1.0, "step": 0.05},
                {"label": "Stop na konturze znaku", "var": self.dirt_flow_stop_on_contour_var, "type": "check"},
                {"label": "Prędkość/uderzenie", "type": "slider", "var": self.vehicle_speed_var, "from": 0.0, "to": 1.0, "step": 0.05},
                {"label": "Połysk mokrego błota", "type": "slider", "var": self.wet_mud_gloss_var, "from": 0.0, "to": 1.0, "step": 0.01},
            ]
        if key == "illumination":
            return [
                {
                    "type": "section",
                    "label": "Układ sceny",
                    "text": "Kamera patrzy frontalnie w środek rektyfikowanej tablicy. Z reguluje dystans obserwatora i siłę odbić widocznych w kamerze.",
                    "priority": True,
                },
                {
                    "label": "Dystans obserwatora (Z)",
                    "var": self.scene_camera_z_var,
                    "from": 0.35,
                    "to": 4.0,
                    "step": 0.01,
                    "width": 8,
                    "hint": "Kamera pozostaje frontalna i patrzy w środek tablicy. Z nie obraca sceny; określa dystans obserwatora, więc wpływa głównie na połysk, odbicia i ilość światła wracającego do kamery.",
                },
                {
                    "type": "section",
                    "label": "Ekspozycja światła",
                    "text": "Światło bazowe jest ambientem: równomiernie doświetla całą scenę. Reflektory R1/R2/R3 budują dopiero lokalne plamy światła.",
                },
                {"label": "Noc", "type": "slider", "var": self.night_var, "from": 0.0, "to": 1.0, "step": 0.01},
                {
                    "label": "Światło bazowe",
                    "type": "slider",
                    "var": self.night_light_var,
                    "from": 0.0,
                    "to": 1.0,
                    "step": 0.01,
                    "hint": "Globalny ambient sceny: równomiernie rozjaśnia całą tablicę, bez kierunku, hotspotów i plam. Reflektory odpowiadają za lokalne światło.",
                },
                {
                    "label": "Barwa ambientu (zimna -> ciepła)",
                    "type": "slider",
                    "var": self.night_warmth_var,
                    "from": 0.0,
                    "to": 1.0,
                    "step": 0.01,
                    "hint": "Kolor światła bazowego: lewa strona daje chłodniejszy ambient, prawa cieplejszy. Jasność nadal kontroluje suwak światła bazowego.",
                },
                *(
                    [
                        {
                            "type": "section",
                            "label": "Odbicia wtórne",
                            "text": "Rozproszenie energii światła po wypukłych krawędziach znaków.",
                        },
                        {
                            "label": "Głębia odbić",
                            "var": self.relief_bounce_depth_var,
                            "from": 1,
                            "to": 4,
                            "step": 1,
                            "width": 5,
                            "hint": "Liczba przebiegów rozproszenia światła po wypukłych krawędziach. 1 oznacza model bez odbić wtórnych.",
                        },
                        {
                            "label": "Siła odbić wtórnych",
                            "type": "slider",
                            "var": self.relief_bounce_strength_var,
                            "from": 0.0,
                            "to": 1.0,
                            "step": 0.01,
                            "hint": "Ile energii z pierwszej oświetlonej krawędzi ma trafić na pobliskie kontury znaków.",
                        },
                    ]
                    if self.target != "plate"
                    else []
                ),
                *(
                    [
                        {
                            "type": "section",
                            "label": "Przesłonięcia światła",
                            "text": "Cień rzucany przez element nad tablicą, np. rant karoserii lub daszek nad wnęką.",
                        },
                        {"label": "Cień daszka", "type": "slider", "var": self.overhang_shadow_var, "from": 0.0, "to": 1.0, "step": 0.01},
                        {"label": "Zasięg cienia", "type": "slider", "var": self.overhang_shadow_depth_var, "from": 0.0, "to": 0.60, "step": 0.01},
                        {"label": "Skos cienia", "type": "slider", "var": self.overhang_shadow_skew_var, "from": -1.0, "to": 1.0, "step": 0.01},
                    ]
                    if self.target != "plate"
                    else []
                ),
            ]
        if key == "rain":
            return [
                {"label": "Gęstość deszczu", "type": "slider", "var": self.rain_var, "from": 0.0, "to": 1.0, "step": 0.01},
                {
                    "label": "Rozmiar kropli",
                    "type": "range_slider",
                    "min_var": self.rain_drop_size_min_var,
                    "max_var": self.rain_drop_size_max_var,
                    "from": 0.0,
                    "to": 1.0,
                    "step": 0.005,
                    "hint": "Zakres losowanego rozmiaru kropli. Lewy suwak ustawia najmniejsze krople, prawy największe; generator losuje wartości pomiędzy nimi.",
                },
                {"label": "Alfa kropli", "type": "slider", "var": self.rain_alpha_var, "from": 0.0, "to": 1.0, "step": 0.01},
                {"label": "Soczewka kropli", "type": "slider", "var": self.rain_lens_var, "from": 0.0, "to": 1.0, "step": 0.01},
                {"label": "Pole wektorowe", "type": "slider", "var": self.rain_vector_field_var, "from": 0.0, "to": 1.0, "step": 0.01},
                {"label": "Wiry lokalne", "type": "slider", "var": self.rain_vortex_var, "from": 0.0, "to": 1.0, "step": 0.01},
                {
                    "label": "Efekt Tyndalla",
                    "type": "slider",
                    "var": self.tyndall_var,
                    "from": 0.0,
                    "to": 1.0,
                    "step": 0.01,
                    "hint": "Widocznosc smugi swiatla w deszczu lub mgielce przy konturach. To nie jest samodzielne zrodlo swiatla.",
                },
                {
                    "label": "Mgiełka przy konturach",
                    "type": "slider",
                    "var": self.rain_edge_mist_var,
                    "from": 0.0,
                    "to": 1.0,
                    "step": 0.01,
                    "hint": "To nie jest ogólna mgła sceny. Ten efekt symuluje drobną mgiełkę/rozbryzg tam, gdzie krople uderzają w wykryte kontury znaków lub krawędzi.",
                },
                {
                    "label": "Zasięg mgiełki",
                    "type": "slider",
                    "var": self.rain_edge_mist_radius_var,
                    "from": 0.0,
                    "to": 1.0,
                    "step": 0.01,
                    "hint": "Odległość rozproszenia od konturu trafionego kroplą. Większa wartość daje szerszą poświatę/mgiełkę wokół uderzonej krawędzi.",
                },
            ]
        return []

    def _add_toolbox_spin(self, parent, row: int, col: int, label: str, variable, from_, to, increment, width=7):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky=tk.W, padx=(0, 4), pady=1)
        ttk.Spinbox(
            parent,
            from_=from_,
            to=to,
            increment=increment,
            textvariable=variable,
            width=width,
        ).grid(row=row, column=col + 1, sticky=tk.EW, padx=(0, 8), pady=1)

    def _add_toolbox_range(self, parent, row: int, col: int, label: str, min_var, max_var, from_, to, increment, width=6):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky=tk.W, padx=(0, 4), pady=2)
        pair = ttk.Frame(parent)
        pair.grid(row=row, column=col + 1, sticky=tk.EW, padx=(0, 12), pady=2)
        pair.columnconfigure(0, weight=1)
        pair.columnconfigure(2, weight=1)
        ttk.Spinbox(
            pair,
            from_=from_,
            to=to,
            increment=increment,
            textvariable=min_var,
            width=width,
        ).grid(row=0, column=0, sticky=tk.EW)
        ttk.Label(pair, text=" - ").grid(row=0, column=1, padx=2)
        ttk.Spinbox(
            pair,
            from_=from_,
            to=to,
            increment=increment,
            textvariable=max_var,
            width=width,
        ).grid(row=0, column=2, sticky=tk.EW)

    def _format_toolbox_value(self, value, step) -> str:
        try:
            value = float(value)
            step = float(step)
        except Exception:
            return str(value)
        if step >= 1:
            return str(int(round(value)))
        if step < 0.01:
            return f"{value:.3f}"
        return f"{value:.2f}"

    def _add_toolbox_slider(self, parent, row: int, col: int, label: str, variable, from_, to, increment):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky=tk.W, padx=(0, 4), pady=1)
        box = ttk.Frame(parent)
        box.grid(row=row, column=col + 1, sticky=tk.EW, padx=(0, 8), pady=1)
        box.columnconfigure(0, weight=1)
        value_lbl = ttk.Label(box, text=self._format_toolbox_value(variable.get(), increment), width=5, anchor=tk.E)
        scale = ttk.Scale(
            box,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=lambda value, lbl=value_lbl, step=increment: lbl.configure(text=self._format_toolbox_value(value, step)),
        )
        scale.grid(row=0, column=0, sticky=tk.EW)
        value_lbl.grid(row=0, column=1, sticky=tk.E, padx=(4, 0))

    def _add_toolbox_range_slider(self, parent, row: int, col: int, label: str, min_var, max_var, from_, to, increment):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky=tk.NW, padx=(0, 4), pady=1)
        pair = ttk.Frame(parent)
        pair.grid(row=row, column=col + 1, columnspan=5, sticky=tk.EW, padx=(0, 8), pady=1)
        pair.columnconfigure(1, weight=1)

        def add_line(line: int, text: str, variable):
            ttk.Label(pair, text=text, width=3).grid(row=line, column=0, sticky=tk.W)
            value_lbl = ttk.Label(pair, text=self._format_toolbox_value(variable.get(), increment), width=5, anchor=tk.E)
            scale = ttk.Scale(
                pair,
                from_=from_,
                to=to,
                orient=tk.HORIZONTAL,
                variable=variable,
                command=lambda value, lbl=value_lbl, step=increment: lbl.configure(text=self._format_toolbox_value(value, step)),
            )
            scale.grid(row=line, column=1, sticky=tk.EW, pady=0)
            value_lbl.grid(row=line, column=2, sticky=tk.E, padx=(4, 0))

        add_line(0, "od", min_var)
        add_line(1, "do", max_var)

    def _bind_range_guards(self):
        pairs = (
            (self.rain_drop_size_min_var, self.rain_drop_size_max_var, 0.0, 1.0),
            (self.night_luma_min_var, self.night_luma_max_var, 0.0, 1.0),
            (self.dirt_flow_mass_min_var, self.dirt_flow_mass_max_var, 0.05, 2.8),
            (self.dirt_flow_stickiness_min_var, self.dirt_flow_stickiness_max_var, 0.0, 1.0),
            (self.dirt_flow_opacity_min_var, self.dirt_flow_opacity_max_var, 0.0, 1.0),
        )
        for min_var, max_var, lower, upper in pairs:
            min_var.trace_add(
                "write",
                lambda *_args, a=min_var, b=max_var, lo=lower, hi=upper: self._coerce_range_pair(a, b, lo, hi, "min"),
            )
            max_var.trace_add(
                "write",
                lambda *_args, a=min_var, b=max_var, lo=lower, hi=upper: self._coerce_range_pair(a, b, lo, hi, "max"),
            )
            self._coerce_range_pair(min_var, max_var, lower, upper, "max")

    def _coerce_range_pair(self, min_var, max_var, lower: float, upper: float, changed: str):
        if getattr(self, "_range_sync_guard", False):
            return
        try:
            min_value = float(min_var.get())
            max_value = float(max_var.get())
        except Exception:
            return
        min_value = max(lower, min(upper, min_value))
        max_value = max(lower, min(upper, max_value))
        if min_value > max_value:
            if changed == "min":
                min_value = max_value
            else:
                max_value = min_value
        self._range_sync_guard = True
        try:
            if abs(float(min_var.get()) - min_value) > 1e-9:
                min_var.set(min_value)
            if abs(float(max_var.get()) - max_value) > 1e-9:
                max_var.set(max_value)
        except Exception:
            pass
        finally:
            self._range_sync_guard = False

    def _profile_from_vars(self) -> AugmentationProfile:
        self._enforce_scene_invariants()
        is_plate_dataset = self.target == "plate"
        sample_size = self._read_int_var(
            self.sample_var,
            default=32,
            minimum=1,
            maximum=int(getattr(self, "sample_pool_limit", 10000) or 10000),
            remember_attr="_last_valid_sample_size",
            repair=True,
        )
        extra_count = self._read_int_var(
            self.extra_var,
            default=0,
            minimum=0,
            maximum=100000,
            remember_attr="_last_valid_extra_count",
            repair=True,
        )
        rain_strength = float(self.rain_var.get() or 0.0)
        rain_drop_size_min = float(self.rain_drop_size_min_var.get() or 0.0)
        rain_drop_size_max = float(self.rain_drop_size_max_var.get() or 0.0)
        rain_vector_field = float(self.rain_vector_field_var.get() or 0.0)
        rain_vortex = float(self.rain_vortex_var.get() or 0.0)
        rain_alpha = float(self.rain_alpha_var.get() or 0.0)
        rain_lens = float(self.rain_lens_var.get() or 0.0)
        rain_edge_mist = float(self.rain_edge_mist_var.get() or 0.0)
        rain_edge_mist_radius = float(self.rain_edge_mist_radius_var.get() or 0.0)
        tyndall_strength = float(self.tyndall_var.get() or 0.0)
        if rain_drop_size_min > rain_drop_size_max:
            rain_drop_size_min, rain_drop_size_max = rain_drop_size_max, rain_drop_size_min
        rain_drop_size = (rain_drop_size_min + rain_drop_size_max) / 2.0
        relief_strength = float(self.dark_relief_var.get() or 0.0)
        relief_profile_preset = normalize_relief_profile_preset(self.relief_profile_preset_var.get())
        relief_profile_curve = relief_profile_curve_for_preset(
            relief_profile_preset,
            getattr(self, "_relief_profile_curve", DEFAULT_RELIEF_PROFILE_CURVE),
        )
        contour_detection_sensitivity = self._read_float_var(self.contour_detection_sensitivity_var, 0.55)
        relief_bounce_depth = max(1, min(4, int(float(self.relief_bounce_depth_var.get() or 1))))
        relief_bounce_strength = float(self.relief_bounce_strength_var.get() or 0.0)
        camera_z = max(0.35, min(4.0, self._read_float_var(self.scene_camera_z_var, 1.65)))
        camera_target_x = 0.0
        camera_target_y = 0.0
        light_normal = self._camera_axis_strength_from_distance(camera_z, camera_target_x, camera_target_y)
        overhang_shadow = 0.0 if is_plate_dataset else float(self.overhang_shadow_var.get() or 0.0)
        overhang_shadow_depth = 0.60 if is_plate_dataset else float(self.overhang_shadow_depth_var.get() or 0.0)
        overhang_shadow_skew = 0.0 if is_plate_dataset else float(self.overhang_shadow_skew_var.get() or 0.0)
        reflect_gradient = 0.0 if is_plate_dataset else float(self.plate_reflect_gradient_var.get() or 0.0)
        reflect_glare = 0.0 if is_plate_dataset else float(self.plate_reflect_glare_var.get() or 0.0)
        reflect_curve = 0.0 if is_plate_dataset else float(self.plate_reflect_curve_var.get() or 0.0)
        night_light = float(self.night_light_var.get() or 0.0)
        night_bloom = float(self.night_bloom_var.get() or 0.0)
        night_iso_noise = float(self.night_iso_noise_var.get() or 0.0)
        night_warmth = float(self.night_warmth_var.get() or 0.0)
        traffic_headlight_raw = float(self.traffic_headlight_var.get() or 0.0)
        traffic_headlight = traffic_headlight_raw if self._headlight_effect_enabled(1) else 0.0
        traffic_headlight_1_warmth = float(self.traffic_headlight_1_warmth_var.get() or 0.0)
        traffic_headlight_1_r = float(self.traffic_headlight_1_r_var.get() or 0.0)
        traffic_headlight_1_g = float(self.traffic_headlight_1_g_var.get() or 0.0)
        traffic_headlight_1_b = float(self.traffic_headlight_1_b_var.get() or 0.0)
        traffic_headlight_1_cone = float(self.traffic_headlight_1_cone_var.get() or 0.45)
        traffic_headlight_1_source_radius = float(self.traffic_headlight_1_source_radius_var.get() or 0.0)
        traffic_headlight_2_raw = float(self.traffic_headlight_2_var.get() or 0.0)
        traffic_headlight_2 = traffic_headlight_2_raw if self._headlight_effect_enabled(2) else 0.0
        traffic_headlight_2_warmth = float(self.traffic_headlight_2_warmth_var.get() or 0.0)
        traffic_headlight_2_r = float(self.traffic_headlight_2_r_var.get() or 0.0)
        traffic_headlight_2_g = float(self.traffic_headlight_2_g_var.get() or 0.0)
        traffic_headlight_2_b = float(self.traffic_headlight_2_b_var.get() or 0.0)
        traffic_headlight_2_cone = float(self.traffic_headlight_2_cone_var.get() or 0.45)
        traffic_headlight_2_source_radius = float(self.traffic_headlight_2_source_radius_var.get() or 0.0)
        traffic_headlight_3_raw = float(self.traffic_headlight_3_var.get() or 0.0)
        traffic_headlight_3 = traffic_headlight_3_raw if self._headlight_effect_enabled(3) else 0.0
        traffic_headlight_3_warmth = float(self.traffic_headlight_3_warmth_var.get() or 0.0)
        traffic_headlight_3_r = float(self.traffic_headlight_3_r_var.get() or 0.0)
        traffic_headlight_3_g = float(self.traffic_headlight_3_g_var.get() or 0.0)
        traffic_headlight_3_b = float(self.traffic_headlight_3_b_var.get() or 0.0)
        traffic_headlight_3_cone = float(self.traffic_headlight_3_cone_var.get() or 0.45)
        traffic_headlight_3_source_radius = float(self.traffic_headlight_3_source_radius_var.get() or 0.0)
        traffic_headlight_count = sum(1 for value in (traffic_headlight, traffic_headlight_2, traffic_headlight_3) if value > 0.001)
        active_light_strength = max(0.0, min(1.0, max(night_light, traffic_headlight, traffic_headlight_2, traffic_headlight_3)))
        wet_mud_gloss = 0.0 if is_plate_dataset else float(self.wet_mud_gloss_var.get() or 0.0)
        water_film = 0.0 if is_plate_dataset else float(self.water_film_var.get() or 0.0)
        water_film_unevenness_value = self.water_film_unevenness_var.get()
        water_film_unevenness = 0.35 if is_plate_dataset or water_film_unevenness_value is None else float(water_film_unevenness_value)
        water_film_lens = 0.0 if is_plate_dataset else float(self.water_film_lens_var.get() or 0.0)
        water_film_contour_value = self.water_film_contour_var.get()
        water_film_contour = 0.55 if is_plate_dataset or water_film_contour_value is None else float(water_film_contour_value)
        water_film_gloss_value = self.water_film_gloss_var.get()
        water_film_gloss = 0.45 if is_plate_dataset or water_film_gloss_value is None else float(water_film_gloss_value)
        if not is_plate_dataset and water_film <= 0.001 and water_film_lens > 0.001:
            water_film = min(1.0, max(0.18, water_film_lens * 0.42))
        # Legacy fields stay at defaults for old profile compatibility; they
        # no longer control the night effect.
        night_luma_min = 0.56
        night_luma_max = 0.92
        wet_reflection_strength = max(0.0, min(1.0, rain_strength * active_light_strength * (0.80 + 0.60 * light_normal))) if is_plate_dataset else 0.0
        flare_strength = max(0.0, min(1.0, (active_light_strength - 0.58) * light_normal * 0.36))
        overexposure_strength = max(0.0, min(1.0, (active_light_strength - 0.62) * light_normal * 0.46))
        dirt_flow_points = 0 if is_plate_dataset else int(float(self.dirt_flow_points_var.get() or 0))
        dirt_flow_mass_min = 0.18 if is_plate_dataset else float(self.dirt_flow_mass_min_var.get() or 0.18)
        dirt_flow_mass_max = 1.0 if is_plate_dataset else float(self.dirt_flow_mass_max_var.get() or 1.0)
        dirt_flow_splash_scale = 0.65 if not is_plate_dataset else 0.0
        dirt_flow_trail_length = 0.0 if is_plate_dataset else float(self.dirt_flow_trail_length_var.get() or 0.0)
        dirt_flow_humidity = 0.0 if is_plate_dataset else float(self.dirt_flow_humidity_var.get() or 0.0)
        dirt_flow_stickiness_min = 0.0 if is_plate_dataset else float(self.dirt_flow_stickiness_min_var.get() or 0.0)
        dirt_flow_stickiness_max = 0.0 if is_plate_dataset else float(self.dirt_flow_stickiness_max_var.get() or 0.0)
        if dirt_flow_stickiness_min > dirt_flow_stickiness_max:
            dirt_flow_stickiness_min, dirt_flow_stickiness_max = dirt_flow_stickiness_max, dirt_flow_stickiness_min
        dirt_flow_stickiness = (dirt_flow_stickiness_min + dirt_flow_stickiness_max) / 2.0
        dirt_flow_opacity_min = 0.0 if is_plate_dataset else float(self.dirt_flow_opacity_min_var.get() or 0.0)
        dirt_flow_opacity_max = 0.0 if is_plate_dataset else float(self.dirt_flow_opacity_max_var.get() or 0.0)
        has_visible_effect = any(
            [
                abs(float(self.rotation_var.get() or 0.0)) > 0.001,
                float(self.brightness_var.get() or 0.0) > 0.001,
                float(self.contrast_var.get() or 0.0) > 0.001,
                abs(float(self.saturation_var.get() if self.saturation_var.get() is not None else 1.0) - 1.0) > 0.001,
                float(self.blur_strength_var.get() or 0.0) > 0.001,
                float(self.noise_var.get() or 0.0) > 0.001,
                rain_strength > 0.001,
                rain_strength > 0.001 and rain_vector_field > 0.001,
                rain_strength > 0.001 and rain_vortex > 0.001,
                rain_strength > 0.001 and rain_lens > 0.001,
                rain_strength > 0.001 and rain_edge_mist > 0.001,
                rain_strength > 0.001 and tyndall_strength > 0.001 and any(
                    value > 0.001 for value in (traffic_headlight, traffic_headlight_2, traffic_headlight_3)
                ),
                float(self.night_var.get() or 0.0) > 0.001,
                night_light > 0.001,
                night_bloom > 0.001,
                night_iso_noise > 0.001,
                traffic_headlight > 0.001,
                traffic_headlight_2 > 0.001,
                traffic_headlight_3 > 0.001,
                wet_mud_gloss > 0.001,
                water_film > 0.001,
                float(self.vehicle_speed_var.get() or 0.0) > 0.001,
                relief_strength > 0.001 and active_light_strength > 0.001,
                relief_strength > 0.001 and active_light_strength > 0.001 and relief_bounce_depth > 1 and relief_bounce_strength > 0.001,
                overhang_shadow > 0.001,
                reflect_gradient > 0.001,
                reflect_glare > 0.001,
                reflect_curve > 0.001,
                dirt_flow_points > 0,
                dirt_flow_trail_length > 0.001,
            ]
        )
        enabled = extra_count > 0 or has_visible_effect
        def headlight_norm_from_world(index: int, handle: str, fallback_x_var, fallback_y_var, *, source: bool) -> tuple[float, float]:
            try:
                world_x_var, world_y_var = self._headlight_world_xy_vars(index, handle)
                world_x = self._read_float_var(world_x_var, -999.0)
                world_y = self._read_float_var(world_y_var, -999.0)
                if world_x > -998.0 and world_y > -998.0:
                    return self._headlight_world_xy_to_norm(world_x, world_y, source=source)
            except Exception:
                pass
            return (
                self._read_float_var(fallback_x_var, -1.0),
                self._read_float_var(fallback_y_var, -1.0),
            )

        headlight_1_source_x, headlight_1_source_y = headlight_norm_from_world(
            1,
            "source",
            self.traffic_headlight_source_x_var,
            self.traffic_headlight_source_y_var,
            source=True,
        )
        headlight_1_target_x, headlight_1_target_y = headlight_norm_from_world(
            1,
            "target",
            self.traffic_headlight_target_x_var,
            self.traffic_headlight_target_y_var,
            source=False,
        )
        headlight_2_source_x, headlight_2_source_y = headlight_norm_from_world(
            2,
            "source",
            self.traffic_headlight_2_source_x_var,
            self.traffic_headlight_2_source_y_var,
            source=True,
        )
        headlight_2_target_x, headlight_2_target_y = headlight_norm_from_world(
            2,
            "target",
            self.traffic_headlight_2_target_x_var,
            self.traffic_headlight_2_target_y_var,
            source=False,
        )
        headlight_3_source_x, headlight_3_source_y = headlight_norm_from_world(
            3,
            "source",
            self.traffic_headlight_3_source_x_var,
            self.traffic_headlight_3_source_y_var,
            source=True,
        )
        headlight_3_target_x, headlight_3_target_y = headlight_norm_from_world(
            3,
            "target",
            self.traffic_headlight_3_target_x_var,
            self.traffic_headlight_3_target_y_var,
            source=False,
        )
        return AugmentationProfile(
            enabled=enabled,
            sample_size=sample_size,
            extra_count=extra_count,
            rotation_limit=float(self.rotation_var.get() or 0.0),
            translate_limit=0.0,
            scale_limit=0.0,
            brightness_limit=float(self.brightness_var.get() or 0.0),
            contrast_limit=float(self.contrast_var.get() or 0.0),
            saturation_limit=float(self.saturation_var.get() or 0.0),
            noise_strength=float(self.noise_var.get() or 0.0),
            noise_grain_size=int(float(self.noise_grain_var.get() or 1)),
            rain_strength=rain_strength,
            rain_drop_size=rain_drop_size,
            rain_drop_size_min=rain_drop_size_min,
            rain_drop_size_max=rain_drop_size_max,
            rain_vector_field_strength=rain_vector_field,
            rain_vortex_strength=rain_vortex,
            rain_alpha=rain_alpha,
            rain_lens_strength=rain_lens,
            rain_edge_mist_strength=rain_edge_mist,
            rain_edge_mist_radius=rain_edge_mist_radius,
            tyndall_strength=tyndall_strength,
            wet_reflection_strength=wet_reflection_strength,
            vehicle_speed=float(self.vehicle_speed_var.get() or 0.0),
            night_strength=float(self.night_var.get() or 0.0),
            night_luma_min=night_luma_min,
            night_luma_max=night_luma_max,
            night_light_strength=night_light,
            night_bloom_strength=night_bloom,
            night_iso_noise_strength=night_iso_noise,
            night_light_warmth=night_warmth,
            scene_plate_width=1.0,
            scene_plate_height=0.24,
            scene_camera_x=0.0,
            scene_camera_y=0.0,
            scene_camera_z=camera_z,
            scene_camera_target_x=camera_target_x,
            scene_camera_target_y=camera_target_y,
            scene_camera_target_z=0.0,
            scene_view_yaw=self._read_float_var(self.scene_view_yaw_var, 35.0),
            scene_view_pitch=self._read_float_var(self.scene_view_pitch_var, -24.0),
            scene_view_roll=self._read_float_var(self.scene_view_roll_var, 0.0),
            scene_plate_texture_enabled=bool(self.scene_plate_texture_var.get()),
            traffic_headlight_strength=traffic_headlight,
            traffic_headlight_count=traffic_headlight_count,
            traffic_headlight_source_x=headlight_1_source_x,
            traffic_headlight_source_y=headlight_1_source_y,
            traffic_headlight_target_x=headlight_1_target_x,
            traffic_headlight_target_y=headlight_1_target_y,
            traffic_headlight_source_world_x=self._read_float_var(self.traffic_headlight_source_world_x_var, -999.0),
            traffic_headlight_source_world_y=self._read_float_var(self.traffic_headlight_source_world_y_var, -999.0),
            traffic_headlight_source_world_z=self._read_float_var(self.traffic_headlight_source_world_z_var, 0.95),
            traffic_headlight_target_world_x=self._read_float_var(self.traffic_headlight_target_world_x_var, -999.0),
            traffic_headlight_target_world_y=self._read_float_var(self.traffic_headlight_target_world_y_var, -999.0),
            traffic_headlight_target_world_z=0.0,
            traffic_headlight_1_warmth=traffic_headlight_1_warmth,
            traffic_headlight_1_r=traffic_headlight_1_r,
            traffic_headlight_1_g=traffic_headlight_1_g,
            traffic_headlight_1_b=traffic_headlight_1_b,
            traffic_headlight_1_cone=traffic_headlight_1_cone,
            traffic_headlight_1_source_radius=traffic_headlight_1_source_radius,
            traffic_headlight_2_strength=traffic_headlight_2,
            traffic_headlight_2_warmth=traffic_headlight_2_warmth,
            traffic_headlight_2_r=traffic_headlight_2_r,
            traffic_headlight_2_g=traffic_headlight_2_g,
            traffic_headlight_2_b=traffic_headlight_2_b,
            traffic_headlight_2_cone=traffic_headlight_2_cone,
            traffic_headlight_2_source_radius=traffic_headlight_2_source_radius,
            traffic_headlight_2_source_x=headlight_2_source_x,
            traffic_headlight_2_source_y=headlight_2_source_y,
            traffic_headlight_2_target_x=headlight_2_target_x,
            traffic_headlight_2_target_y=headlight_2_target_y,
            traffic_headlight_2_source_world_x=self._read_float_var(self.traffic_headlight_2_source_world_x_var, -999.0),
            traffic_headlight_2_source_world_y=self._read_float_var(self.traffic_headlight_2_source_world_y_var, -999.0),
            traffic_headlight_2_source_world_z=self._read_float_var(self.traffic_headlight_2_source_world_z_var, 0.95),
            traffic_headlight_2_target_world_x=self._read_float_var(self.traffic_headlight_2_target_world_x_var, -999.0),
            traffic_headlight_2_target_world_y=self._read_float_var(self.traffic_headlight_2_target_world_y_var, -999.0),
            traffic_headlight_2_target_world_z=0.0,
            traffic_headlight_3_strength=traffic_headlight_3,
            traffic_headlight_3_warmth=traffic_headlight_3_warmth,
            traffic_headlight_3_r=traffic_headlight_3_r,
            traffic_headlight_3_g=traffic_headlight_3_g,
            traffic_headlight_3_b=traffic_headlight_3_b,
            traffic_headlight_3_cone=traffic_headlight_3_cone,
            traffic_headlight_3_source_radius=traffic_headlight_3_source_radius,
            traffic_headlight_3_source_x=headlight_3_source_x,
            traffic_headlight_3_source_y=headlight_3_source_y,
            traffic_headlight_3_target_x=headlight_3_target_x,
            traffic_headlight_3_target_y=headlight_3_target_y,
            traffic_headlight_3_source_world_x=self._read_float_var(self.traffic_headlight_3_source_world_x_var, -999.0),
            traffic_headlight_3_source_world_y=self._read_float_var(self.traffic_headlight_3_source_world_y_var, -999.0),
            traffic_headlight_3_source_world_z=self._read_float_var(self.traffic_headlight_3_source_world_z_var, 0.95),
            traffic_headlight_3_target_world_x=self._read_float_var(self.traffic_headlight_3_target_world_x_var, -999.0),
            traffic_headlight_3_target_world_y=self._read_float_var(self.traffic_headlight_3_target_world_y_var, -999.0),
            traffic_headlight_3_target_world_z=0.0,
            wet_mud_gloss_strength=wet_mud_gloss,
            water_film_strength=water_film,
            water_film_unevenness=water_film_unevenness,
            water_film_lens_strength=water_film_lens,
            water_film_contour_response=water_film_contour,
            water_film_gloss_strength=water_film_gloss,
            flare_strength=flare_strength,
            overexposure_strength=overexposure_strength,
            dirt_streak_strength=0.0,
            dirt_flow_strength=0.0,
            dirt_flow_points=dirt_flow_points,
            dirt_flow_mass_min=dirt_flow_mass_min,
            dirt_flow_mass_max=dirt_flow_mass_max,
            dirt_flow_splash_scale=dirt_flow_splash_scale,
            dirt_flow_trail_length=dirt_flow_trail_length,
            dirt_flow_humidity=dirt_flow_humidity,
            dirt_flow_stickiness=dirt_flow_stickiness,
            dirt_flow_stickiness_min=dirt_flow_stickiness_min,
            dirt_flow_stickiness_max=dirt_flow_stickiness_max,
            dirt_flow_air_angle=float(self.dirt_flow_air_angle_var.get() or 0.0),
            dirt_flow_wind_strength=float(self.dirt_flow_wind_strength_var.get() or 0.0),
            dirt_flow_gravity_angle=90.0,
            dirt_flow_gravity_strength=1.0,
            dirt_flow_opacity_min=dirt_flow_opacity_min,
            dirt_flow_opacity_max=dirt_flow_opacity_max,
            dirt_flow_stop_on_dark_contour=(False if is_plate_dataset else bool(self.dirt_flow_stop_on_contour_var.get())),
            contour_detection_sensitivity=contour_detection_sensitivity,
            dark_relief_strength=relief_strength,
            dark_relief_light_angle=135.0,
            relief_profile_preset=relief_profile_preset,
            relief_profile_curve=relief_profile_curve,
            relief_bounce_depth=relief_bounce_depth,
            relief_bounce_strength=relief_bounce_strength,
            light_normal_strength=light_normal,
            overhang_shadow_strength=overhang_shadow,
            overhang_shadow_depth=overhang_shadow_depth,
            overhang_shadow_skew=overhang_shadow_skew,
            plate_reflect_gradient_strength=reflect_gradient,
            plate_reflect_glare_strength=reflect_glare,
            plate_reflect_curve_strength=reflect_curve,
            blur_strength=float(self.blur_strength_var.get() or 0.0),
            blur_enabled=float(self.blur_strength_var.get() or 0.0) > 0.001,
            randomness_mode=self._selected_randomness_mode(),
            seed=int(getattr(self, "_effect_seed", 42) or 42),
            class_name=str(self.class_var.get() or "").strip(),
            task_target=self.target,
            manual_randomness=self._manual_randomness_config_for_profile(),
        ).normalized()

    def _reset_all_parameters(self):
        self.sample_var.set(min(32, int(getattr(self, "sample_pool_limit", 32) or 32)))
        self.extra_var.set(0)
        self.randomness_mode_var.set(self._randomness_mode_to_label("realistic"))
        self._manual_randomness = default_manual_randomness_config(self.target)
        self.rotation_var.set(0.0)
        self.brightness_var.set(0.0)
        self.contrast_var.set(0.0)
        self.saturation_var.set(1.0)
        self.noise_var.set(0.0)
        self.noise_grain_var.set(1)
        self.rain_var.set(0.0)
        self.rain_drop_size_var.set(0.07)
        self.rain_drop_size_min_var.set(0.01)
        self.rain_drop_size_max_var.set(0.13)
        self.rain_vector_field_var.set(0.0)
        self.rain_vortex_var.set(0.0)
        self.rain_alpha_var.set(0.22)
        self.rain_lens_var.set(0.0)
        self.rain_edge_mist_var.set(0.0)
        self.rain_edge_mist_radius_var.set(0.45)
        self.rain_edge_debug_var.set(False)
        self.tyndall_var.set(0.55)
        self.wet_reflection_var.set(0.0)
        self.vehicle_speed_var.set(0.0)
        self.night_var.set(0.0)
        self.night_luma_min_var.set(0.56)
        self.night_luma_max_var.set(0.92)
        self.night_light_var.set(DEFAULT_BASE_LIGHT_STRENGTH)
        self.night_bloom_var.set(0.0)
        self.night_iso_noise_var.set(0.0)
        self.night_warmth_var.set(0.35)
        self.scene_plate_width_var.set(1.0)
        self.scene_plate_height_var.set(0.24)
        self.scene_camera_x_var.set(0.0)
        self.scene_camera_y_var.set(0.0)
        self.scene_camera_z_var.set(1.65)
        self.scene_camera_target_x_var.set(0.0)
        self.scene_camera_target_y_var.set(0.0)
        self.scene_camera_target_z_var.set(0.0)
        self._scene_projection_active = "xy"
        self._scene_viewport_mode = "xy"
        self._scene_viewport_visible = True
        self._scene_view_pan_x = 0.0
        self._scene_view_pan_y = 0.0
        self._scene_view_zoom = 1.0
        self._scene_pan_drag = None
        self._apply_scene_view_preset("xy")
        self.scene_plate_texture_var.set(True)
        self.traffic_headlight_var.set(0.0)
        self.traffic_headlight_count_var.set(1)
        self.traffic_headlight_1_warmth_var.set(0.35)
        self.traffic_headlight_1_r_var.set(1.0)
        self.traffic_headlight_1_g_var.set(0.88)
        self.traffic_headlight_1_b_var.set(0.54)
        self.traffic_headlight_1_cone_var.set(0.45)
        self.traffic_headlight_1_source_radius_var.set(0.08)
        self.traffic_headlight_source_x_var.set(-1.0)
        self.traffic_headlight_source_y_var.set(-1.0)
        self.traffic_headlight_target_x_var.set(-1.0)
        self.traffic_headlight_target_y_var.set(-1.0)
        self.traffic_headlight_source_world_x_var.set(-0.56)
        self.traffic_headlight_source_world_y_var.set(-0.15)
        self.traffic_headlight_source_world_z_var.set(0.95)
        self.traffic_headlight_target_world_x_var.set(-0.08)
        self.traffic_headlight_target_world_y_var.set(0.01)
        self.traffic_headlight_target_world_z_var.set(0.0)
        self.traffic_headlight_2_var.set(0.0)
        self.traffic_headlight_2_warmth_var.set(0.35)
        self.traffic_headlight_2_r_var.set(1.0)
        self.traffic_headlight_2_g_var.set(0.88)
        self.traffic_headlight_2_b_var.set(0.54)
        self.traffic_headlight_2_cone_var.set(0.45)
        self.traffic_headlight_2_source_radius_var.set(0.08)
        self.traffic_headlight_2_source_x_var.set(-1.0)
        self.traffic_headlight_2_source_y_var.set(-1.0)
        self.traffic_headlight_2_target_x_var.set(-1.0)
        self.traffic_headlight_2_target_y_var.set(-1.0)
        self.traffic_headlight_2_source_world_x_var.set(0.56)
        self.traffic_headlight_2_source_world_y_var.set(-0.14)
        self.traffic_headlight_2_source_world_z_var.set(0.95)
        self.traffic_headlight_2_target_world_x_var.set(0.08)
        self.traffic_headlight_2_target_world_y_var.set(0.00)
        self.traffic_headlight_2_target_world_z_var.set(0.0)
        self.traffic_headlight_3_var.set(0.0)
        self.traffic_headlight_3_warmth_var.set(0.35)
        self.traffic_headlight_3_r_var.set(1.0)
        self.traffic_headlight_3_g_var.set(0.88)
        self.traffic_headlight_3_b_var.set(0.54)
        self.traffic_headlight_3_cone_var.set(0.45)
        self.traffic_headlight_3_source_radius_var.set(0.08)
        self.traffic_headlight_3_source_x_var.set(-1.0)
        self.traffic_headlight_3_source_y_var.set(-1.0)
        self.traffic_headlight_3_target_x_var.set(-1.0)
        self.traffic_headlight_3_target_y_var.set(-1.0)
        self.traffic_headlight_3_source_world_x_var.set(0.00)
        self.traffic_headlight_3_source_world_y_var.set(-0.18)
        self.traffic_headlight_3_source_world_z_var.set(0.95)
        self.traffic_headlight_3_target_world_x_var.set(0.00)
        self.traffic_headlight_3_target_world_y_var.set(0.03)
        self.traffic_headlight_3_target_world_z_var.set(0.0)
        self.traffic_headlight_1_enabled_var.set(False)
        self.traffic_headlight_2_enabled_var.set(False)
        self.traffic_headlight_3_enabled_var.set(False)
        self._active_headlight_index = 1
        self._headlight_visibility = {1: True, 2: False, 3: False}
        for index, visible in self._headlight_visibility.items():
            try:
                visibility_var = (getattr(self, "_headlight_visibility_vars", {}) or {}).get(index)
                if visibility_var is not None:
                    visibility_var.set(bool(visible))
            except Exception:
                pass
        self.wet_mud_gloss_var.set(0.0)
        self.water_film_var.set(0.0)
        self.water_film_unevenness_var.set(0.35)
        self.water_film_lens_var.set(0.0)
        self.water_film_contour_var.set(0.55)
        self.water_film_gloss_var.set(0.45)
        self.flare_var.set(0.0)
        self.overexposure_var.set(0.0)
        self.dirt_streak_var.set(0.0)
        self.dirt_flow_points_var.set(0)
        self.dirt_flow_mass_min_var.set(0.18)
        self.dirt_flow_mass_max_var.set(1.0)
        self.dirt_flow_splash_scale_var.set(0.45)
        self.dirt_flow_trail_length_var.set(0.55)
        self.dirt_flow_humidity_var.set(0.45)
        self.dirt_flow_stickiness_var.set(0.45)
        self.dirt_flow_stickiness_min_var.set(0.30)
        self.dirt_flow_stickiness_max_var.set(0.70)
        self.dirt_flow_air_angle_var.set(0.0)
        self.dirt_flow_wind_strength_var.set(0.45)
        self._close_wind_strength_editor(commit=False)
        self.dirt_flow_gravity_angle_var.set(90.0)
        self.dirt_flow_gravity_strength_var.set(1.0)
        self.dirt_flow_opacity_min_var.set(0.25)
        self.dirt_flow_opacity_max_var.set(0.80)
        self.dirt_flow_stop_on_contour_var.set(False)
        self.contour_detection_sensitivity_var.set(0.55)
        self.dark_relief_var.set(0.30)
        self.dark_relief_light_angle_var.set(135.0)
        self.relief_profile_preset_var.set(DEFAULT_RELIEF_PROFILE_PRESET)
        self._relief_profile_curve = normalize_relief_profile_curve(DEFAULT_RELIEF_PROFILE_CURVE)
        self._relief_profile_drag_index = None
        self._invalidate_contour_wireframe_cache()
        self.relief_bounce_depth_var.set(1)
        self.relief_bounce_strength_var.set(0.28)
        self.light_normal_var.set(0.0)
        self.overhang_shadow_var.set(0.0)
        self.overhang_shadow_depth_var.set(0.60)
        self.overhang_shadow_skew_var.set(0.0)
        self.plate_reflect_gradient_var.set(0.0)
        self.plate_reflect_glare_var.set(0.0)
        self.plate_reflect_curve_var.set(0.0)
        self.blur_strength_var.set(0.0)
        self.blur_var.set(False)
        if self.target == "plate":
            self.class_var.set("plate")
        else:
            self.class_var.set("")
        self._vector_tool = None
        self._vector_drag_start = None
        self._vector_drag_end = None
        self._wind_panel_drag = None
        self._headlight_drag = None
        self._headlight_radius_drag = None
        self._headlight_config_index = None
        self._headlight_panel_pos = None
        self._headlight_panel_anchor_index = None
        self._vector_positions.clear()
        self._schedule_preview_refresh(delay_ms=40)

    def _refresh_extra_count_title(self):
        count = self._read_int_var(
            self.extra_var,
            default=0,
            minimum=0,
            maximum=100000,
            remember_attr="_last_valid_extra_count",
            repair=False,
        )
        if count == 1:
            noun = "zdjęcie"
        elif count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
            noun = "zdjęcia"
        else:
            noun = "zdjęć"
        try:
            self.extra_count_title_var.set(f"Generuj dodatkowe {count} {noun} do train")
        except Exception:
            pass

    def _bind_realtime_preview(self):
        watched_vars = (
            self.rotation_var,
            self.brightness_var,
            self.contrast_var,
            self.saturation_var,
            self.noise_var,
            self.noise_grain_var,
            self.rain_var,
            self.rain_drop_size_var,
            self.rain_drop_size_min_var,
            self.rain_drop_size_max_var,
            self.rain_vector_field_var,
            self.rain_vortex_var,
            self.rain_alpha_var,
            self.rain_lens_var,
            self.rain_edge_mist_var,
            self.rain_edge_mist_radius_var,
            self.tyndall_var,
            self.vehicle_speed_var,
            self.night_var,
            self.night_light_var,
            self.night_bloom_var,
            self.night_iso_noise_var,
            self.night_warmth_var,
            self.scene_camera_z_var,
            self.scene_camera_target_x_var,
            self.scene_camera_target_y_var,
            self.traffic_headlight_var,
            self.traffic_headlight_count_var,
            self.traffic_headlight_1_warmth_var,
            self.traffic_headlight_1_r_var,
            self.traffic_headlight_1_g_var,
            self.traffic_headlight_1_b_var,
            self.traffic_headlight_1_cone_var,
            self.traffic_headlight_1_source_radius_var,
            self.traffic_headlight_source_x_var,
            self.traffic_headlight_source_y_var,
            self.traffic_headlight_target_x_var,
            self.traffic_headlight_target_y_var,
            self.traffic_headlight_source_world_x_var,
            self.traffic_headlight_source_world_y_var,
            self.traffic_headlight_source_world_z_var,
            self.traffic_headlight_target_world_x_var,
            self.traffic_headlight_target_world_y_var,
            self.traffic_headlight_2_var,
            self.traffic_headlight_2_warmth_var,
            self.traffic_headlight_2_r_var,
            self.traffic_headlight_2_g_var,
            self.traffic_headlight_2_b_var,
            self.traffic_headlight_2_cone_var,
            self.traffic_headlight_2_source_radius_var,
            self.traffic_headlight_2_source_x_var,
            self.traffic_headlight_2_source_y_var,
            self.traffic_headlight_2_target_x_var,
            self.traffic_headlight_2_target_y_var,
            self.traffic_headlight_2_source_world_x_var,
            self.traffic_headlight_2_source_world_y_var,
            self.traffic_headlight_2_source_world_z_var,
            self.traffic_headlight_2_target_world_x_var,
            self.traffic_headlight_2_target_world_y_var,
            self.traffic_headlight_3_var,
            self.traffic_headlight_3_warmth_var,
            self.traffic_headlight_3_r_var,
            self.traffic_headlight_3_g_var,
            self.traffic_headlight_3_b_var,
            self.traffic_headlight_3_cone_var,
            self.traffic_headlight_3_source_radius_var,
            self.traffic_headlight_3_source_x_var,
            self.traffic_headlight_3_source_y_var,
            self.traffic_headlight_3_target_x_var,
            self.traffic_headlight_3_target_y_var,
            self.traffic_headlight_3_source_world_x_var,
            self.traffic_headlight_3_source_world_y_var,
            self.traffic_headlight_3_source_world_z_var,
            self.traffic_headlight_3_target_world_x_var,
            self.traffic_headlight_3_target_world_y_var,
            self.traffic_headlight_1_enabled_var,
            self.traffic_headlight_2_enabled_var,
            self.traffic_headlight_3_enabled_var,
            self.wet_mud_gloss_var,
            self.water_film_var,
            self.water_film_unevenness_var,
            self.water_film_lens_var,
            self.water_film_contour_var,
            self.water_film_gloss_var,
            self.dirt_flow_points_var,
            self.dirt_flow_mass_min_var,
            self.dirt_flow_mass_max_var,
            self.dirt_flow_splash_scale_var,
            self.dirt_flow_trail_length_var,
            self.dirt_flow_humidity_var,
            self.dirt_flow_stickiness_min_var,
            self.dirt_flow_stickiness_max_var,
            self.dirt_flow_air_angle_var,
            self.dirt_flow_wind_strength_var,
            self.dirt_flow_opacity_min_var,
            self.dirt_flow_opacity_max_var,
            self.dirt_flow_stop_on_contour_var,
            self.contour_detection_sensitivity_var,
            self.dark_relief_var,
            self.relief_bounce_depth_var,
            self.relief_bounce_strength_var,
            self.overhang_shadow_var,
            self.overhang_shadow_depth_var,
            self.overhang_shadow_skew_var,
            self.plate_reflect_gradient_var,
            self.plate_reflect_glare_var,
            self.plate_reflect_curve_var,
            self.blur_strength_var,
        )
        for var in watched_vars:
            try:
                var.trace_add("write", lambda *_args: self._schedule_preview_refresh())
            except Exception:
                pass
        for index, var in (
            (1, self.traffic_headlight_var),
            (2, self.traffic_headlight_2_var),
            (3, self.traffic_headlight_3_var),
        ):
            try:
                var.trace_add("write", lambda *_args, idx=index, strength_var=var: self._arm_headlight_when_strength_is_set(idx, strength_var))
            except Exception:
                pass
        try:
            self.extra_var.trace_add("write", lambda *_args: self._refresh_extra_count_title())
        except Exception:
            pass
        try:
            self.rain_edge_debug_var.trace_add("write", lambda *_args: self._on_rain_edge_debug_toggle())
        except Exception:
            pass
        try:
            self.contour_detection_sensitivity_var.trace_add("write", lambda *_args: self._on_contour_geometry_setting_changed())
        except Exception:
            pass
        try:
            self.plate_reflect_curve_var.trace_add("write", lambda *_args: self._on_contour_geometry_setting_changed())
        except Exception:
            pass

    def _on_rain_edge_debug_toggle(self) -> None:
        try:
            enabled = bool(self.rain_edge_debug_var.get())
        except Exception:
            enabled = False
        self._scene_plate_texture_cache_key = None
        self._scene_plate_texture_photo_ref = None
        self._scene_plate_texture_quad_cache_key = None
        self._scene_plate_texture_quad_photo_ref = None
        self._invalidate_contour_wireframe_cache()
        try:
            self._draw_vector_overlay()
        except Exception:
            pass
        if not enabled:
            self._refresh_preview(redraw_only=True)
            return
        payload = getattr(self, "_last_preview_payload", {}) if hasattr(self, "_last_preview_payload") else {}
        if isinstance(payload, dict) and payload.get("rain_edge_debug_mask") is not None:
            self._refresh_preview(redraw_only=True)
        else:
            self._refresh_preview(redraw_only=False)

    def _invalidate_contour_wireframe_cache(self) -> None:
        self._scene_contour_wireframe_cache_key = None
        self._scene_contour_wireframe_cache = []

    def _on_contour_geometry_setting_changed(self) -> None:
        self._invalidate_contour_wireframe_cache()
        if hasattr(self, "_last_preview_payload"):
            try:
                delattr(self, "_last_preview_payload")
            except Exception:
                pass
        try:
            self._draw_vector_overlay()
        except Exception:
            pass
        if bool(self.rain_edge_debug_var.get()):
            self._schedule_preview_refresh(delay_ms=90)

    def _arm_headlight_when_strength_is_set(self, index: int, strength_var) -> None:
        try:
            if self._read_float_var(strength_var, 0.0) <= 0.001:
                return
            self._select_headlight(index, reveal=False)
            enabled_var = self._headlight_enabled_var(index)
            if not bool(enabled_var.get()):
                enabled_var.set(True)
            self._refresh_effect_inspector_title()
        except Exception:
            pass

    def _schedule_preview_refresh(self, delay_ms: int = 180):
        self._refresh_scene_summary()
        if bool(getattr(self, "_preview_refresh_suspended", False)):
            if self._preview_live_edit_active():
                self._preview_refresh_dirty = True
                return
            self._preview_refresh_suspended = False
            self._preview_refresh_dirty = False
        if self._preview_after_id is not None:
            try:
                self.window.after_cancel(self._preview_after_id)
            except Exception:
                pass
            self._preview_after_id = None
        try:
            self._preview_after_id = self.window.after(max(40, int(delay_ms)), self._run_scheduled_preview_refresh)
        except Exception:
            self._preview_after_id = None

    def _preview_live_edit_active(self) -> bool:
        return bool(
            getattr(self, "_canvas_slider_drag", None) is not None
            or bool(getattr(self, "_inspector_scale_drag_active", False))
            or getattr(self, "_relief_profile_drag_index", None) is not None
            or getattr(self, "_headlight_drag", None) is not None
            or getattr(self, "_scene_projection_drag", None) is not None
            or getattr(self, "_scene_view_drag", None) is not None
            or getattr(self, "_scene_pan_drag", None) is not None
        )

    def _run_scheduled_preview_refresh(self):
        self._preview_after_id = None
        self._refresh_preview()

    def _begin_canvas_live_edit(self):
        self._preview_refresh_suspended = True
        self._preview_refresh_dirty = False
        self._cancel_pending_preview()

    def _end_canvas_live_edit(self, delay_ms: int = 80):
        self._preview_refresh_suspended = False
        self._preview_refresh_dirty = False
        self._schedule_preview_refresh(delay_ms=delay_ms)

    def _grab_augmented_canvas_for_live_edit(self):
        try:
            canvas = getattr(self, "augmented_canvas", None)
            if canvas is not None:
                canvas.grab_set()
        except Exception:
            pass

    def _release_augmented_canvas_live_grab(self):
        try:
            canvas = getattr(self, "augmented_canvas", None)
            if canvas is not None:
                canvas.grab_release()
        except Exception:
            pass

    def _refresh_dependency_status(self):
        status = get_albumentations_status()
        available = bool(status.get("available"))
        self.dep_status_lbl.configure(text="Albumentations: dostępne" if available else "Albumentations: brak")
        self.dep_install_btn.configure(state=(tk.DISABLED if available else tk.NORMAL))

    def _install_dependency(self):
        if callable(self.install_callback):
            self.install_callback()
        self.dep_install_btn.configure(state=tk.DISABLED)
        self.dep_status_lbl.configure(text="Albumentations: instalacja w toku...")
        self.window.after(1200, self._poll_dependency_after_install)

    def _poll_dependency_after_install(self):
        self._refresh_dependency_status()
        if not bool(get_albumentations_status().get("available")):
            self.window.after(1200, self._poll_dependency_after_install)

    def _preview_sample_candidates(self) -> list[Path]:
        candidates = list(getattr(self, "sample_images", []) or [])
        if not candidates:
            return []
        try:
            limit = int(self.sample_var.get() or 0)
        except Exception:
            limit = 0
        if limit > 0:
            candidates = candidates[: max(1, min(len(candidates), limit))]
        return candidates

    def _preview_candidates_scope_key(self, candidates: list[Path]) -> str:
        if not candidates:
            return ""
        try:
            limit = int(self.sample_var.get() or 0)
        except Exception:
            limit = 0
        sample = list(candidates[:3])
        if len(candidates) > 6:
            sample.extend(candidates[-3:])
        else:
            sample = list(candidates)
        return f"{len(candidates)}|{limit}|" + "|".join(str(path) for path in sample)

    def _pick_random_sample(self):
        candidates = self._preview_sample_candidates()
        if not candidates:
            self.preview_status_lbl.configure(text="Brak obrazów źródłowych do losowania.")
            return
        if len(candidates) == 1:
            self._current_sample = candidates[0]
            self.preview_status_lbl.configure(text=f"{self._current_sample.name} | To jedyna próbka dostępna w podglądzie.")
        else:
            current_sample = self._current_sample
            pool_key = self._preview_candidates_scope_key(candidates)
            if pool_key != self._sample_shuffle_scope_key:
                self._sample_shuffle_scope_key = pool_key
                self._sample_shuffle_bag = []
            self._sample_shuffle_bag = [
                path
                for path in self._sample_shuffle_bag
                if path != current_sample
            ]
            if not self._sample_shuffle_bag:
                self._sample_shuffle_bag = [
                    path
                    for path in candidates
                    if path != current_sample
                ] or list(candidates)
                self._ui_random.shuffle(self._sample_shuffle_bag)
            self._current_sample = self._sample_shuffle_bag.pop()
        if hasattr(self, "_last_preview_payload"):
            try:
                delattr(self, "_last_preview_payload")
            except Exception:
                pass
        try:
            self.preview_status_lbl.configure(text=f"{Path(self._current_sample).name} | Losuję podgląd...")
        except Exception:
            pass
        self._request_raw_sample_preview()
        self._schedule_preview_refresh(delay_ms=900)

    def _request_raw_sample_preview(self):
        self._raw_preview_request_id += 1
        request_id = int(self._raw_preview_request_id)
        self._raw_preview_pending = True
        if self._raw_preview_worker_running:
            return
        self._start_raw_sample_preview_worker(request_id)

    def _start_raw_sample_preview_worker(self, request_id: int):
        sample = getattr(self, "_current_sample", None)
        if sample is None or not PIL_AVAILABLE or np is None:
            return
        try:
            canvas = getattr(self, "augmented_canvas", None)
            canvas_w = max(320, int(canvas.winfo_width() or 640)) if canvas is not None else 640
            canvas_h = max(220, int(canvas.winfo_height() or 420)) if canvas is not None else 420
        except Exception:
            canvas_w, canvas_h = 640, 420
        self._raw_preview_worker_running = True
        self._raw_preview_pending = False
        sample_path = Path(sample)

        def _worker():
            payload = None
            try:
                image = Image.open(sample_path).convert("RGB")
                resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR", 2)
                image.thumbnail((max(1, canvas_w - 12), max(1, canvas_h - 12)), resampling)
                payload = np.array(image)
            except Exception:
                payload = None

            def _finish():
                self._finish_raw_sample_preview(request_id, sample_path, payload)

            try:
                self.window.after(0, _finish)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_raw_sample_preview(self, request_id: int, sample_path: Path, array):
        self._raw_preview_worker_running = False
        try:
            if not bool(self.window.winfo_exists()):
                return
        except Exception:
            return
        is_current = Path(getattr(self, "_current_sample", "")) == sample_path
        if request_id != int(getattr(self, "_raw_preview_request_id", 0) or 0) or not is_current:
            if bool(getattr(self, "_raw_preview_pending", False)):
                self._start_raw_sample_preview_worker(int(getattr(self, "_raw_preview_request_id", request_id) or request_id))
            return
        if array is not None:
            try:
                self.preview_status_lbl.configure(text=f"{sample_path.name} | Podgląd surowy, efekt renderuje się po chwili.")
            except Exception:
                pass
            if bool(self.rain_edge_debug_var.get()):
                array = self._compose_rain_edge_debug_preview(array, None)
            self._draw_array(self.augmented_canvas, array)
        if bool(getattr(self, "_raw_preview_pending", False)):
            self._start_raw_sample_preview_worker(int(getattr(self, "_raw_preview_request_id", request_id) or request_id))

    def _reroll_effect_layout(self):
        self._effect_seed = self._ui_random.randint(1, 2_147_483_647)
        self._refresh_preview()

    def _capture_memory_snapshot(self, stage: str = "") -> dict:
        snapshot = {"available": False, "stage": stage}
        if psutil is None:
            self._memory_snapshot = snapshot
            return snapshot
        try:
            process = self._process_handle
            if process is None:
                process = psutil.Process()
                self._process_handle = process
            memory_info = process.memory_info()
            rss = float(memory_info.rss)
            peak_wset = float(getattr(memory_info, "peak_wset", rss) or rss)
            virtual = psutil.virtual_memory()
            snapshot = {
                "available": True,
                "stage": stage,
                "process_mb": rss / (1024.0 * 1024.0),
                "process_peak_mb": peak_wset / (1024.0 * 1024.0),
                "system_total_gb": float(virtual.total) / (1024.0 ** 3),
                "system_used_gb": float(virtual.total - virtual.available) / (1024.0 ** 3),
                "system_free_gb": float(virtual.available) / (1024.0 ** 3),
                "system_percent": float(virtual.percent),
            }
        except Exception:
            snapshot = {"available": False, "stage": stage}
        self._memory_snapshot = snapshot
        return snapshot

    def _format_memory_snapshot(self) -> str:
        snapshot = getattr(self, "_memory_snapshot", {}) or self._capture_memory_snapshot()
        stage = str(snapshot.get("stage") or "")
        prefix = f"{stage}: " if stage else ""
        if not snapshot.get("available"):
            return f"{prefix}RAM: brak danych"
        return (
            f"{prefix}RAM procesu {snapshot.get('process_mb', 0.0):.0f} MB | "
            f"peak {snapshot.get('process_peak_mb', 0.0):.0f} MB | "
            f"system {snapshot.get('system_used_gb', 0.0):.1f}/"
            f"{snapshot.get('system_total_gb', 0.0):.1f} GB "
            f"({snapshot.get('system_percent', 0.0):.0f}%) | "
            f"wolne {snapshot.get('system_free_gb', 0.0):.1f} GB"
        )

    def _refresh_preview(self, redraw_only: bool = False):
        if not redraw_only:
            self._raw_preview_request_id += 1
            self._raw_preview_pending = False
        if not redraw_only and self._preview_after_id is not None:
            try:
                self.window.after_cancel(self._preview_after_id)
            except Exception:
                pass
            self._preview_after_id = None
        if not PIL_AVAILABLE:
            self.preview_status_lbl.configure(text="Podgląd wymaga biblioteki Pillow.")
            return
        if self._current_sample is None:
            self.preview_status_lbl.configure(text="Brak obrazu źródłowego do podglądu. Wskaż źródło datasetu w PZ1.")
            original_canvas = getattr(self, "original_canvas", None)
            if original_canvas is not None:
                self._draw_canvas_message(original_canvas, "Brak obrazu")
            self._draw_canvas_message(self.augmented_canvas, "Brak obrazu")
            return

        if redraw_only and hasattr(self, "_last_preview_payload"):
            payload = getattr(self, "_last_preview_payload", {})
            if bool(self.rain_edge_debug_var.get()) and isinstance(payload, dict) and payload.get("rain_edge_debug_mask") is None:
                self._refresh_preview(redraw_only=False)
                return
            self._draw_preview_images(payload)
            return

        self._capture_memory_snapshot("przed renderem")
        try:
            self._draw_vector_overlay()
            self.window.update_idletasks()
        except Exception:
            pass
        ok, msg, payload = preview_augmentation_image(self._current_sample, self._profile_from_vars())
        self._capture_memory_snapshot("po renderze")
        self.preview_status_lbl.configure(text=f"{Path(self._current_sample).name} | {msg}")
        if ok:
            self._last_preview_payload = payload
            self._draw_preview_images(payload)
        else:
            original_canvas = getattr(self, "original_canvas", None)
            if original_canvas is not None:
                self._draw_canvas_message(original_canvas, Path(self._current_sample).name)
            self._draw_canvas_message(self.augmented_canvas, msg)

    def _draw_preview_images(self, payload: dict):
        original = payload.get("original_rgb")
        augmented = payload.get("augmented_rgb")
        debug_mask = payload.get("rain_edge_debug_mask")
        debug_edges = bool(self.rain_edge_debug_var.get())
        original_canvas = getattr(self, "original_canvas", None)
        if original_canvas is not None and original is not None:
            original_display = self._compose_rain_edge_debug_preview(original, debug_mask) if debug_edges else original
            self._draw_array(original_canvas, original_display)
        if augmented is not None:
            augmented_display = self._compose_rain_edge_debug_preview(augmented, debug_mask) if debug_edges else augmented
            if bool(getattr(self, "_fullscreen_show_original", False)) and original is not None:
                display = self._compose_rain_edge_debug_preview(original, debug_mask) if debug_edges else original
            else:
                display = augmented_display
            self._draw_array(self.augmented_canvas, display)

    def _compose_rain_edge_debug_preview(self, array, mask):
        if np is None:
            return array
        try:
            try:
                from ..training.dataset_augmentation import build_contour_profile_debug_overlay

                profile_overlay = build_contour_profile_debug_overlay(array, self._profile_from_vars())
                if profile_overlay is not None:
                    profile_overlay = np.asarray(profile_overlay)
                    if profile_overlay.shape[:2] == array.shape[:2]:
                        try:
                            diff = float(
                                np.mean(
                                    np.abs(
                                        profile_overlay.astype("float32")
                                        - np.asarray(array).astype("float32")
                                    )
                                )
                            )
                        except Exception:
                            diff = 1.0
                        if diff > 0.003:
                            return np.clip(profile_overlay, 0, 255).astype("uint8")
            except Exception:
                pass
            if mask is None:
                try:
                    from ..training.dataset_augmentation import build_rain_edge_debug_mask

                    mask = build_rain_edge_debug_mask(array, self._profile_from_vars())
                except Exception:
                    mask = None
            if mask is None:
                return array
            edge_mask = np.clip(mask.astype("float32"), 0.0, 1.0)
            if edge_mask.shape[:2] != array.shape[:2] and PIL_AVAILABLE:
                mask_image = Image.fromarray(np.clip(edge_mask * 255.0, 0, 255).astype("uint8"))
                mask_image = mask_image.resize((int(array.shape[1]), int(array.shape[0])))
                edge_mask = np.asarray(mask_image).astype("float32") / 255.0
            edge_peak = float(np.percentile(edge_mask, 99.2) or 0.0)
            edge_max = float(edge_mask.max() or 0.0)
            edge_scale = edge_peak if edge_peak > 0.0001 else edge_max
            if edge_scale <= 0.0001:
                try:
                    from ..training.dataset_augmentation import build_rain_edge_debug_mask

                    fallback_mask = build_rain_edge_debug_mask(array, self._profile_from_vars())
                    if fallback_mask is None:
                        return array
                    edge_mask = np.clip(fallback_mask.astype("float32"), 0.0, 1.0)
                    if edge_mask.shape[:2] != array.shape[:2] and PIL_AVAILABLE:
                        mask_image = Image.fromarray(np.clip(edge_mask * 255.0, 0, 255).astype("uint8"))
                        mask_image = mask_image.resize((int(array.shape[1]), int(array.shape[0])))
                        edge_mask = np.asarray(mask_image).astype("float32") / 255.0
                    edge_peak = float(np.percentile(edge_mask, 99.2) or 0.0)
                    edge_max = float(edge_mask.max() or 0.0)
                    edge_scale = edge_peak if edge_peak > 0.0001 else edge_max
                    if edge_scale <= 0.0001:
                        return array
                except Exception:
                    return array
            edge_mask = np.clip(edge_mask / edge_scale, 0.0, 1.0)
            edge_core = edge_mask
            try:
                edge_u8 = np.clip(edge_core * 255.0, 0, 255).astype("uint8")
                edge_halo = cv2.dilate(edge_u8, np.ones((3, 3), dtype="uint8"), iterations=1).astype("float32") / 255.0
                edge_halo = cv2.GaussianBlur(edge_halo, (0, 0), sigmaX=0.60, sigmaY=0.60)
            except Exception:
                edge_halo = edge_core
            edge_alpha = np.clip(np.maximum(edge_core * 0.96, edge_halo * 0.42), 0.0, 0.96)
            base = array.astype("float32")
            color = np.array([255.0, 70.0, 35.0], dtype="float32")
            glow = np.clip(edge_halo * 0.34, 0.0, 0.34)
            out = base * (1.0 - edge_alpha[:, :, None]) + color * edge_alpha[:, :, None]
            out = out * (1.0 - glow[:, :, None]) + np.array([255.0, 210.0, 70.0], dtype="float32") * glow[:, :, None]
            return np.clip(out, 0, 255).astype("uint8")
        except Exception:
            return array

    def _draw_array(self, canvas: tk.Canvas, array):
        if canvas is None:
            return
        canvas.delete("all")
        try:
            image = Image.fromarray(array)
        except Exception:
            self._draw_canvas_message(canvas, "Nie można wyświetlić obrazu")
            return
        original_image_w = max(1, int(image.width))
        original_image_h = max(1, int(image.height))
        width = max(1, int(canvas.winfo_width() or 1))
        height = max(1, int(canvas.winfo_height() or 1))
        zoom = float(getattr(self, "_preview_zoom", self._default_preview_zoom()) or self._default_preview_zoom()) if canvas in (getattr(self, "original_canvas", None), getattr(self, "augmented_canvas", None)) else 1.0
        zoom = max(0.25, min(6.0, zoom))
        max_w = max(1, width - 12)
        max_h = max(1, height - 12)
        base_scale = min(max_w / max(1, image.width), max_h / max(1, image.height))
        scale = max(0.01, base_scale * zoom)
        resized_size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
        image = image.resize(resized_size, resampling)
        resized_w = max(1, int(image.width))
        resized_h = max(1, int(image.height))
        crop_left = 0
        crop_top = 0
        display_pan_x = 0.0
        display_pan_y = 0.0
        raw_pan_x = float(getattr(self, "_preview_pan_x", 0.0) or 0.0)
        raw_pan_y = float(getattr(self, "_preview_pan_y", 0.0) or 0.0)
        if image.width > max_w:
            max_pan_x = max(0.0, (image.width - max_w) / 2.0)
            pan_x = max(-max_pan_x, min(max_pan_x, raw_pan_x))
            center_x = image.width / 2.0 - pan_x
            crop_left = max(0, int(round(center_x - max_w / 2.0)))
            self._preview_pan_x = pan_x
        else:
            max_pan_x = max(0.0, (max_w - image.width) / 2.0)
            pan_x = max(-max_pan_x, min(max_pan_x, raw_pan_x))
            display_pan_x = pan_x
            self._preview_pan_x = pan_x
        if image.height > max_h:
            max_pan_y = max(0.0, (image.height - max_h) / 2.0)
            pan_y = max(-max_pan_y, min(max_pan_y, raw_pan_y))
            center_y = image.height / 2.0 - pan_y
            crop_top = max(0, int(round(center_y - max_h / 2.0)))
            self._preview_pan_y = pan_y
        else:
            max_pan_y = max(0.0, (max_h - image.height) / 2.0)
            pan_y = max(-max_pan_y, min(max_pan_y, raw_pan_y))
            display_pan_y = pan_y
            self._preview_pan_y = pan_y
        if image.width > max_w or image.height > max_h:
            image = image.crop((crop_left, crop_top, min(image.width, crop_left + max_w), min(image.height, crop_top + max_h)))
        display_w = max(1, int(image.width))
        display_h = max(1, int(image.height))
        display_left = (float(width) - float(display_w)) / 2.0 + display_pan_x
        display_top = (float(height) - float(display_h)) / 2.0 + display_pan_y
        self._preview_canvas_mappings[id(canvas)] = {
            "canvas_w": float(width),
            "canvas_h": float(height),
            "image_w": float(original_image_w),
            "image_h": float(original_image_h),
            "scale": float(scale),
            "resized_w": float(resized_w),
            "resized_h": float(resized_h),
            "crop_left": float(crop_left),
            "crop_top": float(crop_top),
            "display_left": float(display_left),
            "display_top": float(display_top),
            "display_w": float(display_w),
            "display_h": float(display_h),
        }
        photo = ImageTk.PhotoImage(image)
        self._photo_refs.append(photo)
        self._photo_refs = self._photo_refs[-8:]
        canvas.create_image(display_left, display_top, image=photo, anchor=tk.NW)
        if canvas is getattr(self, "augmented_canvas", None):
            self._draw_vector_overlay()

    def _register_canvas_overlay_region(self, kind: str, rect: tuple[float, float, float, float], **payload):
        payload.update({"kind": kind, "rect": tuple(rect)})
        self._canvas_overlay_regions.append(payload)

    def _format_overlay_value(self, variable, step) -> str:
        try:
            return self._format_toolbox_value(variable.get(), step)
        except Exception:
            return "0"

    def _draw_canvas_tool_icon(self, canvas: tk.Canvas, key: str, label: str, x: int, y: int):
        key = self._normalize_toolbox_key(key)
        special_tool_active = str(getattr(self, "_vector_tool", "") or "") == "wind"
        active = (
            not special_tool_active
            and key == self._normalize_toolbox_key(self._active_toolbox)
        )
        fill = "#20342f" if active else "#171b20"
        outline = "#5fd29c" if active else "#596269"
        color = "#7ff0b4" if active else "#d7dde1"
        muted = "#9ff0bc" if active else "#7b8790"
        canvas.create_rectangle(x, y, x + 42, y + 38, fill=fill, outline=outline, width=(2 if active else 1), tags=("aug_overlay",))
        cx = x + 21
        cy = y + 15
        if key in ("scene", "plate_surface", "geometry"):
            canvas.create_rectangle(cx - 10, cy - 7, cx + 10, cy + 7, outline=color, width=2, tags=("aug_overlay",))
            canvas.create_line(cx - 8, cy + 7, cx + 8, cy - 7, fill=color, width=1, tags=("aug_overlay",))
        elif key == "color":
            for offset, fill in ((-7, "#f05252"), (0, "#22c55e"), (7, "#3b82f6")):
                canvas.create_oval(cx + offset - 4, cy - 4, cx + offset + 4, cy + 4, fill=(fill if active else color), outline="", tags=("aug_overlay",))
        elif key in ("rain", "weather"):
            canvas.create_arc(cx - 12, cy - 9, cx + 3, cy + 7, start=40, extent=230, outline=color, width=2, style=tk.ARC, tags=("aug_overlay",))
            canvas.create_arc(cx - 2, cy - 9, cx + 13, cy + 7, start=-90, extent=230, outline=color, width=2, style=tk.ARC, tags=("aug_overlay",))
            for offset in (-8, 0, 8):
                canvas.create_line(cx + offset - 1, cy + 8, cx + offset - 5, cy + 15, fill=muted, width=2, tags=("aug_overlay",))
        elif key == "material":
            canvas.create_rectangle(cx - 12, cy - 8, cx + 12, cy + 8, outline=color, width=2, tags=("aug_overlay",))
            canvas.create_line(cx - 9, cy + 1, cx - 4, cy - 2, cx + 1, cy + 2, cx + 7, cy - 1, fill=muted, width=2, smooth=True, tags=("aug_overlay",))
        elif key == "relief":
            canvas.create_line(cx - 11, cy + 8, cx - 5, cy - 8, cx + 2, cy + 8, cx + 8, cy - 8, fill=color, width=2, smooth=True, tags=("aug_overlay",))
            for offset in (-7, -2, 3, 8):
                canvas.create_line(cx + offset, cy - 6, cx + offset - 3, cy + 7, fill=muted, width=1, tags=("aug_overlay",))
        elif key in ("mud", "dirt"):
            canvas.create_oval(cx - 10, cy - 5, cx + 4, cy + 7, fill=color, outline="", tags=("aug_overlay",))
            canvas.create_oval(cx + 2, cy - 8, cx + 11, cy + 3, fill=color, outline="", tags=("aug_overlay",))
            canvas.create_oval(cx - 2, cy + 8, cx + 4, cy + 14, fill=muted, outline="", tags=("aug_overlay",))
        elif key in ("light", "illumination"):
            canvas.create_oval(cx - 13, cy - 5, cx - 5, cy + 5, fill=color, outline="", tags=("aug_overlay",))
            canvas.create_polygon(cx - 3, cy - 8, cx + 14, cy - 15, cx + 14, cy + 15, cx - 3, cy + 8, fill="", outline=muted, width=2, tags=("aug_overlay",))
            canvas.create_line(cx - 3, cy, cx + 12, cy, fill=color, width=1, arrow=tk.LAST, tags=("aug_overlay",))
        elif key == "sensor":
            canvas.create_rectangle(cx - 13, cy - 8, cx + 12, cy + 9, outline=color, width=2, tags=("aug_overlay",))
            canvas.create_oval(cx - 6, cy - 6, cx + 7, cy + 7, outline=muted, width=2, tags=("aug_overlay",))
            canvas.create_oval(cx - 1, cy - 1, cx + 2, cy + 2, fill=color, outline="", tags=("aug_overlay",))
        canvas.create_text(cx, y + 31, text=str(label or key)[:6], fill=color, font=("Segoe UI", 6), tags=("aug_overlay",))
        self._register_canvas_overlay_region("toolbox", (x, y, x + 42, y + 38), key=key)

    def _draw_canvas_vector_tool_icon(self, canvas: tk.Canvas, tool: str, x: int, y: int):
        specs = self._vector_specs()
        spec = specs.get(tool)
        if not spec:
            return
        active = self._vector_tool == tool
        fill = spec["active"] if active else "#171b20"
        outline = spec["color"] if active else "#596269"
        color = spec["color"] if active else "#d7dde1"
        tags = ("vector_overlay", "vector_control", f"vector_tool:{tool}")
        canvas.create_rectangle(x, y, x + 42, y + 38, fill=fill, outline=outline, width=(2 if active else 1), tags=tags)
        self._draw_vector_icon(canvas, tool, x + 21, y + 15, color, active)
        canvas.create_text(x + 21, y + 31, text="Wiatr", fill=color, font=("Segoe UI", 6), tags=tags)
        self._register_canvas_overlay_region("vector_tool", (x, y, x + 42, y + 38), tool=tool)

    def _draw_canvas_action_icon(self, canvas: tk.Canvas, action: str, label: str, x: int, y: int):
        outline = "#69737a"
        color = "#d7dde1"
        if action == "fullscreen" and bool(getattr(self, "_preview_fullscreen", False)):
            outline = "#f3c86a"
            color = "#ffd56e"
        if action == "zoom" and abs(float(getattr(self, "_preview_zoom", self._default_preview_zoom()) or self._default_preview_zoom()) - self._default_preview_zoom()) > 0.01:
            outline = "#76d9ff"
            color = "#76d9ff"
        if action == "dice":
            outline = "#8df0b7"
            color = "#8df0b7"
        if action == "layout":
            outline = "#76d9ff"
            color = "#76d9ff"
        if action == "reset":
            outline = "#f3c86a"
            color = "#ffd56e"
        canvas.create_rectangle(x, y, x + 38, y + 38, fill="#171b20", outline=outline, width=1, tags=("aug_overlay",))
        cx = x + 19
        cy = y + 16
        if action == "zoom":
            canvas.create_oval(cx - 7, cy - 7, cx + 5, cy + 5, outline=color, width=2, tags=("aug_overlay",))
            canvas.create_line(cx + 4, cy + 4, cx + 10, cy + 10, fill=color, width=2, tags=("aug_overlay",))
            canvas.create_text(cx, y + 32, text=f"{float(getattr(self, '_preview_zoom', 1.0) or 1.0):.1f}x", fill=color, font=("Segoe UI", 6), tags=("aug_overlay",))
        elif action == "fullscreen":
            canvas.create_line(cx - 10, cy - 7, cx - 10, cy - 12, cx - 5, cy - 12, fill=color, width=2, tags=("aug_overlay",))
            canvas.create_line(cx + 10, cy - 7, cx + 10, cy - 12, cx + 5, cy - 12, fill=color, width=2, tags=("aug_overlay",))
            canvas.create_line(cx - 10, cy + 7, cx - 10, cy + 12, cx - 5, cy + 12, fill=color, width=2, tags=("aug_overlay",))
            canvas.create_line(cx + 10, cy + 7, cx + 10, cy + 12, cx + 5, cy + 12, fill=color, width=2, tags=("aug_overlay",))
            canvas.create_text(cx, y + 32, text=label, fill=color, font=("Segoe UI", 6), tags=("aug_overlay",))
        elif action == "dice":
            die_size = 16
            die_left = int(round(cx - die_size / 2))
            die_top = y + 6
            canvas.create_rectangle(
                die_left,
                die_top,
                die_left + die_size,
                die_top + die_size,
                fill="#10211a",
                outline=color,
                width=1,
                tags=("aug_overlay",),
            )
            dot_r = 1.7
            for dot_x, dot_y in ((4, 4), (12, 4), (8, 8), (4, 12), (12, 12)):
                px = die_left + dot_x
                py = die_top + dot_y
                canvas.create_oval(px - dot_r, py - dot_r, px + dot_r, py + dot_r, fill=color, outline="", tags=("aug_overlay",))
            canvas.create_text(cx, y + 32, text=label, fill=color, font=("Segoe UI", 6), tags=("aug_overlay",))
        elif action == "layout":
            points = ((cx - 9, cy - 5), (cx + 8, cy - 7), (cx - 2, cy + 8))
            canvas.create_line(points[0][0], points[0][1], points[1][0], points[1][1], fill=color, width=1, tags=("aug_overlay",))
            canvas.create_line(points[1][0], points[1][1], points[2][0], points[2][1], fill=color, width=1, tags=("aug_overlay",))
            canvas.create_line(points[2][0], points[2][1], points[0][0], points[0][1], fill=color, width=1, dash=(2, 2), tags=("aug_overlay",))
            for px, py in points:
                canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill=color, outline="", tags=("aug_overlay",))
            canvas.create_text(cx, y + 32, text=label, fill=color, font=("Segoe UI", 6), tags=("aug_overlay",))
        elif action == "reset":
            canvas.create_arc(cx - 9, cy - 9, cx + 9, cy + 9, start=35, extent=285, style=tk.ARC, outline=color, width=2, tags=("aug_overlay",))
            canvas.create_line(cx - 1, cy - 10, cx + 6, cy - 10, cx + 6, cy - 3, fill=color, width=2, tags=("aug_overlay",))
            canvas.create_text(cx, y + 32, text=label, fill=color, font=("Segoe UI", 6), tags=("aug_overlay",))
        self._register_canvas_overlay_region("action", (x, y, x + 38, y + 38), action=action)

    def _draw_canvas_slider(self, canvas: tk.Canvas, x: int, y: int, w: int, label: str, variable, from_, to, step, role: str | None = None):
        try:
            value = float(variable.get())
        except Exception:
            value = 0.0
        from_ = float(from_)
        to = float(to)
        span = max(0.000001, to - from_)
        ratio = max(0.0, min(1.0, (value - from_) / span))
        role_key = str(role or label or "").strip().casefold()
        fill_color = {
            "r": "#ff6b6b",
            "red": "#ff6b6b",
            "g": "#65d99b",
            "green": "#65d99b",
            "b": "#76d9ff",
            "blue": "#76d9ff",
        }.get(role_key, "#65d99b")
        label_color = fill_color if role_key in {"r", "red", "g", "green", "b", "blue"} else "#e4ecec"
        knob_fill = {
            "r": "#ffd6d6",
            "red": "#ffd6d6",
            "g": "#dfffe9",
            "green": "#dfffe9",
            "b": "#d9f2ff",
            "blue": "#d9f2ff",
        }.get(role_key, "#dfffe9")
        canvas.create_text(x, y, text=f"{label}: {self._format_overlay_value(variable, step)}", anchor=tk.W, fill=label_color, font=("Segoe UI", 8), tags=("aug_overlay",))
        bar_y = y + 12
        canvas.create_line(x, bar_y, x + w, bar_y, fill="#56616a", width=3, tags=("aug_overlay",))
        canvas.create_line(x, bar_y, x + w * ratio, bar_y, fill=fill_color, width=3, tags=("aug_overlay",))
        knob_x = x + w * ratio
        canvas.create_oval(knob_x - 4, bar_y - 4, knob_x + 4, bar_y + 4, fill=knob_fill, outline="#0d1f18", width=1, tags=("aug_overlay",))
        self._register_canvas_overlay_region(
            "slider",
            (x - 4, y - 2, x + w + 8, y + 20),
            variable=variable,
            from_=from_,
            to=to,
            step=step,
            bar=(x, x + w),
            role=role,
        )

    def _draw_canvas_range_slider(self, canvas: tk.Canvas, x: int, y: int, w: int, label: str, min_var, max_var, from_, to, step):
        try:
            min_value = float(min_var.get())
            max_value = float(max_var.get())
        except Exception:
            min_value = float(from_)
            max_value = float(to)
        from_ = float(from_)
        to = float(to)
        span = max(0.000001, to - from_)
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        min_ratio = max(0.0, min(1.0, (min_value - from_) / span))
        max_ratio = max(0.0, min(1.0, (max_value - from_) / span))
        min_x = x + w * min_ratio
        max_x = x + w * max_ratio
        canvas.create_text(x, y, text=label, anchor=tk.W, fill="#e4ecec", font=("Segoe UI", 8), tags=("aug_overlay",))
        box_w = 42
        box_h = 14
        min_text = self._format_overlay_value(min_var, step)
        max_text = self._format_overlay_value(max_var, step)
        canvas.create_rectangle(x + w - box_w * 2 - 8, y - 7, x + w - box_w - 6, y + box_h - 7, fill="#162027", outline="#3f555e", width=1, tags=("aug_overlay",))
        canvas.create_rectangle(x + w - box_w, y - 7, x + w, y + box_h - 7, fill="#162027", outline="#3f555e", width=1, tags=("aug_overlay",))
        canvas.create_text(x + w - box_w - 7, y, text=min_text, anchor=tk.E, fill="#8df0b7", font=("Segoe UI", 7, "bold"), tags=("aug_overlay",))
        canvas.create_text(x + w - 3, y, text=max_text, anchor=tk.E, fill="#76d9ff", font=("Segoe UI", 7, "bold"), tags=("aug_overlay",))
        bar_y = y + 15
        canvas.create_line(x, bar_y, x + w, bar_y, fill="#56616a", width=3, tags=("aug_overlay",))
        canvas.create_line(min_x, bar_y, max_x, bar_y, fill="#65d99b", width=4, tags=("aug_overlay",))
        canvas.create_oval(min_x - 5, bar_y - 6, min_x + 5, bar_y + 6, fill="#41c973", outline="#0d1f18", width=1, tags=("aug_overlay",))
        canvas.create_rectangle(max_x - 5, bar_y - 6, max_x + 5, bar_y + 6, fill="#4aa8ff", outline="#0d1f18", width=1, tags=("aug_overlay",))
        self._register_canvas_overlay_region(
            "range_slider",
            (x - 5, y - 8, x + w + 8, y + 24),
            min_var=min_var,
            max_var=max_var,
            from_=from_,
            to=to,
            step=step,
            bar=(x, x + w),
            min_x=min_x,
            max_x=max_x,
        )

    def _draw_canvas_check(self, canvas: tk.Canvas, x: int, y: int, label: str, variable):
        active = bool(variable.get())
        fill = "#61d998" if active else "#10161a"
        outline = "#61d998" if active else "#6f7a82"
        canvas.create_rectangle(x, y, x + 12, y + 12, fill=fill, outline=outline, width=1, tags=("aug_overlay",))
        if active:
            canvas.create_line(x + 3, y + 6, x + 5, y + 9, x + 10, y + 3, fill="#0b1712", width=1, tags=("aug_overlay",))
        canvas.create_text(x + 19, y + 6, text=label, anchor=tk.W, fill=("#bfffd6" if active else "#dce5e7"), font=("Segoe UI", 8), tags=("aug_overlay",))
        self._register_canvas_overlay_region("check", (x - 4, y - 4, x + 190, y + 17), variable=variable)

    def _draw_canvas_toolbox_panel(self, canvas: tk.Canvas, width: int, height: int) -> int:
        fields = self._toolbox_fields(self._active_toolbox)
        if not fields:
            return 54
        panel_w = max(230, min(292, width - 28))
        panel_pos = getattr(self, "_toolbox_panel_pos", None)
        if panel_pos is None:
            x = max(10, width - panel_w - 10)
            y = 72
        else:
            x = int(panel_pos[0])
            y = int(panel_pos[1])
        x = max(4, min(x, max(4, width - panel_w - 4)))
        y = max(64, min(y, max(64, height - 38)))
        self._toolbox_panel_pos = (x, y)
        def _field_height(field: dict) -> int:
            if field.get("type") == "section":
                return 34 if str(field.get("text", "")).strip() else 24
            return 29

        row_step = 29
        collapsed = bool(getattr(self, "_toolbox_panel_collapsed", False))
        content_h = sum(_field_height(field) for field in fields)
        max_panel_h = min(230, max(126, height - y - 16))
        natural_h = 44 + content_h
        panel_h = 30 if collapsed else min(max_panel_h, max(112, natural_h))
        viewport_top = y + 34
        viewport_bottom = y + panel_h - 10
        viewport_h = max(24, viewport_bottom - viewport_top)
        max_scroll = max(0.0, float(content_h - viewport_h))
        scroll = max(0.0, min(max_scroll, float(getattr(self, "_toolbox_panel_scroll", 0.0) or 0.0)))
        self._toolbox_panel_scroll = scroll
        self._toolbox_panel_max_scroll = max_scroll
        self._toolbox_panel_rect = (x, y, x + panel_w, y + panel_h)
        slider_w = panel_w - (34 if max_scroll > 0.5 else 24)
        canvas.create_rectangle(x, y, x + panel_w, y + panel_h, fill="#10161a", outline="#3d4a50", width=1, tags=("aug_overlay",))
        self._register_canvas_overlay_region("panel", (x, y, x + panel_w, y + panel_h))
        canvas.create_text(x + 10, y + 13, text=self._active_toolbox_title(), anchor=tk.W, fill="#8df0b7", font=("Segoe UI", 10, "bold"), tags=("aug_overlay",))
        icon_y = y + 14
        reset_x = x + panel_w - 48
        toggle_x = x + panel_w - 24
        canvas.create_text(reset_x, icon_y, text="↺", anchor=tk.CENTER, fill="#d7dde1", font=("Segoe UI", 10, "bold"), tags=("aug_overlay",))
        canvas.create_text(toggle_x, icon_y, text=("+" if collapsed else "−"), anchor=tk.CENTER, fill="#d7dde1", font=("Segoe UI", 12, "bold"), tags=("aug_overlay",))
        canvas.create_line(x + panel_w - 78, y + 9, x + panel_w - 66, y + 9, fill="#71808a", width=1, tags=("aug_overlay",))
        canvas.create_line(x + panel_w - 78, y + 14, x + panel_w - 66, y + 14, fill="#71808a", width=1, tags=("aug_overlay",))
        canvas.create_line(x + panel_w - 78, y + 19, x + panel_w - 66, y + 19, fill="#71808a", width=1, tags=("aug_overlay",))
        self._register_canvas_overlay_region("panel_drag", (x, y, x + panel_w - 58, y + 28))
        self._register_canvas_overlay_region("panel_reset", (reset_x - 11, y + 3, reset_x + 11, y + 25))
        self._register_canvas_overlay_region("panel_toggle", (toggle_x - 11, y + 3, toggle_x + 11, y + 25))
        if collapsed:
            return y + panel_h
        self._register_canvas_overlay_region("panel_body", (x + 4, viewport_top, x + panel_w - 4, viewport_bottom))
        row_y = viewport_top - scroll
        for field in fields:
            item_h = _field_height(field)
            item_top = row_y
            item_bottom = row_y + item_h
            if item_bottom < viewport_top or item_top > viewport_bottom:
                row_y += item_h
                continue
            if item_top < viewport_top or item_bottom > viewport_bottom:
                row_y += item_h
                continue
            ftype = field.get("type")
            if ftype == "section":
                label = str(field.get("label", "") or "").strip()
                text = str(field.get("text", "") or "").strip()
                priority = bool(field.get("priority", False))
                header_fill = "#0f2d1d" if priority else "#14211d"
                header_outline = "#3fc978" if priority else "#46665a"
                section_h = 28 if text else 18
                canvas.create_rectangle(
                    x + 8,
                    row_y - 5,
                    x + panel_w - 8,
                    row_y - 5 + section_h,
                    fill=header_fill,
                    outline=header_outline,
                    width=1,
                    tags=("aug_overlay",),
                )
                canvas.create_text(
                    x + 14,
                    row_y + 4,
                    text=label[:42],
                    anchor=tk.W,
                    fill="#a4f6c7",
                    font=("Segoe UI", 7, "bold"),
                    tags=("aug_overlay",),
                )
                if text:
                    canvas.create_text(
                        x + 14,
                        row_y + 17,
                        text=text[:70],
                        anchor=tk.W,
                        fill="#c8d7cf",
                        font=("Segoe UI", 6),
                        tags=("aug_overlay",),
                    )
                row_y += section_h + 6
            elif ftype == "check":
                self._draw_canvas_check(canvas, x + 10, row_y - 4, str(field["label"]), field["var"])
                row_y += item_h
            elif ftype in ("range", "range_slider"):
                self._draw_canvas_range_slider(canvas, x + 10, row_y, slider_w, str(field["label"]), field["min_var"], field["max_var"], field["from"], field["to"], field["step"])
                row_y += item_h
            elif "var" not in field:
                row_y += item_h
            else:
                self._draw_canvas_slider(canvas, x + 10, row_y, slider_w, str(field["label"]), field.get("var"), field["from"], field["to"], field["step"])
                row_y += item_h
        if max_scroll > 0.5:
            track_x = x + panel_w - 9
            canvas.create_line(track_x, viewport_top, track_x, viewport_bottom, fill="#2f3b42", width=3, tags=("aug_overlay",))
            thumb_h = max(22.0, viewport_h * (viewport_h / max(float(content_h), 1.0)))
            thumb_y = viewport_top + (viewport_h - thumb_h) * (scroll / max_scroll if max_scroll > 0 else 0.0)
            canvas.create_rectangle(
                track_x - 3,
                thumb_y,
                track_x + 3,
                thumb_y + thumb_h,
                fill="#65d99b",
                outline="#0d1f18",
                width=1,
                tags=("aug_overlay",),
            )
        return y + panel_h

    def _bind_vector_canvas(self, canvas: tk.Canvas):
        canvas.bind("<Button-1>", self._on_vector_canvas_press, add="+")
        canvas.bind("<Double-Button-1>", self._on_vector_canvas_double_press, add="+")
        canvas.bind("<B1-Motion>", self._on_vector_canvas_drag, add="+")
        canvas.bind("<ButtonRelease-1>", self._on_vector_canvas_release, add="+")
        canvas.bind("<Button-3>", self._on_vector_canvas_context_press, add="+")
        canvas.bind("<B3-Motion>", self._on_vector_canvas_context_drag, add="+")
        canvas.bind("<ButtonRelease-3>", self._on_vector_canvas_context_release, add="+")
        canvas.bind("<Motion>", self._on_vector_canvas_motion, add="+")
        canvas.bind("<Leave>", self._on_vector_canvas_leave, add="+")
        canvas.bind("<space>", self._on_space_key, add="+")
        canvas.bind("<Key-space>", self._on_space_key, add="+")

    def _bind_preview_zoom_canvas(self, canvas: tk.Canvas):
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.bind(sequence, self._on_preview_mousewheel, add="+")

    def _vector_specs(self) -> dict[str, dict[str, str]]:
        wind_label = "Wiatr/deszcz" if self.target == "plate" else "Wiatr/błoto"
        return {
            "wind": {"label": wind_label, "color": "#76d9ff", "active": "#123946"},
        }

    def _vector_value_text(self, tool: str) -> str:
        try:
            if tool == "wind":
                return f"{float(self.dirt_flow_wind_strength_var.get()):.2f}"
        except Exception:
            return "0.00"
        return "0.00"

    def _draw_memory_overlay(self, canvas: tk.Canvas, width: int, height: int):
        try:
            text = self._format_memory_snapshot()
            percent = float((getattr(self, "_memory_snapshot", {}) or {}).get("system_percent", 0.0) or 0.0)
            outline = "#58d68d"
            if percent >= 88:
                outline = "#ff6b6b"
            elif percent >= 76:
                outline = "#f3c86a"
            max_w = max(80, width - 20)
            text_w = min(max(228, int(len(text) * 5.7) + 18), max_w)
            x1 = max(10, width - text_w - 10)
            y2 = max(74, height - 10)
            y1 = max(52, y2 - 24)
            canvas.create_rectangle(
                x1,
                y1,
                x1 + text_w,
                y2,
                fill="#10161a",
                outline=outline,
                width=1,
                tags=("aug_overlay", "memory_overlay"),
            )
            canvas.create_text(
                x1 + 9,
                y1 + 12,
                text=text,
                anchor=tk.W,
                fill="#d7dde1",
                font=("Segoe UI", 7),
                tags=("aug_overlay", "memory_overlay"),
            )
        except Exception:
            pass

    def _draw_vector_overlay(self):
        canvas = getattr(self, "augmented_canvas", None)
        if canvas is None:
            return
        try:
            canvas.delete("vector_overlay")
            canvas.delete("aug_overlay")
            self._canvas_overlay_regions = []
            width = max(1, int(canvas.winfo_width() or 1))
            height = max(1, int(canvas.winfo_height() or 1))
            scene_visible = bool(getattr(self, "_scene_viewport_visible", False))
            if scene_visible:
                self._draw_scene_viewport(canvas, width, height)
            x = 10
            y = 10
            specs = self._vector_specs()
            for key, label in self._domain_toolboxes():
                self._draw_canvas_tool_icon(canvas, key, label, x, y)
                x += 48
            if "wind" in specs:
                self._draw_canvas_vector_tool_icon(canvas, "wind", x, y)
                x += 48
            actions = (("dice", "obraz"), ("layout", "układ"), ("reset", "zero"), ("zoom", "zoom"), ("fullscreen", "ent"))
            actions_left = max(10, width - (len(actions) * 44 - 6))
            mode_icons_right = max(x + 150, actions_left - 118)
            try:
                self._draw_scene_viewport_mode_icons(canvas, width, height, start_x=x + 4, start_y=10, max_right=mode_icons_right)
            except Exception:
                pass
            action_x = actions_left
            for action, label in actions:
                self._draw_canvas_action_icon(canvas, action, label, action_x, 10)
                action_x += 44
            mode_text = "ORYGINAŁ" if bool(getattr(self, "_fullscreen_show_original", False)) else "EFEKT"
            pill_w = 102
            pill_x = max(10, actions_left - pill_w - 8)
            pill_y = 10
            canvas.create_rectangle(pill_x, pill_y, pill_x + pill_w, pill_y + 20, fill="#171b20", outline="#77612b", width=1, tags=("aug_overlay",))
            canvas.create_text(pill_x + pill_w / 2, pill_y + 10, text=f"TAB: {mode_text}", fill="#ffd56e", font=("Segoe UI", 7), tags=("aug_overlay",))
            vector_drag_active = bool(
                self._vector_drag_start
                and self._vector_drag_end
                and self._vector_tool in specs
                and self._vector_tool != "wind"
            )
            active_headlight = getattr(self, "_active_headlight_index", None)
            if active_headlight not in (1, 2, 3):
                active_headlight = None
            wind_active = self._vector_tool == "wind"
            if self._vector_tool in specs and self._vector_tool != "wind" and not vector_drag_active:
                self._draw_vector_line(self._vector_tool, width, height)
            if vector_drag_active:
                self._draw_arrow(
                    canvas,
                    self._vector_drag_start,
                    self._vector_drag_end,
                    specs[self._vector_tool]["color"],
                    dash=(4, 3),
                )
            if not (
                self._headlight_drag is not None
                or self._canvas_slider_drag is not None
                or self._toolbox_panel_drag is not None
                or self._headlight_panel_drag is not None
                or self._scene_projection_drag is not None
                or self._scene_view_drag is not None
                or self._scene_pan_drag is not None
                or self._vector_drag_start is not None
            ):
                self._draw_memory_overlay(canvas, width, height)
            if (
                self._normalize_toolbox_key(getattr(self, "_active_toolbox", "")) == "illumination"
                or self._headlight_cones_are_visible()
                or active_headlight is not None
            ):
                self._draw_headlight_handles(canvas, width, height)
            if "wind" in specs and wind_active:
                self._draw_wind_vector_panel(canvas, width, height, active=wind_active)
            canvas.tag_raise("headlight_control")
            canvas.tag_raise("scene_view_icons")
            if "wind" in specs and wind_active:
                canvas.tag_raise("wind_vector_panel")
            if (
                bool(getattr(self, "_preview_fullscreen", False))
                and bool(getattr(self, "_toolbox_panel_visible", False))
                and bool(getattr(self, "_effect_inspector_visible", True))
                and not wind_active
                and getattr(self, "_headlight_config_index", None) not in (1, 2, 3)
            ):
                self._draw_canvas_toolbox_panel(canvas, width, height)
        except Exception:
            pass

    def _request_vector_overlay_redraw(self, delay_ms: int = 24):
        if self._overlay_redraw_after_id is not None:
            return

        def _run():
            self._overlay_redraw_after_id = None
            self._draw_vector_overlay()

        try:
            self._overlay_redraw_after_id = self.window.after(max(8, int(delay_ms)), _run)
        except Exception:
            self._overlay_redraw_after_id = None
            self._draw_vector_overlay()

    def _draw_vector_icon(self, canvas: tk.Canvas, tool: str, cx: float, cy: float, color: str, active: bool):
        tags = ("vector_overlay", "vector_control", f"vector_tool:{tool}")
        if tool == "wind":
            canvas.create_line(cx - 13, cy - 5, cx - 3, cy - 9, cx + 12, cy - 5, fill=color, width=2, smooth=True, tags=tags)
            canvas.create_line(cx - 10, cy + 1, cx + 9, cy + 1, fill=color, width=2, arrow=tk.LAST, arrowshape=(6, 7, 3), tags=tags)
            canvas.create_line(cx - 13, cy + 7, cx - 1, cy + 10, cx + 11, cy + 7, fill=color, width=1, smooth=True, tags=tags)
            return
        if tool == "light":
            radius = 5
            canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline=color, fill=("#5a4212" if active else ""), width=2, tags=tags)
            for angle in (0, 45, 90, 135, 180, 225, 270, 315):
                rad = math.radians(angle)
                canvas.create_line(
                    cx + math.cos(rad) * 8,
                    cy + math.sin(rad) * 8,
                    cx + math.cos(rad) * 13,
                    cy + math.sin(rad) * 13,
                    fill=color,
                    width=1,
                    tags=tags,
                )
            return
        canvas.create_arc(cx - 11, cy - 9, cx + 11, cy + 13, start=20, extent=140, outline=color, width=2, style=tk.ARC, tags=tags)
        canvas.create_line(cx + 8, cy + 6, cx + 13, cy + 11, fill=color, width=2, tags=tags)

    def _draw_vector_line(self, tool: str, width: int, height: int):
        canvas = getattr(self, "augmented_canvas", None)
        specs = self._vector_specs()
        if canvas is None or tool not in specs:
            return
        if tool in self._vector_positions:
            start, end = self._vector_positions[tool]
        else:
            start, end = self._default_vector_position(tool, width, height)
        if start == end:
            return
        if tool == "wind":
            self._draw_wind_vector_line(canvas, start, end, specs[tool]["color"])
            return
        self._draw_arrow(canvas, start, end, specs[tool]["color"])
        canvas.create_text(
            end[0] + 8,
            end[1],
            text=specs[tool]["label"],
            anchor=tk.W,
            fill=specs[tool]["color"],
            font=("Segoe UI", 8),
            tags=("vector_overlay",),
        )

    def _draw_wind_vector_line(self, canvas: tk.Canvas, start: tuple[int, int], end: tuple[int, int], color: str, dash=None):
        sx, sy = float(start[0]), float(start[1])
        ex, ey = float(end[0]), float(end[1])
        min_x = min(sx, ex)
        max_x = max(sx, ex)
        min_y = min(sy, ey)
        max_y = max(sy, ey)
        label = "WIATR"
        try:
            strength = float(self.dirt_flow_wind_strength_var.get() or 0.0)
            label = f"WIATR {strength:.2f}"
        except Exception:
            pass
        canvas_w = max(1, int(canvas.winfo_width() or 1))
        canvas_h = max(1, int(canvas.winfo_height() or 1))
        if dash is None:
            canvas.create_line(
                sx,
                sy,
                ex,
                ey,
                fill="#031015",
                width=6,
                tags=("vector_overlay",),
            )
        canvas.create_line(
            sx,
            sy,
            ex,
            ey,
            fill=color,
            width=4,
            arrow=tk.LAST,
            arrowshape=(15, 18, 6),
            dash=dash,
            tags=("vector_overlay",),
        )
        for px, py, fill, outline in (
            (sx, sy, "#071218", color),
            (ex, ey, color, "#071218"),
        ):
            canvas.create_oval(px - 6, py - 6, px + 6, py + 6, fill=fill, outline=outline, width=2, tags=("vector_overlay",))
        label_w = min(116, max(70, int(len(label) * 7 + 18)))
        label_h = 18
        safe_top = 46

        def clamp(value: float, low: float, high: float) -> float:
            if high < low:
                return low
            return max(low, min(high, value))

        label_candidates = (
            (clamp(min_x, 4, canvas_w - label_w - 4), min_y - label_h - 8),
            (clamp(min_x, 4, canvas_w - label_w - 4), max_y + 8),
            (min_x - label_w - 8, clamp(sy - label_h / 2, safe_top, canvas_h - label_h - 4)),
            (max_x + 8, clamp(sy - label_h / 2, safe_top, canvas_h - label_h - 4)),
        )
        label_rect = None
        for lx, ly in label_candidates:
            if lx < 4 or ly < safe_top or lx + label_w > canvas_w - 4 or ly + label_h > canvas_h - 4:
                continue
            overlaps_vector_box = not (lx + label_w < min_x or lx > max_x or ly + label_h < min_y or ly > max_y)
            if not overlaps_vector_box:
                label_rect = (int(lx), int(ly), int(lx + label_w), int(ly + label_h))
                break
        if label_rect is not None:
            lx1, ly1, lx2, ly2 = label_rect
            canvas.create_rectangle(
                lx1,
                ly1,
                lx2,
                ly2,
                fill="#071218",
                outline="#24566a",
                width=1,
                tags=("vector_overlay",),
            )
            canvas.create_text(
                lx1 + 8,
                ly1 + label_h / 2,
                text=label,
                anchor=tk.W,
                fill="#bfefff",
                font=("Segoe UI", 7, "bold"),
                tags=("vector_overlay",),
            )
        else:
            canvas.create_text(
                sx + 10,
                sy - 12,
                text=label,
                anchor=tk.W,
                fill="#bfefff",
                font=("Segoe UI", 7, "bold"),
                tags=("vector_overlay",),
            )

    def _wind_vector_panel_geometry(self, width: int, height: int) -> dict[str, tuple[float, float, float, float] | tuple[float, float] | float]:
        panel_w = 218.0
        panel_h = 218.0
        if bool(getattr(self, "_scene_viewport_visible", False)):
            _rect, body_rect = self._scene_viewport_rects(width, height)
            right = float(body_rect[2]) - 14.0
            top = float(body_rect[1]) + 10.0
        else:
            right = float(width) - 16.0
            top = 58.0
        left = max(12.0, right - panel_w)
        top = max(52.0, min(top, max(52.0, float(height) - panel_h - 12.0)))
        panel = (left, top, left + panel_w, top + panel_h)
        value_field = (left + panel_w - 58.0, top + 27.0, left + panel_w - 12.0, top + 48.0)
        plot = (left + 18.0, top + 54.0, left + panel_w - 18.0, top + panel_h - 18.0)
        start = ((float(plot[0]) + float(plot[2])) / 2.0, (float(plot[1]) + float(plot[3])) / 2.0)
        scale = max(46.0, min((float(plot[2]) - float(plot[0])) / 2.0, (float(plot[3]) - float(plot[1])) / 2.0) - 8.0)
        return {"panel": panel, "plot": plot, "start": start, "scale": scale, "value_field": value_field}

    def _wind_vector_panel_points(self, geometry: dict, active: bool) -> tuple[tuple[float, float], tuple[float, float], str]:
        start_x, start_y = geometry.get("start", (0.0, 0.0))
        scale = float(geometry.get("scale", 60.0) or 60.0)
        drag = getattr(self, "_wind_panel_drag", None)
        if isinstance(drag, dict):
            end = drag.get("end", (start_x + scale * 0.45, start_y))
            return (float(start_x), float(start_y)), (float(end[0]), float(end[1])), "ustawiam"
        try:
            strength = max(0.0, min(1.0, float(self.dirt_flow_wind_strength_var.get() or 0.0)))
        except Exception:
            strength = 0.0
        try:
            angle = float(self.dirt_flow_air_angle_var.get() or 0.0)
        except Exception:
            angle = 0.0
        display_strength = strength
        if active and display_strength <= 0.001:
            display_strength = 0.45
        if display_strength <= 0.001:
            end_x = start_x + scale * 0.30
            end_y = start_y
        else:
            radians = math.radians(angle)
            end_x = start_x + math.cos(radians) * scale * display_strength
            end_y = start_y - math.sin(radians) * scale * display_strength
        label = f"{strength:.2f} / {angle:+.0f}°"
        label = f"{strength:.1f} / {angle:+.0f} deg"
        return (float(start_x), float(start_y)), (float(end_x), float(end_y)), label

    def _draw_wind_vector_panel(self, canvas: tk.Canvas, width: int, height: int, *, active: bool) -> None:
        geometry = self._wind_vector_panel_geometry(width, height)
        panel = geometry["panel"]
        plot = geometry["plot"]
        value_field = geometry["value_field"]
        start, end, label = self._wind_vector_panel_points(geometry, active)
        x1, y1, x2, y2 = [float(value) for value in panel]
        px1, py1, px2, py2 = [float(value) for value in plot]
        outline = "#76d9ff" if active else "#3f5562"
        canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            fill="#071014",
            outline=outline,
            width=2 if active else 1,
            tags=("aug_overlay", "wind_vector_panel"),
        )
        self._register_canvas_overlay_region(
            "wind_panel_guard",
            (x1, y1, x2, y2),
        )
        canvas.create_text(
            x1 + 10,
            y1 + 13,
            text="Wiatr XY",
            anchor=tk.W,
            fill="#bfefff",
            font=("Segoe UI", 9, "bold"),
            tags=("aug_overlay", "wind_vector_panel"),
        )
        canvas.create_text(
            x2 - 10,
            y1 + 13,
            text="I ćwiartka",
            anchor=tk.E,
            fill="#9aa7ad",
            font=("Segoe UI", 7),
            tags=("aug_overlay", "wind_vector_panel"),
        )
        canvas.create_rectangle(
            x2 - 74,
            y1 + 4,
            x2 - 6,
            y1 + 22,
            fill="#071014",
            outline="",
            tags=("aug_overlay", "wind_vector_panel"),
        )
        canvas.create_text(
            x2 - 10,
            y1 + 13,
            text="360 deg",
            anchor=tk.E,
            fill="#9aa7ad",
            font=("Segoe UI", 7),
            tags=("aug_overlay", "wind_vector_panel"),
        )
        fx1, fy1, fx2, fy2 = [float(value) for value in value_field]
        canvas.create_text(
            fx1 - 5,
            (fy1 + fy2) / 2.0,
            text="V",
            anchor=tk.E,
            fill="#9aa7ad",
            font=("Segoe UI", 7, "bold"),
            tags=("aug_overlay", "wind_vector_panel"),
        )
        canvas.create_rectangle(
            fx1,
            fy1,
            fx2,
            fy2,
            fill="#10161a",
            outline="#47616f" if active else "#344149",
            width=1,
            tags=("aug_overlay", "wind_vector_panel"),
        )
        try:
            wind_value_text = f"{float(self.dirt_flow_wind_strength_var.get() or 0.0):.1f}"
        except Exception:
            wind_value_text = "0.0"
        canvas.create_text(
            (fx1 + fx2) / 2.0,
            (fy1 + fy2) / 2.0,
            text=wind_value_text,
            anchor=tk.CENTER,
            fill="#bfefff",
            font=("Segoe UI", 8, "bold"),
            tags=("aug_overlay", "wind_vector_panel"),
        )
        self._register_canvas_overlay_region("wind_strength_field", (fx1, fy1, fx2, fy2))
        canvas.create_rectangle(
            px1,
            py1,
            px2,
            py2,
            fill="#0b1218",
            outline="#243743",
            width=1,
            tags=("aug_overlay", "wind_vector_panel"),
        )
        dial_cx, dial_cy = start
        radius = float(geometry["scale"])
        canvas.create_oval(
            dial_cx - radius,
            dial_cy - radius,
            dial_cx + radius,
            dial_cy + radius,
            fill="#0d171d",
            outline="#3b5c68",
            width=1,
            tags=("aug_overlay", "wind_vector_panel"),
        )
        for ratio, color in ((0.50, "#1e3942"), (0.75, "#284b56")):
            inner = radius * ratio
            canvas.create_oval(
                dial_cx - inner,
                dial_cy - inner,
                dial_cx + inner,
                dial_cy + inner,
                outline=color,
                width=1,
                tags=("aug_overlay", "wind_vector_panel"),
            )
        canvas.create_line(dial_cx - radius, dial_cy, dial_cx + radius, dial_cy, fill="#596269", width=1, tags=("aug_overlay", "wind_vector_panel"))
        canvas.create_line(dial_cx, dial_cy - radius, dial_cx, dial_cy + radius, fill="#596269", width=1, tags=("aug_overlay", "wind_vector_panel"))
        canvas.create_text(dial_cx + radius + 5.0, dial_cy, text="+X", anchor=tk.W, fill="#ffb0b0", font=("Segoe UI", 8, "bold"), tags=("aug_overlay", "wind_vector_panel"))
        canvas.create_text(dial_cx - radius - 5.0, dial_cy, text="-X", anchor=tk.E, fill="#ffb0b0", font=("Segoe UI", 8, "bold"), tags=("aug_overlay", "wind_vector_panel"))
        canvas.create_text(dial_cx, dial_cy - radius - 5.0, text="+Y", anchor=tk.S, fill="#a7f0c2", font=("Segoe UI", 8, "bold"), tags=("aug_overlay", "wind_vector_panel"))
        canvas.create_text(dial_cx, dial_cy + radius + 5.0, text="-Y", anchor=tk.N, fill="#a7f0c2", font=("Segoe UI", 8, "bold"), tags=("aug_overlay", "wind_vector_panel"))
        canvas.create_line(
            start[0],
            start[1],
            end[0],
            end[1],
            fill="#031015",
            width=5,
            arrow=tk.LAST,
            arrowshape=(10, 12, 4),
            tags=("aug_overlay", "wind_vector_panel"),
        )
        canvas.create_line(
            start[0],
            start[1],
            end[0],
            end[1],
            fill="#76d9ff",
            width=2,
            arrow=tk.LAST,
            arrowshape=(10, 12, 4),
            tags=("aug_overlay", "wind_vector_panel"),
        )
        canvas.create_oval(start[0] - 4, start[1] - 4, start[0] + 4, start[1] + 4, fill="#071218", outline="#76d9ff", width=1, tags=("aug_overlay", "wind_vector_panel"))
        canvas.create_oval(end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4, fill="#76d9ff", outline="#071218", width=1, tags=("aug_overlay", "wind_vector_panel"))
        canvas.create_text(
            x1 + 10,
            y2 - 10,
            text=f"siła/kąt: {label}",
            anchor=tk.W,
            fill="#d7dde1",
            font=("Segoe UI", 7, "bold"),
            tags=("aug_overlay", "wind_vector_panel"),
        )
        self._register_canvas_overlay_region(
            "wind_vector_panel",
            (px1, py1, px2, py2),
            start=start,
            plot=plot,
            scale=float(geometry["scale"]),
        )

    def _headlight_var_group(self, index: int):
        if index == 1:
            return (
                self.traffic_headlight_source_x_var,
                self.traffic_headlight_source_y_var,
                self.traffic_headlight_target_x_var,
                self.traffic_headlight_target_y_var,
                self.traffic_headlight_var,
                self.traffic_headlight_1_warmth_var,
            )
        if index == 2:
            return (
                self.traffic_headlight_2_source_x_var,
                self.traffic_headlight_2_source_y_var,
                self.traffic_headlight_2_target_x_var,
                self.traffic_headlight_2_target_y_var,
                self.traffic_headlight_2_var,
                self.traffic_headlight_2_warmth_var,
            )
        return (
            self.traffic_headlight_3_source_x_var,
            self.traffic_headlight_3_source_y_var,
            self.traffic_headlight_3_target_x_var,
            self.traffic_headlight_3_target_y_var,
            self.traffic_headlight_3_var,
            self.traffic_headlight_3_warmth_var,
        )

    def _headlight_control_vars(self, index: int):
        if index == 1:
            return (
                self.traffic_headlight_var,
                self.traffic_headlight_1_r_var,
                self.traffic_headlight_1_g_var,
                self.traffic_headlight_1_b_var,
                self.traffic_headlight_1_cone_var,
                self.traffic_headlight_1_source_radius_var,
            )
        if index == 2:
            return (
                self.traffic_headlight_2_var,
                self.traffic_headlight_2_r_var,
                self.traffic_headlight_2_g_var,
                self.traffic_headlight_2_b_var,
                self.traffic_headlight_2_cone_var,
                self.traffic_headlight_2_source_radius_var,
            )
        return (
            self.traffic_headlight_3_var,
            self.traffic_headlight_3_r_var,
            self.traffic_headlight_3_g_var,
            self.traffic_headlight_3_b_var,
            self.traffic_headlight_3_cone_var,
            self.traffic_headlight_3_source_radius_var,
        )

    def _headlight_world_xy_vars(self, index: int, handle: str):
        handle = "target" if str(handle or "").lower() == "target" else "source"
        if index == 1:
            return (
                (self.traffic_headlight_target_world_x_var, self.traffic_headlight_target_world_y_var)
                if handle == "target"
                else (self.traffic_headlight_source_world_x_var, self.traffic_headlight_source_world_y_var)
            )
        if index == 2:
            return (
                (self.traffic_headlight_2_target_world_x_var, self.traffic_headlight_2_target_world_y_var)
                if handle == "target"
                else (self.traffic_headlight_2_source_world_x_var, self.traffic_headlight_2_source_world_y_var)
            )
        return (
            (self.traffic_headlight_3_target_world_x_var, self.traffic_headlight_3_target_world_y_var)
            if handle == "target"
            else (self.traffic_headlight_3_source_world_x_var, self.traffic_headlight_3_source_world_y_var)
        )

    def _headlight_norm_to_world_xy(self, norm_x: float, norm_y: float, *, source: bool) -> tuple[float, float]:
        try:
            plate_w = max(0.25, min(3.0, float(self.scene_plate_width_var.get() or 1.0)))
            plate_h = max(0.08, min(1.2, float(self.scene_plate_height_var.get() or 0.24)))
        except Exception:
            plate_w, plate_h = 1.0, 0.24
        return (float(norm_x) - 0.5) * plate_w, (0.5 - float(norm_y)) * plate_h

    def _headlight_world_xy_to_norm(self, world_x: float, world_y: float, *, source: bool) -> tuple[float, float]:
        try:
            plate_w = max(0.25, min(3.0, float(self.scene_plate_width_var.get() or 1.0)))
            plate_h = max(0.08, min(1.2, float(self.scene_plate_height_var.get() or 0.24)))
        except Exception:
            plate_w, plate_h = 1.0, 0.24
        x_norm = float(world_x) / max(0.0001, plate_w) + 0.5
        y_norm = 0.5 - float(world_y) / max(0.0001, plate_h)
        if source:
            return x_norm, y_norm
        return max(0.0, min(1.0, x_norm)), max(0.0, min(1.0, y_norm))

    def _scene_world_xy_to_canvas_point(self, world_x: float, world_y: float, width: int, height: int) -> tuple[int, int]:
        try:
            plate_w = max(0.25, min(3.0, float(self.scene_plate_width_var.get() or 1.0)))
            plate_h = max(0.08, min(1.2, float(self.scene_plate_height_var.get() or 0.24)))
        except Exception:
            plate_w, plate_h = 1.0, 0.24
        x_norm = float(world_x) / max(0.0001, plate_w) + 0.5
        y_norm = 0.5 - float(world_y) / max(0.0001, plate_h)
        return self._canvas_point_from_image_norm_unclamped(x_norm, y_norm, width, height)

    def _headlight_scene_world_point(self, index: int, handle: str) -> tuple[float, float, float]:
        handle = "target" if str(handle or "").strip().lower() == "target" else "source"
        defaults = (0.0, 0.0, 0.0) if handle == "target" else (0.0, 0.0, 0.95)
        try:
            x_var, y_var, z_var = self._headlight_world_var_triplet(index, handle)
            return (
                self._read_float_var(x_var, defaults[0]),
                self._read_float_var(y_var, defaults[1]),
                self._read_float_var(z_var, defaults[2]),
            )
        except Exception:
            return defaults

    def _headlight_scene_world_points(self, *, visible_only: bool = False) -> list[tuple[float, float, float]]:
        points: list[tuple[float, float, float]] = []
        for index in (1, 2, 3):
            if visible_only and not self._headlight_cones_are_visible(index):
                continue
            points.append(self._headlight_scene_world_point(index, "source"))
            points.append(self._headlight_scene_world_point(index, "target"))
        return points

    def _scene_3d_projection_context(self, width: int, height: int) -> dict | None:
        try:
            _rect, body_rect = self._scene_viewport_rects(int(width), int(height))
            x1, y1, x2, y2 = [float(value) for value in body_rect]
            plot_w = max(1.0, x2 - x1)
            plot_h = max(1.0, y2 - y1)
            camera_z = self._read_float_var(self.scene_camera_z_var, 1.65)
            camera_target_x = self._read_float_var(self.scene_camera_target_x_var, 0.0)
            camera_target_y = self._read_float_var(self.scene_camera_target_y_var, 0.0)
            plate = [(-0.5, -0.12, 0.0), (0.5, -0.12, 0.0), (0.5, 0.12, 0.0), (-0.5, 0.12, 0.0)]
            all_points = plate + [(0.0, 0.0, camera_z), (camera_target_x, camera_target_y, 0.0)]
            all_points.extend(self._headlight_scene_world_points(visible_only=True))
            for segment in self._scene_contour_wireframe_world_segments():
                try:
                    all_points.extend([segment["start"], segment["end"]])
                except Exception:
                    continue
            yaw, pitch, _roll = self._scene_view_angles()
            scale_matrix = self._scene_view_matrix_from_angles(yaw, pitch, 0.0)
            scale_rotated = [
                self._scene_view_rotate_point_with_matrix(point, scale_matrix)
                for point in all_points
            ]
            # Roll rotates the screen plane. Keep auto-fit scale based on the
            # unrolled view so roll never feels like zoom.
            scale_values = [(item[0], -item[2]) for item in scale_rotated]
            min_x = min(value[0] for value in scale_values)
            max_x = max(value[0] for value in scale_values)
            min_y = min(value[1] for value in scale_values)
            max_y = max(value[1] for value in scale_values)
            scale = min(plot_w / max(0.15, max_x - min_x + 0.18), plot_h / max(0.15, max_y - min_y + 0.18))
            scale *= max(0.20, min(8.0, float(getattr(self, "_scene_view_zoom", 1.0) or 1.0)))
            center_x = (x1 + x2) / 2.0 + float(getattr(self, "_scene_view_pan_x", 0.0) or 0.0)
            center_y = (y1 + y2) / 2.0 + float(getattr(self, "_scene_view_pan_y", 0.0) or 0.0)
            mid_x = (min_x + max_x) / 2.0
            mid_y = (min_y + max_y) / 2.0

            def project(point: tuple[float, float, float]) -> tuple[float, float]:
                rx, _ry, rz = self._scene_view_rotate_point(point)
                return center_x + (rx - mid_x) * scale, center_y + ((-rz) - mid_y) * scale

            return {
                "project": project,
                "scale": float(scale),
                "body_rect": body_rect,
                "center_x": float(center_x),
                "center_y": float(center_y),
                "mid_x": float(mid_x),
                "mid_y": float(mid_y),
                "matrix": self._scene_view_matrix(),
            }
        except Exception:
            return None

    def _scene_world_point_to_canvas_point(
        self,
        point: tuple[float, float, float],
        width: int,
        height: int,
    ) -> tuple[int, int] | None:
        context = self._scene_3d_projection_context(width, height)
        if not context:
            return None
        try:
            px, py = context["project"](point)
            return int(round(px)), int(round(py))
        except Exception:
            return None

    def _scene_canvas_point_to_headlight_world(
        self,
        index: int,
        handle: str,
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> tuple[float, float, float] | None:
        context = self._scene_3d_projection_context(width, height)
        if not context:
            return None
        try:
            scale = max(0.0001, float(context.get("scale", 1.0) or 1.0))
            desired_rx = (float(x) - float(context.get("center_x", 0.0))) / scale + float(context.get("mid_x", 0.0))
            desired_rz = -((float(y) - float(context.get("center_y", 0.0))) / scale + float(context.get("mid_y", 0.0)))
            matrix = context.get("matrix") or self._scene_view_matrix()
            row_x = matrix[0]
            row_z = matrix[2]
            current_x, current_y, current_z = self._headlight_scene_world_point(index, handle)
            normalized_handle = "target" if str(handle or "").strip().lower() == "target" else "source"
            mode = self._active_scene_viewport_mode()

            def solve_pair(a: float, b: float, c: float, d: float, rhs1: float, rhs2: float) -> tuple[float, float] | None:
                det = a * d - b * c
                if abs(det) <= 0.000001:
                    return None
                return (rhs1 * d - b * rhs2) / det, (a * rhs2 - rhs1 * c) / det

            def solve_single(coeff_x: float, coeff_z: float, fixed_z: float, fixed_y: float) -> float:
                rhs_x = desired_rx - row_x[1] * fixed_y - row_x[2] * (fixed_z - 0.42)
                rhs_z = desired_rz - row_z[1] * fixed_y - row_z[2] * (fixed_z - 0.42)
                if abs(coeff_x) >= abs(coeff_z) and abs(coeff_x) > 0.000001:
                    return rhs_x / coeff_x
                if abs(coeff_z) > 0.000001:
                    return rhs_z / coeff_z
                return current_x

            if normalized_handle == "target":
                if mode == "xz":
                    next_x = solve_single(row_x[0], row_z[0], 0.0, current_y)
                    return max(-0.5, min(0.5, next_x)), max(-0.12, min(0.12, current_y)), 0.0
                if mode == "yz":
                    rhs_x = desired_rx - row_x[0] * current_x - row_x[2] * -0.42
                    rhs_z = desired_rz - row_z[0] * current_x - row_z[2] * -0.42
                    if abs(row_x[1]) >= abs(row_z[1]) and abs(row_x[1]) > 0.000001:
                        next_y = rhs_x / row_x[1]
                    elif abs(row_z[1]) > 0.000001:
                        next_y = rhs_z / row_z[1]
                    else:
                        next_y = current_y
                    return max(-0.5, min(0.5, current_x)), max(-0.12, min(0.12, next_y)), 0.0
                solved = solve_pair(
                    row_x[0],
                    row_x[1],
                    row_z[0],
                    row_z[1],
                    desired_rx - row_x[2] * -0.42,
                    desired_rz - row_z[2] * -0.42,
                )
                if solved is None:
                    return None
                next_x, next_y = solved
                return max(-0.5, min(0.5, next_x)), max(-0.12, min(0.12, next_y)), 0.0

            if mode == "xz":
                solved = solve_pair(
                    row_x[0],
                    row_x[2],
                    row_z[0],
                    row_z[2],
                    desired_rx - row_x[1] * current_y + row_x[2] * 0.42,
                    desired_rz - row_z[1] * current_y + row_z[2] * 0.42,
                )
                if solved is None:
                    return None
                next_x, next_z = solved
                return max(-3.0, min(3.0, next_x)), max(-3.0, min(3.0, current_y)), max(0.0, min(8.0, next_z))
            if mode == "yz":
                solved = solve_pair(
                    row_x[1],
                    row_x[2],
                    row_z[1],
                    row_z[2],
                    desired_rx - row_x[0] * current_x + row_x[2] * 0.42,
                    desired_rz - row_z[0] * current_x + row_z[2] * 0.42,
                )
                if solved is None:
                    return None
                next_y, next_z = solved
                return max(-3.0, min(3.0, current_x)), max(-3.0, min(3.0, next_y)), max(0.0, min(8.0, next_z))
            solved = solve_pair(
                row_x[0],
                row_x[1],
                row_z[0],
                row_z[1],
                desired_rx - row_x[2] * (current_z - 0.42),
                desired_rz - row_z[2] * (current_z - 0.42),
            )
            if solved is None:
                return None
            next_x, next_y = solved
            return max(-3.0, min(3.0, next_x)), max(-3.0, min(3.0, next_y)), max(0.0, min(8.0, current_z))
        except Exception:
            return None

    def _scene_canvas_point_to_world_xy(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        clamp_to_plate: bool = False,
    ) -> tuple[float, float]:
        try:
            plate_w = max(0.25, min(3.0, float(self.scene_plate_width_var.get() or 1.0)))
            plate_h = max(0.08, min(1.2, float(self.scene_plate_height_var.get() or 0.24)))
        except Exception:
            plate_w, plate_h = 1.0, 0.24
        norm_x, norm_y = self._image_norm_from_canvas_point_unclamped(x, y, width, height)
        world_x = (float(norm_x) - 0.5) * plate_w
        world_y = (0.5 - float(norm_y)) * plate_h
        if clamp_to_plate:
            world_x = max(-plate_w / 2.0, min(plate_w / 2.0, world_x))
            world_y = max(-plate_h / 2.0, min(plate_h / 2.0, world_y))
        return world_x, world_y

    def _headlight_enabled_var(self, index: int):
        if index == 1:
            return self.traffic_headlight_1_enabled_var
        if index == 2:
            return self.traffic_headlight_2_enabled_var
        return self.traffic_headlight_3_enabled_var

    def _headlight_effect_enabled(self, index: int) -> bool:
        try:
            return bool(self._headlight_enabled_var(index).get())
        except Exception:
            return False

    def _read_float_var(self, variable, default: float = 0.0) -> float:
        try:
            value = variable.get()
            if value is None or str(value).strip() == "":
                return float(default)
            value = float(value)
            return value if math.isfinite(value) else float(default)
        except Exception:
            return float(default)

    def _read_int_var(
        self,
        variable,
        *,
        default: int = 0,
        minimum: int | None = None,
        maximum: int | None = None,
        remember_attr: str | None = None,
        repair: bool = False,
    ) -> int:
        fallback = default
        if remember_attr:
            try:
                fallback = int(getattr(self, remember_attr, default))
            except Exception:
                fallback = default
        try:
            raw = variable.get()
            if raw is None or str(raw).strip() == "":
                raise ValueError("empty numeric field")
            value = int(float(raw))
        except Exception:
            value = int(fallback)
        if minimum is not None:
            value = max(int(minimum), value)
        if maximum is not None:
            value = min(int(maximum), value)
        if remember_attr:
            try:
                setattr(self, remember_attr, int(value))
            except Exception:
                pass
        if repair:
            try:
                raw = variable.get()
                if raw is None or str(raw).strip() == "" or int(float(raw)) != int(value):
                    variable.set(int(value))
            except Exception:
                try:
                    variable.set(int(value))
                except Exception:
                    pass
        return int(value)

    def _set_headlight_representation_visible(self, index: int, visible: bool) -> None:
        try:
            index = int(index or 0)
        except Exception:
            index = 0
        if index not in (1, 2, 3):
            return
        state = self._headlight_visibility_state()
        state[index] = bool(visible)
        try:
            visibility_var = (getattr(self, "_headlight_visibility_vars", {}) or {}).get(index)
            if visibility_var is not None:
                visibility_var.set(bool(visible))
        except Exception:
            pass
        self._refresh_scene_summary()

    def _select_headlight(self, index: int | None, *, reveal: bool = True):
        previous = getattr(self, "_active_headlight_index", None)
        if index is None:
            self._active_headlight_index = None
            self._headlight_config_index = None
            self._headlight_panel_pos = None
            self._headlight_panel_anchor_index = None
            if previous is not None:
                self._refresh_effect_inspector_title()
            return
        index = int(index or 0)
        if index not in (1, 2, 3):
            self._active_headlight_index = None
            self._headlight_config_index = None
            self._headlight_panel_pos = None
            self._headlight_panel_anchor_index = None
            if previous is not None:
                self._refresh_effect_inspector_title()
            return
        self._active_headlight_index = index
        if self._headlight_config_index not in (None, index):
            self._headlight_config_index = None
            self._headlight_panel_pos = None
            self._headlight_panel_anchor_index = None
        if reveal:
            self._set_headlight_representation_visible(index, True)
        if previous != index:
            self._refresh_effect_inspector_title()

    def _toggle_headlight_effect(self, index: int):
        try:
            self._select_headlight(index, reveal=False)
            enabled_var = self._headlight_enabled_var(index)
            next_value = not bool(enabled_var.get())
            enabled_var.set(next_value)
            strength_var = self._headlight_control_vars(index)[0]
            if next_value and self._read_float_var(strength_var, 0.0) <= 0.001:
                strength_var.set(0.65)
            self._refresh_effect_inspector_title()
        except Exception:
            pass

    def _on_headlight_effect_checkbox_changed(self, index: int):
        try:
            index = int(index or 0)
        except Exception:
            index = 0
        if index not in (1, 2, 3):
            return
        try:
            self._select_headlight(index, reveal=False)
            enabled = self._headlight_effect_enabled(index)
            strength_var = self._headlight_control_vars(index)[0]
            if enabled and self._read_float_var(strength_var, 0.0) <= 0.001:
                strength_var.set(0.65)
            self._refresh_scene_summary()
            self._refresh_effect_inspector_title()
            self._draw_vector_overlay()
            self._schedule_preview_refresh(delay_ms=40)
        except Exception:
            pass

    def _headlight_visibility_state(self) -> dict[int, bool]:
        state = getattr(self, "_headlight_visibility", None)
        if not isinstance(state, dict):
            legacy_visible = bool(getattr(self, "_headlight_cones_visible", True))
            state = {1: legacy_visible, 2: legacy_visible, 3: legacy_visible}
            self._headlight_visibility = state
        for index in (1, 2, 3):
            state.setdefault(index, True)
        return state

    def _headlight_cones_are_visible(self, index: int | None = None) -> bool:
        state = self._headlight_visibility_state()
        if index is None:
            return any(bool(state.get(item, True)) for item in (1, 2, 3))
        try:
            index = int(index or 0)
        except Exception:
            index = 0
        if index not in (1, 2, 3):
            return False
        return bool(state.get(index, True))

    def _toggle_headlight_cone_visibility(self, index: int | None = None):
        try:
            index = int(index or 0)
        except Exception:
            index = 0
        if index not in (1, 2, 3):
            return
        self._select_headlight(index, reveal=False)
        state = self._headlight_visibility_state()
        next_value = not bool(state.get(index, True))
        self._set_headlight_representation_visible(index, next_value)

    def _default_headlight_norm(self, index: int) -> tuple[tuple[float, float], tuple[float, float]]:
        defaults = {
            1: ((0.14, 0.91), (0.42, 0.44)),
            2: ((0.86, 0.90), (0.58, 0.48)),
            3: ((0.50, 0.98), (0.50, 0.36)),
        }
        return defaults.get(index, defaults[1])

    def _preview_mapping_for_canvas(self, canvas: tk.Canvas | None = None) -> dict[str, float] | None:
        canvas = canvas or getattr(self, "augmented_canvas", None)
        if canvas is None:
            return None
        try:
            return (getattr(self, "_preview_canvas_mappings", {}) or {}).get(id(canvas))
        except Exception:
            return None

    def _canvas_point_from_image_norm(self, x_norm: float, y_norm: float, width: int, height: int) -> tuple[int, int]:
        x_norm = max(0.0, min(1.0, float(x_norm)))
        y_norm = max(0.0, min(1.0, float(y_norm)))
        return self._canvas_point_from_image_norm_unclamped(x_norm, y_norm, width, height)

    def _canvas_point_from_image_norm_unclamped(self, x_norm: float, y_norm: float, width: int, height: int) -> tuple[int, int]:
        x_norm = float(x_norm)
        y_norm = float(y_norm)
        mapping = self._preview_mapping_for_canvas(getattr(self, "augmented_canvas", None))
        if mapping:
            resized_x = x_norm * float(mapping.get("image_w", 1.0)) * float(mapping.get("scale", 1.0))
            resized_y = y_norm * float(mapping.get("image_h", 1.0)) * float(mapping.get("scale", 1.0))
            x = float(mapping.get("display_left", 0.0)) + resized_x - float(mapping.get("crop_left", 0.0))
            y = float(mapping.get("display_top", 0.0)) + resized_y - float(mapping.get("crop_top", 0.0))
            return (int(round(x)), int(round(y)))
        return (int(round(x_norm * max(1, width))), int(round(y_norm * max(1, height))))

    def _image_norm_from_canvas_point(self, x: float, y: float, width: float, height: float) -> tuple[float, float]:
        return self._image_norm_from_canvas_point_for_canvas(
            getattr(self, "augmented_canvas", None),
            x,
            y,
            width,
            height,
        )

    def _image_norm_from_canvas_point_for_canvas(
        self,
        canvas: tk.Canvas | None,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> tuple[float, float]:
        mapping = self._preview_mapping_for_canvas(canvas)
        if mapping:
            image_w = max(1.0, float(mapping.get("image_w", 1.0)))
            image_h = max(1.0, float(mapping.get("image_h", 1.0)))
            scale = max(0.000001, float(mapping.get("scale", 1.0)))
            resized_x = float(mapping.get("crop_left", 0.0)) + float(x) - float(mapping.get("display_left", 0.0))
            resized_y = float(mapping.get("crop_top", 0.0)) + float(y) - float(mapping.get("display_top", 0.0))
            return (
                max(0.0, min(1.0, resized_x / (image_w * scale))),
                max(0.0, min(1.0, resized_y / (image_h * scale))),
            )
        return (
            max(0.0, min(1.0, float(x) / max(1.0, float(width)))),
            max(0.0, min(1.0, float(y) / max(1.0, float(height)))),
        )

    def _image_norm_from_canvas_point_unclamped(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> tuple[float, float]:
        mapping = self._preview_mapping_for_canvas(getattr(self, "augmented_canvas", None))
        if mapping:
            image_w = max(1.0, float(mapping.get("image_w", 1.0)))
            image_h = max(1.0, float(mapping.get("image_h", 1.0)))
            scale = max(0.000001, float(mapping.get("scale", 1.0)))
            resized_x = float(mapping.get("crop_left", 0.0)) + float(x) - float(mapping.get("display_left", 0.0))
            resized_y = float(mapping.get("crop_top", 0.0)) + float(y) - float(mapping.get("display_top", 0.0))
            return resized_x / (image_w * scale), resized_y / (image_h * scale)
        return float(x) / max(1.0, float(width)), float(y) / max(1.0, float(height))

    def _set_preview_pan_for_canvas_anchor(
        self,
        canvas: tk.Canvas | None,
        x_norm: float,
        y_norm: float,
        pointer_x: float,
        pointer_y: float,
        zoom: float,
    ) -> bool:
        mapping = self._preview_mapping_for_canvas(canvas)
        if not mapping:
            return False
        try:
            canvas_w = max(1.0, float(canvas.winfo_width() if canvas is not None else mapping.get("canvas_w", 1.0)) or 1.0)
            canvas_h = max(1.0, float(canvas.winfo_height() if canvas is not None else mapping.get("canvas_h", 1.0)) or 1.0)
            image_w = max(1.0, float(mapping.get("image_w", 1.0)))
            image_h = max(1.0, float(mapping.get("image_h", 1.0)))
            max_w = max(1.0, canvas_w - 12.0)
            max_h = max(1.0, canvas_h - 12.0)
            base_scale = min(max_w / image_w, max_h / image_h)
            scale = max(0.01, base_scale * max(0.25, min(6.0, float(zoom))))
            resized_w = max(1.0, round(image_w * scale))
            resized_h = max(1.0, round(image_h * scale))

            display_w = min(resized_w, max_w)
            display_h = min(resized_h, max_h)
            display_left = (canvas_w - display_w) / 2.0
            display_top = (canvas_h - display_h) / 2.0

            if resized_w > max_w:
                desired_crop_left = float(x_norm) * image_w * scale - (float(pointer_x) - display_left)
                desired_crop_left = max(0.0, min(resized_w - max_w, desired_crop_left))
                self._preview_pan_x = (resized_w / 2.0) - (max_w / 2.0) - desired_crop_left
            else:
                max_pan_x = max(0.0, (max_w - resized_w) / 2.0)
                desired_pan_x = float(pointer_x) - display_left - float(x_norm) * image_w * scale
                self._preview_pan_x = max(-max_pan_x, min(max_pan_x, desired_pan_x))

            if resized_h > max_h:
                desired_crop_top = float(y_norm) * image_h * scale - (float(pointer_y) - display_top)
                desired_crop_top = max(0.0, min(resized_h - max_h, desired_crop_top))
                self._preview_pan_y = (resized_h / 2.0) - (max_h / 2.0) - desired_crop_top
            else:
                max_pan_y = max(0.0, (max_h - resized_h) / 2.0)
                desired_pan_y = float(pointer_y) - display_top - float(y_norm) * image_h * scale
                self._preview_pan_y = max(-max_pan_y, min(max_pan_y, desired_pan_y))
            return True
        except Exception:
            return False

    def _clamp_canvas_point_to_preview_image(self, x: float, y: float) -> tuple[float, float]:
        mapping = self._preview_mapping_for_canvas(getattr(self, "augmented_canvas", None))
        if not mapping:
            return float(x), float(y)
        left = float(mapping.get("display_left", 0.0))
        top = float(mapping.get("display_top", 0.0))
        right = left + max(1.0, float(mapping.get("display_w", 1.0)))
        bottom = top + max(1.0, float(mapping.get("display_h", 1.0)))
        return (
            max(left, min(right, float(x))),
            max(top, min(bottom, float(y))),
        )

    def _headlight_display_unit(self, width: int, height: int) -> float:
        if bool(getattr(self, "_scene_viewport_visible", False)):
            context = self._scene_3d_projection_context(width, height)
            if context:
                try:
                    return max(1.0, float(context.get("scale", 1.0) or 1.0) * 0.16)
                except Exception:
                    pass
        mapping = self._preview_mapping_for_canvas(getattr(self, "augmented_canvas", None))
        if mapping:
            return max(
                1.0,
                min(float(mapping.get("image_w", 1.0)), float(mapping.get("image_h", 1.0)))
                * float(mapping.get("scale", 1.0)),
            )
        return max(1.0, float(min(width, height)))

    def _headlight_radius_knob_extension(self) -> float:
        return 16.0

    def _headlight_radius_axis(
        self,
        center: tuple[int, int],
        handle: str,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        """Keep source and footprint radius handles visually separable."""
        cx = float(center[0])
        cy = float(center[1])
        handle = str(handle or "").strip().lower()
        margin = 72.0
        if handle == "source":
            return (0.0, 1.0) if cy < margin else (0.0, -1.0)
        return (-1.0, 0.0) if cx > float(width) - margin else (1.0, 0.0)

    def _default_headlight_points(self, index: int, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        source, target = self._default_headlight_norm(index)
        return (
            self._canvas_point_from_image_norm(source[0], source[1], width, height),
            self._canvas_point_from_image_norm(target[0], target[1], width, height),
        )

    def _headlight_points(self, index: int, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        sx_var, sy_var, tx_var, ty_var, _strength_var, _warmth_var = self._headlight_var_group(index)
        default_source, default_target = self._default_headlight_points(index, width, height)
        try:
            if bool(getattr(self, "_scene_viewport_visible", False)):
                source = self._scene_world_point_to_canvas_point(
                    self._headlight_scene_world_point(index, "source"),
                    width,
                    height,
                )
                target = self._scene_world_point_to_canvas_point(
                    self._headlight_scene_world_point(index, "target"),
                    width,
                    height,
                )
                if source is not None and target is not None:
                    return source, target
            sx = self._read_float_var(sx_var, -1.0)
            sy = self._read_float_var(sy_var, -1.0)
            tx = self._read_float_var(tx_var, -1.0)
            ty = self._read_float_var(ty_var, -1.0)
            source = default_source
            target = default_target
            source_world_x_var, source_world_y_var = self._headlight_world_xy_vars(index, "source")
            target_world_x_var, target_world_y_var = self._headlight_world_xy_vars(index, "target")
            source_world_x = self._read_float_var(source_world_x_var, -999.0)
            source_world_y = self._read_float_var(source_world_y_var, -999.0)
            target_world_x = self._read_float_var(target_world_x_var, -999.0)
            target_world_y = self._read_float_var(target_world_y_var, -999.0)
            if source_world_x > -998.0 and source_world_y > -998.0:
                source = self._scene_world_xy_to_canvas_point(source_world_x, source_world_y, width, height)
            elif 0.0 <= sx <= 1.0 and 0.0 <= sy <= 1.0:
                source = self._canvas_point_from_image_norm(sx, sy, width, height)
            if target_world_x > -998.0 and target_world_y > -998.0:
                target = self._scene_world_xy_to_canvas_point(target_world_x, target_world_y, width, height)
            elif 0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0:
                target = self._canvas_point_from_image_norm(tx, ty, width, height)
            return source, target
        except Exception:
            pass
        return default_source, default_target

    def _ensure_headlight_pair_defaults(self, index: int, width: float, height: float) -> None:
        sx_var, sy_var, tx_var, ty_var, _strength_var, _warmth_var = self._headlight_var_group(index)
        default_source, default_target = self._default_headlight_norm(index)

        def is_valid_pair(x_var, y_var) -> bool:
            try:
                x = self._read_float_var(x_var, -1.0)
                y = self._read_float_var(y_var, -1.0)
                return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
            except Exception:
                return False

        def set_normalized_if_missing(x_var, y_var, point: tuple[int, int]) -> None:
            if is_valid_pair(x_var, y_var):
                return
            try:
                x_var.set(max(0.0, min(1.0, float(point[0]))))
                y_var.set(max(0.0, min(1.0, float(point[1]))))
            except Exception:
                pass

        set_normalized_if_missing(sx_var, sy_var, default_source)
        set_normalized_if_missing(tx_var, ty_var, default_target)

    def _headlight_color(self, warmth: float, strength: float) -> str:
        warmth = max(0.0, min(1.0, float(warmth)))
        cold = (118, 217, 255)
        warm = (255, 211, 106)
        color = tuple(int(cold[i] * (1.0 - warmth) + warm[i] * warmth) for i in range(3))
        if strength <= 0.001:
            color = tuple(int(channel * 0.56) for channel in color)
        return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

    def _headlight_color_from_vars(self, index: int, strength: float) -> str:
        try:
            _strength, red_var, green_var, blue_var, _cone, _source_radius = self._headlight_control_vars(index)
            red = max(0.0, min(1.0, float(red_var.get() or 0.0)))
            green = max(0.0, min(1.0, float(green_var.get() or 0.0)))
            blue = max(0.0, min(1.0, float(blue_var.get() or 0.0)))
            color = (int(red * 255.0), int(green * 255.0), int(blue * 255.0))
            if strength <= 0.001:
                color = tuple(int(channel * 0.72) for channel in color)
            luminance = color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722
            if luminance < 118:
                # UI handles must stay readable even when the simulated light is dark.
                fallback = (255, 211, 106)
                blend = 0.68 if strength > 0.001 else 0.82
                color = tuple(int(color[i] * (1.0 - blend) + fallback[i] * blend) for i in range(3))
            return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        except Exception:
            return self._headlight_color(0.35, strength)

    def _draw_headlight_toggle_bar(self, canvas: tk.Canvas, width: int, height: int, active_index: int):
        label_w = 72
        col_w = 42
        row_h = 20
        panel_w = label_w + col_w * 3 + 12
        panel_h = min(68, max(58, height - 84))
        x = 10
        y = min(max(54, height - panel_h - 12), max(4, height - panel_h - 4))
        canvas.create_rectangle(
            x,
            y,
            x + panel_w,
            y + panel_h,
            fill="#0d1418",
            outline="#344149",
            width=1,
            tags=("vector_overlay", "headlight_control"),
        )
        canvas.create_text(
            x + 8,
            y + 10,
            text="Reflektory",
            anchor=tk.W,
            fill="#d7dde1",
            font=("Segoe UI", 8),
            tags=("vector_overlay", "headlight_control"),
        )
        header_y = y + 12
        for offset, index in enumerate((1, 2, 3)):
            cx = x + label_w + col_w * offset + col_w / 2
            canvas.create_text(
                cx,
                header_y,
                text=f"R{index}",
                anchor=tk.CENTER,
                fill="#fff3b0" if index == active_index else "#d7dde1",
                font=("Segoe UI", 8, "bold" if index == active_index else "normal"),
                tags=("vector_overlay", "headlight_control"),
            )

        rows = (
            ("Widok", "headlight_cone_toggle"),
            ("Efekt", "headlight_toggle"),
        )
        for row_index, (row_label, action_kind) in enumerate(rows):
            row_y = y + 21 + row_index * row_h
            canvas.create_text(
                x + 8,
                row_y + row_h / 2,
                text=row_label,
                anchor=tk.W,
                fill="#d7dde1",
                font=("Segoe UI", 8),
                tags=("vector_overlay", "headlight_control"),
            )
            canvas.create_line(
                x + 6,
                row_y,
                x + panel_w - 6,
                row_y,
                fill="#243039",
                width=1,
                tags=("vector_overlay", "headlight_control"),
            )
            for offset, index in enumerate((1, 2, 3)):
                cell_x = x + label_w + col_w * offset + 4
                cell_y = row_y + 2
                cell_w = col_w - 8
                cell_h = row_h - 4
                if action_kind == "headlight_cone_toggle":
                    active = self._headlight_cones_are_visible(index)
                    outline = "#ff9f1c"
                    fill = "#2a1a0d"
                    text = "TAK" if active else "NIE"
                    text_fill = "#ffd29a" if active else "#9aa7ad"
                else:
                    enabled = self._headlight_effect_enabled(index)
                    active = enabled
                    outline = "#58d68d"
                    fill = "#14291f"
                    text = "ON" if active else "OFF"
                    text_fill = "#bfffd6" if active else "#9aa7ad"
                if not active:
                    outline = "#4b555c"
                    fill = "#151a1f"
                canvas.create_rectangle(
                    cell_x,
                    cell_y,
                    cell_x + cell_w,
                    cell_y + cell_h,
                    fill=fill,
                    outline=outline,
                    width=1,
                    tags=("vector_overlay", "headlight_control"),
                )
                canvas.create_text(
                    cell_x + cell_w / 2,
                    cell_y + cell_h / 2,
                    text=text,
                    anchor=tk.CENTER,
                    fill=text_fill,
                    font=("Segoe UI", 7),
                    tags=("vector_overlay", "headlight_control"),
                )
                self._register_canvas_overlay_region(
                    action_kind,
                    (cell_x, cell_y, cell_x + cell_w, cell_y + cell_h),
                    index=index,
                )

    def _headlight_config_lollipop_geometry(
        self,
        source: tuple[int, int],
        target: tuple[int, int],
        width: int,
        height: int,
        source_half_width: float = 0.0,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        sx = float(source[0])
        sy = float(source[1])
        tx = float(target[0])
        ty = float(target[1])
        vx = sx - tx
        vy = sy - ty
        raw_distance = math.hypot(vx, vy)
        if raw_distance <= 4.0:
            ux = 0.86
            uy = -0.50
        else:
            distance = max(1.0, raw_distance)
            ux = vx / distance
            uy = vy / distance
        margin = 16.0
        start_dist = max(24.0, float(source_half_width or 0.0) + 10.0)
        end_dist = start_dist + 42.0
        limits = [end_dist]
        if abs(ux) > 0.001:
            limits.append(((float(width) - margin - sx) / ux) if ux > 0 else ((margin - sx) / ux))
        if abs(uy) > 0.001:
            limits.append(((float(height) - margin - sy) / uy) if uy > 0 else ((margin - sy) / uy))
        positive_limits = [value for value in limits if value > start_dist + 8.0]
        if positive_limits:
            end_dist = max(start_dist + 8.0, min(end_dist, min(positive_limits)))
        return (
            (sx + ux * start_dist, sy + uy * start_dist),
            (sx + ux * end_dist, sy + uy * end_dist),
        )

    def _draw_headlight_config_lollipop(
        self,
        canvas: tk.Canvas,
        index: int,
        source: tuple[int, int],
        target: tuple[int, int],
        width: int,
        height: int,
        *,
        color: str,
        expanded: bool,
        source_half_width: float,
    ):
        (x1, y1), (cx, cy) = self._headlight_config_lollipop_geometry(source, target, width, height, source_half_width)
        hover = getattr(self, "_headlight_config_hover_index", None) == index
        outline = "#ffad33" if expanded or hover else "#ff8f1f"
        fill = "#111820"
        button_w = 48 if hover or expanded else 44
        button_h = 20 if hover or expanded else 18
        canvas.create_line(
            x1,
            y1,
            cx,
            cy,
            fill=outline,
            width=1,
            dash=(2, 3),
            tags=("vector_overlay", "headlight_control"),
        )
        canvas.create_rectangle(
            cx - button_w / 2,
            cy - button_h / 2,
            cx + button_w / 2,
            cy + button_h / 2,
            fill=fill,
            outline=outline,
            width=2 if hover or expanded else 1,
            tags=("vector_overlay", "headlight_control"),
        )
        canvas.create_text(
            cx - button_w / 2 + 8,
            cy,
            text=f"R{index}",
            anchor=tk.W,
            fill=outline,
            font=("Segoe UI", 8, "bold"),
            tags=("vector_overlay", "headlight_control"),
        )
        menu_x1 = cx + button_w / 2 - 17
        menu_x2 = cx + button_w / 2 - 6
        for offset in (-5, 0, 5):
            y = cy + offset
            canvas.create_line(
                menu_x1,
                y,
                menu_x2,
                y,
                fill=color if offset == 0 else outline,
                width=1,
                tags=("vector_overlay", "headlight_control"),
            )
        hit_pad_x = button_w / 2 + 8
        hit_pad_y = button_h / 2 + 8
        self._register_canvas_overlay_region(
            "headlight_config_toggle",
            (cx - hit_pad_x, cy - hit_pad_y, cx + hit_pad_x, cy + hit_pad_y),
            index=index,
        )

    def _draw_headlight_target_footprint(
        self,
        canvas: tk.Canvas,
        index: int,
        canvas_width: int,
        canvas_height: int,
        radius_px: float,
        *,
        outline: str,
        dash: tuple[int, int] | None = None,
        line_width: int = 1,
    ) -> bool:
        if not bool(getattr(self, "_scene_viewport_visible", False)):
            return False
        context = self._scene_3d_projection_context(canvas_width, canvas_height)
        if not context:
            return False
        try:
            project = context["project"]
            scale = max(0.0001, float(context.get("scale", 1.0) or 1.0))
            radius_world = max(0.001, float(radius_px) / scale)
            target_x, target_y, _target_z = self._headlight_scene_world_point(index, "target")
            coords: list[float] = []
            steps = 48
            for step in range(steps + 1):
                angle = (math.tau * float(step)) / float(steps)
                px, py = project(
                    (
                        float(target_x) + math.cos(angle) * radius_world,
                        float(target_y) + math.sin(angle) * radius_world,
                        0.0,
                    )
                )
                coords.extend((float(px), float(py)))
            canvas.create_line(
                *coords,
                fill=outline,
                width=line_width,
                dash=dash,
                smooth=True,
                tags=("vector_overlay", "headlight_control"),
            )
            return True
        except Exception:
            return False

    def _draw_headlight_scene_cone_guides(
        self,
        canvas: tk.Canvas,
        index: int,
        canvas_width: int,
        canvas_height: int,
        source_radius_px: float,
        target_radius_px: float,
        *,
        outline: str,
        dash: tuple[int, int] | None = None,
        line_width: int = 1,
    ) -> bool:
        if not bool(getattr(self, "_scene_viewport_visible", False)):
            return False
        context = self._scene_3d_projection_context(canvas_width, canvas_height)
        if not context:
            return False
        try:
            project = context["project"]
            scale = max(0.0001, float(context.get("scale", 1.0) or 1.0))
            source_radius_world = max(0.0, float(source_radius_px) / scale)
            target_radius_world = max(0.001, float(target_radius_px) / scale)
            source_world = self._headlight_scene_world_point(index, "source")
            target_x, target_y, target_z = self._headlight_scene_world_point(index, "target")
            for angle in (0.0, math.pi / 2.0, math.pi, math.pi * 1.5):
                source_edge_x, source_edge_y = project(
                    (
                        float(source_world[0]) + math.cos(angle) * source_radius_world,
                        float(source_world[1]) + math.sin(angle) * source_radius_world,
                        float(source_world[2]),
                    )
                )
                edge_x, edge_y = project(
                    (
                        float(target_x) + math.cos(angle) * target_radius_world,
                        float(target_y) + math.sin(angle) * target_radius_world,
                        float(target_z),
                    )
                )
                canvas.create_line(
                    source_edge_x,
                    source_edge_y,
                    edge_x,
                    edge_y,
                    fill=outline,
                    width=line_width,
                    dash=dash,
                    tags=("vector_overlay", "headlight_control"),
                )
            return True
        except Exception:
            return False

    def _draw_headlight_handles(self, canvas: tk.Canvas, width: int, height: int):
        active_index = getattr(self, "_active_headlight_index", None)
        if active_index not in (1, 2, 3):
            active_index = None
            self._active_headlight_index = None
        config_index = getattr(self, "_headlight_config_index", None)
        if config_index not in (1, 2, 3):
            config_index = None
            self._headlight_config_index = None
        self._draw_headlight_toggle_bar(canvas, width, height, active_index)

        def _draw_one(index: int) -> None:
            is_active = index == active_index
            is_configured = index == config_index
            is_dragged = bool(
                getattr(self, "_headlight_drag", None)
                and int((getattr(self, "_headlight_drag", {}) or {}).get("index", 0) or 0) == index
            )
            is_visible = self._headlight_cones_are_visible(index)
            if not is_visible:
                return
            _sx_var, _sy_var, _tx_var, _ty_var, strength_var, _warmth_var = self._headlight_var_group(index)
            try:
                raw_strength = max(0.0, min(1.0, float(strength_var.get() or 0.0)))
            except Exception:
                raw_strength = 0.0
            enabled = self._headlight_effect_enabled(index)
            visible_strength = raw_strength if enabled else 0.0
            source, target = self._headlight_points(index, width, height)
            color = self._headlight_color_from_vars(index, visible_strength)
            try:
                _strength_var, _red_var, _green_var, _blue_var, cone_var, source_radius_var = self._headlight_control_vars(index)
                cone = max(0.02, min(2.5, float(cone_var.get() or 0.45)))
                source_radius = max(0.0, min(2.5, float(source_radius_var.get() or 0.0)))
            except Exception:
                cone = 0.45
                source_radius = 0.08
            is_prominent = is_active or is_configured or is_dragged
            line_width = 1
            guide_width = 2 if is_prominent else 1
            dash = None if is_visible and enabled and raw_strength > 0.001 else (4, 4)
            vx = float(target[0] - source[0])
            vy = float(target[1] - source[1])
            raw_distance = math.hypot(vx, vy)
            distance = max(1.0, raw_distance)
            if raw_distance <= 0.0001:
                ux = 0.0
                uy = 1.0
            else:
                ux = vx / distance
                uy = vy / distance
            px = -uy
            py = ux
            unit = self._headlight_display_unit(width, height)
            source_value_half_width = max(0.0, source_radius * unit)
            target_value_half_width = max(0.0, cone * unit)
            source_half_width = max(3.0, source_value_half_width)
            target_half_width = max(3.0, target_value_half_width)
            representation_outline = "#43c56f" if is_prominent else "#ff8f1f"
            axis_color = "#43c56f" if is_prominent else color
            marker_fill = "#0f2d1d" if is_prominent else "#111820"
            cone_outline = representation_outline
            cone_width = 1
            target_cone_footprint_drawn = False
            if target_half_width > 1.0 or source_half_width > 1.0:
                scene_cone_drawn = self._draw_headlight_scene_cone_guides(
                    canvas,
                    index,
                    width,
                    height,
                    source_value_half_width,
                    target_half_width,
                    outline=cone_outline,
                    dash=(3, 3),
                    line_width=cone_width,
                )
                if scene_cone_drawn:
                    target_cone_footprint_drawn = self._draw_headlight_target_footprint(
                        canvas,
                        index,
                        width,
                        height,
                        target_half_width,
                        outline=cone_outline,
                        dash=(3, 3),
                        line_width=cone_width,
                    )
                elif raw_distance <= max(6.0, min(source_half_width, target_half_width) * 0.30):
                    cx = (float(source[0]) + float(target[0])) / 2.0
                    cy = (float(source[1]) + float(target[1])) / 2.0
                    target_cone_footprint_drawn = self._draw_headlight_target_footprint(
                        canvas,
                        index,
                        width,
                        height,
                        target_half_width,
                        outline=cone_outline,
                        dash=(3, 3),
                        line_width=cone_width,
                    )
                    if not target_cone_footprint_drawn:
                        canvas.create_oval(
                            cx - target_half_width,
                            cy - target_half_width,
                            cx + target_half_width,
                            cy + target_half_width,
                            fill="",
                            outline=cone_outline,
                            width=cone_width,
                            dash=(3, 3),
                            tags=("vector_overlay", "headlight_control"),
                        )
                    canvas.create_oval(
                        cx - source_half_width,
                        cy - source_half_width,
                        cx + source_half_width,
                        cy + source_half_width,
                        fill="",
                        outline=cone_outline,
                        width=cone_width,
                        dash=(2, 4),
                        tags=("vector_overlay", "headlight_control"),
                    )
                else:
                    cone_points = (
                        source[0] + px * source_half_width,
                        source[1] + py * source_half_width,
                        target[0] + px * target_half_width,
                        target[1] + py * target_half_width,
                        target[0] - px * target_half_width,
                        target[1] - py * target_half_width,
                        source[0] - px * source_half_width,
                        source[1] - py * source_half_width,
                    )
                    canvas.create_polygon(
                        *cone_points,
                        fill="",
                        outline=cone_outline,
                        width=cone_width,
                        dash=(3, 3),
                        tags=("vector_overlay", "headlight_control"),
                    )
            canvas.create_line(
                source[0],
                source[1],
                target[0],
                target[1],
                fill=representation_outline,
                width=guide_width,
                dash=dash,
                tags=("vector_overlay", "headlight_control"),
            )
            canvas.create_line(
                source[0],
                source[1],
                target[0],
                target[1],
                fill=axis_color,
                width=line_width,
                dash=dash,
                tags=("vector_overlay", "headlight_control"),
            )
            radius = 7 if enabled and raw_strength > 0.001 else 5
            if is_prominent:
                radius += 1
            canvas.create_oval(
                source[0] - source_half_width,
                source[1] - source_half_width,
                source[0] + source_half_width,
                source[1] + source_half_width,
                fill="",
                outline=representation_outline,
                width=1,
                dash=(2, 3),
                tags=("vector_overlay", "headlight_control"),
            )
            target_footprint_drawn = target_cone_footprint_drawn or self._draw_headlight_target_footprint(
                canvas,
                index,
                width,
                height,
                target_half_width,
                outline=representation_outline,
                dash=(3, 3),
            )
            if not target_footprint_drawn:
                canvas.create_oval(
                    target[0] - target_half_width,
                    target[1] - target_half_width,
                    target[0] + target_half_width,
                    target[1] + target_half_width,
                    fill="",
                    outline=representation_outline,
                    width=1,
                    dash=(3, 3),
                    tags=("vector_overlay", "headlight_control"),
                )
            canvas.create_oval(
                source[0] - radius,
                source[1] - radius,
                source[0] + radius,
                source[1] + radius,
                fill=marker_fill,
                outline=representation_outline,
                width=1,
                tags=("vector_overlay", "headlight_control"),
            )
            canvas.create_oval(
                target[0] - 10,
                target[1] - 10,
                target[0] + 10,
                target[1] + 10,
                fill="",
                outline=representation_outline,
                width=1,
                tags=("vector_overlay", "headlight_control"),
            )
            canvas.create_oval(
                target[0] - 6,
                target[1] - 6,
                target[0] + 6,
                target[1] + 6,
                fill=marker_fill,
                outline=representation_outline,
                width=1,
                tags=("vector_overlay", "headlight_control"),
            )
            hit_pad = 17
            self._register_canvas_overlay_region(
                "headlight",
                (source[0] - hit_pad, source[1] - hit_pad, source[0] + hit_pad + 6, source[1] + hit_pad + 6),
                index=index,
                handle="source",
            )
            self._register_canvas_overlay_region(
                "headlight",
                (target[0] - hit_pad, target[1] - hit_pad, target[0] + hit_pad, target[1] + hit_pad),
                index=index,
                handle="target",
            )
            if is_prominent:
                def draw_radius_knob(
                    center: tuple[int, int],
                    visual_half_width: float,
                    value_half_width: float,
                    handle_name: str,
                ) -> None:
                    extension = self._headlight_radius_knob_extension()
                    axis_x, axis_y = self._headlight_radius_axis(center, handle_name, width, height)
                    edge_x = float(center[0]) + axis_x * float(visual_half_width)
                    edge_y = float(center[1]) + axis_y * float(visual_half_width)
                    knob_x = float(center[0]) + axis_x * (float(value_half_width) + extension)
                    knob_y = float(center[1]) + axis_y * (float(value_half_width) + extension)
                    knob_r = 4.5
                    canvas.create_line(
                        edge_x,
                        edge_y,
                        knob_x,
                        knob_y,
                        fill="#ff4d4d",
                        width=1,
                        dash=(2, 2),
                        tags=("vector_overlay", "headlight_control"),
                    )
                    canvas.create_oval(
                        knob_x - knob_r - 0.7,
                        knob_y - knob_r - 0.7,
                        knob_x + knob_r + 0.7,
                        knob_y + knob_r + 0.7,
                        fill="#ff3030",
                        outline="#fff3b0",
                        width=1,
                        tags=("vector_overlay", "headlight_control"),
                    )
                    arrow_len = 16.0
                    tip_x = knob_x + axis_x * arrow_len
                    tip_y = knob_y + axis_y * arrow_len
                    if handle_name == "source":
                        arrow_start = (knob_x + axis_x * (knob_r + 2.0), knob_y + axis_y * (knob_r + 2.0))
                        arrow_end = (tip_x, tip_y)
                    else:
                        arrow_start = (tip_x, tip_y)
                        arrow_end = (knob_x + axis_x * (knob_r + 2.0), knob_y + axis_y * (knob_r + 2.0))
                    canvas.create_line(
                        arrow_start[0],
                        arrow_start[1],
                        arrow_end[0],
                        arrow_end[1],
                        fill="#ff3030",
                        width=2,
                        arrow=tk.LAST,
                        arrowshape=(7, 9, 3),
                        tags=("vector_overlay", "headlight_control"),
                    )
                    hit_pad = 10.0
                    self._register_canvas_overlay_region(
                        "headlight_radius",
                        (knob_x - hit_pad, knob_y - hit_pad, knob_x + hit_pad, knob_y + hit_pad),
                        index=index,
                        handle=handle_name,
                    )

                draw_radius_knob(source, source_half_width, source_value_half_width, "source")
                draw_radius_knob(target, target_half_width, target_value_half_width, "target")
            self._draw_headlight_config_lollipop(
                canvas,
                index,
                source,
                target,
                width,
                height,
                color=color,
                expanded=config_index == index,
                source_half_width=source_half_width,
            )

        for headlight_index in (1, 2, 3):
            _draw_one(headlight_index)
        return

    def _set_headlight_radius_handle(self, region: dict, x: float, y: float):
        canvas = getattr(self, "augmented_canvas", None)
        if canvas is None:
            return
        width = max(1, int(canvas.winfo_width() or 1))
        height = max(1, int(canvas.winfo_height() or 1))
        try:
            index = int(region.get("index", 1) or 1)
        except Exception:
            index = 1
        handle = "source" if str(region.get("handle") or "").strip().lower() == "source" else "target"
        try:
            source, target = self._headlight_points(index, width, height)
            center = source if handle == "source" else target
            unit = max(1.0, self._headlight_display_unit(width, height))
            extension = self._headlight_radius_knob_extension()
            axis_x, axis_y = self._headlight_radius_axis(center, handle, width, height)
            projected = (float(x) - float(center[0])) * axis_x + (float(y) - float(center[1])) * axis_y
            radius_px = max(0.0, projected - extension)
            next_value = radius_px / unit
            _strength_var, _red_var, _green_var, _blue_var, cone_var, source_radius_var = self._headlight_control_vars(index)
            if handle == "source":
                next_value = max(0.0, min(2.5, next_value))
                current = self._read_float_var(source_radius_var, 0.0)
                if abs(current - next_value) > 0.002:
                    source_radius_var.set(next_value)
            else:
                next_value = max(0.02, min(2.5, next_value))
                current = self._read_float_var(cone_var, 0.45)
                if abs(current - next_value) > 0.002:
                    cone_var.set(next_value)
            self._refresh_scene_summary()
        except Exception:
            pass

    def _set_headlight_handle(self, region: dict, x: float, y: float):
        canvas = getattr(self, "augmented_canvas", None)
        if canvas is None:
            return
        width = max(1.0, float(canvas.winfo_width() or 1))
        height = max(1.0, float(canvas.winfo_height() or 1))
        index = int(region.get("index", 1) or 1)
        handle = str(region.get("handle") or "source")
        sx_var, sy_var, tx_var, ty_var, _strength_var, _warmth_var = self._headlight_var_group(index)
        self._ensure_headlight_pair_defaults(index, width, height)
        def set_if_changed(variable, value: float) -> None:
            try:
                if abs(self._read_float_var(variable, -1.0) - float(value)) > 0.0008:
                    variable.set(value)
            except Exception:
                variable.set(value)
        try:
            if bool(getattr(self, "_scene_viewport_visible", False)):
                world_point = self._scene_canvas_point_to_headlight_world(
                    index,
                    handle,
                    float(x),
                    float(y),
                    int(width),
                    int(height),
                )
                if world_point is not None:
                    world_x, world_y, world_z = world_point
                    world_x_var, world_y_var, world_z_var = self._headlight_world_var_triplet(index, handle)
                    set_if_changed(world_x_var, world_x)
                    set_if_changed(world_y_var, world_y)
                    set_if_changed(world_z_var, world_z)
                    nx, ny = self._headlight_world_xy_to_norm(world_x, world_y, source=handle != "target")
                    if handle == "target":
                        set_if_changed(tx_var, nx)
                        set_if_changed(ty_var, ny)
                    else:
                        set_if_changed(sx_var, nx)
                        set_if_changed(sy_var, ny)
                    self._refresh_scene_summary()
                    return
            if handle == "target":
                clamped_x, clamped_y = self._clamp_canvas_point_to_preview_image(float(x), float(y))
                nx, ny = self._image_norm_from_canvas_point(clamped_x, clamped_y, width, height)
                set_if_changed(tx_var, nx)
                set_if_changed(ty_var, ny)
                world_x, world_y = self._scene_canvas_point_to_world_xy(
                    clamped_x,
                    clamped_y,
                    width,
                    height,
                    clamp_to_plate=True,
                )
            else:
                nx, ny = self._image_norm_from_canvas_point_unclamped(float(x), float(y), width, height)
                set_if_changed(sx_var, nx)
                set_if_changed(sy_var, ny)
                world_x, world_y = self._scene_canvas_point_to_world_xy(
                    float(x),
                    float(y),
                    width,
                    height,
                    clamp_to_plate=False,
                )
                world_x = max(-3.0, min(3.0, world_x))
                world_y = max(-3.0, min(3.0, world_y))
            world_x_var, world_y_var = self._headlight_world_xy_vars(index, handle)
            set_if_changed(world_x_var, world_x)
            set_if_changed(world_y_var, world_y)
            if handle == "target":
                try:
                    _x_var, _y_var, z_var = self._headlight_world_var_triplet(index, "target")
                    set_if_changed(z_var, 0.0)
                except Exception:
                    pass
            self._refresh_scene_summary()
        except Exception:
            pass

    def _draw_headlight_config_toolbox(self, canvas: tk.Canvas, width: int, height: int):
        index = getattr(self, "_headlight_config_index", None)
        if index not in (1, 2, 3):
            return
        panel_w = min(260, max(220, width - 24))
        panel_h = 188
        index = int(index)
        panel_pos = getattr(self, "_headlight_panel_pos", None)
        if panel_pos is None or getattr(self, "_headlight_panel_anchor_index", None) != index:
            source, target = self._headlight_points(index, width, height)
            try:
                _strength_var, _red_var, _green_var, _blue_var, _cone_var, source_radius_var = self._headlight_control_vars(index)
                source_radius = max(0.0, min(2.5, float(source_radius_var.get() or 0.0)))
            except Exception:
                source_radius = 0.08
            source_half_width = max(3.0, source_radius * self._headlight_display_unit(width, height))
            _stem_start, anchor = self._headlight_config_lollipop_geometry(
                source,
                target,
                width,
                height,
                source_half_width,
            )
            x = int(anchor[0] + 24)
            y = int(anchor[1] + 24)
            if x + panel_w > width - 8:
                x = int(anchor[0] - panel_w - 24)
            if y + panel_h > height - 8:
                y = int(anchor[1] - panel_h - 24)
            toggle_panel_w = 72 + 42 * 3 + 12
            toggle_panel_h = 68
            toggle_x = 10
            toggle_y = max(72, height - toggle_panel_h - 12)
            overlaps_toggle = (
                x < toggle_x + toggle_panel_w + 8
                and x + panel_w > toggle_x - 8
                and y < toggle_y + toggle_panel_h + 8
                and y + panel_h > toggle_y - 8
            )
            if overlaps_toggle:
                y = max(52, toggle_y - panel_h - 10)
            self._headlight_panel_anchor_index = index
        else:
            x = int(panel_pos[0])
            y = int(panel_pos[1])
        x = max(6, min(x, max(6, width - panel_w - 6)))
        y = max(52, min(y, max(52, height - panel_h - 10)))
        self._headlight_panel_pos = (x, y)
        strength_var, red_var, green_var, blue_var, cone_var, source_radius_var = self._headlight_control_vars(index)
        enabled = self._headlight_effect_enabled(index)
        color = self._headlight_color_from_vars(index, float(strength_var.get() or 0.0) if enabled else 0.0)
        panel_outline = "#ffd36a"
        panel_title = "#fff3b0"
        canvas.create_rectangle(x, y, x + panel_w, y + panel_h, fill="#10161a", outline=panel_outline, width=1, tags=("aug_overlay",))
        self._register_canvas_overlay_region("headlight_panel", (x, y, x + panel_w, y + panel_h), index=index)
        canvas.create_rectangle(x + 7, y + 4, x + 130, y + 24, fill="#172027", outline="#584b25", width=1, tags=("aug_overlay",))
        canvas.create_text(x + 12, y + 14, text=f"Reflektor R{index} {'ON' if enabled else 'OFF'}", anchor=tk.W, fill=panel_title, font=("Segoe UI", 9, "bold"), tags=("aug_overlay",))
        for yy in (y + 9, y + 14, y + 19):
            canvas.create_line(x + panel_w - 34, yy, x + panel_w - 14, yy, fill="#9aa7ad", width=1, tags=("aug_overlay",))
        self._register_canvas_overlay_region("headlight_panel_drag", (x, y, x + panel_w, y + 27), index=index)
        slider_x = x + 10
        slider_w = panel_w - 20
        rows = (
            ("Nat.", strength_var, 0.0, 1.0, 0.01),
            ("R", red_var, 0.0, 1.0, 0.01),
            ("G", green_var, 0.0, 1.0, 0.01),
            ("B", blue_var, 0.0, 1.0, 0.01),
            ("Prom. końca", cone_var, 0.02, 2.5, 0.01),
            ("Prom. startu", source_radius_var, 0.0, 2.5, 0.01),
        )
        row_y = y + 38
        for label, variable, from_, to, step in rows:
            role = str(label).strip().casefold() if str(label).strip().casefold() in {"r", "g", "b"} else None
            self._draw_canvas_slider(canvas, slider_x, row_y, slider_w, label, variable, from_, to, step, role=role)
            row_y += 24

    def _draw_arrow(self, canvas: tk.Canvas, start: tuple[int, int], end: tuple[int, int], color: str, dash=None):
        canvas.create_line(
            start[0],
            start[1],
            end[0],
            end[1],
            fill=color,
            width=2,
            arrow=tk.LAST,
            arrowshape=(10, 12, 4),
            dash=dash,
            tags=("vector_overlay",),
        )
        canvas.create_oval(
            start[0] - 3,
            start[1] - 3,
            start[0] + 3,
            start[1] + 3,
            fill=color,
            outline="",
            tags=("vector_overlay",),
        )

    def _default_vector_position(self, tool: str, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        value = 0.0
        angle = 0.0
        try:
            if tool == "wind":
                value = float(self.dirt_flow_wind_strength_var.get() or 0.0)
                angle_deg = float(self.dirt_flow_air_angle_var.get() or 0.0)
                if self._vector_tool == tool and abs(value) < 0.001 and abs(angle_deg) < 0.001:
                    angle_deg = 28.0
                angle = math.radians(angle_deg)
                direction = (math.sin(angle), math.cos(angle))
                mapping = {}
                try:
                    canvas = getattr(self, "augmented_canvas", None)
                    if canvas is not None:
                        mapping = self._preview_canvas_mappings.get(id(canvas), {}) or {}
                except Exception:
                    mapping = {}
                image_left = float(mapping.get("display_left", 0.0) or 0.0)
                image_top = float(mapping.get("display_top", 0.0) or 0.0)
                image_w = float(mapping.get("display_w", width) or width)
                image_h = float(mapping.get("display_h", height) or height)
                start = (
                    int(round(max(72.0, min(width - 92.0, image_left + image_w * 0.13)))),
                    int(round(max(72.0, min(height - 92.0, image_top + image_h * 0.17)))),
                )
            else:
                return (0, 0), (0, 0)
            display_value = max(value, 0.32) if self._vector_tool == tool else value
            length = self._vector_canvas_scale(width, height) * display_value
            end = (
                int(round(start[0] + direction[0] * length)),
                int(round(start[1] + direction[1] * length)),
            )
            return start, end
        except Exception:
            return (0, 0), (0, 0)

    def _vector_canvas_scale(self, width: int | None = None, height: int | None = None) -> float:
        canvas = getattr(self, "augmented_canvas", None)
        width = int(width if width is not None else (canvas.winfo_width() if canvas is not None else 320))
        height = int(height if height is not None else (canvas.winfo_height() if canvas is not None else 240))
        return max(60.0, min(180.0, min(width, height) * 0.34))

    def _canvas_overlay_hit(self, event) -> dict | None:
        x = float(getattr(event, "x", 0) or 0)
        y = float(getattr(event, "y", 0) or 0)
        hits: list[dict] = []
        for region in reversed(list(getattr(self, "_canvas_overlay_regions", []))):
            try:
                x1, y1, x2, y2 = region.get("rect", (0, 0, 0, 0))
                if x1 <= x <= x2 and y1 <= y <= y2:
                    hits.append(region)
            except Exception:
                continue
        if not hits:
            return None
        for region in hits:
            if region.get("kind") in {"scene_viewport_mode", "scene_plate_texture_toggle"}:
                return region
        first = hits[0]
        if first.get("kind") != "headlight":
            return first
        active_index = int(getattr(self, "_active_headlight_index", 1) or 1)
        for region in hits:
            if region.get("kind") == "headlight" and int(region.get("index", 0) or 0) == active_index:
                return region
        return first

    def _scroll_canvas_toolbox_panel_at_point(self, x: float, y: float, steps: float) -> bool:
        if not (
            bool(getattr(self, "_preview_fullscreen", False))
            and bool(getattr(self, "_toolbox_panel_visible", False))
            and float(getattr(self, "_toolbox_panel_max_scroll", 0.0) or 0.0) > 0.5
        ):
            return False
        try:
            x1, y1, x2, y2 = getattr(self, "_toolbox_panel_rect", (0, 0, 0, 0))
            if not (float(x1) <= float(x) <= float(x2) and float(y1) <= float(y) <= float(y2)):
                return False
            max_scroll = max(0.0, float(getattr(self, "_toolbox_panel_max_scroll", 0.0) or 0.0))
            current = max(0.0, min(max_scroll, float(getattr(self, "_toolbox_panel_scroll", 0.0) or 0.0)))
            next_scroll = max(0.0, min(max_scroll, current - float(steps) * 58.0))
            if abs(next_scroll - current) <= 0.1:
                return True
            self._toolbox_panel_scroll = next_scroll
            self._draw_vector_overlay()
            return True
        except Exception:
            return False

    def _set_canvas_slider_value(self, region: dict, x: float):
        variable = region.get("variable")
        if variable is None:
            return
        try:
            bar_x1, bar_x2 = region.get("bar", (0.0, 1.0))
            ratio = max(0.0, min(1.0, (float(x) - float(bar_x1)) / max(1.0, float(bar_x2) - float(bar_x1))))
            from_ = float(region.get("from_", 0.0))
            to = float(region.get("to", 1.0))
            step = float(region.get("step", 0.01) or 0.01)
            value = from_ + (to - from_) * ratio
            if step > 0:
                value = round(value / step) * step
            if step >= 1:
                value = int(round(value))
            variable.set(value)
        except Exception:
            pass

    def _set_canvas_range_slider_value(self, region: dict, x: float, handle: str | None = None):
        min_var = region.get("min_var")
        max_var = region.get("max_var")
        if min_var is None or max_var is None:
            return
        try:
            bar_x1, bar_x2 = region.get("bar", (0.0, 1.0))
            ratio = max(0.0, min(1.0, (float(x) - float(bar_x1)) / max(1.0, float(bar_x2) - float(bar_x1))))
            from_ = float(region.get("from_", 0.0))
            to = float(region.get("to", 1.0))
            step = float(region.get("step", 0.01) or 0.01)
            value = from_ + (to - from_) * ratio
            if step > 0:
                value = round(value / step) * step
            if step >= 1:
                value = int(round(value))
            if handle not in ("min", "max"):
                min_x = float(region.get("min_x", bar_x1))
                max_x = float(region.get("max_x", bar_x2))
                if abs(min_x - max_x) <= 2.0:
                    handle = "max" if float(x) >= max_x else "min"
                else:
                    handle = "min" if abs(float(x) - min_x) <= abs(float(x) - max_x) else "max"
            current_min = float(min_var.get())
            current_max = float(max_var.get())
            if abs(current_min - current_max) <= max(1e-9, step * 0.5):
                if value > current_max:
                    handle = "max"
                elif value < current_min:
                    handle = "min"
            if handle == "min":
                min_var.set(min(value, current_max))
                region["handle"] = "min"
            else:
                max_var.set(max(value, current_min))
                region["handle"] = "max"
            self._coerce_range_pair(min_var, max_var, from_, to, str(region.get("handle") or "max"))
        except Exception:
            pass

    def _default_preview_zoom(self) -> float:
        # Keep a working margin around the plate so HUD tools and headlights do
        # not cover the most important image area right after opening the modal.
        return 0.58

    def _toggle_preview_zoom(self):
        current = float(getattr(self, "_preview_zoom", self._default_preview_zoom()) or self._default_preview_zoom())
        steps = tuple(sorted({0.35, 0.5, round(self._default_preview_zoom(), 2), 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}))
        next_value = steps[0]
        for value in steps:
            if current < value - 0.01:
                next_value = value
                break
        else:
            next_value = steps[0]
        self._preview_zoom = next_value
        if next_value <= 1.01:
            self._preview_pan_x = 0.0
            self._preview_pan_y = 0.0
        self._refresh_preview(redraw_only=True)

    def _preview_wheel_canvas_and_point(self, event):
        candidates = [
            getattr(self, "augmented_canvas", None),
            getattr(self, "original_canvas", None),
        ]
        mappings = getattr(self, "_preview_canvas_mappings", {}) or {}
        widget = getattr(event, "widget", None)
        if widget in candidates and mappings.get(id(widget)):
            return widget, float(getattr(event, "x", 0.0) or 0.0), float(getattr(event, "y", 0.0) or 0.0)

        try:
            root_x = float(getattr(event, "x_root"))
            root_y = float(getattr(event, "y_root"))
        except Exception:
            root_x = root_y = None

        if root_x is not None and root_y is not None:
            for canvas in candidates:
                if canvas is None or not mappings.get(id(canvas)):
                    continue
                try:
                    left = float(canvas.winfo_rootx())
                    top = float(canvas.winfo_rooty())
                    width = float(canvas.winfo_width() or 0)
                    height = float(canvas.winfo_height() or 0)
                    if left <= root_x <= left + width and top <= root_y <= top + height:
                        return canvas, root_x - left, root_y - top
                except Exception:
                    continue

        canvas = getattr(self, "augmented_canvas", None)
        if canvas is not None:
            if root_x is not None and root_y is not None:
                try:
                    return canvas, root_x - float(canvas.winfo_rootx()), root_y - float(canvas.winfo_rooty())
                except Exception:
                    pass
            return canvas, float(getattr(event, "x", 0.0) or 0.0), float(getattr(event, "y", 0.0) or 0.0)
        return widget, float(getattr(event, "x", 0.0) or 0.0), float(getattr(event, "y", 0.0) or 0.0)

    def _on_preview_mousewheel(self, event):
        try:
            current = max(0.25, min(6.0, float(getattr(self, "_preview_zoom", self._default_preview_zoom()) or self._default_preview_zoom())))
            canvas, event_x, event_y = self._preview_wheel_canvas_and_point(event)
            try:
                canvas_w = float(canvas.winfo_width() if canvas is not None else 1.0)
                canvas_h = float(canvas.winfo_height() if canvas is not None else 1.0)
            except Exception:
                canvas_w = 1.0
                canvas_h = 1.0
            anchor_norm = self._image_norm_from_canvas_point_for_canvas(
                canvas,
                event_x,
                event_y,
                canvas_w,
                canvas_h,
            )
            if getattr(event, "num", None) == 4:
                steps = 1.0
            elif getattr(event, "num", None) == 5:
                steps = -1.0
            else:
                steps = float(getattr(event, "delta", 0) or 0) / 120.0
            if abs(steps) <= 0.001:
                return None
            if (
                canvas is getattr(self, "augmented_canvas", None)
                and self._scroll_canvas_toolbox_panel_at_point(event_x, event_y, steps)
            ):
                return "break"
            scene_visible = bool(getattr(self, "_scene_viewport_visible", False)) and canvas is getattr(self, "augmented_canvas", None)
            scene_current = max(0.20, min(8.0, float(getattr(self, "_scene_view_zoom", 1.0) or 1.0)))
            base_value = scene_current if scene_visible else current
            next_value = max(0.20, min(8.0, base_value * (1.10 ** steps))) if scene_visible else max(0.25, min(6.0, current * (1.10 ** steps)))
            if abs(next_value - base_value) > 0.001:
                if scene_visible:
                    _rect, body_rect = self._scene_viewport_rects(int(canvas_w), int(canvas_h))
                    body_center_x = (float(body_rect[0]) + float(body_rect[2])) / 2.0
                    body_center_y = (float(body_rect[1]) + float(body_rect[3])) / 2.0
                    old_center_x = body_center_x + float(getattr(self, "_scene_view_pan_x", 0.0) or 0.0)
                    old_center_y = body_center_y + float(getattr(self, "_scene_view_pan_y", 0.0) or 0.0)
                    ratio = max(0.01, float(next_value) / max(0.01, float(scene_current)))
                    new_center_x = float(event_x) - (float(event_x) - old_center_x) * ratio
                    new_center_y = float(event_y) - (float(event_y) - old_center_y) * ratio
                    self._scene_view_zoom = next_value
                    self._scene_view_pan_x = new_center_x - body_center_x
                    self._scene_view_pan_y = new_center_y - body_center_y
                    self._draw_vector_overlay()
                    return "break"
                self._preview_zoom = next_value
                anchored = self._set_preview_pan_for_canvas_anchor(
                    canvas,
                    anchor_norm[0],
                    anchor_norm[1],
                    event_x,
                    event_y,
                    next_value,
                )
                if not anchored and next_value <= 1.01:
                    self._preview_pan_x = 0.0
                    self._preview_pan_y = 0.0
                self._refresh_preview(redraw_only=True)
            return "break"
        except Exception:
            return None

    def _toggle_preview_fullscreen(self):
        self._preview_fullscreen = not bool(getattr(self, "_preview_fullscreen", False))
        self._apply_preview_fullscreen_layout(bool(self._preview_fullscreen))
        try:
            self.window.attributes("-fullscreen", bool(self._preview_fullscreen))
        except Exception:
            try:
                self.window.state("zoomed" if self._preview_fullscreen else "normal")
            except Exception:
                pass
        if self._preview_fullscreen:
            try:
                self.augmented_canvas.focus_set()
            except Exception:
                pass
        self._refresh_preview(redraw_only=True)

    def _apply_preview_fullscreen_layout(self, active: bool):
        try:
            if active:
                snapshot: dict[str, dict] = {}
                for widget_name in ("header_frame", "controls_host", "preview_toolbar", "footer_frame", "left_preview_frame", "effect_inspector_shell"):
                    widget = getattr(self, widget_name, None)
                    if widget is not None:
                        try:
                            snapshot[widget_name] = dict(widget.grid_info() or {})
                        except Exception:
                            pass
                        widget.grid_remove()
                right = getattr(self, "right_preview_frame", None)
                if right is not None:
                    try:
                        snapshot["right_preview_frame"] = dict(right.grid_info() or {})
                    except Exception:
                        pass
                    right.grid_configure(row=0, column=0, columnspan=2, sticky=tk.NSEW, padx=0)
                self._preview_fullscreen_grid_snapshot = snapshot
            else:
                snapshot = dict(getattr(self, "_preview_fullscreen_grid_snapshot", {}) or {})

                def _restore_grid(widget_name: str, fallback: dict | None = None) -> None:
                    widget = getattr(self, widget_name, None)
                    if widget is None:
                        return
                    info = dict(snapshot.get(widget_name) or fallback or {})
                    try:
                        if info:
                            widget.grid(**info)
                        else:
                            widget.grid()
                    except Exception:
                        try:
                            widget.grid()
                        except Exception:
                            pass

                _restore_grid("header_frame")
                if not bool(getattr(self, "_effects_controls_panel_hidden", False)):
                    _restore_grid("controls_host")
                _restore_grid("preview_toolbar")
                _restore_grid("footer_frame")
                _restore_grid("left_preview_frame", {"row": 0, "column": 0, "sticky": tk.NSEW, "padx": (0, 5)})
                if bool(getattr(self, "_effect_inspector_visible", True)):
                    _restore_grid("effect_inspector_shell", {"row": 0, "column": 1, "sticky": tk.NSEW, "padx": (10, 0)})
                _restore_grid("right_preview_frame", {"row": 0, "column": 0, "sticky": tk.NSEW})
                self._preview_fullscreen_grid_snapshot = {}
        except Exception:
            pass

    def _on_escape_key(self, _event=None):
        if bool(getattr(self, "_preview_fullscreen", False)):
            self._preview_fullscreen = False
            self._apply_preview_fullscreen_layout(False)
            try:
                self.window.attributes("-fullscreen", False)
            except Exception:
                try:
                    self.window.state("normal")
                except Exception:
                    pass
            self._refresh_preview(redraw_only=True)
            return "break"
        return None

    def _on_tab_key(self, _event=None):
        widget = getattr(_event, "widget", None)
        try:
            widget_class = str(widget.winfo_class() or "")
            if widget_class in {"Entry", "TEntry", "Text", "Spinbox", "TSpinbox"}:
                return None
        except Exception:
            pass
        self._fullscreen_show_original = not bool(getattr(self, "_fullscreen_show_original", False))
        try:
            self._scene_plate_texture_cache_key = None
            self._scene_plate_texture_photo_ref = None
            self._scene_plate_texture_quad_cache_key = None
            self._scene_plate_texture_quad_photo_ref = None
        except Exception:
            pass
        try:
            self.augmented_canvas.focus_set()
        except Exception:
            pass
        self._refresh_preview(redraw_only=True)
        return "break"

    def _on_enter_key(self, event=None):
        widget = getattr(event, "widget", None)
        try:
            widget_class = str(widget.winfo_class() or "")
            if widget_class in {"Entry", "TEntry", "Text", "Spinbox", "TSpinbox"}:
                return None
        except Exception:
            pass
        self._toggle_preview_fullscreen()
        return "break"

    def _on_space_key(self, event=None):
        widget = getattr(event, "widget", None)
        try:
            widget_class = str(widget.winfo_class() or "")
            if widget_class in {"Entry", "TEntry", "Text", "Spinbox", "TSpinbox"}:
                return None
        except Exception:
            pass
        current = getattr(self, "_active_headlight_index", None)
        if current not in (1, 2, 3):
            next_index = 1
        elif current == 1:
            next_index = 2
        elif current == 2:
            next_index = 3
        else:
            next_index = None
        self._select_headlight(next_index)
        self._cancel_pending_preview()
        self._draw_vector_overlay()
        return "break"

    def _on_scene_keyboard_nudge(self, event=None):
        widget = getattr(event, "widget", None)
        try:
            widget_class = str(widget.winfo_class() or "")
            if widget_class in {"Entry", "TEntry", "Text", "Spinbox", "TSpinbox"}:
                return None
        except Exception:
            pass
        keysym = str(getattr(event, "keysym", "") or "")
        state = int(getattr(event, "state", 0) or 0)
        shift = bool(state & 0x0001)
        control = bool(state & 0x0004)
        if keysym not in {"Left", "Right", "Up", "Down", "Prior", "Next"}:
            return None

        delta = 0.02
        if control and shift and keysym in {"Left", "Right", "Up", "Down"}:
            return None
        elif control:
            if keysym in {"Left", "Right"}:
                return None
            variables = (self.scene_camera_z_var,)
            ranges = ((0.35, 4.0),)
            axis = 0
            delta = 0.05 if keysym in {"Up", "Prior"} else -0.05
        else:
            index = self._current_headlight_inspector_index()
            if keysym in {"Prior", "Next"}:
                variables = self._headlight_world_var_triplet(index, "source")
                ranges = ((-3.0, 3.0), (-3.0, 3.0), (0.0, 8.0))
                axis = 2
            elif shift:
                variables = self._headlight_world_var_triplet(index, "source")
                ranges = ((-3.0, 3.0), (-3.0, 3.0), (0.0, 8.0))
                axis = 1 if keysym in {"Up", "Down"} else 0
            else:
                variables = self._headlight_world_var_triplet(index, "target")
                ranges = ((-0.5, 0.5), (-0.12, 0.12), (0.0, 0.0))
                axis = 1 if keysym in {"Up", "Down"} else 0
            if keysym in {"Left", "Down", "Next"}:
                delta = -delta

        try:
            variable = variables[axis]
            min_value, max_value = ranges[axis]
            current = self._read_float_var(variable, 0.0)
            variable.set(max(float(min_value), min(float(max_value), current + delta)))
            self._enforce_scene_invariants()
            self._refresh_scene_summary()
            self._draw_vector_overlay()
            self._schedule_preview_refresh()
            return "break"
        except Exception:
            return None

    def _event_has_control(self, event) -> bool:
        try:
            return bool(int(getattr(event, "state", 0) or 0) & 0x0004)
        except Exception:
            return False

    def _event_has_shift(self, event) -> bool:
        try:
            return bool(int(getattr(event, "state", 0) or 0) & 0x0001)
        except Exception:
            return False

    def _scene_drag_event_coords(self, event) -> tuple[int, int]:
        try:
            x_root = getattr(event, "x_root", None)
            y_root = getattr(event, "y_root", None)
            if x_root is not None and y_root is not None:
                return int(x_root), int(y_root)
        except Exception:
            pass
        return int(getattr(event, "x", 0) or 0), int(getattr(event, "y", 0) or 0)

    def _scene_drag_view_center_coords(self, event) -> tuple[float, float]:
        canvas = getattr(self, "augmented_canvas", None) or getattr(event, "widget", None)
        try:
            width = max(1, int(canvas.winfo_width() or 1))
            height = max(1, int(canvas.winfo_height() or 1))
            _rect, body_rect = self._scene_viewport_rects(width, height)
            center_x = (float(body_rect[0]) + float(body_rect[2])) / 2.0
            center_y = (float(body_rect[1]) + float(body_rect[3])) / 2.0
            if getattr(event, "x_root", None) is not None and getattr(event, "y_root", None) is not None:
                center_x += float(canvas.winfo_rootx())
                center_y += float(canvas.winfo_rooty())
            return center_x, center_y
        except Exception:
            return self._scene_drag_event_coords(event)

    def _scene_view_rotation_available(self) -> bool:
        return True

    def _wind_panel_event_point(self, region: dict, event) -> tuple[float, float]:
        x = float(getattr(event, "x", 0) or 0)
        y = float(getattr(event, "y", 0) or 0)
        try:
            px1, py1, px2, py2 = [float(value) for value in region.get("plot", (0, 0, 1, 1))]
            x = max(px1, min(px2, x))
            y = max(py1, min(py2, y))
        except Exception:
            pass
        return self._wind_panel_constrain_point(region, (x, y))

    def _wind_panel_constrain_point(self, region: dict, point: tuple[float, float]) -> tuple[float, float]:
        try:
            start_x, start_y = [float(value) for value in region.get("start", (0.0, 0.0))]
            scale = max(1.0, float(region.get("scale", 60.0) or 60.0))
            dx = float(point[0]) - start_x
            dy = start_y - float(point[1])
            length = min(scale, math.hypot(dx, dy))
            if length < 0.001:
                return start_x, start_y
            angle = math.degrees(math.atan2(dy, dx))
            radians = math.radians(angle)
            return start_x + math.cos(radians) * length, start_y - math.sin(radians) * length
        except Exception:
            return point

    def _is_wind_panel_overlay(self, region: dict | None) -> bool:
        if not isinstance(region, dict):
            return False
        return str(region.get("kind") or "") in {"wind_panel_guard", "wind_strength_field", "wind_vector_panel"}

    def _begin_wind_panel_drag(self, region: dict, event) -> None:
        start = region.get("start", (0.0, 0.0))
        point = self._wind_panel_event_point(region, event)
        self._vector_tool = "wind"
        self._wind_panel_drag = {
            "start": (float(start[0]), float(start[1])),
            "end": point,
            "plot": region.get("plot", (0.0, 0.0, 1.0, 1.0)),
            "scale": float(region.get("scale", 60.0) or 60.0),
        }
        self._draw_vector_overlay()

    def _update_wind_panel_drag(self, event) -> None:
        drag = getattr(self, "_wind_panel_drag", None)
        if not isinstance(drag, dict):
            return
        self._wind_panel_drag["end"] = self._wind_panel_event_point(drag, event)

    def _finish_wind_panel_drag(self, event) -> None:
        drag = getattr(self, "_wind_panel_drag", None)
        if not isinstance(drag, dict):
            return
        self._update_wind_panel_drag(event)
        start = drag.get("start", (0.0, 0.0))
        end = drag.get("end", start)
        dx = float(end[0]) - float(start[0])
        dy = float(start[1]) - float(end[1])
        length = math.hypot(dx, dy)
        scale = max(1.0, float(drag.get("scale", 60.0) or 60.0))
        if length < 4.0:
            value = 0.0
            angle = 0.0
        else:
            value = max(0.0, min(1.0, length / scale))
            angle = math.degrees(math.atan2(dy, dx))
        try:
            self.dirt_flow_wind_strength_var.set(value)
            self.dirt_flow_air_angle_var.set(angle)
        except Exception:
            pass
        self._wind_panel_drag = None
        self._draw_vector_overlay()
        self._schedule_preview_refresh(delay_ms=40)

    def _deactivate_wind_tool_ui(self, *, commit_editor: bool = True) -> None:
        self._wind_panel_drag = None
        self._close_wind_strength_editor(commit=commit_editor)
        if self._vector_tool == "wind":
            self._vector_tool = None

    def _open_wind_strength_editor(self, region: dict) -> None:
        canvas = getattr(self, "augmented_canvas", None)
        if canvas is None:
            return
        self._close_wind_strength_editor(commit=False)
        try:
            x1, y1, x2, y2 = [float(value) for value in region.get("rect", region.get("rect_", (0, 0, 46, 21)))]
        except Exception:
            x1, y1, x2, y2 = 0.0, 0.0, 46.0, 21.0
        try:
            value = max(0.0, min(1.0, float(self.dirt_flow_wind_strength_var.get() or 0.0)))
        except Exception:
            value = 0.0
        var = tk.StringVar(value=f"{value:.1f}")
        entry = tk.Entry(
            canvas,
            textvariable=var,
            justify=tk.CENTER,
            font=("Segoe UI", 8, "bold"),
            bg="#10161a",
            fg="#bfefff",
            insertbackground="#bfefff",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground="#76d9ff",
            highlightcolor="#76d9ff",
        )
        self._wind_strength_editor = entry
        self._wind_strength_editor_var = var
        entry.place(x=int(x1) + 1, y=int(y1) + 1, width=max(24, int(x2 - x1) - 2), height=max(16, int(y2 - y1) - 2))
        try:
            window_tag = str(self.window)
            entry.bindtags(tuple(tag for tag in entry.bindtags() if str(tag) != window_tag))
        except Exception:
            pass
        entry.bind("<Return>", lambda _event: self._close_wind_strength_editor(commit=True), add="+")
        entry.bind("<KP_Enter>", lambda _event: self._close_wind_strength_editor(commit=True), add="+")
        entry.bind("<Escape>", lambda _event: self._close_wind_strength_editor(commit=False), add="+")
        entry.bind("<Button-3>", lambda _event: "break", add="+")
        entry.bind("<B3-Motion>", lambda _event: "break", add="+")
        entry.bind("<ButtonRelease-3>", lambda _event: "break", add="+")
        entry.bind("<FocusOut>", lambda _event: self._close_wind_strength_editor(commit=True), add="+")
        try:
            entry.focus_set()
            entry.selection_range(0, tk.END)
        except Exception:
            pass

    def _close_wind_strength_editor(self, *, commit: bool) -> str:
        entry = getattr(self, "_wind_strength_editor", None)
        var = getattr(self, "_wind_strength_editor_var", None)
        if entry is None:
            return "break"
        self._wind_strength_editor = None
        self._wind_strength_editor_var = None
        if commit and var is not None:
            try:
                raw = str(var.get() or "0").replace(",", ".").strip()
                value = round(max(0.0, min(1.0, float(raw))), 1)
                self.dirt_flow_wind_strength_var.set(value)
            except Exception:
                pass
        try:
            entry.destroy()
        except Exception:
            pass
        self._draw_vector_overlay()
        if commit:
            self._schedule_preview_refresh(delay_ms=40)
        return "break"

    def _set_scene_view_value(self, variable, value: float, low: float, high: float) -> None:
        try:
            variable.set(max(float(low), min(float(high), float(value))))
        except Exception:
            pass

    def _begin_scene_view_drag(self, event, mode: str | None = None) -> None:
        current_mode = self._active_scene_viewport_mode()
        requested_mode = self._scene_view_mode_from_projection(mode or current_mode) or current_mode
        if requested_mode != current_mode:
            self._scene_viewport_mode = requested_mode
        mode = requested_mode
        self._scene_projection_active = mode
        self._active_toolbox = "illumination"
        start_x, start_y = self._scene_drag_event_coords(event)
        center_x, center_y = self._scene_drag_view_center_coords(event)
        self._scene_view_drag = {
            "start_x": start_x,
            "start_y": start_y,
            "center_x": center_x,
            "center_y": center_y,
            "base_matrix": self._scene_view_matrix(),
            "base_yaw": self._read_float_var(self.scene_view_yaw_var, 0.0),
            "base_pitch": self._read_float_var(self.scene_view_pitch_var, 89.0),
            "base_roll": self._read_float_var(self.scene_view_roll_var, 0.0),
            "roll": self._event_has_shift(event),
            "primed": False,
        }
        self._grab_augmented_canvas_for_live_edit()
        self._draw_vector_overlay()

    def _begin_scene_pan_drag(self, event) -> None:
        self._active_toolbox = "illumination"
        start_x, start_y = self._scene_drag_event_coords(event)
        self._scene_pan_drag = {
            "start_x": start_x,
            "start_y": start_y,
            "base_x": float(getattr(self, "_scene_view_pan_x", 0.0) or 0.0),
            "base_y": float(getattr(self, "_scene_view_pan_y", 0.0) or 0.0),
            "primed": False,
        }
        self._grab_augmented_canvas_for_live_edit()
        self._draw_vector_overlay()

    def _update_scene_pan_drag(self, event) -> None:
        drag = getattr(self, "_scene_pan_drag", None)
        if not isinstance(drag, dict):
            return
        current_x, current_y = self._scene_drag_event_coords(event)
        if not bool(drag.get("primed", False)):
            drag["start_x"] = current_x
            drag["start_y"] = current_y
            drag["primed"] = True
            return
        dx = current_x - int(drag.get("start_x", 0) or 0)
        dy = current_y - int(drag.get("start_y", 0) or 0)
        if abs(dx) <= 1 and abs(dy) <= 1:
            return
        self._scene_view_pan_x = float(drag.get("base_x", 0.0) or 0.0) + dx
        self._scene_view_pan_y = float(drag.get("base_y", 0.0) or 0.0) + dy

    def _end_scene_pan_drag(self) -> None:
        self._scene_pan_drag = None
        self._release_augmented_canvas_live_grab()

    def _update_scene_view_drag(self, event) -> None:
        drag = getattr(self, "_scene_view_drag", None)
        if not isinstance(drag, dict):
            return
        current_x, current_y = self._scene_drag_event_coords(event)
        if not bool(drag.get("primed", False)):
            drag["start_x"] = current_x
            drag["start_y"] = current_y
            drag["primed"] = True
            return
        dx = current_x - int(drag.get("start_x", 0) or 0)
        dy = current_y - int(drag.get("start_y", 0) or 0)
        if abs(dx) <= 1 and abs(dy) <= 1:
            return
        def _drag_float(name: str, default: float) -> float:
            try:
                value = drag.get(name, default)
                if value is None:
                    value = default
                return float(value)
            except Exception:
                return float(default)

        base_matrix = drag.get("base_matrix")
        if not (isinstance(base_matrix, tuple) and len(base_matrix) == 3):
            base_matrix = self._scene_view_matrix_from_angles(
                _drag_float("base_yaw", 0.0),
                _drag_float("base_pitch", 89.0),
                _drag_float("base_roll", 0.0),
            )

        if bool(drag.get("roll")):
            delta_angle = dx * 0.42
            roll_delta = self._scene_axis_rotation_matrix("y", delta_angle)
            next_matrix = self._scene_matrix_multiply(roll_delta, base_matrix)
            self._set_scene_view_value(self.scene_view_roll_var, _drag_float("base_roll", 0.0) + delta_angle, -180.0, 180.0)
            self._set_scene_view_rotation_matrix(next_matrix)
        else:
            yaw_delta = dx * 0.32
            pitch_delta = -dy * 0.24
            yaw_matrix = self._scene_axis_rotation_matrix("z", yaw_delta)
            pitch_matrix = self._scene_axis_rotation_matrix("x", pitch_delta)
            screen_delta = self._scene_matrix_multiply(pitch_matrix, yaw_matrix)
            next_matrix = self._scene_matrix_multiply(screen_delta, base_matrix)
            self._set_scene_view_value(self.scene_view_yaw_var, _drag_float("base_yaw", 0.0) + yaw_delta, -180.0, 180.0)
            self._set_scene_view_value(self.scene_view_pitch_var, _drag_float("base_pitch", 89.0) + pitch_delta, -89.0, 89.0)
            self._set_scene_view_rotation_matrix(next_matrix)
        self._enforce_scene_invariants()
        self._refresh_scene_summary()

    def _end_scene_view_drag(self) -> None:
        self._scene_view_drag = None
        self._release_augmented_canvas_live_grab()

    def _handle_canvas_overlay_press(self, event) -> bool:
        region = self._canvas_overlay_hit(event)
        if not region:
            return False
        kind = region.get("kind")
        if kind == "wind_panel_guard":
            return True
        if kind == "vector_tool":
            tool = str(region.get("tool") or "")
            if not tool:
                return True
            self._toolbox_panel_visible = False
            if self._vector_tool == tool:
                if tool == "wind":
                    self._deactivate_wind_tool_ui()
                else:
                    self._vector_tool = None
            else:
                if tool != "wind":
                    self._deactivate_wind_tool_ui()
                self._vector_tool = tool
            self._vector_drag_start = None
            self._vector_drag_end = None
            self._draw_vector_overlay()
            return True
        if kind == "toolbox":
            self._deactivate_wind_tool_ui()
            self._select_toolbox(str(region.get("key") or self._active_toolbox))
            return True
        if kind == "action":
            action = str(region.get("action") or "")
            if action == "zoom":
                self._toggle_preview_zoom()
            elif action == "fullscreen":
                self._toggle_preview_fullscreen()
            elif action == "dice":
                self._pick_random_sample()
            elif action == "layout":
                self._reroll_effect_layout()
            elif action == "reset":
                self._reset_all_parameters()
            return True
        if kind == "check":
            variable = region.get("variable")
            try:
                variable.set(not bool(variable.get()))
            except Exception:
                pass
            self._refresh_preview(redraw_only=True)
            return True
        if kind == "slider":
            self._begin_canvas_live_edit()
            self._canvas_slider_drag = region
            self._grab_augmented_canvas_for_live_edit()
            self._set_canvas_slider_value(region, getattr(event, "x", 0))
            self._draw_vector_overlay()
            return True
        if kind == "range_slider":
            self._begin_canvas_live_edit()
            region["handle"] = None
            self._canvas_slider_drag = region
            self._grab_augmented_canvas_for_live_edit()
            self._set_canvas_range_slider_value(region, getattr(event, "x", 0))
            self._draw_vector_overlay()
            return True
        if kind == "panel_drag":
            self._toolbox_panel_drag = {
                "start_x": int(getattr(event, "x", 0)),
                "start_y": int(getattr(event, "y", 0)),
                "base": tuple(getattr(self, "_toolbox_panel_pos", None) or (10, 78)),
            }
            return True
        if kind == "panel_reset":
            self._toolbox_panel_pos = None
            self._toolbox_panel_collapsed = False
            self._draw_vector_overlay()
            return True
        if kind == "panel_toggle":
            self._toolbox_panel_collapsed = not bool(getattr(self, "_toolbox_panel_collapsed", False))
            self._draw_vector_overlay()
            return True
        if kind == "scene_viewport_mode":
            mode = str(region.get("mode", "3d") or "3d").lower()
            valid = {key for key, _label in self._scene_view_modes()}
            if mode not in valid:
                mode = "xy"
            self._toolbox_panel_visible = False
            self._scene_viewport_mode = mode
            self._scene_projection_active = mode
            self._scene_viewport_visible = True
            self._apply_scene_view_preset(mode)
            self._draw_vector_overlay()
            return True
        if kind == "wind_strength_field":
            self._open_wind_strength_editor(region)
            return True
        if kind == "wind_vector_panel":
            self._begin_wind_panel_drag(region, event)
            return True
        if kind == "scene_viewport_panel":
            mode = self._scene_view_mode_from_projection(region.get("projection", "")) or self._active_scene_viewport_mode()
            if self._event_has_control(event):
                self._begin_scene_view_drag(event, mode=mode)
            else:
                self._begin_scene_pan_drag(event)
            return True
        if kind == "scene_plate_texture_toggle":
            try:
                self.scene_plate_texture_var.set(not bool(self.scene_plate_texture_var.get()))
                self._scene_plate_texture_cache_key = None
                self._scene_plate_texture_photo_ref = None
                self._scene_plate_texture_quad_cache_key = None
                self._scene_plate_texture_quad_photo_ref = None
            except Exception:
                pass
            self._active_toolbox = "illumination"
            self._draw_vector_overlay()
            return True
        if kind == "scene_projection_panel":
            projection = str(region.get("projection", "xy") or "xy").lower()
            self._toolbox_panel_visible = False
            if self._event_has_control(event):
                drag_mode = self._scene_view_mode_from_projection(projection)
                self._begin_scene_view_drag(event, mode=drag_mode)
                return True
            if projection in {"3d", "xy", "xz", "yz"}:
                self._scene_projection_active = projection
                view_mode = self._scene_view_mode_from_projection(projection)
                if view_mode is not None:
                    previous_mode = self._active_scene_viewport_mode()
                    self._scene_viewport_mode = view_mode
                    self._scene_viewport_visible = True
                    if view_mode != previous_mode:
                        self._apply_scene_view_preset(view_mode)
                self._draw_vector_overlay()
            return True
        if kind == "scene_projection_toggle":
            projection = str(region.get("projection", "xy") or "xy").lower()
            if projection in {"3d", "xy", "xz", "yz"}:
                self._toolbox_panel_visible = False
                was_active = projection == str(getattr(self, "_scene_projection_active", "3d") or "3d").lower()
                self._scene_projection_active = projection
                view_mode = self._scene_view_mode_from_projection(projection)
                if view_mode is not None:
                    previous_mode = self._active_scene_viewport_mode()
                    self._scene_viewport_mode = view_mode
                    self._scene_viewport_visible = True
                    if view_mode != previous_mode:
                        self._apply_scene_view_preset(view_mode)
                self._scene_projection_expanded = not bool(getattr(self, "_scene_projection_expanded", False)) if was_active else True
                self._draw_vector_overlay()
            return True
        if kind == "headlight":
            self._begin_canvas_live_edit()
            self._headlight_drag = region
            self._select_headlight(int(region.get("index", 1) or 1))
            if str(region.get("handle") or "") == "source":
                self._headlight_config_index = None
                self._headlight_panel_pos = None
                self._headlight_panel_anchor_index = None
            self._set_headlight_handle(region, getattr(event, "x", 0), getattr(event, "y", 0))
            try:
                canvas = getattr(self, "augmented_canvas", None)
                if canvas is not None:
                    canvas.grab_set()
            except Exception:
                pass
            self._draw_vector_overlay()
            return True
        if kind == "headlight_radius":
            self._begin_canvas_live_edit()
            self._headlight_radius_drag = region
            self._select_headlight(int(region.get("index", 1) or 1), reveal=True)
            self._set_headlight_radius_handle(region, getattr(event, "x", 0), getattr(event, "y", 0))
            try:
                canvas = getattr(self, "augmented_canvas", None)
                if canvas is not None:
                    canvas.grab_set()
            except Exception:
                pass
            self._draw_vector_overlay()
            return True
        if kind == "headlight_toggle":
            self._toggle_headlight_effect(int(region.get("index", 1) or 1))
            if bool(getattr(self, "_effect_inspector_visible", True)):
                self._render_active_toolbox()
            self._schedule_preview_refresh()
            self._draw_vector_overlay()
            return True
        if kind == "headlight_cone_toggle":
            self._toggle_headlight_cone_visibility(int(region.get("index", 1) or 1))
            if bool(getattr(self, "_effect_inspector_visible", True)):
                self._render_active_toolbox()
            self._draw_vector_overlay()
            return True
        if kind == "headlight_config_toggle":
            self._deactivate_wind_tool_ui()
            self._toolbox_panel_visible = False
            index = int(region.get("index", 1) or 1)
            if getattr(self, "_headlight_config_index", None) == index:
                self._select_headlight(index)
                self._headlight_config_index = None
                self._headlight_panel_pos = None
                self._headlight_panel_anchor_index = None
            else:
                if getattr(self, "_headlight_config_index", None) != index:
                    self._headlight_panel_pos = None
                    self._headlight_panel_anchor_index = None
                self._select_headlight(index, reveal=True)
                self._headlight_config_index = index
                self._active_toolbox = "illumination"
                self._pending_inspector_scroll_anchor = "headlights"
                self._show_effect_inspector(refresh=False)
                self._refresh_toolbox_tab_labels()
                self._refresh_effect_inspector_title()
                self._render_active_toolbox()
            self._draw_vector_overlay()
            return True
        if kind == "headlight_panel_drag":
            self._headlight_panel_drag = {
                "start_x": int(getattr(event, "x", 0)),
                "start_y": int(getattr(event, "y", 0)),
                "base": tuple(getattr(self, "_headlight_panel_pos", None) or (10, 52)),
                "index": int(region.get("index", getattr(self, "_headlight_config_index", 1)) or 1),
            }
            return True
        if kind == "headlight_panel_reset":
            self._headlight_panel_pos = None
            self._headlight_panel_anchor_index = None
            self._draw_vector_overlay()
            return True
        if kind == "headlight_panel":
            return True
        return True

    def _event_vector_tool(self, event) -> str | None:
        canvas = getattr(self, "augmented_canvas", None)
        if canvas is None:
            return None
        try:
            current = canvas.find_withtag("current")
            for item in current:
                for tag in canvas.gettags(item):
                    if str(tag).startswith("vector_tool:"):
                        return str(tag).split(":", 1)[1]
        except Exception:
            return None
        return None

    def _nearest_headlight_index(self, x: float, y: float, tolerance: float = 12.0) -> int | None:
        canvas = getattr(self, "augmented_canvas", None)
        if canvas is None:
            return None
        width = max(1, int(canvas.winfo_width() or 1))
        height = max(1, int(canvas.winfo_height() or 1))
        best_index = None
        best_distance = float(tolerance)
        for index in (1, 2, 3):
            source, target = self._headlight_points(index, width, height)
            sx, sy = float(source[0]), float(source[1])
            tx, ty = float(target[0]), float(target[1])
            vx = tx - sx
            vy = ty - sy
            line_len2 = max(1.0, vx * vx + vy * vy)
            ratio = max(0.0, min(1.0, ((float(x) - sx) * vx + (float(y) - sy) * vy) / line_len2))
            px = sx + vx * ratio
            py = sy + vy * ratio
            distance = math.hypot(float(x) - px, float(y) - py)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _on_vector_canvas_double_press(self, event):
        return None

    def _on_vector_canvas_press(self, event):
        if self._handle_canvas_overlay_press(event):
            return "break"
        if self._event_has_control(event) and self._scene_view_rotation_available():
            self._begin_scene_view_drag(event)
            return "break"
        tool = self._event_vector_tool(event)
        if tool:
            if self._vector_tool == tool:
                if tool == "wind":
                    self._deactivate_wind_tool_ui()
                else:
                    self._vector_tool = None
            else:
                if tool != "wind":
                    self._deactivate_wind_tool_ui()
                self._vector_tool = tool
            self._vector_drag_start = None
            self._vector_drag_end = None
            self._draw_vector_overlay()
            return "break"
        if self._vector_tool:
            if self._vector_tool == "wind":
                return None
            point = (int(event.x), int(event.y))
            self._vector_drag_start = point
            self._vector_drag_end = point
            self._draw_vector_overlay()
            return "break"
        if float(getattr(self, "_preview_zoom", 1.0) or 1.0) > 1.01:
            self._preview_pan_drag = (
                int(event.x),
                int(event.y),
                float(getattr(self, "_preview_pan_x", 0.0) or 0.0),
                float(getattr(self, "_preview_pan_y", 0.0) or 0.0),
            )
            return "break"
        return None

    def _on_vector_canvas_context_press(self, event):
        overlay = self._canvas_overlay_hit(event)
        if self._is_wind_panel_overlay(overlay):
            return "break"
        return None

    def _on_vector_canvas_context_drag(self, event):
        overlay = self._canvas_overlay_hit(event)
        if self._is_wind_panel_overlay(overlay):
            return "break"
        return None

    def _on_vector_canvas_context_release(self, event):
        overlay = self._canvas_overlay_hit(event)
        if self._is_wind_panel_overlay(overlay):
            return "break"
        return None

    def _on_vector_canvas_drag(self, event):
        if self._wind_panel_drag is not None:
            self._update_wind_panel_drag(event)
            self._request_vector_overlay_redraw(delay_ms=8)
            return "break"
        if self._scene_pan_drag is not None:
            self._update_scene_pan_drag(event)
            self._request_vector_overlay_redraw(delay_ms=8)
            return "break"
        if self._scene_view_drag is not None:
            self._update_scene_view_drag(event)
            self._request_vector_overlay_redraw(delay_ms=8)
            return "break"
        if self._scene_projection_drag is not None:
            self._set_scene_projection_overlay_point(self._scene_projection_drag, event)
            self._request_vector_overlay_redraw()
            return "break"
        if self._headlight_panel_drag is not None:
            start_x = int(self._headlight_panel_drag.get("start_x", 0) or 0)
            start_y = int(self._headlight_panel_drag.get("start_y", 0) or 0)
            base_x, base_y = self._headlight_panel_drag.get("base", (10, 52))
            self._headlight_panel_pos = (
                int(base_x) + int(getattr(event, "x", 0)) - start_x,
                int(base_y) + int(getattr(event, "y", 0)) - start_y,
            )
            try:
                self._headlight_panel_anchor_index = int(self._headlight_panel_drag.get("index", 0) or 0)
            except Exception:
                self._headlight_panel_anchor_index = getattr(self, "_headlight_config_index", None)
            self._request_vector_overlay_redraw()
            return "break"
        if self._toolbox_panel_drag is not None:
            start_x = int(self._toolbox_panel_drag.get("start_x", 0) or 0)
            start_y = int(self._toolbox_panel_drag.get("start_y", 0) or 0)
            base_x, base_y = self._toolbox_panel_drag.get("base", (10, 50))
            self._toolbox_panel_pos = (
                int(base_x) + int(getattr(event, "x", 0)) - start_x,
                int(base_y) + int(getattr(event, "y", 0)) - start_y,
            )
            self._request_vector_overlay_redraw()
            return "break"
        if self._headlight_radius_drag is not None:
            self._set_headlight_radius_handle(self._headlight_radius_drag, getattr(event, "x", 0), getattr(event, "y", 0))
            self._request_vector_overlay_redraw()
            return "break"
        if self._headlight_drag is not None:
            self._set_headlight_handle(self._headlight_drag, getattr(event, "x", 0), getattr(event, "y", 0))
            self._request_vector_overlay_redraw()
            return "break"
        if self._canvas_slider_drag is not None:
            if self._canvas_slider_drag.get("kind") == "range_slider":
                self._set_canvas_range_slider_value(
                    self._canvas_slider_drag,
                    getattr(event, "x", 0),
                    str(self._canvas_slider_drag.get("handle") or ""),
                )
            else:
                self._set_canvas_slider_value(self._canvas_slider_drag, getattr(event, "x", 0))
            self._request_vector_overlay_redraw()
            return "break"
        if self._preview_pan_drag is not None:
            start_x, start_y, base_x, base_y = self._preview_pan_drag
            self._preview_pan_x = base_x + (int(event.x) - start_x)
            self._preview_pan_y = base_y + (int(event.y) - start_y)
            self._refresh_preview(redraw_only=True)
            return "break"
        if not self._vector_tool or self._vector_drag_start is None:
            return None
        self._vector_drag_end = (int(event.x), int(event.y))
        self._request_vector_overlay_redraw()
        return "break"

    def _on_vector_canvas_release(self, event):
        if self._wind_panel_drag is not None:
            self._finish_wind_panel_drag(event)
            return "break"
        if self._scene_pan_drag is not None:
            self._update_scene_pan_drag(event)
            self._end_scene_pan_drag()
            self._draw_vector_overlay()
            return "break"
        if self._scene_view_drag is not None:
            self._update_scene_view_drag(event)
            self._end_scene_view_drag()
            self._draw_vector_overlay()
            return "break"
        if self._scene_projection_drag is not None:
            self._set_scene_projection_overlay_point(self._scene_projection_drag, event)
            self._scene_projection_drag = None
            self._release_augmented_canvas_live_grab()
            self._end_canvas_live_edit(delay_ms=40)
            self._draw_vector_overlay()
            return "break"
        if self._headlight_panel_drag is not None:
            self._headlight_panel_drag = None
            return "break"
        if self._toolbox_panel_drag is not None:
            self._toolbox_panel_drag = None
            return "break"
        if self._headlight_radius_drag is not None:
            self._set_headlight_radius_handle(self._headlight_radius_drag, getattr(event, "x", 0), getattr(event, "y", 0))
            self._headlight_radius_drag = None
            try:
                self._release_augmented_canvas_live_grab()
            except Exception:
                pass
            self._end_canvas_live_edit(delay_ms=40)
            self._draw_vector_overlay()
            return "break"
        if self._headlight_drag is not None:
            self._set_headlight_handle(self._headlight_drag, getattr(event, "x", 0), getattr(event, "y", 0))
            self._headlight_drag = None
            try:
                canvas = getattr(self, "augmented_canvas", None)
                if canvas is not None:
                    canvas.grab_release()
            except Exception:
                pass
            self._end_canvas_live_edit(delay_ms=40)
            self._draw_vector_overlay()
            return "break"
        if self._canvas_slider_drag is not None:
            if self._canvas_slider_drag.get("kind") == "range_slider":
                self._set_canvas_range_slider_value(
                    self._canvas_slider_drag,
                    getattr(event, "x", 0),
                    str(self._canvas_slider_drag.get("handle") or ""),
                )
            else:
                self._set_canvas_slider_value(self._canvas_slider_drag, getattr(event, "x", 0))
            self._canvas_slider_drag = None
            self._release_augmented_canvas_live_grab()
            self._end_canvas_live_edit(delay_ms=40)
            self._draw_vector_overlay()
            return "break"
        if self._preview_pan_drag is not None:
            self._preview_pan_drag = None
            return "break"
        if not self._vector_tool or self._vector_drag_start is None:
            return None
        tool = self._vector_tool
        start = self._vector_drag_start
        end = (int(event.x), int(event.y))
        self._vector_drag_start = None
        self._vector_drag_end = None
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 6:
            self._reset_vector_tool(tool)
            self._vector_tool = None
            self._draw_vector_overlay()
            return "break"
        self._vector_positions[tool] = (start, end)
        self._apply_vector_to_profile(tool, dx, dy, length)
        self._draw_vector_overlay()
        return "break"

    def _on_vector_canvas_motion(self, event):
        canvas = getattr(self, "augmented_canvas", None)
        if canvas is None:
            return None
        overlay = self._canvas_overlay_hit(event)
        hover_headlight_config = None
        if overlay and overlay.get("kind") == "headlight_config_toggle":
            try:
                hover_headlight_config = int(overlay.get("index", 0) or 0)
            except Exception:
                hover_headlight_config = None
        if getattr(self, "_headlight_config_hover_index", None) != hover_headlight_config:
            self._headlight_config_hover_index = hover_headlight_config
            self._request_vector_overlay_redraw()
        if self._wind_panel_drag is not None:
            cursor = "crosshair"
        elif self._scene_pan_drag is not None or self._scene_view_drag is not None:
            cursor = "fleur"
        elif overlay and overlay.get("kind") in ("slider", "range_slider"):
            cursor = "sb_h_double_arrow"
        elif overlay and overlay.get("kind") in ("panel_drag", "headlight_panel_drag"):
            cursor = "fleur"
        elif overlay and overlay.get("kind") in ("panel_reset", "panel_toggle", "headlight_panel_reset"):
            cursor = "hand2"
        elif overlay and overlay.get("kind") == "scene_viewport_mode":
            cursor = "hand2"
        elif overlay and overlay.get("kind") == "wind_strength_field":
            cursor = "xterm"
        elif overlay and overlay.get("kind") == "wind_vector_panel":
            cursor = "crosshair"
        elif overlay and overlay.get("kind") == "scene_viewport_panel":
            cursor = "fleur"
        elif overlay and overlay.get("kind") == "scene_plate_texture_toggle":
            cursor = "hand2"
        elif overlay and overlay.get("kind") == "scene_projection_panel" and self._event_has_control(event):
            cursor = "fleur"
        elif overlay and overlay.get("kind") in ("scene_projection_panel", "scene_projection_toggle"):
            cursor = "hand2"
        elif overlay and overlay.get("kind") == "headlight":
            cursor = "fleur"
        elif overlay and overlay.get("kind") == "headlight_radius":
            cursor = "crosshair"
        elif overlay and overlay.get("kind") == "headlight_config_toggle":
            cursor = "hand2"
        elif overlay and overlay.get("kind") == "headlight_panel":
            cursor = "hand2"
        elif overlay:
            cursor = "hand2"
        else:
            cursor = "hand2" if self._event_vector_tool(event) else ("fleur" if float(getattr(self, "_preview_zoom", 1.0) or 1.0) > 1.01 else "")
        try:
            if str(canvas.cget("cursor") or "") != cursor:
                canvas.configure(cursor=cursor)
        except Exception:
            pass
        return None

    def _on_vector_canvas_leave(self, _event):
        if self._headlight_drag is not None:
            return None
        if self._headlight_radius_drag is not None:
            return None
        if self._scene_projection_drag is not None:
            return None
        if self._scene_view_drag is not None:
            return None
        if self._scene_pan_drag is not None:
            return None
        if getattr(self, "_headlight_config_hover_index", None) is not None:
            self._headlight_config_hover_index = None
            self._request_vector_overlay_redraw()
        canvas = getattr(self, "augmented_canvas", None)
        if canvas is not None:
            try:
                canvas.configure(cursor="")
            except Exception:
                pass
        return None

    def _reset_vector_tool(self, tool: str):
        try:
            self._vector_positions.pop(tool, None)
            if tool == "wind":
                self.dirt_flow_wind_strength_var.set(0.0)
                self.dirt_flow_air_angle_var.set(0.0)
        except Exception:
            pass

    def _apply_vector_to_profile(self, tool: str, dx: float, dy: float, length: float):
        canvas = getattr(self, "augmented_canvas", None)
        width = int(canvas.winfo_width() if canvas is not None else 320)
        height = int(canvas.winfo_height() if canvas is not None else 240)
        value = max(0.0, min(1.0, length / self._vector_canvas_scale(width, height)))
        try:
            if tool == "wind":
                angle = math.degrees(math.atan2(-dy, dx))
                self.dirt_flow_air_angle_var.set(((angle + 180.0) % 360.0) - 180.0)
                self.dirt_flow_wind_strength_var.set(value)
        except Exception:
            pass

    def _draw_canvas_message(self, canvas: tk.Canvas, text: str):
        if canvas is None:
            return
        canvas.delete("all")
        width = max(1, int(canvas.winfo_width() or 1))
        height = max(1, int(canvas.winfo_height() or 1))
        canvas.create_text(
            width / 2,
            height / 2,
            text=str(text or ""),
            fill="#d0d0d0",
            width=max(120, width - 20),
            justify=tk.CENTER,
        )

    def _accept(self):
        self._release_augmented_canvas_live_grab()
        self._close_wind_strength_editor(commit=True)
        self._canvas_slider_drag = None
        self._wind_panel_drag = None
        self._headlight_drag = None
        self._headlight_radius_drag = None
        self._scene_projection_drag = None
        self._scene_view_drag = None
        self._scene_pan_drag = None
        self._inspector_scale_drag_active = False
        self._relief_profile_drag_index = None
        self._preview_refresh_suspended = False
        self._preview_refresh_dirty = False
        self._cancel_pending_preview()
        profile = self._profile_from_vars()
        # Dataset production values live in PZ1; this modal only stores effect settings.
        if False and bool(profile.enabled) and int(getattr(profile, "extra_count", 0) or 0) <= 0:
            messagebox.showwarning(
                "Liczba dodatkowych zdjęć",
                "Wpisz, ile dodatkowych zdjęć program ma wygenerować do części train.\n\n"
                "Podgląd może działać bez tej liczby, ale zapis profilu augmentacji wymaga jawnej decyzji.",
                parent=self.window,
            )
            return
        self.result = profile
        self._unbind_fullscreen_preview_tab_toggle()
        self.window.destroy()

    def _cancel(self):
        self._release_augmented_canvas_live_grab()
        self._close_wind_strength_editor(commit=False)
        self._canvas_slider_drag = None
        self._wind_panel_drag = None
        self._headlight_drag = None
        self._headlight_radius_drag = None
        self._scene_projection_drag = None
        self._scene_view_drag = None
        self._scene_pan_drag = None
        self._inspector_scale_drag_active = False
        self._relief_profile_drag_index = None
        self._preview_refresh_suspended = False
        self._preview_refresh_dirty = False
        self._cancel_pending_preview()
        self.result = None
        self._unbind_fullscreen_preview_tab_toggle()
        self.window.destroy()

    def _unbind_fullscreen_preview_tab_toggle(self):
        tag = str(getattr(self, "_tab_toggle_bindtag", "") or "")
        if not tag:
            return
        try:
            for sequence in ("<Tab>", "<ISO_Left_Tab>", "<Shift-Tab>"):
                self.window.bind_class(tag, sequence, "")
        except Exception:
            pass

    def _cancel_pending_preview(self):
        self._raw_preview_request_id += 1
        self._raw_preview_pending = False
        if self._preview_after_id is not None:
            try:
                self.window.after_cancel(self._preview_after_id)
            except Exception:
                pass
        self._preview_after_id = None
        if self._overlay_redraw_after_id is not None:
            try:
                self.window.after_cancel(self._overlay_redraw_after_id)
            except Exception:
                pass
        self._overlay_redraw_after_id = None


def ask_step4_augmentation_profile(
    master,
    *,
    target: str,
    profile: AugmentationProfile,
    sample_images: list[Path] | None = None,
    sample_pool_limit: int | None = None,
    install_callback=None,
) -> AugmentationProfile | None:
    return Step4AugmentationModal(
        master,
        target=target,
        profile=profile,
        sample_images=sample_images,
        sample_pool_limit=sample_pool_limit,
        install_callback=install_callback,
    ).show()
