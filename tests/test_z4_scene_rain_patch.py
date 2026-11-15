from pathlib import Path
import runpy
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def patcher():
    return runpy.run_path(str(ROOT / "patch_Z4RAINEDGE001_scene_contour_collision.py"))


def test_patch_removes_blocker_even_when_replacement_is_a_substring(patcher):
    old = '        "rain_edge_mist_strength",\n        "tyndall_strength",\n'
    new = '        "tyndall_strength",\n'
    replace = patcher["replace_fragment"]
    assert replace(old, old, new, "eligibility") == new
    assert replace(new, old, new, "eligibility") == new


def test_patch_repairs_wrong_function_and_preserves_other_code(patcher):
    source = (ROOT / patcher["AUG_REL"]).read_text(encoding="utf-8-sig")
    edit = patcher["replace_in_function"]
    scope, old, new, label = next(step for step in patcher["AUG_STEPS"] if step[3] == "preserve scene for edge map")
    broken = edit(source, scope, new, old, label)
    misplaced, original = patcher["MISPLACED_SCENE_BASE"]
    broken = edit(broken, "_apply_mt_scene_coarse_noise_cuda", original, misplaced, "faulty patch")
    broken = edit(broken, "_mt_scene_cuda_fastpath_eligible", '    blocked = (\n', '    blocked = (\n        "rain_edge_mist_strength",\n', "old blocker")
    assert patcher["update_backend"](broken) == source


def copy_targets(patcher, tmp_path):
    paths = [patcher["MODAL_REL"], patcher["AUG_REL"]]
    paths += [rel for rel, _, _ in patcher["TEST_UPDATES"]]
    paths += list(patcher["TEST_FILES"])
    for rel in paths:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    return paths


def test_reapplying_patch_does_not_rewrite_files(patcher, tmp_path):
    paths = copy_targets(patcher, tmp_path)
    before = {rel: ((tmp_path / rel).read_bytes(), (tmp_path / rel).stat().st_mtime_ns) for rel in paths}
    assert patcher["main"](tmp_path) == 0
    assert patcher["main"](tmp_path) == 0
    after = {rel: ((tmp_path / rel).read_bytes(), (tmp_path / rel).stat().st_mtime_ns) for rel in paths}
    assert before == after


def test_patch_upgrades_previous_straight_splashes(patcher, tmp_path):
    paths = copy_targets(patcher, tmp_path)
    expected = {rel: (tmp_path / rel).read_text(encoding="utf-8-sig") for rel in paths}
    backend = expected[patcher["AUG_REL"]]
    edit = patcher["replace_in_function"]
    for name, old, new, label in reversed(patcher["DENSITY_STEPS"]):
        backend = edit(backend, name, new, old, label)
    for name, old, new, label in reversed(patcher["COLLISION_SOURCE_STEPS"]):
        backend = edit(backend, name, new, old, label)
    for name, old, new, label in reversed(patcher["SPLASH_STEPS"]):
        backend = edit(backend, name, new, old, label)
    for name, previous in patcher["LEGACY_CONTACT_HELPERS"].items():
        start, end = patcher["function_bounds"](backend, name)
        backend = backend[:start] + previous + backend[end:]
    for name in ("_rain_alpha_parameters", "_rain_fragment_active", "_rain_splash_trajectory"):
        start, end = patcher["function_bounds"](backend, name)
        backend = backend[:start] + backend[end + 2:]
    (tmp_path / patcher["AUG_REL"]).write_text(backend, encoding="utf-8")
    modal = expected[patcher["MODAL_REL"]]
    for name, old, new, label in reversed(patcher["DENSITY_MODAL_STEPS"]):
        modal = edit(modal, name, new, old, label)
    for name, old, new, label in reversed(patcher["SPLASH_MODAL_STEPS"]):
        modal = edit(modal, name, new, old, label)
    (tmp_path / patcher["MODAL_REL"]).write_text(modal, encoding="utf-8")
    for name, previous in patcher["LEGACY_TEST_FILES"].items():
        (tmp_path / name).write_text(previous, encoding="utf-8")
    assert patcher["main"](tmp_path) == 0
    assert expected == {rel: (tmp_path / rel).read_text(encoding="utf-8-sig") for rel in paths}


def test_patch_upgrades_collision_detection_from_the_noisy_image(patcher):
    expected = (ROOT / patcher["AUG_REL"]).read_text(encoding="utf-8-sig")
    previous = expected
    for name, old, new, label in reversed(patcher["COLLISION_SOURCE_STEPS"]):
        previous = patcher["replace_in_function"](previous, name, new, old, label)
    assert patcher["update_backend"](previous) == expected


def test_patch_upgrades_previous_non_accumulating_splashes(patcher):
    expected = (ROOT / patcher["AUG_REL"]).read_text(encoding="utf-8-sig")
    previous = expected
    for name, old, new, label in reversed(patcher["DENSITY_STEPS"]):
        previous = patcher["replace_in_function"](previous, name, new, old, label)
    for name, helper in patcher["DENSITY_PREVIOUS_HELPERS"].items():
        start, end = patcher["function_bounds"](previous, name)
        previous = previous[:start] + helper + previous[end:]
    for name in ("_rain_alpha_parameters", "_rain_fragment_active"):
        start, end = patcher["function_bounds"](previous, name)
        previous = previous[:start] + previous[end + 2:]
    assert patcher["update_backend"](previous) == expected


def test_invalid_backend_fails_before_writing_modal(patcher, tmp_path):
    paths = copy_targets(patcher, tmp_path)
    modal = tmp_path / patcher["MODAL_REL"]
    source = modal.read_text(encoding="utf-8-sig")
    scope, old, new, label = patcher["MODAL_STEPS"][0]
    modal.write_text(patcher["replace_in_function"](source, scope, new, old, label), encoding="utf-8")
    (tmp_path / patcher["AUG_REL"]).write_text("# unsupported source\n", encoding="utf-8")
    before = {rel: (tmp_path / rel).read_bytes() for rel in paths}
    with pytest.raises(RuntimeError):
        patcher["main"](tmp_path)
    assert before == {rel: (tmp_path / rel).read_bytes() for rel in paths}
