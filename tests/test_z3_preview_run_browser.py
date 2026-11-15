import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import z3_preview_run_browser as browser


class Value:
    def __init__(self, value=""):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


def make_run(root: Path, name: str, rows: dict):
    run = root / name
    images = run / "images"
    images.mkdir(parents=True)
    for pid in rows:
        (images / f"{pid}.jpg").write_bytes(b"x")
    (run / "metadata.json").write_text(json.dumps(rows), encoding="utf-8")
    return run


def candidate_host(root: Path, linear=False):
    return SimpleNamespace(
        _step3_linear_mode=linear,
        _get_step3_chars_root_dir=lambda **kwargs: root,
        _is_usable_step3_preview_dir=lambda path, **kwargs: True,
    )


def test_collect_candidates_reads_counts_and_uses_run_name_timestamp(tmp_path):
    rows = {
        "plate_000001": {
            "characters": [{"character": "A"}],
            "status": "perfect",
            "gold_state": {"approved": True, "excluded": False},
            "crop_id": "CROP-1",
            "crop_identity_sha256": "a" * 64,
        },
        "plate_000002": {
            "characters": [],
            "status": "needs_fix",
            "gold_state": {"excluded": True},
        },
    }
    run = make_run(tmp_path, "run_001_20260925_115720", rows)
    result = browser.collect_free_preview_run_candidates(candidate_host(tmp_path))
    assert len(result) == 1
    item = result[0]
    assert item["run_dir"] == run
    assert item["plate_count"] == 2
    assert item["with_chars"] == 1
    assert item["gold_count"] == 1
    assert item["excluded_count"] == 1
    assert item["crop_registry_count"] == 1
    assert item["created_at"] == "25.09.2026 11:57"


def test_campaign_does_not_scan_free_preview_history(tmp_path):
    host = candidate_host(tmp_path, linear=True)
    host._get_step3_chars_root_dir = Mock(side_effect=AssertionError("campaign must not scan free history"))
    assert browser.collect_free_preview_run_candidates(host) == []
    host._get_step3_chars_root_dir.assert_not_called()


def test_format_label_is_human_readable():
    label = browser.format_preview_run_label({
        "run_dir": Path("run_001"),
        "created_at": "25.09.2026 11:57",
        "plate_count": 11,
        "with_chars": 3,
    })
    assert label == "run_001 | 25.09.2026 11:57 | tablice: 11 | znaki: 3/11"


def test_even_sample_covers_first_middle_last():
    assert browser._select_evenly_spaced(list(range(9)), 3) == [0, 4, 8]


def test_use_preview_run_clears_xml_source_and_opens_exact_run(tmp_path):
    run = make_run(tmp_path, "run_001_20260925_115720", {"plate_000001": {}})
    opened = []
    saved = []
    host = SimpleNamespace(
        frame=None,
        _is_usable_step3_preview_dir=lambda path, **kwargs: Path(path) == run,
        annotation_run_dir_var=Value("z2-run"),
        xml_path_var=Value("annotations.xml"),
        images_dir_var=Value("images"),
        preview_dir_var=Value(""),
        _source_binding_sync_in_progress=False,
        _extract_last_source_binding_result={"ok": True},
        _reset_preview_cache=Mock(),
        _save_local_setting=lambda key, value: saved.append((key, value)),
        _force_save_all=Mock(),
        _persist_step3_extract_state=Mock(),
        _open_detection_subtab_with_preview=lambda path, **kwargs: (opened.append((Path(path), kwargs)) or True),
    )
    assert browser.use_preview_run_path(host, run)
    assert host.preview_dir_var.get() == str(run)
    assert host.xml_path_var.get() == ""
    assert host.images_dir_var.get() == ""
    assert host.annotation_run_dir_var.get() == ""
    assert host._extract_last_source_binding_result == {"ok": False}
    assert opened == [(run, {"force_reload": True})]
    assert saved == [("char_preview_dir", str(run))]


def test_invalid_preview_run_never_opens_pz2(tmp_path):
    host = SimpleNamespace(
        frame=None,
        _is_usable_step3_preview_dir=lambda path, **kwargs: False,
        _open_detection_subtab_with_preview=Mock(),
    )
    with patch.object(browser.messagebox, "showwarning"):
        assert not browser.use_preview_run_path(host, tmp_path)
    host._open_detection_subtab_with_preview.assert_not_called()


def test_prepare_preview_image_adds_margin_without_cropping():
    from PIL import Image

    source = Image.new("RGB", (100, 40), (20, 30, 40))
    rendered = browser._prepare_preview_image(source, max_size=(500, 500))

    assert rendered.size[0] > 100
    assert rendered.size[1] > 40
    assert rendered.getpixel(
        (rendered.size[0] // 2, rendered.size[1] // 2)
    ) == (20, 30, 40)
