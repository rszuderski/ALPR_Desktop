from pathlib import Path

from PIL import Image

from auto_annotation_tool.gt_pack import ALPRGTPack
from auto_annotation_tool.gt_resource_companions import (
    discover_gt_pack_companions,
    get_campaign_gt_companion_paths,
    import_and_bind_campaign_gt_companions,
)


def _image(path: Path):
    Image.new("RGB", (96, 48), (30, 60, 90)).save(path)
    return path


def _pack(root: Path, image: Path, *, name="truth.alprgt"):
    pack = ALPRGTPack.create(root / name)
    item = pack.add_image(image)
    pack.ensure_plate(
        image_id=item["image_id"],
        polygon=[(10, 10), (70, 10), (70, 30), (10, 30)],
        plate_id="plate-ann-companion",
    )
    pack.set_ground_truth("plate-ann-companion", "ABC123")
    pack.set_plate_layout_gt("plate-ann-companion", "two_row")
    return pack


class FakeCampaign:
    def __init__(self, project_root: Path, image_dir: Path):
        self.project_root = Path(project_root)
        self.image_dir = Path(image_dir)
        self.bundle = {
            "package_id": "pkg_test",
            "image_source": {
                "package_id": "pkg_test",
                "image_set_token": "iset_test",
            },
        }

    def get_active_project_root_dir(self):
        return self.project_root

    def get_current_iteration_num(self):
        return 1

    def get_iteration_image_set_token(self, iteration):
        return "iset_test"

    def get_iteration_artifact_bundle(self, **kwargs):
        return dict(self.bundle)

    def upsert_iteration_artifact_bundle(self, **kwargs):
        updates = kwargs.get("updates") or {}
        image_source = dict(self.bundle.get("image_source") or {})
        image_source.update(dict(updates.get("image_source") or {}))
        self.bundle["image_source"] = image_source
        return dict(self.bundle)


def test_discovers_only_direct_source_packs_and_excludes_default_working(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = _image(image_dir / "source.png")
    _pack(image_dir, image, name="truth.alprgt")
    _pack(image_dir, image, name="current_work.alprgt")
    nested = image_dir / "nested"
    nested.mkdir()
    _pack(nested, image, name="nested.alprgt")

    result = discover_gt_pack_companions(image_dir)
    assert [Path(item["path"]).name for item in result["ground_truth"]] == ["truth.alprgt"]
    assert result["ground_truth"][0]["plates"] == 1
    assert result["ground_truth"][0]["gt_set"] == 1
    assert result["ground_truth"][0]["layout_set"] == 1


def test_campaign_import_is_project_relative_and_survives_project_move(tmp_path):
    image_dir = tmp_path / "source"
    image_dir.mkdir()
    image = _image(image_dir / "source.png")
    _pack(image_dir, image)
    discovery = discover_gt_pack_companions(image_dir)

    project = tmp_path / "project"
    project.mkdir()
    campaign = FakeCampaign(project, image_dir)
    result = import_and_bind_campaign_gt_companions(
        campaign,
        image_dir=image_dir,
        discoveries=discovery["ground_truth"],
    )
    assert len(result["paths"]) == 1
    assert result["paths"][0].is_dir()
    assert get_campaign_gt_companion_paths(campaign, image_dir=image_dir) == result["paths"]

    moved = tmp_path / "moved-project"
    project.rename(moved)
    campaign.project_root = moved
    resolved = get_campaign_gt_companion_paths(campaign, image_dir=image_dir)
    assert len(resolved) == 1
    assert resolved[0].is_dir()
    assert moved in resolved[0].parents


def test_validate_source_pack_does_not_rewrite_manifest(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = _image(image_dir / "source.png")
    pack = _pack(image_dir, image)
    manifest = pack.manifest_path
    before_bytes = manifest.read_bytes()
    before_mtime = manifest.stat().st_mtime_ns

    opened = ALPRGTPack.open(pack.root)
    assert opened.validate(deep=False)["ok"]

    assert manifest.read_bytes() == before_bytes
    assert manifest.stat().st_mtime_ns == before_mtime
