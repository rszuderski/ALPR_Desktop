from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import z2_gt_companion_flow as flow


class FakeSession:
    def __init__(self):
        self.data = {}
        self.saved = 0

    def get(self, tab, key, default=""):
        return self.data.get(tab, {}).get(key, default)

    def set(self, tab, key, value):
        self.data.setdefault(tab, {})[key] = value

    def save_session(self):
        self.saved += 1


def _empty_discovery(root):
    return {
        "root": str(root),
        "schema": "alpr.image-resource.companions.v1",
        "ground_truth": [],
        "invalid": [],
        "signature": "",
    }


def test_free_mode_without_source_gt_still_produces_current_work(tmp_path, monkeypatch):
    session = FakeSession()
    bind = Mock()
    monkeypatch.setattr(flow, "SESSION", session)
    monkeypatch.setattr(flow, "discover_gt_pack_companions", _empty_discovery)
    monkeypatch.setattr(flow.z2_gt_pack_runtime, "set_gt_resource_binding", bind)

    host = SimpleNamespace(frame=None, app=None)
    result = flow.prepare_free_mode_gt_binding(host, tmp_path)

    expected = tmp_path / "current_work.alprgt"
    assert result["found"] is False
    assert result["working_path"] == str(expected)
    assert session.saved == 1
    assert bind.call_args.kwargs["source_paths"] == []
    assert bind.call_args.kwargs["working_path"] == expected


def test_discovered_source_is_auto_bound_without_prompt(tmp_path, monkeypatch):
    session = FakeSession()
    bind = Mock()
    source = tmp_path / "source.alprgt"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(flow, "SESSION", session)
    monkeypatch.setattr(
        flow,
        "discover_gt_pack_companions",
        lambda root: {
            "root": str(root),
            "schema": "alpr.image-resource.companions.v1",
            "ground_truth": [{"path": str(source), "name": source.name}],
            "invalid": [],
            "signature": "sig-1",
        },
    )

    def fail_if_prompted(*args, **kwargs):
        raise AssertionError("free mode must not prompt for GT companion mounts")

    monkeypatch.setattr(flow, "_confirm", fail_if_prompted)
    monkeypatch.setattr(flow.z2_gt_pack_runtime, "set_gt_resource_binding", bind)

    host = SimpleNamespace(frame=None, app=None)
    result = flow.prepare_free_mode_gt_binding(host, tmp_path)

    expected = tmp_path / "current_work.alprgt"
    assert result["found"] is True
    assert result["accepted"] is True
    assert result["auto_bound"] is True
    assert result["paths"] == [source]
    assert result["working_path"] == str(expected)
    assert bind.call_args.kwargs["source_paths"] == [source]
    assert bind.call_args.kwargs["working_path"] == expected

    saved = session.data["annotation"][flow.FREE_BINDINGS_SESSION_KEY]
    entry = next(iter(saved["bindings"].values()))
    assert entry["policy"] == "auto_companion_v1"
    assert entry["accepted"] is True
    assert entry["source_paths"] == [str(source)]


def test_old_declined_binding_is_migrated_to_auto_companion_policy(tmp_path, monkeypatch):
    session = FakeSession()
    bind = Mock()
    source = tmp_path / "source.alprgt"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")

    key = flow._resource_key(tmp_path)
    session.data = {
        "annotation": {
            flow.FREE_BINDINGS_SESSION_KEY: {
                "schema": flow.FREE_BINDINGS_SCHEMA,
                "bindings": {
                    key: {
                        "signature": "sig-1",
                        "accepted": False,
                        "source_paths": [],
                        "working_path": str(tmp_path / "legacy-other.alprgt"),
                    }
                },
            }
        }
    }

    monkeypatch.setattr(flow, "SESSION", session)
    monkeypatch.setattr(
        flow,
        "discover_gt_pack_companions",
        lambda root: {
            "root": str(root),
            "schema": "alpr.image-resource.companions.v1",
            "ground_truth": [{"path": str(source), "name": source.name}],
            "invalid": [],
            "signature": "sig-1",
        },
    )
    monkeypatch.setattr(flow.z2_gt_pack_runtime, "set_gt_resource_binding", bind)

    host = SimpleNamespace(frame=None, app=None)
    result = flow.prepare_free_mode_gt_binding(host, tmp_path)

    expected = tmp_path / "current_work.alprgt"
    assert result["accepted"] is True
    assert result["paths"] == [source]
    assert result["working_path"] == str(expected)
    assert result["restored_binding"] is False

    saved = session.data["annotation"][flow.FREE_BINDINGS_SESSION_KEY]
    entry = saved["bindings"][key]
    assert entry["policy"] == "auto_companion_v1"
    assert entry["accepted"] is True
    assert entry["source_paths"] == [str(source)]
    assert entry["working_path"] == str(expected)
