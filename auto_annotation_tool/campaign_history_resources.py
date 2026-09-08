"""Read-only, iteration-scoped evidence for the project history viewer."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET


HISTORY_RESOURCE_SCHEMA = "alpr.project_history_resources.v1"


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _number(value) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _key(value) -> str:
    return str(value or "").strip().replace("\\", "/").casefold()


def _not_after(value, cutoff: str) -> bool:
    return not cutoff or bool(value and str(value)[:19] <= cutoff[:19])


class HistoryResourceReader:
    """A single read of small manifests per dialog; no image directory scans."""

    def __init__(self, project_root: Path, project: str):
        self.root = Path(project_root)
        self.project = project
        self.registry = _read_json(self.root / "_campaign_state" / "artifact_registry.json")
        self.approved = _read_json(self.root / "_campaign_state" / "plate_approved_set.json")
        self._files: dict[str, dict] = {}
        self._run_counts: dict[tuple, dict] = {}
        self._snapshots: dict[tuple, dict] = {}

    def _json(self, path) -> dict:
        text = str(path or "").strip()
        if not text:
            return {}
        if text not in self._files:
            self._files[text] = _read_json(Path(text))
        return self._files[text]

    def _bundle(self, iteration: int) -> dict:
        key = _mapping(self.registry.get("iteration_index")).get(str(iteration))
        return _mapping(_mapping(self.registry.get("packages")).get(key))

    def _approved_run(self, source: dict, cutoff: str) -> dict:
        run_dir = str(source.get("run_dir") or "")
        if not run_dir or not _not_after(source.get("updated_at"), cutoff):
            return {}
        manifest = self._json(Path(run_dir) / "run_manifest.json")
        if cutoff and any(
            str(manifest.get(field) or "")[:19] > cutoff[:19]
            for field in ("last_manual_edit_at", "resume_preview_saved_at")
        ):
            return {}
        names = {_key(name) for name in (manifest.get("approved_filenames") or []) if name}
        if not names:
            return {}
        cache_key = (run_dir, tuple(sorted(names)))
        if cache_key in self._run_counts:
            return self._run_counts[cache_key]
        xml_path = Path(source.get("xml_path") or (Path(run_dir) / "annotations.xml"))
        input_dir = manifest.get("source_input_dir") or source.get("input_source")
        if not input_dir:
            return {}
        counts = {}
        try:
            for _, element in ET.iterparse(xml_path, events=("end",)):
                if element.tag != "image":
                    continue
                name = element.get("name", "")
                if _key(name) in names:
                    plates = sum(
                        1 for child in element
                        if child.tag in {"polygon", "box"}
                        and str(child.get("label") or "").lower() in {"plate", "tablica", "license_plate", "licence_plate"}
                    )
                    if plates:
                        counts[_key(Path(input_dir) / name)] = plates
                element.clear()
        except (OSError, ET.ParseError):
            return {}
        self._run_counts[cache_key] = counts
        return counts

    def _plate_counts(self, iteration: int, cutoff: str, bundle: dict) -> dict:
        previous, current = {}, {}
        unknown_geometry = False
        unknown_previous_geometry = False
        changed_previous_plates = 0
        for entry in _mapping(self.approved.get("entries")).values():
            if not isinstance(entry, dict):
                continue
            first = _number(entry.get("first_approved_iteration") or entry.get("approved_iteration"))
            approved_at = entry.get("first_approved_at") or entry.get("approved_at")
            if not first or first > iteration or not _not_after(approved_at, cutoff):
                continue
            identity = _key(entry.get("source_image_path") or entry.get("entry_key"))
            count = _number(entry.get("plate_count"))
            if not count:
                count = sum(1 for plate in (entry.get("plates") or []) if isinstance(plate, dict) and len(plate.get("polygon") or []) >= 4)
            if not identity or not count:
                continue
            if cutoff and str(entry.get("approved_at") or "")[:19] > cutoff[:19]:
                # A later edit is not evidence of the earlier polygon count.
                # An unchanged approval count can still be recovered from the frozen run.
                unknown_geometry = True
            if first < iteration and (_number(entry.get("approved_iteration")) or first) >= iteration:
                unknown_previous_geometry = True
            (current if first == iteration else previous)[identity] = count

        source = _mapping(bundle.get("plate_source"))
        last_iteration = _number(bundle.get("iteration_last_seen"))
        source_is_historical = last_iteration in {None, iteration}
        run = self._approved_run(source, cutoff) if source_is_historical else {}
        if run:
            for identity, count in run.items():
                if identity in previous:
                    changed_previous_plates += count - previous[identity]
                else:
                    current[identity] = count
            if set(previous).union(current).issubset(run):
                unknown_geometry = False

        if not previous and not current:
            return {}
        return {
            "images": len(previous) + len(current),
            "plates": None if unknown_geometry else sum(previous.values()) + sum(current.values()) + changed_previous_plates,
            "iteration_images": len(current),
            "iteration_plates": None if unknown_geometry or unknown_previous_geometry else sum(current.values()) + changed_previous_plates,
            "previous_images": len(previous),
            "previous_plates": None if unknown_geometry or unknown_previous_geometry else sum(previous.values()),
            "source": "approved_set_and_run" if run else "approved_set_by_first_iteration",
            "run_dir": source.get("run_dir", "") if run else "",
        }

    def snapshot(self, iteration: int, *, cutoff: str = "") -> dict:
        cache_key = (iteration, cutoff)
        if cache_key in self._snapshots:
            return self._snapshots[cache_key]
        bundle = self._bundle(iteration)
        state = _mapping(_mapping(self.registry.get("iteration_state")).get(str(iteration)))
        result = {"schema": HISTORY_RESOURCE_SCHEMA, "project": self.project, "iteration": iteration}
        at = self._plate_counts(iteration, cutoff, bundle)
        if at:
            result["AT"] = at

        ingest_path = self.root / "_campaign_state" / "ingest" / f"iter_{iteration:03d}_manifest.json"
        ingest = self._json(ingest_path)
        count = _number(ingest.get("selected_count"))
        if count is not None and _not_after(ingest.get("created_at"), cutoff):
            result["O"] = {"images": count, "source": str(ingest_path), "source_iteration": iteration}

        contracts = _mapping(state.get("t06_contracts"))
        export = _mapping(contracts.get("pz3_char_dataset"))
        summary = self._json(export.get("summary_path"))
        summary_iteration = _number(summary.get("created_iteration") or summary.get("iteration"))
        if summary_iteration and summary_iteration <= iteration and _not_after(summary.get("created_at"), cutoff):
            export = summary
        else:
            stamp = export.get("fulfilled_at") or export.get("updated_at")
            if not _not_after(stamp, cutoff):
                export = {}
        created_iteration = _number(export.get("created_iteration") or export.get("source_iteration") or export.get("iteration"))
        if export and created_iteration and created_iteration <= iteration:
            plates = _number(export.get("exportable_plate_count"))
            chars = _number(export.get("exportable_char_count"))
            if plates is not None and chars is not None:
                result["AZ"] = {
                    "plates": plates, "characters": chars, "source_iteration": created_iteration,
                    "source": export.get("summary_path") or contracts.get("pz3_char_dataset", {}).get("summary_path", ""),
                }
                dataset_path = export.get("gold_dataset_path") or export.get("dataset_path")
                if dataset_path and (export.get("gold_dataset_valid") or export.get("gold_dataset_created")):
                    result["DS"] = {
                        "images": plates, "objects": chars, "target": "char",
                        "source_iteration": created_iteration, "path": dataset_path,
                    }
        training_dataset = _mapping(state.get("step4_dataset"))
        if training_dataset and _not_after(training_dataset.get("updated_at"), cutoff):
            result["dataset_variant"] = {
                field: training_dataset.get(field)
                for field in ("dataset_path", "target", "dataset_iteration", "train_images", "val_images", "test_images", "total_images")
            }
        training = _mapping(state.get("step4_training"))
        run_id = str(training.get("run_id") or "")
        if run_id:
            history = self._json(self.root / "5_training_runs" / "training_history.json")
            run = _mapping(_mapping(history.get("runs")).get(run_id))
            if run and _not_after(run.get("created_at"), cutoff):
                result["training_run"] = {
                    "run_id": run_id,
                    "dataset_path": run.get("dataset_path", ""),
                    "training_dataset_snapshot": run.get("training_dataset_snapshot") or {},
                    "source": "training_history",
                }
                target = run.get("training_target") or training.get("target")
                if target in {"plate", "char"} and run.get("status") == "completed" and _not_after(run.get("finished_at"), cutoff):
                    result["MT" if target == "plate" else "MZ"] = {
                        "run_id": run_id,
                        "path": run.get("best_weights", ""),
                        "source_iteration": training.get("trained_iteration") or training.get("iteration") or iteration,
                        "source": "training_history",
                    }
        self._snapshots[cache_key] = result
        return result

    def for_event(self, entry: dict) -> dict:
        recorded = (entry.get("artifacts") or {}).get("resource_snapshot")
        if (
            isinstance(recorded, dict) and recorded.get("schema") == HISTORY_RESOURCE_SCHEMA
            and recorded.get("project") == self.project
            and recorded.get("iteration") == int(entry.get("iteration") or 0)
        ):
            return recorded
        return self.snapshot(int(entry.get("iteration") or 0), cutoff=str(entry.get("created_at") or ""))
