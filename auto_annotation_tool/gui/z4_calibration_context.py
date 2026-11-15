"""UI identities only. Calibration/export policies live in the existing backends."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..exporters import int8_calibration as calibration


def field(obj, key, default=""):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def role_of(candidate):
    return {"plate": "plate", "char": "character", "character": "character"}.get(
        str(candidate.get("role") or candidate.get("target") or ""), "")


def export_image_size(candidate):
    """Explicit series defaults, independent of the training resolution."""
    return int(candidate.get("export_image_size") or {"plate": 512, "character": 320}.get(
        role_of(candidate), candidate.get("img_size") or field(candidate.get("run"), "img_size", 640)))


@dataclass(frozen=True)
class CalibrationContext:
    role: str
    run_id: str
    history_path: str
    checkpoint: str
    image_size: int
    confidence_threshold: float = .25
    iou_threshold: float = .45
    model_label: str = ""
    source: str = "Wybierz trening ręcznie"
    expected_sha256: str = ""

    def identity(self):
        return (self.role, self.run_id, str(Path(self.history_path).resolve()), str(Path(self.checkpoint).resolve()),
                self.image_size, self.confidence_threshold, self.iou_threshold)


def context_from_candidate(candidate, *, image_size=None, confidence_threshold=.25, iou_threshold=.45, source="Wskazany model"):
    if not candidate:
        raise ValueError("Najpierw wskaż model MT lub MZ na liście. Checkbox Eksport nie wskazuje kontekstu kalibracji.")
    role = role_of(candidate)
    if not role or candidate.get("target_conflict"):
        raise ValueError("Kalibracja MT/MZ: wskazany model MP lub konflikt roli. Wskaż jednoznaczny MT albo MZ.")
    run = candidate.get("run") or candidate.get("history_snapshot") or candidate.get("run_snapshot") or {}
    run_id = str(field(run, "id") or candidate.get("run_id") or "")
    history_dir = str(candidate.get("history_dir") or field(run, "_history_dir") or "")
    history = str(candidate.get("history_path") or (Path(history_dir) / "training_history.json" if history_dir else ""))
    checkpoint = str(candidate.get("best_weights") or "")
    if not run_id or not history:
        raise ValueError("Brak jednoznacznej referencji run → historia dla wskazanego modelu. Użyj „Wybierz trening ręcznie”.")
    if not checkpoint or not Path(checkpoint).is_file():
        raise ValueError("Brak checkpointu wskazanego modelu: " + checkpoint)
    return CalibrationContext(role, run_id, str(Path(history).resolve()), str(Path(checkpoint).resolve()),
        int(image_size if image_size is not None else export_image_size(candidate)), float(confidence_threshold),
        float(iou_threshold), str(candidate.get("model_label") or ""), source,
        str(candidate.get("verified_checkpoint_sha256") or ""))


def run_label(run_id, run, role):
    marker = {"plate": "MT", "character": "MZ"}.get(role, "rola nieustalona")
    name = str(run.get("display_name") or run.get("name") or "nazwa nieustalona")
    if len(name) > 60:
        name = name[:57] + "…"
    architecture = str(run.get("architecture") or run.get("model_version") or "")
    stamp = str(run.get("started_at") or run.get("created_at") or "")
    try:
        stamp = datetime.fromisoformat(stamp).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        stamp = "data nieustalona"
    return f"{marker} · {name}" + (f" / {architecture}" if architecture else "") + f" · {stamp} · Run: {run_id}"


def verify_selection(history_path, run_id, role, *, context=None, expected_sha256=""):
    """Read-only; called on a worker, never hash a checkpoint on Tk's thread."""
    history = Path(history_path).resolve()
    run = calibration._read(history)["runs"][run_id]
    if run.get("status") != "completed" or str(run.get("id", run_id)) != run_id:
        raise ValueError("Popraw trening: wymagany dokładny, ukończony run z wybranej historii.")
    inv_role = run.get("training_dataset_snapshot", {}).get("training_source_hash_inventory", {}).get("role")
    if inv_role != role:
        raise ValueError("Popraw rolę: historia/inwentarz treningu nie potwierdza wybranego MT/MZ.")
    checkpoint = Path(run["best_weights"]).resolve()
    digest = calibration.file_sha256(checkpoint)
    if context:
        if (str(history), run_id, role) != (context.history_path, context.run_id, context.role):
            raise ValueError("Zmieniono historię, run lub rolę przypiętego modelu. Wróć do jego wyboru.")
        if calibration.file_sha256(context.checkpoint) != digest:
            raise ValueError("Konflikt SHA: checkpoint wskazanego modelu różni się od wag runu w historii.")
        if context.expected_sha256 and context.expected_sha256 != digest:
            raise ValueError("Konflikt oczekiwanego SHA wskazanego modelu.")
    output = run.get("output_checkpoint_snapshot") or {}
    recorded = output.get("best_checkpoint_sha256") or (output.get("best") or {}).get("sha256")
    if recorded and recorded != digest:
        raise ValueError("Konflikt SHA z historycznym snapshotem checkpointu.")
    if expected_sha256 and expected_sha256 != digest:
        raise ValueError("Popraw SHA: wpisana wartość nie odpowiada bajtom checkpointu wybranego runu.")
    return dict(role=role, run_id=run_id, history_path=str(history), checkpoint=str(checkpoint),
                sha256=digest, expected_sha256=(context.expected_sha256 if context else "") or recorded or "",
                history_sha256=calibration.file_sha256(history), label=run_label(run_id, run, role))


def assert_manifest_context(payload, context, *, mode=None):
    config = payload["config"]
    if (payload["role"], config["run_id"], str(Path(config["history_path"]).resolve())) != (
            context.role, context.run_id, context.history_path):
        raise ValueError("Manifest dotyczy innego modelu, runu lub historii; nie przypisano do slotu.")
    if payload["preprocessing"]["width"] != context.image_size or payload["preprocessing"]["height"] != context.image_size:
        raise ValueError("Konflikt imgsz manifestu i slotu; przygotuj manifest ponownie.")
    for key in ("confidence_threshold", "iou_threshold"):
        if key in config and float(config[key]) != getattr(context, key):
            raise ValueError("Konflikt " + key + " manifestu i slotu.")
    if context.expected_sha256 and payload["checkpoint"]["sha256"] != context.expected_sha256:
        raise ValueError("Konflikt SHA manifestu i wskazanego modelu.")
    actual_mode = "exploratory" if payload["schema"] in {"alpr.int8_calibration.exploratory.v1", "alpr.int8_calibration.exploratory.v2"} else "strict"
    if mode is not None and actual_mode != mode:
        raise ValueError("Konflikt trybu manifestu i slotu. Wybierz zgodny tryb i otwórz kalibrację ponownie.")
