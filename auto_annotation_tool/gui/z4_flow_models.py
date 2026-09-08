from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class TrainingSourceStats:
    train: int = 0
    val: int = 0
    test: int = 0
    total: int = 0
    labels: int = 0
    classes: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, raw: dict | None) -> "TrainingSourceStats":
        data = dict(raw or {})

        def _as_int(*keys: str) -> int:
            for key in keys:
                if key not in data:
                    continue
                raw_value = data.get(key, 0)
                if raw_value in (None, ""):
                    continue
                try:
                    return max(0, int(float(raw_value or 0)))
                except Exception:
                    continue
            return 0

        train = _as_int("train", "train_images")
        val = _as_int("val", "val_images")
        test = _as_int("test", "test_images")
        total = _as_int("total", "total_images") or train + val + test
        warnings_raw = data.get("warnings", ())
        if isinstance(warnings_raw, str):
            warnings = (warnings_raw,) if warnings_raw else ()
        else:
            try:
                warnings = tuple(str(item) for item in warnings_raw if str(item or "").strip())
            except Exception:
                warnings = ()
        return cls(
            train=train,
            val=val,
            test=test,
            total=total,
            labels=_as_int("labels", "label_count", "total_labels", "total_objects", "objects", "annotations"),
            classes=_as_int("classes", "class_count", "nc"),
            warnings=warnings,
        )

    def split_counts(self) -> dict[str, int]:
        return {
            "train": int(self.train),
            "val": int(self.val),
            "test": int(self.test),
            "total": int(self.total),
        }


@dataclass(frozen=True)
class TrainingSource:
    target: str = "char"
    kind: str = ""
    annotation_file: str = ""
    images_dir: str = ""
    dataset_dir: str = ""
    yaml_path: str = ""
    validated: bool = False
    stats: TrainingSourceStats = field(default_factory=TrainingSourceStats)
    provenance: str = ""
    source_stage: str = ""
    message: str = ""

    def has_dataset(self) -> bool:
        return bool(str(self.dataset_dir or self.yaml_path or "").strip())

    def display_path(self) -> str:
        return str(self.dataset_dir or self.yaml_path or self.annotation_file or self.images_dir or "").strip()


@dataclass(frozen=True)
class TrainingSourceValidationResult:
    ok: bool = False
    source: TrainingSource = field(default_factory=TrainingSource)
    message: str = ""
    reason: str = ""
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PlateXmlImagesSourceAdapter:
    provenance: str = "Źródło PZ1"
    source_stage: str = "Z4/PZ1"

    def _result(
        self,
        *,
        ok: bool,
        message: str,
        xml_path: Path | None = None,
        images_dir: Path | None = None,
        reason: str = "",
    ) -> TrainingSourceValidationResult:
        source = TrainingSource(
            target="plate",
            kind="annotation_xml_images",
            annotation_file=str(xml_path or "").strip(),
            images_dir=str(images_dir or "").strip(),
            validated=bool(ok),
            provenance=self.provenance,
            source_stage=self.source_stage,
            message=str(message or "").strip(),
        )
        raw = {}
        if xml_path is not None:
            raw["xml"] = xml_path
        if images_dir is not None:
            raw["images_dir"] = images_dir
        return TrainingSourceValidationResult(
            ok=bool(ok),
            source=source,
            message=str(message or "").strip(),
            reason=str(reason or "").strip(),
            raw=raw,
        )

    def validate(self, xml_raw: str | Path | None, images_raw: str | Path | None) -> TrainingSourceValidationResult:
        xml_text = str(xml_raw or "").strip()
        images_text = str(images_raw or "").strip()

        if not xml_text:
            return self._result(
                ok=False,
                message="W trybie budowy z XML wymagane są dwa zgodne źródła: plik XML i folder obrazów.",
                reason="missing_xml",
            )
        try:
            xml_path = Path(xml_text)
        except Exception:
            return self._result(
                ok=False,
                message="Nie udało się odczytać ścieżki pliku XML.",
                reason="invalid_xml_path",
            )
        if not xml_path.exists() or not xml_path.is_file() or xml_path.suffix.lower() != ".xml":
            return self._result(
                ok=False,
                message="Wskaż istniejący plik anotacji XML.",
                xml_path=xml_path,
                reason="missing_xml_file",
            )

        if not images_text:
            return self._result(
                ok=False,
                message="Wskaż folder obrazów dla tego pliku XML.",
                xml_path=xml_path,
                reason="missing_images_dir",
            )
        try:
            images_dir = Path(images_text)
        except Exception:
            return self._result(
                ok=False,
                message="Nie udało się odczytać ścieżki folderu obrazów.",
                xml_path=xml_path,
                reason="invalid_images_path",
            )
        if not images_dir.exists() or not images_dir.is_dir():
            return self._result(
                ok=False,
                message="Wskaż istniejący folder obrazów.",
                xml_path=xml_path,
                images_dir=images_dir,
                reason="missing_images_dir",
            )

        return self._result(
            ok=True,
            message="Gotowy do utworzenia wariantu datasetu treningowego.",
            xml_path=xml_path,
            images_dir=images_dir,
        )


@dataclass(frozen=True)
class CharYoloDatasetSourceAdapter:
    validator: Callable[..., dict]
    provenance: str = "Źródło PZ1"
    source_stage: str = "Z3/PZ2"

    def validate(self, source_raw: str | Path | None) -> TrainingSourceValidationResult:
        source_text = str(source_raw or "").strip()
        if not source_text:
            source = TrainingSource(
                target="char",
                kind="yolo_dataset",
                provenance=self.provenance,
                source_stage=self.source_stage,
                message=(
                    "Wskaż dataset znaków: katalog z obrazami i etykietami albo plik data.yaml."
                ),
            )
            return TrainingSourceValidationResult(
                ok=False,
                source=source,
                message=source.message,
                reason="missing_dataset",
            )

        try:
            validation = dict(self.validator(source_text, resolve_nested_dataset=True) or {})
        except TypeError:
            validation = dict(self.validator(source_text) or {})
        except Exception as exc:
            source = TrainingSource(
                target="char",
                kind="yolo_dataset",
                dataset_dir=source_text,
                provenance=self.provenance,
                source_stage=self.source_stage,
                message=f"Nie udało się sprawdzić datasetu znaków: {exc}",
            )
            return TrainingSourceValidationResult(
                ok=False,
                source=source,
                message=source.message,
                reason="validation_error",
            )

        src = validation.get("src") or source_text
        try:
            src_path = Path(src)
            dataset_dir = src_path.parent if src_path.is_file() and src_path.name.lower() == "data.yaml" else src_path
            yaml_path = src_path if src_path.is_file() and src_path.name.lower() == "data.yaml" else dataset_dir / "data.yaml"
        except Exception:
            dataset_dir = Path(str(src))
            yaml_path = dataset_dir / "data.yaml"

        stats = TrainingSourceStats.from_mapping(dict(validation.get("stats") or {}))
        message = str(
            validation.get("message")
            or "Gotowy do utworzenia wariantu datasetu treningowego."
        ).strip()
        ok = bool(validation.get("ok"))
        source = TrainingSource(
            target="char",
            kind="yolo_dataset",
            dataset_dir=str(dataset_dir),
            yaml_path=str(yaml_path),
            validated=ok,
            stats=stats,
            provenance=self.provenance,
            source_stage=self.source_stage,
            message=message,
        )
        return TrainingSourceValidationResult(
            ok=ok,
            source=source,
            message=message,
            reason=str(validation.get("reason") or "").strip(),
            raw=validation,
        )
