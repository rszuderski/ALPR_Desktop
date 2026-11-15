"""Validate semantic classes before turning model boxes into plate polygons."""
from ..config import CONFIG


def plate_class_ids(names) -> set[int]:
    def normalize(name):
        return " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())

    mapping = names if isinstance(names, dict) else dict(enumerate(names or []))
    labels = {normalize(label) for label in CONFIG.PLATE_LABELS} | {"number plate", "licence plate"}
    result = {int(key) for key, name in mapping.items() if normalize(name) in labels}
    if not result:
        classes = ", ".join(str(name) for name in list(mapping.values())[:8]) or "brak informacji o klasach"
        raise ValueError(
            f"Model nie zawiera klasy tablicy rejestracyjnej (klasy: {classes}). "
            "Wybierz model tablic. Detekcje pojazdów lub innych obiektów nie mogą być zapisywane jako tablice."
        )
    return result


def plate_result_indices(result, model) -> set[int]:
    names = getattr(result, "names", None) or getattr(model, "names", None)
    allowed = plate_class_ids(names)
    classes = getattr(result.boxes, "cls", None)
    if classes is None:
        raise ValueError("Wynik modelu nie podaje klas detekcji; nie można rozpoznać tablic.")
    return {index for index, class_id in enumerate(classes.cpu().numpy()) if int(class_id) in allowed}
