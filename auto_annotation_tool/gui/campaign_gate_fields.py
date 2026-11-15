"""Presentation of graph fields; gate rules and action tags remain unchanged."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class GateFieldCopy:
    caption: str
    primary: str
    detail: str = ""
    tone: str = ""


_STATUS = {
    "WYBIERZ": "Wybierz bramkę",
    "OTWARTA": "Gotowa do zatwierdzenia",
    "ZAMKNIĘTA": "Warunki niespełnione",
    "PRZERWANE": "Praca przerwana",
    "WYBIERZ WYNIK": "Wybierz wynik treningu",
    "DO KONTROLI": "Potrzebna kontrola",
}
_ACTIONS = {
    "KONTROLUJ AT": "Sprawdź anotacje tablic",
    "UZUPEŁNIJ ZASOBY": "Uzupełnij zasoby",
    "WYBIERZ TOR PRACY": "Wybierz tor pracy",
    "PRACUJ W Z2": "Otwórz pracę w Z2",
    "WZNÓW PRACĘ": "Wznów pracę",
    "WZNÓW PZ2": "Wznów ramki znaków w PZ2",
    "WZNÓW PZ3": "Wznów dataset w PZ3",
    "ROZLICZ [OK]": "Rozlicz zatwierdzone obrazy",
    "UTWÓRZ AZ W PZ3": "Utwórz dataset znaków w PZ3",
    "NAPRAW: PZ2 -> PZ3": "Napraw ramki i dataset",
    "START: PZ2 -> PZ3": "Rozpocznij pracę nad znakami",
    "PRACA: PZ2 -> PZ3": "Otwórz pracę nad znakami",
    "GOTOWE Z POPRZ. ITERACJI": "Gotowe z poprzednich iteracji",
    "WYBIERZ WYNIK": "Wybierz wynik treningu",
}
_RESOURCES = {
    "O": "Obrazy", "AT": "Anotacje tablic", "AZ": "Anotacje znaków",
    "MT": "Model tablic", "MZ": "Model znaków", "MODEL": "Model",
    "MODEL TABLIC": "Model tablic", "MODEL ZNAKÓW": "Model znaków",
    "DATASET ZNAKÓW": "Dataset znaków", "BEZ TRENINGU": "Bez treningu",
}


def _details(text):
    """Expand only known UI markers; preserve filenames and unfamiliar values."""
    pattern = r"\b(" + "|".join(re.escape(key) for key in sorted(_RESOURCES, key=len, reverse=True)) + r")\(IT(\d+)\)"
    text = re.sub(pattern, lambda m: f"{_RESOURCES[m[1]]} · iteracja {m[2]}", text)
    text = re.sub(r"\bz IT(\d+)\b", r"z iteracji \1", text)
    text = re.sub(r"\bw IT(\d+)\b", r"w iteracji \1", text)
    text = re.sub(r"\bIT(\d+)\b", r"iteracja \1", text)
    return re.sub(r"[ \t]{2,}", "\n", text).replace(" | ", "\n").replace(" -> ", " → ").strip()


def gate_field_copy(label, value, *, enabled=False, approve_label="ZATWIERDŹ"):
    value = str(value or "").strip()
    upper = value.upper()
    if label == "BRAMKA":
        tone = "success" if upper == "OTWARTA" else "warning" if upper not in {"", "WYBIERZ"} else ""
        return GateFieldCopy("Stan bramki", _STATUS.get(upper, value or "Brak statusu"), tone=tone)
    if label == "ZATWIERDŹ":
        _heading, _sep, detail = str(approve_label or "").partition("\n")
        return GateFieldCopy("", "Zatwierdź bramkę", _details(detail) if enabled else "Po spełnieniu wymagań",
                             "success" if enabled else "")
    if label == "ZASOBY":
        if upper == "WYBIERZ":
            return GateFieldCopy("Zasoby", "Wybierz zasoby")
        if upper == "OK":
            return GateFieldCopy("Zasoby", "Komplet zasobów", tone="success")
        if upper.startswith("OK:"):
            return GateFieldCopy("Zasoby", "Zasoby gotowe", _details(value.split(":", 1)[1]), "success")
        if upper.startswith("BRAK:"):
            missing = value.split(":", 1)[1].strip()
            missing = {"tablice [ok]": "Zatwierdzone tablice", "obrazy": "Obrazy",
                       "znaki": "Anotacje znaków", "dataset znaków": "Dataset znaków",
                       "wynik treningu": "Wynik treningu"}.get(missing.casefold(), missing)
            return GateFieldCopy("Zasoby", "Brakuje zasobów", missing, "warning")
        if upper == "KOMPLET O-AT":
            return GateFieldCopy("Zasoby", "Obrazy i anotacje tablic", "Komplet O–AT do kontroli", "warning")
        if upper.startswith("AT Z POPRZEDNICH ITERACJI"):
            return GateFieldCopy("Zasoby", "Anotacje z poprzednich iteracji",
                                 _details(value[len("AT z poprzednich iteracji"):].lstrip(" |")))
        primary, separator, detail = value.partition(" | ")
        return GateFieldCopy("Zasoby", primary or ("Sprawdź zasoby" if enabled else "Nie wybrano"),
                             _details(detail) if separator else "")
    if label == "PRACA":
        if upper == "WYKONANE":
            return GateFieldCopy(
                "Praca",
                "Praca wykonana",
                "Dataset znaków jest gotowy",
                "success",
            )
        if upper.startswith("PRZERWANE"):
            count = re.search(r"\+(\d+)\s+OK", upper)
            detail = f"+{count[1]} obrazów oznaczonych OK" if count else ""
            return GateFieldCopy("Praca", "Praca przerwana", detail, "warning")
        match = re.fullmatch(r"ZATWIERDŹ\s*->\s*(Z\d+)", upper)
        if match:
            return GateFieldCopy("Praca", "Gotowe do przejścia", f"Zatwierdź bramkę → {match[1]}", "success")
        match = re.fullmatch(r"DODAJ \+(\d+) W Z2", upper)
        if match:
            return GateFieldCopy("Praca", "Możesz dodać obrazy", f"Pozostało: {match[1]} · praca w Z2")
        match = re.fullmatch(r"BEZ PRZYROSTU IT(\d+)", upper)
        if match:
            return GateFieldCopy("Praca", "Brak nowych wyników", f"Iteracja {match[1]}")
        if upper in _ACTIONS:
            detail = "PZ2 → PZ3" if "PZ2 -> PZ3" in upper else ""
            return GateFieldCopy("Praca", _ACTIONS[upper], detail)
        primary, separator, detail = value.partition("\n")
        return GateFieldCopy("Praca", primary or ("Otwórz narzędzia" if enabled else "Niedostępne"),
                             _details(detail) if separator else "")
    return GateFieldCopy(str(label), value)


def text_height(canvas, text, font, width):
    """Use Tk's real word wrapping, cached across redraws of the same text."""
    if not text:
        return 0.0
    cache = getattr(canvas, "_gate_field_text_heights", None)
    if cache is None:
        cache = canvas._gate_field_text_heights = {}
    key = (text, tuple(font), round(float(width), 2), round(canvas.winfo_fpixels("1i"), 2))
    if key not in cache:
        if len(cache) >= 384:
            cache.clear()
        item = canvas.create_text(-10000, -10000, text=text, font=font, width=max(1, width), anchor="nw")
        try:
            bounds = canvas.bbox(item)
            cache[key] = float(bounds[3] - bounds[1]) if bounds else 0.0
        finally:
            canvas.delete(item)
    return cache[key]


def field_layout(canvas, copy, *, width, scale, caption_font, primary_font, detail_font):
    text_width = max(1, width - 36 * scale)
    y = 8 * scale
    parts = []
    for role, text, font in (("caption", copy.caption, caption_font),
                              ("primary", copy.primary, primary_font),
                              ("detail", copy.detail, detail_font)):
        if text:
            parts.append((role, text, y))
            y += text_height(canvas, text, font, text_width) + 3 * scale
    return {"parts": parts, "height": y + 5 * scale, "text_width": text_width}


def draw_field(canvas, copy, layout, *, x, y, width, scale, zoom, fonts, colors, tags, enabled=False):
    for role, text, offset in layout["parts"]:
        role_tags = tags if role == "primary" else tuple(tag for tag in tags if tag != "gate_approve_blink_text")
        canvas.create_text(x + 12 * scale, y + offset, text=text, anchor="nw", justify="left",
                           width=max(1, layout["text_width"] * zoom), font=fonts[role],
                           fill=colors[role], tags=(*role_tags, "gate_field_text", f"gate_field_{role}"))
    if enabled:
        canvas.create_text(x + width - 11 * scale, y + layout["height"] / 2, text="›",
                           anchor="center", font=fonts["primary"], fill=colors["primary"],
                           tags=(*tags, "gate_field_affordance"))
