#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generowanie raportu treningu (HTML) + eksport wykresów Ultralytics do folderu runa.

Wynik:
- <run_dir>/plots/  (kopie wykresów z train/)
- <run_dir>/training_report.html (samodzielny HTML z osadzonymi obrazkami)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import base64
import csv
import mimetypes
import shutil


class TrainingReportGenerator:
    DEFAULT_PLOT_CANDIDATES = [
        "results.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "labels.jpg",
        "labels.png",
    ]

    @staticmethod
    def _to_float(value) -> float | None:
        text = str(value or "").strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return None

    @classmethod
    def _read_results_csv(cls, csv_path: Path) -> List[Dict[str, str]]:
        if not csv_path.exists() or not csv_path.is_file():
            return []
        rows: List[Dict[str, str]] = []
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if isinstance(row, dict):
                        rows.append({str(k or "").strip(): str(v or "").strip() for k, v in row.items()})
        except Exception:
            return []
        return rows

    @classmethod
    def _draw_csv_line_chart_matplotlib(
        cls,
        rows: List[Dict[str, str]],
        out_path: Path,
        *,
        title: str,
        series_specs: List[tuple[str, tuple[str, ...], str]],
        parameter_legend: List[str] | None = None,
        width: int = 1800,
        height: int = 1050,
    ) -> Path | None:
        if not rows:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception:
            return None

        epochs: list[float] = []
        for idx, row in enumerate(rows):
            epoch = cls._to_float(row.get("epoch") or row.get("Epoch"))
            epochs.append(epoch if epoch is not None else float(idx + 1))

        series: list[tuple[str, list[float], list[float], str]] = []
        for label, keys, color in series_specs:
            xs: list[float] = []
            ys: list[float] = []
            for epoch, row in zip(epochs, rows):
                value = None
                for key in keys:
                    value = cls._to_float(row.get(key))
                    if value is not None:
                        break
                if value is not None:
                    xs.append(float(epoch))
                    ys.append(float(value))
            if xs and ys:
                series.append((label, xs, ys, color))
        if not series:
            return None

        dpi = 180
        fig_w = max(8.0, float(width) / dpi)
        fig_h = max(5.0, float(height) / dpi)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.patch.set_facecolor("#17212b")
        ax.set_facecolor("#1f2933")
        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.88,
            bottom=0.34 if parameter_legend else 0.23,
        )

        for label, xs, ys, color in series:
            ax.plot(xs, ys, label=label, color=color, linewidth=2.4, marker="o", markersize=3.8)

        ax.set_title(title, color="#e5e7eb", fontsize=15, fontweight="bold", pad=16)
        ax.set_xlabel("Epoka", color="#cbd5e1", fontsize=11)
        ax.set_ylabel("Wartość", color="#cbd5e1", fontsize=11)
        ax.grid(True, color="#334155", linewidth=0.8, alpha=0.65)
        ax.tick_params(colors="#cbd5e1", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#8fd19e")
            spine.set_linewidth(1.2)

        legend = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.11),
            ncol=min(4, max(1, len(series))),
            frameon=True,
            fontsize=9,
        )
        legend.get_frame().set_facecolor("#1f2933")
        legend.get_frame().set_edgecolor("#334155")
        for text in legend.get_texts():
            text.set_color("#e5e7eb")

        if parameter_legend:
            fig.text(
                0.02,
                0.055,
                "Legenda parametrów\n" + "\n".join(f"• {line}" for line in parameter_legend),
                color="#dbeafe",
                fontsize=8.8,
                va="bottom",
                ha="left",
                linespacing=1.32,
                bbox={
                    "boxstyle": "round,pad=0.65,rounding_size=0.12",
                    "facecolor": "#111827",
                    "edgecolor": "#4ade80",
                    "linewidth": 1.05,
                    "alpha": 0.96,
                },
            )

        fig.text(
            0.01,
            0.01,
            "Czytelny wykres wygenerowany przez aplikację z results.csv. Surowe wykresy Ultralytics pozostają w folderze runu.",
            color="#94a3b8",
            fontsize=9,
        )
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
            plt.close(fig)
            return out_path
        except Exception:
            try:
                plt.close(fig)
            except Exception:
                pass
            return None

    @classmethod
    def _draw_csv_line_chart(
        cls,
        rows: List[Dict[str, str]],
        out_path: Path,
        *,
        title: str,
        series_specs: List[tuple[str, tuple[str, ...], str]],
        width: int = 1400,
        height: int = 820,
    ) -> Path | None:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return None

        if not rows:
            return None

        epochs: list[float] = []
        for idx, row in enumerate(rows):
            epoch = cls._to_float(row.get("epoch") or row.get("Epoch"))
            epochs.append(epoch if epoch is not None else float(idx + 1))

        series: list[tuple[str, list[tuple[float, float]], str]] = []
        for label, keys, color in series_specs:
            points: list[tuple[float, float]] = []
            for epoch, row in zip(epochs, rows):
                value = None
                for key in keys:
                    value = cls._to_float(row.get(key))
                    if value is not None:
                        break
                if value is not None:
                    points.append((float(epoch), float(value)))
            if points:
                series.append((label, points, color))

        if not series:
            return None

        all_x = [x for _label, points, _color in series for x, _y in points]
        all_y = [y for _label, points, _color in series for _x, y in points]
        if not all_x or not all_y:
            return None

        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        if min_x == max_x:
            min_x -= 0.5
            max_x += 0.5
        if min_y == max_y:
            pad = max(0.05, abs(min_y) * 0.1)
            min_y -= pad
            max_y += pad
        y_pad = max(0.0001, (max_y - min_y) * 0.08)
        min_y -= y_pad
        max_y += y_pad

        bg = "#17212b"
        panel = "#1f2933"
        grid = "#334155"
        axis = "#8fd19e"
        text = "#e5e7eb"
        muted = "#94a3b8"

        image = Image.new("RGB", (int(width), int(height)), bg)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        left = 92
        right = 42
        top = 74
        bottom = 84
        chart_w = width - left - right
        chart_h = height - top - bottom

        draw.rectangle((left, top, width - right, height - bottom), fill=panel, outline=grid, width=1)
        draw.text((left, 28), title, fill=text, font=font)
        draw.text((left, 50), "Awaryjny wykres z results.csv, gdy Ultralytics nie zapisał własnego PNG.", fill=muted, font=font)

        def px(x_value: float) -> float:
            return left + ((x_value - min_x) / (max_x - min_x)) * chart_w

        def py(y_value: float) -> float:
            return top + (1.0 - ((y_value - min_y) / (max_y - min_y))) * chart_h

        for i in range(6):
            ratio = i / 5
            y = top + ratio * chart_h
            value = max_y - ratio * (max_y - min_y)
            draw.line((left, y, width - right, y), fill=grid, width=1)
            draw.text((12, y - 7), f"{value:.4g}", fill=muted, font=font)

        for i in range(6):
            ratio = i / 5
            x = left + ratio * chart_w
            value = min_x + ratio * (max_x - min_x)
            draw.line((x, top, x, height - bottom), fill="#263544", width=1)
            draw.text((x - 14, height - bottom + 12), f"{value:.0f}", fill=muted, font=font)

        draw.line((left, height - bottom, width - right, height - bottom), fill=axis, width=2)
        draw.line((left, top, left, height - bottom), fill=axis, width=2)
        draw.text((width // 2 - 24, height - 34), "epoka", fill=muted, font=font)

        legend_x = left
        legend_y = height - 58
        for label, points, color in series:
            rendered = [(px(x), py(y)) for x, y in points]
            if len(rendered) == 1:
                x, y = rendered[0]
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=color)
            else:
                draw.line(rendered, fill=color, width=3, joint="curve")
                for x, y in rendered:
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=color)

            draw.rectangle((legend_x, legend_y + 4, legend_x + 14, legend_y + 14), fill=color)
            draw.text((legend_x + 20, legend_y), label, fill=text, font=font)
            legend_x += max(130, 8 * len(label) + 46)
            if legend_x > width - 240:
                legend_x = left
                legend_y += 18

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path)
            return out_path
        except Exception:
            return None

    @classmethod
    def generate_csv_charts(cls, train_dir: Path, out_dir: Path) -> List[Path]:
        rows = cls._read_results_csv(Path(train_dir) / "results.csv")
        if not rows:
            return []

        out_dir.mkdir(parents=True, exist_ok=True)
        chart_specs = [
            (
                "results_metrics_from_csv.png",
                "Metryki walidacyjne z results.csv",
                [
                    ("precision(B)", ("metrics/precision(B)", "precision", "box_precision"), "#60a5fa"),
                    ("recall(B)", ("metrics/recall(B)", "recall", "box_recall"), "#34d399"),
                    ("mAP50(B)", ("metrics/mAP50(B)", "metrics/mAP50", "map50", "box_map50"), "#fbbf24"),
                    ("mAP50-95(B)", ("metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95", "box_map50_95"), "#f87171"),
                    ("precision(P)", ("metrics/precision(P)", "pose_precision"), "#a78bfa"),
                    ("recall(P)", ("metrics/recall(P)", "pose_recall"), "#2dd4bf"),
                    ("mAP50(P)", ("metrics/mAP50(P)", "pose_map50"), "#fb7185"),
                    ("mAP50-95(P)", ("metrics/mAP50-95(P)", "pose_map50_95"), "#c084fc"),
                ],
                [
                    "B = ramki obiektów; P = punkty/narożniki w modelu pose.",
                    "precision = jaki odsetek predykcji był trafny; recall = ile obiektów z etykiet model odnalazł.",
                    "mAP50 = łagodniejsza ocena trafienia przy IoU 0.50; mAP50-95 = surowsza, główna miara jakości.",
                    "Dla decyzji projektowej patrz przede wszystkim na stabilny trend mAP50-95 oraz brak spadku recall.",
                ],
            ),
            (
                "train_val_losses_from_csv.png",
                "Straty treningowe i walidacyjne z results.csv",
                [
                    ("train box", ("train/box_loss",), "#60a5fa"),
                    ("train cls", ("train/cls_loss",), "#34d399"),
                    ("train dfl", ("train/dfl_loss",), "#fbbf24"),
                    ("train pose", ("train/pose_loss",), "#a78bfa"),
                    ("val box", ("val/box_loss",), "#f87171"),
                    ("val cls", ("val/cls_loss",), "#fb7185"),
                    ("val dfl", ("val/dfl_loss",), "#2dd4bf"),
                    ("val pose", ("val/pose_loss",), "#c084fc"),
                ],
                [
                    "train = błąd na danych treningowych; val = błąd na danych walidacyjnych.",
                    "box = położenie ramki; cls = klasa; dfl = granice ramki; pose = punkty/narożniki.",
                    "Niżej zwykle znaczy lepiej. Rosnący val przy malejącym train sugeruje przeuczenie.",
                    "Nagłe skoki loss warto zestawić z RAM/VRAM i zmianami learning rate.",
                ],
            ),
            (
                "train_learning_rate_from_csv.png",
                "Learning rate z results.csv",
                [
                    ("lr/pg0", ("lr/pg0",), "#60a5fa"),
                    ("lr/pg1", ("lr/pg1",), "#34d399"),
                    ("lr/pg2", ("lr/pg2",), "#fbbf24"),
                ],
                [
                    "lr = współczynnik uczenia; pg0/pg1/pg2 = grupy parametrów optymalizatora.",
                    "To harmonogram treningu, a nie bezpośrednia jakość modelu.",
                    "Zmiany learning rate pomagają wyjaśnić tempo poprawy metryk albo nagłe skoki loss.",
                ],
            ),
        ]

        generated: List[Path] = []
        for filename, title, specs, parameter_legend in chart_specs:
            rendered = cls._draw_csv_line_chart_matplotlib(
                rows,
                out_dir / filename,
                title=title,
                series_specs=specs,
                parameter_legend=parameter_legend,
            )
            if rendered is None:
                rendered = cls._draw_csv_line_chart(rows, out_dir / filename, title=title, series_specs=specs)
            if rendered is not None:
                generated.append(rendered)
        return generated

    @staticmethod
    def _img_to_data_uri(img_path: Path) -> str:
        mime, _ = mimetypes.guess_type(str(img_path))
        if not mime:
            mime = "image/png"
        data = img_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    @classmethod
    def find_plots(cls, directory: Path) -> List[Path]:
        """Wyszukuje wykresy w katalogu (train/ albo plots/)."""
        if not directory.exists():
            return []

        found: List[Path] = []

        for name in cls.DEFAULT_PLOT_CANDIDATES:
            p = directory / name
            if p.exists() and p.is_file():
                found.append(p)

        already = {p.name for p in found}
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for p in sorted(directory.glob(ext)):
                if p.is_file() and p.name not in already:
                    found.append(p)

        return found

    @classmethod
    def export_plots(cls, train_dir: Path, run_dir: Path) -> Path:
        """
        Kopiuje wykresy z train/ do run_dir/plots/.
        Zwraca ścieżkę do run_dir/plots.
        """
        plots = cls.find_plots(train_dir)
        out_dir = run_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        for p in plots:
            dst = out_dir / p.name
            try:
                shutil.copy2(p, dst)
            except Exception:
                try:
                    shutil.copy(p, dst)
                except Exception:
                    pass

        cls.generate_csv_charts(train_dir, out_dir)
        return out_dir

    @classmethod
    def generate_html(
        cls,
        run_dict: Dict,
        output_path: Path,
        plots_source_dir: Path,
        extra_info: Optional[Dict] = None,
    ) -> Path:
        """Generuje samodzielny raport HTML z wykresami (base64)."""
        plots = cls.find_plots(plots_source_dir)
        extra_info = extra_info or {}

        meta_rows = ""
        for k in [
            "id", "name", "status", "dataset_path", "base_model",
            "epochs", "batch_size", "img_size", "device",
            "current_epoch", "best_map50", "best_map50_95",
            "started_at", "finished_at", "paused_at",
            "best_weights", "last_weights", "error_message",
        ]:
            v = run_dict.get(k)
            if v not in (None, "", []):
                meta_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

        extra_rows = ""
        for k, v in extra_info.items():
            extra_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

        images_html = ""
        if plots:
            for p in plots:
                try:
                    uri = cls._img_to_data_uri(p)
                    images_html += f"""
                    <div class="card">
                      <div class="title">{p.name}</div>
                      <img src="{uri}" />
                    </div>
                    """
                except Exception as e:
                    images_html += f"<p>Nie udało się osadzić {p.name}: {e}</p>"
        else:
            images_html = "<p>Nie znaleziono wykresów. Upewnij się, że trening uruchamiasz z plots=True.</p>"

        html = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8"/>
  <title>Training Report - {run_dict.get('name','run')}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    h1 {{ margin: 0 0 8px 0; }}
    .sub {{ color: #555; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
    td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; }}
    .title {{ font-weight: bold; margin-bottom: 8px; }}
    img {{ width: 100%; height: auto; border: 1px solid #eee; }}
    @media (max-width: 1000px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Raport treningu</h1>
  <div class="sub">Wygenerowano: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>

  <h2>Metadane</h2>
  <table>
    {meta_rows}
    {extra_rows}
  </table>

  <h2>Wykresy (Ultralytics)</h2>
  <div class="grid">
    {images_html}
  </div>
</body>
</html>
"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return output_path
