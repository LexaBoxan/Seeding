"""Панель статистики по найденным объектам."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from seeding.models import OriginalImage

from .i18n import tr


@dataclass(frozen=True)
class StatisticsSummary:
    """Агрегированная статистика по проекту."""

    pages_count: int = 0
    objects_count: int = 0
    seedlings_count: int = 0
    inflorescences_count: int = 0
    stems_count: int = 0
    roots_count: int = 0
    other_parts_count: int = 0
    avg_confidence: float = 0.0
    min_area: int = 0
    max_area: int = 0
    histogram: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)


class StatisticsPanel(QWidget):
    """Виджет с краткой статистикой и экспортом в CSV."""

    export_csv_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        """Инициализирует панель статистики и подготавливает её к обновлению."""
        super().__init__(parent)
        self._language = "ru"
        self._summary = StatisticsSummary()
        self._build_ui()
        self.set_language(self._language)

    def _build_ui(self) -> None:
        """Создаёт структуру панели: метрики, гистограмму и кнопку экспорта."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        self.pages_label = QLabel("")
        self.objects_label = QLabel("")
        self.avg_conf_label = QLabel("")
        self.seedlings_label = QLabel("")
        self.inflorescences_label = QLabel("")
        self.stems_label = QLabel("")
        self.roots_label = QLabel("")
        self.other_parts_label = QLabel("")
        self.min_area_label = QLabel("")
        self.max_area_label = QLabel("")

        grid.addWidget(self.pages_label, 0, 0)
        grid.addWidget(self.objects_label, 0, 1)
        grid.addWidget(self.avg_conf_label, 1, 0, 1, 2)
        grid.addWidget(self.seedlings_label, 2, 0)
        grid.addWidget(self.inflorescences_label, 2, 1)
        grid.addWidget(self.stems_label, 3, 0)
        grid.addWidget(self.roots_label, 3, 1)
        grid.addWidget(self.other_parts_label, 4, 0, 1, 2)
        grid.addWidget(self.min_area_label, 5, 0)
        grid.addWidget(self.max_area_label, 5, 1)
        layout.addLayout(grid)

        self.hist_bars: list[QProgressBar] = []
        labels = [
            "0.00-0.20",
            "0.20-0.40",
            "0.40-0.60",
            "0.60-0.80",
            "0.80-1.00",
        ]
        for label in labels:
            row = QGridLayout()
            txt = QLabel(label)
            bar = QProgressBar(self)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%v")
            row.addWidget(txt, 0, 0)
            row.addWidget(bar, 0, 1)
            layout.addLayout(row)
            self.hist_bars.append(bar)

        self.export_button = QPushButton("", self)
        self.export_button.clicked.connect(self.export_csv_requested.emit)
        layout.addWidget(self.export_button)
        layout.addStretch()

    def set_language(self, language: str) -> None:
        """Обновляет локализуемые подписи панели статистики."""
        self._language = language
        self.export_button.setText(
            tr(language, "stats_export_csv", "Экспорт статистики в CSV")
        )
        self.set_summary(self._summary)

    @staticmethod
    def build_summary(data: OriginalImage) -> StatisticsSummary:
        """Вычисляет сводную статистику по состоянию проекта."""
        pages_count = len(data.images)
        objects: list = []
        if data.class_object_image:
            for page_objects in data.class_object_image:
                objects.extend(page_objects)

        objects_count = len(objects)
        seedlings_count = objects_count
        inflorescences_count = 0
        stems_count = 0
        roots_count = 0
        other_parts_count = 0
        if objects_count == 0:
            return StatisticsSummary(pages_count=pages_count)

        confidences = [float(obj.confidence) for obj in objects]
        areas: list[int] = []
        for obj in objects:
            if not obj.bbox:
                pass
            else:
                x1, y1, x2, y2 = obj.bbox
                area = max(0, int(x2 - x1)) * max(0, int(y2 - y1))
                areas.append(area)
            if not obj.image_all_class:
                continue
            for part in obj.image_all_class:
                class_key = StatisticsPanel._normalize_part_name(part.class_name)
                if class_key == "inflorescence":
                    inflorescences_count += 1
                elif class_key == "stem":
                    stems_count += 1
                elif class_key == "root":
                    roots_count += 1
                else:
                    other_parts_count += 1

        hist, _ = np.histogram(
            np.array(confidences, dtype=np.float32),
            bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.00001],
        )
        return StatisticsSummary(
            pages_count=pages_count,
            objects_count=objects_count,
            seedlings_count=seedlings_count,
            inflorescences_count=inflorescences_count,
            stems_count=stems_count,
            roots_count=roots_count,
            other_parts_count=other_parts_count,
            avg_confidence=float(np.mean(confidences)),
            min_area=min(areas) if areas else 0,
            max_area=max(areas) if areas else 0,
            histogram=tuple(int(v) for v in hist.tolist()),
        )

    @staticmethod
    def _normalize_part_name(name: str | None) -> str:
        """Приводит имя части к одной из категорий статистики."""
        value = (name or "").strip().lower()
        if not value:
            return "other"
        if value in {"соцветие", "цветок", "flower", "inflorescence"}:
            return "inflorescence"
        if value in {"стебель", "stem"}:
            return "stem"
        if value in {"корень", "root"}:
            return "root"
        if value in {"сеянец", "seedling", "seeding"}:
            return "seedling"
        return "other"

    def set_summary(self, summary: StatisticsSummary) -> None:
        """Обновляет отображение сводной статистики."""
        self._summary = summary
        pages_title = tr(self._language, "stats_pages", "Страниц")
        objects_title = tr(self._language, "stats_objects", "Объектов")
        avg_title = tr(
            self._language,
            "stats_avg_confidence",
            "Средняя уверенность",
        )
        min_area_title = tr(self._language, "stats_min_area", "Мин. площадь")
        max_area_title = tr(self._language, "stats_max_area", "Макс. площадь")
        seedlings_title = tr(self._language, "stats_seedlings", "Сеянцев")
        inflorescences_title = tr(
            self._language,
            "stats_inflorescences",
            "Соцветий",
        )
        stems_title = tr(self._language, "stats_stems", "Стеблей")
        roots_title = tr(self._language, "stats_roots", "Корней")
        other_parts_title = tr(self._language, "stats_other_parts", "Прочих частей")
        self.pages_label.setText(
            f"{pages_title}: {summary.pages_count}"
        )
        self.objects_label.setText(
            f"{objects_title}: {summary.objects_count}"
        )
        self.avg_conf_label.setText(
            f"{avg_title}: "
            f"{summary.avg_confidence:.3f}"
        )
        self.seedlings_label.setText(
            f"{seedlings_title}: {summary.seedlings_count}"
        )
        self.inflorescences_label.setText(
            f"{inflorescences_title}: {summary.inflorescences_count}"
        )
        self.stems_label.setText(f"{stems_title}: {summary.stems_count}")
        self.roots_label.setText(f"{roots_title}: {summary.roots_count}")
        self.other_parts_label.setText(
            f"{other_parts_title}: {summary.other_parts_count}"
        )
        self.min_area_label.setText(
            f"{min_area_title}: {summary.min_area}"
        )
        self.max_area_label.setText(
            f"{max_area_title}: {summary.max_area}"
        )

        max_hist_value = max(summary.histogram) if summary.histogram else 0
        bar_max = max(1, max_hist_value)
        for idx, bar in enumerate(self.hist_bars):
            bar.setRange(0, bar_max)
            bar.setValue(summary.histogram[idx] if idx < 5 else 0)

    @staticmethod
    def export_summary_csv(
        path: str | Path,
        summary: StatisticsSummary,
        *,
        language: str = "ru",
    ) -> None:
        """Сохраняет сводную статистику в CSV-файл."""
        output = Path(path)
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=";")
            writer.writerow(
                [
                    tr(language, "stats_csv_metric", "Показатель"),
                    tr(language, "stats_csv_value", "Значение"),
                ]
            )
            writer.writerow(
                [tr(language, "stats_pages", "Страниц"), summary.pages_count]
            )
            writer.writerow(
                [
                    tr(language, "stats_objects", "Объектов"),
                    summary.objects_count,
                ]
            )
            writer.writerow(
                [
                    tr(language, "stats_seedlings", "Сеянцев"),
                    summary.seedlings_count,
                ]
            )
            writer.writerow(
                [
                    tr(language, "stats_inflorescences", "Соцветий"),
                    summary.inflorescences_count,
                ]
            )
            writer.writerow(
                [tr(language, "stats_stems", "Стеблей"), summary.stems_count]
            )
            writer.writerow(
                [tr(language, "stats_roots", "Корней"), summary.roots_count]
            )
            writer.writerow(
                [
                    tr(language, "stats_other_parts", "Прочих частей"),
                    summary.other_parts_count,
                ]
            )
            writer.writerow(
                [
                    tr(
                        language,
                        "stats_avg_confidence",
                        "Средняя уверенность",
                    ),
                    f"{summary.avg_confidence:.6f}",
                ]
            )
            writer.writerow(
                [
                    tr(language, "stats_min_area", "Мин. площадь"),
                    summary.min_area,
                ]
            )
            writer.writerow(
                [
                    tr(language, "stats_max_area", "Макс. площадь"),
                    summary.max_area,
                ]
            )
            writer.writerow([])
            writer.writerow(
                [
                    tr(
                        language,
                        "stats_csv_hist",
                        "Гистограмма уверенности",
                    ),
                    tr(language, "stats_objects", "Объектов"),
                ]
            )
            labels = [
                "0.00-0.20",
                "0.20-0.40",
                "0.40-0.60",
                "0.60-0.80",
                "0.80-1.00",
            ]
            for label, value in zip(labels, summary.histogram):
                writer.writerow([label, value])
