import os

import numpy as np
from PyQt5.QtWidgets import QApplication

from seeding.ui.statistics_panel import StatisticsPanel, StatisticsSummary
from seeding.ui.thumbnails_panel import ThumbnailsPanel


def _ensure_offscreen_qt() -> tuple[QApplication, bool]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    created = app is None
    if app is None:
        app = QApplication([])
    return app, created


def test_statistics_panel_localizes_labels_and_export_button():
    app, created = _ensure_offscreen_qt()

    panel = StatisticsPanel()
    panel.set_summary(
        StatisticsSummary(
            pages_count=2,
            objects_count=3,
            avg_confidence=0.75,
            min_area=100,
            max_area=300,
            histogram=(1, 0, 1, 0, 1),
        )
    )
    panel.set_language("en")

    assert panel.pages_label.text().startswith("Pages:")
    assert panel.objects_label.text().startswith("Objects:")
    assert "Average confidence" in panel.avg_conf_label.text()
    assert panel.export_button.text() == "Export statistics to CSV"

    panel.close()
    if created:
        app.quit()


def test_thumbnails_panel_localizes_tooltip():
    app, created = _ensure_offscreen_qt()

    panel = ThumbnailsPanel()
    panel.set_language("en")
    panel.set_images([np.zeros((16, 16, 3), dtype=np.uint8)])

    item = panel.list_widget.item(0)
    assert item.toolTip() == "Image 1"

    panel.set_language("ru")
    panel.set_images([np.zeros((16, 16, 3), dtype=np.uint8)])
    item = panel.list_widget.item(0)
    assert item.toolTip() == "Изображение 1"

    panel.close()
    if created:
        app.quit()
