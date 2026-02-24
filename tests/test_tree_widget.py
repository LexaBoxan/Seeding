import os

from PyQt5.QtWidgets import QApplication

from seeding.ui.tree_widget import LayerTreeWidget


def _ensure_offscreen_qt() -> tuple[QApplication, bool]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    created = app is None
    if app is None:
        app = QApplication([])
    return app, created


def test_apply_filter_uses_confidence_metadata():
    app, created = _ensure_offscreen_qt()

    tree = LayerTreeWidget()
    root = tree.add_root_item("image_1", "source", 0, "original", None)
    low = tree.add_child_item(
        root,
        "Seeding1",
        "без числа в описании",
        0,
        0,
        "seeding",
        None,
        confidence=0.2,
    )
    high = tree.add_child_item(
        root,
        "Seeding2",
        "тоже без confidence в тексте",
        0,
        1,
        "seeding",
        None,
        confidence=0.9,
    )

    tree.apply_filter(min_confidence=0.5)

    assert low.isHidden()
    assert not high.isHidden()

    tree.deleteLater()
    if created:
        app.quit()


def test_apply_filter_text_fallback_without_metadata():
    app, created = _ensure_offscreen_qt()

    tree = LayerTreeWidget()
    root = tree.add_root_item("image_1", "source", 0, "original", None)
    child = tree.add_child_item(
        root,
        "Seeding1",
        "Уверенность: 0.65",
        0,
        0,
        "seeding",
        None,
    )

    tree.apply_filter(min_confidence=0.7)
    assert child.isHidden()

    tree.apply_filter(min_confidence=0.6)
    assert not child.isHidden()

    tree.deleteLater()
    if created:
        app.quit()


def test_tree_widget_headers_are_localized():
    app, created = _ensure_offscreen_qt()

    tree = LayerTreeWidget()
    tree.set_language("en")

    assert tree.headerItem().text(0) == "Name"
    assert tree.headerItem().text(1) == "Description"

    tree.set_language("ru")
    assert tree.headerItem().text(0) == "Название"
    assert tree.headerItem().text(1) == "Описание"

    tree.deleteLater()
    if created:
        app.quit()
