"""Централизованные стили интерфейса.

Тёмная тема с природными акцентами (зелёный) для приложения анализа сеянцев.
"""

# Цветовая палитра — сдержанные тона
COLORS = {
    "bg_dark": "#1a1d21",
    "bg_panel": "#23262b",
    "bg_elevated": "#2c3036",
    "bg_input": "#1e2226",
    "accent": "#4b5563",       # приглушённый серый вместо яркого зелёного
    "accent_hover": "#3b4048",
    "accent_muted": "#374151",  # для выделения
    "text": "#d1d5db",
    "text_muted": "#9ca3af",
    "border": "#374151",
    "border_light": "#4b5563",
    "progress": "#6b7280",  # индикатор прогресса
}

# Основная таблица стилей (дополняет qt_material)
MAIN_STYLESHEET = f"""
/* Панели с закруглёнными углами */
QGroupBox {{
    font-weight: 600;
    font-size: 12px;
    color: {COLORS["text"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    margin-top: 16px;
    padding: 20px 16px 16px 16px;
    background-color: {COLORS["bg_panel"]};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    padding: 0 10px;
    color: {COLORS["text_muted"]};
    background-color: {COLORS["bg_panel"]};
    font-weight: 500;
    font-size: 11px;
    text-transform: none;
}}

/* Левая панель — блок информации */
QTextEdit#infoPanel {{
    background-color: {COLORS["bg_input"]};
    color: {COLORS["text"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 14px;
    font-size: 13px;
    selection-background-color: {COLORS["border_light"]};
}}

/* Дерево слоёв */
QTreeWidget QHeaderView::section {{
    padding: 8px 10px;
    font-weight: 500;
    text-transform: none;
}}
QTreeWidget {{
    background-color: {COLORS["bg_input"]};
    color: {COLORS["text"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 10px;
}}
QTreeWidget::item {{
    padding: 8px 6px;
    border-radius: 4px;
}}
QTreeWidget::item:hover {{
    background-color: {COLORS["bg_elevated"]};
}}
QTreeWidget::item:selected {{
    background-color: {COLORS["accent_muted"]};
    color: white;
}}
QTreeWidget::branch:has-children:!has-siblings:closed,
QTreeWidget::branch:closed:has-children:has-siblings {{
    border-image: none;
}}

/* Тулбар — монохромные иконки */
QToolBar {{
    spacing: 4px;
    padding: 6px 12px;
    background: transparent;
}}
QToolBar QToolButton {{
    color: {COLORS["text_muted"]};
}}
QToolBar QToolButton:hover {{
    color: {COLORS["text"]};
}}
QToolButton {{
    padding: 6px;
    margin: 0 2px;
    min-width: 28px;
    min-height: 28px;
    border: none;
    border-radius: 4px;
}}
QToolButton:hover {{
    background-color: {COLORS["bg_elevated"]};
    color: {COLORS["text"]};
}}
QToolButton:pressed {{
    background-color: {COLORS["accent_muted"]};
}}
QToolBar::separator {{
    width: 1px;
    margin: 4px 10px;
    background-color: {COLORS["border"]};
}}

/* Прогресс-бар */
QProgressBar {{
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    text-align: center;
    background-color: {COLORS["bg_input"]};
}}
QProgressBar::chunk {{
    background-color: {COLORS["progress"]};
    border-radius: 3px;
}}

/* Статус-бар */
QStatusBar {{
    background-color: {COLORS["bg_panel"]};
    color: {COLORS["text_muted"]};
}}

/* Разделитель сплиттера */
QSplitter::handle {{
    background-color: {COLORS["border"]};
    width: 3px;
}}

/* Скролл-бар (центральная область) */
QScrollBar:vertical {{
    background: {COLORS["bg_input"]};
    width: 12px;
    border-radius: 6px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {COLORS["border"]};
    border-radius: 6px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS["border_light"]};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* Центральная область просмотра */
QScrollArea {{
    background-color: {COLORS["bg_dark"]};
    border: none;
}}
QScrollArea#centralScroll {{
    padding: 6px;
}}
QGraphicsView {{
    background-color: {COLORS["bg_dark"]};
}}
"""

DIALOG_STYLESHEET = f"""
QDialog {{
    background-color: {COLORS["bg_dark"]};
}}
QLabel {{
    color: {COLORS["text"]};
    font-size: 13px;
}}
QDoubleSpinBox {{
    background-color: {COLORS["bg_input"]};
    color: {COLORS["text"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 8px 12px;
}}
QDoubleSpinBox:focus {{
    border-color: {COLORS["border_light"]};
}}
"""
