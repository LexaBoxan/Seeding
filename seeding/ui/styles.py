"""Централизованные стили интерфейса.

Тёмная тема с природными акцентами (зелёный) для приложения анализа сеянцев.
"""

# Цветовая палитра
COLORS = {
    "bg_dark": "#1a1d21",
    "bg_panel": "#23262b",
    "bg_elevated": "#2c3036",
    "bg_input": "#1e2226",
    "accent": "#22c55e",       # зелёный — сеянцы
    "accent_hover": "#16a34a",
    "accent_muted": "#166534",
    "text": "#e4e6eb",
    "text_muted": "#9ca3af",
    "border": "#3b4048",
    "border_light": "#4b5563",
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
    margin-top: 12px;
    padding: 16px 12px 12px 12px;
    background-color: {COLORS["bg_panel"]};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 8px;
    color: {COLORS["accent"]};
    background-color: {COLORS["bg_panel"]};
}}

/* Левая панель — блок информации */
QTextEdit#infoPanel {{
    background-color: {COLORS["bg_input"]};
    color: {COLORS["text"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 12px;
    font-size: 13px;
    selection-background-color: {COLORS["accent_muted"]};
}}

/* Дерево слоёв */
QTreeWidget {{
    background-color: {COLORS["bg_input"]};
    color: {COLORS["text"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 8px;
}}
QTreeWidget::item {{
    padding: 6px 4px;
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

/* Тулбар — компактнее и чище */
QToolBar {{
    spacing: 4px;
    padding: 6px 8px;
}}
QToolButton {{
    padding: 6px 10px;
    border-radius: 6px;
}}
QToolButton:hover {{
    background-color: {COLORS["bg_elevated"]};
}}
QToolButton:pressed {{
    background-color: {COLORS["accent_muted"]};
}}

/* Прогресс-бар */
QProgressBar {{
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    text-align: center;
    background-color: {COLORS["bg_input"]};
}}
QProgressBar::chunk {{
    background-color: {COLORS["accent"]};
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
    width: 2px;
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
    background: {COLORS["accent"]};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* Центральная область просмотра */
QGraphicsView {{
    background-color: {COLORS["bg_dark"]};
}}
QScrollArea {{
    background-color: {COLORS["bg_dark"]};
    border: none;
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
    border-color: {COLORS["accent"]};
}}
"""
