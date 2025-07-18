# 🖼️ ImageEditor с YOLOv8

ImageEditor — кроссплатформенное приложение на Python и PyQt5 для просмотра и
обработки изображений и PDF‑файлов. Программа ищет сеянцы с помощью модели
YOLOv8 и показывает результат в удобном дереве слоёв.

## Основные возможности

- Загрузка изображений форматов PNG/JPG/BMP и документов PDF
- Отображение страниц и найденных объектов в древовидном списке
- Детекция сеянцев через модель YOLOv8
- Масштабирование, вращение и вписывание изображения в окно
- Структурированные `dataclass` для хранения данных
- Формирование PDF‑отчёта по результатам распознавания

## Установка

1. Установите Python 3.9 или новее.
2. Создайте виртуальное окружение и установите зависимости:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r seeding/requirements.txt
```

Для загрузки весов YOLOv8 можно задать переменную окружения
`YOLO_WEIGHTS_PATH` или передать путь через аргумент `--weights` при запуске.

## Структура проекта

```text
.
├── README.md
└── seeding
    ├── __init__.py           # Точка входа пакета
    ├── main.py               # Запуск GUI
    ├── utils.py              # Вспомогательные функции
    ├── processing
    │   └── image_processor.py# Заглушка алгоритмов
    ├── ui
    │   ├── main_window.py    # Основное окно
    │   └── tree_widget.py    # Дерево слоёв
    ├── models
    │   └── data_models.py    # dataclass модели данных
    ├── report.py             # Создание PDF‑отчёта
    └── requirements.txt
```

## Алгоритм работы

1. Пользователь выбирает изображение или PDF через диалог «Открыть файл».
2. Страницы добавляются в дерево слоёв (`LayerTreeWidget`).
3. По команде "Найти сеянцы" запускается модель YOLOv8.
4. Список боксов обрабатывается функцией `simple_nms` из `utils.py`.
5. Найденные объекты сохраняются в `ObjectImage` и отображаются в дереве.
6. Пользователь может вращать, масштабировать и просматривать объекты.
7. По кнопке "Создать отчёт" вызывается `create_pdf_report` из `report.py`.

## Основные файлы и функции

- `ui/main_window.py` – содержит класс `ImageEditor` с методами
  `open_image`, `load_pdf`, `find_seedlings`, `rotate_image`, `create_report` и
  другими действиями интерфейса.
- `utils.py` – функция `simple_nms` для подавления перекрывающихся боксов.
- `models/data_models.py` – dataclass‑структуры `OriginalImage` и
  `ObjectImage` для хранения результатов.
- `report.py` – функция `create_pdf_report` формирует отчёт в формате PDF.

## Запуск приложения

```bash
python -m seeding.main --weights /path/to/best.pt
```

Если `--weights` не указан, используется переменная `YOLO_WEIGHTS_PATH` или
значение по умолчанию `weights/best.pt` из файла `config.py`.

## Тесты

Для запуска тестов выполните:

```bash
PYTHONPATH=. pytest
```

## Лицензия

MIT License

## Автор

Aleshkin Dev — [@aleshkin](https://t.me/aleshkin)
