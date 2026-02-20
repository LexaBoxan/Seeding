# Seeding 4.0.0

Настольное приложение для анализа изображений и PDF с детекцией сеянцев, классификацией частей растения и формированием PDF-отчётов.

## Требования

- Python 3.10+
- Windows / macOS / Linux

## Установка

Из корня проекта:

```bash
python -m pip install -e .
```

Либо установка зависимостей напрямую:

```bash
python -m pip install -r seeding/requirements.txt
```

## Запуск

```bash
python -m seeding.main --weights models/bestCrop.pt
```

Также можно указать веса через переменные окружения:

- `YOLO_WEIGHTS_PATH`
- `YOLO_CLASSIFY_WEIGHTS_PATH`

## Структура проекта

```text
seeding/
  main.py                     # Точка входа GUI
  config.py                   # Константы и пути
  path_utils.py               # Кроссплатформенная проверка путей
  utils.py                    # Геометрия bbox, NMS и утилиты изображений
  report.py                   # Генерация PDF-отчётов

  controllers/
    __init__.py               # AppController (open/rotate/detect/report)

  services/
    __init__.py               # Image/Detection/Classification/Report services

  models/
    __init__.py               # Dataclass-модели + AppState + DTO-типы

  ui/
    main_window.py            # Главное окно и связка UI с контроллером
    settings_dialog.py        # Настройки порогов/темы/языка
    icon_manager.py           # Единая загрузка иконок
    metrics.py                # Единые размеры/отступы интерфейса
    styles.py                 # Подключение QSS из ресурсов
    theme_manager.py          # Применение тем
    layout_state.py           # Сохранение/восстановление layout
    i18n.py                   # Runtime-локализация интерфейса
    tree_widget.py            # Виджет дерева слоёв
    bbox_item.py              # Редактируемый bounding-box на сцене

  resources/
    icons/                    # SVG-иконки
    styles/                   # QSS темы (dark/light)
    translations/             # Файлы переводов Qt (.ts)
```

## Тестирование

Установить dev-зависимости:

```bash
python -m pip install -r requirements-dev.txt
```

Запустить тесты:

```bash
python -m pytest tests
```

Запустить с покрытием:

```bash
python -m pytest --cov=seeding --cov-report=term-missing --cov-report=xml tests
```

## CI

Workflow: `.github/workflows/ci.yml`

CI выполняет:

- установку зависимостей;
- запуск `pytest`;
- запуск `flake8` для ключевых рефакторных модулей;
- проверку наличия файлов переводов.

## Что изменено в 4.0.0

- Редизайн UI: компактный верхний toolbar, отдельный toolbox, единый визуальный стиль.
- Dock-панели и сохранение раскладки окна в `QSettings`.
- Инструменты `Select / Hand / Zoom` с горячими клавишами `V/H/Z`, временная `Hand` по `Space`.
- Архитектурная декомпозиция: `AppController` + сервисы изображений/детекции/классификации/отчётов.
- Подготовлена база локализации и темизации (ru/en + dark/light).
- Добавлены тесты для контроллера, сервисов, иконок и layout-state.
