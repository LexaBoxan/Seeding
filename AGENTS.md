# Repository Guidelines

## Project Structure & Module Organization
- Application code lives in `seeding/`: `main.py` boots the PyQt5 GUI, `config.py` holds default YOLO weight paths (`models/bestCrop.pt`, `models/bestKlassSeg.pt`), `utils.py` contains image helpers (NMS, bbox rotation), `report.py` builds PDF summaries, and `ui/` hosts widgets such as `main_window.py`.
- Model artefacts are in `models/`. Training configs sit in `TrainConfigs/`. Sample outputs and masks are in `results/`, `seg_out/`, and `Photo/`.
- Datasets are versioned under `dataset/` (segmentation and classification splits). Tests live in `tests/`.

## Build, Test, and Development Commands
- Install (editable): `python -m pip install -e .` from the repo root; Python 3.9+ is required.
- Run the GUI with a custom detector: `python -m seeding.main --weights models/bestCrop.pt` (or set `YOLO_WEIGHTS_PATH`).
- Quick smoke of utility functions and PDF generation: `python -m pytest tests`.
- Export a packaged script entrypoint: `python -m pip install .` then call `seeding` (defined in `pyproject.toml`).

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; prefer explicit imports and typed signatures.
- Use descriptive, lower_snake_case for functions/variables; PascalCase for classes and Qt widgets.
- Keep UI logic within `ui/` classes and image logic in `processing/` or `utils.py`; avoid cross-layer side effects.
- Favor small, testable helpers; log with the module logger rather than print.

## Testing Guidelines
- Test suite uses `pytest`; place new tests beside the code under `tests/`.
- Mirror file names (e.g., `seeding/utils.py` → `tests/test_utils.py`); name tests `test_*`.
- When adding image/YOLO logic, cover bbox math (rotations, NMS thresholds) and PDF output size/content checks.

## Commit & Pull Request Guidelines
- Commits: short imperative subject (<72 chars), include context in the body when altering models, configs, or datasets.
- PRs should describe scope, expected behavior, and screenshots/gifs for UI changes; link issues and note weight/config paths used.
- Document any new environment variables (`YOLO_WEIGHTS_PATH`, `YOLO_CLASSIFY_WEIGHTS_PATH`) or data requirements.

## Security & Configuration Tips
- Do not commit large weight files or raw datasets; prefer `.gitignore` updates and paths under `models/` or `dataset/`.
- Validate external paths before loading to avoid accidental overwrites; keep user-writable output under `results/` or `seg_out/`.
