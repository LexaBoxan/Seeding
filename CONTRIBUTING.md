# Contributing

## Setup

1. Use Python `3.10+`.
2. Create and activate a virtual environment.
3. Install the project in editable mode:

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

## Development workflow

1. Create a topic branch from the active integration branch.
2. Keep changes focused on one problem.
3. Add or update tests with every behavior change.
4. Run the local checks before opening a pull request:

```bash
python -m pytest tests
python -m flake8
python -m pip_audit
```

## Code guidelines

- Follow PEP 8 with 4-space indentation.
- Prefer typed function signatures and small helpers.
- Keep UI logic in `seeding/ui/` and domain logic in services/models/storage.
- Route new user-facing strings through `seeding/ui/i18n.py`.
- Use logging instead of `print`.

## Tests

- Place tests in `tests/`.
- Mirror the module name when possible, for example
  `seeding/storage.py` -> `tests/test_storage_service.py`.
- Mock heavy YOLO/model loading in tests instead of running real inference.

## Models, datasets, and large files

- Do not commit raw datasets or large weight files.
- Keep model paths configurable and validate external paths before use.
- Document any new environment variables or external data requirements.

## Pull requests

Please include:

- a short summary of the change;
- user-visible impact and any migration notes;
- tests that were added or updated;
- screenshots for UI changes when relevant.

## Security

Do not open public issues for security vulnerabilities. Follow
[`SECURITY.md`](SECURITY.md) instead.
