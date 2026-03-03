# Windows packaging

Build the onedir Windows distribution with:

```bash
python -m pip install -e .
python -m pip install -r requirements-packaging.txt
python -m PyInstaller --noconfirm --clean packaging/pyinstaller/Seeding.spec
```

The output bundle is created in `dist/Seeding/`.
