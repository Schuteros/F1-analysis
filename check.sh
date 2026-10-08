#!/bin/sh
.venv/Scripts/python.exe -m mypy && .venv/Scripts/python.exe -m ruff check && .venv/Scripts/python.exe -m pytest -q
