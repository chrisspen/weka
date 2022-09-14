#!/bin/bash
set -e
echo "[$(date)] Initializing environment."
[ -d .env ] && rm -Rf .env
python3.9 -m venv .env
. .env/bin/activate
pip install -U pip
pip install -U setuptools wheel
pip install -r pip-requirements.txt -r pip-requirements-test.txt
pre-commit install
