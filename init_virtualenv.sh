#!/bin/bash
set -e
[ -d .env ] && rm -Rf .env
python3.9 -m venv .env
. .env/bin/activate
pip install -U pip
pip install -r pip-requirements.txt -r pip-requirements-test.txt
pre-commit install
