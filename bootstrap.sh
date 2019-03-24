#!/bin/sh
export FLASK_APP=./predictChessPiece/index.py
export FLASK_ENV=production
source $(pipenv --venv)/bin/activate
flask run -h 0.0.0.0
