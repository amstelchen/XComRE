#!/bin/bash

poetry build
poetry export --only main -f requirements.txt -o requirements.txt --without-hashes
poetry export --with dev -f requirements.txt -o requirements-dev.txt --without-hashes
