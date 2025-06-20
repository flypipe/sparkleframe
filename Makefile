SHELL           :=/usr/bin/env bash
PYTEST_THREADS  ?=$(shell echo $$((`getconf _NPROCESSORS_ONLN` / 3)))
LOCAL_DIR		:=./.docker
MIN_COVERAGE	= 79
version			?=

export PYTHONPATH := $(PYTHONPATH):./sparkleframe

# This block checks for .env and exports it for all recipes
ifneq (,$(wildcard .env))
  include .env
  export $(shell sed 's/=.*//' .env)
endif

bash:
	docker compose -f $(LOCAL_DIR)/docker-compose.yaml run --remove-orphans -it sparkleframe bash
.PHONY: bash

black:
	docker compose -f $(LOCAL_DIR)/docker-compose.yaml run --remove-orphans sparkleframe sh -c "black sparkleframe -l 119"
.PHONY: black

black-check:
	docker compose -f $(LOCAL_DIR)/docker-compose.yaml run --remove-orphans sparkleframe sh -c "black sparkleframe -l 119 --check"
.PHONY: black-check

build:
	docker compose -f $(LOCAL_DIR)/docker-compose.yaml build
.PHONY: build

lint:
	docker compose -f $(LOCAL_DIR)/docker-compose.yaml run --remove-orphans sparkleframe sh -c "python -m ruff check --line-length 119 sparkleframe"
.PHONY: lint

coverage:
	docker compose -f $(LOCAL_DIR)/docker-compose.yaml run --remove-orphans sparkleframe sh -c "pytest -n $(PYTEST_THREADS) -k '_test.py' --cov-config=sparkleframe/.coverage --cov=sparkleframe --no-cov-on-fail --cov-fail-under=$(MIN_COVERAGE) -v sparkleframe"
.PHONY: coverage

test:
	docker compose -f $(LOCAL_DIR)/docker-compose.yaml run --remove-orphans sparkleframe sh -c "pytest -n $(PYTEST_THREADS) -k '_test.py' -vv $(f)"
.PHONY: test

wheel:
	flit build --format=wheel
.PHONY: wheel

pr-check:
	make black
	make lint
	make coverage
.PHONY: pr-check

githooks:
	git config core.hooksPath .github/hooks
	echo "Custom Git hooks enabled (core.hooksPath set to .githooks)"
.PHONY: githooks

pip-compile:
	pip install -r requirements-pkg.in
	pip-compile requirements-pkg.in --no-annotate --no-header
	pip-compile requirements-dev.in --no-annotate --no-header
.PHONY: pip-compile

setup: pip-compile
	pip install -r requirements-dev.txt
	make build
	make githooks
.PHONY: setup

docs:
	@if [ ! -f changelog.md ]; then \
		echo "changelog.md does not exist, running command..."; \
		python scripts/generate_changelog.py; \
	fi
	@if [ ! -f docs/supported_api_doc.md ]; then \
		echo "docs/supported_api_doc.md does not exist, running command..."; \
		python docs/generate_supported_api.py; \
	fi
	cp changelog.md ./docs
	mkdocs serve
.PHONY: docs


docs-deploy:
	@[ -n "$(version)" ] || (echo "ERROR: version is required"; exit 1)
	cp changelog.md ./docs
	mike deploy --allow-empty --push --update-aliases $(shell echo $(version) | awk -F. '{print $$1"."$$2}') latest
	mike set-default --push latest
.PHONY: docs-deploy