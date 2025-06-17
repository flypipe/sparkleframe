SHELL           :=/usr/bin/env bash
PYTEST_THREADS  ?=$(shell echo $$((`getconf _NPROCESSORS_ONLN` / 3)))
LOCAL_DIR		:=./.docker
MIN_COVERAGE	= 79
version			?=

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

pip-compile:
	pip-compile requirements-pkg.in && pip-compile requirements-dev.in
.PHONY: pip-compile

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
	pip-compile requirements-pkg.in
	pip-compile requirements-dev.in
	make build
	make githooks
.PHONY: pip-compile

setup:
	export PYTHONPATH=$PYTHONPATH:./sparkleframe
	pip install -r requirements-pkg.in
	pip-compile requirements-pkg.in
	pip-compile requirements-dev.in
	pip install -r requirements-dev.txt
	make build
	make githooks
.PHONY: setup

docs:
	mike delete --all | true
	mike deploy --update-aliases 0.0
	mike deploy --alias-type=redirect --update-aliases 0.1 latest
	mike set-default latest
.PHONY: docs


docs-deploy:
	@[ -n "$(version)" ] || (echo "ERROR: version is required"; exit 1)
	mike delete --all | true
	mike deploy --allow-empty --push --update-aliases $(shell echo $(version) | awk -F. '{print $$1"."$$2}') latest
	mike set-default --push latest
.PHONY: docs-deploy

docs-serve:
	mike serve -a 127.0.0.1:8000
.PHONY: docs-serve