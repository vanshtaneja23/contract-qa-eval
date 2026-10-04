.PHONY: setup up down migrate ingest test test-unit test-integration lint web-check check

setup:            ## install deps and the pre-commit secret scanner
	git config core.hooksPath .githooks
	uv sync
	cd web && npm ci

up:               ## start Postgres 16 + pgvector
	docker compose up -d --wait

down:
	docker compose down

migrate: up
	cd api && uv run alembic upgrade head

ingest: migrate   ## download CUAD and ingest the pinned 40-contract subset
	uv run cqa-eval ingest --n 40 --seed 42 --matters 4

test:             ## all Python tests (integration tests need Docker)
	uv run pytest

test-unit:
	uv run pytest -m "not integration"

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy api/src eval/src scripts/check_secrets.py
	python3 scripts/check_secrets.py --all

web-check:
	cd web && npm run typecheck && npm run lint

check: lint test web-check
