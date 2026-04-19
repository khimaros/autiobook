.PHONY: all venv build build-cpu build-cuda build-rocm test lint format precommit clean

RUN := uv run --no-sync

all: build

venv:
	uv venv

build: build-cuda

build-cpu: venv
	uv sync --extra cpu

build-cuda: venv
	uv sync --extra gpu

build-rocm: venv
	uv sync --extra rocm-gfx1151
	FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" uv pip install flash-attn --no-build-isolation

test:
	uv pip install pytest
	$(RUN) pytest

test-e2e:
	./tests/test_e2e.sh
	./tests/test_e2e_dramatize.sh

lint:
	uv pip install ruff mypy
	$(RUN) ruff check autiobook/
	$(RUN) mypy autiobook/

format:
	uv pip install ruff
	$(RUN) black autiobook/ tests/
	$(RUN) ruff check --fix autiobook/ tests/

precommit: format lint test

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf workdir_test/
	rm -rf .venv/
	find . -type d -name __pycache__ -exec rm -rf {} +
