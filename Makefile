.PHONY: install test lint format-check demo clean-caches

install:
	python -m pip install -r environment/requirements-lock.txt
	python -m pip install -e . --no-deps

test:
	python -m pytest -q

lint:
	python -m ruff check src tests scripts

format-check:
	python -m ruff format --check src tests scripts

demo:
	vascular-edge run --config configs/analysis.yaml --demo

clean-caches:
	python -c "import pathlib,shutil; [shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree('.pytest_cache',ignore_errors=True); shutil.rmtree('.ruff_cache',ignore_errors=True)"
