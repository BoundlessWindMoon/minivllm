.PHONY: test test-unit test-integration lint fmt check-docs serve bench

test-unit:
	python -m pytest test/unit/ -q

test-integration:
	python -m pytest test/integration/ -q

test:
	python -m pytest test/unit/ -q
	python -m pytest test/integration/ -q

lint:
	python -m ruff check .

fmt:
	python -m ruff format .
	python -m ruff check . --fix

check-docs:
	python scripts/tools/check_docs.py

serve:
	python scripts/serve/start_server.py --config configs/runs/batch.yaml

bench:
	python scripts/serve/bench_serving.py --mode fixed --concurrency 4 \
		--num-requests 20 --input-len 64 --shared-prefix-len 512 \
		--output-len 64 --temperature 0.0
