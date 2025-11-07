default: build

# List available targets
help:
	@echo "Available targets:"
	@echo "  default                  - Run build"
	@echo "  build                    - Build this project"
	@echo "  test                     - Run tests with proper resource limits"
	@echo "  test-agents              - Run agent tests specifically"
	@echo "  coverage                 - Run tests with coverage report (terminal)"
	@echo "  coverage-html            - Run tests with HTML coverage report"
	@echo "  coverage-report          - Open HTML coverage report in browser"
	@echo "  format (fmt)             - Format code using cargo fmt and ruff format"
	@echo "  clean                    - Delete all build artifacts"
	@echo "  images                   - Build and push all AMD64 images"
	@echo "  build-images-amd64       - Build AMD64 Docker images"
	@echo "  push-images-amd64-hash   - Push AMD64 images with commit hash tags"
	@echo "  push-images-amd64-latest - Push AMD64 images with latest tags"
	@echo "  help                     - Show this help message"

build:
	uv run --active maturin develop

fmt: format
format:
	cargo fmt
	ruff format

test:
	ulimit -n 4096 && AUTONOMY_WAIT_UNTIL_INTERRUPTED=0 AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
		uv run --active pytest

test-models:
	ulimit -n 4096 && AUTONOMY_WAIT_UNTIL_INTERRUPTED=0 AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
		uv run --active pytest -vv tests/models/test_models.py

test-agents:
	ulimit -n 4096 && AUTONOMY_WAIT_UNTIL_INTERRUPTED=0 AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
		uv run --active pytest -vv tests/agents/test_agents.py

test-tools:
	ulimit -n 4096 && AUTONOMY_WAIT_UNTIL_INTERRUPTED=0 AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
		uv run --active pytest -vv tests/tools/test_tools.py
	ulimit -n 4096 && AUTONOMY_WAIT_UNTIL_INTERRUPTED=0 AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
		uv run --active pytest -vv tests/tools/test_mcp_tools.py

coverage:
	ulimit -n 4096 && AUTONOMY_WAIT_UNTIL_INTERRUPTED=0 AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
		uv run --active pytest --cov=autonomy --cov-report=term-missing || true

coverage-html:
	ulimit -n 4096 && AUTONOMY_WAIT_UNTIL_INTERRUPTED=0 AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
		uv run --active pytest --cov=autonomy --cov-report=html --cov-report=term-missing || true
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"
	@echo "Run 'make coverage-report' to open it in your browser"

coverage-report: coverage-html
	@if command -v open >/dev/null 2>&1; then \
		open htmlcov/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open htmlcov/index.html; \
	else \
		echo "Please open htmlcov/index.html in your browser"; \
	fi

build-images-amd64:
	docker run --rm -v "$(shell pwd)":/io \
		--platform linux/amd64 \
		ghcr.io/pyo3/maturin build \
			--release \
			--target x86_64-unknown-linux-gnu \
			--manifest-path /io/Cargo.toml
	$(eval SHORT_COMMIT_HASH := $(shell git rev-parse --short=9 HEAD))

	docker pull cgr.dev/chainguard/python:latest-dev
	docker pull cgr.dev/chainguard/python:latest

	DOCKER_BUILDKIT=0 docker build --no-cache --platform linux/amd64 \
		-t ghcr.io/build-trust/autonomy-python-dev:latest \
		-t ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH) --target dev --file Dockerfile .

	DOCKER_BUILDKIT=0 docker build --no-cache --platform linux/amd64 \
		-t ghcr.io/build-trust/autonomy-python:latest \
		-t ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH) --file Dockerfile .

push-images-amd64-hash:
	$(eval SHORT_COMMIT_HASH := $(shell git rev-parse --short=9 HEAD))
	docker push ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH)
	docker push ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH)

push-images-amd64-latest:
	$(eval SHORT_COMMIT_HASH := $(shell git rev-parse --short=9 HEAD))

	docker tag ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH) \
		ghcr.io/build-trust/autonomy-python-dev:latest
	docker push ghcr.io/build-trust/autonomy-python-dev:latest

	docker tag ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH) \
		ghcr.io/build-trust/autonomy-python:latest
	docker push ghcr.io/build-trust/autonomy-python:latest

images:
	$(MAKE) build-images-amd64
	$(MAKE) push-images-amd64-hash
	$(MAKE) push-images-amd64-latest

clean:
	rm -rf target
	rm -rf python/autonomy/*.abi3.so

.PHONY: default help fmt format build test test-agents test-models test-tools clean
.PHONY: coverage coverage-html coverage-report
.PHONY: build-images-amd64 push-images-amd64-latest push-images-amd64-hash images
