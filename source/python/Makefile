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
	@echo "  build-images             - Build multiarch Docker images (amd64 + arm64)"
	@echo "  push-images-hash         - Push images with commit hash tags"
	@echo "  push-images-latest       - Push images with latest tags"
	@echo "  images                   - Build and push all multiarch images"
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

build-images:
	$(eval SHORT_COMMIT_HASH := $(shell git rev-parse --short=9 HEAD))

	# Build amd64 wheel
	docker run --rm -v "$(shell pwd)":/io \
		--platform linux/amd64 \
		ghcr.io/pyo3/maturin build \
			--release \
			--target x86_64-unknown-linux-gnu \
			--manifest-path /io/Cargo.toml

	# Build arm64 wheel
	docker run --rm -v "$(shell pwd)":/io \
		--platform linux/arm64 \
		ghcr.io/pyo3/maturin build \
			--release \
			--target aarch64-unknown-linux-gnu \
			--manifest-path /io/Cargo.toml

	# Pull base images for both architectures
	docker pull --platform linux/amd64 cgr.dev/chainguard/python:latest-dev
	docker pull --platform linux/amd64 cgr.dev/chainguard/python:latest
	docker pull --platform linux/arm64 cgr.dev/chainguard/python:latest-dev
	docker pull --platform linux/arm64 cgr.dev/chainguard/python:latest

	# Build amd64 images
	DOCKER_BUILDKIT=0 docker build --no-cache --platform linux/amd64 \
		-t ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH)-amd64 --target dev --file Dockerfile .
	DOCKER_BUILDKIT=0 docker build --no-cache --platform linux/amd64 \
		-t ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH)-amd64 --file Dockerfile .

	# Build arm64 images
	DOCKER_BUILDKIT=0 docker build --no-cache --platform linux/arm64 \
		-t ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH)-arm64 --target dev --file Dockerfile .
	DOCKER_BUILDKIT=0 docker build --no-cache --platform linux/arm64 \
		-t ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH)-arm64 --file Dockerfile .

	# Tag as latest for local use (uses native architecture)
	docker tag ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH)-arm64 \
		ghcr.io/build-trust/autonomy-python-dev:latest
	docker tag ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH)-arm64 \
		ghcr.io/build-trust/autonomy-python:latest

	@echo ""
	@echo "Build complete. Local images tagged as :latest (arm64)"
	@echo "To use amd64 locally, use the :$(SHORT_COMMIT_HASH)-amd64 tag"

push-images-hash:
	$(eval SHORT_COMMIT_HASH := $(shell git rev-parse --short=9 HEAD))

	# Push arch-specific images first
	docker push ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH)-amd64
	docker push ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH)-arm64
	docker push ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH)-amd64
	docker push ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH)-arm64

	# Create and push multiarch manifests for commit hash
	-docker manifest rm ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH) 2>/dev/null || true
	docker manifest create ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH) \
		ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH)-amd64 \
		ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH)-arm64
	docker manifest push ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH)

	-docker manifest rm ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH) 2>/dev/null || true
	docker manifest create ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH) \
		ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH)-amd64 \
		ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH)-arm64
	docker manifest push ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH)

push-images-latest:
	$(eval SHORT_COMMIT_HASH := $(shell git rev-parse --short=9 HEAD))

	# Create and push multiarch manifests for latest
	-docker manifest rm ghcr.io/build-trust/autonomy-python-dev:latest 2>/dev/null || true
	docker manifest create ghcr.io/build-trust/autonomy-python-dev:latest \
		ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH)-amd64 \
		ghcr.io/build-trust/autonomy-python-dev:$(SHORT_COMMIT_HASH)-arm64
	docker manifest push ghcr.io/build-trust/autonomy-python-dev:latest

	-docker manifest rm ghcr.io/build-trust/autonomy-python:latest 2>/dev/null || true
	docker manifest create ghcr.io/build-trust/autonomy-python:latest \
		ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH)-amd64 \
		ghcr.io/build-trust/autonomy-python:$(SHORT_COMMIT_HASH)-arm64
	docker manifest push ghcr.io/build-trust/autonomy-python:latest

images:
	$(MAKE) build-images
	$(MAKE) push-images-hash
	$(MAKE) push-images-latest

clean:
	rm -rf target
	rm -rf python/autonomy/*.abi3.so

.PHONY: default help fmt format build test test-agents test-models test-tools clean
.PHONY: coverage coverage-html coverage-report
.PHONY: build-images push-images-hash push-images-latest images
