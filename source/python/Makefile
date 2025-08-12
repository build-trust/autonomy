default: build

# List available targets
help:
	@echo "Available targets:"
	@echo "  default - Run build"
	@echo "  build   - Build this project"
	@echo "  test    - Run tests"
	@echo "  clean   - Delete all build artifacts"
	@echo "  help    - Show this help message"

build:
	uv run --active maturin develop

fmt: format
format:
	cargo fmt
	ruff format

test:
	uv run --active pytest

clean:
	rm -rf target
	rm -rf python/autonomy/*.abi3.so

.PHONY: default help build test clean
