default: lint

# List available targets
help:
	@echo "Available targets:"
	@echo "  default                      - Run lint"
	@echo "  lint                         - Run all lints"
	@echo "  lint-spacing-and-indentation - Lint spacing and indentation conventions"
	@echo "  lint-commit-messages         - Lint commit message conventions"
	@echo "  run-examples                 - Run all examples"
	@echo "  run-examples-root            - Run examples from autonomy/examples/"
	@echo "  run-examples-python          - Run examples from source/python/examples/"
	@echo "  help                         - Show this help message"

lint: lint-spacing-and-indentation lint-commit-messages

# Lint spacing and indentation conventions defined in
# the .editorconfig file.
lint-spacing-and-indentation:
	eclint

# Lint commit message conventions defined in
# the ./tools/commitlint/commitlint.config.js file.
lint-commit-messages:
	commitlint --config ./tools/commitlint/commitlint.config.js \
		--from $$(git rev-list --max-parents=0 HEAD) --to HEAD

# Run all examples
run-examples: run-examples-root run-examples-python

# Run examples from autonomy/examples/
run-examples-root:
	@echo "Running examples from autonomy/examples/..."
	@for example in examples/*/images/main/main.py; do \
		if [ -f "$$example" ]; then \
			echo "Running $$example"; \
			AUTONOMY_WAIT_UNTIL_INTERRUPTED=0 OCKAM_SQLITE_IN_MEMORY=1 uv run --active "$$example" || echo "Example $$example failed"; \
		fi; \
	done

# Run examples from source/python/examples/
run-examples-python:
	@echo "Running examples from source/python/examples/..."
	@cd source/python && for example in examples/*.py; do \
		if [ -f "$$example" ]; then \
			echo "Running $$example"; \
			AUTONOMY_WAIT_UNTIL_INTERRUPTED=0 OCKAM_SQLITE_IN_MEMORY=1 uv run --active "$$example" || echo "Example $$example failed"; \
		fi; \
	done

.PHONY: default help lint lint-spacing-and-indentation lint-commit-messages run-examples run-examples-root run-examples-python
