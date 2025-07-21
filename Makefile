default: lint

# List available targets
help:
	@echo "Available targets:"
	@echo "  default                      - Run lint"
	@echo "  lint                         - Run all lints"
	@echo "  lint-spacing-and-indentation - Lint spacing and indentation conventions"
	@echo "  lint-commit-messages         - Lint commit message conventions"
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

.PHONY: default help lint lint-spacing-and-indentation lint-commit-messages
