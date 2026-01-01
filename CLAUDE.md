# Instructions for Coding Agents

These instructions define how Coding Agents should assist with this project.

## Project Overview

This repository contains source code for the following projects:
- The `autonomy` python package in `source/python`
- The `autonomy` command line tool in `source/rust/autonomy_command`

The `autonomy` python package:
- Is built using python and rust, with help from pyo3
- Pyo3 is a tool for creating native python extension modules in rust
- End users use `autonomy` in their python code
- Rust code for the `autonomy` python package lives in `source/python/src`
- Python code for the `autonomy` python package lives in `source/python/python`

The `autonomy` command line tool:
- Is built using rust
- Code for the `autonomy` command lives in `source/rust/autonomy_command`

## General Development Guidelines

- Use the `.scratch` directory for any notes, temporary tests, or experimental code that shouldn't be committed to the repository.
  - Create a dedicated subfolder within `.scratch` for each project or task (e.g., `.scratch/feature-name`, `.scratch/bug-fix-123`)
  - Before creating a new subfolder, check if one already exists for the current work by exploring the `.scratch` directory
  - This keeps work organized and prevents clutter from multiple simultaneous tasks
  - Use descriptive names for subfolders that indicate the purpose
- When using `git diff`, always pipe output to prevent blocking on large diffs:
  - Use `git diff | cat` to view diffs without pagination
  - Use `git diff --stat | cat` to see a summary of changes
  - Use `git diff --name-only | cat` to see only changed file names
  - Avoid running bare `git diff` commands that may open an interactive pager

## Guidelines for creating and running examples

- Refer existing examples in the `examples`.
- To run examples in the `examples` durectory:
  - Use `autonomy --rm` in that directory.
  - The HTTP API is then available at `http://localhost:32100`.
  - Logs are then available at `http://localhost:32101`.
- To pick models, look at the list of models defined in `source/python/python/autonomy/models`

## Testing Strategy

When testing agents and streaming responses:
- Redirect logs to a file for analysis: `uv run --active main.py > /tmp/test.log 2>&1 &`
- Test the HTTP API separately using curl commands on `http://localhost:8000`
- For streaming responses, check for proper completion: `curl --silent --request POST --header "Content-Type: application/json" --data '{"message":"..."}' "http://localhost:8000/agents/AGENT_NAME?stream=true"`
- Verify no timeout errors in logs: `grep -i "error\|exception\|timeout" /tmp/test.log`
- Test multi-turn conversations by sending multiple requests to verify state management
- When testing `ask_user_for_input` tool:
  - First request should pause with `"phase": "waiting_for_input"`
  - Stream should complete (not timeout) when waiting for input
  - Second request with user response should resume correctly
  - Verify no duplicate messages in conversation history
- Use `jq` to parse and filter streaming JSON responses for specific fields
- Enable debug logging for detailed flow analysis: `AUTONOMY_LOG_LEVEL="INFO,agent=DEBUG,model=DEBUG"`

## Guidelines for developing in Python

- The source code of the Autonomy python library lives in `source/python`.
- This library is built using Python and Rust, with help from PyO3.
- PyO3 is a tool for creating native Python extension modules in Rust.
- End users use Autonomy in their Python code.
- Run any python related commands inside `source/python`.
- Indent all code using 2 spaces.
- Prefer `from ... import ...` over importing the full module.
- Run `make format` to format code.
- Use `uv` commands to manage the project.
- The venv for is always at .venv at the root of this repo.
- To add python deps: `cd source/python && uv add --active <package>`
- To remove python deps: `cd source/python && uv remove --active <package>`
- IMPORTANT: always use `--active` with `uv` commands.
- IMPORTANT: always refer `source/python/Makefile` for the right build commands.

## Guidelines for developing in Rust

1. Rust crates live in the path `source/rust`.
2. Run any rust related commands inside `source/rust`.

## Guidelines for writing commit messages

- Use imperative mood and active voice
- Start the subject line with a verb: "Add", "Fix", "Update", "Remove", "Refactor"
- Keep the subject line under 50 characters
- Capitalize the first letter of the subject line
- Do not end the subject line with a period
- Separate the subject from the body with a blank line
- Wrap the body at 72 characters
- Use the body to explain what and why, not how
- Focus on the change itself, not the process of making it
- Write as if completing the sentence: "If applied, this commit will..."

## Documentation
- Don't create too many summary documents and markdown files.

### Use Active Verb Forms in Documentation

When writing comments, docstrings, commit messages, or documentation, prefer **active verb phrases** over **nominalized noun phrases**.

#### Why

- Active verbs are clearer and more direct
- They specify *who/what* performs the action
- They reduce ambiguity
- They're easier to scan and understand

#### Examples

| ❌ Avoid (Nominalized) | ✅ Prefer (Active) |
|------------------------|-------------------|
| User authentication handling | Authenticate users |
| WebSocket connection management | Manage WebSocket connections |
| Error logging and reporting | Log and report errors |
| Data validation | Validate data |
| Cache invalidation | Invalidate cache |
| Request processing | Process requests |
| for downloading/uploading files | to download and upload files |
| for extracting audio | to extract audio |
| for translating text | to translate text |

#### Prefer "to + verb" Over "for + gerund"

When describing what a module or function does, prefer infinitive phrases ("to + verb") over gerund phrases ("for + gerund"):

| ❌ Avoid | ✅ Prefer |
|----------|----------|
| "provides functions for extracting audio" | "provides functions to extract audio" |
| "for downloading and uploading files" | "to download and upload files" |
| "for managing webhooks" | "to manage webhooks" |

#### Apply To

- Function/method docstrings
- TODO comments
- Commit messages
- PR descriptions
- README sections
- Inline comments explaining intent

#### Exceptions

Nominalized forms are acceptable for:
- Class names (`ConnectionManager`, `RequestHandler`)
- Module names
- Category labels in structured docs
- When the noun form is the actual domain term

#### Quick Test

If you can ask **"Who does what?"** and rewrite to answer that question with a subject + verb, use the active form.
