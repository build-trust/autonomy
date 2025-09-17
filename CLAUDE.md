# Instructions for Agents

These instructions define how Agents should assist with this project.

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

## Guidelines for creating and running examples

- Refer existing examples in the `examples`.
- To run examples in the `examples` durectory:
  - Use `autonomy --rm` in that directory.
  - The HTTP API is then available at `http://localhost:32100`.
  - Logs are then available at `http://localhost:32101`.
- To pick models, look at the list of models defined in `source/python/python/autonomy/models`


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

## Guidelines for developing in Rust

1. Rust crates live in the path `source/rust`.
2. Run any rust related commands inside `source/rust`.
