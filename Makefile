.PHONY: build test test-rust test-python lint fmt clean develop check

# Build the Rust extension and install into venv
build:
	maturin develop --release

# Run all tests
test: test-rust test-python

test-rust:
	cargo test

test-python:
	pytest tests/python -v

# Lint everything
lint:
	cargo clippy -- -D warnings
	cargo fmt --check

# Format
fmt:
	cargo fmt

# Full check (lint + test)
check: lint test

# Build for development (debug mode, faster compile)
develop:
	maturin develop

# Clean build artifacts
clean:
	cargo clean
	rm -rf .pytest_cache __pycache__ dist build *.egg-info

# CLI shortcuts
detect:
	fatcrash detect --asset BTC --source sample

backtest:
	fatcrash backtest --asset BTC --start 2017-01-01 --end 2018-06-01

# Setup: create venv and install everything
setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install maturin numpy pandas scipy httpx pyarrow pytest rich typer plotly matplotlib
	. .venv/bin/activate && maturin develop --release
