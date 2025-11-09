#!/bin/bash
set -e

echo "Running all checks and tests..."
echo ""

echo "=== Running pytest ==="
uv run python -m pytest

echo ""
echo "=== Running pyright ==="
uv run pyright

echo ""
echo "=== Running ruff check ==="
uv run ruff check

echo ""
echo "✓ All checks passed!"
