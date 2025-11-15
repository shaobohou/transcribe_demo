#!/bin/bash
set -e

# Parse arguments
USE_CI_PROJECT=false
for arg in "$@"; do
  case $arg in
    --ci)
      USE_CI_PROJECT=true
      shift
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--ci]"
      echo "  --ci: Use CI project (CPU-only, avoids CUDA downloads)"
      exit 1
      ;;
  esac
done

# Set UV command prefix based on mode
if [ "$USE_CI_PROJECT" = true ]; then
  UV_CMD="uv --project ci run"
  echo "Running all checks and tests (CI mode - CPU only)..."
else
  UV_CMD="uv run"
  echo "Running all checks and tests..."
fi
echo ""

echo "=== Running pytest ==="
$UV_CMD python -m pytest

echo ""
echo "=== Running pyright ==="
$UV_CMD pyright

echo ""
echo "=== Running ruff check ==="
$UV_CMD ruff check

echo ""
echo "✓ All checks passed!"
