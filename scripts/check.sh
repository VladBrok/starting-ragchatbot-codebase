#!/bin/bash

# Code quality check script
echo "🔍 Running code quality checks..."

echo "📝 Checking code formatting with black..."
if ! uv run black --check .; then
    echo "❌ Code formatting issues found. Run './scripts/format.sh' to fix them."
    exit 1
fi

echo "🧪 Running tests..."
if ! uv run pytest backend/tests/ -v; then
    echo "❌ Tests failed."
    exit 1
fi

echo "✅ All quality checks passed!"