#!/usr/bin/env bash
# Publish ultimate-utils to PyPI.
#
# Usage:
#   ./scripts/publish_to_pypi.sh          # upload to production PyPI
#   ./scripts/publish_to_pypi.sh --test   # upload to test.pypi.org
#
# Requires: ~/keys/push-pypi-all.txt (production) or ~/keys/pypi_test_token.txt (test)
set -euo pipefail

cd "$(dirname "$0")/.."

PYPI_TOKEN_FILE="$HOME/keys/push-pypi-all.txt"
REPO_URL="https://upload.pypi.org/legacy/"
PROJECT_URL="https://pypi.org/project/ultimate-utils/"

if [[ "${1:-}" == "--test" ]]; then
    PYPI_TOKEN_FILE="$HOME/keys/pypi_test_token.txt"
    REPO_URL="https://test.pypi.org/legacy/"
    PROJECT_URL="https://test.pypi.org/project/ultimate-utils/"
    echo "==> Uploading to TEST PyPI"
else
    echo "==> Uploading to PRODUCTION PyPI"
fi

for module_name in build twine; do
    if ! python -c "import ${module_name}" >/dev/null 2>&1; then
        echo "ERROR: Missing Python module '${module_name}' needed for publishing."
        echo "Install publishing tools with: python -m pip install build twine"
        exit 1
    fi
done

if [[ ! -f "$PYPI_TOKEN_FILE" ]]; then
    echo "ERROR: PyPI token not found at $PYPI_TOKEN_FILE"
    echo "Generate one at https://pypi.org/manage/account/token/"
    echo "Then: echo 'pypi-YOUR_TOKEN' > $PYPI_TOKEN_FILE && chmod 600 $PYPI_TOKEN_FILE"
    exit 1
fi

PYPI_TOKEN="$(cat "$PYPI_TOKEN_FILE")"

echo "==> Cleaning old builds..."
rm -rf build dist *.egg-info

echo "==> Building distribution..."
python -m build --sdist --wheel

echo "==> Checking distribution..."
python -m twine check dist/*

echo "==> Uploading..."
python -m twine upload \
    --repository-url "$REPO_URL" \
    --username __token__ \
    --password "$PYPI_TOKEN" \
    dist/*

echo "==> Cleaning up..."
rm -rf build dist *.egg-info

echo "==> Done! Check $PROJECT_URL"
