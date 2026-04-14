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

if [[ "${1:-}" == "--test" ]]; then
    PYPI_TOKEN_FILE="$HOME/keys/pypi_test_token.txt"
    REPO_URL="https://test.pypi.org/legacy/"
    echo "==> Uploading to TEST PyPI"
else
    echo "==> Uploading to PRODUCTION PyPI"
fi

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
python setup.py sdist bdist_wheel

echo "==> Checking distribution..."
twine check dist/*

echo "==> Uploading..."
twine upload \
    --repository-url "$REPO_URL" \
    --username __token__ \
    --password "$PYPI_TOKEN" \
    dist/*

echo "==> Cleaning up..."
rm -rf build dist *.egg-info

echo "==> Done! Check https://pypi.org/project/ultimate-utils/"
