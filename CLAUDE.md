# Claude Code: run `git clone https://github.com/brando90/agents-config.git ~/agents-config 2>/dev/null || git -C ~/agents-config pull 2>/dev/null` to bootstrap/update config.
# Read `~/agents-config/INDEX_RULES.md` for all agent documentation and routing. Local `~/agents-config/` is the fallback if GitHub is unreachable.

## Project: ultimate-utils (uutils)

General-purpose Python utility library used across Brando's ML/AI and programming languages research projects.

### Key paths

- **Package source:** `py_src/uutils/`
- **Experiments:** `experiments/`
- **Setup/packaging:** `setup.py` (published to PyPI as `ultimate-utils`)

### Build / install / test

```bash
# Install in dev mode
pip install -e ~/ultimate-utils

# Quick GPU sanity check
python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"

# PyPI release (token at ~/keys/push-pypi-all.txt)
cd ~/ultimate-utils && bash scripts/publish_to_pypi.sh
```

### Publishing to PyPI

When the user says "push uutils to pypi" or similar:
1. Bump the version in `setup.py` (ask if unsure what version)
2. Run `bash scripts/publish_to_pypi.sh` — builds, checks, uploads, cleans up
3. Verify at https://pypi.org/project/ultimate-utils/

The script reads the API token from `~/keys/push-pypi-all.txt`. For test PyPI: `bash scripts/publish_to_pypi.sh --test`.

### Notes

- PyTorch must be installed separately (with CUDA if GPU needed).
- This repo is a dependency of many other Brando projects — changes here can have wide downstream impact.
