"""Datasets, the fixed validation carve, and the fixed tokenizer.

Vision uses the official CIFAR-10 and MNIST training images. A validation
split of 5000 images is carved from that training split with seed 123 and
saved. The official test split is a separate constructor and is not used to
choose the learning rate.

The language-model corpus is WikiText-103 raw, with WikiText-2 raw as the
fallback if WikiText-103 cannot be stored. The tokenizer is the fixed GPT-2
byte-pair encoding from tiktoken. It is not fit on these files. Train,
validation, and test files are encoded separately and never concatenated.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import torch

VAL_SEED = 123
VAL_COUNT = 5000
WIKITEXT_103 = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
WIKITEXT_2 = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"


def carve_validation(n: int, n_val: int, seed: int) -> tuple[list[int], list[int]]:
    if n_val <= 0 or n_val >= n:
        raise ValueError(f"validation count {n_val} does not fit in {n}")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    if set(val_idx) & set(train_idx):
        raise AssertionError("validation indices overlap training indices")
    if len(val_idx) + len(train_idx) != n:
        raise AssertionError("split does not cover the official training set")
    return train_idx, val_idx


def load_or_make_split(path: Path, n: int, n_val: int = VAL_COUNT, seed: int = VAL_SEED) -> dict:
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["n"] != n or saved["seed"] != seed or saved["n_val"] != n_val:
            raise AssertionError(f"saved split at {path} does not match this dataset")
        overlap = set(saved["train_idx"]) & set(saved["val_idx"])
        if overlap:
            raise AssertionError("saved split overlaps")
        return saved
    train_idx, val_idx = carve_validation(n, n_val, seed)
    saved = {"n": n, "n_val": n_val, "seed": seed, "train_idx": train_idx, "val_idx": val_idx}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved))
    return saved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    subprocess.check_call(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "2",
            "--retry-delay",
            "2",
            "--max-time",
            "900",
            "-o",
            str(tmp),
            url,
        ]
    )
    if tmp.stat().st_size > 2_000_000_000:
        tmp.unlink()
        raise RuntimeError(f"download exceeded 2GB: {url}")
    tmp.replace(dest)


def prepare_wikitext(root: Path, prefer_103: bool = True) -> dict:
    """Return token files' parent and which corpus was stored.

    The zip and the extracted text stay under root. A second call reuses them.
    """
    root.mkdir(parents=True, exist_ok=True)
    choices = [("wikitext-103-raw-v1", WIKITEXT_103)]
    if prefer_103:
        choices.append(("wikitext-2-raw-v1", WIKITEXT_2))
    else:
        choices = [("wikitext-2-raw-v1", WIKITEXT_2)]
    # Try the larger corpus first. Fall back only before any training.
    errors = []
    for name, url in choices:
        try:
            return _prepare_one(root, name, url)
        except Exception as exc:  # noqa: BLE001 — fallback is the pre-registered alternative
            errors.append(f"{name}: {exc}")
            if name.startswith("wikitext-2"):
                break
    raise RuntimeError("no WikiText corpus could be stored: " + " | ".join(errors))


def _prepare_one(root: Path, name: str, url: str) -> dict:
    folder = root / name
    train = folder / "wiki.train.raw"
    valid = folder / "wiki.valid.raw"
    test = folder / "wiki.test.raw"
    if not (train.exists() and valid.exists() and test.exists()):
        zpath = root / f"{name}.zip"
        if not zpath.exists():
            _download(url, zpath)
        with zipfile.ZipFile(zpath) as archive:
            archive.extractall(root)
        if not train.exists():
            # Some zips unpack into the current name already; others add one directory.
            found = list(root.rglob("wiki.train.raw"))
            if not found:
                raise FileNotFoundError(f"wiki.train.raw missing after extracting {name}")
            folder = found[0].parent
            train = folder / "wiki.train.raw"
            valid = folder / "wiki.valid.raw"
            test = folder / "wiki.test.raw"
    return {
        "name": name,
        "url": url,
        "dir": str(folder),
        "train": str(train),
        "valid": str(valid),
        "test": str(test),
        "zip_sha256": _sha256(root / f"{name}.zip") if (root / f"{name}.zip").exists() else None,
    }


def encode_file(path: Path, encoder) -> torch.Tensor:
    text = Path(path).read_text(encoding="utf-8")
    ids = encoder.encode(text)
    if len(ids) < 2:
        raise RuntimeError(f"{path} encoded to {len(ids)} tokens")
    return torch.tensor(ids, dtype=torch.int32)


def pack_sequences(ids: torch.Tensor, seq_len: int) -> torch.Tensor:
    n = (int(ids.numel()) // seq_len) * seq_len
    if n < seq_len:
        raise RuntimeError(f"only {ids.numel()} tokens, need at least {seq_len}")
    return ids[:n].view(-1, seq_len)


def add_one_unigram(train_ids: torch.Tensor, vocab: int) -> torch.Tensor:
    counts = torch.bincount(train_ids.long().view(-1), minlength=vocab).float()
    return (counts + 1.0) / (counts.sum() + vocab)
