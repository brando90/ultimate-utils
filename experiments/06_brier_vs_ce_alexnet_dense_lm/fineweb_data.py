"""FineWeb-Edu sample used for the matched token-budget replication.

The pinned revision is sample-10BT at
87f09149ef4734204d70ed1d046ddc9ca3f2b8f9. Shards are about 2 GB. Only the
shards required to fill the token quotas are downloaded. A file that is not
parquet, or whose size does not match the pinned revision, is deleted.
"""

from __future__ import annotations

import array
import hashlib
import json
import subprocess
import time
from pathlib import Path

REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
DATASET = "HuggingFaceFW/fineweb-edu"
SHARD_BYTES = {
    "sample/10BT/000_00000.parquet": 2_152_819_114,
    "sample/10BT/001_00000.parquet": 2_152_222_432,
    "sample/10BT/002_00000.parquet": 2_151_796_315,
    "sample/10BT/003_00000.parquet": 2_152_437_524,
    "sample/10BT/004_00000.parquet": 2_152_338_550,
    "sample/10BT/005_00000.parquet": 2_152_189_947,
    "sample/10BT/006_00000.parquet": 2_152_689_867,
    "sample/10BT/007_00000.parquet": 2_150_686_637,
    "sample/10BT/008_00000.parquet": 2_151_274_846,
    "sample/10BT/009_00000.parquet": 2_151_913_277,
    "sample/10BT/010_00000.parquet": 2_152_798_864,
    "sample/10BT/011_00000.parquet": 2_152_323_681,
    "sample/10BT/012_00000.parquet": 2_152_069_689,
    "sample/10BT/013_00000.parquet": 540_632_672,
}


def _urls(relpath: str) -> list[str]:
    pinned = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{relpath}"
    # Same file name on the dataset's default branch. Accepted only when the byte size matches the pin.
    default = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{relpath}"
    return [pinned, default]


def _download_one(relpath: str, dest: Path) -> None:
    expected = SHARD_BYTES[relpath]
    if dest.exists() and dest.stat().st_size == expected and dest.open("rb").read(4) == b"PAR1":
        print(f"shard present {dest.name} {expected}", flush=True)
        return
    if dest.exists():
        dest.unlink()
    errors = []
    for attempt, url in enumerate(_urls(relpath), start=1):
        tmp = dest.with_suffix(dest.suffix + ".partial")
        if tmp.exists():
            tmp.unlink()
        print(f"download attempt {attempt} {url}", flush=True)
        proc = subprocess.run(
            [
                "curl", "-L", "--fail", "--retry", "2", "--retry-delay", "5",
                "--speed-limit", "1024", "--speed-time", "120",
                "-o", str(tmp), url,
            ],
            text=True,
        )
        if proc.returncode != 0 or not tmp.exists():
            errors.append(f"{url} exit {proc.returncode}")
            if tmp.exists():
                tmp.unlink()
            time.sleep(15 * attempt)
            continue
        size = tmp.stat().st_size
        magic = tmp.open("rb").read(4)
        if size != expected or magic != b"PAR1":
            tmp.unlink()
            errors.append(f"{url} size {size} magic {magic!r}")
            time.sleep(15 * attempt)
            continue
        tmp.replace(dest)
        print(f"shard ok {dest.name} {size}", flush=True)
        return
    raise RuntimeError(f"could not download {relpath}: {errors}")


def _bucket(doc_id: str) -> int:
    digest = hashlib.sha256(doc_id.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % 1000


def _as_id(value) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int):
        return str(value)
    return None


def _split_name(bucket: int) -> str | None:
    if bucket <= 7:
        return "valid"
    if bucket <= 15:
        return "test"
    return "train"


def collect_tokens(data_root: Path, train_tokens: int, heldout_tokens: int, seq_len: int) -> dict:
    import tiktoken
    import pyarrow.parquet as pq

    encoder = tiktoken.get_encoding("gpt2")
    if encoder.n_vocab != 50257:
        raise RuntimeError(f"gpt2 vocabulary is {encoder.n_vocab}")
    targets = {
        "train": ((train_tokens + seq_len - 1) // seq_len) * seq_len,
        "valid": ((heldout_tokens + seq_len - 1) // seq_len) * seq_len,
        "test": ((heldout_tokens + seq_len - 1) // seq_len) * seq_len,
    }
    buffers = {name: array.array("I") for name in targets}
    seen: set[str] = set()
    ident = hashlib.sha256()
    counts = {"docs": 0, "duplicates": 0, "empty": 0, "accepted": 0}
    used_shards = []
    data_root.mkdir(parents=True, exist_ok=True)

    def full() -> bool:
        return all(len(buffers[name]) >= targets[name] for name in targets)

    for relpath in sorted(SHARD_BYTES):
        if full():
            break
        dest = data_root / Path(relpath).name
        _download_one(relpath, dest)
        used_shards.append({"path": relpath, "bytes": SHARD_BYTES[relpath]})
        parquet = pq.ParquetFile(dest)
        names = parquet.schema_arrow.names
        if "text" not in names or "id" not in names:
            raise RuntimeError(f"{dest.name} columns are {names}")
        for batch in parquet.iter_batches(batch_size=1024, columns=["text", "id"]):
            texts = batch.column(0).to_pylist()
            ids = batch.column(1).to_pylist()
            for text, raw_id in zip(texts, ids):
                counts["docs"] += 1
                doc_id = _as_id(raw_id)
                if doc_id is None:
                    counts["empty"] += 1
                    continue
                if doc_id in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(doc_id)
                if not isinstance(text, str) or not text.strip():
                    counts["empty"] += 1
                    continue
                name = _split_name(_bucket(doc_id))
                if len(buffers[name]) >= targets[name]:
                    continue
                piece = text if text.endswith("\n") else text + "\n"
                tokens = encoder.encode(piece)
                room = targets[name] - len(buffers[name])
                buffers[name].extend(tokens[:room])
                ident.update(doc_id.encode("utf-8") + b"\n")
                counts["accepted"] += 1
            if counts["docs"] % 20000 == 0:
                print(
                    "scan "
                    + " ".join(f"{name} {len(buffers[name])}/{targets[name]}" for name in targets),
                    flush=True,
                )
            if full():
                break
        print(f"finished {dest.name} accepted {counts['accepted']}", flush=True)
    if not full():
        raise RuntimeError(f"token quotas not filled: {[len(buffers[name]) for name in targets]} counts {counts}")
    return {
        "buffers": buffers,
        "targets": targets,
        "counts": counts,
        "id_sha256": ident.hexdigest(),
        "shards": used_shards,
        "tokenizer": "tiktoken gpt2",
        "vocab": encoder.n_vocab,
        "revision": REVISION,
    }
