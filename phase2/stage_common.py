#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DRIVE_PREFIX = Path("/content/drive")
DEFAULT_SAFETY_BYTES = 5 * 1024**3


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk_size), b""):
            h.update(b)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def require_drive_mounted() -> None:
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError(
            "Google Drive is not mounted. Mount it from a Colab notebook cell first:\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive', force_remount=False)"
        )


def assert_local_path(path: Path) -> None:
    p = path.resolve()
    if str(p).startswith(str(DRIVE_PREFIX.resolve())):
        raise RuntimeError(f"Hot/local path must not be on Drive: {p}")
    if not str(p).startswith("/content/"):
        raise RuntimeError(f"Expected /content local path, got: {p}")


def assert_drive_path(path: Path) -> None:
    p = path.resolve()
    if not str(p).startswith(str(DRIVE_PREFIX.resolve())):
        raise RuntimeError(f"Expected mounted Drive path, got: {p}")


def load_registry(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        reg = json.load(f)
    if reg.get("registry_version") != "1.0.0":
        raise RuntimeError(f"Unsupported registry version: {reg.get('registry_version')}")
    if set(reg.get("datasets", {})) != {"wisig", "radioml2016", "radioml2018"}:
        raise RuntimeError(
            "Frozen registry must contain exactly wisig/radioml2016/radioml2018"
        )
    return reg


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pack_map(packs: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for p in packs:
        fn = p["filename"]
        if fn in out:
            raise RuntimeError(f"Duplicate pack filename: {fn}")
        out[fn] = p
    return out


def copy_drive_file_verified(
    src: Path,
    dst: Path,
    expected_sha256: str,
    expected_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    require_drive_mounted()
    assert_drive_path(src)
    assert_local_path(dst.parent)
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_suffix(dst.suffix + ".partial")
    partial.unlink(missing_ok=True)

    h = hashlib.sha256()
    n = 0
    t0 = time.perf_counter()
    with src.open("rb") as fi, partial.open("wb", buffering=0) as fo:
        for b in iter(lambda: fi.read(16 * 1024 * 1024), b""):
            fo.write(b)
            h.update(b)
            n += len(b)
        fo.flush()
        os.fsync(fo.fileno())
    elapsed = time.perf_counter() - t0
    digest = h.hexdigest()

    if expected_bytes is not None and n != expected_bytes:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{src.name}: copied bytes {n} != {expected_bytes}")
    if digest != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{src.name}: SHA mismatch expected={expected_sha256}, observed={digest}"
        )
    os.replace(partial, dst)
    return {
        "source": str(src),
        "destination": str(dst),
        "bytes": n,
        "sha256": digest,
        "elapsed_s": elapsed,
        "MiB_s": (n / 1024**2) / max(elapsed, 1e-9),
    }


def inspect_tar_members(tar_path: Path) -> List[str]:
    assert_local_path(tar_path)
    names, seen = [], set()
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            if m.issym() or m.islnk() or m.isdev() or m.isfifo():
                raise RuntimeError(f"Unsafe TAR member type: {m.name}")
            if not (m.isfile() or m.isdir()):
                raise RuntimeError(f"Unsupported TAR member type: {m.name}")
            pure = Path(m.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise RuntimeError(f"Unsafe TAR path: {m.name}")
            if m.isfile():
                if m.name in seen:
                    raise RuntimeError(f"Duplicate TAR file member: {m.name}")
                seen.add(m.name)
                names.append(m.name)
    return names


def safe_extract_tar(
    tar_path: Path,
    extract_root: Path,
    expected_members: Optional[List[str]] = None,
) -> None:
    assert_local_path(tar_path)
    assert_local_path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    actual = inspect_tar_members(tar_path)
    if expected_members is not None and set(actual) != set(expected_members):
        missing = sorted(set(expected_members) - set(actual))
        extra = sorted(set(actual) - set(expected_members))
        raise RuntimeError(
            f"TAR member-set mismatch for {tar_path.name}: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    root = extract_root.resolve()
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            target = (extract_root / m.name).resolve()
            if target != root and not str(target).startswith(str(root) + os.sep):
                raise RuntimeError(f"TAR path traversal blocked: {m.name}")
        tf.extractall(extract_root)


class DatasetStageLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(
                f"Stage lock already exists: {self.path}. "
                "Use clean only if no stage-in is running."
            )
        os.write(
            self.fd,
            f"pid={os.getpid()}\ncreated_utc={utc_now()}\n".encode(),
        )
        os.fsync(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def space_preflight(entry: Dict[str, Any]) -> Dict[str, Any]:
    total_pack = sum(int(p["bytes"]) for p in entry["packs"])
    max_pack = max(int(p["bytes"]) for p in entry["packs"])
    required = total_pack + max_pack + DEFAULT_SAFETY_BYTES
    usage = shutil.disk_usage("/content")
    if usage.free < required:
        raise RuntimeError(
            f"Insufficient /content free space: free={usage.free/1024**3:.2f} GiB, "
            f"required={required/1024**3:.2f} GiB"
        )
    return {
        "free_bytes": usage.free,
        "required_bytes": required,
        "total_transport_bytes": total_pack,
        "largest_pack_bytes": max_pack,
        "safety_bytes": DEFAULT_SAFETY_BYTES,
    }
