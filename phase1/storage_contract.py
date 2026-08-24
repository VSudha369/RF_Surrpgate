#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, shutil, tarfile, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DRIVE_PREFIX = Path("/content/drive")
LOCAL_PREFIX = Path("/content")

def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
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

def assert_local_hot_path(path: Path) -> None:
    p = str(path.resolve())
    if p.startswith(str(DRIVE_PREFIX.resolve())):
        raise RuntimeError(f"HOT-PATH VIOLATION: Google Drive path is forbidden here: {p}")
    if not p.startswith("/content/"):
        raise RuntimeError(f"HOT-PATH VIOLATION: expected local /content path, got: {p}")

def copy_file_verified(src: Path, dst: Path, expected_sha256: Optional[str] = None,
                       chunk_size: int = 16 * 1024 * 1024) -> Dict[str, Any]:
    if not src.is_file():
        raise FileNotFoundError(src)
    # Direction-specific callers enforce Drive/local policy.
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_suffix(dst.suffix + ".partial")
    if partial.exists():
        partial.unlink()

    h = hashlib.sha256()
    n = 0
    t0 = time.perf_counter()
    with src.open("rb") as fi, partial.open("wb", buffering=0) as fo:
        while True:
            b = fi.read(chunk_size)
            if not b:
                break
            fo.write(b)
            h.update(b)
            n += len(b)
        fo.flush()
        os.fsync(fo.fileno())
    elapsed = time.perf_counter() - t0
    digest = h.hexdigest()

    if expected_sha256 and digest.lower() != expected_sha256.lower():
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA mismatch for {src}: expected={expected_sha256}, observed={digest}"
        )
    os.replace(partial, dst)
    return {
        "source": str(src), "destination": str(dst), "bytes": n,
        "sha256": digest, "elapsed_s": elapsed,
        "MiB_s": (n / 1024**2) / max(elapsed, 1e-9),
    }

def stage_in_file(drive_path: Path, local_dir: Path,
                  expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    if not str(drive_path.resolve()).startswith(str(DRIVE_PREFIX.resolve())):
        raise RuntimeError(f"stage_in_file source must be on mounted Drive: {drive_path}")
    assert_local_hot_path(local_dir)
    return copy_file_verified(
        drive_path, local_dir / drive_path.name, expected_sha256=expected_sha256
    )

def stage_in_directory_small_files(drive_dir: Path, local_dir: Path,
                                   patterns: Iterable[str] = ("*.npy", "*.json")) -> Dict[str, Any]:
    """For small frozen manifests/split arrays only. Never use for bulk RF signal data."""
    if not drive_dir.is_dir():
        raise FileNotFoundError(drive_dir)
    assert_local_hot_path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for pattern in patterns:
        for src in sorted(drive_dir.glob(pattern)):
            if src.is_file():
                copied.append(copy_file_verified(src, local_dir / src.name))
    return {"source": str(drive_dir), "destination": str(local_dir), "files": copied}

def create_transport_packs(source_root: Path, files: List[Path], out_dir: Path,
                           target_pack_bytes: int = 2 * 1024**3) -> List[Dict[str, Any]]:
    """Create uncompressed TAR packs for fast sequential Drive transfer."""
    assert_local_hot_path(source_root)
    assert_local_hot_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: List[List[Path]] = []
    cur: List[Path] = []
    cur_bytes = 0
    for p in sorted(files, key=lambda x: str(x.relative_to(source_root))):
        size = p.stat().st_size
        if cur and cur_bytes + size > target_pack_bytes:
            groups.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(p)
        cur_bytes += size
    if cur:
        groups.append(cur)

    manifest = []
    for i, group in enumerate(groups):
        pack = out_dir / f"pack_{i:04d}.tar"
        partial = pack.with_suffix(".tar.partial")
        if partial.exists():
            partial.unlink()
        with tarfile.open(partial, mode="w") as tf:
            for p in group:
                tf.add(p, arcname=str(p.relative_to(source_root)), recursive=False)
        os.replace(partial, pack)
        manifest.append({
            "pack_id": i,
            "filename": pack.name,
            "path": str(pack),
            "bytes": pack.stat().st_size,
            "sha256": sha256_file(pack),
            "members": [str(p.relative_to(source_root)) for p in group],
        })
    return manifest


def safe_extract_tar(tar_path: Path, extract_root: Path) -> None:
    """Extract TAR locally with path-traversal protection."""
    assert_local_hot_path(tar_path)
    assert_local_hot_path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    root = extract_root.resolve()
    with tarfile.open(tar_path, mode="r") as tf:
        for member in tf.getmembers():
            target = (extract_root / member.name).resolve()
            if not str(target).startswith(str(root) + os.sep) and target != root:
                raise RuntimeError(f"Unsafe TAR member path: {member.name}")
        tf.extractall(extract_root)

def stage_in_transport_pack(drive_pack: Path, expected_sha256: str,
                            local_staging_dir: Path, extract_root: Path,
                            delete_local_tar_after_extract: bool = True) -> Dict[str, Any]:
    """
    Reusable later-phase stage-in:
    Drive TAR -> local TAR -> SHA verify -> safe local extraction.
    No downstream DataLoader should ever read the Drive TAR directly.
    """
    if not str(drive_pack.resolve()).startswith(str(DRIVE_PREFIX.resolve())):
        raise RuntimeError(f"Transport source must be on Drive: {drive_pack}")
    assert_local_hot_path(local_staging_dir)
    assert_local_hot_path(extract_root)
    copy_info = stage_in_file(
        drive_pack, local_staging_dir, expected_sha256=expected_sha256
    )
    local_tar = local_staging_dir / drive_pack.name
    safe_extract_tar(local_tar, extract_root)
    if delete_local_tar_after_extract:
        local_tar.unlink(missing_ok=True)
    return {
        "drive_pack": str(drive_pack),
        "expected_sha256": expected_sha256,
        "copy": copy_info,
        "extract_root": str(extract_root),
        "local_tar_deleted": bool(delete_local_tar_after_extract),
    }

def copy_artifact_tree_to_drive(local_root: Path, drive_root: Path) -> Dict[str, Any]:
    """
    Stage-out completed Phase artifact. Bulk transport packs are copied as large files;
    compact manifests are copied individually. Does not delete local data.
    """
    assert_local_hot_path(local_root)
    if not str(drive_root.resolve()).startswith(str(DRIVE_PREFIX.resolve())):
        raise RuntimeError(f"stage-out destination must be Drive: {drive_root}")
    drive_root.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in sorted(local_root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(local_root)
        dst = drive_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        local_sha = sha256_file(src)
        info = copy_file_verified(src, dst, expected_sha256=local_sha)
        copied.append({"relative_path": str(rel), **info})
    return {"local_root": str(local_root), "drive_root": str(drive_root), "files": copied}
