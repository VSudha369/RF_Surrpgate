#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PHASE4_VERSION = "1.0.0"
PHASE4_LOCAL_ROOT = Path("/content/surrogate_xai_v2/04_phase4_wisig_teacher_representation_prestudy")
PHASE4_DRIVE_ROOT = Path("/content/drive/MyDrive/Surrogate_XAI_V2/04_PHASE4_WISIG_TEACHER_REPRESENTATION_PRESTUDY")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, block: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def assert_local_hot_path(path: Path) -> None:
    p = path.resolve()
    if str(p).startswith("/content/drive/"):
        raise RuntimeError(f"Scientific hot path must not be on Drive: {p}")
    if not str(p).startswith("/content/"):
        raise RuntimeError(f"Expected /content local scientific path: {p}")


def copy_tree_verified(local_root: Path, drive_root: Path) -> Dict[str, Any]:
    """Persist a completed local artifact tree to Drive with file-level SHA verification."""
    if str(local_root.resolve()).startswith("/content/drive/"):
        raise RuntimeError("Source artifact tree must be local")
    if not str(drive_root.resolve()).startswith("/content/drive/"):
        raise RuntimeError("Destination artifact tree must be on mounted Drive")
    drive_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in sorted(p for p in local_root.rglob("*") if p.is_file()):
        rel = src.relative_to(local_root)
        dst = drive_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".partial")
        tmp.unlink(missing_ok=True)
        shutil.copyfile(src, tmp)
        expected = sha256_file(src)
        observed = sha256_file(tmp)
        if observed != expected:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Stage-out SHA mismatch: {rel}")
        os.replace(tmp, dst)
        rows.append({"path": str(rel), "bytes": src.stat().st_size, "sha256": expected})
    return {"files": rows, "file_count": len(rows), "completed_utc": utc_now()}
