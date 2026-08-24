#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from stage_common import (
    assert_drive_path,
    assert_local_path,
    pack_map,
    read_json,
    require_drive_mounted,
    sha256_file,
    utc_now,
)


def _validate_dataset_manifest(dm: Dict[str, Any], entry: Dict[str, Any]) -> None:
    checks = {
        "n_samples": (dm.get("n_samples"), entry["n_samples"]),
        "seq_len": (dm.get("seq_len"), entry["seq_len"]),
        "n_shards": (dm.get("n_shards"), entry["n_shards"]),
        "source_sha256": (dm.get("source_sha256"), entry["source_sha256"]),
        "phase_version": (dm.get("phase_version"), entry["phase1_version"]),
    }
    failures = [
        f"{k}: observed={o!r}, expected={e!r}"
        for k, (o, e) in checks.items()
        if o != e
    ]
    if failures:
        raise RuntimeError("DATASET_MANIFEST mismatch: " + " | ".join(failures))


def validate_drive_target(
    entry: Dict[str, Any],
    hash_packs: bool = False,
) -> Dict[str, Any]:
    require_drive_mounted()
    ds_root = Path(entry["drive_dataset_root"])
    assert_drive_path(ds_root)
    if not ds_root.is_dir():
        raise FileNotFoundError(ds_root)

    stage_complete = ds_root / "DATASET_STAGE_COMPLETE.json"
    transport_dir = ds_root / "transport"
    tm_path = transport_dir / "TRANSPORT_MANIFEST.json"
    canonical_dir = ds_root / "canonical_manifest"
    dm_path = canonical_dir / "DATASET_MANIFEST.json"
    cv_path = canonical_dir / "CANONICAL_VERIFICATION.json"

    missing = [
        str(p)
        for p in (stage_complete, tm_path, dm_path, cv_path)
        if not p.is_file()
    ]
    if missing:
        raise RuntimeError(f"Frozen Phase-1 target missing files: {missing}")

    tm = read_json(tm_path)
    dm = read_json(dm_path)
    cv = read_json(cv_path)
    expected_dataset = ds_root.name
    dataset = dm.get("dataset") or tm.get("dataset")
    if dataset != expected_dataset:
        raise RuntimeError(
            f"Dataset identity mismatch: {dataset!r} != {expected_dataset!r}"
        )

    _validate_dataset_manifest(dm, entry)

    if cv.get("verified") is not True:
        raise RuntimeError("CANONICAL_VERIFICATION is not verified=true")
    if (
        cv.get("n_samples") != entry["n_samples"]
        or cv.get("seq_len") != entry["seq_len"]
    ):
        raise RuntimeError("CANONICAL_VERIFICATION count/length mismatch")

    reg_packs = pack_map(entry["packs"])
    tm_packs = pack_map(tm.get("packs", []))
    if set(reg_packs) != set(tm_packs):
        raise RuntimeError("Transport pack set differs from frozen registry")
    if len(tm_packs) != entry["n_transport_packs"]:
        raise RuntimeError("Transport pack-count mismatch")

    pack_results = []
    for fn in sorted(reg_packs):
        rp, mp = reg_packs[fn], tm_packs[fn]
        if mp.get("bytes") != rp["bytes"] or mp.get("sha256") != rp["sha256"]:
            raise RuntimeError(f"{fn}: transport manifest differs from registry")
        drive_pack = transport_dir / fn
        if not drive_pack.is_file():
            raise FileNotFoundError(drive_pack)
        size = drive_pack.stat().st_size
        if size != rp["bytes"]:
            raise RuntimeError(f"{fn}: Drive size mismatch")
        digest = None
        if hash_packs:
            digest = sha256_file(drive_pack)
            if digest != rp["sha256"]:
                raise RuntimeError(f"{fn}: Drive SHA mismatch")
        pack_results.append(
            {
                "filename": fn,
                "bytes": size,
                "sha256_expected": rp["sha256"],
                "sha256_observed": digest,
            }
        )

    return {
        "dataset": expected_dataset,
        "drive_dataset_root": str(ds_root),
        "phase1_run_id": entry["phase1_run_id"],
        "phase1_version": entry["phase1_version"],
        "n_samples": entry["n_samples"],
        "seq_len": entry["seq_len"],
        "n_shards": entry["n_shards"],
        "n_transport_packs": entry["n_transport_packs"],
        "source_sha256": entry["source_sha256"],
        "transport_manifest_sha256": sha256_file(tm_path),
        "dataset_manifest_sha256": sha256_file(dm_path),
        "packs": pack_results,
        "hash_packs": bool(hash_packs),
        "validated_utc": utc_now(),
    }


def validate_local_canonical(
    canonical_root: Path,
    entry: Dict[str, Any],
    verification: str = "full",
) -> Dict[str, Any]:
    assert_local_path(canonical_root)
    dm_path = canonical_root / "DATASET_MANIFEST.json"
    cv_path = canonical_root / "CANONICAL_VERIFICATION.json"
    if not dm_path.is_file() or not cv_path.is_file():
        raise RuntimeError(
            "Missing local DATASET_MANIFEST/CANONICAL_VERIFICATION"
        )

    dm = read_json(dm_path)
    cv = read_json(cv_path)
    _validate_dataset_manifest(dm, entry)
    if cv.get("verified") is not True:
        raise RuntimeError("Local CANONICAL_VERIFICATION is not verified=true")

    shards = dm.get("shards", [])
    if len(shards) != entry["n_shards"]:
        raise RuntimeError(
            f"Local shard count {len(shards)} != {entry['n_shards']}"
        )

    expected_gid = 0
    n_total = 0
    labels = set()
    shard_results = []

    for r in shards:
        sig = canonical_root / r["signal_path"]
        meta = canonical_root / r["metadata_path"]
        if not sig.is_file() or not meta.is_file():
            raise RuntimeError(
                f"Missing shard files for shard_id={r.get('shard_id')}"
            )

        if verification == "full":
            if sha256_file(sig) != r["signal_sha256"]:
                raise RuntimeError(f"Signal SHA failure: {sig}")
            if sha256_file(meta) != r["metadata_sha256"]:
                raise RuntimeError(f"Metadata SHA failure: {meta}")

        x = np.load(sig, mmap_mode="r", allow_pickle=False)
        if (
            x.dtype != np.float32
            or x.ndim != 3
            or x.shape[1] != 2
            or x.shape[2] != entry["seq_len"]
        ):
            raise RuntimeError(
                f"Canonical layout/dtype failure: {sig} {x.shape} {x.dtype}"
            )
        if x.shape[0] != r["n_samples"]:
            raise RuntimeError(f"Shard sample-count failure: {sig}")

        with np.load(meta, allow_pickle=False) as m:
            if "global_index" not in m or "label" not in m:
                raise RuntimeError(
                    f"Metadata missing global_index/label: {meta}"
                )
            gids = m["global_index"]
            if len(gids) != x.shape[0]:
                raise RuntimeError(
                    f"Metadata/signal count mismatch: {meta}"
                )
            if len(gids) and int(gids[0]) != expected_gid:
                raise RuntimeError(
                    f"Global-index discontinuity at shard {r['shard_id']}: "
                    f"{int(gids[0])} != {expected_gid}"
                )
            if len(gids):
                expected_gid = int(gids[-1]) + 1
            labels.update(np.unique(m["label"]).astype(int).tolist())

        n_total += int(x.shape[0])
        shard_results.append(
            {
                "shard_id": int(r["shard_id"]),
                "n_samples": int(x.shape[0]),
                "shape": list(x.shape),
                "full_sha_verified": verification == "full",
            }
        )

    if n_total != entry["n_samples"]:
        raise RuntimeError(
            f"Local total samples {n_total} != {entry['n_samples']}"
        )
    if expected_gid != entry["n_samples"]:
        raise RuntimeError(
            f"Final global index {expected_gid} != {entry['n_samples']}"
        )
    if len(labels) != entry["expected_unique_label_count"]:
        raise RuntimeError(
            f"Unique label count {len(labels)} != "
            f"{entry['expected_unique_label_count']}"
        )

    return {
        "verified": True,
        "verification_level": verification,
        "n_samples": n_total,
        "seq_len": entry["seq_len"],
        "n_shards": len(shards),
        "unique_label_count": len(labels),
        "first_label": min(labels) if labels else None,
        "last_label": max(labels) if labels else None,
        "dataset_manifest_sha256": sha256_file(dm_path),
        "validated_utc": utc_now(),
        "shards": shard_results,
    }
