#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from stage_common import (
    DatasetStageLock,
    assert_drive_path,
    assert_local_path,
    atomic_write_json,
    copy_drive_file_verified,
    inspect_tar_members,
    pack_map,
    read_json,
    safe_extract_tar,
    sha256_file,
    space_preflight,
    utc_now,
)
from stage_validation import (
    validate_drive_target,
    validate_local_canonical,
)


def stage_paths(
    registry: Dict[str, Any],
    dataset: str,
    entry: Dict[str, Any],
) -> Dict[str, Path]:
    base = Path(registry["local_stage_root"]) / dataset
    run_root = base / entry["phase1_run_id"]
    return {
        "base": base,
        "run_root": run_root,
        "canonical": run_root / "canonical",
        "marker": run_root / "LOCAL_STAGE_COMPLETE.json",
        "current": base / "current",
        "lock": base / ".stage.lock",
    }


def _update_current_symlink(current: Path, run_root: Path) -> None:
    current.parent.mkdir(parents=True, exist_ok=True)
    tmp = current.parent / f".current.tmp.{os.getpid()}"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(run_root.name)
    os.replace(tmp, current)


def stage_in_dataset(
    registry: Dict[str, Any],
    dataset: str,
    verification: str = "full",
    keep_tars: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    entry = registry["datasets"][dataset]
    drive_validation = validate_drive_target(
        entry,
        hash_packs=False,
    )
    space = space_preflight(entry)
    paths = stage_paths(registry, dataset, entry)
    base, final_root = paths["base"], paths["run_root"]
    assert_local_path(base)
    base.mkdir(parents=True, exist_ok=True)

    with DatasetStageLock(paths["lock"]):
        if final_root.exists():
            if not force:
                local = validate_local_canonical(
                    paths["canonical"],
                    entry,
                    verification=verification,
                )
                _update_current_symlink(paths["current"], final_root)
                return {
                    "dataset": dataset,
                    "status": "ALREADY_STAGED",
                    "local_canonical_root": str(paths["canonical"]),
                    "local_validation": local,
                }
            shutil.rmtree(final_root)

        for p in base.glob(".partial_*"):
            if p.is_dir():
                shutil.rmtree(p)

        partial = base / (
            f".partial_{entry['phase1_run_id']}_{os.getpid()}"
        )
        canonical = partial / "canonical"
        tars = partial / "_transport"
        provenance = partial / "provenance"
        canonical.mkdir(parents=True, exist_ok=True)
        tars.mkdir(parents=True, exist_ok=True)
        provenance.mkdir(parents=True, exist_ok=True)

        ds_root = Path(entry["drive_dataset_root"])
        tm = read_json(
            ds_root / "transport" / "TRANSPORT_MANIFEST.json"
        )
        atomic_write_json(
            provenance / "DRIVE_TARGET_VALIDATION.json",
            drive_validation,
        )
        atomic_write_json(
            provenance / "TRANSPORT_MANIFEST.json",
            tm,
        )
        atomic_write_json(
            provenance / "FROZEN_REGISTRY_ENTRY.json",
            entry,
        )

        copies = []
        expected_all = set()
        extracted_all = set()
        frozen_packs = pack_map(entry["packs"])

        for p in sorted(tm["packs"], key=lambda x: x["pack_id"]):
            fn = p["filename"]
            reg_pack = frozen_packs[fn]
            src = ds_root / "transport" / fn
            dst = tars / fn
            print(
                f"[STAGE-IN] {dataset}: copying {fn} "
                f"({reg_pack['bytes']/1024**3:.2f} GiB)..."
            )
            copy_info = copy_drive_file_verified(
                src,
                dst,
                expected_sha256=reg_pack["sha256"],
                expected_bytes=reg_pack["bytes"],
            )

            actual_members = inspect_tar_members(dst)
            expected_members = list(p["members"])
            if set(actual_members) != set(expected_members):
                raise RuntimeError(
                    f"{fn}: TAR members differ from transport manifest"
                )
            overlap = extracted_all.intersection(actual_members)
            if overlap:
                raise RuntimeError(
                    f"Duplicate members across packs: "
                    f"{sorted(overlap)[:10]}"
                )
            expected_all.update(expected_members)
            extracted_all.update(actual_members)

            print(
                f"[STAGE-IN] {dataset}: extracting {fn} locally..."
            )
            safe_extract_tar(
                dst,
                canonical,
                expected_members=expected_members,
            )
            if not keep_tars:
                dst.unlink(missing_ok=True)
            copies.append(copy_info)

        actual_files = {
            str(p.relative_to(canonical))
            for p in canonical.rglob("*")
            if p.is_file()
        }
        if actual_files != expected_all:
            missing = sorted(expected_all - actual_files)
            extra = sorted(actual_files - expected_all)
            raise RuntimeError(
                f"Extracted member-set mismatch: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

        print(
            f"[VERIFY] {dataset}: validating extracted canonical "
            f"shards ({verification})..."
        )
        local_validation = validate_local_canonical(
            canonical,
            entry,
            verification=verification,
        )

        marker = {
            "phase": "PHASE2_LOCAL_STAGE_MANAGER",
            "phase_version": "1.0.0",
            "dataset": dataset,
            "status": "LOCAL_STAGE_COMPLETE",
            "phase1_run_id": entry["phase1_run_id"],
            "phase1_version": entry["phase1_version"],
            "source_sha256": entry["source_sha256"],
            "registry_entry_sha256": hashlib.sha256(
                json.dumps(
                    entry,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "verification_level": verification,
            "local_validation": local_validation,
            "drive_validation": drive_validation,
            "space_preflight": space,
            "pack_copies": copies,
            "completed_utc": utc_now(),
        }
        atomic_write_json(
            partial / "LOCAL_STAGE_COMPLETE.json",
            marker,
        )

        os.replace(partial, final_root)
        _update_current_symlink(
            paths["current"],
            final_root,
        )

        return {
            "dataset": dataset,
            "status": "LOCAL_STAGE_COMPLETE",
            "local_run_root": str(final_root),
            "local_canonical_root": str(final_root / "canonical"),
            "current_symlink": str(paths["current"]),
            "verification_level": verification,
            "n_samples": entry["n_samples"],
            "seq_len": entry["seq_len"],
            "n_shards": entry["n_shards"],
        }


def validate_staged_dataset(
    registry: Dict[str, Any],
    dataset: str,
    verification: str = "manifest",
) -> Dict[str, Any]:
    entry = registry["datasets"][dataset]
    paths = stage_paths(registry, dataset, entry)
    if not paths["canonical"].is_dir():
        raise RuntimeError(
            f"Dataset not staged: {paths['canonical']}"
        )
    result = validate_local_canonical(
        paths["canonical"],
        entry,
        verification=verification,
    )
    return {
        "dataset": dataset,
        "status": "LOCAL_STAGE_VALID",
        "local_canonical_root": str(paths["canonical"]),
        "validation": result,
    }


def clean_staged_dataset(
    registry: Dict[str, Any],
    dataset: str,
    include_all_runs: bool = False,
) -> Dict[str, Any]:
    entry = registry["datasets"][dataset]
    paths = stage_paths(registry, dataset, entry)
    base = paths["base"]
    assert_local_path(base)

    removed = []
    paths["lock"].unlink(missing_ok=True)
    for p in base.glob(".partial_*"):
        if p.is_dir():
            shutil.rmtree(p)
            removed.append(str(p))

    if include_all_runs:
        if base.exists():
            shutil.rmtree(base)
            removed.append(str(base))
    else:
        if paths["run_root"].exists():
            shutil.rmtree(paths["run_root"])
            removed.append(str(paths["run_root"]))
        if paths["current"].exists() or paths["current"].is_symlink():
            paths["current"].unlink(missing_ok=True)

    return {
        "dataset": dataset,
        "removed": removed,
        "cleaned_utc": utc_now(),
    }


def resolve_local_canonical_root(
    registry: Dict[str, Any],
    dataset: str,
) -> Path:
    entry = registry["datasets"][dataset]
    paths = stage_paths(registry, dataset, entry)
    if not paths["marker"].is_file():
        raise RuntimeError(
            f"No LOCAL_STAGE_COMPLETE marker for {dataset}"
        )
    marker = read_json(paths["marker"])
    if marker.get("status") != "LOCAL_STAGE_COMPLETE":
        raise RuntimeError(
            f"Invalid local stage marker for {dataset}"
        )
    if marker.get("source_sha256") != entry["source_sha256"]:
        raise RuntimeError(
            f"Local stage marker source hash mismatch for {dataset}"
        )
    assert_local_path(paths["canonical"])
    return paths["canonical"]


def atomic_stage_out_file(
    local_file: Path,
    drive_file: Path,
) -> Dict[str, Any]:
    assert_local_path(local_file)
    assert_drive_path(drive_file.parent)
    if not local_file.is_file():
        raise FileNotFoundError(local_file)
    drive_file.parent.mkdir(parents=True, exist_ok=True)

    expected = sha256_file(local_file)
    partial = drive_file.with_suffix(
        drive_file.suffix + ".partial"
    )
    partial.unlink(missing_ok=True)
    shutil.copyfile(local_file, partial)
    observed = sha256_file(partial)
    if observed != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Stage-out SHA mismatch for {local_file}"
        )
    os.replace(partial, drive_file)
    return {
        "local_file": str(local_file),
        "drive_file": str(drive_file),
        "sha256": expected,
        "bytes": local_file.stat().st_size,
        "completed_utc": utc_now(),
    }
