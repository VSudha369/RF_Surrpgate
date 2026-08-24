#!/usr/bin/env python3
"""
PHASE 1 — Dataset Canonicalization and Storage Pipeline v1.0.2

Adds source-authentic WiSig ManyTx support:
- source: ManyTx.pkl.zip -> ManyTx.pkl
- native schema: tx × receiver × capture_date × equalization
- native signal length: L=256
- deterministic flatten order and full nuisance metadata
- hard schema/count validation

RadioML behavior remains inherited from validated Phase 1 v1.0.0/v1.0.1.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from storage_contract import (
    atomic_write_json,
    assert_local_hot_path,
    copy_file_verified,
    create_transport_packs,
    sha256_file,
    stage_in_file,
    utc_now,
)
from wisig_manytx_adapter import (
    inspect_manytx_object,
    iter_manytx_object,
    load_manytx_pickle,
    resolve_manytx_member,
    safe_extract_manytx_pickle,
    validate_manytx_inspection,
)

HERE = Path(__file__).resolve().parent
BASE = HERE / "Phase1_Dataset_Canonicalization_v1_0_0.py"

spec = importlib.util.spec_from_file_location("phase1_v1_0_0_impl", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load Phase-1 base implementation: {BASE}")
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)

impl.VERSION = "1.0.2"

_ORIG_CANONICALIZE_ONE = impl.canonicalize_one
_ORIG_DISCOVER = impl.discover_candidates


def require_drive_mounted() -> None:
    mydrive = Path("/content/drive/MyDrive")
    if not mydrive.is_dir():
        raise RuntimeError(
            "Google Drive is not mounted. Run this in a Colab NOTEBOOK CELL first:\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive', force_remount=False)\n"
            "Then rerun the Phase-1 command."
        )


impl.mount_drive = require_drive_mounted


def discover_candidates_v102(max_depth: int = 5) -> Dict[str, List[str]]:
    out = _ORIG_DISCOVER(max_depth=max_depth)

    roots = [
        Path("/content/drive/MyDrive"),
        Path("/content/drive/MyDrive/Datasets"),
        Path("/content/drive/MyDrive/RF_Datasets"),
        Path("/content/drive/MyDrive/Surrogate-XAI"),
        Path("/content/drive/MyDrive/Surrogate_XAI_V2"),
    ]
    found = set(out.get("wisig", []))
    for root in roots:
        if not root.exists():
            continue
        root_parts = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            dp = Path(dirpath)
            if len(dp.parts) - root_parts >= max_depth:
                dirnames[:] = []
            if str(dp).startswith(str(impl.DRIVE_PHASE)):
                dirnames[:] = []
                continue
            for fn in filenames:
                low = fn.lower()
                if low in ("manytx.pkl.zip", "manytx.zip") or (
                    "manytx" in low and low.endswith(".zip")
                ):
                    found.add(str(dp / fn))
    out["wisig"] = sorted(found)
    return out


impl.discover_candidates = discover_candidates_v102


def write_config_template_v102(path: Path,
                               candidates: Dict[str, List[str]] | None = None) -> None:
    candidates = candidates or discover_candidates_v102()

    def unique(k: str) -> str:
        vals = candidates.get(k, [])
        # Prefer source-authentic ManyTx ZIP over derived WiSig HDF5.
        if k == "wisig":
            zips = [v for v in vals if "manytx" in Path(v).name.lower() and v.lower().endswith(".zip")]
            if len(zips) == 1:
                return zips[0]
            return ""
        return vals[0] if len(vals) == 1 else ""

    cfg = {
        "phase_version": "1.0.2",
        "project": impl.PROJECT,
        "storage": {
            "shard_mib_candidates": impl.DEFAULT_SHARD_MIB_CANDIDATES,
            "transport_pack_gib": impl.DEFAULT_PACK_GIB,
            "local_work_root": str(impl.LOCAL_PROJECT),
            "drive_project_root": str(impl.DRIVE_PROJECT),
        },
        "datasets": {
            "wisig": {
                "enabled": True,
                "format": "wisig_manytx_native_zip",
                "source_drive_path": unique("wisig"),
                "expected_sha256": "",
                "archive_member": "ManyTx.pkl",
                "expected_schema": {
                    "n_samples": 1020643,
                    "sequence_length": 256,
                    "tx_count": 150,
                    "receiver_count": 18,
                    "capture_date_count": 4,
                    "equalization_count": 2,
                    "nonempty_cells": 20759,
                    "empty_cells": 841,
                },
            },
            "radioml2016": {
                "enabled": True,
                "format": "radioml2016_pickle",
                "source_drive_path": unique("radioml2016"),
                "expected_sha256": "",
            },
            "radioml2018": {
                "enabled": True,
                "format": "radioml2018_hdf5",
                "source_drive_path": unique("radioml2018"),
                "expected_sha256": "",
                "hdf5": {"x_key": "X", "y_key": "Y", "z_key": "Z"},
            },
        },
    }
    atomic_write_json(path, cfg)


impl.write_config_template = write_config_template_v102


def canonicalize_manytx_native(name: str, dcfg: Dict[str, Any], run_root: Path,
                               shard_mib: int, transport_pack_gib: int
                               ) -> Dict[str, Any]:
    source_drive = Path(dcfg["source_drive_path"])
    if not source_drive.is_file():
        raise FileNotFoundError(
            f"{name}: ManyTx source ZIP missing/not a file: {source_drive}. "
            "Run --discover and set datasets.wisig.source_drive_path."
        )
    if not source_drive.name.lower().endswith(".zip"):
        raise RuntimeError(
            f"{name}: v1.0.2 native adapter requires a ZIP containing ManyTx.pkl; "
            f"got {source_drive}"
        )

    local_source_dir = run_root / name / "source"
    canonical_root = run_root / name / "canonical"
    transport_root = run_root / name / "transport"
    local_source_dir.mkdir(parents=True, exist_ok=True)
    canonical_root.mkdir(parents=True, exist_ok=True)
    assert_local_hot_path(local_source_dir)
    assert_local_hot_path(canonical_root)

    # Preflight against ZIP size before stage-in.
    source_bytes = source_drive.stat().st_size
    free_bytes = shutil.disk_usage("/content").free
    min_required = int(2.0 * source_bytes + 8 * 1024**3)
    if free_bytes < min_required:
        raise RuntimeError(
            f"{name}: insufficient /content free space before ManyTx stage-in. "
            f"source_zip={source_bytes/1024**3:.2f} GiB, "
            f"free={free_bytes/1024**3:.2f} GiB, "
            f"minimum_required={min_required/1024**3:.2f} GiB"
        )

    expected_sha = dcfg.get("expected_sha256") or None
    stage = stage_in_file(
        source_drive, local_source_dir, expected_sha256=expected_sha
    )
    local_zip = local_source_dir / source_drive.name

    member_name = dcfg.get("archive_member") or "ManyTx.pkl"
    info = resolve_manytx_member(local_zip, member_name)

    # Stronger disk preflight using ZIP central-directory uncompressed member size.
    canonical_expected_bytes = (
        int(dcfg.get("expected_schema", {}).get("n_samples", 1_020_643))
        * 2
        * int(dcfg.get("expected_schema", {}).get("sequence_length", 256))
        * 4
    )
    free_after_zip = shutil.disk_usage("/content").free
    required_after_zip = int(
        info.file_size + 2.15 * canonical_expected_bytes + 4 * 1024**3
    )
    if free_after_zip < required_after_zip:
        local_zip.unlink(missing_ok=True)
        raise RuntimeError(
            f"{name}: insufficient /content free space for extracted ManyTx + "
            f"canonical shards + transport TAR. "
            f"pickle_uncompressed={info.file_size/1024**3:.2f} GiB, "
            f"canonical_estimate={canonical_expected_bytes/1024**3:.2f} GiB, "
            f"free={free_after_zip/1024**3:.2f} GiB, "
            f"required={required_after_zip/1024**3:.2f} GiB"
        )

    print(
        f"[RUN] Extracting {member_name} locally "
        f"({info.file_size/1024**3:.2f} GiB uncompressed)..."
    )
    local_pickle = safe_extract_manytx_pickle(
        local_zip, local_source_dir, preferred=member_name
    )
    extracted_sha = sha256_file(local_pickle)

    print("[RUN] Loading native ManyTx pickle from local SSD...")
    obj = load_manytx_pickle(local_pickle)

    print("[RUN] Inspecting and validating native ManyTx schema...")
    inspection = inspect_manytx_object(obj)
    validate_manytx_inspection(
        inspection, expected=dcfg.get("expected_schema") or {}
    )
    inspection.update({
        "source_zip_drive_path": str(source_drive),
        "source_zip_sha256": stage["sha256"],
        "archive_member": member_name,
        "archive_member_bytes": int(info.file_size),
        "archive_member_sha256": extracted_sha,
    })
    atomic_write_json(canonical_root / "SOURCE_INSPECTION.json", inspection)
    atomic_write_json(canonical_root / "SOURCE_VALUE_MAPPINGS.json", {
        "tx_values": inspection["tx_values"],
        "receiver_values": inspection["receiver_values"],
        "capture_date_values": inspection["capture_date_values"],
        "equalization_values": inspection["equalization_values"],
        "session_id": "NOT_REPRESENTED",
    })

    writer = impl.ShardWriter(canonical_root, shard_mib, name)
    t0 = time.perf_counter()
    for x, meta in iter_manytx_object(obj):
        if x.shape[2] != 256:
            raise RuntimeError(f"WiSig native sequence length changed: {x.shape}")
        if not __import__("numpy").isfinite(x).all():
            raise RuntimeError("WiSig ManyTx contains non-finite signal values")
        writer.add(x, meta)
    dataset_manifest = writer.close()
    elapsed = time.perf_counter() - t0

    if dataset_manifest["n_samples"] != inspection["n_samples"]:
        raise RuntimeError(
            f"WiSig writer count mismatch: {dataset_manifest['n_samples']} "
            f"vs inspection {inspection['n_samples']}"
        )
    if dataset_manifest["seq_len"] != 256:
        raise RuntimeError(
            f"WiSig canonical seq_len must be 256, got {dataset_manifest['seq_len']}"
        )

    dataset_manifest.update({
        "phase": impl.PHASE,
        "phase_version": "1.0.2",
        "source_drive_path": str(source_drive),
        "source_local_zip_path": str(local_zip),
        "source_sha256": stage["sha256"],
        "source_archive_member": member_name,
        "source_archive_member_sha256": extracted_sha,
        "source_schema": inspection["schema"],
        "canonical_layout": "[N,2,L]",
        "canonical_dtype": "float32",
        "signal_value_transform": "layout_dtype_only; no DC/RMS normalization",
        "native_sequence_length_preserved": True,
        "metadata_semantics": {
            "label": "transmitter index; identical to tx_id",
            "tx_id": "index into SOURCE_VALUE_MAPPINGS.tx_values",
            "receiver": "index into SOURCE_VALUE_MAPPINGS.receiver_values",
            "day": "capture-date index alias used by downstream nuisance analyses",
            "capture_date": "index into SOURCE_VALUE_MAPPINGS.capture_date_values",
            "equalization": "index into SOURCE_VALUE_MAPPINGS.equalization_values",
            "source_cell_id": (
                "row-major source cell id over tx,receiver,capture_date,equalization"
            ),
            "sample_in_cell": "source-preserved sample order within a nonempty cell",
        },
        "canonicalization_elapsed_s": elapsed,
        "canonicalization_samples_per_s": (
            dataset_manifest["n_samples"] / max(elapsed, 1e-9)
        ),
        "shard_target_mib": shard_mib,
    })
    atomic_write_json(canonical_root / "DATASET_MANIFEST.json", dataset_manifest)

    verification = impl.verify_canonical_dataset(canonical_root, dataset_manifest)
    if verification["n_samples"] != 1_020_643 or verification["seq_len"] != 256:
        raise RuntimeError(f"WiSig final verification invariant failure: {verification}")
    atomic_write_json(
        canonical_root / "CANONICAL_VERIFICATION.json", verification
    )

    # Release the very large nested Python object before transport pack creation.
    del obj
    gc.collect()

    # Delete staged source files only after canonical verification succeeds.
    local_pickle.unlink(missing_ok=True)
    local_zip.unlink(missing_ok=True)

    pack_files = [p for p in canonical_root.rglob("*") if p.is_file()]
    packs = create_transport_packs(
        canonical_root,
        pack_files,
        transport_root,
        target_pack_bytes=transport_pack_gib * 1024**3,
    )
    atomic_write_json(transport_root / "TRANSPORT_MANIFEST.json", {
        "dataset": name,
        "phase_version": "1.0.2",
        "pack_target_gib": transport_pack_gib,
        "packs": packs,
    })

    return {
        "dataset": name,
        "source_stage_in": stage,
        "inspection": inspection,
        "dataset_manifest": dataset_manifest,
        "verification": verification,
        "transport": {"packs": packs},
        "split_copy": None,
    }


def canonicalize_one_v102(name: str, dcfg: Dict[str, Any], run_root: Path,
                          shard_mib: int, transport_pack_gib: int
                          ) -> Dict[str, Any]:
    if dcfg.get("format") == "wisig_manytx_native_zip":
        if name != "wisig":
            raise RuntimeError(
                "wisig_manytx_native_zip format is only valid for dataset name 'wisig'"
            )
        return canonicalize_manytx_native(
            name, dcfg, run_root, shard_mib, transport_pack_gib
        )
    return _ORIG_CANONICALIZE_ONE(
        name, dcfg, run_root, shard_mib, transport_pack_gib
    )


impl.canonicalize_one = canonicalize_one_v102


if __name__ == "__main__":
    impl.main()
