#!/usr/bin/env python3
"""
PHASE 1 — Dataset Canonicalization and Storage Pipeline v1.0.0

Canonical datasets:
- WiSig: real OTA RF fingerprinting source / frozen benchmark source
- RadioML 2016.10A: pickle dictionary, native L=128
- RadioML 2018.01A: HDF5, native L=1024

Key rules:
1. Drive is persistence/stage-in/stage-out only.
2. Source dataset file is copied to /content and SHA-hashed before parsing.
3. Canonical signals are float32 [N,2,L] with VALUES PRESERVED.
4. No irreversible DC/RMS normalization in Phase 1.
5. Deterministic global indices and metadata shards.
6. Optimized logical shard size benchmark on local SSD.
7. Bulk persistence uses large uncompressed TAR transport packs.
8. All manifests and transport packs are SHA-256 protected.
"""
from __future__ import annotations
import argparse, gc, json, math, os, shutil, sys, time, tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from storage_contract import (
    utc_now, sha256_file, atomic_write_json, assert_local_hot_path,
    stage_in_file, stage_in_directory_small_files, create_transport_packs,
    copy_file_verified,
)
from dataset_adapters import (
    inspect_wisig_hdf5, inspect_radioml2016, inspect_radioml2018,
    iter_wisig, iter_radioml2016, iter_radioml2018,
)

PHASE = "PHASE1_DATASET_CANONICALIZATION"
VERSION = "1.0.0"
PROJECT = "Surrogate_XAI_V2"
DRIVE_PROJECT = Path("/content/drive/MyDrive") / PROJECT
LOCAL_PROJECT = Path("/content") / PROJECT.lower()
DRIVE_PHASE = DRIVE_PROJECT / "01_PHASE1_DATASET_CANONICALIZATION"
LOCAL_PHASE = LOCAL_PROJECT / "01_phase1_dataset_canonicalization"

DEFAULT_SHARD_MIB_CANDIDATES = [16, 32, 64, 128]
DEFAULT_PACK_GIB = 2

def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception as e:
        raise RuntimeError("Google Drive mount failed; run inside Google Colab.") from e

def latest_phase0_status() -> Dict[str, Any]:
    """
    Verify the newest completed Phase-0 artifact locally and require
    GPU_AUTOTUNE_COMPLETE. A CPU-only preflight is never accepted.
    """
    root = DRIVE_PROJECT / "00_PHASE0_RUNTIME_AUTOTUNE"
    runs = sorted(root.glob("run_*"))
    if not runs:
        raise RuntimeError(f"No Phase-0 run found in {root}")

    provenance_local = LOCAL_PROJECT / "_phase0_provenance_check"
    provenance_local.mkdir(parents=True, exist_ok=True)
    assert_local_hot_path(provenance_local)

    failures = []
    for r in reversed(runs):
        p = r / "STAGE_COMPLETE.json"
        if not p.exists():
            continue
        try:
            with p.open() as f:
                complete = json.load(f)
            archive_name = complete.get("archive")
            archive_sha = complete.get("archive_sha256")
            if not archive_name or not archive_sha:
                failures.append(f"{r.name}: missing archive/archive_sha256")
                continue

            drive_archive = r / archive_name
            copy_info = stage_in_file(
                drive_archive, provenance_local, expected_sha256=archive_sha
            )
            local_archive = provenance_local / archive_name

            with tarfile.open(local_archive, mode="r:*") as tf:
                candidates = [
                    m for m in tf.getmembers()
                    if m.isfile() and m.name.endswith("outputs/PHASE0_FINAL_STATUS.json")
                ]
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"Expected exactly one PHASE0_FINAL_STATUS.json in {archive_name}; "
                        f"found {len(candidates)}"
                    )
                fh = tf.extractfile(candidates[0])
                if fh is None:
                    raise RuntimeError("Could not read PHASE0_FINAL_STATUS.json")
                phase0_status = json.loads(fh.read().decode("utf-8"))

            local_archive.unlink(missing_ok=True)
            if phase0_status.get("status") != "GPU_AUTOTUNE_COMPLETE":
                failures.append(
                    f"{r.name}: status={phase0_status.get('status')!r}, not GPU_AUTOTUNE_COMPLETE"
                )
                continue

            return {
                "run_dir": str(r),
                "run_id": phase0_status.get("run_id"),
                "status": phase0_status.get("status"),
                "phase_version": phase0_status.get("phase_version"),
                "archive": archive_name,
                "archive_sha256": archive_sha,
                "verified_stage_in_sha256": copy_info["sha256"],
            }
        except Exception as e:
            failures.append(f"{r.name}: {type(e).__name__}: {e}")

    raise RuntimeError(
        "No verified Phase-0 GPU_AUTOTUNE_COMPLETE artifact found. "
        + " | ".join(failures[:10])
    )

def discover_candidates(max_depth: int = 5) -> Dict[str, List[str]]:
    """
    Metadata-only Drive discovery. Not used for scientific hot I/O.
    Searches likely dataset filenames and limits recursion depth.
    """
    roots = [
        Path("/content/drive/MyDrive"),
        Path("/content/drive/MyDrive/Datasets"),
        Path("/content/drive/MyDrive/RF_Datasets"),
        Path("/content/drive/MyDrive/Surrogate_XAI"),
        Path("/content/drive/MyDrive/Surrogate_XAI_V2"),
    ]
    pats = {
        "wisig": ("wisig",),
        "radioml2016": ("2016.10", "rml2016", "2016_10"),
        "radioml2018": ("2018.01", "rml2018", "gold_xyz_osc", "2018_01"),
    }
    exts = {".h5", ".hdf5", ".pkl", ".pickle", ".dat"}
    out = {k: [] for k in pats}
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        root_parts = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            dp = Path(dirpath)
            if len(dp.parts) - root_parts >= max_depth:
                dirnames[:] = []
            # Avoid our generated output tree.
            if str(dp).startswith(str(DRIVE_PROJECT / "01_PHASE1_DATASET_CANONICALIZATION")):
                dirnames[:] = []
                continue
            for fn in filenames:
                p = dp / fn
                if p.suffix.lower() not in exts:
                    continue
                sp = str(p)
                if sp in seen:
                    continue
                seen.add(sp)
                low = fn.lower()
                for key, tokens in pats.items():
                    if any(t in low for t in tokens):
                        out[key].append(sp)
    for k in out:
        out[k] = sorted(out[k])
    return out

def write_config_template(path: Path, candidates: Optional[Dict[str,List[str]]] = None) -> None:
    candidates = candidates or {k: [] for k in ("wisig","radioml2016","radioml2018")}
    def unique(k):
        return candidates[k][0] if len(candidates[k]) == 1 else ""
    cfg = {
        "phase_version": VERSION,
        "project": PROJECT,
        "storage": {
            "shard_mib_candidates": DEFAULT_SHARD_MIB_CANDIDATES,
            "transport_pack_gib": DEFAULT_PACK_GIB,
            "local_work_root": str(LOCAL_PROJECT),
            "drive_project_root": str(DRIVE_PROJECT),
        },
        "datasets": {
            "wisig": {
                "enabled": True,
                "format": "wisig_hdf5",
                "source_drive_path": unique("wisig"),
                "expected_sha256": "",
                "hdf5": {
                    "signal_dataset": "",
                    "label_dataset": "",
                    "receiver_dataset": "",
                    "day_dataset": "",
                    "equalization_dataset": "",
                },
                "frozen_split_dir_drive": "",
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

def benchmark_shard_sizes(bench_dir: Path, candidates_mib: List[int]) -> Dict[str, Any]:
    """
    Execution-only local SSD benchmark. Creates same total logical bytes at each shard size
    and measures reopen+full-read throughput. Chooses smallest size within 98% of max.
    """
    assert_local_hot_path(bench_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)
    total_mib = 512
    block = np.random.default_rng(12345).standard_normal(1024 * 1024 // 4, dtype=np.float32)
    trials = []
    for mib in candidates_mib:
        td = bench_dir / f"{mib}MiB"
        td.mkdir(parents=True, exist_ok=True)
        nfiles = max(1, total_mib // mib)
        bytes_target = mib * 1024 * 1024
        floats_target = bytes_target // 4
        t0 = time.perf_counter()
        for i in range(nfiles):
            p = td / f"s_{i:04d}.npy"
            mm = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(floats_target,))
            pos = 0
            while pos < floats_target:
                take = min(len(block), floats_target-pos)
                mm[pos:pos+take] = block[:take]
                pos += take
            mm.flush()
            del mm
        write_s = time.perf_counter() - t0

        total_bytes = 0
        t1 = time.perf_counter()
        checksum = 0.0
        for p in sorted(td.glob("*.npy")):
            a = np.load(p, mmap_mode="r")
            # Force complete sequential read in ~4 MiB slices.
            step = 1024 * 1024
            for s in range(0, len(a), step):
                checksum += float(np.sum(a[s:s+step], dtype=np.float64))
            total_bytes += a.nbytes
            del a
        read_s = time.perf_counter() - t1
        trials.append({
            "shard_mib": mib, "nfiles": nfiles, "bytes": total_bytes,
            "write_MiB_s": (total_bytes/1024**2)/max(write_s,1e-9),
            "read_MiB_s": (total_bytes/1024**2)/max(read_s,1e-9),
            "checksum": checksum,
        })
        shutil.rmtree(td)

    max_read = max(t["read_MiB_s"] for t in trials)
    eligible = [t for t in trials if t["read_MiB_s"] >= 0.98*max_read]
    chosen = min(eligible, key=lambda t: t["shard_mib"])
    return {
        "trials": trials,
        "selection_rule": "smallest shard size within 98% of maximum local sequential read throughput",
        "chosen_shard_mib": chosen["shard_mib"],
        "max_read_MiB_s": max_read,
    }

class ShardWriter:
    def __init__(self, root: Path, target_mib: int, dataset_name: str):
        assert_local_hot_path(root)
        self.root = root
        self.target_bytes = target_mib * 1024 * 1024
        self.dataset_name = dataset_name
        self.signal_dir = root / "signals"
        self.meta_dir = root / "metadata"
        self.signal_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self._buf_x: List[np.ndarray] = []
        self._buf_meta: Dict[str, List[np.ndarray]] = {}
        self._buf_bytes = 0
        self._next_shard = 0
        self.records: List[Dict[str,Any]] = []
        self.n_total = 0
        self.seq_len: Optional[int] = None
        self.meta_fields: Optional[List[str]] = None

    def add(self, x: np.ndarray, meta: Dict[str,np.ndarray]) -> None:
        if x.dtype != np.float32 or x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Canonical x must be float32 [N,2,L], got {x.dtype} {x.shape}")
        if self.seq_len is None:
            self.seq_len = int(x.shape[2])
            self.meta_fields = sorted(meta)
        if x.shape[2] != self.seq_len:
            raise ValueError(f"Sequence length changed within dataset: {x.shape[2]} vs {self.seq_len}")
        if sorted(meta) != self.meta_fields:
            raise ValueError("Metadata fields changed between chunks")
        n = x.shape[0]
        if any(len(v) != n for v in meta.values()):
            raise ValueError("Metadata length mismatch")

        # Split incoming chunk if needed so logical shards stay near target size.
        per_sample = x[0].nbytes + sum(np.asarray(v[:1]).nbytes for v in meta.values())
        samples_target = max(1, self.target_bytes // max(per_sample,1))
        start = 0
        while start < n:
            current_samples = sum(a.shape[0] for a in self._buf_x)
            room = max(1, samples_target - current_samples)
            take = min(room, n-start)
            self._buf_x.append(np.ascontiguousarray(x[start:start+take]))
            for k,v in meta.items():
                self._buf_meta.setdefault(k,[]).append(np.ascontiguousarray(v[start:start+take]))
            self._buf_bytes += x[start:start+take].nbytes + sum(v[start:start+take].nbytes for v in meta.values())
            start += take
            if sum(a.shape[0] for a in self._buf_x) >= samples_target:
                self.flush()

    def flush(self) -> None:
        if not self._buf_x:
            return
        x = np.concatenate(self._buf_x, axis=0)
        meta = {k: np.concatenate(v, axis=0) for k,v in self._buf_meta.items()}
        sid = self._next_shard
        sig = self.signal_dir / f"signals_{sid:06d}.npy"
        met = self.meta_dir / f"meta_{sid:06d}.npz"
        np.save(sig, x, allow_pickle=False)
        np.savez(met, **meta)

        gids = meta["global_index"]
        rec = {
            "shard_id": sid,
            "signal_path": str(sig.relative_to(self.root)),
            "metadata_path": str(met.relative_to(self.root)),
            "n_samples": int(len(x)),
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "first_global_index": int(gids[0]),
            "last_global_index": int(gids[-1]),
            "signal_bytes": sig.stat().st_size,
            "metadata_bytes": met.stat().st_size,
            "signal_sha256": sha256_file(sig),
            "metadata_sha256": sha256_file(met),
        }
        self.records.append(rec)
        self.n_total += len(x)
        self._next_shard += 1
        self._buf_x.clear()
        self._buf_meta.clear()
        self._buf_bytes = 0

    def close(self) -> Dict[str,Any]:
        self.flush()
        return {
            "dataset": self.dataset_name,
            "n_samples": self.n_total,
            "seq_len": self.seq_len,
            "n_shards": len(self.records),
            "meta_fields": self.meta_fields,
            "shards": self.records,
        }

def verify_canonical_dataset(root: Path, manifest: Dict[str,Any]) -> Dict[str,Any]:
    expected_gid = 0
    n = 0
    seq_len = manifest["seq_len"]
    labels = set()
    snrs = set()
    for r in manifest["shards"]:
        sig = root / r["signal_path"]
        met = root / r["metadata_path"]
        if sha256_file(sig) != r["signal_sha256"]:
            raise RuntimeError(f"Signal shard SHA failure: {sig}")
        if sha256_file(met) != r["metadata_sha256"]:
            raise RuntimeError(f"Metadata shard SHA failure: {met}")
        x = np.load(sig, mmap_mode="r")
        m = np.load(met, allow_pickle=False)
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] != seq_len or x.dtype != np.float32:
            raise RuntimeError(f"Canonical shape/dtype failure: {sig} {x.shape} {x.dtype}")
        gids = m["global_index"]
        if int(gids[0]) != expected_gid:
            raise RuntimeError(f"Global index discontinuity at shard {r['shard_id']}")
        expected_gid = int(gids[-1]) + 1
        n += len(x)
        if "label" in m: labels.update(np.unique(m["label"]).tolist())
        if "snr" in m: snrs.update(np.unique(m["snr"]).tolist())
    if n != manifest["n_samples"]:
        raise RuntimeError(f"Sample count mismatch {n} != {manifest['n_samples']}")
    return {
        "verified": True, "n_samples": n, "seq_len": seq_len,
        "unique_labels": sorted(map(int, labels)),
        "unique_snrs": sorted(map(int, snrs)),
    }

def canonicalize_one(name: str, dcfg: Dict[str,Any], run_root: Path,
                     shard_mib: int, transport_pack_gib: int) -> Dict[str,Any]:
    source_drive = Path(dcfg["source_drive_path"])
    if not source_drive.is_file():
        raise FileNotFoundError(
            f"{name}: source_drive_path is missing/not a file: {source_drive}. "
            "Run --discover and update the config."
        )
    local_source_dir = run_root / name / "source"
    canonical_root = run_root / name / "canonical"
    transport_root = run_root / name / "transport"
    local_source_dir.mkdir(parents=True, exist_ok=True)
    canonical_root.mkdir(parents=True, exist_ok=True)

    expected = dcfg.get("expected_sha256") or None

    # Conservative local-disk preflight. Canonical float32 IQ is usually similar
    # in size to the source, and transport TAR temporarily duplicates canonical bytes.
    # By deleting the staged raw source before TAR construction, peak usage is kept
    # near ~2x source/canonical rather than ~3x.
    source_bytes = source_drive.stat().st_size
    free_bytes = shutil.disk_usage("/content").free
    required_bytes = int(2.25 * source_bytes + 5 * 1024**3)
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"{name}: insufficient /content free space. "
            f"source={source_bytes/1024**3:.2f} GiB, "
            f"free={free_bytes/1024**3:.2f} GiB, "
            f"conservative_required={required_bytes/1024**3:.2f} GiB"
        )

    stage = stage_in_file(source_drive, local_source_dir, expected_sha256=expected)
    local_source = local_source_dir / source_drive.name

    # Structural inspection and iterator resolution.
    fmt = dcfg["format"]
    if fmt == "radioml2016_pickle":
        inspection = inspect_radioml2016(local_source)
        iterator = iter_radioml2016(local_source)
    elif fmt == "radioml2018_hdf5":
        h = dcfg.get("hdf5", {})
        inspection = inspect_radioml2018(
            local_source, h.get("x_key") or None, h.get("y_key") or None, h.get("z_key") or None
        )
        iterator = iter_radioml2018(
            local_source, inspection["x_key"], inspection["y_key"], inspection["z_key"]
        )
    elif fmt == "wisig_hdf5":
        h = dcfg.get("hdf5", {})
        inspection = inspect_wisig_hdf5(local_source, h.get("signal_dataset") or None)
        signal_key = h.get("signal_dataset") or inspection["detected_signal_dataset"]
        # Metadata keys may be explicit. We intentionally do not silently pick ambiguous labels.
        label_key = h.get("label_dataset") or None
        receiver_key = h.get("receiver_dataset") or None
        day_key = h.get("day_dataset") or None
        equalization_key = h.get("equalization_dataset") or None
        iterator = iter_wisig(
            local_source, signal_key, label_key, receiver_key, day_key, equalization_key
        )
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    atomic_write_json(canonical_root / "SOURCE_INSPECTION.json", inspection)
    writer = ShardWriter(canonical_root, shard_mib, name)
    t0 = time.perf_counter()
    for x, meta in iterator:
        if not np.isfinite(x).all():
            raise RuntimeError(f"{name}: non-finite signal value detected")
        writer.add(x, meta)
    dataset_manifest = writer.close()
    elapsed = time.perf_counter() - t0
    dataset_manifest.update({
        "phase": PHASE, "phase_version": VERSION,
        "source_drive_path": str(source_drive),
        "source_local_path": str(local_source),
        "source_sha256": stage["sha256"],
        "canonical_layout": "[N,2,L]",
        "canonical_dtype": "float32",
        "signal_value_transform": "layout_dtype_only; no DC/RMS normalization",
        "canonicalization_elapsed_s": elapsed,
        "canonicalization_samples_per_s": dataset_manifest["n_samples"]/max(elapsed,1e-9),
        "shard_target_mib": shard_mib,
    })
    atomic_write_json(canonical_root / "DATASET_MANIFEST.json", dataset_manifest)

    # Preserve frozen WiSig split arrays if configured.
    split_copy = None
    split_dir = dcfg.get("frozen_split_dir_drive") or ""
    if split_dir:
        split_copy = stage_in_directory_small_files(
            Path(split_dir), canonical_root / "frozen_splits",
            patterns=("*.npy","*.json")
        )
        atomic_write_json(canonical_root / "FROZEN_SPLIT_COPY_MANIFEST.json", split_copy)

    verification = verify_canonical_dataset(canonical_root, dataset_manifest)
    atomic_write_json(canonical_root / "CANONICAL_VERIFICATION.json", verification)

    # Recover raw-source disk before transport TAR duplicates canonical bytes.
    local_source.unlink(missing_ok=True)

    # Pack canonical signals + metadata + manifests.
    pack_files = [p for p in canonical_root.rglob("*") if p.is_file()]
    packs = create_transport_packs(
        canonical_root, pack_files, transport_root,
        target_pack_bytes=transport_pack_gib * 1024**3
    )
    atomic_write_json(transport_root / "TRANSPORT_MANIFEST.json", {
        "dataset": name, "pack_target_gib": transport_pack_gib, "packs": packs
    })

    return {
        "dataset": name, "source_stage_in": stage, "inspection": inspection,
        "dataset_manifest": dataset_manifest, "verification": verification,
        "transport": {"packs": packs}, "split_copy": split_copy,
    }

def stage_out_dataset(run_root: Path, dataset: str, result: Dict[str,Any], drive_run: Path) -> Dict[str,Any]:
    """
    Copy only compact canonical manifests + large transport packs to Drive.
    Individual canonical shard files intentionally remain local and are NOT
    persisted loose on Drive.
    """
    local_ds = run_root / dataset
    drive_ds = drive_run / dataset
    drive_ds.mkdir(parents=True, exist_ok=True)
    copied = []

    # Compact canonical manifests/splits.
    canonical = local_ds / "canonical"
    for p in sorted(canonical.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(canonical)
        # Skip bulk signal/meta shards; those are inside transport packs.
        if rel.parts and rel.parts[0] in ("signals","metadata"):
            continue
        dst = drive_ds / "canonical_manifest" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        copied.append(copy_file_verified(p, dst, expected_sha256=sha256_file(p)))

    # Large transport packs + transport manifest.
    transport = local_ds / "transport"
    for p in sorted(transport.glob("*")):
        if p.is_file():
            dst = drive_ds / "transport" / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            copied.append(copy_file_verified(p, dst, expected_sha256=sha256_file(p)))

    atomic_write_json(drive_ds / "DATASET_STAGE_COMPLETE.json", {
        "dataset": dataset, "completed_utc": utc_now(),
        "source_sha256": result["dataset_manifest"]["source_sha256"],
        "n_samples": result["dataset_manifest"]["n_samples"],
        "seq_len": result["dataset_manifest"]["seq_len"],
        "n_shards": result["dataset_manifest"]["n_shards"],
        "transport_packs": len(result["transport"]["packs"]),
    })
    return {"dataset": dataset, "drive_root": str(drive_ds), "copied_files": len(copied)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--write-config-template", type=str, default="")
    ap.add_argument("--dataset", choices=["all","wisig","radioml2016","radioml2018"], default="all")
    args = ap.parse_args()

    mount_drive()

    if args.discover:
        print(json.dumps(discover_candidates(), indent=2))
        return

    if args.write_config_template:
        p = Path(args.write_config_template)
        if not str(p.resolve()).startswith("/content/"):
            raise RuntimeError("Write config template under /content, then optionally copy to Drive.")
        candidates = discover_candidates()
        write_config_template(p, candidates)
        print(f"Wrote config template: {p}")
        print(json.dumps(candidates, indent=2))
        return

    if not args.config:
        raise SystemExit(
            "Provide --config /content/phase1_config.json. "
            "Run --write-config-template /content/phase1_config.json first."
        )

    cfg_path = Path(args.config)
    assert_local_hot_path(cfg_path)
    with cfg_path.open() as f:
        cfg = json.load(f)

    phase0 = latest_phase0_status()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_root = LOCAL_PHASE / f"run_{run_id}"
    drive_run = DRIVE_PHASE / f"run_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)
    assert_local_hot_path(run_root)

    phase_status = {
        "phase": PHASE, "phase_version": VERSION, "run_id": run_id,
        "status": "RUNNING", "started_utc": utc_now(),
        "phase0_provenance": phase0,
        "config_sha256": sha256_file(cfg_path),
    }
    atomic_write_json(run_root / "PHASE1_STATUS.json", phase_status)
    shutil.copy2(cfg_path, run_root / "CONFIG.json")

    try:
        storage = cfg["storage"]
        candidates = [int(x) for x in storage.get("shard_mib_candidates", DEFAULT_SHARD_MIB_CANDIDATES)]
        print("[RUN] Local logical-shard benchmark...")
        bench = benchmark_shard_sizes(run_root / "storage_benchmark", candidates)
        atomic_write_json(run_root / "SHARD_SIZE_BENCHMARK.json", bench)
        shard_mib = int(bench["chosen_shard_mib"])
        pack_gib = int(storage.get("transport_pack_gib", DEFAULT_PACK_GIB))
        print(f"[OK] Chosen logical shard size: {shard_mib} MiB")

        selected = [args.dataset] if args.dataset != "all" else ["wisig","radioml2016","radioml2018"]
        results = {}
        for name in selected:
            dcfg = cfg["datasets"][name]
            if not dcfg.get("enabled", True):
                print(f"[SKIP] {name} disabled")
                continue
            print(f"[RUN] Canonicalizing {name}...")
            res = canonicalize_one(name, dcfg, run_root, shard_mib, pack_gib)
            results[name] = res
            print(
                f"[OK] {name}: N={res['dataset_manifest']['n_samples']:,}, "
                f"L={res['dataset_manifest']['seq_len']}, "
                f"shards={res['dataset_manifest']['n_shards']}, "
                f"packs={len(res['transport']['packs'])}"
            )
            print(f"[RUN] Stage-out {name} transport/manifests to Drive...")
            stageout = stage_out_dataset(run_root, name, res, drive_run)
            print(f"[OK] {name} stage-out: {stageout['drive_root']}")

        summary = {
            "phase": PHASE, "phase_version": VERSION, "run_id": run_id,
            "completed_utc": utc_now(),
            "chosen_shard_mib": shard_mib,
            "transport_pack_gib": pack_gib,
            "datasets": {
                k: {
                    "source_sha256": v["dataset_manifest"]["source_sha256"],
                    "n_samples": v["dataset_manifest"]["n_samples"],
                    "seq_len": v["dataset_manifest"]["seq_len"],
                    "n_shards": v["dataset_manifest"]["n_shards"],
                    "n_transport_packs": len(v["transport"]["packs"]),
                    "verification": v["verification"],
                } for k,v in results.items()
            }
        }
        atomic_write_json(run_root / "PHASE1_SUMMARY.json", summary)
        # Copy root compact results to Drive.
        drive_run.mkdir(parents=True, exist_ok=True)
        for fn in ("CONFIG.json","SHARD_SIZE_BENCHMARK.json","PHASE1_SUMMARY.json"):
            src = run_root / fn
            copy_file_verified(src, drive_run / fn, expected_sha256=sha256_file(src))

        phase_status.update({
            "status": "PHASE1_COMPLETE",
            "completed_utc": utc_now(),
            "drive_run": str(drive_run),
            "datasets_completed": sorted(results),
            "chosen_shard_mib": shard_mib,
        })
        atomic_write_json(run_root / "PHASE1_STATUS.json", phase_status)
        copy_file_verified(
            run_root / "PHASE1_STATUS.json",
            drive_run / "PHASE1_STATUS.json",
            expected_sha256=sha256_file(run_root / "PHASE1_STATUS.json")
        )
        atomic_write_json(drive_run / "STAGE_COMPLETE.json", {
            "phase": PHASE, "phase_version": VERSION, "run_id": run_id,
            "status": "PHASE1_COMPLETE", "completed_utc": utc_now(),
            "summary_sha256": sha256_file(run_root / "PHASE1_SUMMARY.json"),
            "config_sha256": sha256_file(run_root / "CONFIG.json"),
        })

        print("="*80)
        print("PHASE 1 STATUS: PHASE1_COMPLETE")
        print(f"Local run: {run_root}")
        print(f"Drive run: {drive_run}")
        print("="*80)
    except Exception as e:
        phase_status.update({
            "status": "FAILED", "failed_utc": utc_now(),
            "error_type": type(e).__name__, "error": str(e),
        })
        atomic_write_json(run_root / "PHASE1_STATUS.json", phase_status)
        raise

if __name__ == "__main__":
    main()
