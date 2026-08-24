#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from access_guard import UNKNOWN_DIAGNOSTIC_ROLES, resolve_authorized_indices, resolve_phase3_wisig_root
from evaluation import fixed_frame_metrics
from phase4_common import atomic_json, sha256_json
from wisig_dataset import IndexedWiSigDataset


PHASE4_BENCHMARK_VERSION = "1.1.0"


@dataclass
class BenchmarkConfig:
    """Frozen short benchmark profile for a single Colab T4 session."""

    profile: str = "benchmark"
    seeds: Tuple[int, ...] = (42, 123, 2026)
    arms: Tuple[str, ...] = ("A0", "A1")
    max_epochs: int = 8
    minimum_epochs: int = 4
    early_stopping_patience: int = 2
    batch_size: int = 256
    samples_per_tx: int = 4
    eval_batch_size: int = 1024
    num_workers: int = 1
    pin_memory: bool = True
    embedding_dim: int = 128
    num_classes: int = 98
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.07
    prototype_momentum: float = 0.95
    label_smoothing: float = 0.0
    dropout: float = 0.10
    phase_rotation_radians: float = 0.12
    amplitude_jitter: float = 0.05
    awgn_std: float = 0.01
    maximum_circular_shift: int = 4
    amp_enabled: bool = True
    gradient_clip_norm: float = 5.0
    max_consecutive_amp_overflows: int = 32
    train_samples_per_known_tx: int = 600
    p0_samples_per_known_tx: int = 150
    shift_samples_per_present_tx: int = 150
    calibration_unknown_limit: int = 5_000
    covariance_fit_limit: int = 50_000
    practical_f1_delta: float = 0.002
    require_t4: bool = True

    def validate(self) -> None:
        if self.profile != "benchmark":
            raise ValueError("Only the frozen benchmark profile is supported")
        if tuple(self.arms) != ("A0", "A1"):
            raise RuntimeError("Benchmark arms are frozen to A0/A1")
        if tuple(self.seeds) != (42, 123, 2026):
            raise RuntimeError("Benchmark seeds are frozen to 42/123/2026")
        if not (1 <= self.minimum_epochs <= self.max_epochs <= 8):
            raise RuntimeError("Benchmark epochs must satisfy 1 <= min <= max <= 8")
        if self.train_samples_per_known_tx != 600:
            raise RuntimeError("Canonical benchmark requires 600 Train-Known samples per transmitter")
        if self.p0_samples_per_known_tx != 150:
            raise RuntimeError("Canonical benchmark requires 150 P0 samples per transmitter")
        if self.batch_size % self.samples_per_tx:
            raise RuntimeError("batch_size must be divisible by samples_per_tx")

    def digest(self) -> str:
        return sha256_json(asdict(self))


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(_worker_id: int) -> None:
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def cuda_preflight(config: BenchmarkConfig) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 4 benchmark requires a CUDA GPU")
    name = torch.cuda.get_device_name(0)
    if config.require_t4 and "T4" not in name.upper():
        raise RuntimeError(f"Canonical Phase 4 benchmark requires Tesla T4; observed {name}")
    props = torch.cuda.get_device_properties(0)
    return {
        "device_name": name,
        "total_memory_bytes": int(props.total_memory),
        "compute_capability": [int(props.major), int(props.minor)],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "amp_enabled": bool(config.amp_enabled),
    }


def deterministic_per_class_positions(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen: List[np.ndarray] = []
    for cls in sorted(np.unique(labels).tolist()):
        positions = np.flatnonzero(labels == cls)
        if len(positions) > per_class:
            positions = np.sort(rng.choice(positions, size=per_class, replace=False))
        chosen.append(positions)
    return np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)


def build_known_label_mapping(store) -> Tuple[Dict[int, int], Dict[str, Any]]:
    phase3 = resolve_phase3_wisig_root()
    with (phase3 / "CLASS_ROLE_MANIFEST.json").open("r", encoding="utf-8") as handle:
        roles = json.load(handle)
    known_ids = list(roles["known_transmitter_identities"])
    if len(known_ids) != 98:
        raise RuntimeError("Phase-3 known identity count changed")
    source = {value: index for index, value in enumerate(store.tx_values)}
    missing = [value for value in known_ids if value not in source]
    if missing:
        raise RuntimeError(f"Known identities missing from Phase-1 mapping: {missing}")
    mapping = {source[value]: cls for cls, value in enumerate(known_ids)}
    evidence = {
        "known_identity_order": known_ids,
        "source_label_to_known_class": {str(key): int(value) for key, value in mapping.items()},
        "mapping_sha256": sha256_json({str(key): int(value) for key, value in mapping.items()}),
    }
    return mapping, evidence


def make_benchmark_dataset(
    store,
    role: str,
    purpose: str,
    mapping: Mapping[int, int],
    config: BenchmarkConfig,
    subset_seed: int = 777,
):
    indices, evidence = resolve_authorized_indices(role, purpose)
    allow_unknown = role in UNKNOWN_DIAGNOSTIC_ROLES
    full = IndexedWiSigDataset(store, indices, mapping, allow_unknown_labels=allow_unknown)
    rng = np.random.default_rng(subset_seed + len(role))
    if allow_unknown:
        count = min(config.calibration_unknown_limit, len(full))
        positions = np.sort(rng.choice(len(full), size=count, replace=False))
    else:
        per_class = (
            config.train_samples_per_known_tx
            if role == "train_known"
            else config.p0_samples_per_known_tx
            if role == "p0_known"
            else config.shift_samples_per_present_tx
        )
        positions = deterministic_per_class_positions(full.known_class, per_class, subset_seed + len(role))
    subset = IndexedWiSigDataset(store, full.indices[positions], mapping, allow_unknown_labels=allow_unknown)
    evidence = dict(evidence)
    evidence.update(
        {
            "benchmark_subset_count": int(len(subset)),
            "full_role_count": int(len(full)),
            "subset_position_sha256": hashlib.sha256(np.asarray(positions, dtype=np.int64).tobytes()).hexdigest(),
            "selection_authorized": role == "p0_known",
        }
    )
    return subset, evidence


def make_loader(dataset, batch_size: int, config: BenchmarkConfig):
    generator = torch.Generator().manual_seed(12345)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=2 if config.num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def collect_outputs(model, dataset, device, config: BenchmarkConfig, include_logits: bool = True):
    loader = make_loader(dataset, config.eval_batch_size, config)
    model.eval()
    embeddings, logits, labels, global_indices = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            signals = batch["x"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.amp_enabled):
                outputs = model(signals)
            embeddings.append(outputs["embedding_normalized"].float().cpu().numpy())
            if include_logits:
                logits.append(outputs["logits"].float().cpu().numpy())
            labels.append(batch["y"].numpy())
            global_indices.append(batch["global_index"].numpy())
    return {
        "embedding": np.concatenate(embeddings),
        "logits": np.concatenate(logits) if logits else None,
        "y": np.concatenate(labels),
        "global_index": np.concatenate(global_indices),
    }


def evaluate_known(model, dataset, device, config: BenchmarkConfig) -> Dict[str, float]:
    outputs = collect_outputs(model, dataset, device, config, include_logits=True)
    return fixed_frame_metrics(
        outputs["y"], outputs["logits"].argmax(axis=1), outputs["logits"], config.num_classes
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def persist_file_verified(local_path: Path, drive_path: Path) -> Dict[str, Any]:
    """Copy one small recovery/output file to Drive; scientific signals remain local."""
    if str(local_path.resolve()).startswith("/content/drive/"):
        raise RuntimeError("Persistence source must be local")
    if not str(drive_path.resolve()).startswith("/content/drive/"):
        raise RuntimeError("Persistence destination must be on Drive")
    drive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = drive_path.with_suffix(drive_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    shutil.copyfile(local_path, temporary)
    expected = sha256_file(local_path)
    observed = sha256_file(temporary)
    if observed != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Drive persistence SHA mismatch: {drive_path.name}")
    os.replace(temporary, drive_path)
    return {"path": str(drive_path), "bytes": int(local_path.stat().st_size), "sha256": expected}


def restore_file_verified(drive_path: Path, local_path: Path, expected_sha256: str) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_suffix(local_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    shutil.copyfile(drive_path, temporary)
    observed = sha256_file(temporary)
    if observed != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Recovery checkpoint SHA mismatch: {drive_path}")
    os.replace(temporary, local_path)


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
