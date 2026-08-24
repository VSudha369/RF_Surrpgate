#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from access_guard import (
    KNOWN_EVAL_ROLES,
    TRAIN_ROLES,
    UNKNOWN_DIAGNOSTIC_ROLES,
    resolve_authorized_indices,
    resolve_phase3_wisig_root,
    strict_access_self_test,
    verify_phase3_contract,
)
from evaluation import (
    binary_attribute_probe_auc,
    calibration_unknown_diagnostic,
    domain_probe_auc,
    fit_novelty_geometry,
    fixed_frame_metrics,
    geometry_summary,
    novelty_scores,
)
from model_and_losses import (
    ARM_DEFINITIONS,
    EXPECTED_STAGE26_ARCHITECTURE_SIGNATURE,
    EXPECTED_STAGE26_PARAMETER_COUNT,
    EMAPrototypeBank,
    WiSigRepresentationNet,
    apply_rf_augmentation,
    architecture_signature,
    objective_loss,
)
from phase4_common import atomic_json, atomic_text, load_json, sha256_file, sha256_json, utc_now
from wisig_dataset import CanonicalWiSigStore, DomainBalancedTxBatchSampler, IndexedWiSigDataset


@dataclass
class PreStudyConfig:
    profile: str = "full"
    seeds: Tuple[int, ...] = (42, 123, 2026)
    max_epochs: int = 40
    minimum_epochs: int = 12
    early_stopping_patience: int = 10
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
    covariance_fit_limit: int = 100_000
    domain_sample_per_tx: int = 100
    bootstrap_iterations: int = 1000
    practical_f1_delta: float = 0.002
    noninferiority_f1_delta: float = 0.001
    require_t4_for_full: bool = True

    def apply_profile(self) -> None:
        if self.profile == "full":
            if tuple(self.seeds) != (42, 123, 2026):
                raise RuntimeError("Full Phase-4 pre-study freezes seeds=(42,123,2026)")
            return
        if self.profile != "pilot":
            raise ValueError("profile must be full or pilot")
        self.seeds = (42,)
        self.max_epochs = 2
        self.minimum_epochs = 1
        self.early_stopping_patience = 1
        self.bootstrap_iterations = 100
        self.covariance_fit_limit = 10_000
        self.domain_sample_per_tx = 20



def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)

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
    s = int(torch.initial_seed() % (2**32))
    random.seed(s)
    np.random.seed(s)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def deterministic_subset_positions(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for c in sorted(np.unique(labels).tolist()):
        p = np.flatnonzero(labels == c)
        if len(p) > per_class:
            p = np.sort(rng.choice(p, size=per_class, replace=False))
        out.append(p)
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def cuda_preflight(config: PreStudyConfig) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 4 requires a CUDA GPU runtime")
    name = torch.cuda.get_device_name(0)
    if config.profile == "full" and config.require_t4_for_full and "T4" not in name.upper():
        raise RuntimeError(f"Canonical full Phase-4 run requires Tesla T4; observed {name}")
    props = torch.cuda.get_device_properties(0)
    return {
        "device_name": name,
        "total_memory_bytes": int(props.total_memory),
        "compute_capability": [int(props.major), int(props.minor)],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "amp_enabled": bool(config.amp_enabled),
    }


def build_known_label_mapping(store: CanonicalWiSigStore) -> Tuple[Dict[int, int], Dict[str, Any]]:
    p3 = resolve_phase3_wisig_root()
    roles = load_json(p3 / "CLASS_ROLE_MANIFEST.json")
    known_ids = list(roles["known_transmitter_identities"])
    if len(known_ids) != 98:
        raise RuntimeError("Phase-3 known identity count changed")
    source = {v: i for i, v in enumerate(store.tx_values)}
    missing = [v for v in known_ids if v not in source]
    if missing:
        raise RuntimeError(f"Phase-3 known identities missing from Phase-1 source mapping: {missing}")
    mapping = {source[v]: cls for cls, v in enumerate(known_ids)}
    return mapping, {
        "known_identity_order": known_ids,
        "source_label_to_known_class": {str(k): int(v) for k, v in mapping.items()},
        "mapping_sha256": sha256_json({str(k): int(v) for k, v in mapping.items()}),
    }


def make_dataset(store, role: str, purpose: str, mapping: Mapping[int, int], profile: str, seed: int):
    idx, evidence = resolve_authorized_indices(role, purpose)
    allow_unknown = role in UNKNOWN_DIAGNOSTIC_ROLES
    ds = IndexedWiSigDataset(store, idx, mapping, allow_unknown_labels=allow_unknown)
    if profile == "pilot":
        rng = np.random.default_rng(seed + len(role))
        if allow_unknown:
            take = min(10_000, len(ds))
            pos = np.sort(rng.choice(len(ds), size=take, replace=False))
        else:
            # Keep all 98 classes represented.
            per_class = 200 if role == "train_known" else 50
            pos = deterministic_subset_positions(ds.known_class, per_class, seed + len(role))
        ds = IndexedWiSigDataset(store, ds.indices[pos], mapping, allow_unknown_labels=allow_unknown)
        evidence = dict(evidence)
        evidence["pilot_subset_count"] = len(ds)
    return ds, evidence


def make_loader(ds, batch_size: int, config: PreStudyConfig, shuffle: bool = False):
    g = torch.Generator().manual_seed(12345)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=2 if config.num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        generator=g,
    )


def evaluate_known(model, ds, device, config) -> Dict[str, Any]:
    loader = make_loader(ds, config.eval_batch_size, config)
    out = collect_outputs(model, loader, device)
    return fixed_frame_metrics(out["logits"], out["labels"], 98)


def collect_outputs(model, loader, device) -> Dict[str, np.ndarray]:
    model.eval()
    emb, logits, labels, receiver, day, eq, gid = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            o = model(x)
            emb.append(o["embedding_normalized"].detach().cpu().numpy())
            logits.append(o["logits"].detach().float().cpu().numpy())
            labels.append(batch["y"].numpy())
            receiver.append(batch["receiver"].numpy())
            day.append(batch["day"].numpy())
            eq.append(batch["equalization"].numpy())
            gid.append(batch["global_index"].numpy())
    return {
        "embedding": np.concatenate(emb),"logits": np.concatenate(logits),"labels": np.concatenate(labels),
        "receiver": np.concatenate(receiver),"day": np.concatenate(day),"equalization": np.concatenate(eq),"global_index": np.concatenate(gid),
    }
