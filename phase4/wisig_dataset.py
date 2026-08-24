#!/usr/bin/env python3
from __future__ import annotations

import bisect
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from phase4_common import load_json


class CanonicalWiSigStore:
    """Read-only local Phase-1 canonical WiSig shards; signals are memory-mapped."""

    def __init__(self, canonical_root: Path):
        self.root = Path(canonical_root).resolve()
        if str(self.root).startswith("/content/drive/"):
            raise RuntimeError("WiSig scientific data must be staged under /content; Drive hot I/O is forbidden")
        dm = load_json(self.root / "DATASET_MANIFEST.json")
        if dm.get("n_samples") != 1_020_643 or dm.get("seq_len") != 256:
            raise RuntimeError("Unexpected canonical WiSig N/L")
        if dm.get("source_sha256") != "a8fc3e35134a240bfb4dab8862a6e482cef44de000b813d42417b853c47ccc7e":
            raise RuntimeError("Unexpected canonical WiSig source SHA")
        mappings = load_json(self.root / "SOURCE_VALUE_MAPPINGS.json")
        self.tx_values = list(mappings["tx_values"])
        self.receiver_values = list(mappings["receiver_values"])
        self.capture_date_values = list(mappings["capture_date_values"])
        self.equalization_values = list(mappings["equalization_values"])
        self.shards = []
        self.ends: List[int] = []
        cursor = 0
        for row in dm["shards"]:
            sig = self.root / row["signal_path"]
            meta = self.root / row["metadata_path"]
            n = int(row["n_samples"])
            self.shards.append((cursor, cursor + n, sig, meta))
            cursor += n
            self.ends.append(cursor)
        if cursor != 1_020_643:
            raise RuntimeError("Canonical shard coverage does not span WiSig universe")
        self._signal_maps: Dict[int, np.ndarray] = {}
        self._meta_cache: Dict[int, Dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return 1_020_643

    def _locate(self, global_index: int) -> Tuple[int, int]:
        if global_index < 0 or global_index >= len(self):
            raise IndexError(global_index)
        sid = bisect.bisect_right(self.ends, global_index)
        start = self.shards[sid][0]
        return sid, global_index - start

    def _signals(self, sid: int) -> np.ndarray:
        if sid not in self._signal_maps:
            self._signal_maps[sid] = np.load(self.shards[sid][2], mmap_mode="r", allow_pickle=False)
        return self._signal_maps[sid]

    def _metadata(self, sid: int) -> Dict[str, np.ndarray]:
        if sid not in self._meta_cache:
            with np.load(self.shards[sid][3], allow_pickle=False) as z:
                self._meta_cache[sid] = {k: z[k] for k in z.files}
        return self._meta_cache[sid]

    def signal(self, global_index: int) -> np.ndarray:
        sid, off = self._locate(int(global_index))
        x = np.asarray(self._signals(sid)[off], dtype=np.float32)
        if x.shape != (2, 256):
            raise RuntimeError(f"Unexpected signal shape {x.shape}")
        return np.array(x, copy=True)

    def metadata_for_indices(self, indices: np.ndarray, fields: Sequence[str]) -> Dict[str, np.ndarray]:
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("indices must be 1D")
        out = {f: np.empty(len(idx), dtype=np.int64) for f in fields}
        order = np.argsort(idx, kind="mergesort")
        sorted_idx = idx[order]
        for sid, (start, end, _, _) in enumerate(self.shards):
            lo = np.searchsorted(sorted_idx, start, side="left")
            hi = np.searchsorted(sorted_idx, end, side="left")
            if lo == hi:
                continue
            meta = self._metadata(sid)
            g = sorted_idx[lo:hi]
            offs = g - start
            target = order[lo:hi]
            for f in fields:
                if f not in meta:
                    raise RuntimeError(f"Canonical WiSig metadata missing {f}")
                out[f][target] = np.asarray(meta[f][offs], dtype=np.int64)
        return out


def preprocess_iq_numpy(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Complex DC removal + per-sample complex RMS normalization; native length preserved."""
    y = np.asarray(x, dtype=np.float32).copy()
    y -= y.mean(axis=1, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(y.astype(np.float64) ** 2, axis=0))))
    if not np.isfinite(rms) or rms < eps:
        raise FloatingPointError(f"Invalid sample RMS: {rms}")
    y /= np.float32(max(rms, eps))
    if not np.isfinite(y).all():
        raise FloatingPointError("Non-finite preprocessed sample")
    return y


class IndexedWiSigDataset(Dataset):
    def __init__(
        self,
        store: CanonicalWiSigStore,
        indices: np.ndarray,
        source_label_to_known_class: Mapping[int, int],
        allow_unknown_labels: bool = False,
    ):
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.source_to_known = dict(source_label_to_known_class)
        self.allow_unknown = allow_unknown_labels
        self.meta = store.metadata_for_indices(
            self.indices,
            ["label", "receiver", "capture_date", "equalization"],
        )
        known = np.array([self.source_to_known.get(int(v), -1) for v in self.meta["label"]], dtype=np.int64)
        if not allow_unknown_labels and np.any(known < 0):
            bad = np.unique(self.meta["label"][known < 0])[:10]
            raise RuntimeError(f"Known-role indices contain non-known transmitter labels: {bad.tolist()}")
        self.known_class = known

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        gid = int(self.indices[pos])
        x = preprocess_iq_numpy(self.store.signal(gid))
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(int(self.known_class[pos]), dtype=torch.long),
            "source_label": torch.tensor(int(self.meta["label"][pos]), dtype=torch.long),
            "receiver": torch.tensor(int(self.meta["receiver"][pos]), dtype=torch.long),
            "capture_date": torch.tensor(int(self.meta["capture_date"][pos]), dtype=torch.long),
            "equalization": torch.tensor(int(self.meta["equalization"][pos]), dtype=torch.long),
            "global_index": torch.tensor(gid, dtype=torch.long),
        }


class DomainBalancedTxBatchSampler(Sampler[List[int]]):
    """Exact Stage-2.6M Tx-primary sampler over V2 canonical metadata."""

    def __init__(
        self,
        dataset: IndexedWiSigDataset,
        batch_size: int = 256,
        samples_per_tx: int = 4,
        seed: int = 0,
        epoch: int = 0,
    ):
        if samples_per_tx <= 1 or batch_size % samples_per_tx:
            raise ValueError("batch_size must be divisible by samples_per_tx > 1")
        self.dataset = dataset
        self.labels = np.asarray(dataset.known_class, dtype=np.int64)
        self.receiver = np.asarray(dataset.meta["receiver"], dtype=object)
        self.day = np.asarray(dataset.meta["capture_date"], dtype=object)
        self.equalized = np.asarray(dataset.meta["equalization"], dtype=np.int8)
        self.batch_size = int(batch_size)
        self.samples_per_tx = int(samples_per_tx)
        self.classes_per_batch = self.batch_size // self.samples_per_tx
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.steps = math.ceil(len(self.labels) / self.batch_size)
        self.class_cells: Dict[int, Dict[Tuple[str, str, int], np.ndarray]] = {}
        for label in sorted(np.unique(self.labels).tolist()):
            positions = np.flatnonzero(self.labels == label)
            cells: Dict[Tuple[str, str, int], List[int]] = defaultdict(list)
            for pos in positions:
                cells[(str(self.receiver[pos]), str(self.day[pos]), int(self.equalized[pos]))].append(int(pos))
            self.class_cells[int(label)] = {
                key: np.asarray(values, dtype=np.int64)
                for key, values in sorted(cells.items(), key=lambda kv: str(kv[0]))
            }
        if len(self.class_cells) != 98:
            raise RuntimeError(f"Sampler received {len(self.class_cells)} classes instead of 98")

    def __len__(self) -> int:
        return self.steps

    def materialize(self, epoch: Optional[int] = None) -> List[List[int]]:
        if epoch is not None:
            self.epoch = int(epoch)
        return list(iter(self))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 104_729 * self.epoch)
        classes = np.asarray(sorted(self.class_cells), dtype=np.int64)
        class_order = rng.permutation(classes).tolist()
        class_cursor = 0
        cell_orders: Dict[int, List[Tuple[str, str, int]]] = {}
        cell_cursors: Dict[int, int] = {}
        sample_orders: Dict[Tuple[int, Tuple[str, str, int]], np.ndarray] = {}
        sample_cursors: Dict[Tuple[int, Tuple[str, str, int]], int] = {}
        for label, cells in self.class_cells.items():
            keys = list(cells)
            rng.shuffle(keys)
            cell_orders[label] = keys
            cell_cursors[label] = 0
            for key, values in cells.items():
                sample_orders[(label, key)] = rng.permutation(values)
                sample_cursors[(label, key)] = 0

        def next_class() -> int:
            nonlocal class_cursor, class_order
            if class_cursor >= len(class_order):
                class_order = rng.permutation(classes).tolist()
                class_cursor = 0
            value = int(class_order[class_cursor])
            class_cursor += 1
            return value

        def next_position(label: int) -> int:
            keys = cell_orders[label]
            cell_cursor = cell_cursors[label]
            key = keys[cell_cursor % len(keys)]
            cell_cursors[label] = cell_cursor + 1
            compound = (label, key)
            order = sample_orders[compound]
            cursor = sample_cursors[compound]
            if cursor >= len(order):
                order = rng.permutation(self.class_cells[label][key])
                sample_orders[compound] = order
                cursor = 0
            value = int(order[cursor])
            sample_cursors[compound] = cursor + 1
            return value

        for _ in range(self.steps):
            chosen_classes: List[int] = []
            while len(chosen_classes) < self.classes_per_batch:
                candidate = next_class()
                if candidate not in chosen_classes:
                    chosen_classes.append(candidate)
            batch: List[int] = []
            for label in chosen_classes:
                for _sample in range(self.samples_per_tx):
                    batch.append(next_position(label))
            if len(batch) != self.batch_size:
                raise RuntimeError("Sampler produced an incorrectly sized batch")
            yield batch

