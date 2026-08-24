#!/usr/bin/env python3
from __future__ import annotations
import json, math, pickle, re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import numpy as np
import h5py

def _to_iq_n2l(x: np.ndarray) -> np.ndarray:
    """Convert common RF layouts to contiguous float32 [N,2,L] without changing values."""
    x = np.asarray(x)
    if np.iscomplexobj(x):
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            raise ValueError(f"Complex RF array must be [N,L], got {x.shape}")
        x = np.stack([x.real, x.imag], axis=1)
    elif x.ndim == 2:
        # Single [2,L] sample or [L,2] sample.
        if x.shape[0] == 2:
            x = x[None, :, :]
        elif x.shape[1] == 2:
            x = x.T[None, :, :]
        else:
            raise ValueError(f"Cannot infer I/Q axes from shape {x.shape}")
    elif x.ndim == 3:
        if x.shape[1] == 2:
            pass
        elif x.shape[2] == 2:
            x = np.transpose(x, (0, 2, 1))
        else:
            raise ValueError(f"Expected one I/Q axis of length 2, got {x.shape}")
    else:
        raise ValueError(f"Unsupported RF array rank/shape: {x.shape}")
    return np.ascontiguousarray(x, dtype=np.float32)

def inspect_hdf5(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                rows.append({
                    "name": name,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "ndim": obj.ndim,
                })
        f.visititems(visitor)
    return rows

def auto_detect_hdf5_signal_dataset(path: Path) -> str:
    rows = inspect_hdf5(path)
    scored = []
    for r in rows:
        shape = r["shape"]
        if len(shape) not in (2, 3):
            continue
        dtype = np.dtype(r["dtype"])
        looks = False
        if dtype.kind == "c" and len(shape) == 2:
            looks = True
        if len(shape) == 3 and (shape[1] == 2 or shape[2] == 2):
            looks = True
        if not looks:
            continue
        name = r["name"].lower()
        score = int(np.prod(shape))
        if any(k in name for k in ("signal", "iq", "samples", "x")):
            score *= 4
        scored.append((score, r["name"]))
    if not scored:
        raise RuntimeError(
            f"No HDF5 signal dataset could be auto-detected. Tree={rows[:50]}"
        )
    scored.sort(reverse=True)
    return scored[0][1]

def _metadata_candidates(rows: List[Dict[str, Any]], n: int, tokens: Tuple[str, ...]) -> List[str]:
    out = []
    for r in rows:
        if not r["shape"] or r["shape"][0] != n:
            continue
        name = r["name"].lower()
        if any(tok in name for tok in tokens):
            out.append(r["name"])
    return out

def inspect_wisig_hdf5(path: Path, explicit_signal: Optional[str] = None) -> Dict[str, Any]:
    rows = inspect_hdf5(path)
    signal = explicit_signal or auto_detect_hdf5_signal_dataset(path)
    sigrow = next(r for r in rows if r["name"] == signal)
    n = sigrow["shape"][0]
    return {
        "hdf5_tree": rows,
        "detected_signal_dataset": signal,
        "n_samples": int(n),
        "candidate_label_datasets": _metadata_candidates(rows, n, ("label","tx","transmitter","device")),
        "candidate_receiver_datasets": _metadata_candidates(rows, n, ("receiver","rx")),
        "candidate_day_datasets": _metadata_candidates(rows, n, ("day","capture","date")),
        "candidate_equalization_datasets": _metadata_candidates(rows, n, ("equal",)),
    }

def inspect_radioml2018(path: Path, x_key: Optional[str] = None,
                        y_key: Optional[str] = None, z_key: Optional[str] = None) -> Dict[str, Any]:
    rows = inspect_hdf5(path)
    names = {r["name"] for r in rows}
    x_key = x_key or ("X" if "X" in names else auto_detect_hdf5_signal_dataset(path))
    with h5py.File(path, "r") as f:
        x_shape = list(f[x_key].shape)
        n = x_shape[0]
        if y_key is None:
            for cand in ("Y", "labels", "label"):
                if cand in f:
                    y_key = cand; break
        if z_key is None:
            for cand in ("Z", "snr", "SNR"):
                if cand in f:
                    z_key = cand; break
        if y_key is None or z_key is None:
            raise RuntimeError(f"Could not resolve Y/Z datasets. HDF5 tree={rows[:50]}")
        return {
            "hdf5_tree": rows, "x_key": x_key, "y_key": y_key, "z_key": z_key,
            "n_samples": int(n), "x_shape": x_shape,
            "y_shape": list(f[y_key].shape), "z_shape": list(f[z_key].shape),
        }

def inspect_radioml2016(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f, encoding="latin1")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict pickle for RadioML2016, got {type(obj)}")
    keys = sorted(obj.keys(), key=lambda k: (str(k[0]), float(k[1])))
    mods = sorted({str(k[0]) for k in keys})
    snrs = sorted({int(k[1]) for k in keys})
    total = 0
    example_shape = None
    for k in keys:
        a = np.asarray(obj[k])
        total += a.shape[0]
        if example_shape is None:
            example_shape = list(a.shape)
    return {
        "n_samples": int(total), "n_cells": len(keys), "modulations": mods,
        "snrs": snrs, "example_cell_shape": example_shape,
    }

def iter_radioml2016(path: Path, chunk_samples: int = 8192):
    with path.open("rb") as f:
        obj = pickle.load(f, encoding="latin1")
    keys = sorted(obj.keys(), key=lambda k: (str(k[0]), float(k[1])))
    mods = sorted({str(k[0]) for k in keys})
    mod_to_id = {m: i for i, m in enumerate(mods)}
    gid = 0
    for mod_raw, snr_raw in keys:
        mod = str(mod_raw)
        snr = int(snr_raw)
        arr = _to_iq_n2l(np.asarray(obj[(mod_raw, snr_raw)]))
        for start in range(0, arr.shape[0], chunk_samples):
            x = arr[start:start+chunk_samples]
            n = len(x)
            meta = {
                "global_index": np.arange(gid, gid+n, dtype=np.int64),
                "label": np.full(n, mod_to_id[mod], dtype=np.int32),
                "snr": np.full(n, snr, dtype=np.int16),
            }
            yield x, meta
            gid += n

def iter_radioml2018(path: Path, x_key: str, y_key: str, z_key: str,
                     chunk_samples: int = 8192):
    with h5py.File(path, "r") as f:
        X, Y, Z = f[x_key], f[y_key], f[z_key]
        n_total = X.shape[0]
        gid = 0
        for start in range(0, n_total, chunk_samples):
            stop = min(start + chunk_samples, n_total)
            x = _to_iq_n2l(X[start:stop])
            y = np.asarray(Y[start:stop])
            if y.ndim == 2:
                label = np.argmax(y, axis=1).astype(np.int32)
            else:
                label = y.reshape(-1).astype(np.int32)
            snr = np.asarray(Z[start:stop]).reshape(-1).astype(np.int16)
            n = len(x)
            meta = {
                "global_index": np.arange(gid, gid+n, dtype=np.int64),
                "label": label,
                "snr": snr,
            }
            yield x, meta
            gid += n

def _read_h5_vector(f: h5py.File, key: Optional[str], start: int, stop: int,
                    default: int = -1, dtype=np.int32) -> np.ndarray:
    n = stop - start
    if not key:
        return np.full(n, default, dtype=dtype)
    a = np.asarray(f[key][start:stop])
    if a.dtype.fields:
        raise ValueError(f"Compound dataset {key} needs an explicit field adapter.")
    if a.ndim > 1:
        if a.shape[1] == 1:
            a = a[:,0]
        else:
            raise ValueError(f"Metadata dataset {key} is not a vector: {a.shape}")
    if a.dtype.kind in ("S","U","O"):
        # Stable categorical coding by lexicographic values within the entire dataset
        full = np.asarray(f[key][:]).astype(str)
        vals = sorted(set(full.tolist()))
        mapping = {v:i for i,v in enumerate(vals)}
        return np.array([mapping[str(v)] for v in np.asarray(f[key][start:stop]).astype(str)], dtype=dtype)
    return np.asarray(a, dtype=dtype).reshape(-1)

def iter_wisig(path: Path, signal_key: str, label_key: Optional[str],
               receiver_key: Optional[str], day_key: Optional[str],
               equalization_key: Optional[str], chunk_samples: int = 8192):
    with h5py.File(path, "r") as f:
        X = f[signal_key]
        n_total = X.shape[0]
        gid = 0
        for start in range(0, n_total, chunk_samples):
            stop = min(start + chunk_samples, n_total)
            x = _to_iq_n2l(X[start:stop])
            n = len(x)
            meta = {
                "global_index": np.arange(gid, gid+n, dtype=np.int64),
                "label": _read_h5_vector(f, label_key, start, stop, dtype=np.int32),
                "receiver": _read_h5_vector(f, receiver_key, start, stop, dtype=np.int16),
                "day": _read_h5_vector(f, day_key, start, stop, dtype=np.int16),
                "equalization": _read_h5_vector(f, equalization_key, start, stop, dtype=np.int8),
            }
            yield x, meta
            gid += n
