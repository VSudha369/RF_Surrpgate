#!/usr/bin/env python3
from __future__ import annotations

import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


REQUIRED_ROOT_KEYS = (
    "tx_list",
    "rx_list",
    "capture_date_list",
    "equalized_list",
    "max_sig",
    "data",
)


def resolve_manytx_member(zip_path: Path, preferred: str = "ManyTx.pkl") -> zipfile.ZipInfo:
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        exact = [i for i in infos if i.filename == preferred]
        if len(exact) == 1:
            return exact[0]
        by_basename = [i for i in infos if Path(i.filename).name.lower() == preferred.lower()]
        if len(by_basename) == 1:
            return by_basename[0]
        raise RuntimeError(
            f"Expected exactly one {preferred!r} member in {zip_path}; "
            f"members={[i.filename for i in infos[:50]]}"
        )


def safe_extract_manytx_pickle(zip_path: Path, out_dir: Path,
                               preferred: str = "ManyTx.pkl") -> Path:
    """
    Extract only the ManyTx pickle, locally, without trusting archive paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    info = resolve_manytx_member(zip_path, preferred)
    dst = out_dir / preferred
    partial = dst.with_suffix(dst.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf, zf.open(info, "r") as src, partial.open("wb") as dstf:
        while True:
            b = src.read(16 * 1024 * 1024)
            if not b:
                break
            dstf.write(b)
        dstf.flush()
        import os
        os.fsync(dstf.fileno())
    partial.replace(dst)
    return dst


def load_manytx_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f, encoding="latin1")
    if not isinstance(obj, dict):
        raise TypeError(f"ManyTx root must be dict, got {type(obj).__name__}")
    missing = [k for k in REQUIRED_ROOT_KEYS if k not in obj]
    if missing:
        raise RuntimeError(f"ManyTx root missing required keys: {missing}")
    return obj


def _cell_to_n2l(cell: Any) -> Tuple[np.ndarray, str]:
    """
    Convert one source cell to contiguous float32 [N,2,L], preserving values.

    Source-authentic WiSig ManyTx is expected to be [N,L,IQ], but this converter
    also recognizes [N,IQ,L] and complex [N,L] for defensive validation.
    """
    a = np.asarray(cell)

    if a.size == 0:
        return np.empty((0, 2, 0), dtype=np.float32), "EMPTY"

    if np.iscomplexobj(a):
        if a.ndim == 1:
            a = a[None, :]
        if a.ndim != 2:
            raise ValueError(f"Complex ManyTx cell must be [N,L], got {a.shape}")
        x = np.stack([a.real, a.imag], axis=1)
        return np.ascontiguousarray(x, dtype=np.float32), "N,L_COMPLEX"

    if a.ndim == 2:
        # Defensive support for one [L,2] or [2,L] sample.
        if a.shape[-1] == 2:
            a = a[None, ...]
        elif a.shape[0] == 2:
            a = a[None, ...]
        else:
            raise ValueError(f"Cannot infer I/Q axis from ManyTx cell {a.shape}")

    if a.ndim != 3:
        raise ValueError(f"ManyTx cell must be rank 3, got {a.shape}")

    if a.shape[-1] == 2:
        # Source report: [N,L,IQ].
        x = np.transpose(a, (0, 2, 1))
        orientation = "N,L,IQ"
    elif a.shape[1] == 2:
        x = a
        orientation = "N,IQ,L"
    else:
        raise ValueError(f"ManyTx cell has no I/Q dimension of length 2: {a.shape}")

    return np.ascontiguousarray(x, dtype=np.float32), orientation


def inspect_manytx_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    tx_list = list(obj["tx_list"])
    rx_list = list(obj["rx_list"])
    date_list = list(obj["capture_date_list"])
    eq_list = list(obj["equalized_list"])
    data = obj["data"]

    if len(data) != len(tx_list):
        raise RuntimeError(f"axis0 mismatch data={len(data)} tx={len(tx_list)}")

    total = 0
    nonempty = 0
    empty = 0
    lengths = set()
    orientations = set()
    cell_shape_examples: List[List[int]] = []

    for ti in range(len(tx_list)):
        if len(data[ti]) != len(rx_list):
            raise RuntimeError(f"axis1 mismatch at tx={ti}")
        for ri in range(len(rx_list)):
            if len(data[ti][ri]) != len(date_list):
                raise RuntimeError(f"axis2 mismatch at tx={ti}, rx={ri}")
            for di in range(len(date_list)):
                if len(data[ti][ri][di]) != len(eq_list):
                    raise RuntimeError(f"axis3 mismatch at tx={ti}, rx={ri}, date={di}")
                for ei in range(len(eq_list)):
                    cell = data[ti][ri][di][ei]
                    a = np.asarray(cell)
                    if a.size == 0:
                        empty += 1
                        continue
                    x, orientation = _cell_to_n2l(cell)
                    nonempty += 1
                    total += int(x.shape[0])
                    lengths.add(int(x.shape[2]))
                    orientations.add(orientation)
                    if len(cell_shape_examples) < 10:
                        cell_shape_examples.append(list(a.shape))

    return {
        "schema": "WISIG_MANYTX_NATIVE_NESTED_V1",
        "root_keys": list(obj.keys()),
        "tx_count": len(tx_list),
        "receiver_count": len(rx_list),
        "capture_date_count": len(date_list),
        "equalization_count": len(eq_list),
        "total_cells": len(tx_list) * len(rx_list) * len(date_list) * len(eq_list),
        "nonempty_cells": nonempty,
        "empty_cells": empty,
        "n_samples": total,
        "signal_lengths": sorted(lengths),
        "source_orientations": sorted(orientations),
        "max_sig": int(obj["max_sig"]),
        "tx_values": [str(v) for v in tx_list],
        "receiver_values": [str(v) for v in rx_list],
        "capture_date_values": [str(v) for v in date_list],
        "equalization_values": [
            int(v) if isinstance(v, (int, np.integer)) else str(v) for v in eq_list
        ],
        "cell_shape_examples": cell_shape_examples,
        "axis_mapping": {
            "axis0": "tx_list",
            "axis1": "rx_list",
            "axis2": "capture_date_list",
            "axis3": "equalized_list",
        },
        "deterministic_flatten_order": (
            "tx_index -> receiver_index -> capture_date_index -> "
            "equalization_index -> sample_index_within_cell"
        ),
        "session_id": "NOT_REPRESENTED",
    }


def validate_manytx_inspection(inspection: Dict[str, Any],
                               expected: Optional[Dict[str, Any]] = None) -> None:
    expected = expected or {}
    checks = {
        "n_samples": expected.get("n_samples", 1_020_643),
        "sequence_length": expected.get("sequence_length", 256),
        "tx_count": expected.get("tx_count", 150),
        "receiver_count": expected.get("receiver_count", 18),
        "capture_date_count": expected.get("capture_date_count", 4),
        "equalization_count": expected.get("equalization_count", 2),
        "nonempty_cells": expected.get("nonempty_cells", 20_759),
        "empty_cells": expected.get("empty_cells", 841),
    }

    observed = {
        "n_samples": inspection["n_samples"],
        "sequence_length": (
            inspection["signal_lengths"][0]
            if len(inspection["signal_lengths"]) == 1 else inspection["signal_lengths"]
        ),
        "tx_count": inspection["tx_count"],
        "receiver_count": inspection["receiver_count"],
        "capture_date_count": inspection["capture_date_count"],
        "equalization_count": inspection["equalization_count"],
        "nonempty_cells": inspection["nonempty_cells"],
        "empty_cells": inspection["empty_cells"],
    }

    failures = []
    for k, exp in checks.items():
        if exp is not None and observed[k] != exp:
            failures.append(f"{k}: expected={exp!r}, observed={observed[k]!r}")

    if inspection["source_orientations"] != ["N,L,IQ"]:
        failures.append(
            "source_orientations: expected source-authentic ['N,L,IQ'], "
            f"observed={inspection['source_orientations']!r}"
        )

    if failures:
        raise RuntimeError(
            "WiSig ManyTx native schema invariant failure: " + " | ".join(failures)
        )


def iter_manytx_object(obj: Dict[str, Any], chunk_samples: int = 8192
                       ) -> Iterator[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    tx_list = list(obj["tx_list"])
    rx_list = list(obj["rx_list"])
    date_list = list(obj["capture_date_list"])
    eq_list = list(obj["equalized_list"])
    data = obj["data"]

    gid = 0
    cell_id = 0

    for ti in range(len(tx_list)):
        for ri in range(len(rx_list)):
            for di in range(len(date_list)):
                for ei in range(len(eq_list)):
                    cell = data[ti][ri][di][ei]
                    a = np.asarray(cell)
                    if a.size == 0:
                        cell_id += 1
                        continue

                    x, orientation = _cell_to_n2l(cell)
                    if orientation != "N,L,IQ":
                        raise RuntimeError(
                            f"Unexpected ManyTx orientation at cell "
                            f"({ti},{ri},{di},{ei}): {orientation}"
                        )

                    for start in range(0, len(x), chunk_samples):
                        chunk = x[start:start + chunk_samples]
                        n = len(chunk)
                        g = np.arange(gid, gid + n, dtype=np.int64)
                        meta = {
                            "global_index": g,
                            "label": np.full(n, ti, dtype=np.int32),
                            "tx_id": np.full(n, ti, dtype=np.int32),
                            "receiver": np.full(n, ri, dtype=np.int16),
                            "day": np.full(n, di, dtype=np.int16),
                            "capture_date": np.full(n, di, dtype=np.int16),
                            "equalization": np.full(n, ei, dtype=np.int8),
                            "source_cell_id": np.full(n, cell_id, dtype=np.int32),
                            "sample_in_cell": np.arange(
                                start, start + n, dtype=np.int32
                            ),
                        }
                        yield chunk, meta
                        gid += n
                    cell_id += 1
