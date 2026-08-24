#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

from phase4_common import load_json, sha256_file

PHASE3_CANONICAL_RUN_ID = "20260824T100504Z"
PHASE3_DRIVE_WISIG = Path(
    "/content/drive/MyDrive/Surrogate_XAI_V2/03_PHASE3_OPEN_SET_SPLITS/"
    f"run_{PHASE3_CANONICAL_RUN_ID}/wisig"
)
PHASE3_LOCAL_WISIG = Path(
    "/content/surrogate_xai_v2/03_phase3_open_set_splits/"
    f"run_{PHASE3_CANONICAL_RUN_ID}/wisig"
)
EXPECTED_SOURCE_SHA256 = "a8fc3e35134a240bfb4dab8862a6e482cef44de000b813d42417b853c47ccc7e"

TRAIN_ROLES = frozenset({"train_known"})
KNOWN_EVAL_ROLES = frozenset({"p0_known", "p1_cross_day", "p2_cross_receiver", "p3_cross_day_receiver"})
UNKNOWN_DIAGNOSTIC_ROLES = frozenset({"calibration_unknown"})
AUTHORIZED_ROLES = TRAIN_ROLES | KNOWN_EVAL_ROLES | UNKNOWN_DIAGNOSTIC_ROLES
STRICT_TOKENS = ("strict_zero_day", "strict_zero_day_shift", "strict_unknown")


class StrictAccessViolation(RuntimeError):
    pass


def resolve_phase3_wisig_root() -> Path:
    # Prefer the canonical completed local run when still present; otherwise use Drive manifests.
    for root in (PHASE3_LOCAL_WISIG, PHASE3_DRIVE_WISIG):
        if (root / "PUBLIC_SPLIT_MANIFEST.json").is_file() and (root / "DATA_ACCESS_POLICY.json").is_file():
            return root
    raise FileNotFoundError(
        "Canonical Phase-3 WiSig run not found. Expected run_20260824T100504Z locally or on Drive."
    )


def _reject_role(role: str) -> None:
    low = role.lower()
    if any(tok in low for tok in STRICT_TOKENS):
        raise StrictAccessViolation(f"Phase 4 permanently forbids strict role access: {role}")
    if role not in AUTHORIZED_ROLES:
        raise RuntimeError(f"Unauthorized Phase-4 role: {role}")


def verify_phase3_contract() -> Dict[str, Any]:
    root = resolve_phase3_wisig_root()
    public = load_json(root / "PUBLIC_SPLIT_MANIFEST.json")
    policy = load_json(root / "DATA_ACCESS_POLICY.json")
    roles = public.get("allowed_roles", {})
    if set(roles) != set(AUTHORIZED_ROLES):
        raise RuntimeError(f"Phase-3 allowed role set changed: {sorted(roles)}")
    if public.get("source_archive_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Phase-3 WiSig source SHA changed")
    if policy.get("unknown_gradients_forbidden") is not True:
        raise RuntimeError("Phase-3 unknown-gradient policy is not frozen true")
    if policy.get("strict_arrays_loaded") is not False or policy.get("strict_index_files_exported") is not False:
        raise StrictAccessViolation("Phase-3 strict-array safety contract changed")
    # A structural audit: exported role files must be exactly the six authorized files.
    exported = {p.name for p in root.glob("*_indices.npy")}
    expected = {f"{r}_indices.npy" for r in AUTHORIZED_ROLES}
    if exported != expected:
        extra = sorted(exported - expected)
        missing = sorted(expected - exported)
        if any("strict" in x for x in extra):
            raise StrictAccessViolation(f"Strict index unexpectedly exported: {extra}")
        raise RuntimeError(f"Phase-3 exported-index set mismatch: missing={missing}, extra={extra}")
    return {
        "phase3_root": str(root),
        "phase3_run_id": PHASE3_CANONICAL_RUN_ID,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "authorized_roles": sorted(AUTHORIZED_ROLES),
        "strict_roles_resolvable": False,
        "policy": policy,
    }


def resolve_authorized_indices(role: str, purpose: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    _reject_role(role)
    if purpose == "gradient" and role not in TRAIN_ROLES:
        raise RuntimeError(f"Gradients are only authorized for train_known; got {role}")
    if purpose not in {"gradient", "model_selection", "known_eval", "unknown_diagnostic"}:
        raise RuntimeError(f"Unknown access purpose: {purpose}")
    if role in UNKNOWN_DIAGNOSTIC_ROLES and purpose != "unknown_diagnostic":
        raise RuntimeError("Calibration Unknown is diagnostic-only and cannot be used for model selection or gradients")
    if role in KNOWN_EVAL_ROLES and purpose == "gradient":
        raise RuntimeError("Known validation/evaluation roles cannot contribute gradients")

    root = resolve_phase3_wisig_root()
    public = load_json(root / "PUBLIC_SPLIT_MANIFEST.json")
    spec = public["allowed_roles"][role]
    path = root / spec["file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise RuntimeError(f"Phase-3 split hash mismatch for {role}")
    arr = np.load(path, allow_pickle=False)
    if arr.dtype != np.int64 or arr.ndim != 1 or len(arr) != int(spec["count"]):
        raise RuntimeError(f"Phase-3 index shape/count mismatch for {role}")
    return arr, {"role": role, "count": len(arr), "sha256": digest, "path": str(path), "purpose": purpose}


def strict_access_self_test() -> Dict[str, Any]:
    blocked = []
    for role in ("strict_zero_day", "strict_zero_day_shift", "strict_unknown"):
        try:
            resolve_authorized_indices(role, "known_eval")
        except StrictAccessViolation:
            blocked.append(role)
        else:
            raise RuntimeError(f"Strict access self-test FAILED for {role}")
    return {"strict_access_blocked": True, "blocked_roles": blocked}
