#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import torch

from teacher_architectures import ARCHITECTURE_REGISTRY, build_teacher, parameter_count

HERE = Path(__file__).resolve().parent
FREEZE = HERE.parent / "phase4" / "PHASE4_V1_1_0_FREEZE.json"
CONFIG = HERE / "phase5_config_v1_0_0.json"
LOCK_TEMPLATE = HERE / "PHASE5_SELECTION_LOCK_TEMPLATE.json"
EXPECTED_ARCH = {"R0_RESNET", "T0_TCN", "X0_TRANSFORMER", "F0_TIME_FREQ"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    checks = {}
    freeze = json.loads(FREEZE.read_text())
    config = json.loads(CONFIG.read_text())
    lock = json.loads(LOCK_TEMPLATE.read_text())

    checks["phase4_frozen"] = freeze["freeze_status"] == "PHASE4_V1_1_0_FROZEN"
    checks["phase4_a0_locked"] = freeze["selection_lock"]["selected_arm"] == "A0"
    checks["phase4_ce_locked"] = freeze["selection_lock"]["selected_objective"] == "Cross-Entropy"
    checks["freeze_sha_matches_config"] = sha256(FREEZE) == config["phase4_freeze"]["freeze_manifest_sha256"]
    checks["objective_immutable_a0_ce"] = config["objective"] == {"arm":"A0","name":"Cross-Entropy","label_smoothing":0.0}
    checks["known_gradient_only"] = config["access"]["gradient_roles"] == ["train_known"]
    checks["p0_selection_only"] = config["access"]["selection_roles"] == ["p0_known"]
    checks["calibration_unknown_forbidden"] = config["access"]["calibration_unknown_allowed"] is False
    checks["strict_forbidden"] = config["access"]["strict_roles_allowed"] is False
    checks["native_l256"] = config["dataset"]["native_sequence_length"] == 256
    checks["architecture_registry_exact"] = set(ARCHITECTURE_REGISTRY) == EXPECTED_ARCH == set(config["architectures"])
    checks["selection_lock_not_preopened"] = lock["status"] == "NOT_LOCKED" and lock["selected_architecture"] is None
    checks["shift_locked_before_eval"] = config["selection"]["selection_must_lock_before_shift_evaluation"] is True

    source = (HERE / "teacher_architectures.py").read_text()
    ast.parse(source)
    checks["python_ast_parse"] = True
    checks["full_complex_fft_present"] = "torch.fft.fft(" in source and "torch.fft.fftshift" in source
    checks["legacy_65bin_slice_absent"] = not re.search(r"(?:[:\[,]\s*65\s*\])", source)

    counts = {}
    for name in sorted(EXPECTED_ARCH):
        model = build_teacher(name)
        counts[name] = parameter_count(model)
        for length in (128, 256, 1024):
            x = torch.randn(2, 2, length)
            with torch.no_grad():
                out = model(x)
            if out["logits"].shape != (2, 98) or out["embedding_raw"].shape != (2, 128) or out["embedding_normalized"].shape != (2, 128):
                raise RuntimeError(f"Output-contract failure: {name} L={length}")
            if not all(torch.isfinite(v).all() for v in out.values()):
                raise RuntimeError(f"Non-finite output: {name} L={length}")
    lo, hi = config["parameter_budget"]["minimum"], config["parameter_budget"]["maximum"]
    checks["parameter_budget"] = all(lo <= n <= hi for n in counts.values())
    checks["length_agnostic_output_contract"] = True

    if not all(checks.values()):
        failed = sorted(k for k,v in checks.items() if not v)
        raise RuntimeError(f"Phase 5 validation failed: {failed}")

    report = {
        "phase5_version":"1.0.0",
        "overall":"PASS",
        "checks":{k:"PASS" for k in sorted(checks)},
        "parameter_counts":counts,
        "phase4_freeze_sha256":sha256(FREEZE),
        "config_sha256":sha256(CONFIG),
        "claims_boundary":config["claims_boundary"]
    }
    out = HERE / "DELIVERY_VALIDATION.json"
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
