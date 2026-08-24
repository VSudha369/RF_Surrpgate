#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from teacher_architectures import ARCHITECTURE_REGISTRY, build_teacher, parameter_count

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 v1.0.0 — WiSig Known-only teacher architecture screening")
    parser.add_argument("--preflight-only", action="store_true", help="Validate frozen Phase-4 lock and architecture registry without accessing RF data")
    args = parser.parse_args()

    freeze = json.loads((HERE.parent / "phase4" / "PHASE4_V1_1_0_FREEZE.json").read_text())
    config = json.loads((HERE / "phase5_config_v1_0_0.json").read_text())
    if freeze["freeze_status"] != "PHASE4_V1_1_0_FROZEN":
        raise RuntimeError("Phase 4 is not frozen")
    if freeze["selection_lock"]["selected_arm"] != "A0" or freeze["selection_lock"]["selected_objective"] != "Cross-Entropy":
        raise RuntimeError("Phase 5 requires frozen A0 Cross-Entropy objective")

    print("Phase 4 objective lock: A0 = Cross-Entropy")
    print("Selection data: P0 Known only")
    print("Calibration Unknown authorized: False")
    print("Strict Zero-Day authorized: False")
    print("Teacher candidates:")
    for name in config["architectures"]:
        print(f"  {name}: {parameter_count(build_teacher(name)):,} parameters")

    if not args.preflight_only:
        raise RuntimeError(
            "Phase 5 training runner is intentionally not enabled in the starter commit. "
            "Run the preflight validator first; the next implementation step is the resumable matched Colab screening runner."
        )
    print("PHASE5_SCREENING_PREFLIGHT_READY")


if __name__ == "__main__":
    main()
