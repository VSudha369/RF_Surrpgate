#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE2 = HERE.parent / "phase2"
if str(PHASE2) not in sys.path:
    sys.path.insert(0, str(PHASE2))

from phase4_common import atomic_json, utc_now
from phase4_benchmark_runtime_v1_1_0 import BenchmarkConfig, persist_file_verified
from phase4_benchmark_runner_v1_1_0 import run_benchmark

try:
    from stage_common import load_registry as load_phase2_registry
    from stage_manager import resolve_local_canonical_root
except Exception as exc:
    raise RuntimeError("Phase 2 modules are required next to phase4/") from exc


LOCAL_ROOT = Path("/content/surrogate_xai_v2/04_phase4_wisig_teacher_representation_benchmark")
DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Surrogate_XAI_V2/"
    "04_PHASE4_WISIG_TEACHER_REPRESENTATION_BENCHMARK"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 v1.1.0 — short, leakage-free WiSig representation benchmark"
    )
    parser.add_argument(
        "--run-id",
        default="benchmark_v1_1_0",
        help="Stable identifier used for interruption-safe resume across Colab sessions",
    )
    parser.add_argument(
        "--canonical-root",
        default=None,
        help="Advanced testing override; must be a local /content path",
    )
    parser.add_argument(
        "--allow-non-t4",
        action="store_true",
        help="Noncanonical escape hatch; hardware remains recorded",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", args.run_id):
        raise ValueError("--run-id may contain only letters, numbers, dot, underscore, and hyphen")

    local_run = LOCAL_ROOT / f"run_{args.run_id}"
    drive_run = DRIVE_ROOT / f"run_{args.run_id}"
    local_run.mkdir(parents=True, exist_ok=True)
    drive_run.mkdir(parents=True, exist_ok=True)
    config = BenchmarkConfig(require_t4=not args.allow_non_t4)

    registry = load_phase2_registry(PHASE2 / "frozen_phase1_registry_v1_0_0.json")
    if args.canonical_root:
        canonical_root = Path(args.canonical_root).resolve()
        if str(canonical_root).startswith("/content/drive/"):
            raise RuntimeError("--canonical-root cannot point to Drive")
        if not str(canonical_root).startswith("/content/"):
            raise RuntimeError("--canonical-root must be under /content")
    else:
        canonical_root = resolve_local_canonical_root(registry, "wisig")

    try:
        summary = run_benchmark(canonical_root, local_run, drive_run, config)
    except Exception as exc:
        failure = {
            "status": "PHASE4_BENCHMARK_FAILED",
            "failed_utc": utc_now(),
            "run_id": args.run_id,
            "config_sha256": config.digest(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "recovery_instruction": (
                "Rerun the same command and --run-id. The latest fully completed epoch "
                "will be restored from Drive after SHA verification."
            ),
        }
        local_failure = local_run / "PHASE4_FAILURE.json"
        atomic_json(local_failure, failure)
        try:
            persist_file_verified(local_failure, drive_run / "PHASE4_FAILURE.json")
        except Exception:
            pass
        raise

    print("=" * 88)
    print("PHASE 4 v1.1.0 STATUS:", summary["status"])
    print("Selected Phase-5 objective arm:", summary["selected_arm_for_phase5"])
    print("Selection protocol: P0 Known only")
    print("Strict Zero-Day accessed: False")
    print("Maximum epochs: 8 | Train samples: 58,800 | P0 samples: 14,700")
    print("Local run:", local_run)
    print("Drive run:", drive_run)
    print("=" * 88)


if __name__ == "__main__":
    main()
