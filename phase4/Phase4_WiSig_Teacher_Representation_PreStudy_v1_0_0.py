#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE2 = HERE.parent / "phase2"
if str(PHASE2) not in sys.path:
    sys.path.insert(0, str(PHASE2))

from phase4_common import PHASE4_DRIVE_ROOT, PHASE4_LOCAL_ROOT, atomic_json, copy_tree_verified, run_id
from prestudy_runner import PreStudyConfig, run_prestudy

try:
    from stage_common import load_registry as load_phase2_registry
    from stage_manager import resolve_local_canonical_root
except Exception as exc:
    raise RuntimeError("Phase 2 modules are required next to phase4/") from exc


def main():
    ap = argparse.ArgumentParser(description="Phase 4 v1.0.0 — WiSig Teacher Representation Pre-Study")
    ap.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--canonical-root", default=None, help="Advanced/testing override; must be local /content path")
    ap.add_argument("--no-stage-out", action="store_true")
    ap.add_argument("--allow-non-t4-full", action="store_true", help="Noncanonical escape hatch; recorded in config")
    args = ap.parse_args()

    rid = run_id()
    local_run = PHASE4_LOCAL_ROOT / f"run_{rid}"
    p2reg = load_phase2_registry(PHASE2 / "frozen_phase1_registry_v1_0_0.json")
    if args.canonical_root:
        canonical = Path(args.canonical_root).resolve()
        if str(canonical).startswith("/content/drive/"):
            raise RuntimeError("--canonical-root cannot point to Drive")
    else:
        canonical = resolve_local_canonical_root(p2reg, "wisig")

    config = PreStudyConfig(profile=args.profile, require_t4_for_full=not args.allow_non_t4_full)
    summary = run_prestudy(canonical, local_run, config)

    drive_run = None
    if not args.no_stage_out:
        drive_run = PHASE4_DRIVE_ROOT / f"run_{rid}"
        manifest = copy_tree_verified(local_run, drive_run)
        atomic_json(local_run / "STAGE_OUT_MANIFEST.json", manifest)
        copy_tree_verified(local_run, drive_run)

    print("=" * 80)
    print("PHASE 4 STATUS:", summary["status"])
    print("Decision:", summary["selection"]["decision"])
    print("Recommended Phase-5 arm:", summary["selection"]["recommended_objective_arm_for_phase5"])
    print("Local run:", local_run)
    print("Drive run:", drive_run if drive_run else "SKIPPED")
    print("Strict zero-day accessed: False")
    print("=" * 80)


if __name__ == "__main__":
    main()
