#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage_common import load_registry
from stage_manager import (
    clean_staged_dataset,
    resolve_local_canonical_root,
    stage_in_dataset,
    validate_staged_dataset,
)
from stage_validation import validate_drive_target

HERE = Path(__file__).resolve().parent
DEFAULT_REGISTRY = HERE / "frozen_phase1_registry_v1_0_0.json"


def datasets_from_arg(reg, name):
    return list(reg["datasets"]) if name == "all" else [name]


def main():
    ap = argparse.ArgumentParser(
        description="Phase 2 v1.0.0 — frozen Phase-1 local stage manager"
    )
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list")
    p = sub.add_parser("validate-drive")
    p.add_argument("--dataset", choices=["all","wisig","radioml2016","radioml2018"], default="all")
    p.add_argument("--hash-packs", action="store_true")

    p = sub.add_parser("stage-in")
    p.add_argument("--dataset", choices=["all","wisig","radioml2016","radioml2018"], required=True)
    p.add_argument("--verification", choices=["full","manifest"], default="full")
    p.add_argument("--keep-tars", action="store_true")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("validate-local")
    p.add_argument("--dataset", choices=["all","wisig","radioml2016","radioml2018"], required=True)
    p.add_argument("--verification", choices=["full","manifest"], default="manifest")

    p = sub.add_parser("resolve")
    p.add_argument("--dataset", choices=["wisig","radioml2016","radioml2018"], required=True)

    p = sub.add_parser("clean")
    p.add_argument("--dataset", choices=["all","wisig","radioml2016","radioml2018"], required=True)
    p.add_argument("--all-runs", action="store_true")

    args = ap.parse_args()
    reg = load_registry(Path(args.registry))

    if args.command == "list":
        print(json.dumps(reg, indent=2))
        return

    if args.command == "validate-drive":
        out = {}
        for ds in datasets_from_arg(reg, args.dataset):
            print(f"[VALIDATE-DRIVE] {ds}...")
            out[ds] = validate_drive_target(reg["datasets"][ds], hash_packs=args.hash_packs)
            print(
                f"[OK] {ds}: N={out[ds]['n_samples']:,}, "
                f"L={out[ds]['seq_len']}, packs={out[ds]['n_transport_packs']}"
            )
        print(json.dumps(out, indent=2))
        return

    if args.command == "stage-in":
        out = {}
        for ds in datasets_from_arg(reg, args.dataset):
            out[ds] = stage_in_dataset(
                reg, ds,
                verification=args.verification,
                keep_tars=args.keep_tars,
                force=args.force,
            )
            print(
                f"[OK] {ds}: {out[ds]['status']} -> "
                f"{out[ds]['local_canonical_root']}"
            )
        print(json.dumps(out, indent=2))
        return

    if args.command == "validate-local":
        out = {}
        for ds in datasets_from_arg(reg, args.dataset):
            out[ds] = validate_staged_dataset(
                reg, ds, verification=args.verification
            )
            print(f"[OK] {ds}: LOCAL_STAGE_VALID")
        print(json.dumps(out, indent=2))
        return

    if args.command == "resolve":
        p = resolve_local_canonical_root(reg, args.dataset)
        print(str(p))
        return

    if args.command == "clean":
        out = {}
        for ds in datasets_from_arg(reg, args.dataset):
            out[ds] = clean_staged_dataset(reg, ds, include_all_runs=args.all_runs)
            print(f"[OK] cleaned {ds}")
        print(json.dumps(out, indent=2))
        return


if __name__ == "__main__":
    main()
