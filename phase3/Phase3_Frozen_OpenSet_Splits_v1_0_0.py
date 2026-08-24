#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PHASE2=HERE.parent/'phase2'
if str(PHASE2) not in sys.path: sys.path.insert(0,str(PHASE2))
from split_common import PHASE3_DRIVE,PHASE3_LOCAL,atomic_json,copy_tree_verified,load_json,run_id,utc_now
from radioml_open_set import build_radioml
from wisig_protocol_bridge import build_wisig
try:
    from stage_common import load_registry as load_phase2_registry
    from stage_manager import resolve_local_canonical_root
except Exception:
    load_phase2_registry=None; resolve_local_canonical_root=None

def main():
    ap=argparse.ArgumentParser(description='Phase 3 v1.0.0 — Frozen Deterministic Open-Set Split Construction')
    ap.add_argument('--dataset',choices=['radioml2016','radioml2018','wisig','all'],required=True)
    ap.add_argument('--canonical-root',default=None,help='Testing/advanced override for one RadioML dataset only')
    ap.add_argument('--no-stage-out',action='store_true')
    args=ap.parse_args()
    rid=run_id(); local_run=PHASE3_LOCAL/f'run_{rid}'; local_run.mkdir(parents=True,exist_ok=True)
    catalogs=load_json(HERE/'class_catalogs_v1_0_0.json')['datasets']
    frozen=load_json(HERE/'frozen_class_roles_v1_0_0.json')['datasets']
    results={}
    targets=['radioml2016','radioml2018','wisig'] if args.dataset=='all' else [args.dataset]
    p2reg=None
    if any(d.startswith('radioml') for d in targets):
        if load_phase2_registry is None: raise RuntimeError('Phase 2 modules are required next to phase3/')
        p2reg=load_phase2_registry(PHASE2/'frozen_phase1_registry_v1_0_0.json')
    for ds in targets:
        print(f'[RUN] Phase 3 split construction: {ds}')
        out=local_run/ds
        if ds.startswith('radioml'):
            if args.canonical_root:
                if len(targets)!=1: raise RuntimeError('--canonical-root only supports a single RadioML target')
                root=Path(args.canonical_root)
            else:
                root=resolve_local_canonical_root(p2reg,ds)
            results[ds]=build_radioml(ds,root,out,catalogs[ds],frozen[ds])
        else:
            wreg=load_json(HERE/'wisig_frozen_protocol_registry_v1_0_0.json')
            results[ds]=build_wisig(wreg,out)
        print(f"[OK] {ds}: {results[ds]}")
    summary={'phase':'PHASE3_FROZEN_OPEN_SET_SPLITS','phase_version':'1.0.0','run_id':rid,'datasets':results,'completed_utc':utc_now()}
    atomic_json(local_run/'PHASE3_SUMMARY.json',summary)
    atomic_json(local_run/'PHASE3_COMPLETE.json',{'status':'PHASE3_COMPLETE','run_id':rid,'datasets':targets})
    drive_run=None
    if not args.no_stage_out:
        drive_run=PHASE3_DRIVE/f'run_{rid}'
        manifest=copy_tree_verified(local_run,drive_run)
        atomic_json(local_run/'STAGE_OUT_MANIFEST.json',manifest)
        # copy stage-out manifest after generation
        copy_tree_verified(local_run,drive_run)
    print('='*80); print('PHASE 3 STATUS: PHASE3_COMPLETE'); print('Local run:',local_run); print('Drive run:',drive_run if drive_run else 'SKIPPED'); print('='*80)
if __name__=='__main__': main()
