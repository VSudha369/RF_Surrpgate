#!/usr/bin/env python3
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any,Dict
import numpy as np
from split_common import atomic_json, sha256_file, utc_now

ALLOWED=('train_known','p0_known','p1_cross_day','p2_cross_receiver','p3_cross_day_receiver','calibration_unknown')
STRICT=('strict_zero_day','strict_zero_day_shift')

def _verify_npy(path:Path,count:int,expected_sha:str,load:bool):
    if not path.is_file(): raise FileNotFoundError(path)
    observed=sha256_file(path)
    if observed!=expected_sha: raise RuntimeError(f'WiSig frozen split SHA mismatch: {path.name}')
    info={'filename':path.name,'bytes':path.stat().st_size,'sha256':observed,'expected_count':count}
    if load:
        a=np.load(path,allow_pickle=False)
        if a.dtype!=np.int64 or a.ndim!=1 or len(a)!=count: raise RuntimeError(f'WiSig split shape/count mismatch: {path.name}')
        if len(np.unique(a))!=len(a): raise RuntimeError(f'Duplicate indices inside {path.name}')
        if len(a) and (a.min()<0 or a.max()>=1020643): raise RuntimeError(f'Out-of-range index in {path.name}')
        info['min']=int(a.min()) if len(a) else None; info['max']=int(a.max()) if len(a) else None
        return info,a
    return info,None

def build_wisig(registry:Dict[str,Any],out_root:Path):
    if registry['source_archive_sha256']!='a8fc3e35134a240bfb4dab8862a6e482cef44de000b813d42417b853c47ccc7e': raise RuntimeError('WiSig source bridge SHA changed')
    ids=registry['identity_sets']; sets={k:set(v) for k,v in ids.items()}
    if [len(sets['known']),len(sets['calibration_unknown']),len(sets['strict_zero_day'])]!=[98,22,30]: raise RuntimeError('WiSig identity cardinality mismatch')
    if sets['known']&sets['calibration_unknown'] or sets['known']&sets['strict_zero_day'] or sets['calibration_unknown']&sets['strict_zero_day']: raise RuntimeError('WiSig identity leakage')
    split_root=Path(registry['old_split_root']); out_root.mkdir(parents=True,exist_ok=True)
    arrays={}; file_info={}
    for role in ALLOWED:
        spec=registry['split_files'][role]; src=split_root/spec['filename']; info,a=_verify_npy(src,spec['count'],spec['sha256'],True)
        dst=out_root/f'{role}_indices.npy'; shutil.copyfile(src,dst)
        if sha256_file(dst)!=spec['sha256']: raise RuntimeError('WiSig allowed split copy hash mismatch')
        arrays[role]=a; file_info[role]={'file':dst.name,'count':spec['count'],'sha256':spec['sha256']}
    # Strict files are byte-hashed only; they are never np.load'ed or copied into normal Phase-3 output.
    strict_commit={}
    for role in STRICT:
        spec=registry['split_files'][role]; src=split_root/spec['filename']; info,_=_verify_npy(src,spec['count'],spec['sha256'],False)
        strict_commit[role]={'sample_count':spec['count'],'source_file_sha256':spec['sha256'],'index_file_exported':False,'access':'FINAL_EVALUATION_ONLY_AFTER_FINAL_SELECTION_LOCK'}
    # Allowed-role disjointness. P0/P1/P2/P3 are protocol-specific distinct sets in the old freeze.
    allowed_names=list(ALLOWED)
    for i in range(len(allowed_names)):
        for j in range(i+1,len(allowed_names)):
            if np.intersect1d(arrays[allowed_names[i]],arrays[allowed_names[j]],assume_unique=False).size: raise RuntimeError(f'WiSig allowed split overlap: {allowed_names[i]} vs {allowed_names[j]}')
    strict_commit['strict_zero_day']['transmitter_identities']=ids['strict_zero_day']
    strict_commit['strict_zero_day_shift']['transmitter_identities']=ids['strict_zero_day']
    strict_commit['identity_count']=30
    atomic_json(out_root/'STRICT_ROLE_COMMITMENT.json',strict_commit)
    atomic_json(out_root/'CLASS_ROLE_MANIFEST.json',{
      'dataset':'wisig','source_archive_sha256':registry['source_archive_sha256'],'partition_seed':registry['partition_seed'],
      'known_transmitter_identities':ids['known'],'calibration_unknown_transmitter_identities':ids['calibration_unknown'],'strict_zero_day_transmitter_identities':ids['strict_zero_day'],
      'heldout_day':registry['heldout_day'],'heldout_receiver':registry['heldout_receiver'],'session_id':'NOT_REPRESENTED'
    })
    atomic_json(out_root/'DATA_ACCESS_POLICY.json',{
      'train_known':'TRAINING_ALLOWED','p0_known':'MODEL_SELECTION_ALLOWED','p1_cross_day':'SHIFT_EVAL_ALLOWED','p2_cross_receiver':'SHIFT_EVAL_ALLOWED','p3_cross_day_receiver':'SHIFT_EVAL_ALLOWED',
      'calibration_unknown':'UNKNOWN_CALIBRATION_ONLY_NO_GRADIENTS','strict_zero_day':'FINAL_EVALUATION_ONLY_AFTER_FINAL_SELECTION_LOCK','strict_zero_day_shift':'FINAL_EVALUATION_ONLY_AFTER_FINAL_SELECTION_LOCK',
      'strict_arrays_loaded':False,'strict_index_files_exported':False,'unknown_gradients_forbidden':True
    })
    atomic_json(out_root/'PUBLIC_SPLIT_MANIFEST.json',{
      'dataset':'wisig','phase3_version':'1.0.0','source_archive_sha256':registry['source_archive_sha256'],'phase1_run_id':registry['new_phase1_run_id'],'seq_len':registry['new_phase1_seq_len'],
      'allowed_roles':file_info,'strict_commitment_file':'STRICT_ROLE_COMMITMENT.json','class_role_manifest':'CLASS_ROLE_MANIFEST.json','created_utc':utc_now()
    })
    validation={'verified':True,'dataset':'wisig','identity_counts':{'known':98,'calibration_unknown':22,'strict_zero_day':30},'allowed_counts':{r:registry['split_files'][r]['count'] for r in ALLOWED},'strict_counts':{r:registry['split_files'][r]['count'] for r in STRICT},'allowed_pairwise_disjoint':True,'strict_files_hash_only_not_loaded':True,'p2_missing_known_identities':registry['cross_checks']['p2_missing_known_identities'],'p3_missing_known_identities':registry['cross_checks']['p3_missing_known_identities']}
    atomic_json(out_root/'SPLIT_VALIDATION.json',validation)
    return validation
