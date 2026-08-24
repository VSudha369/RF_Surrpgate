#!/usr/bin/env python3
from __future__ import annotations
import hashlib, math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from split_common import atomic_json, atomic_npy, load_json, sha256_file, sha256_int64_array, sha256_json, utc_now

SEED=20260824

def _h(*parts):
    s='|'.join(map(str,parts)).encode()
    return int.from_bytes(hashlib.sha256(s).digest()[:8],'big')

def regenerate_class_roles(dataset:str,cfg:Dict[str,Any]):
    fams=cfg['families']; unknown=cfg['calibration_unknown_count']+cfg['strict_unknown_count']
    capacity={f:max(0,len(cs)-1) for f,cs in fams.items()}
    cap_total=sum(capacity.values())
    raw={f:(unknown*capacity[f]/cap_total if cap_total else 0.0) for f in fams}
    alloc={f:int(math.floor(raw[f])) for f in fams}
    remaining=unknown-sum(alloc.values())
    order=sorted(fams,key=lambda f:(-(raw[f]-alloc[f]),_h(SEED,dataset,'family_remainder',f)))
    for f in order:
        if remaining<=0: break
        if alloc[f]<capacity[f]: alloc[f]+=1; remaining-=1
    if remaining: raise RuntimeError('Could not allocate unknown class quota')
    selected=[]
    for fam,classes in fams.items():
        ranked=sorted(classes,key=lambda c:_h(SEED,dataset,fam,c))
        selected.extend((fam,c) for c in ranked[:alloc[fam]])
    selected=sorted(selected,key=lambda fc:_h(SEED,dataset,'unknown_role',fc[0],fc[1]))
    calib={c for _,c in selected[:cfg['calibration_unknown_count']]}
    strict={c for _,c in selected[cfg['calibration_unknown_count']:]}
    known=[c for c in cfg['classes'] if c not in calib|strict]
    return {
      'family_unknown_allocation':alloc,
      'known':known,
      'calibration_unknown':[c for c in cfg['classes'] if c in calib],
      'strict_unknown':[c for c in cfg['classes'] if c in strict],
    }

def _splitmix64(x):
    x=np.asarray(x,dtype=np.uint64)
    x=(x+np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z=x.copy(); z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9); z&=np.uint64(0xFFFFFFFFFFFFFFFF)
    z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB); z&=np.uint64(0xFFFFFFFFFFFFFFFF)
    return z^(z>>np.uint64(31))

def _stable_stratum_split(gids:np.ndarray,label:int,snr:int):
    gids=np.asarray(gids,dtype=np.int64)
    key=np.uint64((_h(SEED,'sample_split',label,snr)) & ((1<<64)-1))
    scores=_splitmix64(gids.astype(np.uint64)^key)
    order=np.argsort(scores,kind='stable'); n=len(gids)
    n_train=int(math.floor(0.70*n)); n_val=int(math.floor(0.15*n))
    return gids[order[:n_train]],gids[order[n_train:n_train+n_val]],gids[order[n_train+n_val:]]

def _read_metadata(canonical_root:Path):
    dm=load_json(canonical_root/'DATASET_MANIFEST.json')
    parts=[]
    for s in dm['shards']:
        with np.load(canonical_root/s['metadata_path'],allow_pickle=False) as z:
            if not {'global_index','label','snr'} <= set(z.files):
                raise RuntimeError('RadioML metadata requires global_index,label,snr')
            parts.append((z['global_index'].astype(np.int64),z['label'].astype(np.int32),z['snr'].astype(np.int16)))
    return dm, np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])

def build_radioml(dataset:str,canonical_root:Path,out_root:Path,catalog:Dict[str,Any],frozen_roles:Dict[str,Any]):
    dm,gid,label,snr=_read_metadata(canonical_root)
    if dm['n_samples']!=catalog['expected_n_samples'] or dm['seq_len']!=catalog['expected_seq_len'] or dm['source_sha256']!=catalog['source_sha256']:
        raise RuntimeError('Phase-1 provenance mismatch')
    if not np.array_equal(np.sort(gid),np.arange(catalog['expected_n_samples'],dtype=np.int64)):
        raise RuntimeError('Global index universe is not exactly 0..N-1')
    if sorted(np.unique(label).tolist())!=list(range(len(catalog['classes']))): raise RuntimeError('Label catalog mismatch')
    if sorted(np.unique(snr).tolist())!=catalog['snrs']: raise RuntimeError('SNR catalog mismatch')
    regenerated=regenerate_class_roles(dataset,catalog)
    if regenerated!=frozen_roles: raise RuntimeError('Frozen class-role manifest does not regenerate exactly')
    name_to_id={n:i for i,n in enumerate(catalog['classes'])}
    known_ids={name_to_id[n] for n in frozen_roles['known']}; calib_ids={name_to_id[n] for n in frozen_roles['calibration_unknown']}; strict_ids={name_to_id[n] for n in frozen_roles['strict_unknown']}
    train=[]; val=[]; test=[]
    expected_per=catalog['samples_per_class_snr']
    for lid in sorted(known_ids):
        for s in catalog['snrs']:
            ids=gid[(label==lid)&(snr==s)]
            if len(ids)!=expected_per: raise RuntimeError(f'Unexpected stratum count label={lid} snr={s}: {len(ids)}')
            a,b,c=_stable_stratum_split(ids,lid,s); train.append(a); val.append(b); test.append(c)
    train=np.sort(np.concatenate(train)); val=np.sort(np.concatenate(val)); test=np.sort(np.concatenate(test))
    calib=np.sort(gid[np.isin(label,list(calib_ids))]); strict=np.sort(gid[np.isin(label,list(strict_ids))])
    all_parts=[train,val,test,calib,strict]
    for i in range(len(all_parts)):
        for j in range(i+1,len(all_parts)):
            if np.intersect1d(all_parts[i],all_parts[j],assume_unique=True).size: raise RuntimeError('Split overlap detected')
    if sum(map(len,all_parts))!=catalog['expected_n_samples']: raise RuntimeError('Split universe does not cover dataset')
    out_root.mkdir(parents=True,exist_ok=True)
    files={}
    for role,a in [('known_train',train),('known_val',val),('known_test',test),('calibration_unknown',calib)]:
        p=out_root/f'{role}_indices.npy'; atomic_npy(p,a); files[role]={'file':p.name,'count':len(a),'sha256':sha256_file(p)}
    strict_commit={
      'role':'strict_unknown','class_names':frozen_roles['strict_unknown'],'class_ids':sorted(strict_ids),'sample_count':len(strict),
      'sorted_global_indices_raw_int64_sha256':sha256_int64_array(strict),
      'index_file_exported':False,
      'access':'FINAL_EVALUATION_ONLY_AFTER_FINAL_SELECTION_LOCK'
    }
    atomic_json(out_root/'STRICT_ROLE_COMMITMENT.json',strict_commit)
    rows=[]
    for i,n in enumerate(catalog['classes']):
        fam=next(f for f,cs in catalog['families'].items() if n in cs)
        role='known' if n in frozen_roles['known'] else ('calibration_unknown' if n in frozen_roles['calibration_unknown'] else 'strict_unknown')
        rows.append({'class_id':i,'class_name':n,'family':fam,'role':role})
    class_manifest={'dataset':dataset,'phase3_version':'1.0.0','seed':SEED,'algorithm':'family_capacity_largest_remainder_v1 + sha256 ranking','rows':rows,'manifest_sha256':None}
    class_manifest['manifest_sha256']=sha256_json({k:v for k,v in class_manifest.items() if k!='manifest_sha256'})
    atomic_json(out_root/'CLASS_ROLE_MANIFEST.json',class_manifest)
    policy={
      'known_train':'TRAINING_ALLOWED','known_val':'MODEL_SELECTION_ALLOWED','known_test':'FINAL_KNOWN_EVALUATION_ONLY',
      'calibration_unknown':'THRESHOLD_AND_UNKNOWN_CALIBRATION_ONLY_NO_GRADIENTS','strict_unknown':'FINAL_EVALUATION_ONLY_AFTER_FINAL_SELECTION_LOCK',
      'strict_index_file_present':False,'unknown_gradients_forbidden':True
    }
    atomic_json(out_root/'DATA_ACCESS_POLICY.json',policy)
    public={'dataset':dataset,'phase3_version':'1.0.0','phase1_source_sha256':catalog['source_sha256'],'n_samples':catalog['expected_n_samples'],'seq_len':catalog['expected_seq_len'],'roles':files,'strict_commitment_file':'STRICT_ROLE_COMMITMENT.json','class_role_manifest':'CLASS_ROLE_MANIFEST.json','created_utc':utc_now()}
    atomic_json(out_root/'PUBLIC_SPLIT_MANIFEST.json',public)
    validation={'verified':True,'dataset':dataset,'counts':{k:len(v) for k,v in [('known_train',train),('known_val',val),('known_test',test),('calibration_unknown',calib),('strict_unknown',strict)]},'universe_count':sum(map(len,all_parts)),'pairwise_disjoint':True,'class_roles_regenerate_exactly':True,'stratification':'class_x_snr','strict_index_exported':False}
    atomic_json(out_root/'SPLIT_VALIDATION.json',validation)
    return validation
