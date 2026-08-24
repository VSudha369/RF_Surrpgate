#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import numpy as np

DRIVE_PROJECT=Path('/content/drive/MyDrive/Surrogate_XAI_V2')
LOCAL_PROJECT=Path('/content/surrogate_xai_v2')
PHASE3_DRIVE=DRIVE_PROJECT/'03_PHASE3_OPEN_SET_SPLITS'
PHASE3_LOCAL=LOCAL_PROJECT/'03_phase3_open_set_splits'


def utc_now(): return datetime.now(timezone.utc).isoformat()
def run_id(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

def sha256_file(path: Path, chunk=16*1024*1024):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

def sha256_json(obj: Any):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def sha256_int64_array(a: np.ndarray):
    x=np.asarray(a,dtype='<i8')
    return hashlib.sha256(x.tobytes(order='C')).hexdigest()

def atomic_json(path:Path,obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,sort_keys=True)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def atomic_npy(path:Path,a:np.ndarray):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(path)+'.tmp')
    with tmp.open('wb') as f:
        np.save(f,np.asarray(a,dtype=np.int64),allow_pickle=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def assert_local(path:Path):
    p=path.resolve()
    if str(p).startswith('/content/drive/') or not str(p).startswith('/content/'):
        raise RuntimeError(f'Expected local /content path, got {p}')

def require_drive():
    if not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError("Mount Drive in a Colab notebook cell before running Phase 3.")

def copy_tree_verified(src:Path,dst:Path):
    require_drive()
    if not src.is_dir(): raise FileNotFoundError(src)
    dst.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for p in sorted(src.rglob('*')):
        if not p.is_file(): continue
        rel=p.relative_to(src); q=dst/rel; q.parent.mkdir(parents=True,exist_ok=True)
        expected=sha256_file(p)
        tmp=Path(str(q)+'.partial'); tmp.unlink(missing_ok=True)
        shutil.copyfile(p,tmp)
        observed=sha256_file(tmp)
        if observed!=expected: raise RuntimeError(f'Stage-out hash mismatch: {rel}')
        os.replace(tmp,q)
        manifest.append({'path':str(rel),'bytes':q.stat().st_size,'sha256':expected})
    return manifest

def load_json(path:Path):
    return json.load(path.open('r',encoding='utf-8'))
