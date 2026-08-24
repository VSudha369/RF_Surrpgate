#!/usr/bin/env python3
"""Phase 0 — Colab runtime autotune v1.0.1 for Surrogate-XAI V2.

Contract: Drive is persistence only. All benchmarks/hot I/O run from /content.
T4/Turing is forced to FP16 autocast + GradScaler; native BF16 is used only
for Ampere-or-newer devices with reported BF16 support.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PHASE = "PHASE0_RUNTIME_AUTOTUNE"
VERSION = "1.0.1"
PROJECT = "Surrogate_XAI_V2"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DRIVE_MOUNT = Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT / "MyDrive" / PROJECT
DRIVE_PHASE = DRIVE_ROOT / "00_PHASE0_RUNTIME_AUTOTUNE"
LOCAL_ROOT = Path("/content") / PROJECT.lower()
RUN_ROOT = LOCAL_ROOT / "00_phase0_runtime_autotune" / f"run_{RUN_ID}"
OUT = RUN_ROOT / "outputs"
BENCH = RUN_ROOT / "bench"
TMP = RUN_ROOT / "tmp"

DISK_MIB = 256
TARGET_MEM_FRAC = 0.88
GPU_WARMUP = 20
GPU_ITERS = 60


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def run_cmd(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as e:
        return {"returncode": None, "stdout": "", "stderr": repr(e)}


def assert_local(path: Path):
    p = str(path.resolve())
    if p.startswith(str(DRIVE_MOUNT.resolve())) or not p.startswith("/content/"):
        raise RuntimeError(f"Hot scientific path must be local /content, got: {p}")


def init_dirs():
    for p in (RUN_ROOT, OUT, BENCH, TMP):
        p.mkdir(parents=True, exist_ok=True)
        assert_local(p)


def mount_drive():
    from google.colab import drive
    if not DRIVE_MOUNT.exists() or not any(DRIVE_MOUNT.iterdir()):
        drive.mount(str(DRIVE_MOUNT), force_remount=False)


def probe():
    import psutil
    import torch
    du = shutil.disk_usage("/content")
    vm = psutil.virtual_memory()
    d = {
        "phase": PHASE, "version": VERSION, "run_id": RUN_ID, "utc": now(),
        "python": sys.version, "platform": platform.platform(),
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_total_bytes": int(vm.total), "ram_available_bytes": int(vm.available),
        "local_disk_total_bytes": int(du.total), "local_disk_free_bytes": int(du.free),
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "torch_compile_available": hasattr(torch, "compile"),
        "nvidia_smi": run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap", "--format=csv,noheader,nounits"]),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d["gpu"] = {
            "name": torch.cuda.get_device_name(0), "total_memory_bytes": int(p.total_memory),
            "multi_processor_count": int(p.multi_processor_count), "major": int(p.major),
            "minor": int(p.minor), "compute_capability": f"{p.major}.{p.minor}",
            "bf16_reported": bool(torch.cuda.is_bf16_supported()),
        }
    else:
        d["gpu"] = None
    return d


def precision_policy(profile):
    if not profile["cuda_available"]:
        return {"device": "cpu", "train_autocast_dtype": "float32", "use_grad_scaler": False,
                "scientific_eval_dtype": "float32", "reason": "CUDA unavailable"}
    g = profile["gpu"] or {}; name = g.get("name", "").lower(); major = int(g.get("major", 0)); minor = int(g.get("minor", 0))
    if "t4" in name or (major == 7 and minor == 5):
        return {"device": "cuda", "train_autocast_dtype": "float16", "use_grad_scaler": True,
                "scientific_eval_dtype": "float32", "reason": "T4/Turing CC7.5: FP16 + GradScaler"}
    if major >= 8 and g.get("bf16_reported", False):
        return {"device": "cuda", "train_autocast_dtype": "bfloat16", "use_grad_scaler": False,
                "scientific_eval_dtype": "float32", "reason": "Ampere-or-newer native-BF16 policy"}
    return {"device": "cuda", "train_autocast_dtype": "float16", "use_grad_scaler": True,
            "scientific_eval_dtype": "float32", "reason": "Conservative CUDA fallback: FP16 + GradScaler"}


def autocast_ctx(policy):
    import torch
    from contextlib import nullcontext
    if policy["device"] != "cuda": return nullcontext()
    dt = torch.bfloat16 if policy["train_autocast_dtype"] == "bfloat16" else torch.float16
    return torch.autocast("cuda", dtype=dt)


def scaler_for(policy):
    import torch
    enabled = policy["device"] == "cuda" and policy["use_grad_scaler"]
    try: return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception: return torch.cuda.amp.GradScaler(enabled=enabled)


def local_disk_bench():
    path = BENCH / "disk.bin"; assert_local(path)
    total = DISK_MIB * 1024 * 1024; block = os.urandom(8 * 1024 * 1024)
    t = time.perf_counter()
    with path.open("wb", buffering=0) as f:
        n = 0
        while n < total:
            b = block[:min(len(block), total - n)]; f.write(b); n += len(b)
        f.flush(); os.fsync(f.fileno())
    ws = time.perf_counter() - t
    t = time.perf_counter(); n = 0
    with path.open("rb", buffering=0) as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): n += len(b)
    rs = time.perf_counter() - t
    path.unlink(missing_ok=True)
    return {"bytes": n, "write_MiB_s": DISK_MIB / ws, "read_MiB_s": DISK_MIB / rs,
            "note": "local /content; read may benefit from OS page cache"}


def h2d_bench(profile):
    import torch
    if not profile["cuda_available"]: return {"skipped": True, "reason": "CUDA unavailable"}
    rows = []
    for mib in (16, 64, 256):
        x = torch.empty((mib * 1024 * 1024) // 4, dtype=torch.float32, pin_memory=True)
        for _ in range(4): y = x.to("cuda", non_blocking=True)
        torch.cuda.synchronize(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record()
        for _ in range(20): y = x.to("cuda", non_blocking=True)
        b.record(); torch.cuda.synchronize(); ms = a.elapsed_time(b)
        rows.append({"MiB": mib, "throughput_GiB_s": (mib * 20 / 1024) / (ms / 1000)})
        del x, y
    return {"skipped": False, "results": rows}


def proxy_model():
    import torch.nn as nn
    class DS(nn.Module):
        def __init__(self, ci, co, s=1):
            super().__init__(); self.m = nn.Sequential(nn.Conv1d(ci, ci, 5, s, 2, groups=ci, bias=False), nn.Conv1d(ci, co, 1, bias=False), nn.BatchNorm1d(co), nn.SiLU())
        def forward(self, x): return self.m(x)
    return nn.Sequential(nn.Conv1d(2,32,7,2,3,bias=False), nn.BatchNorm1d(32), nn.SiLU(), DS(32,64,2), DS(64,96,2), DS(96,128,2), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128,24))


def train_step(model, opt, crit, x, y, policy, scaler):
    opt.zero_grad(set_to_none=True)
    with autocast_ctx(policy): out = model(x); loss = crit(out.float(), y)
    if scaler.is_enabled(): scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    else: loss.backward(); opt.step()


def gpu_proxy_bench(profile, policy):
    import torch
    if not profile["cuda_available"]: return {"skipped": True, "reason": "CUDA unavailable"}
    model = proxy_model().cuda().train(); opt = torch.optim.AdamW(model.parameters(), 1e-3); crit = torch.nn.CrossEntropyLoss(); scaler = scaler_for(policy)
    total = torch.cuda.get_device_properties(0).total_memory; target = int(total * TARGET_MEM_FRAC); trials=[]
    for bs in (32,64,128,256,512,768,1024,1536,2048,3072):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            x=torch.randn(bs,2,1024,device="cuda"); y=torch.randint(0,24,(bs,),device="cuda")
            train_step(model,opt,crit,x,y,policy,scaler); torch.cuda.synchronize(); peak=torch.cuda.max_memory_allocated(); trials.append({"batch":bs,"peak_bytes":int(peak)})
            del x,y
            if peak >= target: break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); break
    under=[r for r in trials if r["peak_bytes"]<=target]; chosen=(under[-1] if under else trials[0])["batch"]
    x=torch.randn(chosen,2,1024,device="cuda"); y=torch.randint(0,24,(chosen,),device="cuda")
    for _ in range(GPU_WARMUP): train_step(model,opt,crit,x,y,policy,scaler)
    torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(GPU_ITERS): train_step(model,opt,crit,x,y,policy,scaler)
    b.record(); torch.cuda.synchronize(); ms=a.elapsed_time(b)
    return {"skipped":False,"batch_trials":trials,"recommended_proxy_train_batch_size":chosen,
            "samples_per_second":chosen*GPU_ITERS/(ms/1000),"precision":policy,
            "note":"proxy only; real teacher/surrogate must re-autotune"}


class MemmapIQ:
    def __init__(self, path, n, L=1024): self.path=str(path); self.n=n; self.L=L; self.mm=None
    def __len__(self): return self.n
    def __getitem__(self, i):
        import numpy as np, torch
        if self.mm is None: self.mm=np.memmap(self.path,dtype=np.float32,mode="r",shape=(self.n,2,self.L))
        return torch.from_numpy(np.array(self.mm[i],copy=True)), 0


def dataloader_gpu_bench(profile, policy, gpu):
    import numpy as np, torch
    from torch.utils.data import DataLoader
    path=BENCH/"loader.f32"; L=1024; n=max(8192,(256*1024*1024)//(2*L*4)); rng=np.random.default_rng(12345)
    mm=np.memmap(path,dtype=np.float32,mode="w+",shape=(n,2,L))
    for s in range(0,n,2048): mm[s:min(s+2048,n)]=rng.standard_normal((min(s+2048,n)-s,2,L),dtype=np.float32)
    mm.flush(); del mm
    use_cuda=profile["cuda_available"]; device="cuda" if use_cuda else "cpu"; cpu=int(profile.get("cpu_logical") or 2)
    candidates=[w for w in (0,1,2,4,6,8) if w<=cpu]; bs=max(64,min(int(gpu.get("recommended_proxy_train_batch_size",256) or 256),512))
    model=proxy_model().to(device).train(); opt=torch.optim.AdamW(model.parameters(),1e-3); crit=torch.nn.CrossEntropyLoss(); scaler=scaler_for(policy); ds=MemmapIQ(path,n,L); rows=[]
    for w in candidates:
        for pf in ([None] if w==0 else [2,4]):
            kw=dict(dataset=ds,batch_size=bs,shuffle=False,num_workers=w,pin_memory=use_cuda,persistent_workers=w>0,drop_last=True)
            if w>0: kw["prefetch_factor"]=pf
            loader=DataLoader(**kw); it=iter(loader)
            for _ in range(6):
                try: xc,_=next(it)
                except StopIteration: it=iter(loader); xc,_=next(it)
                yc=torch.zeros(xc.shape[0],dtype=torch.long); x=xc.to(device,non_blocking=use_cuda); y=yc.to(device,non_blocking=use_cuda); train_step(model,opt,crit,x,y,policy,scaler)
            if use_cuda: torch.cuda.synchronize()
            wait=0.0; samples=0; t0=time.perf_counter()
            for _ in range(30):
                tf=time.perf_counter()
                try: xc,_=next(it)
                except StopIteration: it=iter(loader); xc,_=next(it)
                wait += time.perf_counter()-tf; yc=torch.zeros(xc.shape[0],dtype=torch.long); x=xc.to(device,non_blocking=use_cuda); y=yc.to(device,non_blocking=use_cuda); train_step(model,opt,crit,x,y,policy,scaler); samples += xc.shape[0]
            if use_cuda: torch.cuda.synchronize()
            elapsed=time.perf_counter()-t0; rows.append({"num_workers":w,"prefetch_factor":pf,"samples_per_second":samples/elapsed,"data_wait_fraction":wait/elapsed})
            del loader,it; gc.collect()
    path.unlink(missing_ok=True); mx=max(r["samples_per_second"] for r in rows); near=[r for r in rows if r["samples_per_second"]>=0.98*mx]; near.sort(key=lambda r:(r["num_workers"],999 if r["prefetch_factor"] is None else r["prefetch_factor"])); best=near[0]
    return {"gpu_fed":use_cuda,"trials":rows,"selection_rule":"smallest worker count within 98% of max throughput",
            "recommended_num_workers":best["num_workers"],"recommended_prefetch_factor":best["prefetch_factor"],
            "recommended_data_wait_fraction":best["data_wait_fraction"],"max_samples_per_second":mx}


def compile_bench(profile, policy, batch):
    import torch
    if not profile["cuda_available"] or not hasattr(torch,"compile"): return {"skipped":True,"recommend_torch_compile":False}
    bs=max(16,min(int(batch),512))
    def bench(model):
        model=model.cuda().train(); opt=torch.optim.AdamW(model.parameters(),1e-3); crit=torch.nn.CrossEntropyLoss(); scaler=scaler_for(policy); x=torch.randn(bs,2,1024,device="cuda"); y=torch.randint(0,24,(bs,),device="cuda")
        for _ in range(8): train_step(model,opt,crit,x,y,policy,scaler)
        torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
        for _ in range(30): train_step(model,opt,crit,x,y,policy,scaler)
        b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)
    try:
        eager=bench(proxy_model()); gc.collect(); torch.cuda.empty_cache(); compiled=bench(torch.compile(proxy_model())); gain=(eager-compiled)/eager
        return {"skipped":False,"eager_ms":eager,"compiled_ms":compiled,"steady_state_gain_fraction":gain,"recommend_torch_compile":gain>=0.05,"decision_rule":"enable only if >=5% steady-state gain"}
    except Exception as e:
        return {"skipped":True,"recommend_torch_compile":False,"reason":repr(e)}


def hash_manifest():
    return {str(p.relative_to(RUN_ROOT)):sha256(p) for p in sorted(RUN_ROOT.rglob("*")) if p.is_file() and p.name!="HASH_MANIFEST.json"}


def stage_out():
    mount_drive(); DRIVE_PHASE.mkdir(parents=True,exist_ok=True); dest=DRIVE_PHASE/f"run_{RUN_ID}"; dest.mkdir(parents=True,exist_ok=True)
    base=TMP/f"{PHASE}_{VERSION}_{RUN_ID}"; archive=Path(shutil.make_archive(str(base),"gztar",root_dir=RUN_ROOT)); local_sha=sha256(archive); partial=dest/(archive.name+".partial"); final=dest/archive.name
    shutil.copy2(archive,partial); copied=sha256(partial)
    if copied!=local_sha: raise RuntimeError("Drive stage-out SHA mismatch")
    os.replace(partial,final); done={"archive":final.name,"archive_sha256":local_sha,"drive_run_root":str(dest),"completed_utc":now()}; write_json(dest/"STAGE_COMPLETE.json",done); write_json(DRIVE_ROOT/"LATEST_PHASE0.json",{"run_id":RUN_ID,**done}); return done


def main():
    init_dirs(); print("="*80); print(f"{PHASE} v{VERSION}\nRun ID: {RUN_ID}\nLocal root: {RUN_ROOT}\nDrive project: {DRIVE_ROOT}"); print("="*80)
    status={"phase":PHASE,"version":VERSION,"run_id":RUN_ID,"started_utc":now(),"status":"RUNNING"}; write_json(OUT/"PHASE0_FINAL_STATUS.json",status)
    try:
        profile=probe(); write_json(OUT/"RUNTIME_PROFILE.json",profile); print("[OK] Runtime probe:",profile.get("gpu"))
        policy=precision_policy(profile); write_json(OUT/"PRECISION_POLICY.json",policy); print("[OK] Precision:",policy)
        disk=local_disk_bench(); write_json(OUT/"LOCAL_DISK_BENCHMARK.json",disk); print(f"[OK] SSD read={disk['read_MiB_s']:.1f} write={disk['write_MiB_s']:.1f} MiB/s")
        h2d=h2d_bench(profile); write_json(OUT/"H2D_BENCHMARK.json",h2d)
        gpu=gpu_proxy_bench(profile,policy); write_json(OUT/"GPU_PROXY_AUTOTUNE.json",gpu); print("[OK] Proxy GPU:",gpu.get("recommended_proxy_train_batch_size"),gpu.get("samples_per_second"))
        loader=dataloader_gpu_bench(profile,policy,gpu); write_json(OUT/"DATALOADER_AUTOTUNE.json",loader); print("[OK] DataLoader:",loader["recommended_num_workers"],loader["recommended_prefetch_factor"],"wait",loader["recommended_data_wait_fraction"])
        comp=compile_bench(profile,policy,gpu.get("recommended_proxy_train_batch_size",64)); write_json(OUT/"TORCH_COMPILE_BENCHMARK.json",comp); print("[OK] torch.compile:",comp.get("recommend_torch_compile",False))
        canonical="GPU_AUTOTUNE_COMPLETE" if profile["cuda_available"] else "CPU_PREFLIGHT_COMPLETE"
        rec={"phase":PHASE,"version":VERSION,"run_id":RUN_ID,"precision":policy,
             "dataloader":{"num_workers":loader["recommended_num_workers"],"prefetch_factor":loader["recommended_prefetch_factor"],"pin_memory":profile["cuda_available"],"persistent_workers":loader["recommended_num_workers"]>0,"non_blocking_h2d":profile["cuda_available"],"proxy_data_wait_fraction":loader["recommended_data_wait_fraction"]},
             "gpu_proxy":{"provisional_train_batch_size_L1024":gpu.get("recommended_proxy_train_batch_size"),"must_reautotune_per_real_model":True},
             "torch_compile":{"provisional_recommendation":comp.get("recommend_torch_compile",False),"must_rebenchmark_per_real_model":True},
             "storage":{"local_disk_read_MiB_s":disk["read_MiB_s"],"local_disk_write_MiB_s":disk["write_MiB_s"],"recommended_logical_shard_target_MiB":64,"recommended_drive_transport_pack_GiB":2},
             "execution_contract":{"drive_role":"persistence only","hot_io_root":str(LOCAL_ROOT),"forbid_drive_in_hot_loop":True},
             "phase1_gpu_stage_gate":{"requires_status":"GPU_AUTOTUNE_COMPLETE","cuda_required":True},
             "later_reautotunes":["real teacher batch","real surrogate batch","evaluation batch","IG sample x alpha microbatch","real-model torch.compile"]}
        write_json(OUT/"AUTOTUNE_RECOMMENDATIONS.json",rec); status.update({"status":canonical,"completed_local_utc":now()}); write_json(OUT/"PHASE0_FINAL_STATUS.json",status); write_json(OUT/"HASH_MANIFEST.json",hash_manifest())
        done=stage_out(); status.update({"status":canonical,"completed_utc":now(),**done}); write_json(OUT/"PHASE0_FINAL_STATUS.json",status)
        print("="*80); print("PHASE 0 STATUS:",canonical); print("Drive run:",done["drive_run_root"]); print("="*80)
    except Exception as e:
        status.update({"status":"FAILED","failed_utc":now(),"error_type":type(e).__name__,"error":str(e)}); write_json(OUT/"PHASE0_FINAL_STATUS.json",status); raise


if __name__ == "__main__":
    main()
