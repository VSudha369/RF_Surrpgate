# RF_Surrpgate — Surrogate-XAI V2

Reproducible implementation of the restructured RF Surrogate-XAI research pipeline.

## Execution contract

Google Drive is persistent storage only. Scientific hot I/O, DataLoader work, training, evaluation, XAI generation, and profiling run from the Colab VM local filesystem under `/content`.

Canonical data flow:

`Google Drive -> /content local SSD -> CPU/GPU computation -> local artifact -> Google Drive`

## Implemented phases

- `phase0/` — Colab hardware/runtime autotuning; canonical version `v1.0.1`, frozen.
- `phase1/` — dataset canonicalization/storage; RadioML canonicalization plus native WiSig ManyTx support through `v1.0.2`, frozen.
- `phase2/` — local stage manager for frozen Phase-1 transport artifacts; canonical version `v1.0.0`.

### Phase 0

Canonical T4 result is frozen as `GPU_AUTOTUNE_COMPLETE`: FP16 autocast + GradScaler, scientific evaluation FP32, `torch.compile` disabled.

### Phase 1 — frozen canonical datasets

| Dataset | Samples | Native L | Shards | Transport packs |
|---|---:|---:|---:|---:|
| WiSig ManyTx | 1,020,643 | 256 | 127 | 1 |
| RadioML 2016.10A | 220,000 | 128 | 4 | 1 |
| RadioML 2018.01A | 2,555,904 | 1024 | 157 | 10 |

Phase 1 preserves RF values and native sequence lengths. It standardizes storage only to float32 `[N,2,L]` and persists verified large TAR transport packs to Drive.

The native WiSig source is `ManyTx.pkl.zip`; the older derived `WiSig_tensors.h5` at length 128 is not the canonical source for this pipeline.

### Phase 2 — Local Stage Manager

Mount Drive from a Colab notebook cell, update the repository, then validate the frozen Phase-1 targets:

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

%cd /content/RF_Surrpgate
!git pull origin main
%cd /content/RF_Surrpgate/phase2

!python Phase2_Local_Stage_Manager_v1_0_0.py validate-drive --dataset all
```

Perform the first full real-artifact smoke test with RadioML 2016:

```python
!python Phase2_Local_Stage_Manager_v1_0_0.py stage-in \
    --dataset radioml2016 --verification full

!python Phase2_Local_Stage_Manager_v1_0_0.py resolve \
    --dataset radioml2016
```

Phase 2 freezes exact Phase-1 run IDs, source hashes, transport-pack names, byte sizes, and SHA-256 values. Stage-in copies each pack to `/content`, hashes it during copy, rejects unsafe or unexpected TAR members, extracts locally, verifies canonical shard hashes/layout/counts/global IDs, writes `LOCAL_STAGE_COMPLETE.json`, and exposes a stable local canonical root to downstream phases.

## Persistent paths

Drive artifacts:

`/content/drive/MyDrive/Surrogate_XAI_V2/`

Local hot workspace:

`/content/surrogate_xai_v2/`

Phase-2 staged datasets:

`/content/surrogate_xai_v2/staged_datasets/`

## Reproducibility

Datasets, generated shards, caches, checkpoints, Drive artifacts, and local runtime outputs are intentionally excluded from Git. Each scientific stage records configuration, provenance, hashes, validation results, and completion markers before downstream use.
