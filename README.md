# RF_Surrpgate — Surrogate-XAI V2

Reproducible implementation of the restructured RF Surrogate-XAI research pipeline.

## Execution contract

Google Drive is persistent storage only. Scientific hot I/O, DataLoader work, training, evaluation, XAI generation, and profiling run from the Colab VM local filesystem under `/content`.

Canonical data flow:

`Google Drive -> /content local SSD -> CPU/GPU computation -> local artifact -> Google Drive`

## Implemented phases

- `phase0/` — Google Colab hardware/runtime autotuning, canonical version `v1.0.1`.
- `phase1/` — dataset canonicalization and storage pipeline, canonical version `v1.0.0`.

### Phase 0

Run on a Colab GPU runtime:

```python
!rm -rf /content/RF_Surrpgate
!git clone https://github.com/VSudha369/RF_Surrpgate.git /content/RF_Surrpgate
%cd /content/RF_Surrpgate/phase0
%run Phase0_Colab_Runtime_Autotune_v1_0_1.py
```

Expected T4 gate:

```text
GPU_AUTOTUNE_COMPLETE
float16
GradScaler=True
```

### Phase 1

Recommended runtime: CPU / High-RAM. GPU is not required.

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

!rm -rf /content/RF_Surrpgate
!git clone https://github.com/VSudha369/RF_Surrpgate.git /content/RF_Surrpgate
%cd /content/RF_Surrpgate/phase1

!python Phase1_Dataset_Canonicalization_v1_0_0.py --discover
!python Phase1_Dataset_Canonicalization_v1_0_0.py --write-config-template /content/phase1_config.json
```

Verify/edit `/content/phase1_config.json`, then canonicalize datasets incrementally:

```python
!python Phase1_Dataset_Canonicalization_v1_0_0.py --config /content/phase1_config.json --dataset wisig
!python Phase1_Dataset_Canonicalization_v1_0_0.py --config /content/phase1_config.json --dataset radioml2018
!python Phase1_Dataset_Canonicalization_v1_0_0.py --config /content/phase1_config.json --dataset radioml2016
```

Phase 1 preserves RF values and native sequence lengths. It standardizes only storage layout/dtype to `[N,2,L]` float32 and writes large verified TAR transport packs to Drive instead of thousands of loose bulk files.

## Persistent paths

Drive artifacts:

`/content/drive/MyDrive/Surrogate_XAI_V2/`

Local hot workspace:

`/content/surrogate_xai_v2/`

## Reproducibility

Datasets, generated shards, caches, checkpoints, Drive artifacts, and local runtime outputs are intentionally excluded from Git. Each scientific stage writes configuration, provenance, hashes, metrics, and completion manifests before persistence.
