# RF_Surrpgate — Surrogate-XAI V2

Reproducible implementation of the restructured RF Surrogate-XAI research pipeline.

## Execution contract

Google Drive is persistent storage only. Scientific hot I/O, DataLoader work, training, evaluation, XAI generation, and profiling run from the Colab VM local filesystem under `/content`.

Canonical data flow:

`Google Drive -> /content local SSD -> CPU/GPU computation -> local artifact -> Google Drive`

## Current implementation

- `phase0/` — Google Colab hardware/runtime autotuning.
- Canonical Phase-0 version: `v1.0.1`.
- Tesla T4 policy: FP16 autocast + GradScaler.
- Phase 1+ GPU work is gated on `GPU_AUTOTUNE_COMPLETE`.

## Run Phase 0 in Google Colab

1. Start a Colab **GPU runtime**.
2. Clone this repository:

```python
!git clone https://github.com/VSudha369/RF_Surrpgate.git
%cd /content/RF_Surrpgate/phase0
```

3. Run the canonical script:

```python
%run Phase0_Colab_Runtime_Autotune_v1_0_1.py
```

Alternatively, open `phase0/Phase0_Colab_Runtime_Autotune_v1_0_1.ipynb` directly in Colab and run all cells.

Expected successful T4 gate:

```text
Status: GPU_AUTOTUNE_COMPLETE
Precision: float16
GradScaler: True
PHASE-1 GATE: PASS
```

Runtime outputs are persisted under:

`/content/drive/MyDrive/Surrogate_XAI_V2/`

Local hot workspace:

`/content/surrogate_xai_v2/`

## Reproducibility

Datasets, generated shards, caches, checkpoints, Drive artifacts, and local runtime outputs are intentionally excluded from Git. Each scientific stage writes configuration, provenance, hashes, metrics, and a completion manifest before persistence.
