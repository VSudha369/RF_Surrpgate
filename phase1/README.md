# Phase 1 — Dataset Canonicalization and Storage Pipeline v1.0.0

## Purpose
Canonicalize WiSig, RadioML 2016.10A, and RadioML 2018.01A into deterministic local RF shards while enforcing:

**Drive → local `/content` → processing → local transport packs → Drive**

No scientific parser, canonicalizer, DataLoader, or training stage reads bulk RF samples directly from Google Drive.

## Scientific preservation rule
Phase 1 changes only representation layout/dtype:
- canonical layout: `[N, 2, L]`
- canonical dtype: `float32`
- native sequence length is preserved
- no DC removal, RMS normalization, denoising, resampling, truncation, or padding

This keeps later preprocessing ablations scientifically valid.

## Outputs
Each dataset produces:
- `SOURCE_INSPECTION.json`
- `DATASET_MANIFEST.json`
- `CANONICAL_VERIFICATION.json`
- logical local `signals_*.npy`
- matching `meta_*.npz`
- uncompressed TAR transport packs
- `TRANSPORT_MANIFEST.json`
- Drive completion marker

Loose bulk shards are intentionally **not** persisted to Drive. Drive receives large transport packs plus compact manifests.

## Recommended runtime
Use a CPU/high-RAM Colab runtime for Phase 1. A GPU is not required.

## First run
```python
!rm -rf /content/RF_Surrpgate
!git clone https://github.com/VSudha369/RF_Surrpgate.git /content/RF_Surrpgate
%cd /content/RF_Surrpgate/phase1
!python Phase1_Dataset_Canonicalization_v1_0_0.py --discover
!python Phase1_Dataset_Canonicalization_v1_0_0.py --write-config-template /content/phase1_config.json
```

Open `/content/phase1_config.json` and set/verify the three `source_drive_path` values.
For WiSig, set explicit HDF5 metadata dataset names if auto-detection cannot safely infer them.

Then run:
```python
!python Phase1_Dataset_Canonicalization_v1_0_0.py --config /content/phase1_config.json --dataset all
```

For incremental execution:
```python
!python Phase1_Dataset_Canonicalization_v1_0_0.py --config /content/phase1_config.json --dataset wisig
!python Phase1_Dataset_Canonicalization_v1_0_0.py --config /content/phase1_config.json --dataset radioml2016
!python Phase1_Dataset_Canonicalization_v1_0_0.py --config /content/phase1_config.json --dataset radioml2018
```

## WiSig frozen splits
If the existing validated Stage-1B split-array directory is known, set:
`datasets.wisig.frozen_split_dir_drive`

Only small `.npy`/`.json` split/manifests are copied directly from Drive. Bulk RF signals are not.

## Failure policy
The pipeline aborts on:
- source SHA mismatch
- missing source
- non-finite signal data
- sequence-length changes within a dataset
- non-contiguous global IDs
- shard hash failure
- unexpected I/Q layout

## Reusable later-phase storage API
`storage_contract.py` also exports:
- `stage_in_transport_pack(...)`
- `safe_extract_tar(...)`
- `assert_local_hot_path(...)`

Later phases should import these rather than implementing new Drive readers.

## Phase-0 provenance gate
Before any dataset is canonicalized, Phase 1 stages the latest completed Phase-0
archive locally, verifies its SHA-256, reads `PHASE0_FINAL_STATUS.json`, and requires:

`status == "GPU_AUTOTUNE_COMPLETE"`

A CPU-only Phase-0 preflight is rejected.
