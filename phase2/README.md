# Phase 2 — Local Stage Manager v1.0.0

Phase 2 is the single staging contract for all later scientific phases.

## Frozen real Phase-1 targets

| Dataset | Phase-1 run | N | L | Shards | Packs |
|---|---|---:|---:|---:|---:|
| WiSig | `20260824T082742Z` | 1,020,643 | 256 | 127 | 1 |
| RadioML 2016.10A | `20260824T073604Z` | 220,000 | 128 | 4 | 1 |
| RadioML 2018.01A | `20260824T073812Z` | 2,555,904 | 1024 | 157 | 10 |

The registry freezes the exact source SHA-256 and every Phase-1 TAR filename,
byte size, and SHA-256. A later phase cannot silently stage a different dataset run.

## Contract

`Drive transport pack -> /content TAR -> SHA-256 -> safe extract -> canonical shard SHA-256 -> local completion marker`

Drive is never a DataLoader/training/XAI hot path.

## Commands

Mount Drive in a Colab notebook cell first:

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
```

Then:

```python
%cd /content/RF_Surrpgate
!git pull origin main
%cd /content/RF_Surrpgate/phase2
```

Fast metadata/manifest validation of all real targets:

```python
!python Phase2_Local_Stage_Manager_v1_0_0.py validate-drive --dataset all
```

Full RML16 stage-in smoke test:

```python
!python Phase2_Local_Stage_Manager_v1_0_0.py stage-in   --dataset radioml2016 --verification full
```

Resolve the exact local canonical root for downstream code:

```python
!python Phase2_Local_Stage_Manager_v1_0_0.py resolve --dataset radioml2016
```

Stage WiSig:

```python
!python Phase2_Local_Stage_Manager_v1_0_0.py stage-in   --dataset wisig --verification full
```

Stage RadioML 2018:

```python
!python Phase2_Local_Stage_Manager_v1_0_0.py stage-in   --dataset radioml2018 --verification full
```

Clean a staged dataset:

```python
!python Phase2_Local_Stage_Manager_v1_0_0.py clean --dataset radioml2018
```

## Verification levels

`full` (default for stage-in):
- SHA-256 every Drive TAR while copying;
- reject unsafe TAR member types/paths;
- require TAR member list to equal the frozen transport manifest;
- SHA-256 every extracted signal and metadata shard;
- validate float32 `[N,2,L_native]`;
- validate per-shard sample counts;
- validate contiguous global sample IDs;
- validate total sample count and label cardinality.

`manifest`:
- skips per-shard SHA re-read;
- still checks file presence, shapes, metadata continuity, counts, and labels.

Use `full` for first stage-in on a runtime and for scientific release validation.

## Local layout

```text
/content/surrogate_xai_v2/staged_datasets/
  wisig/
    20260824T082742Z/
      canonical/
      provenance/
      LOCAL_STAGE_COMPLETE.json
    current -> 20260824T082742Z
```

Equivalent fixed run directories are used for RML16 and RML18.

Later phases should call `resolve_local_canonical_root()` or the CLI `resolve`
command rather than hard-coding local paths.

## Reusable modules

- `stage_common.py` — path guards, SHA/copy, TAR safety, locking, disk preflight.
- `stage_validation.py` — frozen Drive-target and extracted-shard validation.
- `stage_manager.py` — orchestration, local resolution/cleanup, atomic file stage-out.
