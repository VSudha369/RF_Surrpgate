# Phase 1 v1.0.2 — Native WiSig ManyTx Canonicalization

This revision adds the source-authentic WiSig adapter to the already validated
Phase-1 RadioML storage pipeline.

## Why v1.0.2 exists

The previously discovered file:

`processed/wisig/WiSig_tensors.h5`

contains only `X` with shape `[1,020,643, 2, 128]`. It is a derived/truncated
representation and does not contain transmitter/receiver/date/equalization metadata.

The source-authentic WiSig ManyTx archive instead contains:
- 1,020,643 samples
- 150 transmitters
- 18 receivers
- 4 capture dates
- 2 equalization states
- 20,759 nonempty cells
- 841 empty cells
- native source signal length 256
- source orientation `[N,L,IQ]`

v1.0.2 therefore canonicalizes `ManyTx.pkl.zip`, not the old HDF5.

## Canonical WiSig representation

Signals:
- dtype: `float32`
- layout: `[N,2,256]`
- values preserved; no DC removal or RMS normalization

Metadata per sample:
- `global_index`
- `label` (transmitter index)
- `tx_id`
- `receiver`
- `day` (capture-date index alias)
- `capture_date`
- `equalization`
- `source_cell_id`
- `sample_in_cell`

`SOURCE_VALUE_MAPPINGS.json` stores the original transmitter, receiver,
capture-date, and equalization values.

## Deterministic flatten order

`tx -> receiver -> capture_date -> equalization -> sample-within-cell`

No session identifier is fabricated. The source has no session axis, so session
is explicitly recorded as `NOT_REPRESENTED`.

## Hard scientific invariants

The run aborts if any of these differ from the expected native ManyTx schema:
- samples: 1,020,643
- L: 256
- transmitters: 150
- receivers: 18
- capture dates: 4
- equalization states: 2
- nonempty cells: 20,759
- empty cells: 841
- source orientation: `[N,L,IQ]`

## Colab run

Mount Drive from a notebook cell:

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
```

Update repository:

```python
%cd /content/RF_Surrpgate
!git pull origin main
%cd /content/RF_Surrpgate/phase1
```

Discover sources:

```python
!python Phase1_Dataset_Canonicalization_v1_0_2.py --discover
```

Use the versioned WiSig config:

```python
!cp phase1_config_v1_0_2_template.json /content/phase1_config_wisig.json
```

Verify `datasets.wisig.source_drive_path`, then run:

```python
!python Phase1_Dataset_Canonicalization_v1_0_2.py \
    --config /content/phase1_config_wisig.json \
    --dataset wisig
```

Recommended runtime: CPU / High-RAM.

## Storage contract

`Drive ZIP -> local ZIP -> local ManyTx.pkl -> local canonical shards -> verified TAR transport packs -> Drive`

No DataLoader or scientific hot loop reads RF samples directly from Drive.
