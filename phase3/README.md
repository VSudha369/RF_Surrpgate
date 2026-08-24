# Phase 3 — Frozen Deterministic Open-Set Split Construction v1.0.0

Phase 3 creates **index manifests only**. It never rewrites RF signal values.

## RadioML

- RML16: 7 Known + 2 Calibration Unknown + 2 Strict Unknown.
- RML18: 16 Known + 4 Calibration Unknown + 4 Strict Unknown.
- Class assignment is deterministic and family-stratified.
- Every modulation family retains at least one Known class.
- Within Known classes, samples are deterministically split inside each `(class, SNR)` stratum into 70% train / 15% validation / remainder test.
- Calibration Unknown samples are exported only for novelty/threshold calibration and never for gradients.
- Strict Unknown class identities are frozen before training, but **strict sample indices are not exported**. Only a SHA-256 commitment to the deterministic sorted strict index set is written.

## WiSig

Phase 3 bridges the authoritative `MANYTX_ZERO_DAY_BRANCH_v1.0.3` split to the new native Phase-1 source because both are proven to derive from the same `ManyTx.pkl.zip` SHA-256:

`a8fc3e35134a240bfb4dab8862a6e482cef44de000b813d42417b853c47ccc7e`

Frozen roles:

- Train Known: 388,139 samples / 98 transmitters
- P0 Known Validation: 68,495
- P1 Cross-Day: 153,529
- P2 Cross-Receiver: 27,088
- P3 Cross-Day+Receiver: 8,992
- Calibration Unknown: 158,400 / 22 transmitters
- Strict Zero-Day: 216,000 / 30 transmitters
- Strict Zero-Day+Shift: 3,000

Held-out day: `2021_03_15`; held-out receiver: `19-1`.

Strict WiSig files are only byte-hashed during Phase 3. They are **never loaded with NumPy and never copied to the normal Phase-3 output tree**.

## Run in Colab

Mount Drive first, pull the repo, then:

```python
%cd /content/RF_Surrpgate
!git pull origin main
%cd /content/RF_Surrpgate/phase3
```

RML16 is already staged from Phase 2:

```python
!python Phase3_Frozen_OpenSet_Splits_v1_0_0.py --dataset radioml2016
```

WiSig split bridge does not require staging WiSig signals:

```python
!python Phase3_Frozen_OpenSet_Splits_v1_0_0.py --dataset wisig
```

Before RML18 split construction, stage it through Phase 2, then:

```python
!python Phase3_Frozen_OpenSet_Splits_v1_0_0.py --dataset radioml2018
```

Outputs go to:

`/content/surrogate_xai_v2/03_phase3_open_set_splits/run_<UTC>/`

and are persisted to:

`/content/drive/MyDrive/Surrogate_XAI_V2/03_PHASE3_OPEN_SET_SPLITS/run_<UTC>/`

## Scientific access policy

`known_train` is the only role permitted for ordinary supervised fitting. `known_val`/P0 is for model selection. Calibration Unknown may tune open-set operating points but must never contribute gradients. Known test / P1 / P2 / P3 are evaluation roles. Strict Unknown / Strict Zero-Day is final-evaluation-only after a final-selection lock.

## Strict-access guard

`strict_access_guard.py` refuses to resolve a strict WiSig index source until a future `FINAL_SELECTION_LOCK.json` records that the teacher, surrogate, open-set threshold, and fidelity gate are frozen and binds them to a final configuration SHA-256. The checked-in template is deliberately `NOT_LOCKED`.
