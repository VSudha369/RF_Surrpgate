# Phase 4 — WiSig Teacher Representation Pre-Study v1.0.0

This phase ports the earlier controlled Stage 2.6M representation-objective ablation into the V2 pipeline while keeping the V2 Phase-1/2/3 contracts authoritative.

## Safety contract

Only Phase-3 `train_known` can produce gradients. P0 is model-selection-only; P1/P2/P3 are known-shift evaluation; Calibration Unknown is threshold-free diagnostic-only. Strict Zero-Day has no resolver route in Phase 4 and the strict-access self-test must fail closed.

## Before running

Use a Tesla T4 runtime for the canonical `full` profile. Mount Drive, pull `main`, and stage native WiSig locally through Phase 2:

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

%cd /content/RF_Surrpgate
!git pull origin main

%cd /content/RF_Surrpgate/phase2
!python Phase2_Local_Stage_Manager_v1_0_0.py stage-in --dataset wisig --verification full
!python Phase2_Local_Stage_Manager_v1_0_0.py resolve --dataset wisig
```

Phase 4 never reads RF signals from Drive in its training/evaluation loop.

## Pilot

Run this first to validate GPU/model/data/access wiring. Pilot is not scientifically selectable evidence.

```python
%cd /content/RF_Surrpgate/phase4
!python Phase4_WiSig_Teacher_Representation_PreStudy_v1_0_0.py --profile pilot
```

Expected terminal status: `PHASE4_PILOT_COMPLETE` and `Strict zero-day accessed: False`.

## Full canonical pre-study

```python
%cd /content/RF_Surrpgate/phase4
!python Phase4_WiSig_Teacher_Representation_PreStudy_v1_0_0.py --profile full
```

Full profile freezes seeds `42,123,2026`, A0–A3, maximum 40 epochs, synchronized early stopping, native `L=256`, and the canonical Phase-3 run `20260824T100504Z`.

Outputs:

```text
/content/surrogate_xai_v2/04_phase4_wisig_teacher_representation_prestudy/run_<UTC>/
/content/drive/MyDrive/Surrogate_XAI_V2/04_PHASE4_WISIG_TEACHER_REPRESENTATION_PRESTUDY/run_<UTC>/
```

The final decision is in `metrics/OBJECTIVE_SELECTION.json` and `PHASE4_SUMMARY.json`.
