# Corrected Phase 4 v1.1.0 — Short WiSig Benchmark

This update keeps the verified v1.0.0 pilot unchanged and adds a corrected, short benchmark.

## What was corrected

- maximum 8 epochs instead of 40;
- 58,800 balanced Train-Known samples instead of all 388,139;
- A0 versus pilot-leading A1 only;
- P0-only checkpointing, early stopping, and objective selection;
- P1/P2/P3 resolved only after selection is locked;
- Calibration Unknown removed from selection;
- complete epoch-level deterministic resume state persisted to Drive;
- component-wise CE/SupCon/prototype losses recorded;
- failed traceback persisted to Drive when possible.

## Colab execution

Mount Drive, clone/pull the repository, and stage WiSig locally through Phase 2:

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

%cd /content/RF_Surrpgate/phase2
!python Phase2_Local_Stage_Manager_v1_0_0.py stage-in --dataset wisig --verification full
!python Phase2_Local_Stage_Manager_v1_0_0.py resolve --dataset wisig
```

Run the corrected benchmark:

```python
%cd /content/RF_Surrpgate/phase4
!python Phase4_WiSig_Teacher_Representation_Benchmark_v1_1_0.py \
    --run-id benchmark_v1_1_0
```

If Colab disconnects, stage WiSig locally again and rerun the identical command. It restores
the last verified completed epoch from Drive.

Expected final terminal status:

```text
PHASE 4 v1.1.0 STATUS: PHASE4_BENCHMARK_COMPLETE
Selection protocol: P0 Known only
Strict Zero-Day accessed: False
```

The final selection is stored in:

```text
metrics/OBJECTIVE_SELECTION_P0_ONLY.json
PHASE4_BENCHMARK_SUMMARY.json
PHASE4_BENCHMARK_COMPLETE.json
```
