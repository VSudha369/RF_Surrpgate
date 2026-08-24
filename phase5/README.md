# Phase 5 — WiSig Teacher Architecture Screening

Phase 5 begins after the frozen Phase-4 v1.1.0 decision `A0 = Cross-Entropy`.

The current v1.0.0 starter freezes the Known-only screening protocol and validates four length-agnostic teacher families before any expensive Colab training is launched. No Calibration Unknown or Strict Zero-Day data is authorized in this phase.

## Files

- `phase5_config_v1_0_0.json` — frozen screening budget/access/selection contract.
- `teacher_architectures.py` — R0/T0/X0/F0 candidate backbones.
- `PHASE5_PROTOCOL.md` — scientific protocol and claims boundary.
- `PHASE5_SELECTION_LOCK_TEMPLATE.json` — fail-closed selection lock template.
- `validate_phase5_screening_v1_0_0.py` — static/model-contract validation.
- `Phase5_WiSig_Teacher_Architecture_Screening_v1_0_0.py` — preflight entrypoint; training runner follows after Colab smoke validation.

## First command

```bash
cd phase5
python validate_phase5_screening_v1_0_0.py
python Phase5_WiSig_Teacher_Architecture_Screening_v1_0_0.py --preflight-only
```

Expected status: `PHASE5_SCREENING_PREFLIGHT_READY`.
