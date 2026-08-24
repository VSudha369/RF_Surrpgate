# Phase 4 v1.1.0 — Resource-Bounded WiSig Representation Benchmark

## Purpose

This version replaces the computationally impractical and selection-leaking v1.0.0 full
profile. It is a paired exploratory benchmark that decides whether the pilot-leading
SupCon arm is practically useful enough to carry into Phase 5.

It does not establish statistical superiority or final teacher quality.

## Frozen benchmark budget

- Train Known: 600 deterministic samples per known transmitter = 58,800 samples.
- P0 Known: 150 deterministic samples per known transmitter = 14,700 samples.
- Arms: A0 CE and A1 CE + 0.1 SupCon.
- Paired seeds: 42, 123, 2026.
- Maximum epochs: 8.
- Minimum epochs: 4.
- Group early-stopping patience: 2.
- Maximum training steps: 11,040 across all arms and seeds.

A2 and A3 are removed because neither beat A0 in the verified pilot and the user requested
a short benchmark rather than a long four-arm experiment.

## Leakage-free selection

Only `train_known` and `p0_known` are resolved before objective selection.

- Gradients: Train Known only.
- Checkpointing: P0 Known fixed-98 macro-F1 only.
- Early stopping: P0 Known only.
- Objective selection: P0 Known macro-F1 and P0 Fisher geometry only.

P1/P2/P3 and Calibration Unknown are not resolved until the selected arm is written to
`OBJECTIVE_SELECTION_P0_ONLY.json`. Their results cannot change the selected arm.

## Practical selection rule

A1 is selected only if all conditions hold:

1. mean paired P0 macro-F1 gain over A0 is at least 0.002;
2. A1 beats A0 for at least two of three seeds;
3. mean P0 Fisher ratio does not regress.

Otherwise A0/CE is retained. Three seeds support an exploratory engineering decision only;
no p-value or statistical-superiority claim is made.

## Post-selection evaluation

After the selection file is persisted:

- both arms are evaluated once on sampled P1/P2/P3 known shifts;
- only the selected arm receives Calibration-Unknown novelty diagnostics;
- no threshold is tuned on Strict Zero-Day;
- Strict Zero-Day remains structurally inaccessible.

## Interruption-safe persistence

RF samples stay on local `/content`. After every fully completed paired-arm epoch, a small
recovery bundle is atomically copied to Drive and SHA-256 verified. It contains model,
prototype bank, optimizer, scheduler, GradScaler, current/best epoch states, early-stop
state, history, RNG states, configuration hash, and matched exposure hashes.

Rerunning the same command and `--run-id` restores the last verified completed epoch. An
exception traceback is persisted to `PHASE4_FAILURE.json` whenever possible.

## Claims boundary

This stage may recommend A0 or A1 for Phase 5. It cannot claim final teacher quality,
strict-zero-day performance, XAI fidelity, surrogate fidelity, latency, or statistical
superiority.
