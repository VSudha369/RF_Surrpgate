# Phase 5 v1.0.0 — WiSig Teacher Architecture Screening Protocol

## Frozen upstream dependency

Phase 4 v1.1.0 is immutable. Every teacher candidate uses **A0 = Cross-Entropy**. Phase 5 changes architecture only.

## Data-access contract

- Gradient-bearing data: `train_known` only.
- Architecture selection: `p0_known` only.
- After the architecture is locked, evaluate `p1_cross_day`, `p2_cross_receiver`, and `p3_cross_day_receiver` once as Known-shift diagnostics.
- Calibration Unknown is not authorized anywhere in Phase 5.
- Strict Zero-Day / Strict Unknown are structurally forbidden.
- WiSig signals must be staged locally under `/content`; Drive is persistence only.
- Native WiSig sequence length remains `L=256`; no global resizing and no legacy 65-bin rFFT representation.

## Candidate teacher families

- `R0_RESNET`: residual temporal CNN.
- `T0_TCN`: multi-scale dilated depthwise-separable TCN.
- `X0_TRANSFORMER`: convolutional tokenization plus compact Transformer encoder.
- `F0_TIME_FREQ`: dual branch over native time-domain IQ and the **full complex FFT spectrum**.

All candidates expose the same output contract: 98-class logits, 128-D raw embedding, and normalized 128-D embedding. They use global/adaptive pooling and are length-agnostic by construction.

## Matched screening budget

Three paired seeds: `42, 123, 2026`. Up to 8 epochs, minimum 4, patience 2. Train subset: 600 samples per Known transmitter (58,800 total). P0 selection subset: 150 samples per Known transmitter (14,700 total). Batch size 256 with the frozen domain-balanced transmitter sampler. Optimizer and conservative RF augmentation remain matched across candidates.

## Selection rule

1. Rank by mean P0 fixed-98 macro-F1 across seeds.
2. If candidates are within `0.002` absolute macro-F1, prefer higher mean P0 Fisher ratio.
3. If still tied, prefer the lower parameter count.
4. Persist `PHASE5_SELECTION_LOCK.json` **before** resolving P1/P2/P3.
5. Do not use shift performance to revise the locked architecture.

Three seeds are exploratory; do not make a significance claim.

## Claims boundary

Phase 5 selects a teacher architecture under a bounded Known-only screen. It does not constitute final teacher training, open-set calibration, strict-zero-day evaluation, XAI evaluation, fidelity evidence, or deployment evidence.
