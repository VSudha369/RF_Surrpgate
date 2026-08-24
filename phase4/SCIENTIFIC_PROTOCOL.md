# Phase 4 v1.0.0 — WiSig Teacher Representation Pre-Study

## Purpose

Phase 4 is a controlled **teacher representation-objective pre-study**. It is not teacher architecture search, final teacher training, strict zero-day evaluation, XAI evaluation, or surrogate evaluation.

It ports the validated Stage 2.6M question into the Surrogate-XAI V2 data/provenance stack:

- Does supervised contrastive learning improve known-transmitter separation and domain robustness over CE?
- Does prototype compactness improve known-manifold geometry and Train-Known-fitted novelty structure?
- Are the two auxiliary objectives complementary?

## Frozen V2 inputs

- WiSig native source SHA-256: `a8fc3e35134a240bfb4dab8862a6e482cef44de000b813d42417b853c47ccc7e`.
- Native signal representation: `float32 [N,2,256]` from Phase 1; Phase 1 stored layout/dtype only.
- Canonical Phase-3 split run: `20260824T100504Z`.
- Train Known: 388,139 samples / 98 transmitters.
- P0: 68,495; P1: 153,529; P2: 27,088; P3: 8,992.
- Calibration Unknown: 158,400 / 22 transmitters.
- Strict Zero-Day: structurally unavailable to Phase 4.

Phase 4 resolves only the six exported Phase-3 allowed index arrays by filename and SHA-256. There is no resolver route for strict roles.

## Preprocessing

Phase 1 remains immutable. Phase 4 preprocesses each sample in memory:

1. complex DC removal (subtract I and Q temporal means),
2. complex RMS normalization with `eps=1e-8`,
3. abort on non-finite or near-zero RMS.

No preprocessed signal dataset is written.

## Controlled arms

```text
A0: CE
A1: CE + 0.1 × SupCon(T=0.07)
A2: CE + 0.1 × Prototype
A3: CE + 0.1 × SupCon(T=0.07) + 0.1 × Prototype
```

Prototype targets are normalized EMA class prototypes with momentum 0.95. SupCon and prototype distances operate in FP32.

## Shared architecture

Input `[B,2,256]` → learned I/Q mixer → stride-2 temporal frontend → residual 64 → 128 → dilated 128 → 256 stack → global mean/std pooling → 128-D projection → 98-class head.

All arms share the same initialization within a seed. Architecture signature equality is recorded.

## Causal controls

Full profile freezes seeds `(42, 123, 2026)`, batch size 256, four samples per transmitter, AdamW (`lr=1e-3`, `weight_decay=1e-4`), cosine schedule, AMP, maximum 40 epochs, minimum 12 epochs, and synchronized group early stopping with patience 10.

Each seed/epoch materializes one deterministic transmitter-primary exposure, diversified over `(receiver, capture-date, equalization)` cells. Every arm consumes the same ordered exposure SHA and resets to the same stochastic stream for augmentation/dropout.

Augmentation is the Stage-2.6M conservative policy: phase rotation ±0.12 rad, amplitude jitter ±5%, RMS-relative AWGN 0.01, circular shift ±4 samples.

## Data access policy

- `train_known`: the only gradient-bearing role.
- `p0_known`: model selection only.
- P1/P2/P3: known-domain evaluation only.
- `calibration_unknown`: frozen-model, threshold-free unknown diagnostic only; never gradients and never model selection.
- strict zero-day / strict zero-day+shift: forbidden regardless of whether a future final-selection lock exists.

The strict self-test explicitly requests strict role names and must receive `StrictAccessViolation` for all of them.

## Metrics

Per-epoch P0–P3: accuracy, top-5 accuracy, CE, fixed-98 macro-F1, fixed-98 balanced accuracy. Missing P2/P3 known identities therefore receive zero recall/F1 in the fixed frame.

Best-primary checkpoints are ranked by macro-average fixed-98 P0–P3 macro-F1.

Downstream best-checkpoint diagnostics report:

- P0 representation Fisher ratio and centroid geometry,
- transmitter-balanced P0-vs-P1, P0-vs-P2, P0-vs-P3 protocol leakage AUROC,
- P0 equalization leakage AUROC,
- Train-Known-fitted centroid + Ledoit-Wolf residual covariance novelty geometry,
- P0 Known versus Calibration Unknown AUROC/AUPRC/Cohen-d for Euclidean, cosine, Mahalanobis, energy, and 1-MSP scores.

Calibration Unknown never fits centroids/covariance and no threshold is tuned.

## Selection

Lexicographic ranking:

1. mean macro-average fixed-98 P0–P3 macro-F1,
2. mean P0 Fisher ratio,
3. lower mean absolute domain-AUROC deviation from 0.5,
4. Calibration Unknown AUROC aggregate.

An auxiliary arm is promoted over CE only if its mean primary gain is at least 0.002, the paired-bootstrap 95% lower bound is at least -0.001, and mean P0 Fisher does not regress. Otherwise the result is `NO_OBJECTIVE_CLEARLY_SUPERIOR` and Phase 5 retains CE.

## Claims boundary

Phase 4 can recommend a representation objective for Phase 5 architecture screening. It cannot claim final teacher quality, strict zero-day generalization, explanation quality, surrogate fidelity, fidelity-gate performance, or deployment latency.
