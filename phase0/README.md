# Phase 0 — Colab Runtime Autotune v1.0.1

Canonical runtime preflight/autotuning for Surrogate-XAI V2.

## Key policy

- Tesla T4 / Turing (compute capability 7.5): **FP16 autocast + GradScaler**.
- Ampere-or-newer GPUs with trusted BF16 support: BF16.
- CPU-only runs are recorded as `CPU_PREFLIGHT_COMPLETE`; GPU-dependent later phases require `GPU_AUTOTUNE_COMPLETE`.
- Google Drive is persistence only. Benchmarks and hot I/O run under `/content/surrogate_xai_v2/`.

## Run

In a Colab GPU runtime:

```python
!git clone https://github.com/VSudha369/RF_Surrpgate.git
%cd /content/RF_Surrpgate/phase0
%run Phase0_Colab_Runtime_Autotune_v1_0_1.py
```

Expected T4 policy:

```text
train_autocast_dtype = float16
use_grad_scaler = True
```

A successful GPU run ends with:

```text
PHASE 0 STATUS: GPU_AUTOTUNE_COMPLETE
```

Outputs are written locally first and then archived to:

`/content/drive/MyDrive/Surrogate_XAI_V2/00_PHASE0_RUNTIME_AUTOTUNE/`
