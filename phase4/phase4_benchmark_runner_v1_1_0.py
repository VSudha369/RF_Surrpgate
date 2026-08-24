#!/usr/bin/env python3
from __future__ import annotations

import gc
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

from access_guard import strict_access_self_test, verify_phase3_contract
from evaluation import calibration_unknown_diagnostic, fit_novelty_geometry, novelty_scores
from model_and_losses import WiSigRepresentationNet
from phase4_common import atomic_json, atomic_text, copy_tree_verified, utc_now
from phase4_benchmark_runtime_v1_1_0 import (
    PHASE4_BENCHMARK_VERSION,
    BenchmarkConfig,
    build_known_label_mapping,
    collect_outputs,
    cuda_preflight,
    evaluate_known,
    make_benchmark_dataset,
    persist_file_verified,
)
from phase4_benchmark_training_v1_1_0 import train_seed
from wisig_dataset import CanonicalWiSigStore


def _persist_json(local_path: Path, drive_path: Path, payload: Mapping[str, Any]) -> None:
    atomic_json(local_path, payload)
    persist_file_verified(local_path, drive_path)


def select_benchmark_arm(seed_summaries: Mapping[int, Dict[str, Any]], config: BenchmarkConfig):
    arm_summary: Dict[str, Any] = {}
    for arm in config.arms:
        p0_values = [
            float(seed_summaries[seed]["arms"][arm]["p0_metrics"]["fixed98_macro_f1"])
            for seed in config.seeds
        ]
        fisher_values = [
            float(seed_summaries[seed]["arms"][arm]["p0_geometry"]["fisher_ratio"])
            for seed in config.seeds
        ]
        arm_summary[arm] = {
            "mean_p0_fixed98_macro_f1": float(np.mean(p0_values)),
            "std_p0_fixed98_macro_f1": float(np.std(p0_values, ddof=1)),
            "mean_p0_fisher_ratio": float(np.mean(fisher_values)),
            "per_seed_p0_fixed98_macro_f1": p0_values,
            "per_seed_p0_fisher_ratio": fisher_values,
        }

    a0 = np.asarray(arm_summary["A0"]["per_seed_p0_fixed98_macro_f1"])
    a1 = np.asarray(arm_summary["A1"]["per_seed_p0_fixed98_macro_f1"])
    difference = a1 - a0
    wins = int(np.sum(difference > 0))
    mean_difference = float(np.mean(difference))
    fisher_nonregression = (
        arm_summary["A1"]["mean_p0_fisher_ratio"]
        >= arm_summary["A0"]["mean_p0_fisher_ratio"]
    )
    selected = "A1" if (
        mean_difference >= config.practical_f1_delta
        and wins >= 2
        and fisher_nonregression
    ) else "A0"
    decision = "SELECT_A1_CE_SUPCON" if selected == "A1" else "RETAIN_A0_CE"
    return {
        "selection_data": "P0 Known only",
        "selection_locked_before_shift_or_unknown_resolution": True,
        "arm_summary": arm_summary,
        "a1_minus_a0_per_seed": difference.tolist(),
        "a1_minus_a0_mean": mean_difference,
        "a1_seed_wins": wins,
        "fisher_nonregression": bool(fisher_nonregression),
        "practical_f1_delta": float(config.practical_f1_delta),
        "decision": decision,
        "selected_arm": selected,
        "statistical_claim": "Exploratory paired benchmark only; no significance claim from three seeds.",
    }


def _load_model(local_root: Path, seed: int, arm: str, config: BenchmarkConfig, device):
    checkpoint_path = local_root / "checkpoints" / f"seed_{seed}" / arm / "best_p0.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("config_sha256") != config.digest():
        raise RuntimeError(f"Checkpoint configuration mismatch: seed={seed} arm={arm}")
    model = WiSigRepresentationNet(config.num_classes, config.embedding_dim, config.dropout).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def run_benchmark(
    canonical_root: Path,
    local_root: Path,
    drive_root: Path,
    config: BenchmarkConfig,
) -> Dict[str, Any]:
    config.validate()
    local_root.mkdir(parents=True, exist_ok=True)
    drive_root.mkdir(parents=True, exist_ok=True)
    for subfolder in ("configs", "checkpoints", "metrics", "manifests", "recovery", "reports"):
        (local_root / subfolder).mkdir(exist_ok=True)
        (drive_root / subfolder).mkdir(exist_ok=True)

    completed_drive = drive_root / "PHASE4_BENCHMARK_COMPLETE.json"
    summary_drive = drive_root / "PHASE4_BENCHMARK_SUMMARY.json"
    if completed_drive.is_file() and summary_drive.is_file():
        from phase4_common import load_json

        summary = load_json(summary_drive)
        if summary.get("config_sha256") != config.digest():
            raise RuntimeError("Existing completed benchmark uses a different frozen configuration")
        print("[COMPLETE] Verified completed Phase 4 v1.1.0 benchmark found on Drive")
        return summary

    access_contract = verify_phase3_contract()
    strict_self_test = strict_access_self_test()
    gpu_runtime = cuda_preflight(config)
    device = torch.device("cuda")
    store = CanonicalWiSigStore(canonical_root)
    mapping, mapping_evidence = build_known_label_mapping(store)

    _persist_json(
        local_root / "configs" / "CONFIG.json",
        drive_root / "configs" / "CONFIG.json",
        {**asdict(config), "config_sha256": config.digest()},
    )
    _persist_json(
        local_root / "manifests" / "ACCESS_CONTRACT.json",
        drive_root / "manifests" / "ACCESS_CONTRACT.json",
        {**access_contract, **strict_self_test},
    )
    _persist_json(
        local_root / "manifests" / "KNOWN_LABEL_MAPPING.json",
        drive_root / "manifests" / "KNOWN_LABEL_MAPPING.json",
        mapping_evidence,
    )
    _persist_json(
        local_root / "manifests" / "GPU_RUNTIME.json",
        drive_root / "manifests" / "GPU_RUNTIME.json",
        gpu_runtime,
    )

    # Selection phase: only Train-Known and P0 are resolved.
    train_dataset, train_evidence = make_benchmark_dataset(
        store, "train_known", "gradient", mapping, config
    )
    p0_dataset, p0_evidence = make_benchmark_dataset(
        store, "p0_known", "model_selection", mapping, config
    )
    selection_split_evidence = {
        "train_known": train_evidence,
        "p0_known": p0_evidence,
        "roles_not_resolved_before_selection": [
            "p1_cross_day",
            "p2_cross_receiver",
            "p3_cross_day_receiver",
            "calibration_unknown",
            "strict_zero_day",
            "strict_zero_day_shift",
            "strict_unknown",
        ],
    }
    _persist_json(
        local_root / "manifests" / "SELECTION_SPLIT_EVIDENCE.json",
        drive_root / "manifests" / "SELECTION_SPLIT_EVIDENCE.json",
        selection_split_evidence,
    )

    estimated_steps = (
        int(np.ceil(len(train_dataset) / config.batch_size))
        * len(config.arms)
        * config.max_epochs
        * len(config.seeds)
    )
    benchmark_budget = {
        "train_samples": int(len(train_dataset)),
        "p0_samples": int(len(p0_dataset)),
        "arms": list(config.arms),
        "seeds": list(config.seeds),
        "maximum_epochs": int(config.max_epochs),
        "maximum_optimizer_steps": int(estimated_steps),
        "selection_evaluations_per_epoch": ["p0_known"],
        "shift_evaluation_cadence": "once after selection lock",
        "calibration_unknown_cadence": "once after selection lock for selected arm only",
    }
    _persist_json(
        local_root / "manifests" / "BENCHMARK_BUDGET.json",
        drive_root / "manifests" / "BENCHMARK_BUDGET.json",
        benchmark_budget,
    )

    seed_summaries = {}
    for seed in config.seeds:
        print(f"[RUN] Phase 4 v1.1.0 paired benchmark seed={seed}")
        seed_summaries[int(seed)] = train_seed(
            int(seed), train_dataset, p0_dataset, local_root, drive_root, config, device
        )

    selection = select_benchmark_arm(seed_summaries, config)
    _persist_json(
        local_root / "metrics" / "OBJECTIVE_SELECTION_P0_ONLY.json",
        drive_root / "metrics" / "OBJECTIVE_SELECTION_P0_ONLY.json",
        selection,
    )
    selected_arm = selection["selected_arm"]
    print(f"[LOCKED] Objective selection={selected_arm}; now resolving shift-evaluation roles")

    # Evaluation phase: these roles cannot alter the already persisted selection decision.
    shift_datasets, evaluation_evidence = {}, {}
    for role in ("p1_cross_day", "p2_cross_receiver", "p3_cross_day_receiver"):
        shift_datasets[role], evaluation_evidence[role] = make_benchmark_dataset(
            store, role, "known_eval", mapping, config
        )
    calibration_dataset, evaluation_evidence["calibration_unknown"] = make_benchmark_dataset(
        store, "calibration_unknown", "unknown_diagnostic", mapping, config
    )
    evaluation_evidence["selection_was_locked_before_resolution"] = True
    evaluation_evidence["selected_arm"] = selected_arm
    _persist_json(
        local_root / "manifests" / "POST_SELECTION_EVALUATION_EVIDENCE.json",
        drive_root / "manifests" / "POST_SELECTION_EVALUATION_EVIDENCE.json",
        evaluation_evidence,
    )

    shift_results: Dict[str, Any] = {}
    for seed in config.seeds:
        shift_results[str(seed)] = {}
        for arm in config.arms:
            model, checkpoint = _load_model(local_root, int(seed), arm, config, device)
            role_metrics = {
                role: evaluate_known(model, dataset, device, config)
                for role, dataset in shift_datasets.items()
            }
            shift_results[str(seed)][arm] = {
                "best_p0_epoch": int(checkpoint["epoch"]),
                "evaluation_only_metrics": role_metrics,
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()
    _persist_json(
        local_root / "metrics" / "LOCKED_SHIFT_EVALUATION.json",
        drive_root / "metrics" / "LOCKED_SHIFT_EVALUATION.json",
        {
            "selection_locked": selection,
            "selection_unchanged_by_these_metrics": True,
            "results": shift_results,
        },
    )

    unknown_results: Dict[str, Any] = {}
    for seed in config.seeds:
        model, checkpoint = _load_model(local_root, int(seed), selected_arm, config, device)
        train_outputs = collect_outputs(model, train_dataset, device, config, include_logits=True)
        p0_outputs = collect_outputs(model, p0_dataset, device, config, include_logits=True)
        calibration_outputs = collect_outputs(
            model, calibration_dataset, device, config, include_logits=True
        )
        centroids, normalized_centroids, precision, shrinkage = fit_novelty_geometry(
            train_outputs["embedding"],
            train_outputs["y"],
            config.num_classes,
            config.covariance_fit_limit,
            int(seed) + 5,
        )
        p0_scores = novelty_scores(
            p0_outputs["embedding"], p0_outputs["logits"], centroids, normalized_centroids, precision
        )
        calibration_scores = novelty_scores(
            calibration_outputs["embedding"],
            calibration_outputs["logits"],
            centroids,
            normalized_centroids,
            precision,
        )
        unknown_results[str(seed)] = {
            "selected_arm": selected_arm,
            "best_p0_epoch": int(checkpoint["epoch"]),
            "covariance_shrinkage": float(shrinkage),
            "calibration_unknown_diagnostic": calibration_unknown_diagnostic(
                p0_scores, calibration_scores
            ),
            "selection_influence": False,
        }
        del model, train_outputs, p0_outputs, calibration_outputs, p0_scores, calibration_scores
        gc.collect()
        torch.cuda.empty_cache()
    _persist_json(
        local_root / "metrics" / "POST_SELECTION_CALIBRATION_UNKNOWN.json",
        drive_root / "metrics" / "POST_SELECTION_CALIBRATION_UNKNOWN.json",
        unknown_results,
    )

    forbidden_files = [
        str(path.relative_to(local_root))
        for path in local_root.rglob("*")
        if path.is_file() and "strict" in path.name.lower()
        and path.suffix.lower() in {".npy", ".npz", ".pt", ".pth", ".csv"}
    ]
    if forbidden_files:
        raise RuntimeError(f"Strict-data artifact leakage detected: {forbidden_files}")
    strict_audit = {
        "strict_role_resolver_available": False,
        "strict_indices_loaded": False,
        "strict_signals_accessed": False,
        "strict_artifacts_found": forbidden_files,
        "strict_access_self_test": strict_self_test,
    }
    _persist_json(
        local_root / "manifests" / "STRICT_ACCESS_AUDIT.json",
        drive_root / "manifests" / "STRICT_ACCESS_AUDIT.json",
        strict_audit,
    )

    summary = {
        "phase": "PHASE4_WISIG_TEACHER_REPRESENTATION_BENCHMARK",
        "phase4_benchmark_version": PHASE4_BENCHMARK_VERSION,
        "status": "PHASE4_BENCHMARK_COMPLETE",
        "completed_utc": utc_now(),
        "config_sha256": config.digest(),
        "phase3_run_id": access_contract["phase3_run_id"],
        "source_sha256": access_contract["source_sha256"],
        "benchmark_budget": benchmark_budget,
        "selection": selection,
        "selected_arm_for_phase5": selected_arm,
        "selection_protocol": "P0-only; P1/P2/P3 and Calibration Unknown resolved after lock",
        "strict_access": strict_audit,
        "claims_boundary": (
            "Resource-bounded representation-objective benchmark only; no strict-zero-day, "
            "final-teacher, XAI-fidelity, statistical-superiority, or deployment claim."
        ),
    }
    _persist_json(
        local_root / "PHASE4_BENCHMARK_SUMMARY.json",
        drive_root / "PHASE4_BENCHMARK_SUMMARY.json",
        summary,
    )
    _persist_json(
        local_root / "PHASE4_BENCHMARK_COMPLETE.json",
        drive_root / "PHASE4_BENCHMARK_COMPLETE.json",
        {
            "status": "PHASE4_BENCHMARK_COMPLETE",
            "selected_arm": selected_arm,
            "config_sha256": config.digest(),
        },
    )
    report = f"""# Phase 4 WiSig Representation Benchmark v1.1.0

Status: **PHASE4_BENCHMARK_COMPLETE**

- Train subset: {len(train_dataset):,} balanced samples (600 per known transmitter).
- P0 selection subset: {len(p0_dataset):,} balanced samples (150 per transmitter).
- Arms: A0 CE versus A1 CE + 0.1 SupCon.
- Seeds: 42, 123, 2026.
- Epoch budget: maximum 8; early stopping from epoch 4.
- Selection data: P0 Known only.
- Locked selected arm: **{selected_arm}**.
- P1/P2/P3 and Calibration Unknown were resolved only after selection was persisted.
- Strict Zero-Day remained structurally inaccessible.

This is an exploratory resource-bounded benchmark. It does not establish statistical
superiority, final teacher quality, strict-zero-day performance, XAI fidelity, or latency.
"""
    local_report = local_root / "reports" / "PHASE4_BENCHMARK_REPORT.md"
    atomic_text(local_report, report)
    persist_file_verified(local_report, drive_root / "reports" / local_report.name)

    stage_out_manifest = copy_tree_verified(local_root, drive_root)
    _persist_json(
        local_root / "STAGE_OUT_MANIFEST.json",
        drive_root / "STAGE_OUT_MANIFEST.json",
        stage_out_manifest,
    )
    return summary
