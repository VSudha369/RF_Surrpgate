#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
from scipy import stats
import torch

from access_guard import TRAIN_ROLES, strict_access_self_test, verify_phase3_contract
from model_and_losses import ARM_DEFINITIONS
from phase4_common import atomic_json, atomic_text, utc_now
from prestudy_runtime import PreStudyConfig, build_known_label_mapping, cuda_preflight, make_dataset
from prestudy_training import train_seed
from wisig_dataset import CanonicalWiSigStore

def paired_bootstrap(diffs: np.ndarray, iterations: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        vals[i] = rng.choice(diffs, size=len(diffs), replace=True).mean()
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def benjamini_hochberg(pvals: Mapping[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running = 1.0
    for rank_rev, (name, p) in enumerate(reversed(items), start=1):
        rank = m - rank_rev + 1
        running = min(running, float(p) * m / rank)
        adj[name] = running
    return adj


def aggregate_and_select(seed_outputs: Mapping[int, Dict[str, Any]], config: PreStudyConfig) -> Dict[str, Any]:
    seeds = sorted(seed_outputs)
    summary = {}
    for arm in ARM_DEFINITIONS:
        rows = [seed_outputs[s]["results"][arm] for s in seeds]
        summary[arm] = {
            "arm_name": ARM_DEFINITIONS[arm]["name"],
            "mean_primary": float(np.mean([r["primary_macro_p0_p3_fixed98_f1"] for r in rows])),
            "mean_p0_fisher": float(np.mean([r["p0_geometry"]["fisher_ratio"] for r in rows])),
            "mean_domain_abs_from_0_5": float(np.mean([r["mean_abs_domain_auc_from_0_5"] for r in rows])),
            "mean_calibration_unknown_auroc": float(np.mean([r["calibration_unknown_auroc_aggregate"] for r in rows])),
            "per_seed_primary": [float(r["primary_macro_p0_p3_fixed98_f1"]) for r in rows],
        }
    ranking = sorted(
        ARM_DEFINITIONS,
        key=lambda a: (
            -summary[a]["mean_primary"],
            -summary[a]["mean_p0_fisher"],
            summary[a]["mean_domain_abs_from_0_5"],
            -summary[a]["mean_calibration_unknown_auroc"],
        ),
    )
    top = ranking[0]
    comparisons = {}
    pvals = {}
    for arm in ("A1", "A2", "A3"):
        a = np.asarray(summary[arm]["per_seed_primary"])
        b = np.asarray(summary["A0"]["per_seed_primary"])
        diff = a - b
        ci = paired_bootstrap(diff, config.bootstrap_iterations, 100 + int(arm[-1]))
        try:
            wil = stats.wilcoxon(diff, alternative="two-sided")
            p = float(wil.pvalue)
        except ValueError:
            p = 1.0
        dz = float(diff.mean() / diff.std(ddof=1)) if len(diff) > 1 and diff.std(ddof=1) > 0 else 0.0
        comparisons[f"{arm}_vs_A0"] = {
            "mean_difference": float(diff.mean()), "bootstrap_ci95": list(ci), "wilcoxon_p": p, "cohen_dz": dz,
        }
        pvals[f"{arm}_vs_A0"] = p
    adj = benjamini_hochberg(pvals)
    for k, q in adj.items():
        comparisons[k]["bh_fdr_q"] = q

    decision = "NO_OBJECTIVE_CLEARLY_SUPERIOR"
    recommended = "A0"
    if top != "A0":
        cmp = comparisons[f"{top}_vs_A0"]
        fisher_ok = summary[top]["mean_p0_fisher"] >= summary["A0"]["mean_p0_fisher"]
        if (
            cmp["mean_difference"] >= config.practical_f1_delta
            and cmp["bootstrap_ci95"][0] >= -config.noninferiority_f1_delta
            and fisher_ok
        ):
            decision = {"A1":"SELECT_CE_SUPCON","A2":"SELECT_CE_PROTOTYPE","A3":"SELECT_CE_SUPCON_PROTOTYPE"}[top]
            recommended = top
    return {
        "arm_summary": summary,
        "lexicographic_ranking": ranking,
        "ranking_top_arm": top,
        "paired_statistics": comparisons,
        "decision": decision,
        "recommended_objective_arm_for_phase5": recommended,
        "claims_boundary": "Representation objective pre-study only; not final teacher, strict zero-day, XAI, fidelity, or latency evidence.",
    }


def run_prestudy(canonical_root: Path, output_root: Path, config: PreStudyConfig) -> Dict[str, Any]:
    config.apply_profile()
    output_root.mkdir(parents=True, exist_ok=True)
    for sub in ("configs", "checkpoints", "metrics", "embeddings", "manifests", "reports"):
        (output_root / sub).mkdir(exist_ok=True)

    access = verify_phase3_contract()
    strict_test = strict_access_self_test()
    gpu = cuda_preflight(config)
    store = CanonicalWiSigStore(canonical_root)
    mapping, mapping_evidence = build_known_label_mapping(store)
    atomic_json(output_root / "configs" / "CONFIG.json", asdict(config))
    atomic_json(output_root / "manifests" / "ACCESS_CONTRACT.json", {**access, **strict_test})
    atomic_json(output_root / "manifests" / "KNOWN_LABEL_MAPPING.json", mapping_evidence)
    atomic_json(output_root / "manifests" / "GPU_RUNTIME.json", gpu)

    datasets = {}
    split_evidence = {}
    purposes = {
        "train_known": "gradient",
        "p0_known": "model_selection",
        "p1_cross_day": "known_eval",
        "p2_cross_receiver": "known_eval",
        "p3_cross_day_receiver": "known_eval",
        "calibration_unknown": "unknown_diagnostic",
    }
    for role, purpose in purposes.items():
        datasets[role], split_evidence[role] = make_dataset(store, role, purpose, mapping, config.profile, 777)
    atomic_json(output_root / "manifests" / "SPLIT_EVIDENCE.json", split_evidence)

    # Hard gradient-bearing partition audit.
    if set(TRAIN_ROLES) != {"train_known"}:
        raise RuntimeError("Unexpected Phase-4 gradient role contract")
    for role in datasets:
        if role != "train_known" and purposes[role] == "gradient":
            raise RuntimeError("Non-train role marked gradient-bearing")

    seed_outputs = {}
    for seed in config.seeds:
        print(f"[RUN] Phase 4 representation pre-study seed={seed}")
        seed_outputs[int(seed)] = train_seed(int(seed), datasets, output_root, config, torch.device("cuda"))
        atomic_json(output_root / "metrics" / f"seed_{seed}_summary.json", seed_outputs[int(seed)])
        print(f"[OK] seed={seed} completed")

    selection = aggregate_and_select(seed_outputs, config)
    atomic_json(output_root / "metrics" / "OBJECTIVE_SELECTION.json", selection)

    # Structural strict-data audit: no strict arrays/checkpoints/embeddings may exist.
    forbidden_files = []
    for p in output_root.rglob("*"):
        if p.is_file() and (p.suffix in {".npy", ".npz", ".pt", ".pth", ".csv"} and "strict" in p.name.lower()):
            forbidden_files.append(str(p.relative_to(output_root)))
    if forbidden_files:
        raise RuntimeError(f"Strict-data artifact leakage detected: {forbidden_files}")
    strict_audit = {
        "strict_role_resolver_available": False,
        "strict_indices_loaded": False,
        "strict_signal_accessed": False,
        "strict_artifacts_found": forbidden_files,
        "strict_access_self_test": strict_test,
    }
    atomic_json(output_root / "manifests" / "STRICT_ACCESS_AUDIT.json", strict_audit)

    status = "PHASE4_COMPLETE" if config.profile == "full" else "PHASE4_PILOT_COMPLETE"
    summary = {
        "phase": "PHASE4_WISIG_TEACHER_REPRESENTATION_PRESTUDY",
        "phase_version": "1.0.0",
        "profile": config.profile,
        "status": status,
        "phase3_run_id": access["phase3_run_id"],
        "source_sha256": access["source_sha256"],
        "canonical_root": str(canonical_root),
        "seeds": list(config.seeds),
        "arms": ARM_DEFINITIONS,
        "selection": selection,
        "strict_access": strict_audit,
        "completed_utc": utc_now(),
    }
    atomic_json(output_root / "PHASE4_SUMMARY.json", summary)
    atomic_json(output_root / f"{status}.json", {"status": status, "decision": selection["decision"], "recommended_arm": selection["recommended_objective_arm_for_phase5"]})
    report = f"""# Phase 4 WiSig Teacher Representation Pre-Study\n\nStatus: **{status}**\n\nThis is a controlled representation-objective pre-study, not final teacher selection.\n\n- Gradient-bearing partition: Train Known only.\n- P0: model selection; P1/P2/P3: known-shift evaluation.\n- Calibration Unknown: threshold-free diagnostic only; no gradients.\n- Strict Zero-Day: structurally inaccessible and not evaluated.\n- Decision: `{selection['decision']}`\n- Recommended Phase-5 objective arm: `{selection['recommended_objective_arm_for_phase5']}`\n\nNo claim about strict zero-day performance, XAI quality, surrogate fidelity, or deployment latency is made here.\n"""
    atomic_text(output_root / "reports" / "PHASE4_PRESTUDY_REPORT.md", report)
    return summary
