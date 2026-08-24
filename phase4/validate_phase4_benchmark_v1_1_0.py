#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VERSION = "1.1.0"
SOURCE_FILES = [
    "Phase4_WiSig_Teacher_Representation_Benchmark_v1_1_0.py",
    "phase4_benchmark_runtime_v1_1_0.py",
    "phase4_benchmark_training_v1_1_0.py",
    "phase4_benchmark_runner_v1_1_0.py",
    "phase4_benchmark_config_v1_1_0.json",
    "Phase4_WiSig_Benchmark_v1_1_0_Colab.ipynb",
    "SCIENTIFIC_PROTOCOL_V1_1_0.md",
    "README_V1_1_0.md",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    checks = {}
    for name in SOURCE_FILES:
        checks[f"file_present::{name}"] = (ROOT / name).is_file()

    python_files = [name for name in SOURCE_FILES if name.endswith(".py")]
    for name in python_files:
        ast.parse((ROOT / name).read_text(encoding="utf-8"))
    checks["python_ast_parse"] = True

    config = json.loads((ROOT / "phase4_benchmark_config_v1_1_0.json").read_text())
    notebook = json.loads((ROOT / "Phase4_WiSig_Benchmark_v1_1_0_Colab.ipynb").read_text())
    checks["config_json_parse"] = True
    checks["notebook_json_parse"] = notebook["nbformat"] == 4
    checks["maximum_epochs_eight"] = config["training"]["maximum_epochs"] == 8
    checks["balanced_train_count_58800"] = config["training"]["train_samples_total"] == 58_800
    checks["p0_count_14700"] = config["selection"]["samples_total"] == 14_700
    checks["maximum_optimizer_steps_11040"] = (
        config["training"]["maximum_optimizer_steps"] == 11_040
    )
    checks["arms_a0_a1_only"] = set(config["arms"]) == {"A0", "A1"}

    training_source = (ROOT / "phase4_benchmark_training_v1_1_0.py").read_text()
    forbidden_selection_roles = (
        "p1_cross_day",
        "p2_cross_receiver",
        "p3_cross_day_receiver",
        "calibration_unknown",
    )
    checks["training_module_has_no_shift_or_unknown_role"] = not any(
        role in training_source for role in forbidden_selection_roles
    )
    checks["p0_only_training_evaluation"] = "evaluate_known(model, p0_dataset" in training_source
    checks["component_losses_recorded"] = all(
        token in training_source
        for token in ("mean_ce_loss", "mean_supcon_loss", "mean_prototype_loss")
    )
    checks["complete_resume_state"] = all(
        token in training_source
        for token in (
            '"optimizer"',
            '"scheduler"',
            '"scaler"',
            '"rng_state"',
            '"history"',
            '"stale"',
            '"config_sha256"',
        )
    )

    runner_source = (ROOT / "phase4_benchmark_runner_v1_1_0.py").read_text()
    lock_position = runner_source.index('"OBJECTIVE_SELECTION_P0_ONLY.json"')
    shift_resolution_position = runner_source.index('for role in ("p1_cross_day"')
    checks["selection_persisted_before_shift_resolution"] = lock_position < shift_resolution_position
    checks["calibration_unknown_post_selection_only"] = (
        runner_source.index('"calibration_unknown", "unknown_diagnostic"') > lock_position
    )
    checks["strict_self_test_present"] = "strict_access_self_test" in runner_source

    main_source = (ROOT / "Phase4_WiSig_Teacher_Representation_Benchmark_v1_1_0.py").read_text()
    checks["failure_traceback_persistence"] = all(
        token in main_source for token in ("traceback.format_exc()", "PHASE4_FAILURE.json")
    )
    checks["drive_not_scientific_hot_path"] = "canonical-root cannot point to Drive" in main_source

    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"Phase 4 v1.1.0 static validation failed: {failed}")

    report = {
        "phase4_benchmark_version": VERSION,
        "validation_scope": (
            "Static source/config/notebook/protocol validation. CUDA execution and dataset-dependent "
            "tests must run in Colab T4."
        ),
        "checks": {name: "PASS" for name in sorted(checks)},
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "overall": "PASS",
    }
    output = ROOT / "DELIVERY_VALIDATION_V1_1_0.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
