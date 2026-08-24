#!/usr/bin/env python3
from __future__ import annotations

import copy
import gc
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation import geometry_summary
from model_and_losses import (
    EXPECTED_STAGE26_ARCHITECTURE_SIGNATURE,
    EXPECTED_STAGE26_PARAMETER_COUNT,
    EMAPrototypeBank,
    WiSigRepresentationNet,
    apply_rf_augmentation,
    architecture_signature,
    objective_loss,
)
from phase4_common import atomic_json, load_json
from phase4_benchmark_runtime_v1_1_0 import (
    PHASE4_BENCHMARK_VERSION,
    BenchmarkConfig,
    atomic_torch_save,
    capture_rng_state,
    collect_outputs,
    evaluate_known,
    make_grad_scaler,
    persist_file_verified,
    restore_file_verified,
    restore_rng_state,
    seed_everything,
    sha256_file,
    worker_init_fn,
    write_csv,
)
from wisig_dataset import DomainBalancedTxBatchSampler


def _state_dict_cpu(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _persist_json(local_path: Path, drive_path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    atomic_json(local_path, payload)
    return persist_file_verified(local_path, drive_path)


def _make_runs(config: BenchmarkConfig, seed: int, device: torch.device):
    seed_everything(seed)
    base = WiSigRepresentationNet(config.num_classes, config.embedding_dim, config.dropout).to(device)
    base_state = copy.deepcopy(base.state_dict())
    signature = architecture_signature(base)
    parameter_count = sum(parameter.numel() for parameter in base.parameters())
    if signature != EXPECTED_STAGE26_ARCHITECTURE_SIGNATURE:
        raise RuntimeError(f"Architecture signature drift: {signature}")
    if parameter_count != EXPECTED_STAGE26_PARAMETER_COUNT:
        raise RuntimeError(f"Parameter-count drift: {parameter_count}")
    del base

    runs: Dict[str, Dict[str, Any]] = {}
    for arm in config.arms:
        model = WiSigRepresentationNet(config.num_classes, config.embedding_dim, config.dropout).to(device)
        model.load_state_dict(base_state)
        bank = EMAPrototypeBank(config.num_classes, config.embedding_dim, config.prototype_momentum).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_epochs, eta_min=config.learning_rate * 0.01
        )
        runs[arm] = {
            "model": model,
            "bank": bank,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": make_grad_scaler(config.amp_enabled),
            "best_p0": -math.inf,
            "best_epoch": 0,
            "best_model": None,
            "best_bank": None,
            "best_p0_metrics": None,
        }
    return runs, signature, parameter_count


def _build_resume_payload(
    seed: int,
    completed_epoch: int,
    stale: int,
    history,
    runs,
    signature: str,
    parameter_count: int,
    config: BenchmarkConfig,
):
    run_payload = {}
    for arm, run in runs.items():
        run_payload[arm] = {
            "model": _state_dict_cpu(run["model"]),
            "prototype_bank": _state_dict_cpu(run["bank"]),
            "optimizer": run["optimizer"].state_dict(),
            "scheduler": run["scheduler"].state_dict(),
            "scaler": run["scaler"].state_dict(),
            "best_p0": float(run["best_p0"]),
            "best_epoch": int(run["best_epoch"]),
            "best_model": run["best_model"],
            "best_bank": run["best_bank"],
            "best_p0_metrics": run["best_p0_metrics"],
        }
    return {
        "phase4_benchmark_version": PHASE4_BENCHMARK_VERSION,
        "config_sha256": config.digest(),
        "seed": int(seed),
        "completed_epoch": int(completed_epoch),
        "stale": int(stale),
        "history": history,
        "architecture_signature": signature,
        "parameter_count": int(parameter_count),
        "arms": list(config.arms),
        "runs": run_payload,
        "rng_state": capture_rng_state(),
    }


def _load_resume(bundle, runs, config: BenchmarkConfig, seed: int) -> tuple[int, int, list]:
    if bundle.get("phase4_benchmark_version") != PHASE4_BENCHMARK_VERSION:
        raise RuntimeError("Recovery version mismatch")
    if bundle.get("config_sha256") != config.digest():
        raise RuntimeError("Recovery configuration hash mismatch")
    if int(bundle.get("seed")) != int(seed):
        raise RuntimeError("Recovery seed mismatch")
    if tuple(bundle.get("arms", [])) != tuple(config.arms):
        raise RuntimeError("Recovery arm mismatch")
    for arm, run in runs.items():
        saved = bundle["runs"][arm]
        run["model"].load_state_dict(saved["model"])
        run["bank"].load_state_dict(saved["prototype_bank"])
        run["optimizer"].load_state_dict(saved["optimizer"])
        run["scheduler"].load_state_dict(saved["scheduler"])
        run["scaler"].load_state_dict(saved["scaler"])
        run["best_p0"] = float(saved["best_p0"])
        run["best_epoch"] = int(saved["best_epoch"])
        run["best_model"] = saved["best_model"]
        run["best_bank"] = saved["best_bank"]
        run["best_p0_metrics"] = saved["best_p0_metrics"]
    restore_rng_state(bundle["rng_state"])
    return int(bundle["completed_epoch"]) + 1, int(bundle["stale"]), list(bundle["history"])


def _restore_best_files_for_completed_seed(
    seed: int, local_root: Path, drive_root: Path, config: BenchmarkConfig
) -> Dict[str, Any] | None:
    drive_summary = drive_root / "metrics" / f"seed_{seed}_benchmark_summary.json"
    drive_complete = drive_root / "recovery" / f"seed_{seed}" / "SEED_COMPLETE.json"
    if not (drive_summary.is_file() and drive_complete.is_file()):
        return None
    completed = load_json(drive_complete)
    if completed.get("config_sha256") != config.digest():
        raise RuntimeError(f"Completed seed {seed} belongs to a different benchmark configuration")
    summary = load_json(drive_summary)
    for arm in config.arms:
        drive_checkpoint = drive_root / "checkpoints" / f"seed_{seed}" / arm / "best_p0.pt"
        local_checkpoint = local_root / "checkpoints" / f"seed_{seed}" / arm / "best_p0.pt"
        expected = summary["arms"][arm]["checkpoint_sha256"]
        restore_file_verified(drive_checkpoint, local_checkpoint, expected)
    print(f"[RESUME] seed={seed} already complete; verified best checkpoints restored")
    return summary


def train_seed(
    seed: int,
    train_dataset,
    p0_dataset,
    local_root: Path,
    drive_root: Path,
    config: BenchmarkConfig,
    device: torch.device,
) -> Dict[str, Any]:
    completed = _restore_best_files_for_completed_seed(seed, local_root, drive_root, config)
    if completed is not None:
        return completed

    runs, signature, parameter_count = _make_runs(config, seed, device)
    history = []
    stale = 0
    start_epoch = 1

    local_recovery = local_root / "recovery" / f"seed_{seed}" / "latest_resume.pt"
    drive_recovery = drive_root / "recovery" / f"seed_{seed}" / "latest_resume.pt"
    local_manifest = local_root / "recovery" / f"seed_{seed}" / "RESUME_MANIFEST.json"
    drive_manifest = drive_root / "recovery" / f"seed_{seed}" / "RESUME_MANIFEST.json"
    if drive_recovery.is_file() and drive_manifest.is_file():
        manifest = load_json(drive_manifest)
        restore_file_verified(drive_recovery, local_recovery, manifest["sha256"])
        bundle = torch.load(local_recovery, map_location="cpu", weights_only=False)
        start_epoch, stale, history = _load_resume(bundle, runs, config, seed)
        print(f"[RESUME] seed={seed} continuing from epoch={start_epoch}")

    sampler = DomainBalancedTxBatchSampler(
        train_dataset, config.batch_size, config.samples_per_tx, seed
    )
    for epoch in range(start_epoch, config.max_epochs + 1):
        batches = sampler.materialize(epoch)
        exposure = np.asarray([position for batch in batches for position in batch], dtype=np.int64)
        exposure_sha256 = hashlib.sha256(exposure.tobytes()).hexdigest()
        any_improvement = False
        epoch_rows = []

        for arm in config.arms:
            run = runs[arm]
            epoch_seed = seed * 1_000_003 + epoch * 10_007
            seed_everything(epoch_seed)
            augmentation_generator = torch.Generator(device=device).manual_seed(epoch_seed + 77)
            model, bank = run["model"], run["bank"]
            model.train()
            losses, ce_losses, supcon_losses, prototype_losses = [], [], [], []
            successful_steps = 0
            consecutive_overflows = 0
            loader = DataLoader(
                train_dataset,
                batch_sampler=batches,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=config.num_workers > 0,
                prefetch_factor=2 if config.num_workers > 0 else None,
                worker_init_fn=worker_init_fn,
            )
            for batch in loader:
                signals = batch["x"].to(device, non_blocking=True)
                labels = batch["y"].to(device, non_blocking=True)
                signals = apply_rf_augmentation(
                    signals,
                    augmentation_generator,
                    config.phase_rotation_radians,
                    config.amplitude_jitter,
                    config.awgn_std,
                    config.maximum_circular_shift,
                )
                run["optimizer"].zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.amp_enabled):
                    outputs = model(signals)
                loss, parts = objective_loss(
                    arm, outputs, labels, bank, config.temperature, config.label_smoothing
                )
                old_scale = float(run["scaler"].get_scale())
                run["scaler"].scale(loss).backward()
                run["scaler"].unscale_(run["optimizer"])
                gradients_finite = all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                )
                if gradients_finite:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                run["scaler"].step(run["optimizer"])
                run["scaler"].update()
                new_scale = float(run["scaler"].get_scale())
                step_ok = gradients_finite and new_scale >= old_scale
                if step_ok:
                    bank.update(outputs["embedding_normalized"], labels)
                    successful_steps += 1
                    consecutive_overflows = 0
                else:
                    consecutive_overflows += 1
                    if consecutive_overflows > config.max_consecutive_amp_overflows:
                        raise FloatingPointError(
                            f"Too many consecutive AMP overflows: seed={seed} arm={arm}"
                        )
                losses.append(float(loss.detach()))
                ce_losses.append(parts["ce"])
                supcon_losses.append(parts["supcon"])
                prototype_losses.append(parts["prototype"])
            del loader
            if successful_steps == 0:
                raise RuntimeError(f"No successful optimizer step: seed={seed} arm={arm} epoch={epoch}")
            run["scheduler"].step()

            # P0 is the only role resolved/evaluated during model selection.
            p0_metrics = evaluate_known(model, p0_dataset, device, config)
            p0_f1 = float(p0_metrics["fixed98_macro_f1"])
            improved = p0_f1 > run["best_p0"] + 1e-8
            if improved:
                run["best_p0"] = p0_f1
                run["best_epoch"] = epoch
                run["best_model"] = _state_dict_cpu(model)
                run["best_bank"] = _state_dict_cpu(bank)
                run["best_p0_metrics"] = p0_metrics
                any_improvement = True

            row = {
                "seed": int(seed),
                "epoch": int(epoch),
                "arm": arm,
                "exposure_sha256": exposure_sha256,
                "successful_optimizer_steps": int(successful_steps),
                "mean_total_loss": float(np.mean(losses)),
                "mean_ce_loss": float(np.mean(ce_losses)),
                "mean_supcon_loss": float(np.mean(supcon_losses)),
                "mean_prototype_loss": float(np.mean(prototype_losses)),
                "p0_fixed98_macro_f1": p0_f1,
                "p0_accuracy": float(p0_metrics["accuracy"]),
                "p0_top5_accuracy": float(p0_metrics["top5_accuracy"]),
                "learning_rate": float(run["optimizer"].param_groups[0]["lr"]),
                "improved": bool(improved),
            }
            history.append(row)
            epoch_rows.append(row)
            print(
                f"[EPOCH] seed={seed} epoch={epoch}/{config.max_epochs} arm={arm} "
                f"P0-F1={100*p0_f1:.3f}% loss={row['mean_total_loss']:.4f}"
            )
            gc.collect()

        if len({row["exposure_sha256"] for row in epoch_rows}) != 1:
            raise RuntimeError("Matched-arm exposure hash diverged")
        stale = 0 if any_improvement else stale + 1

        history_path = local_root / "metrics" / f"seed_{seed}_training_history.csv"
        write_csv(history_path, history)
        persist_file_verified(
            history_path, drive_root / "metrics" / f"seed_{seed}_training_history.csv"
        )

        # Commit new best checkpoints only after both arms finish the epoch.
        for arm, run in runs.items():
            if run["best_model"] is None:
                raise RuntimeError(f"Best P0 state missing for seed={seed} arm={arm}")
            checkpoint = {
                "phase4_benchmark_version": PHASE4_BENCHMARK_VERSION,
                "config_sha256": config.digest(),
                "seed": int(seed),
                "arm": arm,
                "epoch": int(run["best_epoch"]),
                "architecture_signature": signature,
                "parameter_count": int(parameter_count),
                "model": run["best_model"],
                "prototype_bank": run["best_bank"],
                "p0_metrics": run["best_p0_metrics"],
            }
            local_best = local_root / "checkpoints" / f"seed_{seed}" / arm / "best_p0.pt"
            drive_best = drive_root / "checkpoints" / f"seed_{seed}" / arm / "best_p0.pt"
            atomic_torch_save(local_best, checkpoint)
            persist_file_verified(local_best, drive_best)

        resume_payload = _build_resume_payload(
            seed, epoch, stale, history, runs, signature, parameter_count, config
        )
        atomic_torch_save(local_recovery, resume_payload)
        recovery_evidence = persist_file_verified(local_recovery, drive_recovery)
        manifest_payload = {
            "phase4_benchmark_version": PHASE4_BENCHMARK_VERSION,
            "config_sha256": config.digest(),
            "seed": int(seed),
            "completed_epoch": int(epoch),
            "sha256": recovery_evidence["sha256"],
            "bytes": recovery_evidence["bytes"],
            "next_epoch": int(epoch + 1),
        }
        _persist_json(local_manifest, drive_manifest, manifest_payload)

        if epoch >= config.minimum_epochs and stale >= config.early_stopping_patience:
            print(f"[EARLY STOP] seed={seed} epoch={epoch} stale={stale}")
            break

    arm_summaries: Dict[str, Any] = {}
    for arm, run in runs.items():
        local_best = local_root / "checkpoints" / f"seed_{seed}" / arm / "best_p0.pt"
        checkpoint = torch.load(local_best, map_location=device, weights_only=False)
        run["model"].load_state_dict(checkpoint["model"])
        p0_outputs = collect_outputs(run["model"], p0_dataset, device, config, include_logits=True)
        geometry = geometry_summary(p0_outputs["embedding"], p0_outputs["y"], config.num_classes)
        arm_summaries[arm] = {
            "best_epoch": int(checkpoint["epoch"]),
            "p0_metrics": checkpoint["p0_metrics"],
            "p0_geometry": geometry,
            "checkpoint_sha256": sha256_file(local_best),
        }
        del p0_outputs

    seed_summary = {
        "phase4_benchmark_version": PHASE4_BENCHMARK_VERSION,
        "config_sha256": config.digest(),
        "seed": int(seed),
        "architecture_signature": signature,
        "parameter_count": int(parameter_count),
        "epochs_completed": int(max(row["epoch"] for row in history)),
        "selection_role": "p0_known_only",
        "arms": arm_summaries,
    }
    local_summary = local_root / "metrics" / f"seed_{seed}_benchmark_summary.json"
    drive_summary = drive_root / "metrics" / f"seed_{seed}_benchmark_summary.json"
    _persist_json(local_summary, drive_summary, seed_summary)
    complete_payload = {
        "status": "SEED_COMPLETE",
        "seed": int(seed),
        "config_sha256": config.digest(),
        "epochs_completed": seed_summary["epochs_completed"],
    }
    _persist_json(
        local_root / "recovery" / f"seed_{seed}" / "SEED_COMPLETE.json",
        drive_root / "recovery" / f"seed_{seed}" / "SEED_COMPLETE.json",
        complete_payload,
    )
    torch.cuda.empty_cache()
    return seed_summary
