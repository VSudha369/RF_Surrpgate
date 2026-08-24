#!/usr/bin/env python3
from __future__ import annotations

import copy, gc, hashlib, math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation import (binary_attribute_probe_auc, calibration_unknown_diagnostic, domain_probe_auc, fit_novelty_geometry, fixed_frame_metrics, geometry_summary, novelty_scores)
from model_and_losses import (ARM_DEFINITIONS, EXPECTED_STAGE26_ARCHITECTURE_SIGNATURE, EXPECTED_STAGE26_PARAMETER_COUNT, EMAPrototypeBank, WiSigRepresentationNet, apply_rf_augmentation, architecture_signature, objective_loss)
from phase4_common import atomic_json
from prestudy_runtime import PreStudyConfig, collect_outputs, deterministic_subset_positions, evaluate_known, make_grad_scaler, seed_everything, worker_init_fn, write_csv
from wisig_dataset import DomainBalancedTxBatchSampler, IndexedWiSigDataset

def train_seed(seed: int, datasets: Mapping[str, IndexedWiSigDataset], output_root: Path, config: PreStudyConfig, device: torch.device) -> Dict[str, Any]:
    seed_everything(seed)
    base = WiSigRepresentationNet(config.num_classes, config.embedding_dim, config.dropout).to(device)
    base_state = copy.deepcopy(base.state_dict())
    signature = architecture_signature(base)
    parameter_count = sum(p.numel() for p in base.parameters())
    if signature != EXPECTED_STAGE26_ARCHITECTURE_SIGNATURE:
        raise RuntimeError(f"Stage-2.6M architecture signature drift: {signature}")
    if parameter_count != EXPECTED_STAGE26_PARAMETER_COUNT:
        raise RuntimeError(f"Stage-2.6M parameter-count drift: {parameter_count}")
    del base

    runs = {}
    for arm in ARM_DEFINITIONS:
        model = WiSigRepresentationNet(config.num_classes, config.embedding_dim, config.dropout).to(device)
        model.load_state_dict(base_state)
        bank = EMAPrototypeBank(config.num_classes, config.embedding_dim, config.prototype_momentum).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs, eta_min=config.learning_rate * 0.01)
        scaler = make_grad_scaler(config.amp_enabled)
        runs[arm] = {"model": model, "bank": bank, "optimizer": optimizer, "scheduler": scheduler, "scaler": scaler, "best": -math.inf, "best_epoch": 0}

    train_ds = datasets["train_known"]
    history = []
    stale = 0
    sampler = DomainBalancedTxBatchSampler(train_ds, config.batch_size, config.samples_per_tx, seed)

    for epoch in range(1, config.max_epochs + 1):
        batches = sampler.materialize(epoch)
        exposure = np.asarray([p for b in batches for p in b], dtype=np.int64)
        exposure_sha = hashlib.sha256(exposure.tobytes()).hexdigest()
        any_improvement = False
        for arm, run in runs.items():
            # Same-seed/epoch stochastic stream across all objective arms.
            epoch_seed = seed * 1_000_003 + epoch * 10_007
            seed_everything(epoch_seed)
            aug_gen = torch.Generator(device=device).manual_seed(epoch_seed + 77)
            model, bank = run["model"], run["bank"]
            model.train()
            losses = []
            successful_steps = 0
            consecutive_overflows = 0
            loader = DataLoader(
                train_ds,
                batch_sampler=batches,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=config.num_workers > 0,
                prefetch_factor=2 if config.num_workers > 0 else None,
                worker_init_fn=worker_init_fn,
            )
            for batch in loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                x = apply_rf_augmentation(
                    x, aug_gen,
                    config.phase_rotation_radians,
                    config.amplitude_jitter,
                    config.awgn_std,
                    config.maximum_circular_shift,
                )
                run["optimizer"].zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.amp_enabled):
                    outputs = model(x)
                loss, parts = objective_loss(arm, outputs, y, bank, config.temperature, config.label_smoothing)
                old_scale = float(run["scaler"].get_scale())
                run["scaler"].scale(loss).backward()
                run["scaler"].unscale_(run["optimizer"])
                grad_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                if grad_finite:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                run["scaler"].step(run["optimizer"])
                run["scaler"].update()
                new_scale = float(run["scaler"].get_scale())
                step_ok = grad_finite and new_scale >= old_scale
                if step_ok:
                    bank.update(outputs["embedding_normalized"], y)
                    successful_steps += 1
                    consecutive_overflows = 0
                else:
                    consecutive_overflows += 1
                    if consecutive_overflows > config.max_consecutive_amp_overflows:
                        raise FloatingPointError(f"Too many consecutive AMP overflows for {arm} seed={seed}")
                losses.append(float(loss.detach()))
            if successful_steps == 0:
                raise RuntimeError(f"Epoch had no successful optimizer steps: {arm} seed={seed} epoch={epoch}")
            run["scheduler"].step()

            known = {}
            for role in ("p0_known", "p1_cross_day", "p2_cross_receiver", "p3_cross_day_receiver"):
                known[role] = evaluate_known(model, datasets[role], device, config)
            primary = float(np.mean([known[r]["fixed98_macro_f1"] for r in known]))
            worst = float(np.min([known[r]["fixed98_macro_f1"] for r in known]))
            improved = primary > run["best"] + 1e-8
            if improved:
                run["best"] = primary
                run["best_epoch"] = epoch
                any_improvement = True
                ckpt = output_root / "checkpoints" / f"seed_{seed}" / arm / "best_primary.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "phase4_version": "1.0.0", "seed": seed, "arm": arm, "epoch": epoch,
                    "architecture_signature": signature, "model": model.state_dict(), "prototype_bank": bank.state_dict(),
                    "primary_macro_p0_p3_fixed98_f1": primary, "known_metrics": known,
                }, ckpt)
            history.append({
                "seed": seed, "epoch": epoch, "arm": arm,
                "mean_train_loss": float(np.mean(losses)), "successful_optimizer_steps": successful_steps,
                "primary_macro_p0_p3_fixed98_f1": primary, "worst_protocol_fixed98_f1": worst,
                "exposure_sha256": exposure_sha,
                "learning_rate": float(run["optimizer"].param_groups[0]["lr"]),
                **{f"{r}_f1": known[r]["fixed98_macro_f1"] for r in known},
            })
            # Last checkpoint is local recovery evidence.
            last = output_root / "checkpoints" / f"seed_{seed}" / arm / "last.pt"
            last.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"phase4_version":"1.0.0","seed":seed,"arm":arm,"epoch":epoch,"model":model.state_dict(),"prototype_bank":bank.state_dict()}, last)
            del loader
            gc.collect()
        # Causal-control assertion: all arms have the same epoch exposure hash.
        rows = [r for r in history if r["seed"] == seed and r["epoch"] == epoch]
        if len({r["exposure_sha256"] for r in rows}) != 1:
            raise RuntimeError("Matched-arm exposure hash diverged")
        stale = 0 if any_improvement else stale + 1
        write_csv(output_root / "metrics" / "training_history.csv", history)
        if epoch >= config.minimum_epochs and stale >= config.early_stopping_patience:
            break

    # Downstream diagnostics use best-primary checkpoint for each arm.
    seed_results = {}
    for arm, run in runs.items():
        ckpt_path = output_root / "checkpoints" / f"seed_{seed}" / arm / "best_primary.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        run["model"].load_state_dict(ckpt["model"])
        model = run["model"]
        role_outputs = {}
        for role in ("p0_known", "p1_cross_day", "p2_cross_receiver", "p3_cross_day_receiver"):
            role_outputs[role] = collect_outputs(model, datasets[role], device, config, include_logits=True)
        # Train geometry is fitted on deterministic balanced subset, never on Calibration Unknown.
        train_ds_analysis = datasets["train_known"]
        train_per_class = max(1, config.covariance_fit_limit // config.num_classes)
        tpos = deterministic_subset_positions(train_ds_analysis.known_class, train_per_class, seed + 800)
        train_sub = IndexedWiSigDataset(train_ds_analysis.store, train_ds_analysis.indices[tpos], train_ds_analysis.source_to_known, False)
        train_out = collect_outputs(model, train_sub, device, config, include_logits=True)
        cal_out = collect_outputs(model, datasets["calibration_unknown"], device, config, include_logits=True)

        p0 = role_outputs["p0_known"]
        geom = geometry_summary(p0["embedding"], p0["y"], config.num_classes)
        domain = {
            "p0_vs_p1_day_auc": domain_probe_auc(p0["embedding"], p0["y"], role_outputs["p1_cross_day"]["embedding"], role_outputs["p1_cross_day"]["y"], config.domain_sample_per_tx, seed + 1),
            "p0_vs_p2_receiver_auc": domain_probe_auc(p0["embedding"], p0["y"], role_outputs["p2_cross_receiver"]["embedding"], role_outputs["p2_cross_receiver"]["y"], config.domain_sample_per_tx, seed + 2),
            "p0_vs_p3_combined_auc": domain_probe_auc(p0["embedding"], p0["y"], role_outputs["p3_cross_day_receiver"]["embedding"], role_outputs["p3_cross_day_receiver"]["y"], config.domain_sample_per_tx, seed + 3),
            "p0_equalization_auc": binary_attribute_probe_auc(p0["embedding"], p0["equalization"], p0["y"], config.domain_sample_per_tx, seed + 4),
        }
        cent, ncent, precision, shrinkage = fit_novelty_geometry(train_out["embedding"], train_out["y"], config.num_classes, config.covariance_fit_limit, seed + 5)
        p0_scores = novelty_scores(p0["embedding"], p0["logits"], cent, ncent, precision)
        cal_scores = novelty_scores(cal_out["embedding"], cal_out["logits"], cent, ncent, precision)
        cal_diag = calibration_unknown_diagnostic(p0_scores, cal_scores)
        known_metrics = {role: fixed_frame_metrics(o["y"], o["logits"].argmax(axis=1), o["logits"], config.num_classes) for role, o in role_outputs.items()}
        primary = float(np.mean([known_metrics[r]["fixed98_macro_f1"] for r in known_metrics]))
        domain_abs = float(np.mean([abs(v - 0.5) for v in domain.values() if np.isfinite(v)]))
        cal_agg = float(np.mean([v["auroc"] for v in cal_diag.values()]))
        result = {
            "seed": seed, "arm": arm, "best_epoch": int(ckpt["epoch"]),
            "primary_macro_p0_p3_fixed98_f1": primary,
            "known_metrics": known_metrics,
            "p0_geometry": geom,
            "domain_probe_auroc": domain,
            "mean_abs_domain_auc_from_0_5": domain_abs,
            "calibration_unknown": cal_diag,
            "calibration_unknown_auroc_aggregate": cal_agg,
            "novelty_covariance_shrinkage": shrinkage,
            "strict_data_accessed": False,
        }
        seed_results[arm] = result
        atomic_json(output_root / "metrics" / f"seed_{seed}_{arm}_downstream.json", result)
        # Persist only deterministic small embedding samples, not full stores.
        sample_n = min(5000, len(p0["embedding"]))
        sample_pos = np.linspace(0, len(p0["embedding"]) - 1, sample_n, dtype=np.int64)
        np.savez_compressed(
            output_root / "embeddings" / f"seed_{seed}_{arm}_p0_sample.npz",
            embedding=p0["embedding"][sample_pos].astype(np.float16),
            label=p0["y"][sample_pos].astype(np.int16),
            global_index=p0["global_index"][sample_pos].astype(np.int64),
        )
        del role_outputs, train_out, cal_out, p0_scores, cal_scores
        gc.collect()
        torch.cuda.empty_cache()
    return {"architecture_signature": signature, "parameter_count": EXPECTED_STAGE26_PARAMETER_COUNT, "results": seed_results, "epochs_completed": max(r["epoch"] for r in history if r["seed"] == seed)}
