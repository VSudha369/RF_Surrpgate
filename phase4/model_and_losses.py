#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 5, stride=stride, dilation=dilation)
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.Dropout(dropout),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(self.conv1(x)) + self.skip(x))


class WiSigRepresentationNet(nn.Module):
    """Stage-2.6M representation backbone, ported unchanged in structure to native V2 L=256."""

    def __init__(self, num_classes: int = 98, embedding_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.iq_mixer = nn.Sequential(
            nn.Conv1d(2, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            ConvNormAct(32, 64, 7, stride=2),
        )
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(64, 64, dilation=1, dropout=dropout),
            ResidualTemporalBlock(64, 128, stride=2, dilation=1, dropout=dropout),
            ResidualTemporalBlock(128, 128, dilation=2, dropout=dropout),
            ResidualTemporalBlock(128, 256, stride=2, dilation=1, dropout=dropout),
        )
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.temporal(self.iq_mixer(x))
        std = features.float().var(dim=-1, unbiased=False).add(1e-8).sqrt().to(features.dtype)
        pooled = torch.cat((features.mean(dim=-1), std), dim=1)
        embedding_raw = self.projection(pooled)
        embedding_normalized = F.normalize(embedding_raw.float(), dim=1, eps=1e-8)
        logits = self.classifier(embedding_raw)
        return {
            "logits": logits,
            "embedding_raw": embedding_raw,
            "embedding_normalized": embedding_normalized,
        }


EXPECTED_STAGE26_ARCHITECTURE_SIGNATURE = "d5ed7528ab93246c784fe12ed1bee90d4234753293e89bdf3d226db8f2fb5f9c"
EXPECTED_STAGE26_PARAMETER_COUNT = 849634

ARM_DEFINITIONS = {
    "A0": {"name": "CE", "supcon_weight": 0.0, "prototype_weight": 0.0},
    "A1": {"name": "CE + SupCon", "supcon_weight": 0.1, "prototype_weight": 0.0},
    "A2": {"name": "CE + Prototype", "supcon_weight": 0.0, "prototype_weight": 0.1},
    "A3": {"name": "CE + SupCon + Prototype", "supcon_weight": 0.1, "prototype_weight": 0.1},
}


class EMAPrototypeBank(nn.Module):
    def __init__(self, num_classes: int = 98, dim: int = 128, momentum: float = 0.95):
        super().__init__()
        self.momentum = float(momentum)
        self.register_buffer("prototypes", torch.zeros(num_classes, dim, dtype=torch.float32))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, z: torch.Tensor, y: torch.Tensor) -> None:
        z = F.normalize(z.detach().float(), dim=1, eps=1e-8)
        for cls in y.unique().tolist():
            mask = y == int(cls)
            centroid = F.normalize(z[mask].mean(dim=0, keepdim=True), dim=1, eps=1e-8)[0]
            if self.initialized[int(cls)]:
                p = self.momentum * self.prototypes[int(cls)] + (1 - self.momentum) * centroid
                self.prototypes[int(cls)] = F.normalize(p[None, :], dim=1, eps=1e-8)[0]
            else:
                self.prototypes[int(cls)] = centroid
                self.initialized[int(cls)] = True

    def targets(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z.float(), dim=1, eps=1e-8)
        out = torch.empty_like(z)
        for cls in y.unique().tolist():
            mask = y == int(cls)
            if self.initialized[int(cls)]:
                out[mask] = self.prototypes[int(cls)].detach()
            else:
                centroid = F.normalize(z[mask].detach().mean(dim=0, keepdim=True), dim=1, eps=1e-8)[0]
                out[mask] = centroid
        return out


def supervised_contrastive_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z = F.normalize(z.float(), dim=1, eps=1e-8)
    y = y.view(-1)
    sim = (z @ z.T) / float(temperature)
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = y[:, None].eq(y[None, :]) & ~eye
    logits_mask = ~eye
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_count = positive.sum(dim=1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return z.sum() * 0.0
    mean_log_prob = (log_prob * positive).sum(dim=1) / pos_count.clamp_min(1)
    loss = -mean_log_prob[valid].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite supervised contrastive loss")
    return loss


def prototype_compactness_loss(z: torch.Tensor, y: torch.Tensor, bank: EMAPrototypeBank) -> torch.Tensor:
    z = F.normalize(z.float(), dim=1, eps=1e-8)
    target = bank.targets(z, y)
    return (z.float() - target.float()).square().sum(dim=1).mean()


def objective_loss(
    arm: str,
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    bank: EMAPrototypeBank,
    temperature: float = 0.07,
    label_smoothing: float = 0.0,
):
    if arm not in ARM_DEFINITIONS:
        raise KeyError(arm)
    spec = ARM_DEFINITIONS[arm]
    ce = F.cross_entropy(outputs["logits"].float(), labels, label_smoothing=label_smoothing)
    sup = torch.zeros((), device=ce.device, dtype=torch.float32)
    proto = torch.zeros((), device=ce.device, dtype=torch.float32)
    if spec["supcon_weight"]:
        sup = supervised_contrastive_loss(outputs["embedding_normalized"], labels, temperature)
    if spec["prototype_weight"]:
        proto = prototype_compactness_loss(outputs["embedding_normalized"], labels, bank)
    total = ce + spec["supcon_weight"] * sup + spec["prototype_weight"] * proto
    if not torch.isfinite(total):
        raise FloatingPointError(f"Non-finite objective for {arm}")
    return total, {"ce": float(ce.detach()), "supcon": float(sup.detach()), "prototype": float(proto.detach())}


def apply_rf_augmentation(
    x: torch.Tensor,
    generator: torch.Generator,
    phase_rotation_radians: float = 0.12,
    amplitude_jitter: float = 0.05,
    awgn_std: float = 0.01,
    maximum_circular_shift: int = 4,
) -> torch.Tensor:
    batch, _, length = x.shape
    dtype, device = x.dtype, x.device
    phase = (torch.rand(batch, 1, 1, device=device, generator=generator, dtype=dtype) * 2 - 1) * phase_rotation_radians
    c, s = torch.cos(phase), torch.sin(phase)
    i, q = x[:, 0:1], x[:, 1:2]
    x = torch.cat((i * c - q * s, i * s + q * c), dim=1)
    scale = 1 + (torch.rand(batch, 1, 1, device=device, generator=generator, dtype=dtype) * 2 - 1) * amplitude_jitter
    x = x * scale
    if awgn_std > 0:
        rms = x.float().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8).to(dtype)
        noise = torch.randn(x.shape, device=device, generator=generator, dtype=dtype)
        x = x + noise * (awgn_std * rms)
    if maximum_circular_shift > 0:
        shifts = torch.randint(-maximum_circular_shift, maximum_circular_shift + 1, (batch,), device=device, generator=generator)
        base = torch.arange(length, device=device)[None, :]
        gather = (base - shifts[:, None]) % length
        x = x.gather(2, gather[:, None, :].expand(-1, 2, -1))
    return x


def architecture_signature(model: nn.Module) -> str:
    import hashlib, json
    rows = [(n, list(p.shape), str(p.dtype)) for n, p in model.state_dict().items()]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
