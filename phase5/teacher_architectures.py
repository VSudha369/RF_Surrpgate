#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 98
EMBEDDING_DIM = 128


def _groups(channels: int, maximum: int = 16) -> int:
    g = min(maximum, channels)
    while channels % g:
        g -= 1
    return g


class ConvGNAct(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(_groups(cout), cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ConvGNAct(cin, cout, 5, stride=stride, dilation=dilation)
        self.conv2 = nn.Sequential(
            nn.Conv1d(cout, cout, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(_groups(cout), cout),
            nn.Dropout(dropout),
        )
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv1d(cin, cout, 1, stride=stride, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


def _mean_std_pool(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1)
    std = x.float().var(dim=-1, unbiased=False).add(1e-8).sqrt().to(x.dtype)
    return torch.cat([mean, std], dim=1)


class _Head(nn.Module):
    def __init__(self, in_features: int, embedding_dim: int = EMBEDDING_DIM, num_classes: int = NUM_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.proj(pooled)
        norm = F.normalize(raw.float(), dim=1, eps=1e-8)
        return {"logits": self.classifier(raw), "embedding_raw": raw, "embedding_normalized": norm}


class ResidualCNNTeacher(nn.Module):
    """R0: residual temporal CNN with global mean/std pooling."""
    def __init__(self, num_classes: int = NUM_CLASSES, embedding_dim: int = EMBEDDING_DIM, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(ConvGNAct(2, 64, 9, stride=2), ResidualBlock1D(64, 64, dropout=dropout))
        self.body = nn.Sequential(
            ResidualBlock1D(64, 128, stride=2, dropout=dropout),
            ResidualBlock1D(128, 128, dilation=2, dropout=dropout),
            ResidualBlock1D(128, 256, stride=2, dropout=dropout),
            ResidualBlock1D(256, 256, dilation=2, dropout=dropout),
        )
        self.head = _Head(512, embedding_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.head(_mean_std_pool(self.body(self.stem(x))))


class DepthwiseTCNBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.pre = nn.Conv1d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.depth = nn.Conv1d(cout, cout, 5, padding=2 * dilation, dilation=dilation, groups=cout, bias=False)
        self.point = nn.Conv1d(cout, cout, 1, bias=False)
        self.norm = nn.GroupNorm(_groups(cout), cout)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.pre(x)
        y = self.point(self.depth(skip))
        y = self.dropout(self.norm(y))
        return self.act(skip + y)


class DilatedTCNTeacher(nn.Module):
    """T0: multi-scale dilated depthwise-separable TCN."""
    def __init__(self, num_classes: int = NUM_CLASSES, embedding_dim: int = EMBEDDING_DIM, dropout: float = 0.1):
        super().__init__()
        self.stem = ConvGNAct(2, 128, 9, stride=2)
        specs = [(128,1),(160,2),(192,4),(224,8),(256,16),(256,32)]
        blocks = []
        cin = 128
        for cout, dilation in specs:
            blocks.append(DepthwiseTCNBlock(cin, cout, dilation=dilation, dropout=dropout))
            cin = cout
        self.body = nn.Sequential(*blocks)
        self.head = _Head(512, embedding_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.head(_mean_std_pool(self.body(self.stem(x))))


class CompactTransformerTeacher(nn.Module):
    """X0: convolutional tokenization + compact Transformer encoder."""
    def __init__(self, num_classes: int = NUM_CLASSES, embedding_dim: int = EMBEDDING_DIM, dropout: float = 0.1):
        super().__init__()
        d_model = 128
        self.patch = nn.Sequential(
            nn.Conv1d(2, d_model, 9, stride=4, padding=4, bias=False),
            nn.GroupNorm(_groups(d_model), d_model),
            nn.SiLU(inplace=True),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=384,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = _Head(d_model * 2, embedding_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.patch(x).transpose(1, 2)
        tokens = self.final_norm(self.encoder(tokens))
        mean = tokens.mean(dim=1)
        std = tokens.float().var(dim=1, unbiased=False).add(1e-8).sqrt().to(tokens.dtype)
        return self.head(torch.cat([mean, std], dim=1))


class SpectralBranch(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(2, 64, 9, stride=2),
            ResidualBlock1D(64, 128, stride=2, dropout=dropout),
            ResidualBlock1D(128, 192, stride=2, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full complex spectrum; no legacy 65-bin truncation.
        z = torch.complex(x[:, 0].float(), x[:, 1].float())
        spec = torch.fft.fftshift(torch.fft.fft(z, dim=-1, norm="ortho"), dim=-1)
        feat = torch.stack([spec.real, spec.imag], dim=1).to(x.dtype)
        return self.net(feat)


class TimeFrequencyTeacher(nn.Module):
    """F0: native full-spectrum dual branch over time IQ and complex FFT IQ."""
    def __init__(self, num_classes: int = NUM_CLASSES, embedding_dim: int = EMBEDDING_DIM, dropout: float = 0.1):
        super().__init__()
        self.time = nn.Sequential(
            ConvGNAct(2, 64, 9, stride=2),
            ResidualBlock1D(64, 128, stride=2, dropout=dropout),
            ResidualBlock1D(128, 192, stride=2, dropout=dropout),
        )
        self.freq = SpectralBranch(dropout=dropout)
        self.head = _Head(768, embedding_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        time = _mean_std_pool(self.time(x))
        freq = _mean_std_pool(self.freq(x))
        return self.head(torch.cat([time, freq], dim=1))


ARCHITECTURE_REGISTRY = {
    "R0_RESNET": ResidualCNNTeacher,
    "T0_TCN": DilatedTCNTeacher,
    "X0_TRANSFORMER": CompactTransformerTeacher,
    "F0_TIME_FREQ": TimeFrequencyTeacher,
}


def build_teacher(name: str, num_classes: int = NUM_CLASSES, embedding_dim: int = EMBEDDING_DIM, dropout: float = 0.1) -> nn.Module:
    if name not in ARCHITECTURE_REGISTRY:
        raise KeyError(f"Unknown teacher architecture: {name}")
    return ARCHITECTURE_REGISTRY[name](num_classes=num_classes, embedding_dim=embedding_dim, dropout=dropout)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
