#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def fixed_frame_metrics(y_true: np.ndarray, y_pred: np.ndarray, logits: np.ndarray, num_classes: int = 98) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(conf, (y_true, y_pred), 1)
    tp = np.diag(conf).astype(np.float64)
    support = conf.sum(axis=1).astype(np.float64)
    pred_support = conf.sum(axis=0).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_support, out=np.zeros_like(tp), where=pred_support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    acc = float(tp.sum() / max(1, conf.sum()))
    top5 = np.argsort(logits, axis=1)[:, -5:]
    top5_acc = float(np.mean(np.any(top5 == y_true[:, None], axis=1)))
    # Stable CE from logits.
    z = logits.astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    logsum = np.log(np.exp(z).sum(axis=1))
    ce = float(np.mean(logsum - z[np.arange(len(z)), y_true]))
    return {
        "accuracy": acc,
        "top5_accuracy": top5_acc,
        "cross_entropy": ce,
        "fixed98_macro_f1": float(f1.mean()),
        "fixed98_balanced_accuracy": float(recall.mean()),
    }


def geometry_summary(embedding: np.ndarray, labels: np.ndarray, num_classes: int = 98) -> Dict[str, float]:
    x = np.asarray(embedding, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    centroids = np.zeros((num_classes, x.shape[1]), dtype=np.float64)
    within = []
    counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        xc = x[y == c]
        counts[c] = len(xc)
        if len(xc):
            centroids[c] = xc.mean(axis=0)
            within.append(np.mean(np.sum((xc - centroids[c]) ** 2, axis=1)))
    present = counts > 0
    global_c = centroids[present].mean(axis=0)
    between = float(np.mean(np.sum((centroids[present] - global_c) ** 2, axis=1)))
    within_mean = float(np.mean(within)) if within else float("nan")
    fisher = between / max(within_mean, 1e-12)
    norms = np.linalg.norm(centroids[present], axis=1, keepdims=True)
    cn = centroids[present] / np.maximum(norms, 1e-12)
    cosine = cn @ cn.T
    off = cosine[~np.eye(len(cn), dtype=bool)] if len(cn) > 1 else np.array([0.0])
    return {
        "between_centroid_variance": between,
        "within_class_variance": within_mean,
        "fisher_ratio": float(fisher),
        "mean_intercentroid_cosine": float(off.mean()),
    }


def deterministic_per_class_positions(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for c in sorted(np.unique(labels).tolist()):
        p = np.flatnonzero(labels == c)
        if len(p) > per_class:
            p = np.sort(rng.choice(p, size=per_class, replace=False))
        out.append(p)
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def domain_probe_auc(a_emb: np.ndarray, a_tx: np.ndarray, b_emb: np.ndarray, b_tx: np.ndarray, sample_per_tx: int, seed: int) -> float:
    # Restrict to common transmitter identities so protocol separability is not driven by class absence.
    common = np.intersect1d(np.unique(a_tx), np.unique(b_tx))
    if len(common) == 0:
        return float("nan")
    am = np.isin(a_tx, common); bm = np.isin(b_tx, common)
    ae, at = a_emb[am], a_tx[am]
    be, bt = b_emb[bm], b_tx[bm]
    pa = deterministic_per_class_positions(at, sample_per_tx, seed)
    pb = deterministic_per_class_positions(bt, sample_per_tx, seed + 1)
    x = np.concatenate([ae[pa], be[pb]], axis=0)
    y = np.concatenate([np.zeros(len(pa), dtype=np.int64), np.ones(len(pb), dtype=np.int64)])
    if len(np.unique(y)) != 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.zeros(len(y), dtype=np.float64)
    for tr, te in cv.split(x, y):
        clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-4, random_state=seed)
        clf.fit(x[tr], y[tr])
        scores[te] = clf.predict_proba(x[te])[:, 1]
    return float(roc_auc_score(y, scores))


def binary_attribute_probe_auc(emb: np.ndarray, attr: np.ndarray, tx: np.ndarray, sample_per_tx: int, seed: int) -> float:
    pos = deterministic_per_class_positions(tx, sample_per_tx, seed)
    x, y = emb[pos], attr[pos]
    if len(np.unique(y)) != 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.zeros(len(y), dtype=np.float64)
    for tr, te in cv.split(x, y):
        clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-4, random_state=seed)
        clf.fit(x[tr], y[tr])
        scores[te] = clf.predict_proba(x[te])[:, 1]
    return float(roc_auc_score(y, scores))


def fit_novelty_geometry(train_emb: np.ndarray, train_y: np.ndarray, num_classes: int = 98, fit_limit: int = 100_000, seed: int = 0):
    x = np.asarray(train_emb, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.int64)
    centroids = np.stack([x[y == c].mean(axis=0) for c in range(num_classes)], axis=0)
    normalized = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    per_class = max(1, fit_limit // num_classes)
    pos = deterministic_per_class_positions(y, per_class, seed)
    residuals = x[pos] - centroids[y[pos]]
    lw = LedoitWolf(assume_centered=True).fit(residuals)
    return centroids, normalized, lw.precision_, float(lw.shrinkage_)


def novelty_scores(emb: np.ndarray, logits: np.ndarray, centroids: np.ndarray, norm_centroids: np.ndarray, precision: np.ndarray) -> Dict[str, np.ndarray]:
    x = np.asarray(emb, dtype=np.float64)
    eu = np.empty(len(x), dtype=np.float64)
    cos = np.empty(len(x), dtype=np.float64)
    mah = np.empty(len(x), dtype=np.float64)
    # Chunk all prototype-distance calculations; never materialize [N,98,128] for full WiSig.
    for start in range(0, len(x), 1024):
        xb = x[start:start + 1024]
        dd = xb[:, None, :] - centroids[None, :, :]
        eu[start:start + len(xb)] = np.sqrt(np.maximum(np.sum(dd * dd, axis=2), 0.0)).min(axis=1)
        xn = xb / np.maximum(np.linalg.norm(xb, axis=1, keepdims=True), 1e-12)
        cos[start:start + len(xb)] = (1.0 - xn @ norm_centroids.T).min(axis=1)
        vals = np.einsum("bcd,de,bce->bc", dd, precision, dd, optimize=True)
        mah[start:start + len(xb)] = np.sqrt(np.maximum(vals.min(axis=1), 0.0))
    z = np.asarray(logits, dtype=np.float64)
    zmax = z.max(axis=1, keepdims=True)
    energy = -(np.log(np.exp(z - zmax).sum(axis=1)) + zmax[:, 0])
    p = np.exp(z - zmax); p /= p.sum(axis=1, keepdims=True)
    msp = 1.0 - p.max(axis=1)
    return {"euclidean": eu, "cosine": cos, "mahalanobis": mah, "energy": energy, "one_minus_msp": msp}


def calibration_unknown_diagnostic(known_scores: Mapping[str, np.ndarray], unknown_scores: Mapping[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out = {}
    for name in known_scores:
        k = np.asarray(known_scores[name], dtype=np.float64)
        u = np.asarray(unknown_scores[name], dtype=np.float64)
        y = np.concatenate([np.zeros(len(k), dtype=np.int64), np.ones(len(u), dtype=np.int64)])
        s = np.concatenate([k, u])
        auroc = float(roc_auc_score(y, s))
        aupr = float(average_precision_score(y, s))
        pooled = math.sqrt(max(1e-12, ((len(k)-1)*k.var(ddof=1) + (len(u)-1)*u.var(ddof=1)) / max(1, len(k)+len(u)-2)))
        d = float((u.mean() - k.mean()) / pooled)
        out[name] = {"auroc": auroc, "auprc": aupr, "cohen_d": d}
    return out
