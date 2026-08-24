#!/usr/bin/env python3
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Dict

_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')


def validate_final_selection_lock(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            'Strict access denied: FINAL_SELECTION_LOCK.json is missing. '
            'Strict data is final-evaluation-only.'
        )
    obj = json.load(path.open('r', encoding='utf-8'))
    required_true = (
        'teacher_frozen',
        'surrogate_frozen',
        'open_set_threshold_frozen',
        'fidelity_gate_frozen',
        'strict_access_authorized',
    )
    if obj.get('status') != 'FINAL_SELECTION_LOCKED':
        raise RuntimeError('Strict access denied: status is not FINAL_SELECTION_LOCKED')
    for k in required_true:
        if obj.get(k) is not True:
            raise RuntimeError(f'Strict access denied: {k} must be true')
    cfg = obj.get('final_config_sha256', '')
    if not _SHA256_RE.fullmatch(cfg):
        raise RuntimeError('Strict access denied: final_config_sha256 is missing/invalid')
    return obj


def resolve_wisig_strict_source(
    registry: Dict[str, Any], role: str, final_selection_lock: Path
) -> Path:
    if role not in ('strict_zero_day', 'strict_zero_day_shift'):
        raise ValueError(role)
    validate_final_selection_lock(final_selection_lock)
    spec = registry['split_files'][role]
    return Path(registry['old_split_root']) / spec['filename']
