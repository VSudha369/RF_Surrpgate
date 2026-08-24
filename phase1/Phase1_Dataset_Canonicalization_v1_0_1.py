#!/usr/bin/env python3
"""
PHASE 1 — Dataset Canonicalization and Storage Pipeline v1.0.1

Bugfix wrapper for v1.0.0:
- google.colab.drive.mount() must run in the active Colab notebook kernel.
- Phase 1 is normally launched with `!python`, so the subprocess must only
  verify that Drive is already mounted.
- All scientific/storage logic remains the validated v1.0.0 implementation.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "Phase1_Dataset_Canonicalization_v1_0_0.py"

spec = importlib.util.spec_from_file_location("phase1_v1_0_0_impl", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load Phase-1 base implementation: {BASE}")
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)

# Stamp all new manifests/status/configs with the patched canonical version.
impl.VERSION = "1.0.1"


def require_drive_mounted() -> None:
    """Require Drive to have been mounted by the Colab notebook kernel."""
    mydrive = Path("/content/drive/MyDrive")
    if not mydrive.is_dir():
        raise RuntimeError(
            "Google Drive is not mounted. Run this in a Colab NOTEBOOK CELL first:\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive', force_remount=False)\n"
            "Then rerun the Phase-1 command."
        )


# Replace the subprocess-unsafe v1.0.0 mount call.
impl.mount_drive = require_drive_mounted

if __name__ == "__main__":
    impl.main()
