"""
Compatibility shim.

RGBX code imports `from config import config` assuming execution from within the RGBX folder.
When running from repo root (baseline, notebooks, etc), this file provides a consistent `config`.

By default we expose baseline.config.config if available, otherwise fall back to RGBX.config.config.
"""

try:
    from baseline.config import config  # preferred when running baseline
except Exception:
    from RGBX.config import config      # fallback for RGBX runs
