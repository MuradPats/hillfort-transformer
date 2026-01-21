"""
Compatibility shim.

RGBX code expects `import utils.xxx` when run from inside RGBX/.
When running from repo root, we forward `utils.*` to `RGBX.utils.*`.
"""

import importlib
import sys

# Point this package at RGBX.utils so `import utils.foo` works.
_rgbx_engine = importlib.import_module("RGBX.engine")

# Make `utils` behave like `RGBX.utils`
__path__ = _rgbx_engine.__path__  # type: ignore
__all__ = getattr(_rgbx_engine, "__all__", [])
