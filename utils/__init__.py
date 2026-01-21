"""
Compatibility shim.

RGBX code expects `import utils.xxx` when run from inside RGBX/.
When running from repo root, we forward `utils.*` to `RGBX.utils.*`.
"""

import importlib
import sys

# Point this package at RGBX.utils so `import utils.foo` works.
_rgbx_utils = importlib.import_module("RGBX.utils")

# Make `utils` behave like `RGBX.utils`
__path__ = _rgbx_utils.__path__  # type: ignore
__all__ = getattr(_rgbx_utils, "__all__", [])
