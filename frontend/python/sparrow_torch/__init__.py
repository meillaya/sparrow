"""Compatibility shim: the old distribution name maps to :mod:`forge_torch`.

``sparrow_torch`` is kept for one release. New code should import
``forge_torch``; every public attribute and submodule is re-exported here.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import forge_torch as _forge
from forge_torch import *  # noqa: F403 - intentional re-export

__all__ = list(getattr(_forge, "__all__", ()))

for _info in pkgutil.iter_modules(_forge.__path__):
    _name = _info.name
    if _name.startswith("_"):
        continue
    _module_name = f"{__name__}.{_name}"
    if _module_name in sys.modules:
        continue
    try:
        _module = importlib.import_module(f"forge_torch.{_name}")
    except Exception:  # pragma: no cover - optional modules
        continue
    sys.modules[_module_name] = _module
    globals()[_name] = _module

__version__ = getattr(_forge, "__version__", "0.0.0")
