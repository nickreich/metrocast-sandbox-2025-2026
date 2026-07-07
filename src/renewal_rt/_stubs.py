"""Lightweight import-time stubs for idmodels' unused optional dependencies.

Importing `idmodels` eagerly imports GBQRModel and SARIXModel (via idmodels/__init__.py),
which pull in `lightgbm`, `sarix`, `timeseriesutils`, and `tqdm` at module load. This
renewal model uses none of them, and in this environment `lightgbm` is broken (missing
libomp) while `sarix` drags in heavy Bayesian-inference dependencies. These libraries are
only *used* inside GBQR/SARIX methods, never at import, so registering minimal stub modules
in sys.modules lets `from idmodels.model import IDModel` succeed without them.

Import this module BEFORE importing anything from idmodels.
"""

import sys
import types


def _ensure_stub(name: str, attrs: dict | None = None) -> types.ModuleType:
    """Register a stub module under `name` only if the real package is not importable."""
    if name in sys.modules:
        return sys.modules[name]
    try:
        __import__(name)
        return sys.modules[name]
    except Exception:
        mod = types.ModuleType(name)
        for k, v in (attrs or {}).items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        return mod


def install() -> None:
    # lightgbm: gbqr.py does `import lightgbm as lgb`
    _ensure_stub("lightgbm")

    # timeseriesutils: features.py does `from timeseriesutils import featurize`
    _ensure_stub("timeseriesutils", {"featurize": lambda *a, **k: None})

    # sarix: sarix.py does `from sarix import sarix`
    sarix_pkg = _ensure_stub("sarix")
    if not hasattr(sarix_pkg, "sarix"):
        inner = types.ModuleType("sarix.sarix")
        inner.SARIX = object
        sarix_pkg.sarix = inner
        sys.modules["sarix.sarix"] = inner

    # tqdm: gbqr.py does `from tqdm.autonotebook import tqdm`
    if "tqdm" not in sys.modules:
        try:
            __import__("tqdm.autonotebook")
        except Exception:
            tqdm_pkg = types.ModuleType("tqdm")
            auto = types.ModuleType("tqdm.autonotebook")
            auto.tqdm = lambda x=None, *a, **k: x if x is not None else iter(())
            tqdm_pkg.autonotebook = auto
            tqdm_pkg.tqdm = auto.tqdm
            sys.modules["tqdm"] = tqdm_pkg
            sys.modules["tqdm.autonotebook"] = auto
