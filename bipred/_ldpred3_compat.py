"""Lazy compatibility seam for the pinned :mod:`ldpred3` dependency.

The names here are private helpers that bipred deliberately shares with the
exact ldpred3 version pinned in ``pyproject.toml``. Keeping those imports here
makes the next dependency review one small, explicit audit; loading a lightweight
helper such as LDSC does not import the Numba kernels. Public ``LowRankLD`` is
imported from :mod:`ldpred3` by its consumers.
"""

from __future__ import annotations

import importlib


_MODULE_NAMES = {
    "ldpred3._common": (
        "_as_n_vector",
        "_check_h2_p",
        "_finite_control",
        "_integer_at_least",
        "_validate_beta_hat",
        "_validate_blocks",
        "_validate_boolean_controls",
        "_validate_iterations",
        "_validate_seed",
    ),
    "ldpred3.ld_repr": ("_Q8",),
    "ldpred3._numba": (
        "HAVE_NUMBA",
        "_jit",
        "_jit_nogil",
        "_jit_parallel",
        "_set_threads",
        "prange",
    ),
    "ldpred3.ldsc": ("_weights", "_wls"),
}

_NAME_TO_MODULE = {
    name: module for module, names in _MODULE_NAMES.items() for name in names
}

__all__ = sorted(_NAME_TO_MODULE)


def __getattr__(name):
    module = _NAME_TO_MODULE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return list(__all__)
