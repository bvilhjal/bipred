"""Lazy compatibility seam for the bounded :mod:`ldpred3` dependency.

The names here are private helpers that bipred deliberately shares with the
LDpred3 0.5 development line bounded in ``pyproject.toml``. Keeping those
imports here makes the next dependency review one small, explicit audit;
loading a lightweight helper such as LDSC does not import the Numba kernels.
Data-level sibling integration uses the stable :mod:`ldpred3.interop` surface.
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
    # Nesting a thread pool over BLAS is safe only for some builds of it, and
    # ldpred3 already owns that determination. Importing it here rather than
    # re-deriving it keeps one answer to the question across both packages.
    # ``_blas_runtime_info`` backs the screen's why-was-parallelism-blocked
    # hint with the same introspection the gate itself used.
    "ldpred3.ld": ("_blas_pool_safe", "_blas_runtime_info"),
    "ldpred3._numba": (
        "HAVE_NUMBA",
        "_jit",
        "_get_thread_id",
        "_jit_fastmath_nogil",
        "_jit_nogil",
        "_jit_parallel",
        "_set_threads",
        "prange",
        "warn_no_numba",
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
