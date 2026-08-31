"""Lazy compatibility seam for the bounded :mod:`ldpred3` dependency.

The names here are the bounded LDpred3 surface in ``pyproject.toml``
(``>=0.5.5.dev0,<0.7``, including the 0.6 line). Most of them -- the Numba
decorators, the input validators, ``warn_no_numba`` -- are *published* by the
public :mod:`ldpred3.shim` module (added in LDpred3 0.5.3), and the seam
binds those through it. The remainder are still underscore-private helpers
LDpred3 has not published: keeping every borrowing here makes the next
dependency review one small, explicit audit, and loading a lightweight helper
such as LDSC does not import the Numba kernels. Data-level sibling
integration uses the stable :mod:`ldpred3.interop` surface.
"""

from __future__ import annotations

import importlib


_MODULE_NAMES = {
    # Published code-level surface: stable by the same promise as interop.
    "ldpred3.shim": (
        "HAVE_NUMBA",
        "_get_thread_id",
        "_integer_at_least",
        "_jit",
        "_jit_fastmath_nogil",
        "_jit_nogil",
        "_jit_parallel",
        "_set_threads",
        "_validate_beta_hat",
        "_validate_blocks",
        "_validate_iterations",
        "_validate_seed",
        "prange",
        "warn_no_numba",
    ),
    # Still-private helpers below; each entry is an explicit borrowing.
    "ldpred3._common": (
        "_as_n_vector",
        "_check_h2_p",
        "_finite_control",
        "_validate_boolean_controls",
    ),
    "ldpred3.ld_repr": ("_Q8",),
    # ``harmonize`` already builds an identifier -> row-index map and caches it
    # on the variant table, and the web caller pre-warms it once per job. The
    # overlap diagnosis and the identifier re-anchoring need exactly that map,
    # so borrowing it costs nothing; building a private one over a 1.4M-variant
    # reference cost several hundred megabytes per call.
    "ldpred3.harmonize": ("_variant_indices",),
    # Nesting a thread pool over BLAS is safe only for some builds of it, and
    # ldpred3 already owns that determination. Importing it here rather than
    # re-deriving it keeps one answer to the question across both packages.
    # ``_blas_runtime_info`` backs the screen's why-was-parallelism-blocked
    # hint with the same introspection the gate itself used.
    "ldpred3.ld": ("_blas_pool_safe", "_blas_runtime_info"),
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
