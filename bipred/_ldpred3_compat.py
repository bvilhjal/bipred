"""Lazy compatibility seam for the bounded :mod:`ldpred3` dependency.

The names here are the bounded LDpred3 surface in ``pyproject.toml``
(``>=0.7.12,<0.8``; bounded LD cross-products require that provider version).
Single-trait preparation and screening remain provider-owned. Most helpers --
the Numba decorators, the input validators, ``warn_no_numba``, and the helpers
``prepare``/``qc`` share with the provider -- are published by the public
:mod:`ldpred3.shim` module, so siblings never touch the defining modules.
The seam binds the provider's historic *underscore* spellings, the only ones
that exist at the declared floor: LDpred3 gained the non-underscore aliases
(``jit_nogil`` for ``_jit_nogil``, ``cache_variant_table`` for
``_cache_variant_table``, and so on) only at v0.7.23, and keeps the
underscore spellings as aliases on master. Once the floor passes v0.7.23 the
shim-sourced names can switch to the public spellings. The remainder are
helpers reachable only from their defining private modules -- the explicit
``_PRIVATE_NAMES`` list at the bottom, including the pairing helpers
``bipred.prepare`` and ``bipred.qc`` share with the provider.
Keeping every borrowing in this one seam makes the next dependency review one
small, explicit audit, and both tables resolve lazily so a lightweight helper
such as LDSC imports only what it names. Data-level sibling integration uses
the stable :mod:`ldpred3.interop` surface.
"""

from __future__ import annotations

import importlib


_SHIM_MODULE = "ldpred3.shim"

#: Names bound from ``ldpred3.shim`` under the private underscore spellings:
#: the only spellings published at the declared floor (0.7.12). LDpred3
#: >=v0.7.23 also publishes each of these without the underscore (``jit``,
#: ``integer_at_least``, ...); switch to the public spellings once the floor
#: passes the alias release.
_SHIM_NAMES = (
    # Code-level surface: stable by the same promise as interop.
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
)

#: Helpers whose private spellings live in their defining modules rather than
#: ``ldpred3.shim`` at the floor (the shim only re-exports them lazily from
#: v0.7.23 on); each entry is an explicit borrowing, bound here as bind name
#: -> (defining module, attribute).
_PRIVATE_NAMES = {
    # Scalar and control validators shared by every sampler entry point.
    # ``ldpred3.shim`` re-exports them (and the non-underscore twins) lazily
    # from v0.7.23; at the floor they live only here.
    "_as_n_vector": ("ldpred3._common", "_as_n_vector"),
    "_check_h2_p": ("ldpred3._common", "_check_h2_p"),
    "_finite_control": ("ldpred3._common", "_finite_control"),
    "_validate_boolean_controls": ("ldpred3._common",
                                   "_validate_boolean_controls"),
    # The int8 LD quantisation scale the D8 blocks are encoded with.
    "_Q8": ("ldpred3.ld_repr", "_Q8"),
    # ``harmonize`` already builds an identifier -> row-index map and caches it
    # on the variant table, and the web caller pre-warms it once per job. The
    # overlap diagnosis and the identifier re-anchoring need exactly that map,
    # so borrowing it costs nothing; building a private one over a 1.4M-variant
    # reference cost several hundred megabytes per call.
    "_variant_indices": ("ldpred3.harmonize", "_variant_indices"),
    # Nesting a thread pool over BLAS is safe only for some builds of it, and
    # ldpred3 already owns that determination. Importing it here rather than
    # re-deriving it keeps one answer to the question across both packages.
    # ``_blas_runtime_info`` backs the screen's why-was-parallelism-blocked
    # hint with the same introspection the gate itself used.
    "_blas_pool_safe": ("ldpred3.ld", "_blas_pool_safe"),
    "_blas_runtime_info": ("ldpred3.ld", "_blas_runtime_info"),
    # LDSC regression weights and the weighted least-squares solve.
    "_weights": ("ldpred3.ldsc", "_weights"),
    "_wls": ("ldpred3.ldsc", "_wls"),
    # Deterministic λmin for large float LD blocks. DENTIST already owns the
    # Lanczos-or-eigh fallback; the structural PD probe must not re-derive it.
    # ``_ridge_for_floor`` is the one ill-posed-window repair shared by both
    # DENTIST screens, and ``DEFAULT_EIGENVALUE_FLOOR`` its floor: importing
    # them is what makes "both screens repair the same way" true by
    # construction rather than by a docstring kept in step by hand.
    "DEFAULT_EIGENVALUE_FLOOR": ("ldpred3.qc", "DEFAULT_EIGENVALUE_FLOOR"),
    "_ridge_for_floor": ("ldpred3.qc", "_ridge_for_floor"),
    "_smallest_eigenvalue": ("ldpred3.qc", "_smallest_eigenvalue"),
    # The progress-callback contract moved to LDpred3 with the screen; the
    # pairing reports through the same two helpers so one caller callable
    # sees one event vocabulary.
    "report": ("ldpred3._progress", "report"),
    "validate": ("ldpred3._progress", "validate"),
    # The prepared variant-table cache a web caller pre-warms once per job;
    # ``pair_prepared_traits`` reads through it.
    "_cache_variant_table": ("ldpred3.prepare", "_cache_variant_table"),
    # ``ldpred3.prepare`` internals the pairing shares: opening an
    # already-prepared cache without a second payload scan, preparing and
    # validating each trait, resolving its effective N, and refusing a
    # screen-failed trait before the joint fit sees it.
    "_open_cache": ("ldpred3.prepare", "_open_cache"),
    "_prepare_trait": ("ldpred3.prepare", "_prepare_trait"),
    "_require_usable": ("ldpred3.prepare", "_require_usable"),
    "_resolve_n_eff": ("ldpred3.prepare", "_resolve_n_eff"),
    "_validated_trait": ("ldpred3.prepare", "_validated_trait"),
    # ``ldpred3.qc`` internals re-exported name-for-name through
    # ``bipred.qc`` so a caller cannot tell the alias module from the
    # provider's -- ``_ld_consistency_screen_selected`` is the one
    # ``prepare`` resolves.
    "_confirmed_drops": ("ldpred3.qc", "_confirmed_drops"),
    "_dentist_statistic": ("ldpred3.qc", "_dentist_statistic"),
    "_ld_consistency_screen_selected": ("ldpred3.qc",
                                        "_ld_consistency_screen_selected"),
    "_precision_loo": ("ldpred3.qc", "_precision_loo"),
    "_window_ld": ("ldpred3.qc", "_window_ld"),
}

__all__ = sorted(list(_SHIM_NAMES) + list(_PRIVATE_NAMES))


def __getattr__(name):
    if name in _SHIM_NAMES:
        module, source = _SHIM_MODULE, name
    else:
        private = _PRIVATE_NAMES.get(name)
        if private is None:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}")
        module, source = private
    value = getattr(importlib.import_module(module), source)
    globals()[name] = value
    return value


def __dir__():
    return list(__all__)
