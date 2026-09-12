"""Lazy compatibility seam for the bounded :mod:`ldpred3` dependency.

The names here are the bounded LDpred3 surface in ``pyproject.toml``
(``>=0.7.12,<0.8``; bounded LD cross-products require that provider version).
Single-trait preparation and screening remain provider-owned. Most helpers --
the Numba decorators, the input validators, ``warn_no_numba``, and the helpers
``prepare``/``qc`` share with the provider -- are *published* by the public
:mod:`ldpred3.shim` module (added in LDpred3 0.5.3), so siblings never touch
the defining underscore-private modules. Publication renamed them -- the jit
decorators and validators dropped their underscore prefix, and a few moved to
descriptive names (``ldsc_weights``/``ldsc_wls``/``progress_report``/
``progress_validate``) -- so each keeps its historical bind name here and
callers and the seam contract are unchanged. The remainder are still
underscore-private helpers LDpred3 has not published -- the explicit
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

#: Seam bind name -> published ``ldpred3.shim`` attribute.
_SHIM_NAMES = {
    # Code-level surface: stable by the same promise as interop.
    "HAVE_NUMBA": "HAVE_NUMBA",
    "_get_thread_id": "get_thread_id",
    "_integer_at_least": "integer_at_least",
    "_jit": "jit",
    "_jit_fastmath_nogil": "jit_fastmath_nogil",
    "_jit_nogil": "jit_nogil",
    "_jit_parallel": "jit_parallel",
    "_set_threads": "set_threads",
    "_validate_beta_hat": "validate_beta_hat",
    "_validate_blocks": "validate_blocks",
    "_validate_iterations": "validate_iterations",
    "_validate_seed": "validate_seed",
    "prange": "prange",
    "warn_no_numba": "warn_no_numba",
    "_as_n_vector": "as_n_vector",
    "_check_h2_p": "check_h2_p",
    "_finite_control": "finite_control",
    "_validate_boolean_controls": "validate_boolean_controls",
    "_Q8": "Q8",
    # ``harmonize`` already builds an identifier -> row-index map and caches it
    # on the variant table, and the web caller pre-warms it once per job. The
    # overlap diagnosis and the identifier re-anchoring need exactly that map,
    # so borrowing it costs nothing; building a private one over a 1.4M-variant
    # reference cost several hundred megabytes per call.
    "_variant_indices": "variant_indices",
    # Nesting a thread pool over BLAS is safe only for some builds of it, and
    # ldpred3 already owns that determination. Importing it here rather than
    # re-deriving it keeps one answer to the question across both packages.
    # ``_blas_runtime_info`` backs the screen's why-was-parallelism-blocked
    # hint with the same introspection the gate itself used.
    "_blas_pool_safe": "blas_pool_safe",
    "_blas_runtime_info": "blas_runtime_info",
    "_weights": "ldsc_weights",
    "_wls": "ldsc_wls",
    # Deterministic λmin for large float LD blocks. DENTIST already owns the
    # Lanczos-or-eigh fallback; the structural PD probe must not re-derive it.
    # ``_ridge_for_floor`` is the one ill-posed-window repair shared by both
    # DENTIST screens, and ``DEFAULT_EIGENVALUE_FLOOR`` its floor: importing
    # them is what makes "both screens repair the same way" true by
    # construction rather than by a docstring kept in step by hand.
    "DEFAULT_EIGENVALUE_FLOOR": "DEFAULT_EIGENVALUE_FLOOR",
    "_ridge_for_floor": "ridge_for_floor",
    "_smallest_eigenvalue": "smallest_eigenvalue",
    # The progress-callback contract moved to LDpred3 with the screen; the
    # pairing reports through the same two helpers so one caller callable
    # sees one event vocabulary.
    "report": "progress_report",
    "validate": "progress_validate",
    # The prepared variant-table cache a web caller pre-warms once per job;
    # ``pair_prepared_traits`` reads through it.
    "_cache_variant_table": "cache_variant_table",
}

#: Still-private helpers LDpred3 has not published; each entry is an explicit
#: borrowing, bound here as bind name -> (defining module, attribute).
_PRIVATE_NAMES = {
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
    attribute = _SHIM_NAMES.get(name)
    if attribute is not None:
        module, source = _SHIM_MODULE, attribute
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
