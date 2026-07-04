"""bipred — bivariate (two-trait) LDpred.

A joint LDpred model that fits **two traits sharing one LD reference** at once,
built on top of :mod:`ldpred3`. It estimates each trait's SNP heritability, the
**genetic correlation** between them, the per-trait and shared polygenicity
(a MiXeR-style polygenic-overlap summary), and posterior-mean effects that let a
well-powered trait sharpen a correlated under-powered one.

Public API::

    from bipred import ldpred3_auto_bivariate, ldpred3_auto_bivariate_blocks
    res = ldpred3_auto_bivariate(corr, beta_hat1, beta_hat2, n1, n2)
    res.rg, res.h2, res.mixer

``ldpred3_auto_bivariate`` runs on a single dense LD matrix;
``ldpred3_auto_bivariate_blocks`` streams the genome block by block. Both return
a :class:`~bipred.bivariate.BivariateResult`.

Names are imported **lazily** (PEP 562) so ``import bipred`` stays cheap; NumPy is
the only hard dependency (optional Numba accelerates the Gibbs sampler, inherited
from ldpred3's ``[fast]`` extra).
"""

import importlib

__version__ = "0.1.0.dev0"

# public name -> submodule it lives in
_EXPORTS = {
    "bivariate": ["ldpred3_auto_bivariate", "ldpred3_auto_bivariate_blocks",
                  "BivariateResult"],
}

# name -> module, for the lazy loader
_NAME_TO_MODULE = {name: mod for mod, names in _EXPORTS.items() for name in names}

__all__ = ["__version__", *_NAME_TO_MODULE]


def __getattr__(name):
    """Import the owning submodule on first access (PEP 562)."""
    mod = _NAME_TO_MODULE.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    obj = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = obj          # cache so subsequent access skips __getattr__
    return obj


def __dir__():
    return sorted(__all__)
