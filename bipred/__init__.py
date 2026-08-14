"""bipred — bivariate (two-trait) LDpred.

A joint LDpred model that fits **two traits sharing one LD reference** at once,
built on top of :mod:`ldpred3`. It estimates each trait's SNP heritability, the
**genetic correlation** between them, the per-trait and shared polygenicity
(a MiXeR-style polygenic-overlap summary), and posterior-mean effects for
prediction.

Public API::

    from bipred import prepare_bivariate_sumstats, ldpred3_auto_bivariate_blocks
    with prepare_bivariate_sumstats("ld.npz", "t1.tsv", "t2.tsv") as prep:
        res = ldpred3_auto_bivariate_blocks(
            prep.blocks, prep.beta_hat1, prep.beta_hat2,
            prep.n_eff1, prep.n_eff2)
        res.write_weights("t1.weights", trait=1, id=prep.id, chrom=prep.chrom,
                          pos=prep.pos, effect_allele=prep.effect_allele,
                          other_allele=prep.other_allele)

    from bipred import ldpred3_auto_bivariate_chains
    multi = ldpred3_auto_bivariate_chains(blocks, beta_hat1, beta_hat2, n1, n2)
    multi.posterior, multi.basic_split_rhat

``ldpred3_auto_bivariate`` runs on a single dense LD matrix;
``ldpred3_auto_bivariate_blocks`` streams the genome block by block. Both return
a :class:`~bipred.bivariate.BivariateResult`. The multi-chain driver can run
dispersed chains serially or concurrently and pools every finite, equal-length
chain.

Per-locus structure comes from :func:`~bipred.regional.regional_rg`, which turns
a fit into a **regional** genetic correlation. Read its docstring first: it
documents two biases it does not correct (uncorrected sample overlap inflates
or deflates regional covariance, and regional estimates are shrunk toward the
genome-wide value).

Before fitting real summary statistics, perform study-appropriate QC and inspect
their consistency with the fitted LD reference. The lightweight
:func:`~bipred.qc.ld_consistency_screen` detects neighbourhood-level
disagreement that per-variant filters cannot see. It is DENTIST-inspired, not a
full implementation of the published DENTIST workflow; ``dentist`` remains as a
compatibility alias.

For a fast, moment-based genetic-correlation estimate (the cross-check on the
joint fit), :func:`~bipred.ldsc.ldsc_rg` implements cross-trait LD Score
regression, with :func:`~bipred.ldsc.estimate_sample_overlap` for shared
samples. All genetic-correlation estimation lives here; ldpred3 keeps only the
*univariate* LDSC (``ld_scores`` / ``ldsc_h2``) that these build on.

Names are imported **lazily** (PEP 562) so ``import bipred`` stays cheap.
ldpred3 and NumPy are runtime dependencies; optional Numba acceleration comes
from ldpred3's ``[fast]`` extra.
"""

import importlib

__version__ = "0.3.9.dev0"

# public name -> submodule it lives in. No module name may equal one of its own
# exported names: importing a submodule binds it on this package, and the cache
# below then overwrites that binding with the function, so a collision makes
# ``bipred.<name>`` resolve to the module or the function depending on import
# order. That is why the LDSC module is ``ldsc`` and not ``ldsc_rg``.
_EXPORTS = {
    "bivariate": ["ldpred3_auto_bivariate", "ldpred3_auto_bivariate_blocks",
                  "BivariateResult"],
    "prepare": ["prepare_bivariate_sumstats", "PreparedBivariate",
                "subset_blocks"],
    "multichain": ["ldpred3_auto_bivariate_chains",
                   "MultiChainBivariateResult", "BivariateChainSummary",
                   "BivariateBasicSplitRHat"],
    "ldsc": ["ldsc_rg", "LDSCRgResult", "estimate_sample_overlap",
             "ldsc_chi2_mask"],
    "regional": ["regional_rg", "RegionalRgResult"],
    "qc": ["ld_consistency_screen", "dentist", "dentist_statistic",
           "in_long_range_ld",
           "implied_sample_size", "sd_consistency"],
}

# name -> module, for the lazy loader
_NAME_TO_MODULE = {name: mod for mod, names in _EXPORTS.items() for name in names}

assert not (set(_EXPORTS) & set(_NAME_TO_MODULE)), (
    "a submodule name collides with an exported name; see _EXPORTS above")

__all__ = ["__version__", *_NAME_TO_MODULE]


def __getattr__(name):
    """Import the owning submodule on first access (PEP 562)."""
    mod = _NAME_TO_MODULE.get(name)
    if mod is not None:
        obj = getattr(importlib.import_module(f".{mod}", __name__), name)
        globals()[name] = obj      # cache so subsequent access skips __getattr__
        return obj
    if name in _EXPORTS:
        # Submodule access (``bipred.bivariate``). Deliberately *not* cached in
        # globals(): the import machinery already binds the module on this
        # package, and caching it here is what shadowed a submodule whose name
        # matched one of its exports.
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # Submodules are discoverable but stay out of ``__all__``, so
    # ``from bipred import *`` still imports only the public API.
    return sorted({*__all__, *_EXPORTS})
