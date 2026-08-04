"""The lazy public API must remain loadable against the pinned ldpred3."""

import bipred

#: The complete public surface, stated once so an accidental addition or
#: removal fails here rather than shipping silently.
_EXPECTED = {
    "__version__",
    "ldpred3_auto_bivariate",
    "ldpred3_auto_bivariate_blocks",
    "BivariateResult",
    "ldpred3_auto_bivariate_chains",
    "MultiChainBivariateResult",
    "BivariateChainSummary",
    "BivariateBasicSplitRHat",
    "ldsc_rg",
    "LDSCRgResult",
    "estimate_sample_overlap",
    "regional_rg",
    "RegionalRgResult",
}


def test_public_api_surface_is_exact_and_resolves():
    assert set(bipred.__all__) == _EXPECTED
    assert bipred.__dir__() == sorted(bipred.__all__)
    for name in bipred.__all__:
        assert getattr(bipred, name) is not None, name
