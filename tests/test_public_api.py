"""The lazy public API must remain loadable against the pinned ldpred3."""

import subprocess
import sys
import types

import pytest

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
    "ld_consistency_screen",
    "dentist",
    "dentist_statistic",
    "implied_sample_size",
    "in_long_range_ld",
    "sd_consistency",
}


#: The submodules the lazy loader must expose as *modules*, never shadowed by a
#: function of the same name.
_SUBMODULES = {"bivariate", "multichain", "ldsc", "regional", "qc"}


def test_public_api_surface_is_exact_and_resolves():
    assert set(bipred.__all__) == _EXPECTED
    # Submodules are discoverable but excluded from ``__all__`` so that
    # ``from bipred import *`` still imports only the public API.
    assert set(bipred.__dir__()) == _EXPECTED | _SUBMODULES
    for name in bipred.__all__:
        assert getattr(bipred, name) is not None, name


def test_no_submodule_name_collides_with_an_exported_name():
    # ``bipred.<name>`` cannot be both a module and a function. When a submodule
    # shares a name with one of its exports, which one you get depends on import
    # order -- the bug that made ``bipred/ldsc_rg.py`` return a module from
    # ``from bipred import estimate_sample_overlap, ldsc_rg``.
    assert not (set(bipred._EXPORTS) & set(bipred._NAME_TO_MODULE))


@pytest.mark.parametrize("preamble", [
    # The exact snippet documented in docs/rg.md, which used to bind the module.
    "from bipred import estimate_sample_overlap, ldsc_rg",
    "from bipred import ldsc_rg, estimate_sample_overlap",
    # A prior submodule import must not rebind the function either.
    "import bipred.ldsc\nfrom bipred import ldsc_rg",
])
def test_exported_names_resolve_to_callables_in_every_import_order(preamble):
    # Each case runs in a fresh interpreter: the failure is order-dependent, so
    # importing one order inside this process would mask the others.
    code = f"{preamble}\nassert callable(ldsc_rg), type(ldsc_rg)\n"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_submodules_are_reachable_as_attributes():
    code = "\n".join(
        ["import bipred", "import types"]
        + [f"assert isinstance(bipred.{name}, types.ModuleType), {name!r}"
           for name in sorted(_SUBMODULES)]
        # The package docstring cross-references these paths, so they must work.
        + ["assert bipred.ldsc.ldsc_rg is not None",
           "assert bipred.bivariate.BivariateResult is not None"]
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_submodule_as_binds_the_module_not_a_function():
    # ``import a.b as c`` is ``getattr(a, 'b')``, which used to succeed with the
    # wrong object.
    import bipred.ldsc as ldsc_module

    assert isinstance(ldsc_module, types.ModuleType)
    assert callable(ldsc_module.ldsc_rg)
