"""Semantic guards on the private ldpred3 symbols bipred reaches into.

bipred centralizes its underscore-prefixed ldpred3 imports in
``bipred._ldpred3_compat``. The dependency is bounded at
``>=0.5.5.dev0,<0.7`` because that private surface is unversioned. Bipred's
public API and compatibility seam import lazily; these tests force the
complete seam to resolve and guard its *behaviour*, so a dependency bump
fails loudly instead of silently changing bivariate numerics or LDSC-rg
standard errors.
"""

import subprocess
import sys

import numpy as np
import pytest


def test_seam_imports_resolve():
    # The complete borrowed surface, listed explicitly so a partial removal
    # upstream trips a clear failure here rather than an obscure error elsewhere.
    from bipred._ldpred3_compat import (  # noqa: F401
        HAVE_NUMBA,
        _Q8,
        _as_n_vector,
        _check_h2_p,
        _finite_control,
        _get_thread_id,
        _integer_at_least,
        _jit,
        _jit_nogil,
        _jit_parallel,
        _set_threads,
        _validate_beta_hat,
        _validate_blocks,
        _validate_boolean_controls,
        _validate_iterations,
        _validate_seed,
        _weights,
        _wls,
        prange,
    )

    from ldpred3 import LowRankLD  # noqa: F401


def test_deterministic_chain_seeds_are_distinct_and_stable():
    """multichain.py borrows this helper outside the private seam.

    An upstream change silently re-seeds every multi-chain posterior, so the
    contract (distinct uint32 seeds, deterministic in the parent seed) is
    pinned here even though the import is public.
    """
    from ldpred3.diagnostics import deterministic_chain_seeds

    a = deterministic_chain_seeds(17, 4)
    b = deterministic_chain_seeds(17, 4)
    c = deterministic_chain_seeds(18, 4)
    assert a.dtype == np.dtype(np.uint32)
    assert len(set(int(x) for x in a)) == 4
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_q8_int8_scale_is_127():
    # bipred decodes quantised LD as ``R_int8 * (1 / _Q8)`` and the encoder uses
    # ``round(R * _Q8)``. Any change to this constant silently corrupts every
    # int8 block bipred reads, so it is locked here.
    from bipred._ldpred3_compat import _Q8

    assert float(_Q8) == 127.0


def test_public_fit_imports_do_not_load_upstream_sampler_kernels():
    code = """
import sys
from bipred import ldpred3_auto_bivariate_blocks, regional_rg
assert ldpred3_auto_bivariate_blocks is not None
assert regional_rg is not None
assert "ldpred3._kernels" not in sys.modules
assert "ldpred3._inf" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_wls_recovers_exact_linear_fit():
    # ldsc_rg fits its slopes with ``_wls(x, y, w, constrain_intercept)`` and
    # unpacks ``(slope, intercept)``. On an exact line ``y = 2 + 3x`` the fit is
    # analytic for any correct WLS, and the constrained path must hold the
    # intercept fixed while still recovering the slope.
    from bipred._ldpred3_compat import _wls

    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 + 3.0 * x
    w = np.ones_like(x)

    slope, intercept = _wls(x, y, w, None)
    assert slope == pytest.approx(3.0)
    assert intercept == pytest.approx(2.0)

    slope_c, intercept_c = _wls(x, y, w, 2.0)
    assert intercept_c == pytest.approx(2.0)
    assert slope_c == pytest.approx(3.0)


def test_weights_are_positive_and_decreasing():
    # ldsc_rg passes ``_weights(pred_mean, ell_w)`` as the WLS weights, so bipred's
    # rg standard errors depend on them being finite, positive, and down-weighting
    # high-variance / high-LD variants (strictly decreasing in each argument).
    from bipred._ldpred3_compat import _weights

    w_ell = _weights(np.ones(3), np.array([1.0, 2.0, 4.0]))
    assert np.all(np.isfinite(w_ell)) and np.all(w_ell > 0.0)
    assert np.all(np.diff(w_ell) < 0.0)                      # decreasing in ell_w

    w_pred = _weights(np.array([1.0, 2.0, 3.0]), np.ones(3))
    assert np.all(np.isfinite(w_pred)) and np.all(w_pred > 0.0)
    assert np.all(np.diff(w_pred) < 0.0)                     # decreasing in pred_mean


def test_as_n_vector_broadcast_contract():
    # bipred passes a shared scalar N or a per-variant N through
    # ``_as_n_vector(n, m)``: scalars broadcast to length ``m``, a correct-length
    # array passes through unchanged, and a wrong-length array is rejected.
    from bipred._ldpred3_compat import _as_n_vector

    np.testing.assert_array_equal(_as_n_vector(1000.0, 4), np.full(4, 1000.0))
    np.testing.assert_array_equal(
        _as_n_vector(np.array([10.0, 20.0, 30.0, 40.0]), 4),
        np.array([10.0, 20.0, 30.0, 40.0]),
    )
    with pytest.raises((ValueError, IndexError)):
        _as_n_vector(np.array([1.0, 2.0]), 4)


def test_private_ldpred3_imports_stay_centralised():
    """Every underscore-private ldpred3 import must live in ``_ldpred3_compat``.

    The seam only guards what it centralises: a new private import elsewhere in
    the package would drift silently until the next bump. Mirrors the gate
    ``gwfm`` runs for the same reason (gwfm/tests/test_gwfm_public_api.py).
    """
    import pathlib
    import re

    import bipred

    package = pathlib.Path(bipred.__file__).parent
    offenders = []
    for module in sorted(package.glob("*.py")):
        if module.name == "_ldpred3_compat.py":
            continue
        for lineno, line in enumerate(
                module.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            if "ldpred3._" in stripped:
                offenders.append(f"{module.name}:{lineno}: {stripped}")
                continue
            match = re.match(r"from ldpred3\.[\w.]+ import\s*\(?\s*([\w\s,]+)",
                             stripped)
            if match:
                names = match.group(1).replace("\n", " ").split(",")
                private = [n.strip() for n in names if n.strip().startswith("_")]
                if private:
                    offenders.append(f"{module.name}:{lineno}: {stripped}")
    assert offenders == [], (
        "private ldpred3 imports must be centralised in "
        "bipred/_ldpred3_compat.py: " + "; ".join(offenders))
