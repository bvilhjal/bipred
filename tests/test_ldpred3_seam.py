"""Semantic guards on the private ldpred3 symbols bipred reaches into.

bipred builds on a set of underscore-prefixed names from ldpred3's *internal*
modules (``ldpred3.ldpred3``, ``ldpred3._kernels``, ``ldpred3.ldsc``); see the
seam comment in ``bipred/bivariate.py`` and the Notes in ``CHANGELOG.md``. The
install pins an exact ldpred3 commit precisely because that surface is private
and unversioned. Importing ``bipred`` already forces those imports to *resolve*;
these tests guard their *behaviour*, so an ldpred3 bump that changes what a
borrowed helper does fails loudly here instead of silently changing bivariate
numerics or LDSC-rg standard errors.
"""

import numpy as np
import pytest


def test_seam_imports_resolve():
    # The complete borrowed surface, listed explicitly so a partial removal
    # upstream trips a clear failure here rather than an obscure error elsewhere.
    from ldpred3.ldpred3 import (  # noqa: F401
        HAVE_NUMBA,
        LowRankLD,
        _as_n_vector,
        _check_h2_p,
        _finite_control,
        _integer_at_least,
        _jit,
        _jit_parallel,
        _set_threads,
        _validate_beta_hat,
        _validate_blocks,
        _validate_boolean_controls,
        _validate_iterations,
        _validate_seed,
        prange,
    )
    from ldpred3._kernels import _Q8  # noqa: F401
    from ldpred3.ldsc import _wls, _weights  # noqa: F401


def test_q8_int8_scale_is_127():
    # bipred decodes quantised LD as ``R_int8 * (1 / _Q8)`` and the encoder uses
    # ``round(R * _Q8)``. Any change to this constant silently corrupts every
    # int8 block bipred reads, so it is locked here.
    from ldpred3._kernels import _Q8

    assert float(_Q8) == 127.0


def test_wls_recovers_exact_linear_fit():
    # ldsc_rg fits its slopes with ``_wls(x, y, w, constrain_intercept)`` and
    # unpacks ``(slope, intercept)``. On an exact line ``y = 2 + 3x`` the fit is
    # analytic for any correct WLS, and the constrained path must hold the
    # intercept fixed while still recovering the slope.
    from ldpred3.ldsc import _wls

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
    from ldpred3.ldsc import _weights

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
    from ldpred3.ldpred3 import _as_n_vector

    np.testing.assert_array_equal(_as_n_vector(1000.0, 4), np.full(4, 1000.0))
    np.testing.assert_array_equal(
        _as_n_vector(np.array([10.0, 20.0, 30.0, 40.0]), 4),
        np.array([10.0, 20.0, 30.0, 40.0]),
    )
    with pytest.raises((ValueError, IndexError)):
        _as_n_vector(np.array([1.0, 2.0]), 4)
