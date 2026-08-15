"""LD-consistency screening: does it keep clean variants and catch broken ones."""

import numpy as np
import pytest

from bipred.qc import (
    dentist, dentist_statistic, implied_sample_size, in_long_range_ld,
    ld_consistency_screen, sd_consistency,
)


def _ar1(rho, k):
    pos = np.arange(k)
    return rho ** np.abs(pos[:, None] - pos[None, :])


def _clean_panel(k=400, blocks=3, rho=0.7, seed=0):
    """Blocks plus z-scores drawn from the model DENTIST assumes: z ~ N(R b, R)."""
    rng = np.random.default_rng(seed)
    R = _ar1(rho, k)
    chol = np.linalg.cholesky(R + 1e-8 * np.eye(k))
    blocks_out, z = [], []
    for b in range(blocks):
        beta = np.zeros(k)
        beta[rng.choice(k, 4, replace=False)] = rng.standard_normal(4) * 0.05
        z.append(R @ beta * np.sqrt(20000) + chol @ rng.standard_normal(k))
        blocks_out.append((R.astype(np.float32), np.arange(b * k, (b + 1) * k)))
    return blocks_out, np.concatenate(z)


def test_clean_data_survives_screening():
    """A false-positive rate this test would catch: p=5e-8 should drop ~nobody."""
    blocks, z = _clean_panel()
    keep = dentist(blocks, z, rounds=2)
    assert keep.mean() > 0.99, f"dropped {100*(1-keep.mean()):.1f}% of clean data"


def test_a_sign_flipped_variant_is_caught():
    """The error harmonisation cannot see: an allele flip inside strong LD.

    Its own z stays a plausible size, so no per-variant filter (frequency,
    imputation quality, an LDSC chi-square row cap) can flag it. Only its
    disagreement with the correlated variants around it gives it away.
    """
    blocks, z = _clean_panel(rho=0.9, seed=1)
    victim = 150
    z = z.copy()
    z[victim] = -z[victim] - 8.0            # inconsistent with its neighbours
    keep = dentist(blocks, z, rounds=2)
    assert not keep[victim]


def test_statistic_is_calibrated_under_the_null():
    """T should look like chi2_1 when the z really are consistent with the LD."""
    rng = np.random.default_rng(3)
    k = 300
    R = _ar1(0.6, k)
    chol = np.linalg.cholesky(R + 1e-8 * np.eye(k))
    stats = []
    for _ in range(30):
        z = chol @ rng.standard_normal(k)
        order = rng.permutation(k)
        stats.append(dentist_statistic(R, z, order[:k // 2], order[k // 2:]))
    stats = np.concatenate(stats)
    # chi2_1 has median 0.455; allow generous slack for a finite, LD-correlated
    # sample -- this catches a statistic that is wrong by a factor, not by noise.
    assert 0.2 < np.median(stats) < 1.2, np.median(stats)


@pytest.mark.parametrize("floor", [np.nan, -1e-3, 1.0])
def test_public_statistic_validates_eigenvalue_floor(floor):
    R = np.eye(4)
    with pytest.raises(ValueError, match="eigenvalue_floor"):
        dentist_statistic(R, np.zeros(4), [0, 1], [2, 3],
                          eigenvalue_floor=floor)


def test_public_statistic_validates_indices_and_shapes():
    R = np.eye(4)
    with pytest.raises(ValueError, match="disjoint"):
        dentist_statistic(R, np.zeros(4), [0, 1], [1, 2])
    with pytest.raises(ValueError, match="matching finite vector"):
        dentist_statistic(R, np.zeros((4, 1)), [0, 1], [2, 3])


def test_length_and_finiteness_are_checked():
    blocks, z = _clean_panel(k=100, blocks=2)
    with pytest.raises(ValueError, match="blocks span"):
        dentist(blocks, z[:-1])
    bad = z.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        dentist(blocks, bad)


def test_dense_int8_screen_matches_the_corresponding_float_ld():
    """D8 stores round(127 R), not correlations on their natural scale."""
    blocks, z = _clean_panel(k=100, blocks=1, seed=8)
    R, idx = blocks[0]
    d8 = np.rint(np.clip(R, -1.0, 1.0) * 127.0).astype(np.int8)
    float_keep = ld_consistency_screen(blocks, z, rounds=2, seed=3)
    d8_keep = ld_consistency_screen([(d8, idx)], z, rounds=2, seed=3)
    np.testing.assert_array_equal(d8_keep, float_keep)


def test_dense_float_window_matches_the_default_fit_d32_values():
    from bipred.qc import _window_ld

    R = _ar1(0.37, 8).astype(np.float64)
    R[1, 3] = R[3, 1] = np.nextafter(R[1, 3], 1.0)
    local = np.array([1, 3, 6])
    expected = R.astype(np.float32)[np.ix_(local, local)].astype(np.float64)
    np.testing.assert_array_equal(_window_ld(R, local), expected)


def test_lowrank_float_window_matches_the_default_fit_d32_values():
    from ldpred3 import LowRankLD
    from bipred.qc import _window_ld

    rng = np.random.default_rng(24)
    factor = rng.normal(size=(8, 3)).astype(np.float64) * 0.1
    factor[1, 2] = np.nextafter(factor[1, 2], 1.0)
    factor32 = factor.astype(np.float32)
    residual = (1.0 - np.einsum("ij,ij->i", factor32, factor32)).astype(
        np.float32)
    lowrank = LowRankLD(
        U=factor, m=8, scale=1.0, residual_diag=residual)
    local = np.array([1, 3, 6])
    expected_factor = factor32[local].astype(np.float64)
    expected = expected_factor @ expected_factor.T
    expected[np.diag_indices(local.size)] += residual[local]
    np.testing.assert_array_equal(_window_ld(lowrank, local), expected)


def test_all_random_partitions_run_even_after_empty_ones():
    """A useful fourth split must not be hidden by three empty split results."""
    k = 100
    R = np.eye(k)
    R[0, 1] = R[1, 0] = 0.99
    z = np.zeros(k)
    z[:2] = [-5.0, 5.0]
    blocks = [(R, np.arange(k))]

    # With seed 12 the correlated pair lands on the same side of the first
    # three partitions, so its marginal z^2=25 stays below 29.72. The fourth
    # puts the pair on opposite sides and exposes their disagreement.
    #
    # The comparison is only about the fourth split because a block's rounds
    # are spawned from one parent sequence: spawn(3) is a prefix of spawn(4),
    # so both calls below run the *same* first three rounds.
    keep_three = ld_consistency_screen(blocks, z, rounds=3, seed=12)
    keep_four = ld_consistency_screen(blocks, z, rounds=4, seed=12)
    assert keep_three[:2].all()
    assert not keep_four[:2].any()


def test_rounds_are_spawned_so_a_shorter_run_is_a_prefix():
    """What the test above relies on, pinned directly rather than by seed."""
    parent = np.random.SeedSequence(12).spawn(1)[0]
    three = [s.generate_state(4).tolist()
             for s in np.random.SeedSequence(12).spawn(1)[0].spawn(3)]
    four = [s.generate_state(4).tolist() for s in parent.spawn(4)]
    assert three == four[:3]


def test_pooled_screen_matches_the_serial_one(monkeypatch):
    """Blocks are independent, so ncores must not be able to change the mask.

    The pool is forced on here: whether it would be taken for real depends on
    the BLAS the tests happen to run against, and this is asserting that the
    pooled path computes the same thing, not that it is chosen.

    Forcing it means the gate's other precondition has to be met by hand. The
    pool is only ever entered with BLAS pinned to one thread, and nesting it
    over a multi-threaded BLAS is the thing the gate exists to prevent -- under
    numpy's bundled OpenBLAS at its default thread count that does not fail, it
    segfaults, taking the rest of the session with it. So pin BLAS for the
    duration rather than testing a configuration the library refuses to enter.
    """
    threadpoolctl = pytest.importorskip(
        "threadpoolctl",
        reason="without it the reentrancy gate cannot confirm the BLAS, so "
               "the low-rank pooled path is unreachable in the first place")
    import bipred.qc as qc

    blocks, z = _clean_panel(k=120, blocks=4, rho=0.9, seed=13)
    z = z.copy()
    for victim in (30, 200, 415):                 # spread across three blocks
        z[victim] = -z[victim] - 8.0
    serial = ld_consistency_screen(blocks, z, rounds=3, seed=2)
    assert not serial[[30, 200, 415]].any(), "planted errors must be caught"

    monkeypatch.setattr(qc, "_pool_is_worthwhile", lambda ncores, n: True)
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        for ncores in (2, 3, 5):
            pooled = ld_consistency_screen(blocks, z, rounds=3, seed=2,
                                           ncores=ncores)
            np.testing.assert_array_equal(pooled, serial)


def test_pool_is_refused_without_a_confirmed_reentrant_blas(monkeypatch):
    """eigh is the call that miscomputes there, so a guess must not enable it."""
    import bipred.qc as qc
    import bipred._ldpred3_compat as compat

    monkeypatch.setattr(compat, "_blas_pool_safe", lambda lowrank: False)
    assert qc._pool_is_worthwhile(8, 40) is False
    monkeypatch.setattr(compat, "_blas_pool_safe", lambda lowrank: True)
    assert qc._pool_is_worthwhile(8, 40) is True
    # Nothing to spread, or nobody to spread it over.
    assert qc._pool_is_worthwhile(1, 40) is False
    assert qc._pool_is_worthwhile(8, 1) is False


def test_statistic_matches_the_two_matmul_reference():
    """The single product must agree with forming pinv's factor separately."""
    from bipred.qc import _dentist_statistic

    rng = np.random.default_rng(17)
    k = 240
    R = _ar1(0.75, k)
    z = np.linalg.cholesky(R + 1e-8 * np.eye(k)) @ rng.standard_normal(k)
    order = rng.permutation(k)
    predictors, targets = order[:k // 2], order[k // 2:]

    within, across = R[np.ix_(predictors, predictors)], R[np.ix_(targets,
                                                                 predictors)]
    values, vectors = np.linalg.eigh(within)
    keep = values > 1e-3 * max(float(values.max()), 1e-12)
    retained, vals = vectors[:, keep], values[keep]
    scaled = retained / vals
    predicted = across @ (scaled @ (retained.T @ z[predictors]))
    leverage = np.einsum("ij,ij->i", across @ scaled, across @ retained)
    expected = (z[targets] - predicted) ** 2 / np.clip(1.0 - leverage, 1e-6,
                                                       None)

    got = _dentist_statistic(R, z, predictors, targets, 1e-3)
    # Same computation reassociated, so this is float64 rounding -- and the
    # statistic is compared against a threshold of 29.72.
    np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_dentist_compatibility_name_matches_primary_screen():
    blocks, z = _clean_panel(k=100, blocks=1, seed=9)
    np.testing.assert_array_equal(
        dentist(blocks, z, rounds=2, seed=4),
        ld_consistency_screen(blocks, z, rounds=2, seed=4),
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"rounds": 0}, ValueError, "rounds must be an integer >= 1"),
        ({"rounds": 1.5}, ValueError, "rounds must be an integer >= 1"),
        ({"window": 49}, ValueError, "window must be an integer >= 50"),
        ({"window": True}, ValueError, "window must be an integer >= 50"),
        ({"threshold": np.nan}, ValueError, "threshold must be"),
        ({"threshold": 0.0}, ValueError, "threshold must be"),
        ({"eigenvalue_floor": -1e-3}, ValueError, "eigenvalue_floor must be"),
        ({"eigenvalue_floor": 1.0}, ValueError, "eigenvalue_floor must be"),
        ({"ncores": 0}, ValueError, "ncores must be an integer >= 1"),
        ({"ncores": 2.0}, ValueError, "ncores must be an integer >= 1"),
        ({"ncores": True}, ValueError, "ncores must be an integer >= 1"),
    ],
)
def test_ld_consistency_screen_validates_controls(kwargs, error, message):
    blocks, z = _clean_panel(k=100, blocks=1, seed=10)
    with pytest.raises(error, match=message):
        ld_consistency_screen(blocks, z, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"verbose": "False"},
    {"seed": True},
])
def test_ld_consistency_screen_rejects_control_surrogates(kwargs):
    blocks, z = _clean_panel(k=100, blocks=1, seed=11)
    with pytest.raises(ValueError):
        ld_consistency_screen(blocks, z, **kwargs)


def test_low_rank_blocks_need_no_densifying():
    """A low-rank block is screened through its factor, not a dense expansion."""
    from ldpred3 import lowrank_ld
    k = 300
    R = _ar1(0.7, k)
    factor = lowrank_ld(R, variance=0.99, quantize=True)
    rng = np.random.default_rng(5)
    chol = np.linalg.cholesky(R + 1e-8 * np.eye(k))
    z = chol @ rng.standard_normal(k)
    keep = dentist([(factor, np.arange(k))], z, rounds=1)
    assert keep.shape == (k,)
    assert keep.mean() > 0.95


def _sumstats(n_variants=5000, n_eff=100_000.0, af=None, seed=0, binary=False):
    """Summary statistics whose implied SD matches the reference by construction."""
    rng = np.random.default_rng(seed)
    if af is None:
        af = rng.uniform(0.05, 0.95, n_variants)
    sd_ref = np.sqrt(2 * af * (1 - af))
    scale = 2.0 if binary else 1.0
    # Invert sd_ss = scale / sqrt(n se^2 + beta^2) at sd_ss == sd_ref.
    beta = rng.standard_normal(n_variants) * 1e-3
    se = np.sqrt(np.maximum(scale**2 / sd_ref**2 - beta**2, 1e-12) / n_eff)
    return beta, se, af


def test_sd_consistency_keeps_self_consistent_sumstats():
    beta, se, af = _sumstats()
    keep, offset = sd_consistency(beta, se, np.full(beta.size, 100_000.0), af)
    assert keep.mean() > 0.99, f"dropped {100*(1-keep.mean()):.1f}% of clean data"
    assert 0.9 < offset < 1.1, offset


def test_sd_consistency_flags_a_wrong_sample_size():
    """The error it exists for: N inflated, so the implied SD is too small."""
    beta, se, af = _sumstats(seed=1)
    n_true = np.full(beta.size, 100_000.0)
    # Claim 9x the sample size actually used; nothing else changes. The
    # implied SD falls by sqrt(9) = 3, which clears the 0.5x lower bound --
    # a 4x overstatement lands at exactly 0.5 and is deliberately not caught.
    keep, offset = sd_consistency(beta, se, 9.0 * n_true, af, normalise=False)
    assert 0.3 < offset < 0.4, offset
    assert keep.mean() < 0.05


def test_sd_consistency_normalisation_makes_traits_comparable():
    """A binary trait on a mis-specified N must not be judged on a shifted scale.

    Without normalisation the same threshold means different things for two
    traits: on real CARDIoGRAMplusC4D data the unnormalised ratio sat at 0.755
    while a well-specified trait sat near 1, so tightening the bound removed
    83% of the genome from one and almost nothing from the other.
    """
    beta, se, af = _sumstats(seed=2, binary=True)
    n = np.full(beta.size, 100_000.0)
    raw_keep, offset = sd_consistency(beta, se, 9.0 * n, af, binary=True,
                                      normalise=False)
    norm_keep, _ = sd_consistency(beta, se, 9.0 * n, af, binary=True,
                                  normalise=True)
    assert offset < 0.4                        # the scale really is shifted
    assert raw_keep.mean() < norm_keep.mean()  # normalising rescues them
    assert norm_keep.mean() > 0.9


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("beta", np.nan, "beta contains non-finite"),
        ("se", np.nan, "se must contain finite positive"),
        ("se", 0.0, "se must contain finite positive"),
        ("n_eff", np.inf, "n_eff must contain finite positive"),
        ("n_eff", 0.0, "n_eff must contain finite positive"),
        ("af", np.nan, "af must contain finite values"),
        ("af", 1.01, "af must contain finite values"),
    ],
)
def test_sd_consistency_rejects_nonfinite_and_out_of_range_input(
        field, value, message):
    beta, se, af = _sumstats(n_variants=100)
    values = {
        "beta": beta.copy(),
        "se": se.copy(),
        "n_eff": np.full(beta.size, 100_000.0),
        "af": af.copy(),
    }
    values[field][0] = value
    with pytest.raises(ValueError, match=message):
        sd_consistency(values["beta"], values["se"], values["n_eff"],
                       values["af"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lower": 0.0}, "lower must be finite"),
        ({"lower": 1.01}, "lower must be finite"),
        ({"lower": True}, "lower must be a finite number"),
        ({"upper": -0.01}, "upper must be finite"),
        ({"upper": np.nan}, "upper must be a finite number"),
        ({"upper": False}, "upper must be a finite number"),
    ],
)
def test_sd_consistency_validates_thresholds(kwargs, message):
    beta, se, af = _sumstats(n_variants=100)
    with pytest.raises(ValueError, match=message):
        sd_consistency(beta, se, 100_000.0, af, **kwargs)


@pytest.mark.parametrize("field", ["beta", "se", "af", "n_eff"])
def test_sd_consistency_rejects_column_vector_broadcasts(field):
    beta, se, af = _sumstats(n_variants=12)
    values = {"beta": beta, "se": se, "af": af,
              "n_eff": np.full(beta.size, 100_000.0)}
    values[field] = values[field][:, None]
    with pytest.raises(ValueError):
        sd_consistency(values["beta"], values["se"], values["n_eff"],
                       values["af"])


def test_implied_sample_size_recovers_a_known_n():
    beta, se, af = _sumstats(n_eff=80_000.0, seed=3, binary=True)
    out = implied_sample_size(beta, se, af, binary=True,
                              reported_n=np.full(beta.size, 80_000.0))
    assert abs(out["median"] - 80_000.0) / 80_000.0 < 0.02
    assert out["consistent"] is True


def test_implied_sample_size_catches_an_overstated_n():
    """CAD's case: reported 162,973, implied 92,966, ratio 0.570."""
    beta, se, af = _sumstats(n_eff=92_966.0, seed=4, binary=True)
    reported = np.full(beta.size, 162_973.0)
    out = implied_sample_size(beta, se, af, binary=True, reported_n=reported)
    assert out["consistent"] is False
    assert 0.5 < out["ratio"] < 0.62, out["ratio"]


def test_quantitative_implied_n_is_reported_as_unidentified():
    """Calibrating unknown phenotype scale from reported N proves nothing."""
    beta, se, af = _sumstats(n_eff=80_000.0, seed=5, binary=False)
    for reported in (80_000.0, 720_000.0):
        out = implied_sample_size(beta, se, af, reported_n=reported)
        assert np.isnan(out["median"])
        assert np.isnan(out["ratio"])
        assert out["consistent"] is False


@pytest.mark.parametrize("reported", [0.0, -1.0, np.nan, np.inf])
def test_implied_sample_size_rejects_invalid_reported_n(reported):
    beta, se, af = _sumstats(n_variants=12, binary=True)
    with pytest.raises(ValueError, match="finite positive"):
        implied_sample_size(beta, se, af, binary=True, reported_n=reported)


@pytest.mark.parametrize("field", ["beta", "se", "af"])
def test_implied_sample_size_rejects_column_vector_broadcasts(field):
    beta, se, af = _sumstats(n_variants=12, binary=True)
    values = {"beta": beta, "se": se, "af": af}
    values[field] = values[field][:, None]
    with pytest.raises(ValueError, match="equal-length vectors"):
        implied_sample_size(values["beta"], values["se"], values["af"],
                            binary=True, reported_n=100_000.0)


@pytest.mark.parametrize("field,value,match", [
    ("beta", np.nan, "beta contains non-finite"),
    ("se", np.nan, "se must contain finite positive"),
    ("se", 0.0, "se must contain finite positive"),
    ("af", np.nan, "af must contain finite values"),
    ("af", 0.0, "af must contain finite values"),
    ("af", 1.0, "af must contain finite values"),
    ("af", 2.0, "af must contain finite values"),
])
def test_implied_sample_size_rejects_malformed_rows(field, value, match):
    beta, se, af = _sumstats(n_variants=12, binary=True)
    values = {"beta": beta, "se": se, "af": af}
    values[field] = values[field].copy()
    values[field][1] = value
    with pytest.raises(ValueError, match=match):
        implied_sample_size(values["beta"], values["se"], values["af"],
                            binary=True, reported_n=100_000.0)


def test_long_range_regions_cover_the_expected_span():
    from bipred.qc import APOE_HG19, LONG_RANGE_LD_HG19
    assert len(LONG_RANGE_LD_HG19) == 24            # Price et al. 2008
    chrom = np.array(["6", "6", "19", "1", "22"])
    pos = np.array([30_000_000, 40_000_000, 45_400_000, 50_000_000, 20_000_000])
    with_apoe = in_long_range_ld(chrom, pos)
    without = in_long_range_ld(chrom, pos, include_apoe=False)
    assert with_apoe.tolist() == [True, False, True, True, False]
    # APOE is not one of the 24; excluding it must free chr19 alone.
    assert without.tolist() == [True, False, False, True, False]
    assert APOE_HG19[0] == "19"


def test_qc_boolean_controls_reject_strings():
    beta, se, af = _sumstats(n_variants=100, binary=True)
    with pytest.raises(ValueError, match="binary must be a boolean"):
        implied_sample_size(beta, se, af, binary="False")
    with pytest.raises(ValueError, match="normalise must be a boolean"):
        sd_consistency(beta, se, 100_000.0, af, normalise="False")
    with pytest.raises(ValueError, match="include_apoe must be a boolean"):
        in_long_range_ld(["1"], [1], include_apoe="False")


def test_blocked_screen_parallelism_explains_itself(monkeypatch):
    """Requesting ncores>1 and silently getting serial is a no-op flag.

    The conservative BLAS gate is correct; the missing piece was that it never
    said why. Each blocked cause must name itself, and an open gate (or
    ncores=1) must stay quiet.
    """
    import warnings as warnings_module

    import bipred.qc as qc
    import bipred._ldpred3_compat as compat

    blocks, z = _clean_panel(k=100, blocks=2, rho=0.9, seed=5)
    mask = None

    monkeypatch.setattr(compat, "_blas_runtime_info", lambda: (4, True))
    with pytest.warns(RuntimeWarning, match="4 threads"):
        mask = ld_consistency_screen(blocks, z, rounds=1, seed=3, ncores=4)
    assert mask.dtype == bool and mask.size == z.size

    monkeypatch.setattr(compat, "_blas_runtime_info", lambda: (None, None))
    with pytest.warns(RuntimeWarning, match="threadpoolctl"):
        ld_consistency_screen(blocks, z, rounds=1, seed=3, ncores=2)

    monkeypatch.setattr(compat, "_blas_runtime_info", lambda: (1, False))
    with pytest.warns(RuntimeWarning, match="not reentrant"):
        ld_consistency_screen(blocks, z, rounds=1, seed=3, ncores=2)

    monkeypatch.setattr(qc, "_pool_is_worthwhile", lambda ncores, n: True)
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error")
        ld_consistency_screen(blocks, z, rounds=1, seed=3, ncores=4)

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error")
        ld_consistency_screen(blocks, z, rounds=1, seed=3)
