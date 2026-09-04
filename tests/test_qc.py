"""LD-consistency screening: does it keep clean variants and catch broken ones."""

import warnings

import numpy as np
import pytest

from bipred.qc import (
    DEFAULT_DROP_FRACTION_WARN,
    dentist, dentist_statistic, implied_sample_size, in_long_range_ld,
    ld_consistency_screen, sd_consistency,
)


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


# --- progress reporting -----------------------------------------------------


