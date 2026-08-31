"""Public on-ramp: sparse trait preparation, pairing, and write_weights."""

from dataclasses import replace
import warnings

import numpy as np
import pytest

from bipred import (BivariateResult, pair_prepared_traits,
                    prepare_bivariate_sumstats, prepare_trait_sumstats,
                    subset_blocks, ldpred3_auto_bivariate_blocks)
from ldpred3 import LowRankLD, save_ld_blocks
from ldpred3.interop import prepare_ld_cache
from ldpred3.weights import read_weights


def _ar1(k, rho=0.6):
    i = np.arange(k)
    R = rho ** np.abs(i[:, None] - i[None, :])
    return np.ascontiguousarray(R.astype(np.float32))


def test_subset_blocks_retils_to_a_contiguous_cover():
    R0, R1 = _ar1(6), _ar1(6)
    blocks = [(R0, np.arange(6)), (R1, np.arange(6, 12))]
    keep = np.zeros(12, dtype=bool)
    keep[[1, 2, 3, 8, 9, 10]] = True
    tiled, kept = subset_blocks(blocks, keep)
    assert list(kept) == [1, 2, 3, 8, 9, 10]
    assert [tuple(idx) for _, idx in tiled] == [(0, 1, 2), (3, 4, 5)]
    assert tiled[0][0].shape == (3, 3)


def test_subset_blocks_keeps_lowrank_rows_when_rank_fits():
    k = 8
    R = _ar1(k, 0.8).astype(np.float64)
    w, V = np.linalg.eigh(R)
    U = (V[:, -3:] * np.sqrt(np.maximum(w[-3:], 0))).astype(np.float32)
    lr = LowRankLD(U, k, scale=1.0)
    tiled, kept = subset_blocks([(lr, np.arange(k))], [0, 1, 2, 3, 4])
    assert isinstance(tiled[0][0], LowRankLD)
    assert tiled[0][0].U.shape == (5, 3)
    assert list(kept) == [0, 1, 2, 3, 4]


def _write_sumstats(path, ids, a1, a2, beta, se, n, chrom="1", pos=None):
    pos = np.arange(1, len(ids) + 1) if pos is None else pos
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN\n")
        for i, sid in enumerate(ids):
            fh.write(f"{sid}\t{chrom}\t{int(pos[i])}\t{a1[i]}\t{a2[i]}\t"
                     f"{beta[i]:.8g}\t{se[i]:.8g}\t{n}\n")


def _cache_and_sumstats(tmp_path, m=20, *, with_af=True, mmap=False):
    rng = np.random.default_rng(1)
    R = _ar1(m)
    ids = np.array([f"rs{i}" for i in range(m)], dtype=object)
    a1 = np.array(["A"] * m, dtype=object)
    a2 = np.array(["G"] * m, dtype=object)
    chrom = np.array(["1"] * m, dtype=object)
    pos = np.arange(1, m + 1)
    af = np.full(m, 0.3)
    cache = tmp_path / "ld.npz"
    save_ld_blocks(
        str(cache), [(R, np.arange(m))], ids, mmap=mmap,
        counted_allele=a1, other_allele=a2, chrom=chrom, pos=pos,
        reference_af=af if with_af else None, n_ref=500, ridge=0.0)
    n = 10_000
    se = np.full(m, 1.0 / np.sqrt(n))
    b1 = rng.normal(scale=0.02, size=m)
    b2 = 0.6 * b1 + rng.normal(scale=0.015, size=m)
    # Flip trait 2 alleles on a few SNPs so alignment has to sign-flip.
    a1_2, a2_2 = a1.copy(), a2.copy()
    a1_2[::5], a2_2[::5] = a2[::5], a1[::5]
    b2_file = b2.copy()
    b2_file[::5] *= -1
    p1 = tmp_path / "t1.tsv"
    p2 = tmp_path / "t2.tsv"
    _write_sumstats(p1, ids, a1, a2, b1, se, n, pos=pos)
    _write_sumstats(p2, ids, a1_2, a2_2, b2_file, se, n, pos=pos)
    return str(cache), str(p1), str(p2), b1, b2, n, ids, af


def test_prepare_aligns_and_standardizes(tmp_path):
    cache, p1, p2, b1, b2, n, ids, af = _cache_and_sumstats(tmp_path)
    prep = prepare_bivariate_sumstats(cache, p1, p2, n_eff1=n, n_eff2=n, qc=False)
    assert prep.beta_hat1.shape == (20,)
    assert list(prep.id) == list(ids)
    np.testing.assert_array_equal(prep.cache_indices, np.arange(20))
    # Trait 2 was allele-flipped on every 5th SNP; standardized effects
    # should still match the cache-oriented truth.
    from ldpred3 import standardize_betas
    truth2 = standardize_betas(b2, np.full(20, 1.0 / np.sqrt(n)), n)[0]
    assert np.allclose(prep.beta_hat2, truth2, atol=1e-6)
    assert np.allclose(prep.af, af)


def test_prepare_trait_is_sparse_and_sorted_in_cache_order(tmp_path):
    cache, _p1, _p2, b1, _b2, n, ids, _af = _cache_and_sumstats(tmp_path)
    source_order = np.array([9, 2, 15, 0, 7])
    path = tmp_path / "unsorted-trait.tsv"
    _write_sumstats(
        path, ids[source_order], ["A"] * len(source_order),
        ["G"] * len(source_order), b1[source_order],
        np.full(len(source_order), 1.0 / np.sqrt(n)), n,
        pos=source_order + 1)
    events = []
    trait = prepare_trait_sumstats(
        cache, path, n_eff=n, qc=False, label="reusable trait",
        progress=events.append)

    expected = np.sort(source_order)
    np.testing.assert_array_equal(trait.indices, expected)
    assert len(trait) == len(expected) < trait.n_cache == len(ids)
    from ldpred3 import standardize_betas
    truth = standardize_betas(
        b1[expected], np.full(len(expected), 1.0 / np.sqrt(n)), n)[0]
    np.testing.assert_allclose(trait.beta_hat, truth, atol=1e-7)
    np.testing.assert_array_equal(trait.n_eff, np.full(len(expected), n))
    assert trait.indices.flags.c_contiguous and trait.beta_hat.flags.c_contiguous
    assert trait.log["label"] == "reusable trait"
    assert [event["step"] for event in events] == [
        "load LD reference",
        "read, QC, harmonize, and standardize reusable trait",
    ]


def test_pair_prepared_traits_matches_one_shot_preparation(tmp_path):
    cache, _p1, _p2, b1, b2, n, ids, _af = _cache_and_sumstats(tmp_path)
    first = np.array([18, 2, 4, 6, 8, 10, 12, 14, 16, 0, 1])
    second = np.array([17, 1, 4, 6, 8, 10, 12, 14, 16, 2, 3])
    se1 = np.full(len(first), 1.0 / np.sqrt(n))
    se2 = np.full(len(second), 1.0 / np.sqrt(n))
    a1 = np.array(["A"] * len(ids), dtype=object)
    a2 = np.array(["G"] * len(ids), dtype=object)
    a1_2, a2_2 = a1.copy(), a2.copy()
    a1_2[::5], a2_2[::5] = a2[::5], a1[::5]
    b2_file = b2.copy()
    b2_file[::5] *= -1
    p1, p2 = tmp_path / "sparse1.tsv", tmp_path / "sparse2.tsv"
    _write_sumstats(p1, ids[first], a1[first], a2[first], b1[first], se1, n,
                    pos=first + 1)
    _write_sumstats(p2, ids[second], a1_2[second], a2_2[second],
                    b2_file[second], se2, n, pos=second + 1)

    with prepare_ld_cache(cache) as shared:
        one_shot = prepare_bivariate_sumstats(
            shared, p1, p2, n_eff1=n, n_eff2=n, qc=False)
        trait1 = prepare_trait_sumstats(
            shared, p1, n_eff=n, qc=False, label="trait1")
        trait2 = prepare_trait_sumstats(
            shared, p2, n_eff=n, qc=False, label="trait2")
        paired = pair_prepared_traits(shared, trait1, trait2)

        for name in ("beta_hat1", "beta_hat2", "n_eff1", "n_eff2", "id",
                     "chrom", "pos", "effect_allele", "other_allele", "af",
                     "cache_indices"):
            np.testing.assert_array_equal(getattr(paired, name),
                                          getattr(one_shot, name))
        assert paired.log["trait1"] == one_shot.log["trait1"]
        assert paired.log["trait2"] == one_shot.log["trait2"]
        for name in ("n_cache", "n_joint", "n_kept", "n_screen_drop",
                     "screen", "screen_params", "prepared_cache"):
            assert paired.log[name] == one_shot.log[name]
        assert paired.log["prepared_cache"] is True
        assert all(np.isnan(value) for value in paired.log["af_corr"].values())
        assert all(np.isnan(value)
                   for value in one_shot.log["af_corr"].values())
        assert len(paired.blocks) == len(one_shot.blocks) == 1
        np.testing.assert_array_equal(paired.blocks[0][1],
                                      one_shot.blocks[0][1])
        np.testing.assert_array_equal(paired.blocks[0][0],
                                      one_shot.blocks[0][0])


def test_pair_prepared_traits_validates_sparse_cache_indices(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path)
    with prepare_ld_cache(cache) as shared:
        trait1 = prepare_trait_sumstats(
            shared, p1, n_eff=10_000, qc=False, label="trait1")
        trait2 = prepare_trait_sumstats(
            shared, p2, n_eff=10_000, qc=False, label="trait2")

        with pytest.raises(ValueError, match="prepared against 21"):
            pair_prepared_traits(
                shared, replace(trait1, n_cache=21), trait2)
        with pytest.raises(ValueError, match="strictly increasing and unique"):
            pair_prepared_traits(
                shared, replace(trait1, indices=trait1.indices[::-1]), trait2)
        duplicate = trait1.indices.copy()
        duplicate[1] = duplicate[0]
        with pytest.raises(ValueError, match="strictly increasing and unique"):
            pair_prepared_traits(
                shared, replace(trait1, indices=duplicate), trait2)
        outside = trait1.indices.copy()
        outside[-1] = trait1.n_cache
        with pytest.raises(ValueError, match=r"indices must lie in \[0, 20\)"):
            pair_prepared_traits(
                shared, replace(trait1, indices=outside), trait2)
        with pytest.raises(ValueError, match="one-dimensional integer"):
            pair_prepared_traits(
                shared, replace(trait1, indices=trait1.indices.astype(float)),
                trait2)
        bad_z = trait1.z.copy()
        bad_z[0] = np.nan
        with pytest.raises(ValueError, match="z must be finite"):
            pair_prepared_traits(shared, replace(trait1, z=bad_z), trait2,
                                 screen=True)
        bad_eaf = trait1.eaf.copy()
        bad_eaf[0] = 1.1
        with pytest.raises(ValueError, match=r"eaf values must lie in \[0, 1\]"):
            pair_prepared_traits(
                shared, replace(trait1, eaf=bad_eaf), trait2,
                min_af_corr=-1)


def test_prepare_n_cases_uses_ldpred3_n_eff(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path)
    prep = prepare_bivariate_sumstats(
        cache, p1, p2, n_cases2=100, n_controls2=300, n_eff1=10_000, qc=False)
    expected = 4.0 / (1 / 100 + 1 / 300)
    assert np.allclose(prep.n_eff2, expected)


def test_prepare_rejects_scalar_n_eff_with_case_controls(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path)
    with pytest.raises(ValueError, match="scalar n_eff or n_cases"):
        prepare_bivariate_sumstats(
            cache, p1, p2, n_eff1=80_000, n_cases1=12_000, n_controls1=38_000,
            n_eff2=10_000, qc=False)


def test_prepare_names_the_trait_with_no_usable_variants(tmp_path):
    """A trait that loses every variant names itself, for per-trait blame."""
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path)
    m = 20
    off = tmp_path / "off-reference.tsv"
    # Nothing may match: distinct ids, chromosome, positions, and alleles
    # (harmonize falls back to chrom:pos:alleles when ids do not match).
    _write_sumstats(off, [f"off{i}" for i in range(m)], ["C"] * m, ["T"] * m,
                    [0.01] * m, [0.001] * m, 10_000, chrom="2",
                    pos=np.arange(1000, 1000 + m))
    with pytest.raises(ValueError,
                       match="trait1: all GWAS variants were removed"):
        prepare_bivariate_sumstats(cache, str(off), p2, n_eff1=10_000,
                                   n_eff2=10_000, qc=False)
    with pytest.raises(ValueError,
                       match="trait2: all GWAS variants were removed"):
        prepare_bivariate_sumstats(cache, p1, str(off), n_eff1=10_000,
                                   n_eff2=10_000, qc=False)


def test_prepare_accepts_distinct_column_mappings_without_mutating_them(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path)
    path = tmp_path / "trait1-custom.tsv"
    path.write_text(
        (tmp_path / "t1.tsv").read_text(encoding="utf-8").replace(
            "\tN\n", "\tSAMPLES\n", 1),
        encoding="utf-8")
    columns = {"n_eff": "SAMPLES"}
    prep = prepare_bivariate_sumstats(
        cache, path, p2, n_eff2=10_000, columns1=columns, qc=False)
    assert columns == {"n_eff": "SAMPLES"}
    assert np.all(prep.n_eff1 == 10_000)
    with pytest.raises(ValueError, match="either a scalar n_eff or an n_eff column"):
        prepare_bivariate_sumstats(
            cache, path, p2, n_eff1=10_000, n_eff2=10_000,
            columns1=columns, qc=False)


def test_prepare_reads_an_n_eff_column_by_index_name_and_digit_string(tmp_path):
    """columns={"n_eff": 7} must read column 7, not force n_eff=7.0."""
    cache, _, p2, *_ = _cache_and_sumstats(tmp_path)
    m = 20
    n_col = 5_000 + 100 * np.arange(m)          # distinct per variant
    path = tmp_path / "t1-neff-column.tsv"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tSAMPLES\n")
        for i in range(m):
            fh.write(f"rs{i}\t1\t{i + 1}\tA\tG\t0.01\t0.001\t{n_col[i]}\n")
    for override in (7, "7", "SAMPLES", np.int64(7)):
        prep = prepare_bivariate_sumstats(
            cache, str(path), p2, n_eff2=10_000,
            columns1={"n_eff": override}, qc=False)
        # The per-variant column comes through; a forced scalar would make
        # every entry equal.
        np.testing.assert_array_equal(prep.n_eff1, n_col)
    with pytest.raises(ValueError, match="column name or a zero-based"):
        prepare_bivariate_sumstats(
            cache, str(path), p2, n_eff2=10_000,
            columns1={"n_eff": 7.5}, qc=False)


def test_write_weights_uses_hwe_sd_from_cache_af(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path, m=16)
    prep = prepare_bivariate_sumstats(cache, p1, p2, n_eff1=10_000,
                                      n_eff2=10_000, qc=False)
    res = ldpred3_auto_bivariate_blocks(
        prep.blocks, prep.beta_hat1, prep.beta_hat2, prep.n_eff1, prep.n_eff2,
        burn_in=5, num_iter=5, seed=0)
    path = tmp_path / "t1.weights"
    with pytest.warns(RuntimeWarning, match="HWE reference-panel scale"):
        res.write_weights(str(path), trait=1, id=prep.id, chrom=prep.chrom,
                          pos=prep.pos, effect_allele=prep.effect_allele,
                          other_allele=prep.other_allele, af=prep.af)
    wt = read_weights(str(path))
    assert wt.has_scale
    assert np.allclose(wt.weight, res.beta1_est)
    assert np.allclose(wt.sd_ref, np.sqrt(2 * 0.3 * 0.7))


def test_write_weights_rejects_a_length_mismatch():
    res = BivariateResult(
        beta1_est=np.ones(3), beta2_est=np.ones(3), h2=(0.1, 0.1),
        rg=0.0, p=0.02, sigma=np.eye(2))
    with pytest.raises(ValueError, match="provenance"):
        res.write_weights("x", trait=1, id=["a"], chrom=["1"], pos=[1],
                          effect_allele=["A"], other_allele=["G"])


def test_write_weights_refuses_a_diverged_fit_until_allowed(tmp_path):
    res = BivariateResult(
        beta1_est=np.ones(2), beta2_est=np.ones(2), h2=(0.1, 0.1),
        rg=0.0, p=0.02, sigma=np.eye(2),
        divergence_diagnostics={"evaluated": True, "flagged": True})
    ids = ["a", "b"]
    common = dict(
        trait=1, id=ids, chrom=["1", "1"], pos=[1, 2],
        effect_allele=["A", "A"], other_allele=["G", "G"])
    with pytest.raises(ValueError, match="provenance"):
        res.write_weights("x", trait=1, id=["a"], chrom=["1"], pos=[1],
                          effect_allele=["A"], other_allele=["G"])
    with pytest.raises(ValueError, match="divergence diagnostic"):
        res.write_weights(str(tmp_path / "blocked.weights"), **common)
    path = tmp_path / "allowed.weights"
    with pytest.warns(RuntimeWarning, match="divergence diagnostic"):
        res.write_weights(str(path), allow_diverged=True, **common)
    assert path.exists()


def test_unstandardized_z_scores_are_rejected(tmp_path):
    R = _ar1(12)
    z = np.linspace(-2.0, 2.5, 12)
    with pytest.raises(ValueError, match=r"\|beta_hat\| >= 1"):
        ldpred3_auto_bivariate_blocks(
            [(R, np.arange(12))], z, z, 20_000, 20_000,
            burn_in=3, num_iter=3, seed=0)


def test_invalid_sumstat_rows_are_counted_when_qc_is_off(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path, m=8)
    with open(p1, encoding="utf-8") as fh:
        lines = fh.readlines()
    # Zero SE on one otherwise-valid matched row; qc=False used to drop it
    # without recording the loss.
    parts = lines[1].rstrip("\n").split("\t")
    parts[6] = "0"
    lines[1] = "\t".join(parts) + "\n"
    p_bad = tmp_path / "bad.tsv"
    p_bad.write_text("".join(lines), encoding="utf-8")
    from bipred import prepare_trait_sumstats
    trait = prepare_trait_sumstats(cache, p_bad, n_eff=10_000, qc=False)
    assert trait.log["n_harmonized"] == 8
    assert trait.log["n_invalid"] == 1
    assert trait.log["n_matched"] == 7
    assert len(trait) == 7


def test_rsid_match_at_the_wrong_locus_is_dropped(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path, m=6)
    with open(p1, encoding="utf-8") as fh:
        lines = fh.readlines()
    parts = lines[2].rstrip("\n").split("\t")
    parts[2] = "99999"          # same rsID, different BP
    lines[2] = "\t".join(parts) + "\n"
    p_bad = tmp_path / "buildmix.tsv"
    p_bad.write_text("".join(lines), encoding="utf-8")
    from bipred import prepare_trait_sumstats
    trait = prepare_trait_sumstats(cache, p_bad, n_eff=10_000, qc=False)
    assert len(trait) == 5
    dropped = (int(trait.log.get("n_locus_mismatch") or 0)
               + int((trait.log.get("harmonize") or {}).get("n_unmatched") or 0))
    assert dropped >= 1


def test_prepared_trait_rejects_raw_z_as_beta_hat(tmp_path):
    cache, *_ = _cache_and_sumstats(tmp_path, m=8)
    from bipred import PreparedTrait, pair_prepared_traits
    bad = PreparedTrait(
        indices=np.arange(4, dtype=np.int64),
        beta_hat=np.array([0.01, 1.2, -0.03, 0.04]),
        n_eff=np.full(4, 10_000.0),
        z=np.array([1.0, 8.0, -2.0, 0.5]),
        eaf=np.full(4, 0.3),
        n_cache=8,
        log={"label": "z-as-beta"},
    )
    ok = PreparedTrait(
        indices=np.arange(4, dtype=np.int64),
        beta_hat=np.array([0.01, 0.02, -0.03, 0.04]),
        n_eff=np.full(4, 10_000.0),
        z=np.array([1.0, 2.0, -2.0, 0.5]),
        eaf=np.full(4, 0.3),
        n_cache=8,
        log={"label": "ok"},
    )
    with pytest.raises(ValueError, match=r"\|beta_hat\| < 1"):
        pair_prepared_traits(cache, bad, ok)


def test_subset_blocks_strict_indices_sets_singletons_and_views():
    first, second = _ar1(4), _ar1(3)
    blocks = [(first, np.arange(4)), (second, np.arange(4, 7))]

    tiled, kept = subset_blocks(blocks, {1, 6})
    assert kept.tolist() == [1, 6]
    assert [idx.tolist() for _, idx in tiled] == [[0], [1]]
    assert [R.shape for R, _ in tiled] == [(1, 1), (1, 1)]

    whole, kept = subset_blocks(blocks, np.ones(7, dtype=bool))
    assert whole is blocks
    assert whole[0][0] is first
    assert kept.tolist() == list(range(7))
    consecutive, _ = subset_blocks(blocks, [1, 2])
    assert np.shares_memory(consecutive[0][0], first)

    with pytest.raises(ValueError, match="length"):
        subset_blocks(blocks, np.ones(3, dtype=bool))
    with pytest.raises(TypeError, match="integer"):
        subset_blocks(blocks, [1.0])
    with pytest.raises(ValueError, match="duplicate"):
        subset_blocks(blocks, [1, 1])
    with pytest.raises(IndexError, match=r"\[0, 7\)"):
        subset_blocks(blocks, [-1])
    with pytest.raises(IndexError, match=r"\[0, 7\)"):
        subset_blocks(blocks, [7])


def test_prepare_screen_uses_joint_principal_panel_not_zero_filling(
        monkeypatch, tmp_path):
    """A missing neighbour must not become an observed z=0 in the screen."""
    import bipred.qc as qc

    m, n = 120, 10_000
    R = _ar1(m, rho=0.5)
    ids = np.array([f"rs{i}" for i in range(m)], dtype=object)
    alleles1 = np.array(["A"] * m, dtype=object)
    alleles2 = np.array(["G"] * m, dtype=object)
    cache = tmp_path / "screen.npz"
    save_ld_blocks(
        cache, [(R, np.arange(m))], ids,
        counted_allele=alleles1, other_allele=alleles2,
        chrom=np.array(["1"] * m), pos=np.arange(1, m + 1),
        reference_af=np.full(m, 0.3), n_ref=500, ridge=0.0)

    rng = np.random.default_rng(6)
    observed = np.sort(rng.choice(m, 84, replace=False))
    z = rng.normal(size=m)
    se = np.full(observed.size, 1.0 / np.sqrt(n))
    beta = z[observed] * se
    p1, p2 = tmp_path / "screen1.tsv", tmp_path / "screen2.tsv"
    for path in (p1, p2):
        _write_sumstats(
            path, ids[observed], alleles1[observed], alleles2[observed],
            beta, se, n, pos=observed + 1)

    seen = []
    real = qc.ld_consistency_screen

    def spy(blocks, z, **kwargs):
        seen.append((sum(len(idx) for _, idx in blocks), np.asarray(z).size))
        return real(blocks, z, **kwargs)

    monkeypatch.setattr(qc, "ld_consistency_screen", spy)
    prep = prepare_bivariate_sumstats(
        cache, p1, p2, n_eff1=n, n_eff2=n, qc=False,
        screen=True, screen_seed=11)
    # The screen must see the 84 jointly observed variants, not a 120-long
    # vector with missing neighbours filled as z=0.
    assert seen and all(n_sel == n_z == 84 for n_sel, n_z in seen)
    assert prep.log["n_joint"] == 84
    np.testing.assert_array_equal(
        prep.id, ids[np.asarray(prep.cache_indices, dtype=np.int64)])

    trait1 = prepare_trait_sumstats(
        cache, p1, n_eff=n, qc=False, label="trait1")
    trait2 = prepare_trait_sumstats(
        cache, p2, n_eff=n, qc=False, label="trait2")
    events = []
    paired = pair_prepared_traits(
        cache, trait1, trait2, screen=True, screen_seed=11,
        progress=events.append)
    np.testing.assert_array_equal(paired.id, prep.id)
    np.testing.assert_array_equal(paired.beta_hat1, prep.beta_hat1)
    np.testing.assert_array_equal(paired.beta_hat2, prep.beta_hat2)
    assert paired.log["n_joint"] == prep.log["n_joint"]
    assert paired.log["n_screen_drop"] == prep.log["n_screen_drop"]
    assert paired.log["screen_params"] == prep.log["screen_params"]
    assert {event["step"] for event in events} == {
        "LD consistency screen, trait 1",
        "LD consistency screen, trait 2",
    }


def test_prepare_screen_subsets_a_complete_joint_cache(monkeypatch, tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path, mmap=True)

    def screen(blocks, z, **kwargs):
        keep = np.ones(len(z), dtype=bool)
        keep[3] = False
        return keep

    monkeypatch.setattr("bipred.qc.ld_consistency_screen", screen)
    prep = prepare_bivariate_sumstats(
        cache, p1, p2, n_eff1=10_000, n_eff2=10_000,
        qc=False, screen=True)
    assert prep.log["n_joint"] == 20
    assert prep.log["n_screen_drop"] == 1
    assert len(prep.id) == 19 and "rs3" not in prep.id
    np.testing.assert_array_equal(prep.cache_indices,
                                  np.delete(np.arange(20), 3))
    owner = prep._ld_owner
    prep.close()
    assert owner.closed


def test_prepare_keeps_mmap_owner_alive_until_close(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path, mmap=True)
    prep = prepare_bivariate_sumstats(
        cache, p1, p2, n_eff1=10_000, n_eff2=10_000, qc=False)
    owner = prep._ld_owner
    assert owner is not None and not owner.closed
    # The fit reads the mmap-backed view before explicit release.
    result = ldpred3_auto_bivariate_blocks(
        prep.blocks, prep.beta_hat1, prep.beta_hat2,
        prep.n_eff1, prep.n_eff2, burn_in=2, num_iter=2, seed=0)
    assert np.all(np.isfinite(result.beta1_est))
    prep.close()
    assert owner.closed and prep.blocks == []


def test_prepare_reuses_a_caller_owned_validated_cache(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path, mmap=True)
    with prepare_ld_cache(cache) as shared:
        prep = prepare_bivariate_sumstats(
            shared, p1, p2, n_eff1=10_000, n_eff2=10_000, qc=False)
        assert prep.log["prepared_cache"] is True
        assert prep._ld_owner is None
        prep.close()
        assert not shared.closed
        assert len(prep.blocks) == 1
    assert shared.closed


def test_missing_cache_af_writes_safe_target_scaled_weights(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path, with_af=False)
    prep = prepare_bivariate_sumstats(
        cache, p1, p2, n_eff1=10_000, n_eff2=10_000, qc=False)
    assert prep.af is None
    res = BivariateResult(
        beta1_est=np.full(len(prep.id), 0.01),
        beta2_est=np.full(len(prep.id), -0.01), h2=(0.1, 0.1),
        rg=0.0, p=0.02, sigma=np.eye(2))
    path = tmp_path / "target.weights"
    res.write_weights(
        path, trait=1, id=prep.id, chrom=prep.chrom, pos=prep.pos,
        effect_allele=prep.effect_allele, other_allele=prep.other_allele,
        af=prep.af)
    assert not read_weights(path).has_scale


def test_prepare_reports_each_step_and_the_screens_blocks(tmp_path):
    cache, p1, p2, _b1, _b2, n, _ids, _af = _cache_and_sumstats(tmp_path)
    events = []
    with pytest.warns(RuntimeWarning, match="never entered a window"):
        prep = prepare_bivariate_sumstats(
            cache, p1, p2, n_eff1=n, n_eff2=n, qc=False, screen=True,
            screen_rounds=1, screen_seed=3, progress=events.append)
    steps = [e["step"] for e in events if e["unit"] == "step"]
    assert steps == ["load LD reference", "read and QC trait 1",
                     "read and QC trait 2",
                     "harmonize against the LD reference",
                     "LD consistency screen"]
    assert [e["done"] for e in events if e["unit"] == "step"] == [0, 1, 2, 3, 4]
    assert {e["total"] for e in events if e["unit"] == "step"} == {5}
    per_block = [e for e in events if e["unit"] == "block"]
    assert {e["step"] for e in per_block} == {
        "LD consistency screen, trait 1", "LD consistency screen, trait 2"}
    # The reporting controls must not leak into the serialised provenance.
    import json
    json.dumps(prep.log["screen_params"])
    assert "progress" not in prep.log["screen_params"]


def test_prepare_without_a_screen_reports_four_steps(tmp_path):
    cache, p1, p2, _b1, _b2, n, _ids, _af = _cache_and_sumstats(tmp_path)
    events = []
    prepare_bivariate_sumstats(cache, p1, p2, n_eff1=n, n_eff2=n, qc=False,
                               progress=events.append)
    assert {e["total"] for e in events} == {4}
    assert [e["step"] for e in events][-1] == "harmonize against the LD reference"


def test_prepare_rejects_a_non_callable_progress(tmp_path):
    cache, p1, p2, _b1, _b2, n, _ids, _af = _cache_and_sumstats(tmp_path)
    with pytest.raises(TypeError, match="callable"):
        prepare_bivariate_sumstats(cache, p1, p2, n_eff1=n, n_eff2=n,
                                   progress=42)


def _shifted_positions_sumstats(tmp_path, ids, n, *, keep=slice(None),
                                shift=0, name="shifted.tsv"):
    """One trait file whose rows optionally sit at the wrong coordinates."""
    ids = np.asarray(ids, dtype=object)[keep]
    pos = np.arange(1, len(np.asarray(ids)) + 1)
    path = tmp_path / name
    _write_sumstats(
        path, ids, ["A"] * len(ids), ["G"] * len(ids),
        np.full(len(ids), 0.01), np.full(len(ids), 1.0 / np.sqrt(n)), n,
        pos=pos + shift)
    return path


def test_full_reference_coverage_is_recorded_without_a_warning(tmp_path):
    cache, p1, _p2, *_ = _cache_and_sumstats(tmp_path, m=60)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trait = prepare_trait_sumstats(cache, p1, n_eff=10_000, qc=False)
    overlap = trait.log["reference_overlap"]
    assert overlap["n_matched"] == overlap["n_cache"] == 60
    assert overlap["frac_of_reference"] == 1.0
    # The unmatched-row diagnosis costs an index over the whole reference and
    # must not be paid when coverage is fine.
    assert overlap["diagnosed"] is False
    assert not [w for w in caught if "LD-reference" in str(w.message)]


def test_partial_build_shift_warns_and_names_the_genome_build(tmp_path):
    """The 99%-lost-on-coordinates case, which used to be entirely silent."""
    cache, _p1, _p2, _b1, _b2, n, ids, _af = _cache_and_sumstats(
        tmp_path, m=60)
    # Every row but the first six sits at a shifted coordinate, as a GRCh38
    # file does against a GRCh37 reference.
    path = tmp_path / "buildshift.tsv"
    pos = np.arange(1, 61)
    pos[6:] += 137
    _write_sumstats(
        path, ids, ["A"] * 60, ["G"] * 60, np.full(60, 0.01),
        np.full(60, 1.0 / np.sqrt(n)), n, pos=pos)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trait = prepare_trait_sumstats(cache, path, n_eff=n, qc=False,
                                       label="trait")
    overlap = trait.log["reference_overlap"]
    assert overlap["n_matched"] == 6
    assert overlap["frac_of_reference"] == pytest.approx(0.1)
    assert overlap["diagnosed"] is True
    assert overlap["n_unmatched_rows"] == 54
    assert overlap["n_unmatched_id_elsewhere"] == 54
    assert overlap["n_unmatched_id_absent"] == 0
    assert overlap["build_mismatch_suspected"] is True
    messages = [str(w.message) for w in caught
                if issubclass(w.category, RuntimeWarning)]
    assert any("genome-build mismatch" in m and "GRCh38" in m
               for m in messages), messages
    # The old n_locus_mismatch path cannot see this: harmonize rejects the
    # row before bipred can compare loci.
    assert trait.log["n_locus_mismatch"] == 0


def test_a_hits_only_deposition_warns_without_blaming_the_build(tmp_path):
    cache, _p1, _p2, _b1, _b2, n, ids, _af = _cache_and_sumstats(
        tmp_path, m=60)
    path = _shifted_positions_sumstats(
        tmp_path, ids, n, keep=slice(0, 4), name="hitsonly.tsv")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trait = prepare_trait_sumstats(cache, path, n_eff=n, qc=False)
    overlap = trait.log["reference_overlap"]
    assert overlap["n_matched"] == 4
    assert overlap["n_unmatched_rows"] == 0
    assert overlap["build_mismatch_suspected"] is False
    messages = [str(w.message) for w in caught
                if issubclass(w.category, RuntimeWarning)]
    assert any("hits-only" in m for m in messages), messages
    assert not any("genome-build mismatch" in m for m in messages), messages


def test_a_foreign_variant_set_warns_about_absent_identifiers(tmp_path):
    cache, _p1, _p2, _b1, _b2, n, _ids, _af = _cache_and_sumstats(
        tmp_path, m=60)
    # Identifiers the reference does not hold, at coordinates it does not
    # hold either: two different variant sets, not a coordinate error.
    foreign = np.array([f"rs{9_000_000 + i}" for i in range(60)], dtype=object)
    path = _shifted_positions_sumstats(
        tmp_path, foreign, n, shift=500_000, name="foreign.tsv")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="identifier the reference does not hold"):
            prepare_trait_sumstats(cache, path, n_eff=n, qc=False)
    messages = [str(w.message) for w in caught]
    assert any("does not hold at all" in m for m in messages), messages


def test_zero_overlap_from_a_build_shift_says_so_in_the_error(tmp_path):
    cache, _p1, _p2, _b1, _b2, n, ids, _af = _cache_and_sumstats(
        tmp_path, m=60)
    path = _shifted_positions_sumstats(
        tmp_path, ids, n, shift=1_000, name="allshifted.tsv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="check the genome build"):
            prepare_trait_sumstats(cache, path, n_eff=n, qc=False)


def test_reanchoring_recovers_a_build_shifted_trait(tmp_path):
    """The repair path: identifiers agree, coordinates do not."""
    cache, _p1, _p2, _b1, _b2, n, ids, _af = _cache_and_sumstats(
        tmp_path, m=60)
    path = _shifted_positions_sumstats(
        tmp_path, ids, n, shift=137, name="shifted-all.tsv")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="check the genome build"):
            prepare_trait_sumstats(cache, path, n_eff=n, qc=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trait = prepare_trait_sumstats(
            cache, path, n_eff=n, qc=False, label="trait",
            reanchor_on_identifier=True)
    assert len(trait) == 60
    assert trait.log["reference_overlap"]["frac_of_reference"] == 1.0
    log = trait.log["reanchor"]
    assert log["applied"] is True
    assert log["n_anchored"] == log["n_moved"] == 60
    assert log["n_dropped_absent_identifier"] == 0
    messages = [str(w.message) for w in caught]
    assert any("re-anchored 60 of 60 rows" in m for m in messages), messages
    assert any("not a chain-file liftover" in m for m in messages), messages


def test_reanchoring_a_build_matched_trait_moves_nothing(tmp_path):
    cache, p1, _p2, *_ = _cache_and_sumstats(tmp_path, m=60)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plain = prepare_trait_sumstats(cache, p1, n_eff=10_000, qc=False)
        repaired = prepare_trait_sumstats(cache, p1, n_eff=10_000, qc=False,
                                          reanchor_on_identifier=True)
    # Idempotent on correct input: same panel, same effects, no warning.
    np.testing.assert_array_equal(repaired.indices, plain.indices)
    np.testing.assert_allclose(repaired.beta_hat, plain.beta_hat)
    assert repaired.log["reanchor"]["n_moved"] == 0
    assert repaired.log["reanchor"]["n_anchored"] == 60
    assert not [w for w in caught if "re-anchored" in str(w.message)]


def test_reanchoring_refuses_a_positional_match_to_a_different_variant(
        tmp_path):
    """A wrong-variant match is worse than a lost variant."""
    cache, _p1, _p2, _b1, _b2, n, ids, _af = _cache_and_sumstats(
        tmp_path, m=60)
    # Identifiers the reference lacks, sitting exactly on reference positions:
    # harmonize's positional fallback matches them to the wrong variants.
    foreign = np.array([f"rs{9_000_000 + i}" for i in range(60)], dtype=object)
    path = _shifted_positions_sumstats(
        tmp_path, foreign, n, name="colliding.tsv")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loose = prepare_trait_sumstats(cache, path, n_eff=n, qc=False)
        assert len(loose) == 60          # every one a different variant

        with pytest.raises(ValueError, match="no usable variant remains"):
            prepare_trait_sumstats(cache, path, n_eff=n, qc=False,
                                   reanchor_on_identifier=True)


def test_reference_loci_borrows_the_cached_identifier_index(tmp_path):
    """The diagnosis must not build a second full-reference index.

    A private ``{id: {(chrom, pos)}}`` dict over a 1.4M-variant reference cost
    roughly 0.6 GiB per call, and a re-anchored preparation has two call sites
    live at once. ``harmonize`` already memoises an identifier index on the
    variant table, so this asserts the borrowing rather than the byte count.
    """
    from bipred._ldpred3_compat import _variant_indices
    from bipred.prepare import _ReferenceLoci, _cache_variant_table

    cache, *_rest = _cache_and_sumstats(tmp_path, m=12)
    with prepare_ld_cache(cache) as opened:
        variants = _cache_variant_table(opened)
        by_id = _variant_indices(variants)[1]
        first = _ReferenceLoci(variants)
        second = _ReferenceLoci(variants)
        # Same object, not an equal copy: no second index was allocated.
        assert first._by_id is by_id
        assert second._by_id is by_id
        # And it still answers the question the diagnosis asks.
        assert first.loci("rs3") == {("1", 4)}
        assert first.loci("rs9999") is None
