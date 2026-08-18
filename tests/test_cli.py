"""Command-line wiring for scientific QC and deterministic chain controls."""

from types import SimpleNamespace

import numpy as np
import pytest

import bipred
from bipred import BivariateResult
from bipred.cli import main
from ldpred3 import save_ld_blocks
from ldpred3.weights import read_weights


def _prepared(*, af=None):
    prep = SimpleNamespace(
        blocks=[(np.eye(2), np.arange(2))],
        beta_hat1=np.array([0.01, 0.02]),
        beta_hat2=np.array([-0.01, 0.03]),
        n_eff1=np.full(2, 10_000.0), n_eff2=np.full(2, 12_000.0),
        id=np.array(["a", "b"]), chrom=np.array(["1", "1"]),
        pos=np.array([1, 2]), effect_allele=np.array(["A", "A"]),
        other_allele=np.array(["G", "G"]), af=af,
        log={"n_kept": 2, "n_cache": 3, "n_joint": 2,
             "n_screen_drop": 0}, closed=False)

    def close():
        prep.closed = True

    prep.close = close
    return prep


def _result():
    return BivariateResult(
        beta1_est=np.array([0.01, 0.02]),
        beta2_est=np.array([-0.01, 0.03]), h2=(0.1, 0.2),
        rg=0.3, p=0.02, sigma=np.eye(2))


def test_cli_routes_columns_qc_screen_overlap_and_multichain(monkeypatch):
    calls = {}
    prep = _prepared()

    def prepare(*args, **kwargs):
        calls["prepare"] = (args, kwargs)
        return prep

    def chains(*args, **kwargs):
        calls["chains"] = (args, kwargs)
        diagnostic = SimpleNamespace(rhat={"rg": 1.01})
        return SimpleNamespace(posterior=_result(), basic_split_rhat=diagnostic)

    monkeypatch.setattr(bipred, "prepare_bivariate_sumstats", prepare)
    monkeypatch.setattr(bipred, "ldpred3_auto_bivariate_chains", chains)
    assert main([
        "--ld-cache", "ld.npz", "--sumstats1", "one.tsv",
        "--sumstats2", "two.tsv", "--column1", "beta=B",
        "--column2", "ea=ALT", "--min-n-ratio", "0.8",
        "--min-maf", "0.02", "--min-info", "0.9",
        "--min-af-corr", "0.85", "--screen", "--screen-rounds", "5",
        "--screen-seed", "17", "--screen-ncores", "3",
        "--cross-corr", "0.12", "--n-chains", "4",
        "--chain-ncores", "2", "--seed", "9",
    ]) == 0

    prepare_kwargs = calls["prepare"][1]
    assert prepare_kwargs["columns1"] == {"beta": "B"}
    assert prepare_kwargs["columns2"] == {"ea": "ALT"}
    assert prepare_kwargs["qc_params"] == {
        "min_n_ratio": 0.8, "min_maf": 0.02, "min_info": 0.9,
        "max_chisq": None, "drop_duplicates": True}
    assert prepare_kwargs["min_af_corr"] == 0.85
    assert prepare_kwargs["screen_rounds"] == 5
    assert prepare_kwargs["screen_seed"] == 17
    assert prepare_kwargs["screen_ncores"] == 3
    chain_kwargs = calls["chains"][1]
    assert chain_kwargs["n_chains"] == 4
    assert chain_kwargs["chain_ncores"] == 2
    assert chain_kwargs["cross_corr"] == 0.12
    assert chain_kwargs["seed"] == 9
    assert prep.closed


def test_cli_rejects_scalar_n_eff_with_case_controls(capsys):
    with pytest.raises(SystemExit):
        main([
            "--ld-cache", "ld.npz", "--sumstats1", "one.tsv",
            "--sumstats2", "two.tsv", "--n-eff1", "80000",
            "--n-cases1", "12000", "--n-controls1", "38000",
        ])
    assert "either --n-eff" in capsys.readouterr().err


def test_cli_rejects_hwe_scale_without_cache_af_before_fitting(monkeypatch):
    prep = _prepared(af=None)
    fit_called = False

    monkeypatch.setattr(
        bipred, "prepare_bivariate_sumstats", lambda *args, **kwargs: prep)

    def fit(*args, **kwargs):
        nonlocal fit_called
        fit_called = True
        return _result()

    monkeypatch.setattr(bipred, "ldpred3_auto_bivariate_blocks", fit)
    with pytest.raises(SystemExit):
        main([
            "--ld-cache", "ld.npz", "--sumstats1", "one.tsv",
            "--sumstats2", "two.tsv", "--hwe-frozen-scale",
            "--out-weights1", "one.weights",
        ])
    assert not fit_called
    assert prep.closed


def test_cli_real_mmap_cache_to_target_scaled_weights(tmp_path):
    m = 8
    ids = np.array([f"rs{i}" for i in range(m)])
    cache = tmp_path / "ld.npz"
    save_ld_blocks(
        cache, [(np.eye(m, dtype=np.float32), np.arange(m))], ids,
        mmap=True, counted_allele=np.array(["A"] * m),
        other_allele=np.array(["G"] * m), chrom=np.array(["1"] * m),
        pos=np.arange(1, m + 1), n_ref=500, ridge=0.0)
    paths = [tmp_path / "one.tsv", tmp_path / "two.tsv"]
    for trait, path in enumerate(paths, start=1):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN\n")
            for index, variant in enumerate(ids):
                beta = (index + trait) * 1e-3
                handle.write(
                    f"{variant}\t1\t{index + 1}\tA\tG\t{beta}\t0.01\t10000\n")
    out = tmp_path / "trait1.weights"
    assert main([
        "--ld-cache", str(cache), "--sumstats1", str(paths[0]),
        "--sumstats2", str(paths[1]), "--burn-in", "2",
        "--num-iter", "2", "--seed", "4",
        "--out-weights1", str(out),
    ]) == 0
    weights = read_weights(out)
    assert len(weights) == m
    assert not weights.has_scale
