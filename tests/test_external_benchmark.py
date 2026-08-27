"""Plumbing tests for the external-tool benchmarks (MiXeR / LDSC).

These cover the parsers and the probe/fallback behaviour in
``benchmarks/_external_common.py`` plus the truth-simulation invariants in
``benchmarks/external_overlap.py``. They need neither MiXeR nor LDSC
installed; the fixtures are a captured (path-scrubbed) LDSC ``--rg`` log and a
schema-faithful MiXeR ``fit2`` JSON.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))

import _external_common as ec                      # noqa: E402
import external_overlap as eo                      # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def test_parse_ldsc_rg_log_sections():
    """The captured log pins the section-aware parse (h2-block intercepts must
    not leak into the cross-trait intercept)."""
    got = ec.parse_ldsc_rg_log(os.path.join(FIXTURES, "ldsc_rg.log"))
    assert got["rg"] == pytest.approx(0.3328)
    assert got["rg_se"] == pytest.approx(0.078)
    assert got["gcov"] == pytest.approx(0.167)
    # The gencov intercept is 80.0668; the h2 blocks carry -49.58 and 232.59.
    assert got["intercept"] == pytest.approx(80.0668)
    assert got["intercept_se"] == pytest.approx(21.0777)
    assert got["h2_1"] == pytest.approx(0.9844)
    assert got["h2_2"] == pytest.approx(0.2558)
    assert got["mean_chi2_1"] == pytest.approx(428.704)
    assert got["mean_chi2_2"] == pytest.approx(363.1214)


def test_parse_ldsc_rg_log_missing_section_raises():
    bad = os.path.join(FIXTURES, "_bad_ldsc.log")
    with open(bad, "w") as fh:
        fh.write("no estimates here\n")
    try:
        with pytest.raises(ValueError):
            ec.parse_ldsc_rg_log(bad)
    finally:
        os.remove(bad)


def test_parse_mixer_fit2_json_scalar_ci():
    got = ec.parse_mixer_fit2_json(os.path.join(FIXTURES, "mixer_fit2.json"))
    # pi1u = pi_unique1 + pi_shared
    assert got["pi1"] == pytest.approx(0.01)
    assert got["pi2"] == pytest.approx(0.01)
    assert got["pi11"] == pytest.approx(0.005)
    assert got["rho_beta"] == pytest.approx(0.55)
    assert got["rho_zero"] == pytest.approx(0.0)
    assert got["rg"] == pytest.approx(0.275)
    assert got["h2_1"] == pytest.approx(0.25)
    assert got["h2_2"] == pytest.approx(0.251)


def test_parse_mixer_fit2_json_dict_ci():
    """``ci`` entries may be dicts carrying a point estimate."""
    doc = {"params": {"pi": [0.002, 0.003, 0.001], "rho_beta": -0.4,
                      "rho_zero": 0.05},
           "ci": {"rg": {"point estimate": -0.1},
                  "h2_T1": {"estimate": 0.2}, "h2_T2": {"mean": 0.3}}}
    path = os.path.join(FIXTURES, "_mixer_dict_ci.json")
    with open(path, "w") as fh:
        json.dump(doc, fh)
    try:
        got = ec.parse_mixer_fit2_json(path)
    finally:
        os.remove(path)
    assert got["pi1"] == pytest.approx(0.003)
    assert got["pi2"] == pytest.approx(0.004)
    assert got["pi11"] == pytest.approx(0.001)
    assert got["rho_beta"] == pytest.approx(-0.4)
    assert got["rg"] == pytest.approx(-0.1)
    assert got["h2_1"] == pytest.approx(0.2)
    assert got["h2_2"] == pytest.approx(0.3)


def test_probes_fail_closed_on_missing_tools(monkeypatch):
    monkeypatch.setattr(ec, "LDSC_BIN", "/nonexistent-ldsc-bin")
    monkeypatch.setattr(ec, "MIXER_PY", "/nonexistent-python")
    monkeypatch.setattr(ec, "MIXER_SRC", "/nonexistent-mixer-src")
    monkeypatch.setattr(ec, "MIXER_LIB", "")
    assert ec.probe_ldsc() is False
    assert ec.probe_mixer() is False


def test_run_logged_captures_and_raises(tmp_path):
    log = tmp_path / "ok.log"
    dt = ec.run_logged([sys.executable, "-c", "print('hello')"],
                       timeout=60, log_path=str(log))
    assert dt >= 0 and "hello" in log.read_text()
    with pytest.raises(RuntimeError, match="command failed"):
        ec.run_logged(
            [sys.executable, "-c", "raise SystemExit(3)"], timeout=60)


def test_provenance_records_actual_tree_state(tmp_path):
    csv_path = tmp_path / "x.csv"
    csv_path.write_text("a\n1\n")
    sidecar = ec.write_external_provenance(
        str(csv_path), tools={"ldsc": "test"}, run_controls={"m": 5})
    record = json.loads(open(sidecar).read())
    assert record["artifact"] == "x.csv"
    assert record["dependency_sources"] == {"ldsc": "test"}
    assert isinstance(record["source_clean"], bool)
    assert record["run_controls"] == {"m": 5}


def _toy_blocks(nb=3, k=30, seed=0):
    """Small well-conditioned LD blocks for the truth-simulation invariants."""
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(nb):
        a = rng.standard_normal((200, k))
        r = np.corrcoef(a.T)
        blocks.append(r)
    return blocks


def test_sim_effects_exact_mixture_counts():
    blocks = _toy_blocks()
    m = 3 * 30
    rng = np.random.default_rng(1)
    b1, b2, truth = eo.sim_effects(
        rng, p_causal=0.5, frac_shared=0.5, rho_beta=0.6, blocks_r=blocks)
    n_causal = int(round(0.5 * m))
    assert (b1 != 0).sum() == n_causal
    assert (b2 != 0).sum() == n_causal
    both = (b1 != 0) & (b2 != 0)
    assert both.sum() == int(round(0.5 * n_causal))
    assert truth["pi1"] == pytest.approx(n_causal / m)
    assert truth["pi11"] == pytest.approx(both.sum() / m)
    # Shared effects realize the designed sign of rho_beta.
    assert np.corrcoef(b1[both], b2[both])[0, 1] > 0


def test_sim_effects_h2_scaling_and_realized_rg():
    blocks = _toy_blocks()
    rng = np.random.default_rng(2)
    b1, b2, _ = eo.sim_effects(
        rng, p_causal=0.5, frac_shared=0.5, rho_beta=-0.4, blocks_r=blocks)
    assert eo.gv(blocks, b1, b1) == pytest.approx(eo.H2, rel=1e-6)
    assert eo.gv(blocks, b2, b2) == pytest.approx(eo.H2, rel=1e-6)
    rgr = eo.realized_rg(blocks, b1, b2)
    assert -1.0 <= rgr <= 1.0
    assert rgr < 0.0          # negative design correlation, large causal count


def test_sumstats_pair_noise_scale():
    """The noise part of the analytic sumstats has the 1/N marginal variance
    the model promises."""
    blocks = _toy_blocks(nb=2, k=40)
    blocks_c = [np.linalg.cholesky(r + 1e-8 * np.eye(40)) for r in blocks]
    m = 80
    b1 = np.zeros(m)
    b2 = np.zeros(m)
    n = 50_000
    draws = []
    for seed in range(25):
        bh1, bh2 = eo.sumstats_pair(blocks, blocks_c, b1, b2, n,
                                    np.random.default_rng(seed))
        draws.append((float((bh1 ** 2).mean()), float((bh2 ** 2).mean())))
    mean1 = float(np.mean([d[0] for d in draws]))
    mean2 = float(np.mean([d[1] for d in draws]))
    # Null traits: bh is pure noise, so mean(bh^2) ~ 1/N.
    assert mean1 == pytest.approx(1.0 / n, rel=0.15)
    assert mean2 == pytest.approx(1.0 / n, rel=0.15)


def test_write_gwas_files_columns(tmp_path):
    m = 500
    af = np.full(m, 0.3)
    bh = np.zeros(m)
    bh[::10] = 0.001
    stem = str(tmp_path / "t1")
    eo.write_gwas_files(stem, bh, af, 100_000)
    with open(stem + ".ldsc.txt") as fh:
        head = fh.readline().split()
        row1 = fh.readline().split()
    assert head == ["SNP", "A1", "A2", "BETA", "SE", "N", "P"]
    assert len(row1) == 7
    with open(stem + ".mixer.txt") as fh:
        head = fh.readline().split()
    assert head == ["SNP", "A1", "A2", "N", "Z"]


def test_cell_design_scales_with_m():
    """The cells must stay drawable at any panel size (the CI smoke config is
    far smaller than the committed 40k-SNP run)."""
    for spec in eo.CELLS.values():
        n_causal = max(int(round(spec["p_causal"] * eo.M)), 20)
        need = n_causal + 2 * (n_causal - int(round(spec["frac_shared"]
                                                  * n_causal)))
        assert need <= eo.M
