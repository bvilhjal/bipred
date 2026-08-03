"""Chain-level threading for the multi-chain bivariate driver.

The per-block sweep kernels release the GIL, which is what makes running several
chains in threads worthwhile: before that, thread-parallel chains were *slower*
than serial ones because every chain contended for the interpreter lock.
"""

import numpy as np
import pytest

from bipred.bivariate import _bivar_one_sweep_jit, _bivar_one_sweep_lowrank_jit
from bipred.multichain import ldpred3_auto_bivariate_chains

try:
    from bipred._ldpred3_compat import HAVE_NUMBA
except ImportError:                                # pragma: no cover
    HAVE_NUMBA = False


def _panel(nb=3, k=40, n_ind=400, seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    idx = 0
    for _ in range(nb):
        X = rng.standard_normal((k, n_ind)).astype(np.float32)
        X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
        R = (X @ X.T) / n_ind
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        blocks.append((np.ascontiguousarray(R, np.float32),
                       np.arange(idx, idx + k)))
        idx += k
    return blocks, idx


@pytest.fixture(scope="module")
def problem():
    blocks, m = _panel()
    rng = np.random.default_rng(11)
    causal = rng.choice(m, 12, replace=False)
    bt = np.zeros(m)
    bt[causal] = rng.standard_normal(12) * np.sqrt(0.3 / 12)
    n_eff = 2e5
    return (blocks,
            bt + rng.standard_normal(m) / np.sqrt(n_eff),
            bt + rng.standard_normal(m) / np.sqrt(n_eff),
            n_eff)


KW = dict(n_chains=4, seed=7, num_iter=20, burn_in=10)


@pytest.mark.skipif(not HAVE_NUMBA, reason="nogil only applies to the JIT kernels")
def test_sweep_kernels_release_the_gil():
    """Without nogil, threading the chain driver is a pessimisation."""
    for kernel in (_bivar_one_sweep_jit, _bivar_one_sweep_lowrank_jit):
        assert kernel.targetoptions.get("nogil") is True


class TestChainThreading:

    def test_defaults_to_serial(self, problem):
        blocks, b1, b2, n = problem
        result = ldpred3_auto_bivariate_chains(blocks, b1, b2, n, n, **KW)
        assert result.n_chains == 4

    @pytest.mark.parametrize("chain_ncores", [2, 4])
    def test_threaded_is_bit_identical_to_serial(self, problem, chain_ncores):
        """Chains share the LD read-only, own everything they write, and keep
        their own deterministic seeds, so threading cannot change the answer."""
        blocks, b1, b2, n = problem
        serial = ldpred3_auto_bivariate_chains(blocks, b1, b2, n, n,
                                               chain_ncores=1, **KW)
        threaded = ldpred3_auto_bivariate_chains(blocks, b1, b2, n, n,
                                                 chain_ncores=chain_ncores, **KW)
        np.testing.assert_array_equal(threaded.posterior.beta1_est,
                                      serial.posterior.beta1_est)
        np.testing.assert_array_equal(threaded.posterior.beta2_est,
                                      serial.posterior.beta2_est)
        assert threaded.posterior.rg == serial.posterior.rg
        assert threaded.posterior.h2 == serial.posterior.h2

    def test_chain_order_is_preserved(self, problem):
        """Pooling and split-Rhat must see the same chain sequence either way."""
        blocks, b1, b2, n = problem
        serial = ldpred3_auto_bivariate_chains(blocks, b1, b2, n, n,
                                               chain_ncores=1, **KW)
        threaded = ldpred3_auto_bivariate_chains(blocks, b1, b2, n, n,
                                                 chain_ncores=4, **KW)
        assert ([s.seed for s in threaded.chain_summaries]
                == [s.seed for s in serial.chain_summaries])
        for a, b in zip(threaded.chain_summaries, serial.chain_summaries):
            assert a.rg == b.rg and a.h2 == b.h2

    def test_more_workers_than_chains_is_harmless(self, problem):
        blocks, b1, b2, n = problem
        serial = ldpred3_auto_bivariate_chains(blocks, b1, b2, n, n,
                                               chain_ncores=1, **KW)
        wide = ldpred3_auto_bivariate_chains(blocks, b1, b2, n, n,
                                             chain_ncores=64, **KW)
        np.testing.assert_array_equal(wide.posterior.beta1_est,
                                      serial.posterior.beta1_est)

    def test_rejects_nesting_with_block_threading(self, problem):
        """Nesting would oversubscribe the machine."""
        blocks, b1, b2, n = problem
        with pytest.raises(ValueError, match="cannot be combined with ncores"):
            ldpred3_auto_bivariate_chains(blocks, b1, b2, n, n,
                                          chain_ncores=2, ncores=2, **KW)

    def test_block_threading_alone_is_still_allowed(self, problem):
        blocks, b1, b2, n = problem
        ldpred3_auto_bivariate_chains(blocks, b1, b2, n, n,
                                      chain_ncores=1, ncores=2, **KW)

    @pytest.mark.parametrize("bad", [0, -3])
    def test_rejects_invalid_chain_ncores(self, problem, bad):
        blocks, b1, b2, n = problem
        with pytest.raises(ValueError):
            ldpred3_auto_bivariate_chains(blocks, b1, b2, n, n,
                                          chain_ncores=bad, **KW)

    @pytest.mark.parametrize("chain_ncores", [1, 4])
    def test_a_failing_chain_reports_its_index_and_seed(self, problem,
                                                       monkeypatch,
                                                       chain_ncores):
        """An exception raised inside a worker thread must still surface with
        the chain it came from, not as a bare error from the pool."""
        import bipred.multichain as mc
        real = mc._ldpred3_auto_bivariate_prepared
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("synthetic per-chain failure")
            return real(*args, **kwargs)

        monkeypatch.setattr(mc, "_ldpred3_auto_bivariate_prepared", flaky)
        blocks, b1, b2, n = problem
        with pytest.raises(RuntimeError, match=r"chain \d+ \(seed \d+\) failed"):
            ldpred3_auto_bivariate_chains(blocks, b1, b2, n, n,
                                          chain_ncores=chain_ncores, **KW)
