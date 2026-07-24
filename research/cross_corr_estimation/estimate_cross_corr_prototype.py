"""Prototype: estimate the GWAS-overlap noise correlation (``cross_corr``)
*inside* a bivariate summary-statistic Gibbs sampler, jointly with the genetic
correlation.

STATUS: research prototype. This is **not** part of the shipped `bipred` API and
imports nothing from the package -- it is a self-contained proof that the
estimator is identifiable and de-biases r_g, to justify (or not) a future
production feature. See README.md in this directory.

Background. bipred fixes ``cross_corr`` (the cross-trait correlation of the
sampling noise induced by overlapping GWAS samples) as a user input. The
question this settles: can the bivariate Gibbs sampler *estimate* it instead?

Model, per SNP j (infinitesimal bivariate LDpred; the mixture is orthogonal to
the cross_corr update, so the simplest generative model suffices for the proof):

    d_j = beta_j + e_j,   e_j ~ N(0, E),   beta_j ~ N(0, G)
    E = [[1/N1, cc/sqrt(N1 N2)], [cc/sqrt(N1 N2), 1/N2]]   (diagonals FIXED)

where d_j is the LD-residualised marginal, G the 2x2 per-SNP genetic covariance
(free), and cc = cross_corr the noise correlation (free when estimated).

The update. The naive per-SNP residual ``d_j - beta_j`` is a BIASED statistic for
cc: with r_g > 0 the joint per-SNP draw couples the two residuals, and the GWAS
noise is LD-structured (R/N), not per-SNP diagonal. The correct sufficient
statistic whitens the *marginal* residual by the LD Cholesky ``R = L L'``:

    z_t = sqrt(N_t) * L^{-1} (bhat_t - R beta_t)

is i.i.d. bivariate normal with correlation exactly cc, so cc draws from an exact
1-D conditional (a grid over the correlation likelihood that respects the KNOWN
unit-variance diagonals). This is the LDSC-intercept identification (overlap is
LD-flat, genetic covariance scales with LD), expressed as a Gibbs step.

Ground-truth check: simulate two traits with genetic correlation ``RG_TRUE`` and
a known overlap noise correlation ``CC_TRUE``, then compare a sampler that fixes
cc = 0 (bipred today) with one that estimates cc jointly.

Run:  OPENBLAS_NUM_THREADS=1 python research/cross_corr_estimation/estimate_cross_corr_prototype.py
(pure NumPy; a few minutes single-core). Writes ``results.csv``.
"""
import csv
import os

import numpy as np

# ---- configuration ------------------------------------------------------- #
NB, K = 25, 60                     # LD blocks, variants per block  (m = 1500)
M = NB * K
N1 = N2 = 8_000.0                  # modest N so overlap noise actually bites
H1, H2 = 0.5, 0.5                  # SNP heritabilities
RG_TRUE = 0.6                      # true genetic correlation
CC_TRUE = 0.4                      # true overlap noise correlation (the target)
REPS = 5
N_SWEEP, BURN = 1500, 500
SIM_SEED = 0

rng = np.random.default_rng(SIM_SEED)

# block LD: AR(1) with a random rho per block. `chols` factorises the simulation
# covariance; `Lsamp` is the whitening Cholesky the estimator uses.
blocks, chols, Lsamp = [], [], []
for _ in range(NB):
    rho = rng.uniform(0.2, 0.8)
    d = np.abs(np.subtract.outer(np.arange(K), np.arange(K)))
    R = rho ** d
    blocks.append(R)
    chols.append(np.linalg.cholesky(R + 1e-6 * np.eye(K)))
    Lsamp.append(np.linalg.cholesky(R + 1e-8 * np.eye(K)))

# true effects: shared-causal bivariate scaled to (H1, H2) with corr RG_TRUE
Lg = np.array([[1.0, 0.0], [RG_TRUE, np.sqrt(1 - RG_TRUE**2)]])
raw = Lg @ rng.standard_normal((2, M))
beta1_t, beta2_t = raw[0].copy(), raw[1].copy()


def gv(b1, b2):
    return sum(b1[i*K:(i+1)*K] @ (blocks[i] @ b2[i*K:(i+1)*K]) for i in range(NB))


beta1_t *= np.sqrt(H1 / gv(beta1_t, beta1_t))
beta2_t *= np.sqrt(H2 / gv(beta2_t, beta2_t))


def simulate(cc_true, seed):
    """Marginal sumstats with overlap noise correlation cc_true (effects/LD fixed).

    bhat_t = R beta_t + L u_t / sqrt(N_t) with corr(u1_j, u2_j) = cc_true, so the
    cross-trait per-SNP noise covariance is cc_true / sqrt(N1 N2)."""
    r = np.random.default_rng(seed)
    Ln = np.array([[1.0, 0.0], [cc_true, np.sqrt(1 - cc_true**2)]])
    b1 = np.empty(M); b2 = np.empty(M)
    for i in range(NB):
        sl = slice(i*K, (i+1)*K)
        u = Ln @ r.standard_normal((2, K))
        b1[sl] = blocks[i] @ beta1_t[sl] + chols[i] @ u[0] / np.sqrt(N1)
        b2[sl] = blocks[i] @ beta2_t[sl] + chols[i] @ u[1] / np.sqrt(N2)
    return b1, b2


def run(bhat1, bhat2, estimate_cc, seed):
    """Infinitesimal bivariate LDpred Gibbs; optionally estimate cross_corr."""
    r = np.random.default_rng(seed)
    b1 = np.zeros(M); b2 = np.zeros(M)
    rb1 = [blocks[i] @ b1[i*K:(i+1)*K] for i in range(NB)]
    rb2 = [blocks[i] @ b2[i*K:(i+1)*K] for i in range(NB)]
    G = np.array([[H1/M, 0.0], [0.0, H2/M]])
    cc = 0.0
    sd1, sd2 = 1.0/np.sqrt(N1), 1.0/np.sqrt(N2)
    cc_grid = np.linspace(-0.95, 0.95, 191)
    keep_cc, keep_G = [], []

    for sweep in range(N_SWEEP):
        Ginv = np.linalg.inv(G)
        E = np.array([[1.0/N1, cc*sd1*sd2], [cc*sd1*sd2, 1.0/N2]])
        Einv = np.linalg.inv(E)
        V = np.linalg.inv(Ginv + Einv)
        Lv = np.linalg.cholesky(V)
        VEinv = V @ Einv
        Sbeta = np.zeros((2, 2))
        for i in range(NB):
            R = blocks[i]; sl = slice(i*K, (i+1)*K)
            b1b, b2b = b1[sl], b2[sl]
            r1, r2 = rb1[i], rb2[i]
            for j in range(K):
                d1 = bhat1[sl][j] - r1[j] + b1b[j]
                d2 = bhat2[sl][j] - r2[j] + b2b[j]
                mu = VEinv @ np.array([d1, d2])
                draw = mu + Lv @ r.standard_normal(2)
                nb1, nb2 = draw[0], draw[1]
                r1 += R[:, j] * (nb1 - b1b[j])
                r2 += R[:, j] * (nb2 - b2b[j])
                b1b[j], b2b[j] = nb1, nb2
                Sbeta[0, 0] += nb1*nb1; Sbeta[1, 1] += nb2*nb2
                Sbeta[0, 1] += nb1*nb2
        Sbeta[1, 0] = Sbeta[0, 1]

        # genetic covariance G ~ IW(nu0 + M, Psi0 + Sbeta)  (weak prior)
        post_scale = np.eye(2) * 1e-8 + Sbeta
        Lc = np.linalg.cholesky(np.linalg.inv(post_scale))
        A = np.zeros((2, 2))
        A[0, 0] = np.sqrt(r.chisquare(4.0 + M))
        A[1, 1] = np.sqrt(r.chisquare(4.0 + M - 1))
        A[1, 0] = r.standard_normal()
        G = np.linalg.inv(Lc @ A @ A.T @ Lc.T)

        # cross_corr from whitened marginal residuals (exact 1-D conditional)
        if estimate_cc:
            S11 = S22 = S12 = 0.0
            for i in range(NB):
                sl = slice(i*K, (i+1)*K)
                w1 = np.linalg.solve(Lsamp[i], bhat1[sl] - rb1[i]) * np.sqrt(N1)
                w2 = np.linalg.solve(Lsamp[i], bhat2[sl] - rb2[i]) * np.sqrt(N2)
                S11 += w1 @ w1; S22 += w2 @ w2; S12 += w1 @ w2
            g = cc_grid
            ll = -0.5*M*np.log(1 - g*g) - (S11 - 2*g*S12 + S22)/(2*(1 - g*g))
            ll -= ll.max()
            wts = np.exp(ll); wts /= wts.sum()
            cc = float(r.choice(g, p=wts))

        if sweep >= BURN:
            keep_cc.append(cc)
            keep_G.append(G.copy())

    Gm = np.mean(keep_G, axis=0)
    return dict(rg=Gm[0, 1] / np.sqrt(Gm[0, 0] * Gm[1, 1]),
                cc=float(np.mean(keep_cc)))


def main():
    naive_rg, joint_rg, joint_cc, ctrl_cc = [], [], [], []
    for rep in range(REPS):
        bhat1, bhat2 = simulate(CC_TRUE, 100 + rep)
        naive_rg.append(run(bhat1, bhat2, estimate_cc=False, seed=rep)["rg"])
        j = run(bhat1, bhat2, estimate_cc=True, seed=rep)
        joint_rg.append(j["rg"]); joint_cc.append(j["cc"])
        c1, c2 = simulate(0.0, 500 + rep)               # no-overlap control
        ctrl_cc.append(run(c1, c2, estimate_cc=True, seed=rep)["cc"])

    rows = [
        ("rg", "cc_fixed_0", RG_TRUE, np.mean(naive_rg), np.std(naive_rg)),
        ("rg", "cc_estimated", RG_TRUE, np.mean(joint_rg), np.std(joint_rg)),
        ("cross_corr", "cc_estimated", CC_TRUE, np.mean(joint_cc), np.std(joint_cc)),
        ("cross_corr", "control_no_overlap", 0.0, np.mean(ctrl_cc), np.std(ctrl_cc)),
    ]
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["quantity", "sampler", "truth", "mean", "sd"])
        for q, s, t, m, sd in rows:
            w.writerow([q, s, f"{t:.3f}", f"{m:.4f}", f"{sd:.4f}"])

    print(f"TRUTH: rg={RG_TRUE}  cross_corr={CC_TRUE}   ({REPS} reps)\n")
    print(f"cc FIXED at 0 (bipred today):  rg = {np.mean(naive_rg):.3f} +/- {np.std(naive_rg):.3f}"
          f"   (truth {RG_TRUE}, inflated by overlap)")
    print(f"cc ESTIMATED jointly:          rg = {np.mean(joint_rg):.3f} +/- {np.std(joint_rg):.3f}"
          f"   cross_corr = {np.mean(joint_cc):.3f} +/- {np.std(joint_cc):.3f}   (truth {CC_TRUE})")
    print(f"control, no overlap:           cross_corr = {np.mean(ctrl_cc):.3f} +/- {np.std(ctrl_cc):.3f}"
          f"   (truth 0.0)")
    print("\nwrote results.csv")


if __name__ == "__main__":
    main()
