"""Small, self-contained bipred example.

This is an API smoke example, not a calibration benchmark.
"""

import numpy as np

from bipred import ldpred3_auto_bivariate


rng = np.random.default_rng(7)
m = 120
n_eff = 20_000

# Positive-definite AR(1) LD.
position = np.arange(m)
corr = 0.7 ** np.abs(position[:, None] - position[None, :])

# Correlated infinitesimal effects with h2 near 0.3 per trait.
effect_cov = np.array([[1.0, 0.5], [0.5, 1.0]]) * (0.3 / m)
beta1, beta2 = rng.multivariate_normal([0.0, 0.0], effect_cov, size=m).T

# Marginal GWAS effects: beta_hat = R @ beta + noise.
chol = np.linalg.cholesky(corr)
beta_hat1 = corr @ beta1 + chol @ rng.normal(size=m) / np.sqrt(n_eff)
beta_hat2 = corr @ beta2 + chol @ rng.normal(size=m) / np.sqrt(n_eff)

result = ldpred3_auto_bivariate(
    corr,
    beta_hat1,
    beta_hat2,
    n_eff,
    n_eff,
    burn_in=100,
    num_iter=100,
    seed=7,
)

print(result)
print("h2:", result.h2)
print("r_g:", result.rg)
print("shared fraction:", result.mixer["frac_shared"])
