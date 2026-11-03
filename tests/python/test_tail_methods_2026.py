"""Python bridge tests for the 2026 tail, EVT, and backtesting methods."""

import numpy as np
import pytest

from fatcrash._core import (
    asymptotic_tail_persistence,
    compare_quantile_forecasts,
    conservative_hill,
    es_e_backtest,
    expectile_var_es,
    hill_estimator,
    lehmer_evi,
    lognormal_gpd_mixture,
    lognormal_gpd_mixture_var_es,
    power_mean_evi,
    reduced_bias_evi,
    second_order_params,
    tail_homogeneity_covariate_test,
    tail_homogeneity_test,
    tail_persistence,
    tail_persistence_rolling,
)


@pytest.fixture
def student_t():
    return np.random.default_rng(0).standard_t(3, size=5000)


def test_conservative_hill_below_hill(student_t):
    alpha, theta = conservative_hill(student_t, delta=0.05)
    assert 0 < theta < 1
    assert alpha == pytest.approx((1 - theta) * hill_estimator(student_t))


def test_power_and_lehmer_reduce_to_hill(student_t):
    xi_hill = 1 / hill_estimator(student_t)
    assert power_mean_evi(student_t, p=1.0) == pytest.approx(xi_hill)
    assert lehmer_evi(student_t, p=1.0) == pytest.approx(xi_hill)


def test_reduced_bias_evi_near_truth(student_t):
    rho, beta = second_order_params(student_t)
    assert rho < 0
    for kind in ["hill", "pm", "nrb_pm", "lehmer"]:
        xi = reduced_bias_evi(student_t, k=500, kind=kind, p=1.0 if kind == "hill" else 0.5 if kind != "lehmer" else 2.0)
        assert 0.15 < xi < 0.55, kind  # true xi = 1/3
    with pytest.raises(ValueError):
        reduced_bias_evi(student_t, kind="nope")


def test_tail_persistence(student_t):
    p = tail_persistence(student_t, u=0.99)
    assert 1.0 < p < 2.5
    assert asymptotic_tail_persistence(0.5) == pytest.approx(2.0)
    assert asymptotic_tail_persistence(-0.1) == 1.0
    roll = tail_persistence_rolling(student_t, window=500)
    assert np.isnan(roll[:499]).all() and np.isfinite(roll[499:]).all()


def test_homogeneity_groups_and_covariate():
    rng = np.random.default_rng(1)
    same = np.concatenate([rng.pareto(3, 3000) + 1, 4 * (rng.pareto(3, 3000) + 1)])
    diff = np.concatenate([rng.pareto(2, 3000) + 1, rng.pareto(5, 3000) + 1])
    groups = np.repeat([0, 1], 3000).astype(np.int64)
    _, p_same, n_tail = tail_homogeneity_test(same, groups, 150, use_abs=False)
    _, p_diff, _ = tail_homogeneity_test(diff, groups, 150, use_abs=False)
    assert n_tail == 300
    assert p_same > 0.05 and p_diff < 0.01
    # Continuous covariate driving alpha: the global top-k version applies.
    # (With two groups, a global threshold keeps mostly the fatter group,
    # which is why the group test above balances the tail per group.)
    x = rng.uniform(size=6000)
    y = rng.uniform(size=6000) ** (-1 / (1.5 + 4 * x))
    _, p_cov, _ = tail_homogeneity_covariate_test(y, x, 300, use_abs=False)
    assert p_cov < 0.01


def test_mixture_and_expectiles(student_t):
    p_body, mu, s, xi, beta, loglik, converged = lognormal_gpd_mixture(student_t)
    assert 0 < p_body < 1 and xi > 0 and np.isfinite(loglik)
    var, es = lognormal_gpd_mixture_var_es(student_t, p=0.99)
    assert es > var > 0
    e, xes, gamma = expectile_var_es(student_t, tau=0.99)
    assert xes > e > 0 and 0 < gamma < 1


def test_backtests():
    rng = np.random.default_rng(2)
    losses = rng.pareto(3, 4000) + 1
    true_es = 1.5 * 0.025 ** (-1 / 3)
    e, reject, z_lo, lam = es_e_backtest(losses, 0.7 * true_es, n_cal=1000)
    assert len(e) == 3000 and reject is not None and 0 < lam <= 0.025
    y = rng.standard_normal(20000)
    lo, up, mean, n_exc, decision = compare_quantile_forecasts(
        y, np.full_like(y, -3.0), np.full_like(y, -1.96), warmup=200
    )
    assert n_exc > 0 and decision is not None and decision[1] == 1
    assert lo[-1] <= mean[-1] <= up[-1]
