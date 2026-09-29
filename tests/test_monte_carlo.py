import numpy as np
import pytest

from simulation.monte_carlo import call_price_bs, simulate_delta_hedge_paths_mc


def _price(sigma, S0=100.0, K=100.0, T=1.0):
    return float(np.atleast_1d(call_price_bs(S0, K, 0.0, sigma, T))[0])


def test_replication_has_zero_mean_pnl():
    """With realized vol == implied vol, delta hedging replicates the option: mean P&L ~ 0."""
    res = simulate_delta_hedge_paths_mc(sigma_real=0.2, sigma_imp=0.2, n_steps=365, n_paths=20000, seed=0)
    pnl = res["pnl"]
    assert abs(pnl.mean()) < 4 * pnl.std() / np.sqrt(len(pnl))


@pytest.mark.parametrize("sigma_real", [0.1, 0.3])
def test_mean_pnl_equals_bs_price_difference(sigma_real):
    """Expected P&L of hedging at IV while the market realizes RV is C(RV) - C(IV)."""
    res = simulate_delta_hedge_paths_mc(sigma_real=sigma_real, sigma_imp=0.2, n_steps=365, n_paths=20000, seed=1)
    assert res["pnl"].mean() == pytest.approx(_price(sigma_real) - _price(0.2), abs=0.1)


def test_pnl_attribution_adds_up():
    res = simulate_delta_hedge_paths_mc(r=0.03, sigma_real=0.25, sigma_imp=0.2, n_steps=100,
                                        n_paths=500, seed=2, transaction_cost_bps=5)
    total = (res["payoff_component"] + res["hedge_trading_component"]
             + res["financing_component"] + res["transaction_cost_component"])
    np.testing.assert_allclose(total, res["pnl"], atol=0.05)


def test_hedging_error_shrinks_with_frequency():
    std = [simulate_delta_hedge_paths_mc(n_steps=n, n_paths=5000, seed=3)["pnl"].std() for n in (50, 800)]
    assert std[1] < std[0] / 3  # ~ 1/sqrt(n): sqrt(16) = 4
