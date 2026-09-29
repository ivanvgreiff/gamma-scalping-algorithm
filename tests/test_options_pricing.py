import numpy as np
import pytest

from models.options_pricing import bs_price, delta, gamma, vega, theta, implied_volatility

S, K, T, R, SIGMA = 100.0, 95.0, 0.5, 0.02, 0.3


def test_put_call_parity():
    call = bs_price(S, K, T, R, SIGMA, "call")
    put = bs_price(S, K, T, R, SIGMA, "put")
    assert call - put == pytest.approx(S - K * np.exp(-R * T), abs=1e-10)


def test_greeks_match_finite_differences():
    h = 1e-3
    fd_delta = (bs_price(S + h, K, T, R, SIGMA) - bs_price(S - h, K, T, R, SIGMA)) / (2 * h)
    fd_gamma = (delta(S + h, K, T, R, SIGMA) - delta(S - h, K, T, R, SIGMA)) / (2 * h)
    fd_vega = (bs_price(S, K, T, R, SIGMA + h) - bs_price(S, K, T, R, SIGMA - h)) / (2 * h)
    # theta is dV/dt = -dV/dT
    fd_theta = -(bs_price(S, K, T + h, R, SIGMA) - bs_price(S, K, T - h, R, SIGMA)) / (2 * h)
    assert delta(S, K, T, R, SIGMA) == pytest.approx(fd_delta, rel=1e-6)
    assert gamma(S, K, T, R, SIGMA) == pytest.approx(fd_gamma, rel=1e-5)
    assert vega(S, K, T, R, SIGMA) == pytest.approx(fd_vega, rel=1e-6)
    assert theta(S, K, T, R, SIGMA) == pytest.approx(fd_theta, rel=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_volatility_round_trip(option_type):
    price = bs_price(S, K, T, R, SIGMA, option_type)
    assert implied_volatility(price, S, K, T, R, option_type) == pytest.approx(SIGMA, abs=1e-6)
