import pytest

from mcpricer import bs_delta, bs_price, bs_vega, mc_asian_price, mc_greeks, mc_price

PARAMS = dict(spot=100.0, strike=105.0, rate=0.03, vol=0.2, maturity=1.0)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_matches_black_scholes_within_four_std_errors(option_type):
    res = mc_price(**PARAMS, option_type=option_type, n_paths=400_000, seed=1)
    bs = bs_price(**PARAMS, option_type=option_type)
    assert abs(res.price - bs) < 4 * res.std_error


def test_dividend_yield():
    res = mc_price(**PARAMS, q=0.02, n_paths=400_000, seed=2)
    assert abs(res.price - bs_price(**PARAMS, q=0.02)) < 4 * res.std_error


def test_put_call_parity_in_black_scholes():
    call = bs_price(**PARAMS, option_type="call")
    put = bs_price(**PARAMS, option_type="put")
    forward_pv = PARAMS["spot"] - PARAMS["strike"] * 2.718281828459045 ** (-PARAMS["rate"])
    assert call - put == pytest.approx(forward_pv, abs=1e-9)


def test_antithetic_reduces_standard_error():
    plain = mc_price(**PARAMS, n_paths=200_000, antithetic=False, seed=3)
    anti = mc_price(**PARAMS, n_paths=200_000, antithetic=True, seed=3)
    assert anti.std_error < plain.std_error


def test_seed_reproducibility():
    a = mc_price(**PARAMS, n_paths=10_000, seed=42)
    b = mc_price(**PARAMS, n_paths=10_000, seed=42)
    assert a.price == b.price


def test_error_shrinks_with_more_paths():
    small = mc_price(**PARAMS, n_paths=10_000, seed=5)
    large = mc_price(**PARAMS, n_paths=1_000_000, seed=5)
    assert large.std_error < small.std_error / 5  # ~10x expected (1/sqrt(100))


def test_greeks_match_black_scholes():
    g = mc_greeks(**PARAMS, n_paths=400_000, seed=7)
    assert g["delta"] == pytest.approx(bs_delta(**PARAMS), abs=0.01)
    assert g["vega"] == pytest.approx(bs_vega(**PARAMS), rel=0.03)


def test_asian_call_cheaper_than_european_call():
    asian = mc_asian_price(**PARAMS, n_paths=200_000, seed=9)
    assert 0 < asian.price < bs_price(**PARAMS)


def test_invalid_option_type():
    with pytest.raises(ValueError):
        mc_price(**PARAMS, option_type="straddle", n_paths=10)
