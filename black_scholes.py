"""Closed-form Black-Scholes-Merton prices and Greeks (used as a benchmark)."""

from __future__ import annotations

from math import erf, exp, log, pi, sqrt


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _d1_d2(spot: float, strike: float, rate: float, vol: float, maturity: float, q: float):
    d1 = (log(spot / strike) + (rate - q + 0.5 * vol**2) * maturity) / (vol * sqrt(maturity))
    return d1, d1 - vol * sqrt(maturity)


def bs_price(spot, strike, rate, vol, maturity, option_type="call", q=0.0) -> float:
    """Black-Scholes price of a European call or put with continuous dividend yield q."""
    d1, d2 = _d1_d2(spot, strike, rate, vol, maturity, q)
    disc_s = spot * exp(-q * maturity)
    disc_k = strike * exp(-rate * maturity)
    if option_type == "call":
        return disc_s * _norm_cdf(d1) - disc_k * _norm_cdf(d2)
    if option_type == "put":
        return disc_k * _norm_cdf(-d2) - disc_s * _norm_cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_delta(spot, strike, rate, vol, maturity, option_type="call", q=0.0) -> float:
    d1, _ = _d1_d2(spot, strike, rate, vol, maturity, q)
    call_delta = exp(-q * maturity) * _norm_cdf(d1)
    return call_delta if option_type == "call" else call_delta - exp(-q * maturity)


def bs_vega(spot, strike, rate, vol, maturity, q=0.0) -> float:
    """Vega per 1.00 (i.e. 100 vol points) change in volatility; same for calls and puts."""
    d1, _ = _d1_d2(spot, strike, rate, vol, maturity, q)
    return spot * exp(-q * maturity) * _norm_pdf(d1) * sqrt(maturity)
