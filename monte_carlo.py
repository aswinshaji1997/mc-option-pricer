"""Monte Carlo pricing of options under geometric Brownian motion (GBM).

Under the risk-neutral measure the asset follows

    dS = (r - q) S dt + sigma S dW,

which has the exact solution

    S_T = S_0 * exp((r - q - 0.5 sigma^2) T + sigma sqrt(T) Z),   Z ~ N(0, 1).

The option price is the discounted expected payoff, estimated by the sample mean
over simulated paths. The standard error of that estimate shrinks like 1/sqrt(N).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo estimate with its sampling error."""

    price: float
    std_error: float
    n_paths: int

    @property
    def ci_95(self) -> tuple[float, float]:
        half = Z_95 * self.std_error
        return self.price - half, self.price + half


def _payoff(prices: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(prices - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - prices, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def _draw_normals(n_paths: int, antithetic: bool, rng: np.random.Generator) -> np.ndarray:
    """Standard normal draws; with antithetic variates the second half is -Z of the first."""
    if antithetic:
        half = rng.standard_normal((n_paths + 1) // 2)
        return np.concatenate([half, -half])[:n_paths]
    return rng.standard_normal(n_paths)


def _price_from_normals(
    z: np.ndarray, spot, strike, rate, vol, maturity, option_type, q, antithetic
) -> MCResult:
    drift = (rate - q - 0.5 * vol**2) * maturity
    s_t = spot * np.exp(drift + vol * np.sqrt(maturity) * z)
    disc_payoff = np.exp(-rate * maturity) * _payoff(s_t, strike, option_type)

    if antithetic:
        # Average each (Z, -Z) pair first: the pairs are the independent samples.
        half = len(disc_payoff) // 2
        samples = 0.5 * (disc_payoff[:half] + disc_payoff[half : 2 * half])
    else:
        samples = disc_payoff
    n = len(samples)
    return MCResult(
        price=float(samples.mean()),
        std_error=float(samples.std(ddof=1) / np.sqrt(n)),
        n_paths=len(z),
    )


def mc_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    q: float = 0.0,
    antithetic: bool = True,
    seed: int | None = None,
) -> MCResult:
    """Price a European call or put by Monte Carlo simulation of S_T."""
    rng = np.random.default_rng(seed)
    z = _draw_normals(n_paths, antithetic, rng)
    return _price_from_normals(z, spot, strike, rate, vol, maturity, option_type, q, antithetic)


def mc_greeks(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    option_type: str = "call",
    n_paths: int = 200_000,
    q: float = 0.0,
    antithetic: bool = True,
    seed: int | None = None,
    spot_bump: float = 0.01,
    vol_bump: float = 0.01,
) -> dict[str, float]:
    """Delta and vega by central finite differences with common random numbers.

    Re-pricing with the SAME normal draws in the bumped and base runs cancels most of the
    simulation noise, which is what makes bump-and-reprice usable for Monte Carlo Greeks.
    Delta is per unit of spot; vega is per 1.00 change in volatility.
    """
    rng = np.random.default_rng(seed)
    z = _draw_normals(n_paths, antithetic, rng)

    def price(s, v):
        return _price_from_normals(z, s, strike, rate, v, maturity, option_type, q, antithetic).price

    h_s = spot * spot_bump
    delta = (price(spot + h_s, vol) - price(spot - h_s, vol)) / (2 * h_s)
    vega = (price(spot, vol + vol_bump) - price(spot, vol - vol_bump)) / (2 * vol_bump)
    return {"delta": delta, "vega": vega}


def mc_asian_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 50,
    q: float = 0.0,
    antithetic: bool = True,
    seed: int | None = None,
) -> MCResult:
    """Arithmetic-average Asian option (average over n_steps equally spaced fixings).

    The payoff depends on the whole path, so the full path is simulated step by step.
    There is no closed form for the arithmetic average, which is where Monte Carlo is useful.
    """
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    half_paths = (n_paths + 1) // 2 if antithetic else n_paths
    z = rng.standard_normal((half_paths, n_steps))
    if antithetic:
        z = np.concatenate([z, -z])[:n_paths]

    log_increments = (rate - q - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    paths = spot * np.exp(np.cumsum(log_increments, axis=1))
    disc_payoff = np.exp(-rate * maturity) * _payoff(paths.mean(axis=1), strike, option_type)

    if antithetic:
        half = len(disc_payoff) // 2
        samples = 0.5 * (disc_payoff[:half] + disc_payoff[half : 2 * half])
    else:
        samples = disc_payoff
    return MCResult(
        price=float(samples.mean()),
        std_error=float(samples.std(ddof=1) / np.sqrt(len(samples))),
        n_paths=n_paths,
    )
