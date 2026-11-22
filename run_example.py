"""Price a European option by Monte Carlo, compare with Black-Scholes, and plot convergence.

Run from the repository root after `pip install -e .`:
    python examples/run_example.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcpricer import bs_delta, bs_price, bs_vega, mc_asian_price, mc_greeks, mc_price

PARAMS = dict(spot=100.0, strike=105.0, rate=0.03, vol=0.20, maturity=1.0)


def main() -> None:
    bs = bs_price(**PARAMS, option_type="call")
    res = mc_price(**PARAMS, option_type="call", n_paths=1_000_000, seed=42)
    lo, hi = res.ci_95
    print(f"European call  Black-Scholes : {bs:.4f}")
    print(f"               Monte Carlo   : {res.price:.4f}  (SE {res.std_error:.4f}, 95% CI [{lo:.4f}, {hi:.4f}])")

    greeks = mc_greeks(**PARAMS, option_type="call", n_paths=1_000_000, seed=42)
    print(f"Delta          BS {bs_delta(**PARAMS):.4f}   MC {greeks['delta']:.4f}")
    print(f"Vega           BS {bs_vega(**PARAMS):.4f}   MC {greeks['vega']:.4f}")

    asian = mc_asian_price(**PARAMS, option_type="call", n_paths=500_000, n_steps=50, seed=42)
    print(f"Asian call (arithmetic, 50 fixings): {asian.price:.4f}  (SE {asian.std_error:.4f})")

    # Convergence: MC error vs number of paths, against the 1/sqrt(N) rate.
    sizes = np.logspace(2, 6, 17).astype(int)
    errors, std_errors = [], []
    for n in sizes:
        r = mc_price(**PARAMS, option_type="call", n_paths=int(n), seed=1)
        errors.append(abs(r.price - bs))
        std_errors.append(r.std_error)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(sizes, errors, "o-", label="|MC price - Black-Scholes|")
    ax.loglog(sizes, std_errors, "--", label="Monte Carlo standard error")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Error")
    ax.set_title("Monte Carlo convergence (European call)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = Path(__file__).resolve().parents[1] / "docs" / "convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
