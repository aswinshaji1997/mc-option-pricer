# mcpricer: a simple Monte Carlo option pricer in Python

A small, readable Monte Carlo pricer for European and Asian options under geometric Brownian motion (GBM),
benchmarked against the closed-form Black-Scholes-Merton price.

![Convergence of the Monte Carlo estimate](docs/convergence.png)

## What it does

| Feature | Where |
|---|---|
| European call/put by simulating `S_T` with the exact GBM solution | `mc_price` |
| Antithetic variates (variance reduction) | `antithetic=True` |
| Standard error and 95% confidence interval for every estimate | `MCResult` |
| Delta and vega by bump-and-reprice with common random numbers | `mc_greeks` |
| Arithmetic-average Asian option (path-dependent, no closed form) | `mc_asian_price` |
| Black-Scholes price, delta and vega as a benchmark | `bs_price`, `bs_delta`, `bs_vega` |
| Tests that check MC against Black-Scholes within statistical tolerance | `tests/` |

## The method

Under the risk-neutral measure the asset follows `dS = (r - q) S dt + sigma S dW`. This has the exact solution

```
S_T = S_0 * exp((r - q - sigma^2 / 2) * T + sigma * sqrt(T) * Z),   Z ~ N(0, 1)
```

so a European option needs no time-stepping. The price is the discounted mean payoff over `N` simulated
paths, `exp(-rT) * mean(payoff(S_T))`. The standard error falls like `1 / sqrt(N)`: a 10x smaller error needs 100x more paths.

**Antithetic variates.** Each draw `Z` is paired with `-Z`. The two payoffs are negatively correlated, so averaging
each pair gives a lower-variance estimate. The pairs, not the individual paths, are treated as the independent samples
when computing the standard error.

**Greeks.** Delta and vega use central finite differences. Reusing the same random numbers in the bumped and base
runs (common random numbers) cancels most of the simulation noise.

## Quick start

```bash
git clone <your-repo-url>
cd mc-option-pricer
pip install -e ".[dev]"
python examples/run_example.py
pytest
```

```python
from mcpricer import mc_price, bs_price

res = mc_price(spot=100, strike=105, rate=0.03, vol=0.20, maturity=1.0,
               option_type="call", n_paths=1_000_000, seed=42)
print(res.price, res.std_error, res.ci_95)
print(bs_price(100, 105, 0.03, 0.20, 1.0, "call"))
```

## Example output

`spot=100, strike=105, r=3%, vol=20%, T=1y`, 1,000,000 paths, seed 42:

```
European call  Black-Scholes : 7.1281
               Monte Carlo   : 7.1340  (SE 0.0103, 95% CI [7.1138, 7.1542])
Delta          BS 0.5024   MC 0.5025
Vega           BS 39.8935   MC 39.9335
Asian call (arithmetic, 50 fixings): 3.2211  (SE 0.0075)
```

The Black-Scholes price lies inside the Monte Carlo confidence interval. Vega is per 1.00 change in volatility.

## Limitations

- Constant volatility and rate (GBM). There is no smile, stochastic volatility or local volatility.
- European and discretely monitored arithmetic Asian payoffs only. No early exercise (American options).
- Pure NumPy; no C++ or hardware acceleration.
- The Asian price is not validated against a closed form. The tests only check that it is positive and below the European call.

## Possible extensions

- Control variate for the Asian option using the geometric-average closed form.
- Heston or SABR dynamics.
- Longstaff-Schwartz for American options.
- Quasi-random (Sobol) sequences.

## License

MIT
