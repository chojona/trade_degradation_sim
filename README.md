# Trade Degradation Simulator

A hypothetical **reward/risk retention** calculator. It perturbs a trade plan's
entry using assumed delay/slippage distributions and compares actual RR with the
original RR. It does not estimate win probability, expected profitability,
historical alpha, queue fills, or live order latency. No historical data or broker
connections are included.

## Numerical contract

- Original entry, stop and target must be finite, tick-aligned and correctly
  bracketed. Tick size is positive; delay and slippage are nonnegative.
- Generated fills round adversely to the executable tick grid (up for long,
  down for short). Expanded stops round outward to whole ticks. An expansion
  never shrinks an existing stop to conceal a cap violation.
- The ideal RR always uses the **original stop**, including in C++.
- Minimum RR and the stop cap are hard validity rules. Stop distance is measured
  from the **actual fill**, not the ideal entry. Absolute entry-distance limits
  apply to favorable and adverse moves, preserving that explicit original rule.
- `edge_*` fields remain compatibility names for RR quantities. Prefer
  `rr_retention_pct`. Neither is expected profitability.
- A supplied seed reproduces a given Python random stream. This is not evidence
  that the distribution is calibrated. Regime coefficients are hypothetical.
- Boundary functions return `None`/JSON `null` if the zero point fails. Otherwise
  they return the end of the **zero-connected passing sampled prefix**, stopping
  at the first failed point. No monotonicity is assumed. Slippage spacing is one
  tick; delay spacing is at most 0.1 seconds by default, with the cap included.
  No passing-point guarantee is made between delay samples. `P(bad)` is a Monte
  Carlo point estimate, not a confidence bound; increasing its allowed threshold
  loosens the rule. Equality to the threshold fails.

## Run

```sh
python3 python/simulator.py --side long --E 20000 --S 19995 --T 20020 --tick .25 --delay .2 --regime Normal
python3 python/simulator.py --mc 200 --seed 7 --side long --E 20000 --S 19995 --T 20020 --tick .25 --delay 2 --regime High
```

Add `--backend python` or `--backend cpp` to select single-run/Monte Carlo math;
`auto` uses the extension when available. Boundary helpers use the automatic
backend; their numerical contract is shared. Add `--boundary` for sampled limits.

The optional Streamlit UI runs with `streamlit run python/app.py`. Streamlit is
not required for CLI, native builds or numerical tests. UI reruns use the selected
seed for the single simulation; no probabilistic result is an authorization.

## Native build and tests

C++17, CMake >=3.18, Python development headers and pybind11 v2.12.0 are required.
The default build fetches the declared pybind11 source; no data are downloaded.
Use a fresh build directory when changing Python versions. Choose the same interpreter for build and test (example uses Python 3.12):

```sh
cmake -S cpp -B cpp/build/release -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v python3.12)"
cmake --build cpp/build/release --config Release -j 2
TDS_REQUIRE_CPP=1 python3.12 -B -m unittest discover -s python -p 'test*.py' -v
```

Without a native build, run the test command without `TDS_REQUIRE_CPP`; native
checks are explicitly skipped. Do not interpret skipped tests as native parity.

This project is independent of the main backtester. Nothing here reads historical
research artifacts, held-out outcomes, certification, authorization mappings, or
trading accounts. See `docs/stages.md` for staged validation and limitations.

## Experimental P1 batch prototype

`python/batch.py` provides an explicit Python reference and native batch backend.
Python generates common seeded float64 draws; C++ snapshots them and releases the
GIL for numerical evaluation. `run_batch(..., backend='cpp')` includes sampling;
`evaluate_samples` measures already-generated draws. Detailed samples are optional.
See `docs/read-only-integration-proposal.md` for the proposed future boundary.
There is no connection to the main backtester. See `benchmarks/README.md` for
measured workloads and limits, rather than assuming that native code is faster.
