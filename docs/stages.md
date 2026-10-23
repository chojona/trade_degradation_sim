# Staged implementation record

## Stage P0 — numerical correctness

Branch: `codex/p0-numerical-correctness`. Baseline: `0ae0e1e`.

Corrected original-stop RR baseline parity, actual-fill stop distance, minimum RR
invalidation, executable tick rounding and stop rounding, finite/range/grid/input
validation, zero-trial errors, infeasible boundary representation, and unjustified
binary-search monotonicity. Consolidated duplicated Monte Carlo definitions and
removed the silently substituted regime dictionary. Single-run UI now uses its
seed; its threshold advice and distribution binning are corrected. Public labels
say RR retention instead of claiming expected edge.

Boundary API change: no feasible baseline returns `None`; consumers must handle
it. Delay results describe the passing sampled prefix only. Preserved the
absolute-distance gate (including favorable movement), but documented it.

CMake now uses modern Python discovery consistently: the old build could select
Python 3.12 headers/libraries but emit a Python 3.9 extension through pybind11's
legacy discovery. Native validation requires an importable extension explicitly.

Regression suite covers hand-calculated long/short fills, expansion and baseline,
actual stop caps, tick alignment, invalid inputs, infeasible versus zero limits,
nonmonotonic safe islands, seeded Monte Carlo, and complete Python/native results.

No market data, research result, certification, production authorization, or main
backtester behavior is read or changed. No performance claim belongs to P0.

Remaining limits: hypothetical Gaussian assumptions; no commissions or exit-path
model; Monte Carlo probabilities have sampling uncertainty; sampled delay limits
are not continuous safety bounds; Streamlit is optional and not installed by the
numerical workflow. Proceed to measured scalar benchmarking after tests pass.

P0 validation (2026-10-23): **35 tests passed, no skips**, with the Release C++
extension loaded under Python 3.12.13. CLI native/boundary smoke passed and emitted
`null` for an infeasible slippage baseline. UI adapter smoke verified seeded
reruns, infeasible display paths, and histogram counts (not browser rendering).
Python source parsing and `git diff --check` passed. No benchmark yet. Correctness
checks justify proceeding to P1 scalar measurements.

## Stage P1a — scalar benchmark

Branch: `codex/p1-scalar-benchmarks`, based on P0 commit `7b9e110`.
The runner, raw results and environment are under `benchmarks/`. Four paired
10,000-trial cases had **zero numerical error**; timing used seven repeats and
1,000 individual latency samples per backend/case. Scalar native/Python
throughput ratios were **1.131, 1.115, 0.621, 1.092**. No consistent acceleration
is established, and the short/fixed timings are variable. No live latency claim.

The native functions remain scalar while Python owns the trial loop and sampling.
A bounded batch prototype is justified to test this bottleneck, with common draws
and materialization included in measurements. Remaining risks: single machine,
uncontrolled scheduling/power, uncalibrated model, backend RNG portability and
conversion overhead. Numerical tests still pass; the benchmark changes no model.

## Stage P1b — batched prototype and interface proposal

Branch: `codex/p1-batched-prototype`, based on benchmark commit `cd25e42`.
Prototype source snapshot `b452ace`; raw benchmark under `benchmarks/`.

Added a validated explicit-draw native batch kernel, readable Python reference,
seeded shared draw generator and optional detailed samples. Native inputs are
copied under the GIL; evaluation releases it and uses local state. Percentile
indices use nearest/ties-to-even; aggregation uses compensated summation. No new
native RNG, internal threading, external data, or main-project imports.

Final validation: **43 native-enabled tests passed, no skips**. Python 3.9 fallback
ran 43 tests: **32 passed, 11 native tests skipped**. Tests cover all regimes,
both sides, expansion, rule thresholds, decimal/signed price grids, summaries,
individual sample parity, percentile ties, input validation, concurrent calls and
unchanged input buffers. `git diff --check` passed.

Measured 10,000-trial summary-only throughput improvement over the matching Python
reference: **190.55–195.76x evaluation-only**, **5.05–6.07x sampling-to-summary**.
Detailed output reduces these ratios. Maximum observed detailed numerical error
was `7.106e-15`; no classifications differed. All environment, source hashes,
repeat timings and latency samples are retained. One workstation and synthetic
assumptions only; no historical/live performance conclusion.

`docs/read-only-integration-proposal.md` specifies a proposed approved-export-only
boundary with unchanged eligibility, causal timing, lineage and cost ownership.
No adapter was built or connected. P1 ends here. Further P2 integration requires
explicit human approval. Remaining risks: hypothetical distributions, Python
sampling bottleneck, cross-runtime random-stream/float variation, O(n) percentile
storage, detailed output allocations, and target-workload portability.
