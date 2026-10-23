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
