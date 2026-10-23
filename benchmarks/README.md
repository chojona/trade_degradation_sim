# Synthetic computation benchmarks

## Scalar baseline (P1 stage 1)

`scalar-macos-arm64-20261023.json` compares the **P0-corrected** pure-Python and
scalar pybind11 paths on identical seeded synthetic workloads, not the old
mathematically inconsistent implementation. No historical inputs are used.

Environment: Apple M4 / arm64, macOS Darwin 25.6, Python 3.12.13, Apple Clang 21,
CMake 4.2.3, pybind11 v2.12.0, Release `-O3 -DNDEBUG`. Exact compiler, runtime,
source/extension hashes, pybind commit and flags are in the JSON. One process and
one Python thread; scheduling, thermal and power state were not controlled.

Each of four cases uses 10,000 trials per repeat, seven repeats after warmup,
and 1,000 separate clock-instrumented individual latency samples per backend.
Throughput uses the median repeat duration. Backend order alternates by case;
repeats within a backend remain sequential. Raw repeat times are retained.
The timed loop includes sampling, validation, result construction and crossings.

| Scenario | Python trials/s | Scalar C++ trials/s | C++ / Python throughput | Python p50 ns | C++ p50 ns |
|---|---:|---:|---:|---:|---:|
| Long, fixed stop | 127,102 | 143,759 | 1.131 | 7,292 | 7,167 |
| Long, expanded stop | 107,251 | 119,600 | 1.115 | 9,375 | 8,584 |
| Short, fixed stop | 124,995 | 77,566 | 0.621 | 8,292 | 7,208 |
| Short, expanded stop | 104,908 | 114,589 | 1.092 | 9,750 | 8,959 |

Numerical result: zero maximum absolute error over 40,000 paired trials; timed RR
retention checksums agree exactly. Raw counts, entry sums, RR sums, first results,
and latency p50/p95/p99 are retained. Individual latency includes timer overhead;
no overhead subtraction or confidence interval is claimed.

**Decision:** no consistent scalar-extension speed advantage is established.
The short/fixed throughput regression despite lower separately sampled latency
illustrates run variability; do not infer universal speedups from these numbers.
A limited batched prototype is justified to test whether removing the Python
trial loop and repeated binding calls helps. It is not justified for deployment.
Compare both evaluation-only and sampling-plus-summary workloads, including
conversion/materialization, and retain the Python implementation as reference.

Reproduce (exclusive output creation):

```sh
python3.12 -B benchmarks/benchmark_scalar.py --output /tmp/tds-scalar-new.json
```

The runner reads only synthetic specifications, source/build metadata and local
hardware information. It requires the Release extension. Hardware inspection may
require normal terminal permissions outside the agent sandbox.

## Batch prototype (P1 stage 2)

`batch-macos-arm64-20261023.json` records the same machine/runtime, source snapshot
`b452ace`, and a rebuilt extension. Compile flags were
`-O3 -DNDEBUG -std=gnu++17 -fPIC -fvisibility=hidden -flto`; link uses `-flto`.
No fast-math or native CPU tuning flag was enabled. Additional native and Python
batch source hashes are recorded. The tree was clean when measurement began.

The prototype owns evaluation and aggregation in C++, but deliberately leaves
seeded random draw generation in Python. Both backends receive identical arrays.
Evaluation-only timings include plan conversion, input snapshots, native calls,
summary construction and any requested sample materialization. Sampling-to-summary
timings additionally include creation of the seeded RNG and all draws. Both
implementations produce the same summary and optional detailed sample columns.
This compares a readable corrected reference with the native prototype, not the
best conceivable optimized Python/NumPy implementation.

Seven throughput repeats alternate backend order. Small requests run multiple
batches per repeat. Separate individual latency samples number 101 for n<=100
and 21 for large batches; p95 is descriptive, not a tail-latency guarantee. Raw
repeat times, all latency samples and numerical checks are retained.

### Summary-only batches of 10,000 trials

| Scenario / timed scope | Python trials/s | C++ trials/s | Ratio | Python request p50 ms | C++ request p50 ms |
|---|---:|---:|---:|---:|---:|
| Long fixed / evaluation | 161,704 | 30,812,748 | 190.55 | 61.299 | 0.310 |
| Short expanded / evaluation | 126,690 | 24,801,034 | 195.76 | 77.780 | 0.386 |
| Long fixed / sampling-to-summary | 128,940 | 650,680 | 5.05 | 77.437 | 15.422 |
| Short expanded / sampling-to-summary | 105,106 | 637,976 | 6.07 | 93.583 | 15.759 |

Detailed 10,000-sample outputs reduce evaluation throughput ratios to 58.45 and
70.52, and sampling-to-summary ratios to 4.87 and 5.76. Output construction and
Python sampling are material costs and must not be omitted from an end-to-end
claim. Summary mode still uses O(n) storage for the defined percentiles.

Single-trial sampling-to-summary request p50 was 11.79–12.46 microseconds native
versus 14.67–16.25 microseconds through the matching batch reference. These are
scenario summaries with a newly seeded RNG, unlike the scalar benchmark's
individual SimResult calls with a reused RNG. Do not compare them as equivalent
single-order paths or interpret them as broker latency.

Numerical differences: maximum summary difference `3.553e-15`, maximum detailed
sample difference `7.106e-15`; no classification differences. Deterministic tests
use absolute tolerance `1e-10`, not a cross-platform bitwise guarantee. Shared
draws are the backend reproducibility boundary; Python RNG portability is not
assumed across runtime versions.

**Decision:** the measured large-batch improvement supports retaining this
standalone prototype and reviewing a read-only P2 adapter proposal. It does not
authorize that adapter, changing the main backtester, empirical calibration,
live execution, or automatic routing. No universal speedup is claimed beyond
these workloads and this build.

Reproduce with a fresh output filename:

```sh
python3.12 -B benchmarks/benchmark_batch.py --output /tmp/tds-batch-new.json
```
