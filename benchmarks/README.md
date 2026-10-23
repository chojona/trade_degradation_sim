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
