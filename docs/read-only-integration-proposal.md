# Read-only integration interface proposal — P2 NOT authorized

Status: proposal only. No adapter is installed in or imported by the main
backtester. No historical file, research result, held-out outcome, certification,
production authorization map or account is accessed by this prototype.

## Boundary and ownership

The main backtester remains owner of selection, calendars, raw executable prices,
features, causal availability, research eligibility, order lifecycle, commissions,
slippage, positions and risk. A future adapter would accept only a separately
approved export of hypothetical trade plans and return diagnostic scenarios.
It would have no engine callback, broker transport, order-submission method,
filesystem discovery, dataset loader or certification update capability.

Only a human-approved export may supply plans. An adapter must never enumerate
research directories or open sealed manifests/outcomes. The export must exclude
future labels, future realized volatility and held-out outcomes. Hashing or
copying a held-out outcome is not permitted as a shortcut to lineage validation.

## Proposed request schema: `tds-readonly-scenario-v1`

| Field | Meaning / validation |
|---|---|
| `request_id`, `schema_version` | Unique caller identity and exact supported schema |
| `eligibility` | `synthetic`, `strict`, `conditional`, or `prospective`; copied from approved export, never promoted |
| `research_release_id`, `methodology_version` | Required lineage identifiers; synthetic requests explicitly identify a fixture |
| `source_hashes`, `assumption_ids` | Hashes/assumptions supplied by the approved exporter; no arbitrary paths to follow |
| `contract`, `symbol`, `tick_size`, `point_value` | Explicit dated contract and economics checked against the caller's approved specification |
| `decision_time_utc`, `inputs_available_at_utc` | Aware timestamps; every input available at or before decision time |
| `availability_status` | `observed`, `assumed`, or `unknown`; conditional timing remains assumed |
| `plan` | Side, tick-aligned ideal entry, original stop/target and explicit scenario rule caps |
| `regime_parameters`, `calibration_id` | Versioned coefficients with hypothetical/empirical status; never silently derived from future RV |
| `delay_seconds`, `forced_slip_ticks` | Scenario assumptions, not claims about observed fills |
| `seed`, `trials`, `backend`, `include_samples` | Explicit replay settings; supported trial bound and exact draw-array hashes |
| `cost_policy` | Diagnostic entry displacement only; existing commissions/slippage/accounting are not changed or charged again |

Unknown schema/fields, missing lineage, mixed eligibility/release identities,
noncausal timestamps, malformed prices, unsupported backend or absent conditional
opt-in would reject the request. `unknown` availability must not pass as observed;
strict completeness alone does not imply proven historical availability. A
conditional experiment needs explicit caller opt-in and retains every assumption.

## Proposed response

`request_id`, request hash, unchanged eligibility/availability/assumptions,
`model_version`, native/Python source and build hashes, interpreter/compiler,
seed, draw hashes, sample count, RR-retention summary, scenario classifications,
optional samples, and explicit limitations. Never return an order, position size,
trade authorization, updated certification tier or production routing decision.

Classification means a hypothetical model-rule result, not permission to trade.
RR retention is not expectancy or statistically established alpha. Monte Carlo
frequencies are conditional on assumed inputs; no calibrated market probability
or continuous delay safety guarantee is asserted.

## Native prototype contract

Current standalone API: `batch.sample_scenarios(spec, inp, n, seed)` returns two
native float64 arrays. `batch.evaluate_samples(..., backend='python'|'cpp',
include_samples=False)` returns a shared MCStats summary and optional eight-column
sample rows; the final column is 0=Clean, 1=Compromised, 2=Invalid. The maximum batch
is 1,000,000 trials. `batch.run_batch` includes Python draw generation.

Python owns the random stream. Store array bytes, native endianness and SHA-256
when cross-runtime reproducibility matters: a seed alone does not guarantee the
same normal draws across Python versions. C++ snapshots contiguous float64 arrays
before releasing the GIL. No Python objects or borrowed mutable buffers are used
in its kernel. It validates input and either returns a complete result or fails;
it does not persist partial outputs. Summary mode stores an O(n) retention vector
for exact defined percentiles; detailed mode additionally allocates O(n) rows and
Python materialization. Concurrent calls use local state. No fast-math is enabled.

## Acceptance before any P2 implementation

1. Approve the request/response contract and exporter scope explicitly.
2. Use synthetic fixtures only for adapter tests, including attempts to submit
   unknown availability, missing conditional opt-in and forbidden outcome fields.
3. Prove byte-for-byte unchanged main execution/risk behavior and read-only input
   handling. Maintain separate output locations and immutable versioned reports.
4. Preserve existing cost ownership; evaluate any fill-model change as a later,
   separately approved experiment. Stop expansion is a scenario assumption only.
5. Profile the intended workload on its target machine. Single-trial requests may
   be slower through the batch interface; do not automatically reroute callers.
6. Treat HAR-to-subsecond parameter conversion, empirical calibration, options and
   live trading as separate proposals with separate evidence and authorization.

This proposal requests no data purchase, installation in the main project,
production authorization change, certification change, or trading enablement.
