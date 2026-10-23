"""Hypothetical RR-retention scenarios, not expected profitability or a fill model.

Legacy ``edge_*`` result names are retained for callers; they mean reward/risk.
Boundary functions return None when the zero point fails. Delay boundaries describe
only a sampled zero-connected grid prefix, never a certified continuous interval.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Optional

try:
    import trade_degradation_cpp as cpp
except ImportError:
    cpp = None

from regimes import REGIMES, RegimeParams


def finite(name, value, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (minimum is not None and value < minimum) or (positive and value <= 0):
        raise ValueError(f"{name} outside allowed range")
    return float(value)


def count(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def clamp(x, lo, hi):
    if not all(math.isfinite(v) for v in (x, lo, hi)) or lo > hi:
        raise ValueError("clamp requires finite values and ordered bounds")
    return max(lo, min(hi, x))


def side_sign(side):
    if not isinstance(side, str):
        raise ValueError("side must be long or short")
    value = side.strip().lower()
    if value in ("long", "buy"):
        return 1
    if value in ("short", "sell"):
        return -1
    raise ValueError("side must be long or short")


def price_to_ticks(price_diff, tick_size):
    finite("price difference", price_diff)
    finite("tick_size", tick_size, positive=True)
    return finite("tick quotient", price_diff / tick_size)


def ticks_to_price(ticks, tick_size):
    finite("ticks", ticks)
    finite("tick_size", tick_size, positive=True)
    return finite("price delta", ticks * tick_size)


def grid_units(price, tick):
    units = price_to_ticks(price, tick)
    if abs(units) >= 2 ** 50:
        raise ValueError("price/tick outside supported precision range")
    return units


def round_grid(price, tick, direction):
    units = grid_units(price, tick)
    nearest = round(units)
    if abs(units - nearest) <= 1e-8:
        return nearest * tick
    return (math.ceil(units) if direction > 0 else math.floor(units)) * tick


def on_grid(price, tick):
    units = grid_units(price, tick)
    return abs(units - round(units)) <= 1e-8


@dataclass(frozen=True)
class TradeSpec:
    side: str
    ideal_entry: float
    stop: float
    target: float
    tick_size: float
    max_entry_distance_ticks: float = 12.0
    max_late_ticks_gate: float = 4.0
    min_rr: float = 1.0
    allow_stop_expansion: bool = False
    max_stop_ticks: float = 60.0

    def __post_init__(self):
        sgn = side_sign(self.side)
        object.__setattr__(self, "side", "long" if sgn == 1 else "short")
        finite("tick_size", self.tick_size, positive=True)
        for name in ("ideal_entry", "stop", "target"):
            finite(name, getattr(self, name))
            if not on_grid(getattr(self, name), self.tick_size):
                raise ValueError(f"{name} must lie on the tick grid")
        if (self.ideal_entry - self.stop) * sgn <= 0 or (self.target - self.ideal_entry) * sgn <= 0:
            raise ValueError("ideal stop and target must bracket the entry")
        for name in ("max_entry_distance_ticks", "max_late_ticks_gate", "min_rr"):
            finite(name, getattr(self, name), minimum=0)
        finite("max_stop_ticks", self.max_stop_ticks, positive=True)
        if not isinstance(self.allow_stop_expansion, bool):
            raise ValueError("allow_stop_expansion must be bool")


@dataclass(frozen=True)
class SimInput:
    delay_seconds: float
    regime: str = "Normal"
    forced_slip_ticks: Optional[float] = None

    def __post_init__(self):
        finite("delay_seconds", self.delay_seconds, minimum=0)
        if not isinstance(self.regime, str) or self.regime.strip() not in REGIMES:
            raise ValueError(f"Unknown regime {self.regime!r}")
        object.__setattr__(self, "regime", self.regime.strip())
        if self.forced_slip_ticks is not None:
            finite("forced_slip_ticks", self.forced_slip_ticks, minimum=0)


@dataclass
class SimResult:
    actual_entry: float
    used_stop: float
    edge_ideal_R: float
    edge_actual_R: float
    edge_remaining_pct: float
    degradation_score: float
    classification: str
    details: dict

    @property
    def rr_retention_pct(self):
        return self.edge_remaining_pct


def compute_edge_R(side, entry, stop, target):
    for name, value in (("entry", entry), ("stop", stop), ("target", target)):
        finite(name, value)
    sgn = side_sign(side)
    risk = finite("risk", (entry - stop) * sgn)
    reward = finite("reward", (target - entry) * sgn)
    return finite("RR", reward / risk) if risk > 0 else 0.0


def classify(edge_remaining_pct, rule_penalty):
    finite("RR retention", edge_remaining_pct, minimum=0)
    finite("rule penalty", rule_penalty, minimum=0)
    if rule_penalty >= .5:
        return "Invalid"
    return "Clean" if edge_remaining_pct >= 80 else "Compromised" if edge_remaining_pct >= 50 else "Invalid"


def resolve_backend(backend):
    if backend not in ("auto", "python", "cpp"):
        raise ValueError("backend must be auto, python or cpp")
    if backend == "cpp" and cpp is None:
        raise RuntimeError("C++ extension unavailable")
    return cpp is not None and backend != "python"


def evaluate_draw(spec, inp, delta_delay_ticks, slip_ticks, *, backend="auto"):
    """Evaluate explicit draws. Both backends use the same original-stop baseline."""
    finite("delay draw", delta_delay_ticks)
    finite("slippage draw", slip_ticks, minimum=0)
    rp = REGIMES[inp.regime]
    for name in rp.__dataclass_fields__:
        finite(name, getattr(rp, name), minimum=0)
    native = resolve_backend(backend)
    sgn, tick = side_sign(spec.side), spec.tick_size
    if native:
        actual = cpp.actual_entry_from_ticks(spec.side, spec.ideal_entry, tick, delta_delay_ticks, slip_ticks)
    else:
        actual = round_grid(spec.ideal_entry + ticks_to_price(delta_delay_ticks + sgn * slip_ticks, tick), tick, sgn)
    lateness = abs(price_to_ticks(actual - spec.ideal_entry, tick))
    stop = spec.stop
    if spec.allow_stop_expansion:
        base = abs(price_to_ticks(spec.ideal_entry - spec.stop, tick))
        factor = clamp(lateness / max(spec.max_entry_distance_ticks, 1e-9), 0, 2)
        desired = finite("expanded stop", base * (1 + rp.stop_k_regime + rp.stop_k_late * factor))
        # Never shrink an existing stop to hide a cap violation.
        distance = max(base, min(math.ceil(desired - 1e-8), math.floor(spec.max_stop_ticks)))
        stop = round_grid(spec.ideal_entry - sgn * distance * tick, tick, -sgn)
    ideal = compute_edge_R(spec.side, spec.ideal_entry, spec.stop, spec.target)
    if native:
        em = cpp.edge_metrics_with_stops(spec.side, spec.ideal_entry, actual, spec.stop, stop, spec.target)
        ideal, actual_rr, ratio = em.rr_ideal, em.rr_actual, em.edge_ratio
    else:
        actual_rr = compute_edge_R(spec.side, actual, stop, spec.target)
        ratio = finite("RR ratio", actual_rr / ideal)
    retained = clamp(ratio, 0, 1) * 100
    stop_ticks = (actual - stop) * sgn / tick
    invalid = (lateness > spec.max_entry_distance_ticks + 1e-8 or
               lateness > spec.max_late_ticks_gate + 1e-8 or
               stop_ticks > spec.max_stop_ticks + 1e-8 or stop_ticks <= 0 or
               actual_rr < spec.min_rr or (spec.target - actual) * sgn <= 0)
    penalty = .5 if invalid else 0.0
    ideal_risk = (spec.ideal_entry - spec.stop) * sgn
    risk_increase = clamp(((actual - stop) * sgn / ideal_risk) - 1, 0, 1)
    score = 100 * clamp(.7 * clamp(1 - ratio, 0, 1) + .2 * risk_increase + .1 * penalty, 0, 1)
    return SimResult(actual, stop, ideal, actual_rr, retained, score, classify(retained, penalty), {
        "regime": inp.regime, "delay_seconds": inp.delay_seconds,
        "delta_delay_ticks": delta_delay_ticks, "slip_ticks": slip_ticks,
        "lateness_ticks": lateness, "late_gate_ticks": spec.max_late_ticks_gate,
        "edge_ratio": ratio, "edge_remaining_unclamped_pct": clamp(ratio, 0, 10) * 100,
        "cpp_path": native, "rr_ideal": ideal, "rr_actual": actual_rr,
        "rule_penalty": penalty, "stop_ticks_used": stop_ticks,
        "metric": "reward_risk_retention_not_expected_profitability",
    })


def sample_draw(inp, rng):
    rp = REGIMES[inp.regime]
    for name in rp.__dataclass_fields__:
        finite(name, getattr(rp, name), minimum=0)
    # The sign-dependent bias is added in simulate_once.
    sigma = finite("delay sigma", rp.sigma_per_sqrt_s_ticks * math.sqrt(inp.delay_seconds))
    delta = clamp(rng.gauss(0, sigma), -6 * sigma, 6 * sigma) if inp.delay_seconds else 0.0
    slip = max(0, round(rng.gauss(rp.slip_mu_ticks, rp.slip_sigma_ticks))) if inp.forced_slip_ticks is None else inp.forced_slip_ticks
    return delta, slip, sigma


def simulate_once(spec, inp, rng=None, *, backend="auto"):
    rng = rng if rng is not None else random.Random()
    delta, slip, sigma = sample_draw(inp, rng)
    return evaluate_draw(spec, inp, delta + side_sign(spec.side) * .35 * sigma, slip, backend=backend)


def validate_probability(edge_floor_pct, trials, p_bad=None):
    finite("RR floor", edge_floor_pct, minimum=0)
    if edge_floor_pct > 100:
        raise ValueError("RR floor must be <= 100")
    count("trials", trials)
    if p_bad is not None:
        finite("p_bad", p_bad, positive=True)
        if p_bad > 1:
            raise ValueError("p_bad must be <= 1")


def prob_bad(spec, inp, edge_floor_pct, trials=200, seed=123):
    """Monte Carlo point estimate, not a confidence bound or market probability."""
    validate_probability(edge_floor_pct, trials)
    rng = random.Random(seed)
    return sum((r.classification == "Invalid" or r.edge_remaining_pct < edge_floor_pct)
               for r in (simulate_once(spec, inp, rng) for _ in range(trials))) / trials


def _is_bad(spec, inp, edge_floor_pct, seed=123):
    validate_probability(edge_floor_pct, 1)
    r = simulate_once(spec, inp, random.Random(seed))
    return r.classification == "Invalid" or r.edge_remaining_pct < edge_floor_pct


def _prefix_boundary(points, acceptable):
    last = None
    for point in points:
        if not acceptable(point):
            break
        last = point
    return last


def max_slippage_allowed(spec, delay_seconds, regime, edge_floor_pct=50., max_search_ticks=120, seed=123):
    count("max_search_ticks", max_search_ticks, 0)
    return _prefix_boundary(range(max_search_ticks + 1), lambda slip: not _is_bad(
        spec, SimInput(delay_seconds, regime, slip), edge_floor_pct, seed))


def max_slippage_allowed_prob(spec, delay_seconds, regime, edge_floor_pct=50., p_bad=.30,
                              trials=300, max_search_ticks=60, seed=123):
    validate_probability(edge_floor_pct, trials, p_bad)
    count("max_search_ticks", max_search_ticks, 0)
    return _prefix_boundary(range(max_search_ticks + 1), lambda slip: prob_bad(
        spec, SimInput(delay_seconds, regime, slip), edge_floor_pct, trials, seed) < p_bad)


def max_delay_allowed_prob(spec, forced_slip_ticks, regime, edge_floor_pct=50., p_bad=.30,
                           trials=300, max_search_seconds=60., seed=123, *, step_seconds=.1):
    """Last passing zero-connected sampled point (spacing <= step_seconds).

    Stops at the first failing point. No monotonicity or between-grid safety is
    assumed. None is infeasible; 0.0 means the baseline alone passed.
    """
    validate_probability(edge_floor_pct, trials, p_bad)
    finite("max_search_seconds", max_search_seconds, minimum=0)
    finite("step_seconds", step_seconds, positive=True)
    steps = math.ceil(max_search_seconds / step_seconds)
    if steps > 100000:
        raise ValueError("delay grid exceeds 100000 steps")
    points = (min(i * step_seconds, max_search_seconds) for i in range(steps + 1))
    return _prefix_boundary(points, lambda delay: prob_bad(
        spec, SimInput(delay, regime, forced_slip_ticks), edge_floor_pct, trials, seed) < p_bad)


@dataclass
class MCStats:
    n: int
    p_clean: float
    p_compromised: float
    p_invalid: float
    edge_remaining_mean: float
    edge_remaining_p10: float
    edge_remaining_p50: float
    edge_remaining_p90: float
    degradation_mean: float


def _percentile(xs, p):
    if not xs:
        raise ValueError("percentile requires samples")
    finite("percentile", p, minimum=0)
    if p > 100:
        raise ValueError("percentile must be <= 100")
    return sorted(xs)[round(p / 100 * (len(xs) - 1))]


def monte_carlo_samples(spec, inp, n, seed=42, *, backend="auto"):
    count("n", n)
    rng = random.Random(seed)
    results = [simulate_once(spec, inp, rng, backend=backend) for _ in range(n)]
    edges = [r.edge_remaining_pct for r in results]
    stats = MCStats(n, *(sum(r.classification == c for r in results) / n for c in ("Clean", "Compromised", "Invalid")),
                    math.fsum(edges) / n, *(_percentile(edges, p) for p in (10, 50, 90)),
                    math.fsum(r.degradation_score for r in results) / n)
    return stats, edges


def run_monte_carlo(spec, inp, n, seed=42, *, backend="auto"):
    return monte_carlo_samples(spec, inp, n, seed, backend=backend)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", required=True, choices=["long", "short"])
    for flag in ("E", "S", "T", "tick"):
        ap.add_argument("--" + flag, type=float, required=True)
    for flag, default in (("delay", 0.), ("slip", None), ("max_entry_ticks", 12.), ("max_late_gate", 4.),
                          ("min_rr", 1.), ("max_stop_ticks", 60.), ("edge_floor", 70.), ("p_bad", .4), ("max_delay_cap", 60.)):
        ap.add_argument("--" + flag, type=float, default=default)
    for flag, default in (("mc", 0), ("seed", 42), ("trials", 300), ("max_slip_cap", 60)):
        ap.add_argument("--" + flag, type=int, default=default)
    ap.add_argument("--regime", default="Normal", choices=list(REGIMES))
    ap.add_argument("--allow_stop_expansion", action="store_true")
    ap.add_argument("--boundary", action="store_true")
    ap.add_argument("--backend", choices=["auto", "python", "cpp"], default="auto")
    args = ap.parse_args()
    spec = TradeSpec(args.side, args.E, args.S, args.T, args.tick, args.max_entry_ticks,
                     args.max_late_gate, args.min_rr, args.allow_stop_expansion, args.max_stop_ticks)
    inp = SimInput(args.delay, args.regime, args.slip)
    count("mc", args.mc, 0)
    output = {"metric": "reward_risk_retention_not_expected_profitability", "seed": args.seed,
              "result": asdict(run_monte_carlo(spec, inp, args.mc, args.seed, backend=args.backend)
                               if args.mc else simulate_once(spec, inp, random.Random(args.seed), backend=args.backend))}
    if args.boundary:
        output["boundary"] = {
            "max_slip_ticks": max_slippage_allowed_prob(spec, args.delay, args.regime, args.edge_floor, args.p_bad, args.trials, args.max_slip_cap, args.seed),
            "sampled_delay_prefix_seconds": max_delay_allowed_prob(spec, args.slip or 0., args.regime, args.edge_floor, args.p_bad, args.trials, args.max_delay_cap, args.seed),
            "delay_grid_seconds": .1, "infeasible_is_null": True}
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
