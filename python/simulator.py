# python/simulator.py
# Trade Degradation Simulator (core engine)
# - Edge remaining (%), Degradation score, Entry quality classification
# - Optional: max tolerable delay/slippage boundaries
#
# Run examples (from repo root):
#   python3 simulator.py --side long --E 20000 --S 19995 --T 20020 --tick 0.25 --delay 0.2 --regime Normal
#   python3 simulator.py --side long --E 20000 --S 19995 --T 20020 --tick 0.25 --delay 2.5 --regime High --boundary
#   python3 simulator.py --mc 20000 --seed 7 --side long --E 20000 --S 19995 --T 20020 --tick 0.25 --delay 2.0 --regime High

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# C++ fast-path (pybind11)
try:
    import trade_degradation_cpp as cpp
except Exception:
    cpp = None

# -----------------------------
# Volatility regimes (single source of truth)
try:
    from regimes import REGIMES, RegimeParams  # type: ignore
except Exception:
    # Fallback if simulator.py is run standalone without regimes.py
    from dataclasses import dataclass
    from typing import Dict

    @dataclass(frozen=True)
    class RegimeParams:
        sigma_per_sqrt_s_ticks: float
        slip_mu_ticks: float
        slip_sigma_ticks: float
        stop_k_regime: float
        stop_k_late: float

    def _default_regimes() -> Dict[str, RegimeParams]:
        return {
            "Low": RegimeParams(0.6, 0.2, 0.6, 0.05, 0.05),
            "Normal": RegimeParams(1.0, 0.5, 1.0, 0.08, 0.08),
            "High": RegimeParams(1.6, 1.2, 1.8, 0.12, 0.12),
            "Event": RegimeParams(2.5, 2.0, 2.8, 0.18, 0.18),
        }

    REGIMES = _default_regimes()

# -----------------------------
# Helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def side_sign(side: str) -> int:
    s = side.lower().strip()
    if s in ("long", "buy"):
        return +1
    if s in ("short", "sell"):
        return -1
    raise ValueError("side must be 'long' or 'short'")

def price_to_ticks(price_diff: float, tick_size: float) -> float:
    return price_diff / tick_size

def ticks_to_price(ticks: float, tick_size: float) -> float:
    return ticks * tick_size


# -----------------------------
# Core types
# -----------------------------
@dataclass
class TradeSpec:
    side: str              # "long" or "short"
    ideal_entry: float     # E
    stop: float            # S (rule-based stop)
    target: float          # T (TP1 or TP2)
    tick_size: float

    # Rules / constraints
    max_entry_distance_ticks: float = 12.0   # max tolerated distance from ideal entry
    max_late_ticks_gate: float = 4.0         # hard gate: if lateness_ticks exceeds this, trade is Invalid
    min_rr: float = 1.0                      # minimum RR to consider valid
    allow_stop_expansion: bool = False       # if True, stop can expand when late/volatile
    max_stop_ticks: float = 60.0             # hard cap stop size (ticks) if expanding


@dataclass
class SimInput:
    delay_seconds: float
    regime: str = "Normal"
    forced_slip_ticks: Optional[float] = None  # if None, sample slip distribution


@dataclass
class SimResult:
    actual_entry: float
    used_stop: float
    edge_ideal_R: float
    edge_actual_R: float
    edge_remaining_pct: float
    degradation_score: float
    classification: str
    details: Dict[str, Union[float, str]]


# -----------------------------
# Edge & simulation logic
# -----------------------------
def compute_edge_R(side: str, entry: float, stop: float, target: float) -> float:
    """
    Reward/Risk in R-multiples.
    For long: risk = entry-stop, reward = target-entry
    For short: risk = stop-entry, reward = entry-target
    """
    sgn = side_sign(side)
    if sgn == +1:
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target

    if risk <= 0:
        return 0.0
    return reward / risk


def classify(edge_remaining_pct: float, rule_penalty: float) -> str:
    # If major rule violation, invalidate regardless of edge
    if rule_penalty >= 0.5:
        return "Invalid"
    if edge_remaining_pct >= 80:
        return "Clean"
    if edge_remaining_pct >= 50:
        return "Compromised"
    return "Invalid"


def simulate_once(spec: TradeSpec, inp: SimInput, rng: Optional[random.Random] = None) -> SimResult:
    rng = rng or random.Random()

    regime_key = inp.regime.strip()
    if regime_key not in REGIMES:
        raise ValueError(f"Unknown regime '{inp.regime}'. Choose from: {list(REGIMES.keys())}")
    rp = REGIMES[regime_key]

    sgn = side_sign(spec.side)
    tick = spec.tick_size

    # Ideal edge
    edge_ideal = compute_edge_R(spec.side, spec.ideal_entry, spec.stop, spec.target)

    # 1) Delay -> price moves (ticks), with slight adverse bias (optional)
    delay = max(0.0, float(inp.delay_seconds))
    sigma_delay_ticks = rp.sigma_per_sqrt_s_ticks * math.sqrt(max(delay, 1e-9))

    delta_delay_ticks = rng.gauss(0.0, sigma_delay_ticks)
    # Prevent absurd Gaussian outliers from dominating results (truncated normal)
    zcap = 6.0
    max_move = zcap * sigma_delay_ticks
    delta_delay_ticks = clamp(delta_delay_ticks, -max_move, max_move)

    adverse_bias_ticks = 0.35 * sigma_delay_ticks  # small bias: being late is usually worse on average
    delta_delay_ticks += sgn * adverse_bias_ticks

    entry_pre_slip = spec.ideal_entry + ticks_to_price(delta_delay_ticks, tick)

    # 2) Slippage (ticks)
    if inp.forced_slip_ticks is None:
        slip_ticks = max(0.0, round(rng.gauss(rp.slip_mu_ticks, rp.slip_sigma_ticks)))
    else:
        slip_ticks = max(0.0, float(inp.forced_slip_ticks))

    
    if cpp is not None:
        # NOTE: entry_pre_slip already includes delta_delay_ticks.
        # We only need to add slippage directionally on top of that, so keep python here OR
        # just recompute full actual entry from ideal using C++:
        actual_entry = cpp.actual_entry_from_ticks(
            spec.side,
            spec.ideal_entry,
            tick,
            float(delta_delay_ticks),
            float(slip_ticks),
        )
    else:
        actual_entry = entry_pre_slip + sgn * ticks_to_price(slip_ticks, tick)


    # Lateness in ticks from ideal
    lateness_ticks = abs(price_to_ticks(actual_entry - spec.ideal_entry, tick))

    # 3) Stop expansion (optional)
    used_stop = spec.stop
    if spec.allow_stop_expansion:
        base_stop_ticks = abs(price_to_ticks(spec.ideal_entry - spec.stop, tick))
        lateness_factor = clamp(lateness_ticks / max(spec.max_entry_distance_ticks, 1e-9), 0.0, 2.0)
        mult = 1.0 + rp.stop_k_regime + rp.stop_k_late * lateness_factor
        new_stop_ticks = clamp(base_stop_ticks * mult, base_stop_ticks, spec.max_stop_ticks)

        if sgn == +1:
            used_stop = spec.ideal_entry - ticks_to_price(new_stop_ticks, tick)
        else:
            used_stop = spec.ideal_entry + ticks_to_price(new_stop_ticks, tick)

    used_cpp = (cpp is not None)

    # --- Edge metrics (C++ fast-path if available) ---
    if cpp is not None:
        # Use C++ for RR + edge math (faster, same behavior)
        em = cpp.edge_metrics(
            spec.side,
            spec.ideal_entry,
            float(actual_entry),
            float(used_stop),
            float(spec.target),
        )
        edge_ideal = float(em.rr_ideal)
        edge_actual = float(em.rr_actual)
        edge_ratio = float(em.edge_ratio)
        edge_remaining_clamped = float(em.edge_remaining_clamped_pct)
        edge_remaining_unclamped = float(em.edge_remaining_unclamped_pct)
    else:
        # Python fallback (your existing behavior)
        edge_actual = compute_edge_R(spec.side, actual_entry, used_stop, spec.target)
        edge_ratio = (edge_actual / edge_ideal) if edge_ideal > 1e-9 else 0.0
        edge_remaining_clamped = clamp(edge_ratio, 0.0, 1.0) * 100.0
        edge_remaining_unclamped = clamp(edge_ratio, 0.0, 10.0) * 100.0  # cap for display



    # Rule penalties
    rule_penalty = 0.0

    # Rule 1: Entry too far
    if lateness_ticks > spec.max_entry_distance_ticks:
        rule_penalty = max(rule_penalty, 0.5)

    # Hard lateness gate (trader realism): if you're too late, it's invalid regardless of RR
    if lateness_ticks > spec.max_late_ticks_gate:
        rule_penalty = max(rule_penalty, 0.5)

    # Rule 2: Stop too big (if expanded)
    stop_ticks_used = abs(price_to_ticks(spec.ideal_entry - used_stop, tick))
    if stop_ticks_used > spec.max_stop_ticks + 1e-6:
        rule_penalty = max(rule_penalty, 0.5)

    # Rule 3: RR too low
    if edge_actual < spec.min_rr:
        rule_penalty = max(rule_penalty, 0.2)

    # Degradation score components
    edge_loss = clamp(1.0 - (edge_actual / edge_ideal if edge_ideal > 1e-9 else 1.0), 0.0, 1.0)

    # Risk increase proxy
    if sgn == +1:
        risk_ideal = max(spec.ideal_entry - spec.stop, 1e-9)
        risk_actual = max(actual_entry - used_stop, 1e-9)
    else:
        risk_ideal = max(spec.stop - spec.ideal_entry, 1e-9)
        risk_actual = max(used_stop - actual_entry, 1e-9)

    risk_increase = clamp((risk_actual / risk_ideal) - 1.0, 0.0, 1.0)

    degradation_score = 100.0 * clamp(
        0.7 * edge_loss + 0.2 * risk_increase + 0.1 * rule_penalty,
        0.0, 1.0
    )

    cls = classify(edge_remaining_clamped, rule_penalty)


    details: Dict[str, Union[float, str]] = {
        "regime": regime_key,
        "delay_seconds": delay,
        "delta_delay_ticks": float(delta_delay_ticks),
        "slip_ticks": float(slip_ticks),
        "lateness_ticks": float(lateness_ticks),
        "late_gate_ticks": float(spec.max_late_ticks_gate),
        "edge_ratio": float(edge_ratio),
        "edge_remaining_unclamped_pct": float(edge_remaining_unclamped),
        "cpp_path": bool(used_cpp),
        "rr_ideal": float(edge_ideal),
        "rr_actual": float(edge_actual),
        "rule_penalty": float(rule_penalty),
        "stop_ticks_used": float(stop_ticks_used),
    }

    return SimResult(
        actual_entry=actual_entry,
        used_stop=used_stop,
        edge_ideal_R=edge_ideal,
        edge_actual_R=edge_actual,
        edge_remaining_pct=edge_remaining_clamped,
        degradation_score=degradation_score,
        classification=cls,
        details=details,
    )

def prob_bad(
    spec: TradeSpec,
    inp: SimInput,
    edge_floor_pct: float,
    trials: int = 200,
    seed: int = 123
) -> float:
    rng = random.Random(seed)
    bad = 0
    for _ in range(trials):
        r = simulate_once(spec, inp, rng=rng)
        if r.classification == "Invalid" or r.edge_remaining_pct < edge_floor_pct:
            bad += 1
    return bad / trials


# Boundary solvers (max tolerable)
def _is_bad(spec: TradeSpec, inp: SimInput, edge_floor_pct: float, seed: int = 123) -> bool:
    """Deterministic eval for boundary search."""
    r = simulate_once(spec, inp, rng=random.Random(seed))
    if r.classification == "Invalid":
        return True
    return r.edge_remaining_pct < edge_floor_pct


def max_slippage_allowed(
    spec: TradeSpec,
    delay_seconds: float,
    regime: str,
    edge_floor_pct: float = 50.0,
    max_search_ticks: int = 120,
    seed: int = 123
) -> int:
    """Largest slip_ticks allowed at fixed delay."""
    lo, hi = 0, max_search_ticks

    if _is_bad(spec, SimInput(delay_seconds=delay_seconds, regime=regime, forced_slip_ticks=0.0), edge_floor_pct, seed):
        return 0

    if not _is_bad(spec, SimInput(delay_seconds=delay_seconds, regime=regime, forced_slip_ticks=float(hi)), edge_floor_pct, seed):
        return hi

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _is_bad(spec, SimInput(delay_seconds=delay_seconds, regime=regime, forced_slip_ticks=float(mid)), edge_floor_pct, seed):
            hi = mid
        else:
            lo = mid
    return lo


def max_delay_allowed_prob(
    spec: TradeSpec,
    forced_slip_ticks: float,
    regime: str,
    edge_floor_pct: float = 50.0,
    p_bad: float = 0.30,
    trials: int = 300,
    max_search_seconds: float = 60.0,
    seed: int = 123
) -> float:
    lo, hi = 0.0, 1.0



    # Expand hi until probability of bad exceeds threshold
    while hi < max_search_seconds:
        p = prob_bad(
            spec,
            SimInput(delay_seconds=hi, regime=regime, forced_slip_ticks=forced_slip_ticks),
            edge_floor_pct=edge_floor_pct,
            trials=trials,
            seed=seed,
        )
        if p >= p_bad:
            break
        hi *= 2.0

    hi = min(hi, max_search_seconds)

    # If even hi is still acceptable, return cap
    p_hi = prob_bad(
        spec,
        SimInput(delay_seconds=hi, regime=regime, forced_slip_ticks=forced_slip_ticks),
        edge_floor_pct=edge_floor_pct,
        trials=trials,
        seed=seed,
    )
    if p_hi < p_bad:
        return hi

    # Binary search
    for _ in range(25):
        mid = (lo + hi) / 2.0
        p_mid = prob_bad(
            spec,
            SimInput(delay_seconds=mid, regime=regime, forced_slip_ticks=forced_slip_ticks),
            edge_floor_pct=edge_floor_pct,
            trials=trials,
            seed=seed,
        )
        if p_mid >= p_bad:
            hi = mid
        else:
            lo = mid

    return lo

def max_slippage_allowed_prob(
    spec: TradeSpec,
    delay_seconds: float,
    regime: str,
    edge_floor_pct: float = 50.0,
    p_bad: float = 0.30,
    trials: int = 300,
    max_search_ticks: int = 60,
    seed: int = 123
) -> int:
    lo, hi = 0, 1

    # Expand hi until bad
    while hi < max_search_ticks:
        p = prob_bad(
            spec,
            SimInput(delay_seconds=delay_seconds, regime=regime, forced_slip_ticks=float(hi)),
            edge_floor_pct=edge_floor_pct,
            trials=trials,
            seed=seed,
        )
        if p >= p_bad:
            break
        hi *= 2
    hi = min(hi, max_search_ticks)

    # If even hi is still acceptable, return cap
    p_hi = prob_bad(
        spec,
        SimInput(delay_seconds=delay_seconds, regime=regime, forced_slip_ticks=float(hi)),
        edge_floor_pct=edge_floor_pct,
        trials=trials,
        seed=seed,
    )
    if p_hi < p_bad:
        return hi

    # Binary search integer ticks
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        p_mid = prob_bad(
            spec,
            SimInput(delay_seconds=delay_seconds, regime=regime, forced_slip_ticks=float(mid)),
            edge_floor_pct=edge_floor_pct,
            trials=trials,
            seed=seed,
        )
        if p_mid >= p_bad:
            hi = mid
        else:
            lo = mid

    return lo

# Monte Carlo (local fallback)
# (Later you'll move this to python/monte_carlo.py)
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


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    idx = int(round((p / 100.0) * (len(xs_sorted) - 1)))
    idx = int(clamp(idx, 0, len(xs_sorted) - 1))
    return xs_sorted[idx]


def run_monte_carlo(spec: TradeSpec, inp: SimInput, n: int, seed: int = 42) -> MCStats:
    rng = random.Random(seed)
    edges: List[float] = []
    degs: List[float] = []
    counts = {"Clean": 0, "Compromised": 0, "Invalid": 0}

    for _ in range(n):
        r = simulate_once(spec, inp, rng=rng)
        edges.append(r.edge_remaining_pct)
        degs.append(r.degradation_score)
        counts[r.classification] += 1

    return MCStats(
        n=n,
        p_clean=counts["Clean"] / n,
        p_compromised=counts["Compromised"] / n,
        p_invalid=counts["Invalid"] / n,
        edge_remaining_mean=sum(edges) / n,
        edge_remaining_p10=_percentile(edges, 10),
        edge_remaining_p50=_percentile(edges, 50),
        edge_remaining_p90=_percentile(edges, 90),
        degradation_mean=sum(degs) / n,
    )


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Trade Degradation Simulator (python/simulator.py)")

    ap.add_argument("--side", required=True, choices=["long", "short"])
    ap.add_argument("--E", type=float, required=True, help="Ideal entry price")
    ap.add_argument("--S", type=float, required=True, help="Stop price")
    ap.add_argument("--T", type=float, required=True, help="Target price")
    ap.add_argument("--tick", type=float, required=True, help="Tick size (e.g., NQ=0.25, GC=0.10)")

    ap.add_argument("--delay", type=float, default=0.0, help="Entry delay in seconds")
    ap.add_argument("--slip", type=float, default=None, help="Forced slippage ticks (optional)")
    ap.add_argument("--regime", type=str, default="Normal", choices=list(REGIMES.keys()))

    ap.add_argument("--max_entry_ticks", type=float, default=12.0)
    ap.add_argument("--max_late_gate", type=float, default=4.0, help="Hard gate: if lateness (ticks) exceeds this, classify Invalid")
    ap.add_argument("--min_rr", type=float, default=1.0)
    ap.add_argument("--allow_stop_expansion", action="store_true")
    ap.add_argument("--max_stop_ticks", type=float, default=60.0)

    ap.add_argument("--mc", type=int, default=0, help="Monte Carlo runs (0 = single run)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--edge_floor", type=float, default=70.0, help="Min edge remaining % (for boundaries)")
    ap.add_argument("--boundary", action="store_true", help="Compute max tolerable delay/slippage")
    ap.add_argument("--p_bad", type=float, default=0.40, help="Boundary: probability threshold to call it bad")
    ap.add_argument("--trials", type=int, default=300, help="Boundary: Monte Carlo trials per step")
    ap.add_argument("--max_delay_cap", type=float, default=60.0, help="Boundary: max seconds to search")
    ap.add_argument("--max_slip_cap", type=int, default=60, help="Boundary: max ticks to search")


    args = ap.parse_args()

    spec = TradeSpec(
        side=args.side,
        ideal_entry=args.E,
        stop=args.S,
        target=args.T,
        tick_size=args.tick,
        max_entry_distance_ticks=args.max_entry_ticks,
        max_late_ticks_gate=args.max_late_gate,
        min_rr=args.min_rr,
        allow_stop_expansion=bool(args.allow_stop_expansion),
        max_stop_ticks=args.max_stop_ticks,
    )

    inp = SimInput(
        delay_seconds=args.delay,
        regime=args.regime,
        forced_slip_ticks=args.slip,
    )

    if args.mc and args.mc > 0:
        stats = run_monte_carlo(spec, inp, n=args.mc, seed=args.seed)
        print("\n=== Monte Carlo Summary ===")
        print(f"Runs: {stats.n}")
        print(f"P(Clean):        {stats.p_clean:.3f}")
        print(f"P(Compromised):  {stats.p_compromised:.3f}")
        print(f"P(Invalid):      {stats.p_invalid:.3f}")
        print(f"Edge remaining mean: {stats.edge_remaining_mean:.2f}%")
        print(f"Edge remaining p10/p50/p90: {stats.edge_remaining_p10:.2f}% / {stats.edge_remaining_p50:.2f}% / {stats.edge_remaining_p90:.2f}%")
        print(f"Degradation mean: {stats.degradation_mean:.2f}/100\n")
    else:
        r = simulate_once(spec, inp, rng=random.Random(args.seed))
        print("\n=== Single Simulation ===")
        print(f"Ideal edge (RR):   {r.edge_ideal_R:.3f}R")
        print(f"Actual edge (RR):  {r.edge_actual_R:.3f}R")
        print(f"Edge remaining:    {r.edge_remaining_pct:.2f}%")
        print(f"Degradation score: {r.degradation_score:.2f}/100")
        print(f"Classification:    {r.classification}")
        print("\n--- Details ---")
        for k, v in r.details.items():
            print(f"{k}: {v}")
        print(f"actual_entry: {r.actual_entry}")
        print(f"used_stop:    {r.used_stop}\n")

    if args.boundary:
        forced_slip_for_delay = 0.0 if args.slip is None else float(args.slip)

        max_slip = max_slippage_allowed_prob(
            spec,
            delay_seconds=float(args.delay),
            regime=args.regime,
            edge_floor_pct=float(args.edge_floor),
            p_bad=float(args.p_bad),
            trials=int(args.trials),
            max_search_ticks=int(args.max_slip_cap),
            seed=args.seed,
        )

        max_delay = max_delay_allowed_prob(
            spec,
            forced_slip_ticks=forced_slip_for_delay,
            regime=args.regime,
            edge_floor_pct=float(args.edge_floor),
            p_bad=float(args.p_bad),
            trials=int(args.trials),
            max_search_seconds=float(args.max_delay_cap),
            seed=args.seed,
        )

        # Probability at the boundary points (transparency)
        p_at_max_delay = prob_bad(
            spec,
            SimInput(delay_seconds=max_delay, regime=args.regime, forced_slip_ticks=forced_slip_for_delay),
            edge_floor_pct=args.edge_floor,
            trials=args.trials,
            seed=args.seed,
        )

        p_at_max_slip = prob_bad(
            spec,
            SimInput(delay_seconds=args.delay, regime=args.regime, forced_slip_ticks=float(max_slip)),
            edge_floor_pct=args.edge_floor,
            trials=args.trials,
            seed=args.seed,
        )

        print("=== Tolerances (probabilistic) ===")
        print(
            f"Edge floor: {args.edge_floor:.1f}% | "
            f"p_bad: {args.p_bad:.2f} | "
            f"trials: {args.trials} | "
            f"late_gate: {args.max_late_gate} ticks"
        )

        # Context from the *single sim* result `r`
        # (This assumes you ran `r = simulate_once(...)` earlier in main, which you did.)
        print(
            f"Current lateness: {r.details['lateness_ticks']:.2f} ticks "
            f"(gate = {r.details['late_gate_ticks']:.1f} ticks)"
        )

        print(f"Max slippage allowed (ticks) @ delay={args.delay}s: {max_slip}")
        print(f"P(bad) @ max_slip:  {p_at_max_slip:.3f}")

        print(f"Max delay allowed (s) @ slip={forced_slip_for_delay} ticks: {max_delay:.3f}")
        print(f"P(bad) @ max_delay: {p_at_max_delay:.3f}")
        print()

if __name__ == "__main__":
    main()
