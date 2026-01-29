# python/monte_carlo.py
# Monte Carlo utilities for Trade Degradation Simulator

from dataclasses import dataclass
from typing import List, Tuple
import random

from simulator import TradeSpec, SimInput, simulate_once  # core engine


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
    idx = max(0, min(idx, len(xs_sorted) - 1))
    return xs_sorted[idx]


def run_monte_carlo(spec: TradeSpec, inp: SimInput, n: int, seed: int = 42) -> Tuple[MCStats, List[float]]:
    """
    Returns:
      - summary stats
      - list of edge_remaining_pct samples (for histogram plotting in app.py)
    """
    rng = random.Random(seed)

    edges: List[float] = []
    degs: List[float] = []
    counts = {"Clean": 0, "Compromised": 0, "Invalid": 0}

    for _ in range(n):
        r = simulate_once(spec, inp, rng=rng)
        edges.append(r.edge_remaining_pct)
        degs.append(r.degradation_score)
        counts[r.classification] += 1

    stats = MCStats(
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
    return stats, edges
