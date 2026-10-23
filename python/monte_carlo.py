"""Compatibility facade; sampling and summary definitions have one owner."""
from simulator import MCStats, _percentile, monte_carlo_samples


def run_monte_carlo(spec, inp, n, seed=42):
    return monte_carlo_samples(spec, inp, n, seed)
