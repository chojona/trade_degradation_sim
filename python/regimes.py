# python/regimes.py
# Volatility regime parameters for Trade Degradation Simulator

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RegimeParams:
    # Delay impact: price change during delay ~ Normal(0, sigma_per_sqrt_s * sqrt(delay_seconds)) in ticks
    sigma_per_sqrt_s_ticks: float

    # Slippage: slip_ticks ~ max(0, round(Normal(mu, sigma)))
    slip_mu_ticks: float
    slip_sigma_ticks: float

    # Stop expansion multipliers (only used if allow_stop_expansion=True)
    stop_k_regime: float
    stop_k_late: float


# You can tune these later per instrument (NQ vs GC) or per session (NY vs Asia)
REGIMES: Dict[str, RegimeParams] = {
    "Low": RegimeParams(
        sigma_per_sqrt_s_ticks=0.6,
        slip_mu_ticks=0.2,
        slip_sigma_ticks=0.6,
        stop_k_regime=0.05,
        stop_k_late=0.05,
    ),
    "Normal": RegimeParams(
        sigma_per_sqrt_s_ticks=1.0,
        slip_mu_ticks=0.5,
        slip_sigma_ticks=1.0,
        stop_k_regime=0.08,
        stop_k_late=0.08,
    ),
    "High": RegimeParams(
        sigma_per_sqrt_s_ticks=1.6,
        slip_mu_ticks=1.2,
        slip_sigma_ticks=1.8,
        stop_k_regime=0.12,
        stop_k_late=0.12,
    ),
    "Event": RegimeParams(
        sigma_per_sqrt_s_ticks=2.5,
        slip_mu_ticks=2.0,
        slip_sigma_ticks=2.8,
        stop_k_regime=0.18,
        stop_k_late=0.18,
    ),
}
