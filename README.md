# Trade Degradation Simulator

A quantitative tool that measures how much trading edge degrades due to execution delay, slippage, and volatility.

Instead of asking *“does this strategy work?”*, the simulator answers:

> **“How much of my edge is gone by the time I enter?”**

---

## What it does

Given an ideal trade plan (entry, stop, target), the simulator models:

- Entry delay (seconds late)
- Slippage (ticks)
- Volatility regimes (Low / Normal / High / Event)
- Optional stop expansion
- A hard lateness gate (too late = invalid)

It outputs:
- Ideal RR vs Actual RR
- Edge remaining (%)
- Degradation score (0–100)
- Trade classification: **Clean / Compromised / Invalid**
- Probabilistic execution limits (via Monte Carlo)

A C++ core (pybind11) is used optionally for performance-critical math.

---
