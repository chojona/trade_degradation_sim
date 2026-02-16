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

### Run

From the `python/` directory:

```bash
cd python
python3 simulator.py --side long --E 20000 --S 19995 --T 20020 --tick 0.25 --delay 0.2 --regime Normal
python3 simulator.py --mc 200 --side long --E 20000 --S 19995 --T 20020 --tick 0.25 --delay 2.0 --regime High
python3 simulator.py --boundary --side long --E 20000 --S 19995 --T 20020 --tick 0.25 --delay 2.5 --regime High
```

Optional C++ extension (faster): from repo root, `cd cpp && cmake -B build -S . && cmake --build build`. The built module is placed in `python/`.

### Tests

From repo root: `python3 python/test_simulator.py` (uses `unittest`). All 14 tests must pass; 3 are skipped if the C++ extension is not built.

---
