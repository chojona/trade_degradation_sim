# Tests for Trade Degradation Simulator
# Run from repo root: python python/test_simulator.py
# Or: python -m pytest python/test_simulator.py -v  (if pytest installed)

from __future__ import annotations

import sys
import os
import unittest

# Allow importing simulator and regimes when run from repo root
if __name__ == "__main__" or "__file__" in dir():
    _root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    _pydir = os.path.dirname(os.path.abspath(__file__))
    if _pydir not in sys.path:
        sys.path.insert(0, _pydir)

import random
from simulator import (
    clamp,
    side_sign,
    price_to_ticks,
    ticks_to_price,
    compute_edge_R,
    classify,
    TradeSpec,
    SimInput,
    SimResult,
    simulate_once,
    REGIMES,
)


def _approx(a: float, b: float, rel: float = 1e-9) -> bool:
    return abs(a - b) <= rel or (abs(b) > 1e-12 and abs((a - b) / b) <= rel)


# ---- Helpers ----
class TestHelpers(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp(0.5, 0.0, 1.0), 0.5)
        self.assertEqual(clamp(-1.0, 0.0, 1.0), 0.0)
        self.assertEqual(clamp(2.0, 0.0, 1.0), 1.0)

    def test_side_sign(self):
        self.assertEqual(side_sign("long"), 1)
        self.assertEqual(side_sign("Long"), 1)
        self.assertEqual(side_sign("short"), -1)
        self.assertEqual(side_sign("Short"), -1)
        with self.assertRaises(ValueError):
            side_sign("invalid")

    def test_price_ticks_roundtrip(self):
        tick = 0.25
        ticks = 4.0
        self.assertTrue(_approx(price_to_ticks(ticks_to_price(ticks, tick), tick), ticks))
        self.assertTrue(_approx(ticks_to_price(price_to_ticks(1.0, tick), tick), 1.0))


# ---- Edge (RR) ----
class TestEdgeR(unittest.TestCase):
    def test_long(self):
        self.assertAlmostEqual(compute_edge_R("long", 20000.0, 19995.0, 20020.0), 4.0)
        self.assertEqual(compute_edge_R("long", 20000.0, 20000.0, 20020.0), 0.0)

    def test_short(self):
        self.assertAlmostEqual(compute_edge_R("short", 20000.0, 20005.0, 19980.0), 4.0)


# ---- Classification ----
class TestClassify(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(classify(85.0, 0.0), "Clean")
        self.assertEqual(classify(80.0, 0.0), "Clean")
        self.assertEqual(classify(50.0, 0.0), "Compromised")
        self.assertEqual(classify(60.0, 0.0), "Compromised")
        self.assertEqual(classify(40.0, 0.0), "Invalid")
        self.assertEqual(classify(90.0, 0.5), "Invalid")
        self.assertEqual(classify(90.0, 0.2), "Clean")


# ---- Single simulation ----
def _make_spec(side: str = "long") -> TradeSpec:
    return TradeSpec(
        side=side,
        ideal_entry=20000.0,
        stop=19995.0,
        target=20020.0,
        tick_size=0.25,
        max_entry_distance_ticks=12.0,
        max_late_ticks_gate=4.0,
        min_rr=1.0,
        allow_stop_expansion=False,
        max_stop_ticks=60.0,
    )


class TestSimulateOnce(unittest.TestCase):
    def test_deterministic(self):
        spec = _make_spec()
        inp = SimInput(delay_seconds=0.5, regime="Normal", forced_slip_ticks=1.0)
        rng = random.Random(42)
        r1 = simulate_once(spec, inp, rng=rng)
        r2 = simulate_once(spec, inp, rng=random.Random(42))
        self.assertEqual(r1.actual_entry, r2.actual_entry)
        self.assertEqual(r1.edge_remaining_pct, r2.edge_remaining_pct)
        self.assertEqual(r1.classification, r2.classification)

    def test_structure(self):
        spec = _make_spec()
        inp = SimInput(delay_seconds=0.0, regime="Normal", forced_slip_ticks=0.0)
        r = simulate_once(spec, inp, rng=random.Random(123))
        self.assertIsInstance(r, SimResult)
        self.assertGreaterEqual(r.edge_ideal_R, 0)
        self.assertGreaterEqual(r.edge_remaining_pct, 0)
        self.assertLessEqual(r.edge_remaining_pct, 100)
        self.assertGreaterEqual(r.degradation_score, 0)
        self.assertLessEqual(r.degradation_score, 100)
        self.assertIn(r.classification, ("Clean", "Compromised", "Invalid"))
        self.assertIn("lateness_ticks", r.details)
        self.assertIn("regime", r.details)

    def test_short(self):
        spec_short = TradeSpec(
            side="short",
            ideal_entry=20000.0,
            stop=20005.0,
            target=19980.0,
            tick_size=0.25,
        )
        inp = SimInput(delay_seconds=0.0, regime="Normal", forced_slip_ticks=0.0)
        r = simulate_once(spec_short, inp, rng=random.Random(999))
        self.assertAlmostEqual(r.edge_ideal_R, 4.0)
        self.assertIn(r.classification, ("Clean", "Compromised", "Invalid"))


# ---- C++ vs Python parity (when C++ extension is available) ----
try:
    import trade_degradation_cpp as cpp  # noqa: F401
    HAS_CPP = True
except Exception:
    HAS_CPP = False


@unittest.skipIf(not HAS_CPP, "C++ extension not built")
class TestCppParity(unittest.TestCase):
    def test_cpp_actual_entry_matches_python(self):
        for side in ("long", "short"):
            ideal = 20000.0
            tick = 0.25
            delta_delay_ticks = 2.5
            slip_ticks = 1.0
            actual_cpp = cpp.actual_entry_from_ticks(side, ideal, tick, delta_delay_ticks, slip_ticks)
            sgn = 1.0 if side == "long" else -1.0
            total_ticks = delta_delay_ticks + sgn * slip_ticks
            actual_py = ideal + ticks_to_price(total_ticks, tick)
            self.assertAlmostEqual(actual_cpp, actual_py)

    def test_cpp_edge_metrics_matches_python(self):
        side = "long"
        ideal = 20000.0
        actual = 20001.0
        stop = 19995.0
        target = 20020.0
        em = cpp.edge_metrics(side, ideal, actual, stop, target)
        rr_ideal_py = compute_edge_R(side, ideal, stop, target)
        rr_actual_py = compute_edge_R(side, actual, stop, target)
        self.assertAlmostEqual(em.rr_ideal, rr_ideal_py)
        self.assertAlmostEqual(em.rr_actual, rr_actual_py)
        ratio_py = rr_actual_py / rr_ideal_py if rr_ideal_py > 1e-9 else 0.0
        self.assertAlmostEqual(em.edge_ratio, ratio_py)

    def test_simulate_once_cpp_invariants(self):
        spec = _make_spec()
        inp = SimInput(delay_seconds=0.2, regime="Normal", forced_slip_ticks=2.0)
        r = simulate_once(spec, inp, rng=random.Random(7))
        self.assertTrue(r.details["cpp_path"])
        self.assertGreaterEqual(r.edge_remaining_pct, 0)
        self.assertLessEqual(r.edge_remaining_pct, 100)
        self.assertAlmostEqual(r.edge_ideal_R, 4.0)
        self.assertGreaterEqual(r.details["lateness_ticks"], 0)


# ---- Hand-computed math ----
class TestMathSanity(unittest.TestCase):
    """Verify core formulas with hand-computed cases."""

    def test_zero_delay_zero_slip_gives_full_edge(self):
        # delay=0 => sigma_delay=0 => delta_delay_ticks=0; forced_slip=0 => actual_entry = ideal
        spec = TradeSpec(
            side="long",
            ideal_entry=20000.0,
            stop=19995.0,
            target=20020.0,
            tick_size=0.25,
        )
        inp = SimInput(delay_seconds=0.0, regime="Normal", forced_slip_ticks=0.0)
        r = simulate_once(spec, inp, rng=random.Random(999))
        self.assertAlmostEqual(r.actual_entry, 20000.0)
        self.assertAlmostEqual(r.edge_ideal_R, 4.0)
        self.assertAlmostEqual(r.edge_actual_R, 4.0)
        self.assertAlmostEqual(r.edge_remaining_pct, 100.0)
        self.assertEqual(r.classification, "Clean")

    def test_one_tick_worse_long_reduces_edge_correctly(self):
        # Long: ideal 20000, stop 19995, target 20020 => ideal RR = 20/5 = 4
        # If actual_entry = 20000.25 (1 tick worse): risk=5.25, reward=19.75 => RR = 19.75/5.25 = 3.76...
        # edge_ratio = 3.76/4 ≈ 0.940, so edge_remaining ≈ 94%
        spec = TradeSpec(
            side="long",
            ideal_entry=20000.0,
            stop=19995.0,
            target=20020.0,
            tick_size=0.25,
        )
        inp = SimInput(delay_seconds=0.0, regime="Normal", forced_slip_ticks=1.0)
        r = simulate_once(spec, inp, rng=random.Random(0))
        # With delay=0, delta_delay_ticks=0; slip=1, sgn=1 => actual = 20000 + 0.25*1 = 20000.25
        self.assertAlmostEqual(r.actual_entry, 20000.25)
        self.assertAlmostEqual(r.edge_ideal_R, 4.0)
        expected_rr_actual = (20020.0 - 20000.25) / (20000.25 - 19995.0)  # 19.75 / 5.25
        self.assertAlmostEqual(r.edge_actual_R, expected_rr_actual)
        expected_ratio = expected_rr_actual / 4.0
        self.assertAlmostEqual(r.edge_remaining_pct, min(1.0, expected_ratio) * 100.0)

    def test_short_one_tick_worse(self):
        # Short: ideal 20000, stop 20005, target 19980 => ideal RR = 20/5 = 4
        # 1 tick worse for short = lower fill: actual = 20000 - 0.25 = 19999.75
        # risk = 20005 - 19999.75 = 5.25, reward = 19999.75 - 19980 = 19.75 => RR = 19.75/5.25
        spec = TradeSpec(
            side="short",
            ideal_entry=20000.0,
            stop=20005.0,
            target=19980.0,
            tick_size=0.25,
        )
        inp = SimInput(delay_seconds=0.0, regime="Normal", forced_slip_ticks=1.0)
        r = simulate_once(spec, inp, rng=random.Random(0))
        self.assertAlmostEqual(r.actual_entry, 19999.75)
        self.assertAlmostEqual(r.edge_ideal_R, 4.0)
        expected_rr = (19999.75 - 19980.0) / (20005.0 - 19999.75)
        self.assertAlmostEqual(r.edge_actual_R, expected_rr)

    def test_degradation_zero_when_no_loss(self):
        spec = TradeSpec(
            side="long",
            ideal_entry=20000.0,
            stop=19995.0,
            target=20020.0,
            tick_size=0.25,
        )
        inp = SimInput(delay_seconds=0.0, regime="Normal", forced_slip_ticks=0.0)
        r = simulate_once(spec, inp, rng=random.Random(1))
        self.assertAlmostEqual(r.edge_remaining_pct, 100.0)
        self.assertLess(r.degradation_score, 1.0)  # should be ~0 (only tiny float noise)

    def test_inverted_trade_zero_rr(self):
        # Entry past stop (long): risk would be negative; we should get 0 RR
        self.assertEqual(compute_edge_R("long", 19994.0, 19995.0, 20020.0), 0.0)
        self.assertEqual(compute_edge_R("short", 20006.0, 20005.0, 19980.0), 0.0)


# ---- Regimes ----
class TestRegimes(unittest.TestCase):
    def test_defined(self):
        self.assertIn("Normal", REGIMES)
        self.assertIn("High", REGIMES)
        r = REGIMES["Normal"]
        self.assertEqual(r.sigma_per_sqrt_s_ticks, 1.0)
        self.assertEqual(r.slip_mu_ticks, 0.5)

    def test_unknown_regime_raises(self):
        spec = _make_spec()
        inp = SimInput(delay_seconds=0.0, regime="UnknownRegime", forced_slip_ticks=0.0)
        with self.assertRaises(ValueError):
            simulate_once(spec, inp, rng=random.Random(0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
