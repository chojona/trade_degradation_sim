"""Numerical contract regressions, including failures reproduced in the audit."""
import dataclasses
import math
import os
import random
import unittest
from unittest.mock import patch
import simulator as s
from monte_carlo import run_monte_carlo as external_mc

if os.environ.get("TDS_REQUIRE_CPP") == "1" and s.cpp is None:
    raise RuntimeError("Native validation requested but extension is unavailable")


def spec(**kw):
    return dataclasses.replace(s.TradeSpec('long', 20000., 19995., 20020., .25), **kw)


class NumericalRegressions(unittest.TestCase):
    def test_minimum_rr_is_hard_gate(self):
        r = s.simulate_once(spec(min_rr=5), s.SimInput(0, forced_slip_ticks=0))
        self.assertEqual(r.edge_actual_R, 4)
        self.assertEqual(r.classification, 'Invalid')

    def test_stop_cap_is_actual_fill_distance(self):
        r = s.simulate_once(spec(max_stop_ticks=20), s.SimInput(0, forced_slip_ticks=4))
        self.assertEqual(r.details['stop_ticks_used'], 24)
        self.assertEqual(r.classification, 'Invalid')

    def test_original_rr_survives_stop_expansion(self):
        for backend in ('python', 'cpp') if s.cpp else ('python',):
            r = s.simulate_once(spec(allow_stop_expansion=True), s.SimInput(0, forced_slip_ticks=0), backend=backend)
            self.assertEqual(r.used_stop, 19994.5)  # 21.6 ticks rounded outward to 22
            self.assertEqual(r.edge_ideal_R, 4)
            self.assertAlmostEqual(r.edge_actual_R, 20 / 5.5)
            self.assertAlmostEqual(r.rr_retention_pct, 100 * (20 / 5.5) / 4)

    def test_price_and_expanded_stop_are_on_grid(self):
        for side in ('long', 'short'):
            t = spec(side=side, stop=19995 if side == 'long' else 20005,
                     target=20020 if side == 'long' else 19980, allow_stop_expansion=True)
            for seed in range(100):
                r = s.simulate_once(t, s.SimInput(1), random.Random(seed))
                self.assertAlmostEqual(r.actual_entry / .25, round(r.actual_entry / .25))
                self.assertAlmostEqual(r.used_stop / .25, round(r.used_stop / .25))
        r = s.evaluate_draw(spec(), s.SimInput(1), .09411971155, 0, backend='python')
        self.assertEqual(r.actual_entry, 20000.25)

    def test_expansion_does_not_shrink_to_hide_bad_cap(self):
        r = s.simulate_once(spec(allow_stop_expansion=True, max_stop_ticks=10), s.SimInput(0, forced_slip_ticks=0))
        self.assertEqual(r.used_stop, 19995)
        self.assertEqual(r.classification, 'Invalid')

    def test_invalid_specifications_rejected(self):
        for field, values in {
            'tick_size': [0, -.25, math.nan, math.inf, True],
            'ideal_entry': [math.nan, math.inf, 20000.1],
            'stop': [math.nan, 20001, 19995.1],
            'target': [math.inf, 19999, 20020.1],
            'max_stop_ticks': [0, -1, math.inf],
            'min_rr': [-1, math.nan],
            'max_late_ticks_gate': [-1, math.nan],
            'max_entry_distance_ticks': [-1, math.inf],
            'allow_stop_expansion': [1, 'yes'],
            'side': ['', None, 'unknown'],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    spec(**{field: value})

    def test_invalid_inputs_and_draws_rejected(self):
        for delay in (-1, math.nan, math.inf, True):
            with self.assertRaises(ValueError): s.SimInput(delay)
        for slip in (-1, math.nan, math.inf, True):
            with self.assertRaises(ValueError): s.SimInput(0, forced_slip_ticks=slip)
        for delta, slip in ((math.nan, 0), (math.inf, 0), (0, -1)):
            with self.assertRaises(ValueError): s.evaluate_draw(spec(), s.SimInput(0), delta, slip)
        with self.assertRaises(ValueError): s.price_to_ticks(1, 0)
        with self.assertRaises(ValueError): s.ticks_to_price(1, -.25)
        with self.assertRaises(ValueError): s.grid_units(1e20, .25)

    def test_helper_nonfinite_and_grid_precision(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError): s.clamp(value, 0, 1)
        with self.assertRaises(ValueError): s.clamp(0, 2, 1)
        for side, entry, stop, target in [('long', -10., -11., -8.), ('short', 10., 11., 8.)]:
            t = s.TradeSpec(side, entry, stop, target, .1)
            r = s.evaluate_draw(t, s.SimInput(1), .25, 0, backend='python')
            self.assertAlmostEqual(r.actual_entry / .1, round(r.actual_entry / .1))

    def test_invalid_trial_counts_and_probability_inputs(self):
        for n in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): s.run_monte_carlo(spec(), s.SimInput(0), n)
            with self.assertRaises(ValueError): external_mc(spec(), s.SimInput(0), n)
            with self.assertRaises(ValueError): s.prob_bad(spec(), s.SimInput(0), 70, n)
        for p in (0, -1, 1.1, math.nan):
            with self.assertRaises(ValueError): s.max_slippage_allowed_prob(spec(), 0, 'Normal', p_bad=p)
        for floor in (-1, 101, math.nan):
            with self.assertRaises(ValueError): s.prob_bad(spec(), s.SimInput(0), floor)
        with self.assertRaises(ValueError): s.max_delay_allowed_prob(spec(), 0, 'Normal', step_seconds=0)
        with self.assertRaises(ValueError): s.max_slippage_allowed_prob(spec(), 0, 'Normal', max_search_ticks=-1)

    def test_infeasible_and_zero_boundary_are_distinct(self):
        t = spec(min_rr=5)
        self.assertIsNone(s.max_slippage_allowed(t, 0, 'Normal'))
        self.assertIsNone(s.max_slippage_allowed_prob(t, 0, 'Normal', trials=10))
        self.assertIsNone(s.max_delay_allowed_prob(t, 0, 'Normal', trials=10))
        self.assertEqual(s.max_slippage_allowed_prob(spec(max_late_ticks_gate=0), 0, 'Normal', trials=10), 0)
        self.assertEqual(s.max_delay_allowed_prob(spec(), 0, 'Normal', trials=10, max_search_seconds=0), 0)
        self.assertIsNone(s.max_slippage_allowed_prob(spec(), 60, 'Event', 70, .4, 300, 60, 7))

    def test_nonmonotonic_safe_islands_do_not_extend_prefix(self):
        # Passing cap does not excuse failure at an earlier sampled point.
        with patch.object(s, 'prob_bad', side_effect=lambda t, i, *a: .8 if i.forced_slip_ticks == 2 else .1):
            self.assertEqual(s.max_slippage_allowed_prob(spec(), 0, 'Normal', max_search_ticks=4), 1)
        with patch.object(s, 'prob_bad', side_effect=lambda t, i, *a: .8 if .19 < i.delay_seconds < .21 else .1):
            self.assertAlmostEqual(s.max_delay_allowed_prob(spec(), 0, 'Normal', max_search_seconds=.4), .1)

    def test_mc_facades_have_identical_seeded_summaries(self):
        inp = s.SimInput(2, 'High')
        summary = s.run_monte_carlo(spec(), inp, 100, 7)
        other, edges = external_mc(spec(), inp, 100, 7)
        self.assertEqual(summary, other)
        self.assertEqual(len(edges), 100)
        self.assertAlmostEqual(summary.p_clean + summary.p_compromised + summary.p_invalid, 1)

    def test_rr_is_not_expectancy(self):
        self.assertEqual(s.simulate_once(spec(), s.SimInput(0, forced_slip_ticks=0)).details['metric'],
                         'reward_risk_retention_not_expected_profitability')


class UIContractRegressions(unittest.TestCase):
    def test_seed_infeasibility_and_distribution_counts(self):
        # Execute the app against an in-memory UI adapter, without Streamlit/server.
        import runpy
        import sys
        import types
        from pathlib import Path
        class UI:
            def __init__(self): self.charts = []
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def __getattr__(self, name):
                if name == 'sidebar': return self
                if name == 'columns': return lambda sizes, **kw: [self] * (sizes if isinstance(sizes, int) else len(sizes))
                if name == 'selectbox': return lambda label, options, **kw: options[kw.get('index', 0)]
                if name == 'number_input': return lambda label, **kw: kw['value']
                if name == 'slider': return lambda label, *a, **kw: kw.get('value', a[2] if len(a) > 2 else None)
                if name == 'checkbox': return lambda *a, **kw: False
                if name == 'button': return lambda *a, **kw: True
                if name == 'spinner': return lambda *a, **kw: self
                if name == 'bar_chart': return lambda data, **kw: self.charts.append(data)
                return lambda *a, **kw: None
        ui = UI()
        module = types.ModuleType('streamlit')
        module.__getattr__ = ui.__getattr__
        with patch.dict(sys.modules, {'streamlit': module}), \
             patch.object(s, 'max_slippage_allowed_prob', return_value=None), \
             patch.object(s, 'max_delay_allowed_prob', return_value=None):
            a = runpy.run_path(str(Path(__file__).with_name('app.py')))
            b = runpy.run_path(str(Path(__file__).with_name('app.py')))
        self.assertEqual(a['r'], b['r'])
        self.assertEqual(len(ui.charts), 2)
        self.assertEqual(sum(ui.charts[0]['Trials']), 2000)


@unittest.skipIf(s.cpp is None, 'C++ extension not built')
class NativeRegressions(unittest.TestCase):
    def test_full_result_parity(self):
        for side in ('long', 'short', ' BUY ', 'Sell'):
            sign = s.side_sign(side)
            for expand in (False, True):
                t = spec(side=side, stop=20000 - sign * 5, target=20000 + sign * 20, allow_stop_expansion=expand)
                for seed in range(50):
                    a = s.simulate_once(t, s.SimInput(2, 'High'), random.Random(seed), backend='python')
                    b = s.simulate_once(t, s.SimInput(2, 'High'), random.Random(seed), backend='cpp')
                    self.assertEqual(a.classification, b.classification)
                    for field in ('actual_entry', 'used_stop', 'edge_ideal_R', 'edge_actual_R', 'edge_remaining_pct', 'degradation_score'):
                        self.assertAlmostEqual(getattr(a, field), getattr(b, field), places=10)

    def test_native_invalid_inputs_and_aliases(self):
        self.assertEqual(s.cpp.rr(' BUY ', 20000, 19995, 20020), 4)
        for tick in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError): s.cpp.price_to_ticks(1, tick)
            with self.assertRaises(ValueError): s.cpp.actual_entry_from_ticks('long', 20000, tick, 0, 0)
        for entry in (math.nan, math.inf):
            with self.assertRaises(ValueError): s.cpp.rr('long', entry, 19995, 20020)
        with self.assertRaises(ValueError): s.cpp.actual_entry_from_ticks('long', 20000, .25, 0, -1)
        self.assertEqual(s.cpp.actual_entry_from_ticks('long', 20000, .25, .1, 0), 20000.25)
        self.assertEqual(s.cpp.actual_entry_from_ticks('short', 20000, .25, -.1, 0), 19999.75)


if __name__ == '__main__':
    unittest.main()
