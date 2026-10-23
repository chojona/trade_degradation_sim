from array import array
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import math
import unittest
import batch as b
import simulator as s


def plan(side='long', expand=False, tick=.25, entry=20000):
    sign = s.side_sign(side)
    return s.TradeSpec(side, entry, entry - sign * 20 * tick, entry + sign * 80 * tick,
                       tick, allow_stop_expansion=expand)


class BatchReferenceTests(unittest.TestCase):
    def test_seeded_draws_and_scalar_summary(self):
        p, inp = plan(), s.SimInput(2, 'High')
        self.assertEqual(b.sample_scenarios(p, inp, 100, 7), b.sample_scenarios(p, inp, 100, 7))
        batch, _ = b.run_batch(p, inp, 100, 7)
        self.assertEqual(batch, s.run_monte_carlo(p, inp, 100, 7, backend='python'))

    def test_invalid_batch_contract(self):
        p, inp, ok = plan(), s.SimInput(0), array('d', [0])
        for a, z in ((array('d'), array('d')), (ok, array('d', [0, 1])), ([0], ok), (array('f', [0]), ok)):
            with self.assertRaises(ValueError): b.evaluate_samples(p, inp, a, z)
        for n in (0, -1, True, 1000001):
            with self.assertRaises(ValueError): b.sample_scenarios(p, inp, n)
        with self.assertRaises(ValueError): b.evaluate_samples(p, inp, ok, ok, backend='auto')
        with self.assertRaises(ValueError): b.evaluate_samples(p, inp, ok, ok, include_samples=1)
        for delta, slip in ((math.nan, 0), (math.inf, 0), (0, -1)):
            with self.assertRaises(ValueError): b.evaluate_samples(p, inp, array('d', [delta]), array('d', [slip]))


@unittest.skipIf(s.cpp is None or not hasattr(s.cpp, 'evaluate_batch'), 'batch extension unavailable')
class NativeBatchTests(unittest.TestCase):
    def compare(self, p, inp, deltas, slips):
        reference, rows = b.evaluate_samples(p, inp, deltas, slips, backend='python', include_samples=True)
        native, actual_rows = b.evaluate_samples(p, inp, deltas, slips, backend='cpp', include_samples=True)
        for key, expected in asdict(reference).items():
            self.assertAlmostEqual(getattr(native, key), expected, places=10, msg=key)
        self.assertEqual(len(rows), len(actual_rows))
        for expected, actual in zip(rows, actual_rows):
            self.assertEqual(expected[-1], actual[-1])
            for a, z in zip(expected[:-1], actual[:-1]): self.assertAlmostEqual(a, z, places=10)
        compact, compact_rows = b.evaluate_samples(p, inp, deltas, slips, backend='cpp')
        self.assertEqual(native, compact)
        self.assertEqual(compact_rows, [])

    def test_all_regimes_directions_and_expansion(self):
        for side in ('long', 'short'):
            for expand in (False, True):
                for regime in s.REGIMES:
                    p, inp = plan(side, expand), s.SimInput(2.5, regime)
                    self.compare(p, inp, *b.sample_scenarios(p, inp, 1000, 17))

    def test_hand_cases_and_signed_decimal_grids(self):
        for tick, entry in ((.25, 20000), (.1, -10), (.1, 10)):
            for side in ('long', 'short'):
                p = plan(side, True, tick, entry)
                self.compare(p, s.SimInput(0), array('d', [0, .1, -.1, 4, -25, 100]), array('d', [0, 1, 0, .5, 0, 0]))
        native, rows = b.evaluate_samples(plan(), s.SimInput(0), array('d', [0]), array('d', [1]), backend='cpp', include_samples=True)
        self.assertEqual(rows[0][0], 20000.25)
        self.assertAlmostEqual(rows[0][3], 19.75 / 5.25)
        self.assertAlmostEqual(native.edge_remaining_mean, 100 * (19.75 / 5.25) / 4)

    def test_percentiles_ties_to_even(self):
        for n in (1, 2, 6, 16, 36, 46, 96):
            self.compare(plan(), s.SimInput(0), array('d', [i / 10 for i in range(n)]), array('d', [0] * n))

    def test_native_rejects_invalid_buffers_draws_and_specs(self):
        p, inp = plan(), s.SimInput(0)
        rp = asdict(s.REGIMES[inp.regime])
        ok = array('d', [0])
        for a, z in ((array('d'), array('d')), (ok, array('d', [0, 1])), (array('f', [0]), ok),
                     (memoryview(array('d', [0, 0, 0]))[::2], array('d', [0, 0])),
                     (array('d', [math.nan]), ok), (ok, array('d', [-1]))):
            with self.assertRaises(ValueError): s.cpp.evaluate_batch(asdict(p), rp, a, z)
        for key, value in (('tick_size', 0), ('ideal_entry', 20000.1), ('max_late_ticks_gate', -1), ('stop', 20001)):
            fields = asdict(p); fields[key] = value
            with self.assertRaises(ValueError): s.cpp.evaluate_batch(fields, rp, ok, ok)
        with self.assertRaises(ValueError): s.cpp.evaluate_batch({}, rp, ok, ok)
        bad_regime = dict(rp, sigma_per_sqrt_s_ticks=math.nan)
        with self.assertRaises(ValueError): s.cpp.evaluate_batch(asdict(p), bad_regime, ok, ok)

    def test_repeatability_concurrent_calls_and_input_immutability(self):
        p, inp = plan(), s.SimInput(2, 'Event')
        deltas, slips = b.sample_scenarios(p, inp, 2000, 7)
        before = (deltas.tobytes(), slips.tobytes())
        def call(_): return b.evaluate_samples(p, inp, deltas, slips, backend='cpp', include_samples=True)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(call, range(8)))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(before, (deltas.tobytes(), slips.tobytes()))


if __name__ == '__main__':
    unittest.main()
