"""Experimental explicit-draw batch API; independent of all historical research.

Python owns the seeded random stream. Native evaluation never creates its own
RNG: identical float64 draws are the reproducibility boundary. No model is fit,
no certification is promoted and no order is submitted by this module.
"""
from array import array
from dataclasses import asdict
import math
import random
import simulator as s

MAX_TRIALS = 1000000
SAMPLE_FIELDS = ('actual_entry', 'used_stop', 'edge_ideal_R', 'edge_actual_R',
                 'edge_remaining_pct', 'degradation_score', 'stop_ticks_used', 'classification_code')
CLASSIFICATIONS = ('Clean', 'Compromised', 'Invalid')


def sample_scenarios(spec, inp, n, seed=42):
    s.count('n', n)
    if n > MAX_TRIALS:
        raise ValueError('batch exceeds 1000000 trials')
    rng = random.Random(seed)
    deltas, slips = array('d'), array('d')
    sign = s.side_sign(spec.side)
    for _ in range(n):
        delta, slip, sigma = s.sample_draw(inp, rng)
        deltas.append(delta + sign * .35 * sigma)
        slips.append(slip)
    return deltas, slips


def evaluate_samples(spec, inp, deltas, slips, *, backend='python', include_samples=False):
    if backend not in ('python', 'cpp'):
        raise ValueError('batch backend must be explicit: python or cpp')
    if not isinstance(include_samples, bool):
        raise ValueError('include_samples must be bool')
    if any(not isinstance(xs, array) or xs.typecode != 'd' for xs in (deltas, slips)):
        raise ValueError('draws must be native float64 array.array buffers')
    if not 0 < len(deltas) <= MAX_TRIALS or len(deltas) != len(slips):
        raise ValueError('draws must have equal nonzero length <= 1000000')
    if backend == 'cpp':
        if s.cpp is None or not hasattr(s.cpp, 'evaluate_batch'):
            raise RuntimeError('batch extension unavailable; build the P1 prototype')
        raw = s.cpp.evaluate_batch(asdict(spec), asdict(s.REGIMES[inp.regime]), deltas, slips, include_samples)
        return s.MCStats(**raw['stats']), raw['samples']
    # Independent readable Python reference, using the tested scalar contract.
    values, scores, counts, rows = [], [], [0, 0, 0], []
    for delta, slip in zip(deltas, slips):
        result = s.evaluate_draw(spec, inp, delta, slip, backend='python')
        code = CLASSIFICATIONS.index(result.classification)
        counts[code] += 1
        values.append(result.rr_retention_pct)
        scores.append(result.degradation_score)
        if include_samples:
            rows.append([result.actual_entry, result.used_stop, result.edge_ideal_R, result.edge_actual_R,
                         result.rr_retention_pct, result.degradation_score, result.details['stop_ticks_used'], code])
    n = len(values)
    return s.MCStats(n, *(x / n for x in counts), math.fsum(values) / n,
                     *(s._percentile(values, p) for p in (10, 50, 90)), math.fsum(scores) / n), rows


def run_batch(spec, inp, n, seed=42, *, backend='python', include_samples=False):
    deltas, slips = sample_scenarios(spec, inp, n, seed)
    return evaluate_samples(spec, inp, deltas, slips, backend=backend, include_samples=include_samples)
