"""Prototype evaluation and sampling-to-summary timings; synthetic inputs only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time
from benchmark_scalar import environment, sha, percentile, ROOT
sys.path.insert(0, str(ROOT / 'python'))
import simulator as s
import batch as b


def timed(call, loops):
    start = time.perf_counter_ns()
    for _ in range(loops): result = call()
    return (time.perf_counter_ns() - start) / 1e9, result


def verify(a, z):
    errors = {key: abs(value - asdict(z[0])[key]) for key, value in asdict(a[0]).items()}
    if max(errors.values()) > 1e-10: raise AssertionError(errors)
    if len(a[1]) != len(z[1]): raise AssertionError('sample count mismatch')
    max_sample_error = 0.
    for left, right in zip(a[1], z[1]):
        if left[-1] != right[-1]: raise AssertionError('classification mismatch')
        max_sample_error = max(max_sample_error, *(abs(x - y) for x, y in zip(left, right)))
    if max_sample_error > 1e-10: raise AssertionError(max_sample_error)
    return {'summary_absolute_errors': errors, 'maximum_sample_error': max_sample_error,
            'summary': asdict(a[0]), 'materialized_samples': len(a[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    s.count('repeats', args.repeats)
    if s.cpp is None or not hasattr(s.cpp, 'evaluate_batch'): raise RuntimeError('native batch build required')
    env = environment()
    for name in ('cpp/batch.hpp', 'python/batch.py', 'benchmarks/benchmark_batch.py', 'benchmarks/benchmark_scalar.py'):
        env['source_sha256'][name] = sha(ROOT / name)
    report = {'schema': 'tds-batch-benchmark-v1', 'environment': env, 'seed': 7, 'repeats': args.repeats,
              'method': 'paired alternating backend order by repeat; warmed; conversion and output included; independent latency samples',
              'cases': []}
    for side, expand in (('long', False), ('short', True)):
        sign = s.side_sign(side)
        p = s.TradeSpec(side, 20000, 20000-sign*5, 20000+sign*20, .25, allow_stop_expansion=expand)
        inp = s.SimInput(2, 'High')
        for n, include in ((1, False), (100, False), (10000, False), (10000, True)):
            draws = b.sample_scenarios(p, inp, n, 7)
            for mode in ('evaluation_only', 'sampling_to_summary'):
                calls = {backend: (lambda backend=backend: b.evaluate_samples(p, inp, *draws, backend=backend, include_samples=include))
                         if mode == 'evaluation_only' else
                         (lambda backend=backend: b.run_batch(p, inp, n, 7, backend=backend, include_samples=include))
                         for backend in ('python', 'cpp')}
                reference, native = calls['python'](), calls['cpp']()
                case = {'spec': asdict(p), 'input': asdict(inp), 'n': n, 'include_samples': include, 'mode': mode,
                        'numerical': verify(reference, native), 'measurements': {}}
                times = {key: [] for key in calls}
                loops = max(1, 1000 // n)
                for call in calls.values():
                    for _ in range(3): call()
                for repeat in range(args.repeats):
                    for backend in (('python', 'cpp') if repeat % 2 == 0 else ('cpp', 'python')):
                        seconds, result = timed(calls[backend], loops)
                        times[backend].append(seconds)
                        verify(reference, result)
                for backend, call in calls.items():
                    latency = []
                    for _ in range(101 if n <= 100 else 21):
                        start = time.perf_counter_ns(); call(); latency.append(time.perf_counter_ns() - start)
                    case['measurements'][backend] = {
                        'seconds_per_repeat': times[backend], 'batches_per_repeat': loops,
                        'trials_per_second': n * loops / statistics.median(times[backend]),
                        'latency_samples_ns': latency, 'latency_ns_p50': percentile(latency, .5),
                        'latency_ns_p95': percentile(latency, .95),
                    }
                case['cpp_throughput_ratio_to_python'] = case['measurements']['cpp']['trials_per_second'] / case['measurements']['python']['trials_per_second']
                report['cases'].append(case)
                print(side, n, include, mode, round(case['cpp_throughput_ratio_to_python'], 3), flush=True)
    report['limitations'] = ['One workstation/build; power, thermal and scheduling state uncontrolled.',
                            'Python-generated common draws; C++ RNG is deliberately outside this prototype.',
                            'Summary-only and detailed output costs differ; no historical or broker latency claim.',
                            'Latency p95 from 21 samples for large batches is descriptive, not a service-level bound.',
                            'Baseline is the corrected explicit-draw Python reference; scalar baseline is separately recorded.']
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__': main()
