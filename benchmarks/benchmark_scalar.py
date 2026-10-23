"""Matched corrected scalar backends. Synthetic inputs only; no data imports.

Throughput excludes per-call clock instrumentation; latency includes its overhead.
Results are exclusive-create JSON. Timings include Python result construction and
native crossings. These are computation measurements, not live execution latency.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import simulator as s


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(xs, p):
    return sorted(xs)[round(p * (len(xs) - 1))]


def environment():
    cache = ROOT / 'cpp/build/release/CMakeCache.txt'
    entries = {line.split('=', 1)[0].split(':', 1)[0]: line.split('=', 1)[1]
               for line in cache.read_text().splitlines()
               if '=' in line and not line.startswith(('#', '//'))}
    pybind_source = Path(entries['pybind11_SOURCE_DIR'])
    return {
        'benchmark_runner_sha256': sha(__file__),
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'machine': platform.machine(),
        'processor': command('/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'),
        'logical_cpus': command('/usr/sbin/sysctl', '-n', 'hw.logicalcpu'),
        'memory_bytes': command('/usr/sbin/sysctl', '-n', 'hw.memsize'),
        'compiler': command('c++', '--version'), 'cmake': command('cmake', '--version'),
        'pybind11_commit': command('git', '-C', str(pybind_source), 'rev-parse', 'HEAD'),
        'revision': command('git', 'rev-parse', 'HEAD'),
        'tracked_changes': command('git', 'diff', '--name-only'),
        'source_sha256': {name: sha(ROOT / name) for name in
                          ('python/simulator.py', 'python/regimes.py', 'cpp/degradation.hpp', 'cpp/bindings.cpp', 'cpp/CMakeLists.txt')},
        'extension': str(s.cpp.__file__), 'extension_sha256': sha(s.cpp.__file__),
        'build_configuration': [line for line in cache.read_text().splitlines()
                                if line.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_FLAGS:', 'CMAKE_CXX_FLAGS_RELEASE:', 'CMAKE_INTERPROCEDURAL_OPTIMIZATION:'))],
        'execution': 'single process, one Python thread, warmed extension; OS scheduling and power state uncontrolled',
        'timer': 'perf_counter_ns; latency includes timer/invocation overhead; no overhead subtraction',
    }


def numerical_results(spec, inp, n, seed):
    output = {}
    for backend in ('python', 'cpp'):
        rng = random.Random(seed)
        values = [s.simulate_once(spec, inp, rng, backend=backend) for _ in range(n)]
        output[backend] = values
    max_error = 0.
    for a, b in zip(output['python'], output['cpp']):
        if a.classification != b.classification:
            raise AssertionError('classification mismatch')
        for field in ('actual_entry', 'used_stop', 'edge_ideal_R', 'edge_actual_R', 'edge_remaining_pct', 'degradation_score'):
            error = abs(getattr(a, field) - getattr(b, field))
            max_error = max(max_error, error)
            if error > 1e-10:
                raise AssertionError((field, error))
    values = output['python']
    return {'samples': n, 'max_absolute_error': max_error,
            'rr_retention_sum': sum(v.rr_retention_pct for v in values),
            'entry_sum': sum(v.actual_entry for v in values),
            'counts': {c: sum(v.classification == c for v in values) for c in ('Clean', 'Compromised', 'Invalid')},
            'first_result': asdict(values[0])}


def measure(spec, inp, backend, n, repeats, seed):
    for _ in range(200):
        s.simulate_once(spec, inp, random.Random(seed), backend=backend)
    durations, checksums = [], []
    for _ in range(repeats):
        rng = random.Random(seed)
        start = time.perf_counter_ns()
        checksum = sum(s.simulate_once(spec, inp, rng, backend=backend).rr_retention_pct for _ in range(n))
        durations.append((time.perf_counter_ns() - start) / 1e9)
        checksums.append(checksum)
    rng = random.Random(seed)
    latencies = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        s.simulate_once(spec, inp, rng, backend=backend)
        latencies.append(time.perf_counter_ns() - start)
    return {'seconds': durations, 'trials_per_second': n / statistics.median(durations),
            'latency_ns_p50': percentile(latencies, .5), 'latency_ns_p95': percentile(latencies, .95),
            'latency_ns_p99': percentile(latencies, .99), 'latency_samples': len(latencies),
            'checksums': checksums}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--trials', type=int, default=10000)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    s.count('trials', args.trials); s.count('repeats', args.repeats)
    if s.cpp is None:
        raise RuntimeError('native extension required for matched comparison')
    report = {'schema': 'tds-scalar-benchmark-v1', 'environment': environment(),
              'seed': args.seed, 'trials_per_repeat': args.trials, 'repeats': args.repeats,
              'comparison': 'P0-corrected Python versus P0-corrected scalar extension, identical seeded workloads', 'cases': []}
    for side in ('long', 'short'):
        sign = s.side_sign(side)
        for expand in (False, True):
            spec = s.TradeSpec(side, 20000, 20000 - sign * 5, 20000 + sign * 20, .25, allow_stop_expansion=expand)
            inp = s.SimInput(2, 'High')
            case = {'spec': asdict(spec), 'input': asdict(inp),
                    'numerical': numerical_results(spec, inp, args.trials, args.seed), 'measurements': {}}
            # Alternate backend order between cases to reduce systematic order bias.
            order = ('python', 'cpp') if len(report['cases']) % 2 == 0 else ('cpp', 'python')
            for backend in order:
                case['measurements'][backend] = measure(spec, inp, backend, args.trials, args.repeats, args.seed)
            py, cpp = (case['measurements'][b] for b in ('python', 'cpp'))
            case['cpp_throughput_ratio_to_python'] = cpp['trials_per_second'] / py['trials_per_second']
            if py['checksums'] != cpp['checksums']:
                raise AssertionError('timed numerical results differ')
            report['cases'].append(case)
            print(side, 'expanded' if expand else 'fixed', 'cpp/python throughput', round(case['cpp_throughput_ratio_to_python'], 3), flush=True)
    report['limitations'] = ['Synthetic uncalibrated scenarios; not historical or live latency.',
                            'One workstation and build; scheduling, thermal and power state uncontrolled.',
                            'Scalar paths retain Python sampling, validation, classification, and object construction.',
                            'Does not compare the original defective formulas to corrected formulas.']
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
