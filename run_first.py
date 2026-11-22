"""Run serial fresh-process benchmarks with no added pause."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--cpus', type=int, default=None)
args = parser.parse_args()
if args.runs < 1:
    parser.error('--runs must be positive')
root = Path(__file__).resolve().parent
(root / 'run-logs').mkdir(exist_ok=True)
out = Path(tempfile.mkdtemp(prefix='first-', dir=root / 'run-logs'))
env = dict(os.environ, JAX_ENABLE_COMPILATION_CACHE='false', PYTHONDONTWRITEBYTECODE='1')
rows = []
for i in range(1, args.runs + 1):
    print(f'\n=== Independent process {i}/{args.runs} ===', flush=True)
    prefix = out / f'run-{i:02d}'
    cmd = [sys.executable, '-u', str(root / 'benchmark.py'), '--repeats', '0',
           '--output', str(prefix.with_suffix('.json')), '--plot', str(prefix.with_suffix('.png'))]
    if args.cpus is not None:
        cmd += ['--cpus', str(args.cpus)]
    with prefix.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f'Run {i} failed; see {prefix}.log')
    report = json.loads(prefix.with_suffix('.json').read_text())
    checks = report['validation']
    assert checks['all_outputs_and_gradients_finite']
    assert checks['upstream_sensitivity_check_passed']
    assert checks['gradient_check']['passed']
    rows.append({'run': i, 'first_run_seconds': report['timing_seconds']['end_to_end']})
    with (out / 'runs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
values = [r['first_run_seconds'] for r in rows]
summary = {'runs': len(values), 'mean': statistics.mean(values),
           'median': statistics.median(values),
           'sample_sd': statistics.stdev(values) if len(values) > 1 else None,
           'min': min(values), 'max': max(values)}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
print(f'PASS: all {len(values)} runs passed. Results: {out}')
