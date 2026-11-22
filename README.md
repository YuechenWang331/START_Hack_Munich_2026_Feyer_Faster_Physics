# Feyer: Faster Physics — final CPU version

Complete first-run optical evaluation using Quan's sparse signal solver and the
host-components5 component/carrier implementation. The simulator and Benchmark3
are the exact code used for the recorded **1.600749 s mean** across 10 independent
processes on an Apple M4 (median 1.540963 s; sample SD 0.202803 s). All ten runs
passed validation. This is a historical local measurement, not an official
single-CPU Xeon result or a guaranteed runtime.

## Install

Use Python 3.12. From this repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

`requirements.txt` preserves the tested environment versions. SciPy is required
for the host linear solvers. No GPU is required.

## Reproduce

One complete first run with saved physical arrays:

```sh
mkdir -p run-logs
JAX_ENABLE_COMPILATION_CACHE=false python benchmark.py --repeats 0 --output run-logs/check.json --plot run-logs/check.png --save-arrays run-logs/check.npz
```

Ten serial independent processes, without added pauses (the recorded protocol):

```sh
python run_first.py --runs 10
```

For Linux/WSL on one allocated logical CPU:

```sh
python run_first.py --runs 10 --cpus 1
```

Do not use `--cpus` on macOS. This Benchmark3 variant accepts `--repeats 0`
only; it measures fresh-process first evaluation, not warm iterations. Default
XLA optimization is used, with the compilation cache disabled. No custom
level-0 compiler option is enabled.

Imports, backend initialization, input preparation, lowering, compilation,
forward evaluation, readout, gradients and result transfer are counted.
Validation and report/plot generation are outside the timer. Interpreter startup
before the first Python timer is excluded by the existing benchmark.

## Correctness and approach

The workload retains all 101 frequencies, all 193 physical design variables,
and float64/complex128 precision. Every recorded run checks finite outputs and
gradients, positive sensitivities against the public reference with maximum
absolute natural-log error at most 1e-5, and the supplied finite-difference
gradient spot check. That spot check is not an exhaustive proof for all inputs.

The signal solver uses SciPy sparse LU with custom differentiation. Carrier
solves use SciPy dense solves and an implicit reverse derivative. Component
formulas are reused through a NumPy backend with analytic local derivatives,
connected to JAX through custom VJPs. This reduces the graph that XLA must trace
and compile. Computation still occurs inside the timed evaluation. The host
component implementation targets the current component formulas and real
physical parameters; extensions require derivative validation.

## Files and evidence

- `benchmark.py`: measured Benchmark3, unchanged.
- `differometor/`: complete final package, including public helper/plot modules.
- `examples/data/`: the required setup and reference fixtures.
- `run_first.py`: independent-process reproduction helper.
- `reports/final/`: all ten original JSON reports, summary, CSV statistics and
  the phase/totals plots. Original absolute paths in these archived reports
  describe the measurement location; use the commands above for this layout.
- `reports/baseline.json`: historical original baseline report. Different
  batches are descriptive comparisons, not paired trials.
- `SOURCE.json`: source commit and code hashes; `LICENSE`: upstream MIT license.

The repository was flattened from `combined-four-20261121-184759/`. Old
experiments remain in Git history and on `experiment/host-components5`; they
are not required to run this final version. No stored physical answers or
persistent compilation cache are used to accelerate evaluation.
