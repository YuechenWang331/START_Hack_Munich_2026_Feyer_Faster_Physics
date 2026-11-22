# Feyer Faster Physics on CPU

This branch builds on the team's sparse signal solver, host component formulas
and total internal reflection fix in `main` at `623b0a6`. It combines component
evaluation, matrix assembly and solving in one CPU callback for the carrier and
another for signal and noise. Their reverse callbacks propagate derivatives
through the same operations. JAX compiles less indexing and array manipulation.
The simulator still evaluates the optical equations at the supplied parameters.

## First-run performance

Each entry summarizes eight independent Python processes on the same Apple M4
with CPU affinity unrestricted. Run order was counterbalanced, compilation
caching was disabled and XLA used its default settings. All numbers are seconds
from the benchmark's `timing_seconds.end_to_end`.

| Timing program | Simulator | Mean | Median | Min–max | Mean saving vs. main |
| --- | --- | ---: | ---: | ---: | ---: |
| Repository Benchmark3 | Main at `623b0a6` | 0.848 | 0.803 | 0.786–1.173 | — |
| Repository Benchmark3 | Fuse signal and noise | 0.644 | 0.644 | 0.629–0.659 | 24.1% |
| Repository Benchmark3 | Fuse signal, noise and carrier (this branch) | **0.618** | **0.618** | **0.605–0.629** | **27.1%** |
| Original three-stage starter | Main at `623b0a6` | 0.951 | 0.870 | 0.862–1.488 | — |
| Original three-stage starter | This branch | **0.688** | **0.670** | **0.661–0.827** | **27.7%** |

The branch beat main in all eight paired rounds under each program. Benchmark3
trace/lower plus compile fell from 0.374 to 0.178 s. Its forward, readout and
gradient execution was 0.234 s on main and 0.233 s on this branch. A fourth
variant changed sparse residual evaluation and averaged 0.620 s. Its timing
overlapped the selected version, so this branch retains main's refinement.

The public sensitivity check and supplied gradient spot check passed in every
run. Against outputs from the original dense starter, the selected version's
maximum absolute natural-log sensitivity difference was 5.22e-8 or less under
both timing programs. All 193 gradient coordinates passed `rtol=0.01, atol=1e-6`.
Two additional evaluations changed both mirror tuning and linked space length
and passed the same full-vector checks. Small dense reference tests cover two
detectors, complex cotangents and repeated coefficient destinations.

Individual first-run totals, phase means, simulator hashes and numerical errors
are in [the M4 results](reports/host-assembly-m4.json). The earlier team's 10-run
M4 batch averaged 1.601 s. These new comparisons use a fresh measurement of main
in the same rounds. Single-logical-CPU Xeon performance remains unmeasured here.

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

To run the original three-stage program from an unmodified starter alongside
this checkout, set `starter` to the starter directory. Run from this repo's root.
The starter requires at least three warm repetitions and excludes them from its
first-run total.

```sh
starter=/path/to/start-hack-munich-2026-main
mkdir -p run-logs
PYTHONPATH="$PWD:$starter" JAX_ENABLE_COMPILATION_CACHE=false python -P "$starter/benchmark.py" --repo "$PWD" --repeats 3 --output run-logs/starter.json --save-arrays run-logs/starter.npz
python -m unittest discover -s tests -v
```

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
- `SOURCE.json` records provenance for the earlier team's version. `LICENSE` contains the upstream MIT license.

The repository was flattened from `combined-four-20261121-184759/`. Old
experiments remain in Git history and on `experiment/host-components5`; they
are not required to run this final version. No stored physical answers or
persistent compilation cache are used to accelerate evaluation.
