# Feyer: Faster Physics — Silly Entropy

**Complete first run: 20.805 s → 1.538 s (13.53× faster, 92.61% less time).**

Apple M4, CPU only. Mean of 30 independent Python processes per version, with
the compilation cache disabled. All 30 runs of the final version passed the
benchmark checks. The physics is the same: all 101 frequencies, all 193 design
variables, float64 / complex128.

## Results

Means across 30 independent processes per version (original `83f8d2f` vs.
final `623b0a6`).

| Phase | Original (s) | Final (s) | Saved (s) | Less time | Speedup |
|---|---:|---:|---:|---:|---:|
| Setup and imports | 1.280 | 0.463 | 0.816 | 63.79% | 2.76× |
| Trace / lower + compile | 8.015 | 0.626 | 7.389 | 92.19% | 12.80× |
| Forward + readout + gradient | 11.492 | 0.448 | 11.044 | 96.10% | 25.64× |
| Result transfer | 0.0187 | 0.0002 | 0.0186 | 99.06% | 106.75× |
| **Complete first run** | **20.805** | **1.538** | **19.267** | **92.61%** | **13.53×** |

The final runs are stable. Run 1 took 2.104 s, and runs 2–30 averaged 1.518 s.
Most of the extra time in run 1 (about 85%) is in imports, setup building and
compilation. This locates the extra time; it does not prove its cause. A fresh
Python process does not clear operating-system caches.

## What we changed

The component equations and the physics are unchanged. We changed **how** the
calculation runs. The first run is mostly compile time, so the main idea is
simple: give the XLA compiler less to compile, and do the heavy linear algebra
with fast NumPy/SciPy code that has exact, hand-written gradients.

1. **Group component work.** Each component type is evaluated once over all
   its calls, instead of sending every call through one large switch.
2. **Component formulas in NumPy.** The same component equations run through
   a NumPy backend. Exact local derivatives come from dual numbers. They are
   connected to JAX with `pure_callback` and `custom_vjp`, so JAX does not
   trace them.
3. **Sparse signal and noise solves.** These systems are mostly zeros. We
   store only the non-zero values and solve with SciPy sparse LU: one
   factorisation per frequency serves both the signal solve and the detector
   weight solve. Gradients use the adjoint method. One step of iterative
   refinement keeps the result accurate.
4. **Dense carrier solve.** SciPy dense solve, with an implicit reverse
   derivative.
5. **One value-and-gradient function.** Simulation, readout and gradient are
   compiled together as one executable, instead of three separate stages.
6. **Total internal reflection fix.** For `|z| > 1`, `cos(arcsin z)` uses its
   complex value `-i·sign(z)·sqrt(z² − 1)`, as the original complex JAX code
   does, instead of returning NaN.
7. **Lazy plotting imports**, so plotting libraries do not load in the timed
   run.

## Correctness

Every run checks:

- all outputs and gradients are finite,
- sensitivities match the public reference (maximum absolute natural-log error
  at most 1e-5),
- the supplied finite-difference gradient spot check passes.

The spot check is not a full proof for all inputs. The NumPy component code
targets the current component formulas and real physical parameters; new
components or complex parameters would need their derivatives checked again.

No stored results and no persistent compilation cache are used. All work
happens inside the timed evaluation.

## Install

Use Python 3.12. From this repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

`requirements.txt` keeps the tested package versions. SciPy is required for
the linear solvers. No GPU is needed.

## Reproduce

One complete first run, also saving the physical arrays:

```sh
mkdir -p run-logs
JAX_ENABLE_COMPILATION_CACHE=false python benchmark.py --repeats 0 --output run-logs/check.json --plot run-logs/check.png --save-arrays run-logs/check.npz
```

Several independent processes in a row, with no pause (set `--runs 30` for the
protocol above):

```sh
python run_first.py --runs 10
```

On Linux/WSL, pinned to one logical CPU:

```sh
python run_first.py --runs 10 --cpus 1
```

Do not use `--cpus` on macOS. `benchmark.py` accepts `--repeats 0` only: it
measures the fresh-process first evaluation, not warm repeats. Default XLA
options are used.

## How we measured

- **Timed:** imports, backend start, reading inputs, building the setup,
  tracing / lowering, compiling, forward evaluation, readout, gradient and
  result transfer.
- **Not timed:** interpreter start before the first timer, validation, and
  report / plot writing.
- Each run is a new Python process. The compilation cache is off.
- Original and final runs were measured in separate batches, so they are a
  descriptive comparison, not paired trials.
- Spread is ±1 sample standard deviation, not a confidence interval.
- All numbers are from an Apple M4. We have no official single-CPU Xeon result.

## Files

- `benchmark.py`: the benchmark used for all measurements.
- `differometor/`: the full final package.
- `examples/data/`: the required setup and reference fixtures.
- `run_first.py`: runs several independent first runs and summarises them.
- `reports/final/`: an earlier batch of 10 final runs (mean 1.601 s, median
  1.541 s), with JSON reports, statistics and plots. The raw files of the
  30-run batch above are not in this repository.
- `reports/baseline.json`: one example run of the original version.
- `SOURCE.json`: source commit and code hashes.
- `LICENSE`: upstream MIT license.
