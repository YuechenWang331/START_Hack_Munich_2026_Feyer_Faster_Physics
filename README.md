# Feyer: Faster Physics — Silly Entropy

**Complete first run: 20.805 s → 1.538 s (13.53× faster, 92.61% less time).**

Apple M4, CPU only. Mean of 30 independent Python processes per version, with
the compilation cache disabled. All 30 runs of the final version passed the
benchmark checks. The physics is the same: all 101 frequencies, all 193 design
variables, float64 / complex128.

The speedup is in the simulator (`differometor/`). It still takes any setup
definition and input parameters and produces the same outputs, and it passes
the **official, unchanged `benchmark.py`** from the starter repository (see
[Official benchmark check](#official-benchmark-check)).

## Challenge criteria

| Criterion | How this version meets it |
|---|---|
| Any optical input, requested frequencies, design variables and precision | The simulator builds everything from the setup definition at run time: the sparsity pattern, the component groups and the solves. Nothing is specific to the public setup. All frequencies and all design variables are kept, in float64 / complex128. |
| All required physical outputs, sensitivities and gradients | Same outputs as the starter: carrier, signal, noise, sensitivity, powers and the reverse-mode gradient for all 193 variables. |
| Complete first-run timing, including setup and compilation | All timings below are complete first runs in a fresh process: imports, setup, tracing, compiling, evaluation, gradient and result transfer. No persistent compilation cache and no stored results. |
| Same correctness requirements and CPU limits | Every run passes the benchmark checks (finite outputs and gradients, sensitivity within 1e-5 in natural log, finite-difference gradient check). CPU only; `--cpus` pins to one logical CPU on Linux. |

## Results

Means across 30 independent processes per version: the original starter code
vs. the final version (`623b0a6`), measured with the `benchmark.py` in this
repository (see [How we measured](#how-we-measured)).

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

### Other machines

Approximate complete first-run times from tests on other machines (original →
final). These are rough test numbers; their raw files are not in this
repository.

| Machine | Original | Final |
|---|---:|---:|
| Apple M4 (macOS) | ~20 s | ~1.5 s |
| AMD Ryzen 9 | 90–110 s | 2–4 s |
| Intel i7 (Linux) | ~50 s | ~6.0 s |
| Intel i5 | ~60 s | ~5.8 s |

### Official benchmark check

The official `benchmark.py` from the starter repository (unchanged since commit
`02ec741`) was run on this simulator on an Intel i7 laptop (Windows, CPU only):

- **PASS**: outputs and gradients finite, maximum absolute log-sensitivity
  error 5.4e-6 (limit 1e-5), finite-difference gradient check passed.
- Complete first run: 3.14 s (7.96 s on the very first launch, before the
  operating system had cached the libraries). One check, not a timed series.

The official benchmark compiles forward, sensitivity and gradient as three
separate executables, so its first run is slower than with this repository's
`benchmark.py` (see item 6 below).

## What we changed (short version)

The component equations and the physics are unchanged. We changed **how** the
calculation runs. The first run is mostly compile time, so the main idea is
simple: give the XLA compiler less to compile, and do the heavy linear algebra
with fast NumPy/SciPy code that has exact, hand-written gradients.

1. **Group component work.** Each component type is evaluated once over all
   its calls, instead of sending every call through one large switch.
2. **Component formulas in NumPy.** The same equations run through a NumPy
   backend, with exact local derivatives from dual numbers. JAX calls them
   through `pure_callback` and `custom_vjp`, so it does not trace them.
3. **Sparse signal solve.** The signal matrices are mostly zeros. We store
   only the non-zero values and use one SciPy sparse LU per frequency for both
   the signal solve and the detector solve.
4. **No separate noise solve.** The noise power is computed from the detector
   weights we already have, which removes one solve per frequency.
5. **Dense carrier solve in SciPy**, with an implicit reverse derivative.
6. **One value-and-gradient function (measurement script only).** This
   repository's `benchmark.py` compiles simulation, readout and gradient
   together as one executable, instead of in separate stages. This is not part
   of the simulator: items 1–5, 7 and 8 also apply under the official
   benchmark, this one does not.
7. **Total internal reflection fix.** For `|z| > 1`, `cos(arcsin z)` uses its
   complex value `-i·sign(z)·sqrt(z² − 1)`, as the original complex JAX code
   does, instead of returning NaN.
8. **Lazy plotting imports**, so plotting libraries do not load in the timed
   run.

The details are below.

## How the optimizations work

For one design in the public fixture, the simulator solves for 286 steady-state
carrier fields, then 751 coupled signal coordinates at each of 101 frequencies.
The starter's forward pass solves three systems per frequency for one detector:
signal propagation, detector back-propagation and covariance propagation.
The final implementation removes the third system algebraically and shares the
expensive factorization between the first two. The gradient is a separate
reverse pass through this calculation.

### 1. Store the optical connections once and update their values

A mirror or space couples only specific ports. Across a frequency sweep the
coupling coefficients change, while their row and column positions follow from
the optical topology. A dense batch of signal matrices would allocate
$101 \times 751^2$ complex entries, about 869 MiB at 16 bytes per entry,
before counting noise matrices or gradient intermediates.

During setup, `build.to_value_systems` takes the union of the base matrix's
nonzeros and all positions that component instructions can write. This includes
a coupling that happens to be zero at the current design but can acquire a
nonzero derivative. The builder records the signal diagonal, right-hand-side
slots and sideband conjugation mask. During the sweep, `simulate` updates a
frequency-by-stored-entry array. The solver groups those entries by column in
compressed sparse column (CSC) format for SciPy's SuperLU.

### 2. Share one pivoted sparse LU per frequency

At frequency $f$, let $A_f$ be the signal matrix, $b_f$ the injected signal,
and $c_f$ the detector selection vector. The two required systems are

$$
A_f x_f=b_f, \qquad A_f^H w_f=c_f.
$$

`sparse_solve.signal_solves` factorizes a row- and column-permuted $A_f$ into
lower and upper triangular factors. SuperLU uses those factors for the signal
solve and their conjugate transposes for the detector solve. Multiple detectors
are passed as columns of one right-hand side. The 101-frequency forward sweep
therefore uses **101 sparse signal-matrix factorizations**. Each frequency gets
fresh factors because its coefficients depend on frequency.

Both solves receive one step of **iterative refinement**. For an initial
solution $u$ of $Mu=d$, the code computes the residual $r=d-Mu$ using the
assembled operator, solves $M\delta=r$ with the same factors, and returns
$u+\delta$. This corrects numerical error in sensitive optical systems at the
cost of another triangular solve and sparse product.

### 3. Remove the final quantum-noise solve

Let $Q_f$ be the noise-source covariance. The starter forms $Q_f w_f$, solves
for its propagation through $A_f$, and projects the result onto the detector.
Since $A_f^H w_f=c_f$, taking the conjugate transpose gives
$w_f^H=c_f^H A_f^{-1}$. The detector's noise power therefore satisfies

$$
\mathrm{Re}(c_f^H A_f^{-1} Q_f w_f)
= \mathrm{Re}(w_f^H Q_f w_f).
$$

`simulate._sparse_matvec` forms $Q_f w_f$ from the stored noise entries,
including off-diagonal squeezing correlations. A dot product with the conjugate
detector weights gives the power. This saves **101 covariance solves** per
forward evaluation of the public fixture and preserves the normalization used
to convert power to amplitude spectral density.

### 4. Differentiate the linear equations implicitly

Differentiating $Ax=b$ gives $A \mathrm{d}x = \mathrm{d}b - (\mathrm{d}A) x$. For a scalar loss, reverse mode
can use a transposed solve to propagate the derivative to the inputs. The dense
carrier solver saves its matrix and solution. Given the incoming derivative
$g$, it solves $A^T p=g$ and returns $p$ for the right-hand side and
$-p x^T$ for the matrix, following JAX's complex cotangent convention.

The sparse signal solver implements a custom reverse rule for both the signal
and detector equations. It factorizes each frequency's matrix once again during
the backward pass, shares those factors across the reverse solves, and evaluates
the matrix derivatives only at stored row-column pairs. One reverse pass thus
propagates the scalar loss to all 193 design variables. The finite-difference
spot check in the benchmark runs separately during validation.

### 5. Reduce first-run compilation work

`build._group_by_function` groups component calls by formula and remaps their
outputs to the original matrix slots. `host_components.component_outputs`
evaluates each nonempty group in NumPy across its components and sweep rows.
Its custom reverse rule uses dual numbers to differentiate each formula with
respect to its local real inputs and sums contributions wherever components
share a parameter. JAX calls these routines and the SciPy solvers through host
callbacks, keeping their internal operations out of the graph XLA compiles.

The NumPy version must give the same values as the original complex JAX
formulas. One formula needed care: the transmitted angle uses
$\cos(\arcsin(z))$, and with total internal reflection $\lvert z \rvert > 1$. A real
`arcsin` would return NaN there, so the code uses the complex continuation
$-i \cdot \mathrm{sign}(z) \sqrt{z^2-1}$, which is what JAX computes on complex
inputs, together with the matching derivative.

At the benchmark level (this repository's `benchmark.py`, not the simulator),
`jax.value_and_grad` combines simulation, detector readout and differentiation
into one compiled executable. The reported first-run total includes lowering,
compilation, numerical execution and host transfer. The timings measure these
strategies together; they do not assign a separate speedup to any single
change.

### 6. Works for an unknown optical setup

Nothing above depends on the public setup. The stored matrix positions, the
component groups and the solver layout are all derived from the setup
definition during the build step, so a different topology gets its own
pattern. The NumPy component code covers all 19 component formulas in
`components.py`. Each one was checked against the original complex JAX formula
on random inputs, including total internal reflection: the largest relative
difference was 7e-16 for values and 2e-14 for gradients. It expects
real physical parameter values, as in the challenge inputs.

## Correctness

Every run checks:

- all outputs and gradients are finite,
- sensitivities match the public reference (maximum absolute natural-log error
  at most 1e-5),
- the supplied finite-difference gradient spot check passes.

The spot check is not a full proof for all inputs. The NumPy component code
targets the current component formulas and real physical parameters; new
components or complex parameters would need their derivatives checked again.

No stored physical answers and no persistent compilation cache are used. All
work happens inside the timed evaluation.

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

Several independent processes in a row, with no pause (use `--runs 30` for the
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
- The 30-run results are from an Apple M4, with this repository's
  `benchmark.py`. We have no official single-CPU Xeon result.
- The official `benchmark.py` also runs this simulator unchanged and passes
  (see [Official benchmark check](#official-benchmark-check)).

## Implementation guide

The comparison below uses commit `02ec741` of the challenge starter repository
(`start-hack-munich-2026`) as the baseline. For one design, `benchmark.py`
calls `run_build_step` to prepare the optical system, then evaluates `simulate`,
the detector readout and the loss gradient with respect to all 193 parameters.

### Modules and tools added by the team

| File | Role in the final package |
| --- | --- |
| [differometor/host_components.py](differometor/host_components.py) | Evaluates groups of the original component formulas with NumPy for the carrier, signal and noise matrices. During the reverse pass, dual numbers compute local coefficient derivatives and a custom JAX reverse rule accumulates contributions to shared parameters. Handles total internal reflection like the complex JAX formulas. |
| [differometor/carrier_solve.py](differometor/carrier_solve.py) | Solves the dense steady-state carrier system with SciPy. Its JAX reverse rule solves a transposed system and returns derivatives for both the matrix and its right-hand side. |
| [differometor/sparse_solve.py](differometor/sparse_solve.py) | Active signal solver. At each frequency, `signal_solves` builds a compressed sparse column matrix and shares one SuperLU factorization between the signal and adjoint detector solves. Each solve gets one residual correction. The reverse pass factorizes again and computes matrix derivatives only at stored coordinates. |
| [differometor/sparse_signal.py](differometor/sparse_signal.py) | Alternative implementation using JAX's `custom_linear_solve`. It gathers coefficients from a dense signal system using a structural sparsity pattern. The benchmark calls `sparse_solve.signal_solves` instead. |
| [run_first.py](run_first.py) | Launches a fresh Python process for each measurement, checks its validation report, and writes individual logs, `runs.csv` and `summary.json` under a new `run-logs/first-*` directory. |

### Changes to starter files

| File | What changed |
| --- | --- |
| [differometor/build.py](differometor/build.py) | Groups component calls by function and remaps their output indices. Builds static optical coefficients with NumPy. `to_value_systems` records every matrix position that can receive a signal or noise coefficient, including entries initially equal to zero, and maps subsequent updates to those positions. |
| [differometor/simulate.py](differometor/simulate.py) | Calls the host component evaluator and dense carrier solver, then updates only the stored signal and noise entries across the sweep. For detector selection vector `c`, it solves `Aᴴw = c` alongside the signal system. It computes noise power as `Re(wᴴ Qw)` using a sparse product with the noise matrix `Q`, eliminating the starter's final covariance solve. |
| [differometor/__init__.py](differometor/__init__.py) | Loads the plotting module on first access to `visualize_setup`, reducing imports for simulation-only runs. |
| [benchmark.py](benchmark.py) | The starter compiled forward simulation, readout and reverse differentiation separately. This benchmark compiles them together with `jax.value_and_grad`, measures one first run, and reuses that executable for the finite-difference check. It accepts `--repeats 0` and renders its own phase plot. It is the script used for the 30-run results. It is not part of the simulator; the official starter `benchmark.py` also runs this simulator and passes. |
| [pyproject.toml](pyproject.toml) and [.gitignore](.gitignore) | The package declares SciPy as a direct dependency for the CPU solvers. Git excludes the local virtual environment and generated benchmark outputs such as `run-logs/`. |

`differometor/components.py` keeps the starter's optical formulas. The
component optimization lives in `host_components.py`. The final package also
keeps the starter versions of `setups.py`, `plot.py`, `utils.py`, the pinned
`requirements.txt`, the MIT license and both public JSON fixtures in
`examples/data/`.

## Files and evidence

| File or directory | What it records |
| --- | --- |
| [SOURCE.json](SOURCE.json) | Source branch, commit and SHA-256 hashes for the published simulator and benchmark. |
| [reports/baseline.json](reports/baseline.json) | One example run of the original starter code (`02ec741`), from a separate batch. |
| [reports/final/](reports/final/) | An earlier batch of 10 final runs (mean 1.601 s, median 1.541 s): per-process JSON reports, `runs.csv`, phase and total statistics in CSV, a text summary and two timing plots. Absolute paths in these reports point to the original measurement machine. |

The raw files of the 30-run batch in the Results section are not in this
repository. Earlier experiments remain in Git history and on
`experiment/host-components5`; they are not needed to run this final version.
