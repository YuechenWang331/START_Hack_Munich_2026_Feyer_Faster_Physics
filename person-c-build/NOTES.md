# Silly Entropy — Faster Physics log

Score = complete first-run total of `benchmark.py --cpus 1` (official Xeon baseline 44.2 s:
imports 2.6 | build 1.0 | forward lower 2.0 + compile 15.0 | forward run 13.3 | sensitivity ~0.4 |
gradient lower 0.4 + compile 5.2 | gradient run 4.3).

## Tools

| Tool | What it checks |
|---|---|
| `tools/check_build.py` | Builds 11 setups (fixture, uifo_20_5000, aLIGO, Voyager, cavity tuning sweep, 6 random UIFOs 2x2–4x4) with the ORIGINAL code (`../reference-differometor`, git worktree at `02ec741`) and the CURRENT code, in separate processes, and compares every array `run_build_step` returns: values, shape, dtype and JAX `weak_type`. |
| `tools/check_outputs.py` | Same idea for simulation outputs: carrier, signal, noise, sensitivity, loss (+ gradient on small setups; `--fixture-gradient` for the fixture on a ≥32 GB machine). Also applies the official `abs(ln S − ln S_ref) ≤ 1e-5` rule to the fixture. |
| `tools/time_setup.py` | Times imports / CPU init / input load / build exactly like `benchmark.py`, without the 14 GB compile+run stages. |

Reference copy: `git worktree add ../reference-differometor 02ec741`.

Finding: the original `build()` iterates a Python `set` of port names when adding vacuum noise
to unused ports, so the ORDER of noise-matrix instructions changes between processes (hash
randomisation). Same entries, so physics is unaffected; checkers pin `PYTHONHASHSEED=0` to
compare strictly.

## Person C — setup phase (imports + build)

### C1. Lazy Plotly import (`differometor/__init__.py`)
- **Why:** `__init__` imported `plot.py`, which imports `plotly.graph_objects`; the benchmark never plots.
- **Change:** `visualize_setup` is now resolved on first access via a module-level `__getattr__`
  (PEP 562). `df.visualize_setup` and `from differometor import visualize_setup` still work.
- **Effect:** `import differometor` 0.078 s → 0.031 s (laptop, after `jax` is already imported).

### C2. No eager JAX in the build step (`differometor/build.py`)
- **Why (measured):** first build of the fixture took ~1.0 s, but a second build in the same
  process only 0.15 s. The difference was ~100 one-off XLA compilations: every eager JAX op
  (`components.space`, `mirror_matrix`'s jnp helpers, `jnp.array(x, dtype=...)`, jnp indexing in
  `prepare_arrays`) makes XLA compile a tiny program the first time it sees a shape/dtype.
- **Change:**
  1. Constant matrix entries at build time (space phase factor, mirror and beamsplitter blocks) are
     computed with NumPy twins `_space_np`, `_mirror_matrix_np`, `_beamsplitter_matrix_np` that
     follow the JAX formulas in `components.py` operation by operation. The jitted simulation still
     uses the JAX originals.
  2. Host → JAX conversion via `_device_array` = `jnp.asarray(np.asarray(x, dtype))`: same dtype and
     values, zero compiles.
  3. `prepare_arrays` indexes/reshapes on the host instead of with eager JAX ops.
- **Correctness:**
  - `check_build.py`: all arrays identical on 11 setups (values, dtype, shape, weak_type).
  - Fuzz test, 300 random inputs incl. non-zero frequency, refractive index ≠ 1, angles 0–60°:
    `space` and `mirror` twins bit-identical 300/300; beamsplitter twin bit-identical 208/300,
    otherwise ≤ 3.5e-15 relative (1–2 ulp, NumPy vs XLA `sin`/`arcsin`). Only for unusual
    angle + index combinations, and only in starting values that the JAX `surface` function
    recomputes whenever that component is optimised or in the signal matrix.
- **Effect:** fixture build step 1.0–1.2 s → ~0.03 s.

### C1 + C2 combined — measured (laptop i5-10300H, Windows, 1 CPU via process affinity)
`tools/time_setup.py`, original vs new, alternating, 3 runs each:

| Phase | Original | New |
|---|---|---|
| Imports | 0.83–0.96 s | 0.75–0.77 s |
| CPU init | 0.008 s | 0.032 s |
| Build | 1.06–1.07 s | 0.08–0.09 s |
| **Setup total** | **1.90–2.08 s** | **0.87–0.89 s** |

≈ 1.05 s saved per first run (the official Xeon baseline spends 1.0 s in build).
Part of the import cost moved into CPU init (+25 ms); the figure above is the net effect.

`tools/check_outputs.py` (original vs new): carrier, signal, noise, sensitivity, loss bit-identical on the
fixture; gradients also bit-identical on aLIGO, Voyager and two random 2x2 UIFOs. Fixture vs public
reference: max |ln S − ln S_ref| = 5.4e-6 (≤ 1e-5). One random UIFO (2x2 seed 0) has a detector with
no signal, so S is non-finite in the ORIGINAL code as well; identical in both.

Not yet run: full `benchmark.py` PASS — needs ~15 GB free RAM, this laptop has ~7 GB free. To run on
Linux/32 GB before submission (`tools/check_outputs.py --fixture-gradient` as well).

## Quan — reverse-mode gradient, Steps 0–3 (analysis of `combined-four`, no code changed)

Base: `combined-four-20261121-184759` from `main` (Quan's build.py/__init__.py + Ha's simulate.py, which
includes Vinh's batched detectors and Tung's removed third solve). Merged into `quan/build-optimization`.
Environments: `Feyer/.venvs/original` (official clone) and `Feyer/.venvs/dev` (combined-four on this branch).
Note: combined-four's `benchmark.py` is Yuechen's modified copy (`--repeats 0`, readout of detector ports only,
inside the timed part). Official scores use the unmodified `benchmark.py`; run it via
`python <official>/benchmark.py --repo <combined-four folder>`.

### Correctness of combined-four vs original (`profiles/check_outputs_original_vs_combined-four.txt`)
- carrier, signal: bit-identical.
- noise / sensitivity: up to 5.9e-8 relative on the fixture (new noise formula, rounding-level);
  fixture vs public reference 5.375e-6 (original 5.374e-6), limit 1e-5. OK.
- gradients, norm-wise ‖Δg‖/‖g‖: aLIGO 7.4e-10, Voyager 2.2e-15, random UIFO 7.1e-16. OK.
  (Element-wise "100 %" errors were near-zero elements only.)

### Profile (laptop, 1 CPU, `tools/profile_phases.py`, 21 and 41 of 101 frequencies)
| Stage | 21 freq | 41 freq | → 101 freq (extrapolated) |
|---|---|---|---|
| Forward lower + compile | 1.88 + 4.46 s | 1.79 + 5.11 s | ≈ 6.6 s (constant) |
| Forward run | 1.13 s | 2.32 s | ≈ 5.9 s (≈0.06 s/freq) |
| Gradient lower + compile | 0.44 + 3.10 s | 0.45 + 3.00 s | ≈ 3.4 s (constant) |
| Gradient run | 0.22 s | 0.37 s | ≈ 0.8 s |
| AD tape / peak memory | 1.34 / 2.26 GiB | 2.62 / 4.05 GiB | ≈ 6.5 / ≈ 9.5 GiB |

Where the time goes (profiler trace, 21 freq):
- **Gradient stage is dominated by compiling (≈3.4 s), not running (≈0.8 s).** The backward program has
  5,736 HLO lines and 1,525 parameters (tape leaves), largely from differentiating `vmap(lax.switch(FUNCTIONS))`
  in `update()` (all 19 branches evaluated for every entry).
- Gradient run: 30 % copying a dense tape array, 27 % triangular solves (necessary), the rest dense
  backward steps of `.at[].set` / scatter / slice-pad (simulate.py:338, 128, 350) and the noise einsum (417).
  The dense outer product of the solve's matvec does not show up as a hotspot.
- Forward compile: 26,009 HLO lines, dominated by multiply/broadcast/select/compare = the 19-branch switch.
- Forward run: LU 32 % (necessary for dense LU), dense copies/slices/scatters/selects ≈ 55 %
  (simulate.py:350 `system = signal_matrix[:, :, :-1]`, 130 update scatter, 348 conjugation `where`, 39 transpose).

## Quan — Change G1: `update()` grouped by function type (in `combined-four-…` on this branch)

**Why:** the profile showed the gradient stage is mostly compile time. `update()` evaluated
`vmap(lax.switch(FUNCTIONS))`: under `vmap`, all 19 component functions run for every matrix entry and
one result is selected. That made the forward program 26,009 HLO lines and the gradient 5,736 lines.

**Change:**
- `build.py` (`_group_by_function`, called from `pairs_to_arrays`): at build time, sort the function calls
  by function type. `*_function_input_indices` becomes a tuple with one `(K_f, 7)` index array per entry of
  `FUNCTIONS` (empty when unused); `*_output_indices` rows are renumbered to the grouped order;
  `*_function_indices` keeps each group's original call numbers (bookkeeping only).
- `simulate.py` `update()`: for each non-empty group, `vmap(FUNCTIONS[f])` over only that group's calls,
  then concatenate. Group sizes are array shapes, so the choice is fixed at trace time. Applies to the
  carrier, signal and noise matrices (Yuechen's `compile-carrier` did the carrier only).
- Generic: groups come from whatever functions the setup uses; nothing fixture-specific.

**Correctness** (`profiles/check_outputs_combined-four_vs_grouped.txt`, vs unchanged combined-four):
carrier, signal, noise, sensitivity, loss bit-identical on all cases; gradients ‖Δg‖/‖g‖ ≤ 4.9e-17
(summation order). Build arrays: only the regrouped/renumbered arrays differ, everything else identical
on 11 setups. Fixture vs public reference 5.375e-6 (limit 1e-5).

**Effect** (laptop, 1 CPU, 21 frequencies, old/new alternating ×3, `profiles/old_vs_grouped_21freq_alternating.txt`):

| Stage | combined-four | grouped | change |
|---|---|---|---|
| Forward lower | 1.86–1.89 s | 1.18–1.20 s | −37 % |
| Forward compile | 14.06–14.64 s | 8.09–8.21 s | −43 % |
| Forward run | 1.81–1.87 s | 1.75–1.79 s | ≈ −4 % |
| Gradient lower | 0.47–0.48 s | 0.24–0.25 s | −49 % |
| **Gradient compile** | **7.00 s** | **3.07–3.13 s** | **−56 %** |
| Gradient run | 0.32 s | 0.30–0.31 s | ≈ −5 % |

HLO size: forward 26,009 → 17,617 lines; gradient 5,736 → 3,877 lines (tape leaves 1,525 → 1,004).
The laptop ran ~3× slower overall during this series than in the earlier profile (same code, mains power,
probably thermal); an earlier single pair in the fast state gave forward compile 4.46 → 2.61 s and gradient
compile 3.10 → 1.15 s, i.e. the same relative gains. Compile times do not depend on the number of
frequencies, so the gains carry over to 101. Full benchmark not run here (≈9 GB peak); to be confirmed on
the M4 with the unmodified `benchmark.py`.

## Quan — Change B: signal and noise systems as non-zero values (Idea B, on top of grouping)

**Why:** the profile showed about half of the forward run and of the gradient run moving dense
(frequencies, 751, 752) arrays that are 99.7 % zeros, and the AD tape keeping several dense copies.

**Change (in `combined-four-…` on this branch):**
- `build.py` `to_value_systems` (called at the end of `run_build_step`): from the setup's own starting matrix
  and update instructions, list every position of the augmented signal matrix that can ever be non-zero
  (starting non-zeros, diagonal, every update position, carrier-scaling positions; RHS = column N, `-1`
  normalised). Update/carrier-scaling instructions become slots in that list; the lower-sideband
  conjugation becomes a per-slot mask (same block rule as before). Same for the noise matrix.
  The structures replace the dense matrices under the same argument names, so `simulate()`'s signature
  and `benchmark.py` are unchanged. Nothing is fixture-specific.
- `simulate.py`: updates, carrier scaling and conjugation act on a (frequencies, nnz) value array.
  A dense matrix is built only for `lu_factor`, under `stop_gradient`. Ha's shared-LU
  `custom_linear_solve` is kept, but its `matvec` is a sparse sum over the values, so JAX differentiates
  one number per non-zero instead of 751×751, and the tape keeps the LU factors instead of dense copies.
  The noise product N·w is a sparse sum too. Vinh's batched detectors and Tung's removed solve are kept.

**Correctness** (vs combined-four from `main`; `profiles/ideab/`):
- carrier and signal **bit-identical** on fixture, aLIGO, Voyager, random UIFOs (so the value-based
  assembly reproduces the dense matrix exactly); noise/sensitivity identical on the fixture, ≤ 1.8e-15 elsewhere.
- gradients ≤ 5.7e-15 norm-wise on the small setups; **full fixture gradient (193 values)** 4.0e-11
  norm-wise (worst element 1.4e-8; ill-conditioned system).
- **Official, unmodified `benchmark.py`: PASS** (sensitivity vs public reference + finite-difference
  gradient check). First run on this laptop, 1 CPU: **10.78 s** — the first version that fits in this
  laptop's RAM under the official benchmark.

**Effect** (laptop, 1 CPU, three versions alternating ×2; `profiles/ideab/summary_staged41_and_benchmark3.txt`):

| | combined-four | grouped | grouped + B |
|---|---|---|---|
| Official structure, 41 freq: forward run | 2.34 / 3.52 s | 3.85 / 3.43 s | **1.71 / 1.63 s** |
| Official structure, 41 freq: gradient run | 0.39 / 0.64 s | 0.57 / 0.56 s | **0.34 / 0.27 s** |
| benchmark3, 101 freq: run | 6.72 / 6.70 s | 6.37 / 6.36 s | **4.15 / 4.20 s** |
| benchmark3, 101 freq: compile | 20.99 / 21.03 s | 10.57 / 10.72 s | 10.93 / 11.21 s |
| benchmark3, 101 freq: total | 30.7 / 30.7 s | 19.2 / 19.3 s | **17.3 / 17.6 s** |

- Idea B: run time −35 % (benchmark3) to −50 % (forward, official structure); compile unchanged.
- **Caveat on compile times on this laptop:** XLA compile times jump between a "fast" and a ~3× slower
  "slow" state for the same code (combined-four forward compile seen at 4.5, 5.6, 14–16 s; grouped at
  2.6 and 7.6–8.9 s). Only same-state pairs are comparable; those show grouping at about −40 %
  (4.46 → 2.61 s fast, 14.3 → 8.1 s slow). Python-side lowering is stable and shows grouping −35–45 %.
  The M4 measurement decides the compile question.
