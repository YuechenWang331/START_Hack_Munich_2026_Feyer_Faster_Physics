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
