# Person C — faster setup phase (`build.py` + lazy imports)

Changes the two simulator files below. Everything else in `differometor/` is the original
code (`feyerlabs/start-hack-munich-2026` at commit `02ec741`).

| File | Change |
|---|---|
| `differometor/build.py` | No eager JAX during the build: NumPy versions of the build-time formulas (`_space_np`, `_mirror_matrix_np`, `_beamsplitter_matrix_np`), and host→JAX conversion without XLA compiles (`_device_array`). |
| `differometor/__init__.py` | Plotly is imported only when `visualize_setup` is first used. |

**Result (laptop, 1 CPU):** build step 1.06 s → 0.08 s; imports + init + load + build 1.9 s → 0.9 s.
All build arrays and all simulation outputs and gradients are bit-identical to the original code.
Details and measurements: [`NOTES.md`](NOTES.md).

## Apply to your local copy of the official repo

Option A — patch (in your clone of `feyerlabs/start-hack-munich-2026`, on your own branch):

```bash
git apply /path/to/person-c-build/build-optimization.patch
```

Option B — copy the two files over `differometor/build.py` and `differometor/__init__.py`.

If you have already edited `build.py` or `__init__.py` yourself, use Option A and resolve any
conflict, rather than overwriting your files.

## Check it

The checkers compare the ORIGINAL code against yours. Copy `tools/` into the official clone, then
create an untouched copy of the original next to it:

```bash
git worktree add ../reference-differometor 02ec741
python tools/check_build.py      # every build array: values, dtype, shape, weak_type (11 setups) -> BUILD CHECK PASS
python tools/check_outputs.py    # carrier/signal/noise/sensitivity/loss (+ gradients on small setups) -> OUTPUT CHECK PASS
python tools/time_setup.py       # imports / init / load / build timings, like benchmark.py
python benchmark.py --cpus 1 --repeats 3 --output benchmark-check.json   # needs ~15 GB RAM, Linux/WSL
```

`check_outputs.py --fixture-gradient` also differentiates the full benchmark setup (needs ~15 GB RAM).
