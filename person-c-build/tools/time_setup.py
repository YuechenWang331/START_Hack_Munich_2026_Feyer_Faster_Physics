"""Time the benchmark's setup phases (imports, CPU init, input loading, build) for one repo.

Mirrors the timers in benchmark.py up to and including "build", without the
memory-heavy compile/forward/gradient stages, so it runs on small machines.

    python tools/time_setup.py [REPO]          # REPO defaults to this checkout

On Windows, pin to one CPU like `benchmark.py --cpus 1` does on Linux:
    cmd /c start /affinity 1 /wait /b python tools/time_setup.py
"""
from time import perf_counter

IMPORT_START = perf_counter()
import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
REPO = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent).resolve()
sys.path.insert(0, str(REPO))

import jax
import jax.numpy as jnp
import numpy as np
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces

IMPORT_SECONDS = perf_counter() - IMPORT_START
assert Path(df.__file__).resolve().parent == REPO / "differometor", df.__file__
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


def timed(label, function):
    start = perf_counter()
    result = function()
    jax.block_until_ready(result)
    return result, perf_counter() - start


def initialize():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    return jax.devices("cpu")


_, init_seconds = timed("init", initialize)
data_dir = REPO / "examples" / "data"


def load_inputs():
    fixture_bytes = (data_dir / "uifo_800_3000.json").read_bytes()
    reference_bytes = (data_dir / "uifo_800_3000_test_data.json").read_bytes()
    return json.loads(fixture_bytes), json.loads(reference_bytes)


(fixture, reference), load_seconds = timed("load", load_inputs)


def build_inputs():
    # Same body as benchmark.py's build_inputs().
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    initial = []
    for pair in pairs:
        members = pair if isinstance(pair[0], list) else [pair]
        values = [fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop] for name, prop in members]
        initial.append(values[0])
    parameters = jnp.asarray(initial, dtype=jnp.float64)
    frequencies = jnp.asarray(reference["frequencies"], dtype=jnp.float64)
    expected = jnp.asarray(reference["sensitivities"], dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    arrays["optimized_parameters"] = parameters
    return arrays, jnp.asarray(ports), parameters, frequencies, expected, pairs, jnp.asarray(1.0, dtype=jnp.float64)


_, build_seconds = timed("build", build_inputs)
total = IMPORT_SECONDS + init_seconds + load_seconds + build_seconds
affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "n/a (Windows)"
print(json.dumps({"repo": REPO.name, "imports": round(IMPORT_SECONDS, 4), "initialize": round(init_seconds, 4),
                  "load": round(load_seconds, 4), "build": round(build_seconds, 4), "setup_total": round(total, 4)}))
