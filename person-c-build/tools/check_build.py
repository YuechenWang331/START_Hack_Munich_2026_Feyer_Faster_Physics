"""Compare every array produced by the build step: ORIGINAL simulator vs CURRENT one.

Pass the folder of the ORIGINAL simulator as --reference and the one to check as --current.

Usage:
    python tools/check_build.py --reference ../start-hack-munich-2026 --current ../combined-four-20261121-184759

Each side runs in its own subprocess (the build keeps global state in
LINKING_FUNCTIONS, and each side must import its own `differometor`).
Integer arrays must match exactly; float/complex arrays are reported with max
abs/rel error and must agree to within a few ulps (rtol 1e-14).
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
RTOL = 1e-14


def flatten(prefix, value, out):
    """Store nested tuples/lists/dicts of arrays as flat npz keys."""
    import numpy as np
    if value is None:
        out[prefix] = np.array("None")
    elif isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}.{key}", item, out)
    elif isinstance(value, (tuple, list)) and not (len(value) and isinstance(value[0], (int, float, complex))):
        for index, item in enumerate(value):
            flatten(f"{prefix}[{index}]", item, out)
    else:
        out[prefix] = np.asarray(value)
        # JAX's weak_type flag changes how simulate() is traced, so it must match too.
        out[prefix + ".weak_type"] = np.array(str(getattr(value, "weak_type", "not a jax array")))


def worker(repo, output):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    sys.path.insert(0, str(repo))
    import json
    from time import perf_counter
    import numpy as np
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import differometor as df
    from differometor import setups
    assert Path(df.__file__).resolve().parent == Path(repo).resolve() / "differometor", df.__file__

    data = Path(repo) / "examples" / "data"
    reference = json.loads((data / "uifo_800_3000_test_data.json").read_text())
    benchmark_frequencies = jnp.asarray(reference["frequencies"], dtype=jnp.float64)

    def frequency_node(setup):
        for node, attributes in setup.nodes(data=True):
            if attributes.get("component") == "frequency":
                return node
        return None

    def uifo_case(setup, frequencies):
        pairs = setups.constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
        return setup, [(frequency_node(setup), "frequency")], frequencies, pairs

    def generic_case(setup, frequencies):
        pairs = [list(pair) for pair in setup.parameters if pair[1] in PROPERTIES]
        return setup, [(frequency_node(setup), "frequency")], frequencies, pairs

    def cavity():
        setup = df.Setup()
        setup.add("laser", "l0", power=1)
        setup.add("mirror", "m0", reflectivity=0.99, loss=0)
        setup.add("mirror", "m1", reflectivity=0.991, loss=0)
        setup.space("l0", "m0", length=1)
        setup.space("m0", "m1", length=1)
        setup.add("detector", "refl", target="m0", port="left", direction="out")
        setup.add("detector", "circ", target="m1", port="left", direction="in")
        setup.add("detector", "trns", target="m1", port="right", direction="out")
        return setup, [("m0", "tuning")], jnp.linspace(-180, 180, 400), None

    log_frequencies = jnp.logspace(jnp.log10(20.0), jnp.log10(5000.0), 100)
    cases = {
        "uifo_800_3000": lambda: uifo_case(df.Setup.from_data(json.loads((data / "uifo_800_3000.json").read_text())), benchmark_frequencies),
        "uifo_20_5000": lambda: uifo_case(df.Setup.from_data(json.loads((data / "uifo_20_5000.json").read_text())), log_frequencies),
        "aligo": lambda: generic_case(setups.aligo()[0], log_frequencies),
        "voyager": lambda: generic_case(setups.voyager()[0], log_frequencies),
        "cavity_tuning_sweep": cavity,
    }
    for size, seed in [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0)]:
        cases[f"uifo_random_{size}x{size}_seed{seed}"] = (
            lambda size=size, seed=seed: uifo_case(setups.uifo(size, random=True, random_seed=seed)[0], benchmark_frequencies)
        )

    arrays = {}
    for name, make in cases.items():
        setup, changing_pairs, changing_values, pairs = make()
        start = perf_counter()
        simulation_arrays, *metadata = df.run_build_step(setup, changing_pairs, changing_values, pairs)
        jax.block_until_ready(simulation_arrays)
        seconds = perf_counter() - start
        flatten(name + ".arrays", simulation_arrays, arrays)
        flatten(name + ".metadata", [m for m in metadata if not isinstance(m, dict)], arrays)
        port_to_index = [m for m in metadata if isinstance(m, dict)]
        if port_to_index:
            arrays[name + ".port_to_index"] = np.array(sorted(port_to_index[0].items()), dtype=object).astype(str)
        arrays[name + ".seconds"] = np.array(seconds)
    np.savez(output, **arrays)


def compare(reference_file, current_file, reference_repo=None):
    import numpy as np
    # Load fully into memory so Windows can delete the temporary files afterwards.
    with np.load(reference_file, allow_pickle=False) as file:
        reference = {key: file[key] for key in file.files}
    with np.load(current_file, allow_pickle=False) as file:
        current = {key: file[key] for key in file.files}
    failures = []
    keys = sorted(set(reference) | set(current))
    cases = sorted({key.split(".")[0] for key in keys})
    print(f"{'case':34s} {'arrays':>6s} {'exact':>6s} {'max rel err':>12s} {'build ref':>10s} {'build new':>10s}  status")
    for case in cases:
        case_keys = [k for k in keys if k.split(".")[0] == case and not k.endswith(".seconds")]
        exact, worst, problems = 0, 0.0, []
        for key in case_keys:
            if key not in reference or key not in current:
                problems.append(f"{key}: missing on {'new' if key in reference else 'reference'} side")
                continue
            a, b = reference[key], current[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                problems.append(f"{key}: shape/dtype {a.shape} {a.dtype} -> {b.shape} {b.dtype}")
                continue
            if a.dtype.kind in "biuU" or a.dtype == object:
                if not np.array_equal(a, b):
                    problems.append(f"{key}: integer/str values differ")
                else:
                    exact += 1
                continue
            if np.array_equal(a, b):
                exact += 1
                continue
            scale = np.maximum(np.abs(a), np.abs(b))
            rel = np.abs(a - b) / np.where(scale == 0, 1, scale)
            worst = max(worst, float(rel.max()))
            if rel.max() > RTOL:
                problems.append(f"{key}: max rel error {rel.max():.3e} > {RTOL}")
        status = "OK" if not problems else "FAIL"
        print(f"{case:34s} {len(case_keys):6d} {exact:6d} {worst:12.3e} {float(reference[case + '.seconds']):9.3f}s {float(current[case + '.seconds']):9.3f}s  {status}")
        failures += problems
    for problem in failures:
        print("  -", problem)
    print("BUILD CHECK", "PASS" if not failures else "FAIL")
    return not failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, help="folder with the ORIGINAL (or older) simulator, e.g. ../start-hack-munich-2026")
    parser.add_argument("--current", type=Path, help="folder with the simulator to check, e.g. combined-four-.../")
    parser.add_argument("--worker", nargs=2, metavar=("REPO", "OUTPUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(Path(args.worker[0]), args.worker[1])
        return
    if args.reference is None or args.current is None:
        parser.error("both --reference and --current are required")
    if not (args.reference / "differometor").is_dir():
        parser.error(f"No differometor/ folder in {args.reference}")
    if not (args.current / "differometor").is_dir():
        parser.error(f"No differometor/ folder in {args.current}")
    with tempfile.TemporaryDirectory() as temporary:
        files = []
        for label, repo in (("reference", args.reference), ("current", args.current)):
            output = os.path.join(temporary, label + ".npz")
            # Run from the temp dir so neither repo shadows the other. build() iterates a set of
            # port-name strings, so a fixed hash seed is needed for a deterministic entry order.
            environment = dict(os.environ, PYTHONHASHSEED="0")
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(repo.resolve()), output], check=True, cwd=temporary, env=environment)
            files.append(output)
        sys.exit(0 if compare(*files, reference_repo=args.reference) else 1)


if __name__ == "__main__":
    main()
