"""Compare simulation OUTPUTS (and gradients) of the ORIGINAL vs CURRENT simulator.

Runs the same cases with ../reference-differometor (baseline commit) and with this
repo, each in its own subprocess, then compares carrier, signal, noise,
sensitivity, loss and gradient.

    python tools/check_outputs.py --reference ORIGINAL --current NEW                     # fixture forward + small setups with gradients
    python tools/check_outputs.py --reference ORIGINAL --current NEW --fixture-gradient  # also the fixture gradient (lots of RAM)

The fixture sensitivity is also checked against the public reference with the
official rule abs(ln S - ln S_ref) <= 1e-5.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


def worker(repo, output, fixture_gradient):
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
    frequencies = jnp.asarray(reference["frequencies"], dtype=jnp.float64)

    def frequency_node(setup):
        return next(node for node, a in setup.nodes(data=True) if a.get("component") == "frequency")

    def initial_values(arrays, count):
        # Starting values = the setup's own values, read from the built parameter row.
        row = np.asarray(arrays["parameters"])[0].real
        values = np.zeros(count)
        values[np.asarray(arrays["optimized_value_indices"])] = row[np.asarray(arrays["optimized_parameter_indices"])]
        return jnp.asarray(values, dtype=jnp.float64)

    def readout(outputs, ports, target):
        carrier, signal, noise = outputs
        signals = df.signal_detector(carrier, signal)[ports]
        detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
        sensitivity = (noise / jnp.abs(detected)).squeeze()
        return jnp.mean(jnp.log10(sensitivity / target)), sensitivity

    def uifo_pairs(setup):
        return setups.constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)

    def generic_pairs(setup):
        return [list(p) for p in setup.parameters if p[1] in PROPERTIES]

    fixture = json.loads((data / "uifo_800_3000.json").read_text())
    cases = [
        ("uifo_800_3000", lambda: df.Setup.from_data(fixture), uifo_pairs, fixture_gradient),
        ("aligo", lambda: setups.aligo()[0], generic_pairs, True),
        ("voyager", lambda: setups.voyager()[0], generic_pairs, True),
        ("uifo_random_2x2_seed0", lambda: setups.uifo(2, random=True, random_seed=0)[0], uifo_pairs, True),
        ("uifo_random_2x2_seed1", lambda: setups.uifo(2, random=True, random_seed=1)[0], uifo_pairs, True),
    ]
    results = {}
    for name, make, make_pairs, with_gradient in cases:
        setup = make()
        pairs = make_pairs(setup)
        arrays, ports, *_ = df.run_build_step(setup, [(frequency_node(setup), "frequency")], frequencies, pairs)
        parameters = initial_values(arrays, len(pairs))
        ports = jnp.asarray(ports)
        target = jnp.asarray(reference["sensitivities"]) if name == "uifo_800_3000" else 1.0

        def loss(p):
            return readout(df.simulate(**dict(arrays, optimized_parameters=p)), ports, target)[0]

        start = perf_counter()
        outputs = jax.block_until_ready(jax.jit(lambda p: df.simulate(**dict(arrays, optimized_parameters=p)))(parameters))
        value, sensitivity = jax.jit(lambda o: readout(o, ports, target))(outputs)
        for key, item in zip(("carrier", "signal", "noise"), outputs):
            results[f"{name}.{key}"] = np.asarray(item)
        results[f"{name}.sensitivity"] = np.asarray(sensitivity)
        results[f"{name}.loss"] = np.asarray(value)
        if with_gradient:
            results[f"{name}.gradient"] = np.asarray(jax.jit(jax.grad(loss))(parameters))
        print(f"  [{Path(repo).name}] {name}: {perf_counter() - start:.1f}s{' (with gradient)' if with_gradient else ''}", flush=True)
        jax.clear_caches()
    np.savez(output, **results)


def compare(reference_file, current_file, reference_repo=None):
    import json
    import numpy as np
    with np.load(reference_file) as file:
        reference = {k: file[k] for k in file.files}
    with np.load(current_file) as file:
        current = {k: file[k] for k in file.files}
    ok = True
    print(f"\n{'array':36s} {'shape':>12s} {'identical':>9s} {'max abs err':>12s} {'max rel err':>12s}")
    for key in sorted(set(reference) | set(current)):
        if key not in reference or key not in current:
            print(f"{key:36s} missing on one side"); ok = False
            continue
        a, b = reference[key], current[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            print(f"{key:36s} shape/dtype mismatch {a.shape}{a.dtype} vs {b.shape}{b.dtype}"); ok = False
            continue
        same = np.array_equal(a, b, equal_nan=True)
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        # A setup can be degenerate (e.g. a detector that sees no signal gives S = inf/NaN).
        # That is only acceptable if the ORIGINAL code is non-finite at exactly the same entries.
        same_pattern = np.array_equal(finite_a, finite_b)
        mask = finite_a & finite_b
        diff = np.abs(a - b)[mask] if a.size else np.zeros(0)
        scale = np.maximum(np.abs(a), np.abs(b))[mask] if a.size else np.zeros(0)
        abs_err = float(diff.max()) if diff.size else 0.0
        rel_err = float((diff / np.where(scale == 0, 1, scale)).max()) if diff.size else 0.0
        flag = "" if finite_b.all() else ("  non-finite in BOTH (degenerate setup)" if same_pattern else "  NEW NON-FINITE")
        if key.endswith("gradient") and mask.all() and a.size:
            rel_err_for_pass = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))
        else:
            rel_err_for_pass = rel_err
        ok &= same_pattern and rel_err_for_pass <= (1e-8 if key.endswith("gradient") else 1e-10)
        print(f"{key:36s} {str(a.shape):>12s} {str(same):>9s} {abs_err:12.3e} {rel_err:12.3e}{flag}")
        if key.endswith("gradient") and mask.all() and not same:
            # Element-wise relative error explodes for entries near zero; the norm-wise error does not.
            norm_err = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))
            print(f"{'':36s} norm-wise |g_new - g_ref| / |g_ref| = {norm_err:.3e}   (|g_ref| = {np.linalg.norm(a):.3e})")
            for i in np.argsort(-np.abs(a - b))[:3]:
                print(f"{'':36s}   worst index {i}: ref {a[i]: .6e}  new {b[i]: .6e}")
    public = json.loads((reference_repo / "examples/data/uifo_800_3000_test_data.json").read_text())
    log_error = np.abs(np.log(current["uifo_800_3000.sensitivity"]) - np.log(np.asarray(public["sensitivities"])))
    sensitivity_ok = bool((current["uifo_800_3000.sensitivity"] > 0).all() and log_error.max() <= 1e-5)
    print(f"\nfixture vs public reference: max |ln S - ln S_ref| = {log_error.max():.3e} (limit 1e-5) -> {'OK' if sensitivity_ok else 'FAIL'}")
    ok &= sensitivity_ok
    print("OUTPUT CHECK", "PASS" if ok else "FAIL")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, help="folder with the ORIGINAL (or older) simulator, e.g. ../start-hack-munich-2026")
    parser.add_argument("--current", type=Path, help="folder with the simulator to check, e.g. combined-four-.../")
    parser.add_argument("--fixture-gradient", action="store_true", help="also differentiate the full fixture (~15 GB RAM)")
    parser.add_argument("--worker", nargs=2, metavar=("REPO", "OUTPUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(Path(args.worker[0]), args.worker[1], args.fixture_gradient)
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
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(repo.resolve()), output]
            if args.fixture_gradient:
                command.append("--fixture-gradient")
            subprocess.run(command, check=True, cwd=temporary, env=dict(os.environ, PYTHONHASHSEED="0"))
            files.append(output)
        sys.exit(0 if compare(*files, reference_repo=args.reference) else 1)


if __name__ == "__main__":
    main()
