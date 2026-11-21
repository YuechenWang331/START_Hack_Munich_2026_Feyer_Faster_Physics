"""Profile where the forward pass and the reverse-mode gradient spend their time.

Runs the same stages as benchmark.py (forward + AD residuals, sensitivity, backward)
for one simulator folder, optionally on a subset of the 101 frequencies so that it
fits in a laptop's memory (for ANALYSIS only, never for the score). Reports:

- lower / compile / run time of every stage and the process' peak memory,
- XLA's static cost estimate (flops, bytes accessed) of the compiled programs,
- the largest operations in the compiled programs, by output size,
- with --trace: measured time per XLA operation, mapped back to the source line
  (e.g. simulate.py:345) that created it.

    python tools/profile_phases.py --repo ../combined-four-20261121-184759 --frequencies 21 --trace
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")


def peak_memory_gib():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return max(counters.PeakWorkingSetSize, counters.PeakPagefileUsage) / 2**30
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


SHAPE = re.compile(r"=\s*\(?([a-z0-9]+)\[([0-9,]*)\]")
OPCODE = re.compile(r"\]\{?[^ ]*\}?\)?\s+([a-z\-]+)\(")
OP_NAME = re.compile(r'op_name="([^"]*)"')
FRAME = re.compile(r"stack_frame_id=(\d+)")
CALLS = re.compile(r"calls=%([\w.\-]+)")
BYTES = {"c128": 16, "f64": 8, "s64": 8, "s32": 4, "pred": 1, "c64": 8, "f32": 4, "u32": 4, "u8": 1}


def _frame_tables(hlo_text):
    """Parse the FileNames / FileLocations / StackFrames tables of an HLO dump."""
    section, files, locations, frames = None, {}, {}, {}
    for line in hlo_text.splitlines():
        stripped = line.strip()
        if stripped in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
            section = stripped
            continue
        if not stripped or stripped.startswith(("%", "ENTRY", "ROOT", "HloModule", "}")):
            if stripped.startswith(("%", "ENTRY")):
                section = None
            continue
        key, _, value = stripped.partition(" ")
        if not key.isdigit():
            continue
        if section == "FileNames":
            files[int(key)] = value.strip('"')
        elif section == "FileLocations":
            fields = dict(re.findall(r"(\w+)=(\d+)", value))
            locations[int(key)] = (int(fields.get("file_name_id", 0)), int(fields.get("line", 0)))
        elif section == "StackFrames":
            fields = dict(re.findall(r"(\w+)=(\d+)", value))
            frames[int(key)] = (int(fields.get("file_location_id", 0)), int(fields.get("parent_frame_id", 0)))
    return files, locations, frames


def _source(frame_id, tables):
    """Innermost frame inside differometor/ (else the innermost frame), as 'file:line'."""
    files, locations, frames = tables
    first, seen = None, set()
    while frame_id in frames and frame_id not in seen:
        seen.add(frame_id)
        location_id, parent = frames[frame_id]
        file_id, line = locations.get(location_id, (0, 0))
        path = files.get(file_id, "?")
        label = f"{Path(path).name}:{line}"
        first = first or label
        if "differometor" in path.replace("\\", "/"):
            return label
        frame_id = parent
    return first or "?"


def instructions(hlo_text):
    """name -> (opcode, output bytes, 'file:line primitive') for every HLO instruction."""
    tables = _frame_tables(hlo_text)
    table, roots, current = {}, {}, None
    raw = []
    for line in hlo_text.splitlines():
        stripped = line.strip()
        if (stripped.startswith("%") or stripped.startswith("ENTRY")) and stripped.endswith("{") and " = " not in stripped:
            current = stripped.split(" ", 1)[0].lstrip("%") if stripped.startswith("%") else "ENTRY"
            continue
        if not stripped.startswith(("%", "ROOT %")) or " = " not in stripped:
            continue
        name = stripped.replace("ROOT ", "").split(" ", 1)[0].lstrip("%")
        shape = SHAPE.search(stripped)
        size = 0
        if shape:
            size = BYTES.get(shape.group(1), 8)
            for d in shape.group(2).split(","):
                if d:
                    size *= int(d)
        opcode = OPCODE.search(stripped)
        op_name = OP_NAME.search(stripped)
        frame = FRAME.search(stripped)
        where = None
        if op_name or frame:
            where = f"{_source(int(frame.group(1)), tables) if frame else '?'} {op_name.group(1).split('/')[-1] if op_name else ''}".strip()
        calls = CALLS.search(stripped)
        raw.append((name, opcode.group(1) if opcode else "?", size, where, calls.group(1) if calls else None))
        if stripped.startswith("ROOT") and current:
            roots[current] = where
    for name, opcode, size, where, calls in raw:
        if where is None and calls:
            where = roots.get(calls)  # fusions: describe them by the root of the fused computation
        table[name] = (opcode, size, where or "?")
    return table


def top_by_size(hlo_text, count=12):
    table = instructions(hlo_text)
    rows = sorted(table.items(), key=lambda kv: -kv[1][1])[:count]
    return [f"    {size / 2**20:9.1f} MiB  {opcode:14s} {where}" for name, (opcode, size, where) in rows]


def trace_breakdown(directory, hlo_text, count=20):
    """Sum measured durations of XLA ops in a profiler trace, grouped by source line."""
    files = glob.glob(os.path.join(directory, "**", "*.trace.json.gz"), recursive=True)
    if not files:
        return ["    (no trace file produced)"]
    with gzip.open(max(files, key=os.path.getmtime), "rt") as handle:
        events = json.load(handle).get("traceEvents", [])
    table = instructions(hlo_text)
    by_where, total, matched = collections.Counter(), 0.0, 0.0
    for event in events:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        name = event.get("name", "")
        key = name.split(":")[0].lstrip("%")
        if key in table:
            opcode, size, where = table[key]
            by_where[f"{where} [{opcode}]"] += event["dur"] / 1e6
            matched += event["dur"] / 1e6
    lines = [f"    matched XLA op time: {matched:.3f} s"]
    for where, seconds in by_where.most_common(count):
        lines.append(f"    {seconds:8.3f} s  {100 * seconds / max(matched, 1e-12):5.1f}%  {where}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--frequencies", type=int, default=101, help="evenly spaced subset of the 101 frequencies")
    parser.add_argument("--trace", action="store_true", help="record and summarise a profiler trace of the runs")
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))

    import numpy as np
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    import differometor as df
    from differometor.setups import constrain_inter_grid_cell_spaces
    assert Path(df.__file__).resolve().parent == repo / "differometor", df.__file__

    data = repo / "examples" / "data"
    fixture = json.loads((data / "uifo_800_3000.json").read_text())
    reference = json.loads((data / "uifo_800_3000_test_data.json").read_text())
    pick = np.unique(np.linspace(0, len(reference["frequencies"]) - 1, args.frequencies).round().astype(int))
    frequencies = jnp.asarray(np.asarray(reference["frequencies"])[pick], dtype=jnp.float64)
    expected = jnp.asarray(np.asarray(reference["sensitivities"])[pick], dtype=jnp.float64)

    properties = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, properties)
    initial = []
    for pair in pairs:
        name, prop = pair[0] if isinstance(pair[0], list) else pair
        initial.append(fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop])
    parameters = jnp.asarray(initial, dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    arrays["optimized_parameters"] = parameters
    ports = jnp.asarray(ports)
    one = jnp.asarray(1.0, dtype=jnp.float64)

    def forward_with_pullback(p, a):
        return jax.vjp(lambda q: df.simulate(**dict(a, optimized_parameters=q)), p)

    def readout(outputs, detector_ports, target):
        carrier, signal, noise = outputs
        signals = df.signal_detector(carrier, signal)[detector_ports]
        detected = signals.squeeze() if detector_ports.shape[0] == 1 else signals[0] - signals[1]
        sensitivity = (noise / jnp.abs(detected)).squeeze()
        return jnp.mean(jnp.log10(sensitivity / target)), (sensitivity, df.power_detector(carrier))

    def sensitivity_with_pullback(outputs, detector_ports, target):
        return jax.vjp(lambda v: readout(v, detector_ports, target), outputs, has_aux=True)

    def backward(simulation_pullback, readout_pullback, cotangent):
        return simulation_pullback(readout_pullback(cotangent)[0])[0]

    timings = {}

    def timed(key, function):
        start = perf_counter()
        result = function()
        jax.block_until_ready(result)
        timings[key] = perf_counter() - start
        return result

    trace_dir = tempfile.mkdtemp(prefix="jaxtrace-")
    forward_lowered = timed("forward_lower", lambda: jax.jit(forward_with_pullback).lower(parameters, arrays))
    forward = timed("forward_compile", forward_lowered.compile)
    if args.trace:
        jax.profiler.start_trace(os.path.join(trace_dir, "forward"))
    outputs, simulation_pullback = timed("forward_run", lambda: forward(parameters, arrays))
    if args.trace:
        jax.profiler.stop_trace()
    sensitivity_fn = jax.jit(sensitivity_with_pullback).lower(outputs, ports, expected).compile()
    loss, readout_pullback, _ = sensitivity_fn(outputs, ports, expected)
    backward_lowered = timed("gradient_lower", lambda: jax.jit(backward).lower(simulation_pullback, readout_pullback, one))
    gradient_fn = timed("gradient_compile", backward_lowered.compile)
    if args.trace:
        jax.profiler.start_trace(os.path.join(trace_dir, "gradient"))
    gradient = timed("gradient_run", lambda: gradient_fn(simulation_pullback, readout_pullback, one))
    if args.trace:
        jax.profiler.stop_trace()
    assert np.isfinite(np.asarray(gradient)).all()

    print(f"repo: {repo.name} | frequencies: {len(pick)} | gradient norm {float(jnp.linalg.norm(gradient)):.6e} | loss {float(loss):.6e}")
    for key, seconds in timings.items():
        print(f"  {key:18s} {seconds:8.3f} s")
    print(f"  peak memory        {peak_memory_gib():8.2f} GiB")
    residual_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(simulation_pullback) if hasattr(x, "nbytes"))
    print(f"  AD residuals (tape) {residual_bytes / 2**30:7.2f} GiB")

    for label, compiled in (("FORWARD", forward), ("GRADIENT", gradient_fn)):
        text = compiled.as_text()
        cost = compiled.cost_analysis()
        cost = cost[0] if isinstance(cost, list) else cost
        opcodes = collections.Counter(v[0] for v in instructions(text).values())
        print(f"\n{label}: {len(text.splitlines())} HLO lines | flops {cost.get('flops', 0):.3e} | bytes accessed {cost.get('bytes accessed', 0) / 2**30:.2f} GiB")
        print("  most frequent ops: " + ", ".join(f"{k} {v}" for k, v in opcodes.most_common(10)))
        print("  largest outputs:")
        print("\n".join(top_by_size(text)))
        if args.trace:
            print("  measured time by source line:")
            print("\n".join(trace_breakdown(os.path.join(trace_dir, label.lower()), text)))


if __name__ == "__main__":
    main()
