from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.components import power_detector, signal_detector


def __getattr__(name):
    # Plotting pulls in Plotly, which simulations never need. Import it only when
    # `differometor.visualize_setup` is first used (PEP 562 module __getattr__).
    if name == "visualize_setup":
        from differometor.plot import visualize_setup
        return visualize_setup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
