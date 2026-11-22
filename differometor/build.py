"""
differometor.builder
====================

This module turns a `differometor.setups.Setup` graph into:

1. A *carrier* linear system matrix (complex-valued).
2. A *signal* linear system matrix that contains two sidebands plus explicit signal degrees of freedom.
3. A *noise* matrix and selection vectors for quantum noise readout.
4. A set of *instructions* that describe how to update matrix entries when parameters change.

The central idea is:

- Build sparse update instructions that map from a parameter vector to specific matrix entries.
- Keep indexing conventions consistent across carrier, signal, and noise representations.
- Allow optional parameter linking, where some parameters are computed from others by user-provided functions.

Indexing conventions
--------------------

Carrier system:
- `carrier_matrix` has shape `(system_size, system_size + 1)`.
  The last column is the right-hand side.

Signal system:
- `signal_matrix` has shape `(system_size * 2 + signal_size, system_size * 2 + signal_size + 1)`.
  The last column is the right-hand side.
- The first `system_size` block is the upper sideband.
- The next `system_size` block is the lower sideband.
- The final `signal_size` rows and columns represent explicit signal inputs.

Noise system:
- `noise_matrix` matches `signal_matrix[:, :-1]` shape.
- `noise_selection_vectors` encodes which fields are read out by noise detectors.

The code below is intentionally explicit about shapes and index transforms, because silent off-by-one
or block-shift errors are difficult to debug in linearized optical simulations.
"""

import numpy as np
import jax
import jax.numpy as jnp
from differometor.setups import Setup
from collections import defaultdict
from differometor.components import (
    C_LIGHT,
    F0,
    F,
    DEFAULT_REFRACTIVE_INDEX,
    DEFAULT_PROPERTIES,
    FUNCTIONS,
    signal_function,
    laser_np,
    laser,
    vacuum_quantum_noise,
    squeezer,
    nothing_matrix,
    directional_beamsplitter_matrix,
    beamsplitter_matrix,
    mirror_matrix,
    space_modulation,
    space_lower,
    space_modulation_lower,
    space,
    laser_amplitude_modulation,
    laser_frequency_modulation,
    laser_frequency_modulation_lower,
    susceptibility,
    force_calculation_left,
    force_calculation_right,
    surface,
    loss_quantum_noise,
    corrected_optomechanical_phase_left,
    corrected_optomechanical_phase_right,
    dummy_function,
)

# ---------------------------------------------------------------------
# Global state for parameter linking
# ---------------------------------------------------------------------

LINKING_FUNCTIONS = []
"""
Global registry of parameter-linking functions.

A linking function is a callable that is evaluated externally to compute the current value
of a "linked" parameter. The builder stores indices into this list, so downstream code can
apply linking without needing to serialize callables.

Notes
-----
- This is module-level state. If you build multiple setups in the same process and rely on
  strict reproducibility of indices, you should reset this list before each build.
"""


def _device_array(value, dtype=None) -> jnp.ndarray:
    """
    Convert host data to a JAX array without compiling anything.

    `jnp.array(x, dtype=...)` runs an eager convert op, which makes XLA compile a
    tiny program per dtype/shape (~5 ms each, dozens per build). Converting with
    NumPy first and then handing the buffer to JAX gives the same dtype and values.
    """
    return jnp.asarray(np.asarray(value, dtype=dtype))


def _group_by_function(function_input_indices, function_indices, output_indices):
    """
    Regroup component-function calls by function type, at build time.

    `simulate.update` used to evaluate `lax.switch(FUNCTIONS)` for every call. Under
    `vmap` that evaluates all 19 component functions for every matrix entry, which makes
    the forward and gradient programs (and their compile times) large. Grouping lets
    `update` evaluate each function once, vectorised over only its own calls.

    Parameters
    ----------
    function_input_indices:
        Array `(K, W)`: parameter indices for each of the K function calls.
    function_indices:
        Length-K sequence: index into `FUNCTIONS` for each call.
    output_indices:
        Array `(2, M)`: (call row, output column) of each value written to the matrix.

    Returns
    -------
    grouped_inputs:
        Tuple with one `(K_f, W)` index array per entry of `FUNCTIONS` (empty if unused).
    grouped_rows:
        Tuple with the original call numbers of each group (bookkeeping/checks only).
    output_indices:
        Copy of `output_indices` with call rows renumbered to the grouped order
        (groups concatenated in `FUNCTIONS` order).
    """
    function_input_indices = np.asarray(function_input_indices, dtype=np.int64)
    function_indices = np.asarray(function_indices, dtype=np.int64).reshape(-1)
    output_indices = np.asarray(output_indices).astype(np.int64)
    assert len(function_indices) == len(function_input_indices), "one function per call row expected"
    assert output_indices[0].max(initial=0) < len(function_indices), "output row outside the function calls"

    rows_per_function = [np.flatnonzero(function_indices == f) for f in range(len(FUNCTIONS))]
    new_position = np.empty(len(function_indices), dtype=np.int64)
    new_position[np.concatenate(rows_per_function)] = np.arange(len(function_indices))
    remapped = output_indices.copy()
    remapped[0] = new_position[output_indices[0]]

    width = function_input_indices.shape[1]
    grouped_inputs = tuple(_device_array(function_input_indices[rows].reshape(-1, width), dtype=np.int64) for rows in rows_per_function)
    grouped_rows = tuple(_device_array(rows, dtype=np.int64) for rows in rows_per_function)
    return grouped_inputs, grouped_rows, remapped


# ---------------------------------------------------------------------
# Signal and noise systems as lists of non-zero values (Idea B)
# ---------------------------------------------------------------------
# The signal matrix is ~99.7 % zeros. Working on a dense (frequencies, N, N + 1) stack
# makes the forward pass move mostly zeros through memory and makes the AD tape keep
# several dense copies. Instead, `simulate` keeps one complex value per position that
# can ever be non-zero. The positions are derived here from the starting matrix and
# the update instructions of THIS setup, so nothing depends on a particular fixture.


def _normalised_positions(indices, shape):
    """Index pair (rows, cols) -> non-negative positions, like NumPy/JAX indexing (-1 = last)."""
    rows = np.asarray(indices[0], dtype=np.int64).reshape(-1)
    cols = np.asarray(indices[1], dtype=np.int64).reshape(-1)
    assert np.all((rows >= -shape[0]) & (rows < shape[0])), "row index outside the matrix"
    assert np.all((cols >= -shape[1]) & (cols < shape[1])), "column index outside the matrix"
    return rows % shape[0], cols % shape[1]


def _value_layout(matrix, *index_pairs, with_diagonal=False):
    """
    Sorted (row, col) of every position that is non-zero in `matrix` or written by any
    of `index_pairs`, plus a function mapping (rows, cols) to slots in that list.
    """
    matrix = np.asarray(matrix)
    nonzero_rows, nonzero_cols = np.nonzero(matrix)
    rows = [nonzero_rows]
    cols = [nonzero_cols]
    if with_diagonal:
        rows.append(np.arange(min(matrix.shape)))
        cols.append(np.arange(min(matrix.shape)))
    for pair in index_pairs:
        pair_rows, pair_cols = _normalised_positions(pair, matrix.shape)
        rows.append(pair_rows)
        cols.append(pair_cols)
    width = matrix.shape[1]
    keys = np.unique(np.concatenate(rows) * width + np.concatenate(cols))
    layout_rows, layout_cols = np.divmod(keys, width)

    def slots(pair):
        pair_rows, pair_cols = _normalised_positions(pair, matrix.shape)
        return np.searchsorted(keys, pair_rows * width + pair_cols)

    return layout_rows, layout_cols, slots


def to_value_systems(simulation_arrays: dict, carrier_size: int) -> dict:
    """
    Replace the dense signal and noise matrices of `run_build_step`'s output by their
    non-zero values (same argument names, so `simulate`'s signature is unchanged).

    signal_matrix  -> (base_values, conjugate, system_slots, system_rows, system_columns,
                       rhs_slots, rhs_rows)
        base_values      (nnz,)  starting value at every possible non-zero position of the
                                 augmented matrix (N, N + 1); column N is the right-hand side
        conjugate        (nnz,)  True where the lower-sideband conjugation applies
        system_slots     slots with column < N, and their (row, column) in the N x N system
        rhs_slots        slots in the right-hand-side column, and their rows
    signal_matrix_indices           -> (slots,) where each update output is written
    signal_carrier_placing_indices  -> (slots,) of the entries scaled by the carrier
    noise_matrix   -> (base_values, rows, columns)
    noise_system_matrix_indices     -> (slots,)
    """
    arrays = dict(simulation_arrays)

    signal_matrix = np.asarray(arrays["signal_matrix"])
    system_size = signal_matrix.shape[0]
    rows, cols, slots = _value_layout(
        signal_matrix,
        np.asarray(arrays["signal_matrix_indices"]),
        np.asarray(arrays["signal_carrier_placing_indices"]),
        with_diagonal=True,
    )
    # Same rule as the dense conjugation in simulate(): lower-sideband rows from column n on,
    # and the signal rows over the upper-sideband columns (column N, the RHS, counts as >= n).
    lower = (rows >= carrier_size) & (rows < 2 * carrier_size)
    conjugate = (lower & (cols >= carrier_size)) | ((rows >= 2 * carrier_size) & (cols < carrier_size))
    in_system = cols < system_size
    arrays["signal_matrix"] = (
        _device_array(signal_matrix[rows, cols], dtype=np.complex128),
        _device_array(conjugate, dtype=bool),
        _device_array(np.flatnonzero(in_system), dtype=np.int64),
        _device_array(rows[in_system], dtype=np.int64),
        _device_array(cols[in_system], dtype=np.int64),
        _device_array(np.flatnonzero(~in_system), dtype=np.int64),
        _device_array(rows[~in_system], dtype=np.int64),
    )
    arrays["signal_matrix_indices"] = _device_array(slots(np.asarray(arrays["signal_matrix_indices"])), dtype=np.int64)
    arrays["signal_carrier_placing_indices"] = _device_array(
        slots(np.asarray(arrays["signal_carrier_placing_indices"])), dtype=np.int64
    )

    noise_matrix = np.asarray(arrays["noise_matrix"])
    noise_rows, noise_cols, noise_slots = _value_layout(noise_matrix, np.asarray(arrays["noise_system_matrix_indices"]))
    arrays["noise_matrix"] = (
        _device_array(noise_matrix[noise_rows, noise_cols], dtype=np.complex128),
        _device_array(noise_rows, dtype=np.int64),
        _device_array(noise_cols, dtype=np.int64),
    )
    arrays["noise_system_matrix_indices"] = _device_array(
        noise_slots(np.asarray(arrays["noise_system_matrix_indices"])), dtype=np.int64
    )
    return arrays


def _reshape_to_row(values):
    """`values.reshape(1, -1)` on the host when that cannot change the result's JAX type."""
    if isinstance(values, jax.Array) and not values.weak_type:
        return _device_array(np.asarray(values).reshape(1, -1))
    return values.reshape(1, -1)


# ---------------------------------------------------------------------
# NumPy twins of component formulas used for constant build-time entries
# ---------------------------------------------------------------------
# `build` only needs plain numbers here. Calling the JAX versions eagerly makes JAX
# compile one tiny XLA program per primitive (~60 compiles, ~0.9 s on first build).
# These mirror `components.space`, `components.mirror_matrix` and
# `components.beamsplitter_matrix` operation by operation; the jitted simulation
# still uses the JAX originals.


def _space_np(frequency, length, refractive_index):
    """NumPy version of `components.space(...)[0]`."""
    frequency, length, refractive_index = np.array([frequency, length, refractive_index], dtype=np.float64)
    return -np.exp(-1j * 2 * np.pi * frequency * length * refractive_index / C_LIGHT)


def _surface_coefficients_np(loss, reflectivity, tuning, frequency, refractive_index_left, refractive_index_right, alpha=0.0):
    """NumPy version of components.reflectivity_left / transmissivity / reflectivity_right."""

    def phase_shift(refractive_index):
        return 2 * np.radians(tuning) * refractive_index * (1 + frequency / F0)

    def incidence_angle_right(index_in, index_out):
        # Like the JAX original, total internal reflection gives NaN silently.
        with np.errstate(invalid="ignore"):
            return np.cos(np.arcsin(index_in / index_out * np.sin(np.radians(alpha))))

    incidence_angle_left = np.cos(np.radians(alpha))
    absolute_reflectivity = (1 - loss) * reflectivity
    absolute_transmissivity = (1 - loss) * (1 - reflectivity)

    reflectivity_left = -np.sqrt(absolute_reflectivity) * np.exp(
        1j * phase_shift(refractive_index_left) * incidence_angle_left
    )
    reflectivity_right = -np.sqrt(absolute_reflectivity) * np.exp(
        -1j
        * phase_shift(refractive_index_right)
        * incidence_angle_right(refractive_index_right, refractive_index_left)
    )
    left_term = phase_shift(refractive_index_left) * incidence_angle_left
    right_term = phase_shift(refractive_index_right) * incidence_angle_right(refractive_index_left, refractive_index_right)
    transmissivity = -np.sqrt(absolute_transmissivity) * np.exp(1j * (np.pi / 2 + 0.5 * (left_term - right_term)))
    return reflectivity_left, transmissivity, reflectivity_right


def _mirror_matrix_np(loss, reflectivity, tuning, frequency, refractive_index_left, refractive_index_right):
    """NumPy version of `components.mirror_matrix` (same 4x4 layout)."""
    r_left, t, r_right = _surface_coefficients_np(
        loss, reflectivity, tuning, frequency, refractive_index_left, refractive_index_right
    )
    return np.array(
        [
            [1, 0, 0, 0],
            [r_left, 1, t, 0],
            [0, 0, 1, 0],
            [t, 0, r_right, 1],
        ]
    )


def _beamsplitter_matrix_np(loss, reflectivity, tuning, frequency, refractive_index_left, refractive_index_right, alpha):
    """NumPy version of `components.beamsplitter_matrix` (same 8x8 layout)."""
    r_left, t, r_right = _surface_coefficients_np(
        loss, reflectivity, tuning, frequency, refractive_index_left, refractive_index_right, alpha
    )
    return np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, r_left, 0, t, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [r_left, 0, 0, 1, 0, 0, t, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [t, 0, 0, 0, 0, 1, r_right, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, t, 0, r_right, 0, 0, 1],
        ]
    )


def set_instructions(
    instructions: dict,
    node: str,
    function_input_indices: np.ndarray,
    output_column_indices: np.ndarray,
    system_matrix_row_indices: np.ndarray,
    system_matrix_column_indices: np.ndarray,
    function_indices: list,
    output_row_indices: np.ndarray | None = None,
    carrier_indices: np.ndarray | None = None,
    carrier_row_indices: np.ndarray | None = None,
    carrier_column_indices: np.ndarray | None = None,
    system_size_for_sidebands: int | None = None,
) -> None:
    """
    Store standardized update instructions for one logical component.

    These instructions tell the solver how to:
    - Read the relevant parameter indices (`function_input_indices`),
    - Call one or more functions (`function_indices`) from `differometor.components.FUNCTIONS`,
    - Select outputs (`output_row_indices`, `output_column_indices`),
    - Place the resulting values into the system (or noise) matrix at
      (`system_matrix_row_indices`, `system_matrix_column_indices`).

    The function optionally expands indices for a two-sideband signal system by duplicating
    appropriate entries when `system_size_for_sidebands` is provided.

    Parameters
    ----------
    instructions:
        Dictionary that is updated in-place. Keys are node identifiers.
    node:
        Key under which the instruction set is stored.
    function_input_indices:
        Integer array with shape `(N, K)` where each row lists parameter indices used
        by one function call. This function standardizes `K` to 7 by padding with zeros.
    output_column_indices:
        Integer array of length `M`. Each entry selects a column of the called function output.
    system_matrix_row_indices:
        Integer array of length `M`. Target row indices in the destination matrix.
    system_matrix_column_indices:
        Integer array of length `M`. Target column indices in the destination matrix.
    function_indices:
        List of integer indices into `FUNCTIONS`, one per function call.
    output_row_indices:
        Optional integer array that selects output rows when multiple function calls are stacked.
        If omitted, downstream code typically assumes a single-row output per component.
    carrier_indices, carrier_row_indices, carrier_column_indices:
        Optional arrays that specify which entries from the carrier solution are needed to compute
        signal updates, and where those carrier values must be multiplied in.
    system_size_for_sidebands:
        If provided, expand indices to account for two sidebands in the signal/noise matrices.
        Entries in the carrier-sized block (indices `< system_size_for_sidebands`) may be duplicated
        and shifted to represent the second sideband.

    Raises
    ------
    AssertionError
        If shapes and index arrays are inconsistent.
    """
    function_input_indices_shape = 7

    # Standardize function input length to 7 by padding with placeholder zeros.
    if function_input_indices.shape[1] < function_input_indices_shape:
        pad_width = function_input_indices_shape - function_input_indices.shape[1]
        function_input_indices = np.concatenate(
            [function_input_indices, np.zeros((function_input_indices.shape[0], pad_width))],
            axis=1,
        )

    assert (
        function_input_indices.shape[1] == function_input_indices_shape
    ), f"Function input indices have to be standardized to length {function_input_indices_shape}."
    assert len(output_column_indices) == len(
        system_matrix_row_indices
    ), "Output column indices have to match the length of system matrix row indices."

    # Copy carrier indexing arrays to avoid accidental mutations by sideband expansion.
    if carrier_indices is not None:
        carrier_indices = carrier_indices.copy()
        carrier_row_indices = carrier_row_indices.copy()
        carrier_column_indices = carrier_column_indices.copy()

    if system_size_for_sidebands is not None:
        # The signal and noise matrices contain two carrier-sized blocks:
        # - Upper sideband: indices [0, system_size)
        # - Lower sideband: indices [system_size, 2 * system_size)
        #
        # Plus additional signal degrees of freedom appended afterward.
        #
        # Any entry that references the carrier-sized portion must be duplicated and shifted
        # appropriately to also exist in the second sideband block. Entries that already live
        # in the appended signal block must be shifted because the carrier block is duplicated.

        only_row_mask = (system_matrix_row_indices < system_size_for_sidebands) & (
            system_matrix_column_indices >= system_size_for_sidebands
        )
        only_column_mask = (system_matrix_column_indices < system_size_for_sidebands) & (
            system_matrix_row_indices >= system_size_for_sidebands
        )
        row_and_column_mask = (system_matrix_row_indices < system_size_for_sidebands) & (
            system_matrix_column_indices < system_size_for_sidebands
        )
        pure_signal_mask = (system_matrix_row_indices >= system_size_for_sidebands) & (
            system_matrix_column_indices >= system_size_for_sidebands
        )

        # Shift indices for entries that belong to the appended signal block.
        system_matrix_column_indices[pure_signal_mask] += system_size_for_sidebands
        system_matrix_row_indices[pure_signal_mask] += system_size_for_sidebands

        # Shift the side that references the appended signal block in mixed entries.
        system_matrix_column_indices[only_row_mask] += system_size_for_sidebands
        system_matrix_row_indices[only_column_mask] += system_size_for_sidebands

        new_system_matrix_row_indices = system_matrix_row_indices.copy()
        new_system_matrix_column_indices = system_matrix_column_indices.copy()
        new_output_column_indices = output_column_indices.copy()
        new_output_row_indices = output_row_indices.copy() if output_row_indices is not None else None

        # Duplicate entries for the second sideband where needed.
        if only_row_mask.any():
            new_system_matrix_row_indices = np.concatenate(
                [new_system_matrix_row_indices, system_matrix_row_indices[only_row_mask] + system_size_for_sidebands]
            )
            new_system_matrix_column_indices = np.concatenate(
                [new_system_matrix_column_indices, system_matrix_column_indices[only_row_mask]]
            )
            new_output_column_indices = np.concatenate(
                [new_output_column_indices, output_column_indices[only_row_mask]]
            )
            if output_row_indices is not None:
                new_output_row_indices = np.concatenate([new_output_row_indices, output_row_indices[only_row_mask]])

        if only_column_mask.any():
            new_system_matrix_column_indices = np.concatenate(
                [
                    new_system_matrix_column_indices,
                    system_matrix_column_indices[only_column_mask] + system_size_for_sidebands,
                ]
            )
            new_system_matrix_row_indices = np.concatenate(
                [new_system_matrix_row_indices, system_matrix_row_indices[only_column_mask]]
            )
            new_output_column_indices = np.concatenate(
                [new_output_column_indices, output_column_indices[only_column_mask]]
            )
            if output_row_indices is not None:
                new_output_row_indices = np.concatenate([new_output_row_indices, output_row_indices[only_column_mask]])

        if row_and_column_mask.any():
            new_system_matrix_column_indices = np.concatenate(
                [
                    new_system_matrix_column_indices,
                    system_matrix_column_indices[row_and_column_mask] + system_size_for_sidebands,
                ]
            )
            new_system_matrix_row_indices = np.concatenate(
                [
                    new_system_matrix_row_indices,
                    system_matrix_row_indices[row_and_column_mask] + system_size_for_sidebands,
                ]
            )
            new_output_column_indices = np.concatenate(
                [new_output_column_indices, output_column_indices[row_and_column_mask]]
            )
            if output_row_indices is not None:
                new_output_row_indices = np.concatenate([new_output_row_indices, output_row_indices[row_and_column_mask]])

        output_column_indices = new_output_column_indices
        output_row_indices = new_output_row_indices
        system_matrix_row_indices = new_system_matrix_row_indices
        system_matrix_column_indices = new_system_matrix_column_indices

        # Apply the same duplication and shifting logic to carrier-index instructions.
        if carrier_indices is not None:
            only_row_mask = (carrier_row_indices < system_size_for_sidebands) & (
                carrier_column_indices >= system_size_for_sidebands
            )
            only_column_mask = (carrier_column_indices < system_size_for_sidebands) & (
                carrier_row_indices >= system_size_for_sidebands
            )
            row_and_column_mask = (carrier_row_indices < system_size_for_sidebands) & (
                carrier_column_indices < system_size_for_sidebands
            )
            pure_signal_mask = (carrier_row_indices >= system_size_for_sidebands) & (
                carrier_column_indices >= system_size_for_sidebands
            )

            carrier_column_indices[pure_signal_mask] += system_size_for_sidebands
            carrier_row_indices[pure_signal_mask] += system_size_for_sidebands
            carrier_column_indices[only_row_mask] += system_size_for_sidebands
            carrier_row_indices[only_column_mask] += system_size_for_sidebands

            new_carrier_row_indices = carrier_row_indices.copy()
            new_carrier_column_indices = carrier_column_indices.copy()
            new_carrier_indices = carrier_indices.copy()

            if only_row_mask.any():
                new_carrier_row_indices = np.concatenate(
                    [new_carrier_row_indices, carrier_row_indices[only_row_mask] + system_size_for_sidebands]
                )
                new_carrier_column_indices = np.concatenate(
                    [new_carrier_column_indices, carrier_column_indices[only_row_mask]]
                )
                new_carrier_indices = np.concatenate([new_carrier_indices, carrier_indices[only_row_mask]])

            if only_column_mask.any():
                new_carrier_column_indices = np.concatenate(
                    [
                        new_carrier_column_indices,
                        carrier_column_indices[only_column_mask] + system_size_for_sidebands,
                    ]
                )
                new_carrier_row_indices = np.concatenate(
                    [new_carrier_row_indices, carrier_row_indices[only_column_mask]]
                )
                new_carrier_indices = np.concatenate([new_carrier_indices, carrier_indices[only_column_mask]])

            if row_and_column_mask.any():
                new_carrier_column_indices = np.concatenate(
                    [
                        new_carrier_column_indices,
                        carrier_column_indices[row_and_column_mask] + system_size_for_sidebands,
                    ]
                )
                new_carrier_row_indices = np.concatenate(
                    [
                        new_carrier_row_indices,
                        carrier_row_indices[row_and_column_mask] + system_size_for_sidebands,
                    ]
                )
                new_carrier_indices = np.concatenate([new_carrier_indices, carrier_indices[row_and_column_mask]])

            carrier_indices = new_carrier_indices
            carrier_row_indices = new_carrier_row_indices
            carrier_column_indices = new_carrier_column_indices

    assert (
        function_input_indices.shape[1] == function_input_indices_shape
    ), f"Function input indices have to be standardized to length {function_input_indices_shape}."
    assert len(output_column_indices) == len(
        system_matrix_row_indices
    ), "Output column indices have to match the length of system matrix row indices."

    instructions[node] = {
        "function_input_indices": function_input_indices,
        "output_column_indices": output_column_indices,
        "system_matrix_row_indices": system_matrix_row_indices,
        "system_matrix_column_indices": system_matrix_column_indices,
        "function_indices": function_indices,
    }
    if output_row_indices is not None:
        instructions[node]["output_row_indices"] = output_row_indices

    # Carrier-solution coupling information for signal updates.
    if carrier_indices is not None:
        instructions[node]["carrier_indices"] = carrier_indices
        instructions[node]["carrier_row_indices"] = carrier_row_indices
        instructions[node]["carrier_column_indices"] = carrier_column_indices


def parameter_linking(
    new_parameters: list,
    indices_to_link: list,
    linked_names: list,
    linking_function_indices: list,
    parameters: list,
) -> None:
    """
    Append parameters to the parameter list while supporting "linked" parameters.

    A "linked" parameter is represented by a tuple `(name, function)` where:
    - `name` is stored in `linked_names` and later converted to an index in `parameter_names`.
    - `function` is appended to `LINKING_FUNCTIONS` and referenced by an integer index.

    Parameters
    ----------
    new_parameters:
        List whose items are either:
        - A numeric value to append directly, or
        - A tuple `(str, callable)` describing a linked parameter.
    indices_to_link:
        List updated in-place with indices in `parameters` that correspond to linked parameters.
    linked_names:
        List updated in-place with the names of linked parameters.
    linking_function_indices:
        List updated in-place with integer indices into `LINKING_FUNCTIONS`.
    parameters:
        Parameter list updated in-place. Linked parameters are stored as placeholder zeros.

    Raises
    ------
    AssertionError
        If a linked parameter tuple is malformed.
    """
    for new_parameter in new_parameters:
        if type(new_parameter) is not tuple:
            parameters.append(new_parameter)
        else:
            assert type(new_parameter[0]) is str, "Linked parameter name has to be a string."
            assert callable(new_parameter[1]), "Linked parameter value has to be callable."

            # Store a placeholder; the real value is computed via the linking function later.
            parameters.append(0)

            # Register the linking function and store the index mapping.
            LINKING_FUNCTIONS.append(new_parameter[1])
            linking_function_indices.append(len(LINKING_FUNCTIONS) - 1)
            indices_to_link.append(len(parameters) - 1)
            linked_names.append(new_parameter[0])


def build(setup: Setup):
    """
    Build carrier, signal, and noise matrices plus update instructions from a setup graph.

    Parameters
    ----------
    setup:
        A `Setup` graph that contains:
        - Nodes with at least a `"component"` key.
        - Edges representing spaces (propagation paths) with optional `"properties"` dict.

    Returns
    -------
    instructions_tuple:
        `(carrier_instructions, signal_instructions, noise_instructions, signal_components, parameter_names)`
        where each instruction dictionary maps a component key to a standardized instruction set.
    matrices_tuple:
        `(parameters, carrier_matrix, signal_matrix, noise_matrix, noise_selection_vectors,
          qhd_parameter_indices, qhd_placing_indices, linked_indices, linking_function_indices, indices_to_link)`
        as `jax.numpy` arrays.
    metadata_tuple:
        `(detector_indices, mirror_indices, beamsplitter_indices, isolator_indices, port_to_index)`
        where the first four entries are integer arrays and `port_to_index` maps string keys of the form
        `"component.port.direction"` to carrier indices.

    Notes
    -----
    - `parameters` is shaped `(1, N)` to match later vectorized update logic.
    - If no explicit signal exists, a dummy signal degree of freedom is created to avoid conditional
      branches in later JAX computations.

    Raises
    ------
    ValueError
        If invalid graph connections are detected (for example detector as edge source).
    """
    system_size = 0
    signal_size = 0
    matrix_positions = {}
    carrier_instructions = {}
    signal_instructions = {}

    # Parameter vector starts with:
    # - Carrier frequency at index 0
    # - Default refractive index at index 1
    # - Default alpha value of 0 at index 2
    parameters = [F, DEFAULT_REFRACTIVE_INDEX, 0]
    parameter_names = ["", "", ""]

    signal_components = []
    space_to_signals = defaultdict(list)
    laser_to_signals = defaultdict(list)
    parameter_positions = {}
    noise_instructions = {}
    all_ports = set([])
    used_ports = set([])
    quantum_detector_number = 0
    free_masses = []
    signal_frequency_position = 0
    surfaces_to_refractive_index_parameter_position = defaultdict(dict)
    port_to_index = {}
    mirror_indices = []
    beamsplitter_indices = []
    isolator_indices = []
    indices_to_link = []
    linked_names = []
    linking_function_indices = []

    def load_defaults(data: dict, component: str | None = None) -> None:
        """
        Ensure `data["properties"]` exists and includes defaults for missing keys.

        Parameters
        ----------
        data:
            Node or edge attribute dictionary to update in-place.
        component:
            Component name to select defaults from. If not provided, `data["component"]` is used.
        """
        if component is None:
            component = data["component"]
        if "properties" not in data:
            data["properties"] = DEFAULT_PROPERTIES[component].copy()
        else:
            default_dict = DEFAULT_PROPERTIES[component].copy()
            default_dict.update(data["properties"])
            data["properties"] = default_dict

    # -----------------------------------------------------------------
    # First edge loop: determine system size contributions from spaces
    # and register space parameters early (needed by surfaces).
    # -----------------------------------------------------------------
    for (source, target, data) in setup.edges(data=True):
        load_defaults(data, "space")

        # Normalize missing port configuration for edges.
        if "source_port" not in data:
            data["source_port"] = "right"
        if "target_port" not in data:
            data["target_port"] = "left"

        source_node = setup.nodes[source]
        target_node = setup.nodes[target]

        # Add space parameters (length and refractive index).
        parameter_positions[f"{source}_{target}"] = len(parameters)
        parameter_linking(
            [data["properties"]["length"], data["properties"]["refractive_index"]],
            indices_to_link,
            linked_names,
            linking_function_indices,
            parameters,
        )
        parameter_names += [f"{source}_{target}_length", f"{source}_{target}_refractive_index"]

        # Track refractive index parameter positions at surfaces.
        if source_node["component"] == "mirror":
            surfaces_to_refractive_index_parameter_position[source][data["source_port"]] = len(parameters) - 1
        if target_node["component"] == "mirror":
            surfaces_to_refractive_index_parameter_position[target][data["target_port"]] = len(parameters) - 1

        # Beamsplitter refractive index constraints:
        # left and top share one index, right and bottom share the other.
        port_mapping = {"right": "right", "left": "left", "top": "left", "bottom": "right"}

        if source_node["component"] == "beamsplitter":
            port = port_mapping[data["source_port"]]
            if (
                port in surfaces_to_refractive_index_parameter_position[source]
                and parameters[surfaces_to_refractive_index_parameter_position[source][port]] != parameters[-1]
            ):
                raise ValueError(
                    "Beamsplitter has to have the same refractive index on left and top and right and bottom respectively."
                )
            surfaces_to_refractive_index_parameter_position[source][port] = len(parameters) - 1

        if target_node["component"] == "beamsplitter":
            port = port_mapping[data["target_port"]]
            if (
                port in surfaces_to_refractive_index_parameter_position[target]
                and parameters[surfaces_to_refractive_index_parameter_position[target][port]] != parameters[-1]
            ):
                raise ValueError(
                    "Beamsplitter has to have the same refractive index on left and top and right and bottom respectively."
                )
            surfaces_to_refractive_index_parameter_position[target][port] = len(parameters) - 1

        # Space-connected lasers and detectors create explicit "space endpoint" nodes in the matrix.
        if source_node["component"] in ["laser", "squeezer"]:
            matrix_positions[f"{source}_{target}_source"] = system_size
            system_size += 2  # each space endpoint contributes an input and output index
            all_ports.add(f"{source}_{target}.left")
            source_node["target"] = f"{source}_{target}_source"
        elif source_node["component"] in ["detector", "qnoised"]:
            raise ValueError("Detectors can only be targets of edges.")

        if target_node["component"] in ["detector", "qnoised"]:
            matrix_positions[f"{source}_{target}_target"] = system_size
            system_size += 2
            all_ports.add(f"{source}_{target}.right")
            target_node["target"] = f"{source}_{target}_target"
            target_node["direction"] = "out"
        elif target_node["component"] in ["laser", "squeezer"]:
            raise ValueError("Lasers can only be sources of edges.")

    # -----------------------------------------------------------------
    # First node loop: place submatrices and count signal degrees of freedom.
    # -----------------------------------------------------------------
    for node, data in setup.nodes(data=True):
        load_defaults(data)

        # Place matrix components on the diagonal.
        if data["component"] in MATRIX_SIZES:
            matrix_size = MATRIX_SIZES[data["component"]]
            matrix_positions[node] = system_size

            if data["component"] == "mirror":
                surface_indices = mirror_indices
            elif data["component"] == "beamsplitter":
                surface_indices = beamsplitter_indices
            elif data["component"] == "directional_beamsplitter":
                surface_indices = isolator_indices
            else:
                surface_indices = []

            surface_indices.extend(range(system_size, system_size + matrix_size))
            system_size += matrix_size

            # Track all potential ports to later mark unused ports as vacuum-noise inputs.
            if data["component"] in ["mirror"]:
                all_ports.add(node + ".left")
                all_ports.add(node + ".right")
            if data["component"] in ["beamsplitter"]:
                all_ports.add(node + ".top")
                all_ports.add(node + ".bottom")
                all_ports.add(node + ".left")
                all_ports.add(node + ".right")

        # Normalize defaults for sources and sinks.
        if data["component"] in ["laser", "squeezer", "detector", "qnoised"]:
            if "port" not in data:
                data["port"] = "left"
            if "direction" not in data:
                data["direction"] = "in"

        # Squeezers are treated as signal components because their parameters are varied in the signal run.
        if data["component"] == "squeezer":
            signal_components.append(node)

        # Quantum detector counting for noise readout vectors.
        if data["component"] in ["qnoised", "qhd"]:
            if ("auxiliary" not in data) or ("auxiliary" in data and not data["auxiliary"]):
                quantum_detector_number += 1

        # Explicit signal inputs (for example modulation sources).
        if data["component"] == "signal":
            signal_components.append(node)
            matrix_positions[node] = signal_size
            try:
                target_component = setup.nodes[data["target"]]["component"]
                if target_component == "laser":
                    laser_to_signals[data["target"]].append(node)
            except KeyError:
                # Target is a space (spaces are edges, not nodes).
                space_to_signals[data["target"]].append(node)
            signal_size += 1

        # Free mass introduces two signal degrees of freedom (force and displacement).
        if data["component"] == "free_mass":
            signal_components.append(node)
            matrix_positions[node] = signal_size
            signal_size += 2
            free_masses.append(node)

        # Signal frequency parameter (used for sideband computations).
        if data["component"] == "frequency":
            signal_components.append(node)
            signal_frequency_position = len(parameters)
            parameters.append(data["properties"]["frequency"])
            parameter_names.append(f"{node}_frequency")

    detectors = {}

    # Carrier matrix: identity on the left block, with a separate right-hand-side column.
    carrier_matrix = np.zeros((system_size, system_size + 1), dtype=complex)
    carrier_matrix[:, :system_size] = np.eye(system_size, dtype=complex)

    # Ensure signal_size is at least 1 to avoid conditional logic later.
    if signal_size == 0:
        signal_size = 1

    # Signal matrix includes two carrier-sized blocks plus appended signal block, plus a right-hand-side column.
    signal_matrix = np.zeros((system_size * 2 + signal_size, system_size * 2 + signal_size + 1), dtype=complex)
    signal_matrix[:, : system_size * 2 + signal_size] = np.eye(system_size * 2 + signal_size, dtype=complex)

    # Noise matrix has the same shape as the signal matrix without the right-hand side.
    noise_matrix = np.zeros(signal_matrix[:, :-1].shape)
    noise_detectors = {}

    # Selection vectors map fields to detector readouts for quantum noise calculations.
    noise_selection_vectors = np.zeros((quantum_detector_number, 1, system_size * 2 + signal_size), dtype=complex)
    noise_detector_count = 0

    detector_indices = []

    # Quantum homodyne detection (QHD) needs per-detector phase rotation information.
    qhd_parameter_indices = []
    qhd_placing_indices = []

    # -----------------------------------------------------------------
    # Second node loop: resolve laser and detector indices (depends on target placement).
    # Also fill matrices and build update instructions.
    # -----------------------------------------------------------------
    for node, data in setup.nodes(data=True):
        # -----------------------------
        # Signals
        # -----------------------------
        if data["component"] == "signal":
            signal_index = matrix_positions[node]

            # Add signal amplitude and phase parameters.
            parameter_linking(
                [data["properties"]["amplitude"], data["properties"]["phase"]],
                indices_to_link,
                linked_names,
                linking_function_indices,
                parameters,
            )
            parameter_names += [f"{node}_amplitude", f"{node}_phase"]

            # Signal acts as a right-hand-side injection into the explicit signal block.
            set_instructions(
                signal_instructions,
                node,
                function_input_indices=np.array([[len(parameters) - 2, len(parameters) - 1]]),
                output_column_indices=np.array([0]),
                system_matrix_row_indices=np.array([system_size * 2 + signal_index]),
                system_matrix_column_indices=np.array([-1]),
                function_indices=[FUNCTIONS.index(signal_function)],
            )

        # -----------------------------
        # Lasers, detectors, squeezers
        # -----------------------------
        if data["component"] in ["laser", "detector", "qnoised", "squeezer"]:
            # Locate the target's base index in the system matrix.
            try:
                target_index = matrix_positions[data["target"]]
            except KeyError:
                # The target is a synthetic space endpoint created in the edge loop.
                suffix = "_source" if data["port"] == "left" else "_target"
                target_index = matrix_positions[data["target"] + suffix]

            # Compute port offset within the target submatrix.
            try:
                port_offset = PORT_DICTS[setup.nodes[data["target"]]["component"]][data["port"]]
            except KeyError:
                # Space endpoints have a single implicit port.
                port_offset = 0

            if data["direction"] not in ["in", "out"]:
                raise ValueError("Direction has to be either in or out.")

            # Determine if target is a diagonal matrix component (mirror, beamsplitter, etc.).
            target_is_matrix_component = False
            try:
                target_is_matrix_component = setup.nodes[data["target"]]["component"] in MATRIX_SIZES
            except KeyError:
                pass

            # Direction ordering differs for space endpoints versus explicit components.
            if not target_is_matrix_component:
                direction_offset = 1 if data["direction"] == "in" else 0
            else:
                direction_offset = 0 if data["direction"] == "in" else 1

            component_index = target_index + port_offset + direction_offset
            matrix_positions[node] = component_index

            # Ordinary detectors.
            if data["component"] == "detector":
                sideband = data.get("sideband", "upper")
                if sideband == "lower":
                    component_index += system_size
                detectors[node] = component_index
                detector_indices.append(component_index)

            # Quantum-noise detectors contribute to selection vectors.
            if data["component"] == "qnoised":
                if ("auxiliary" not in data) or ("auxiliary" in data and not data["auxiliary"]):
                    noise_detectors[node] = noise_detector_count
                    noise_selection_vectors[noise_detector_count, 0, component_index] = np.sqrt(2)
                    noise_selection_vectors[noise_detector_count, 0, component_index + system_size] = np.sqrt(2)
                    noise_detector_count += 1

            # Lasers inject a carrier field into the right-hand side and also emit vacuum noise.
            if data["component"] == "laser":
                target_port = data["target"].replace("_source", "").replace("_target", "") + "." + data["port"]
                used_ports.add(target_port)

                carrier_matrix[component_index, system_size] = laser_np(**data["properties"])

                parameter_linking(
                    [data["properties"]["power"], data["properties"]["phase"]],
                    indices_to_link,
                    linked_names,
                    linking_function_indices,
                    parameters,
                )
                parameter_names += [f"{node}_power", f"{node}_phase"]

                set_instructions(
                    carrier_instructions,
                    node,
                    function_input_indices=np.array([[len(parameters) - 2, len(parameters) - 1]]),
                    output_column_indices=np.array([0]),
                    system_matrix_row_indices=np.array([component_index]),
                    system_matrix_column_indices=np.array([-1]),
                    function_indices=[FUNCTIONS.index(laser)],
                )

                set_instructions(
                    noise_instructions,
                    node,
                    function_input_indices=np.array([[0]]),
                    output_column_indices=np.array([0]),
                    system_matrix_row_indices=np.array([component_index]),
                    system_matrix_column_indices=np.array([component_index]),
                    function_indices=[FUNCTIONS.index(vacuum_quantum_noise)],
                    system_size_for_sidebands=system_size,
                )

            # Squeezers transform vacuum noise entering the attached port.
            if data["component"] == "squeezer":
                target_port = data["target"].replace("_source", "").replace("_target", "") + "." + data["port"]
                used_ports.add(target_port)

                parameter_linking(
                    [data["properties"]["db"], data["properties"]["angle"]],
                    indices_to_link,
                    linked_names,
                    linking_function_indices,
                    parameters,
                )
                parameter_names += [f"{node}_db", f"{node}_angle"]

                set_instructions(
                    noise_instructions,
                    node,
                    function_input_indices=np.array([[len(parameters) - 2, len(parameters) - 1]]),
                    output_column_indices=np.array([0, 1, 2, 3]),
                    system_matrix_row_indices=np.array(
                        [component_index, component_index, component_index + system_size, component_index + system_size]
                    ),
                    system_matrix_column_indices=np.array(
                        [component_index, component_index + system_size, component_index + system_size, component_index]
                    ),
                    function_indices=[FUNCTIONS.index(squeezer)],
                )

        # -----------------------------
        # Quantum homodyne detection (QHD)
        # -----------------------------
        if data["component"] == "qhd":
            parameter_linking(
                [data["properties"]["phase"]],
                indices_to_link,
                linked_names,
                linking_function_indices,
                parameters,
            )
            parameter_names += [f"{node}_phase"]

            detector1_index = matrix_positions[data["detector1"]]
            detector2_index = matrix_positions[data["detector2"]]

            qhd_parameter_indices.extend([len(parameters) - 1, len(parameters) - 1])
            qhd_placing_indices.extend(
                [[noise_detector_count, 0, detector2_index], [noise_detector_count, 0, detector2_index + system_size]]
            )

            noise_selection_vectors[noise_detector_count, 0, detector1_index] = np.sqrt(2)
            noise_selection_vectors[noise_detector_count, 0, detector2_index] = np.sqrt(2)
            noise_selection_vectors[noise_detector_count, 0, detector1_index + system_size] = np.sqrt(2)
            noise_selection_vectors[noise_detector_count, 0, detector2_index + system_size] = np.sqrt(2)
            noise_detector_count += 1

        # -----------------------------
        # Passive matrix components
        # -----------------------------
        if data["component"] == "nothing":
            nothing_index = matrix_positions[node]
            matrix = nothing_matrix()
            carrier_matrix[nothing_index : nothing_index + matrix.shape[0], nothing_index : nothing_index + matrix.shape[1]] = matrix

        if data["component"] == "directional_beamsplitter":
            directional_beamsplitter_index = matrix_positions[node]
            matrix = directional_beamsplitter_matrix()
            carrier_matrix[
                directional_beamsplitter_index : directional_beamsplitter_index + matrix.shape[0],
                directional_beamsplitter_index : directional_beamsplitter_index + matrix.shape[1],
            ] = matrix

        # -----------------------------
        # Mirrors and beamsplitters
        # -----------------------------
        if data["component"] in ["mirror", "beamsplitter"]:
            component_index = matrix_positions[node]
            parameter_positions[node] = len(parameters)

            if data["component"] == "mirror":
                refractive_index_left_position = surfaces_to_refractive_index_parameter_position[node].get("left", 1)
                refractive_index_right_position = surfaces_to_refractive_index_parameter_position[node].get("right", 1)

                matrix = _mirror_matrix_np(
                    data["properties"]["loss"],
                    data["properties"]["reflectivity"],
                    data["properties"]["tuning"],
                    F,
                    parameters[refractive_index_left_position],
                    parameters[refractive_index_right_position],
                )

                parameter_linking(
                    [data["properties"]["loss"], data["properties"]["reflectivity"], data["properties"]["tuning"]],
                    indices_to_link,
                    linked_names,
                    linking_function_indices,
                    parameters,
                )
                parameter_names += [f"{node}_loss", f"{node}_reflectivity", f"{node}_tuning"]
                loss_position = len(parameters) - 3

                function_input_indices = np.array(
                    [[len(parameters) - 3, len(parameters) - 2, len(parameters) - 1, 0, refractive_index_left_position, refractive_index_right_position, 2]]
                )
                signal_function_input_indices = np.array(
                    [
                        [
                            len(parameters) - 3,
                            len(parameters) - 2,
                            len(parameters) - 1,
                            signal_frequency_position,
                            refractive_index_left_position,
                            refractive_index_right_position,
                            2,
                        ]
                    ]
                )

                output_column_indices = np.array([0, 1, 1, 2])
                system_matrix_row_indices = component_index + np.array([1, 1, 3, 3])
                system_matrix_column_indices = component_index + np.array([0, 2, 0, 2])

                noise_output_column_indices = np.array([0, 0])
                noise_system_matrix_indices = component_index + np.array([1, 3])

            elif data["component"] == "beamsplitter":
                refractive_index_left_position = surfaces_to_refractive_index_parameter_position[node].get("left", 1)
                refractive_index_right_position = surfaces_to_refractive_index_parameter_position[node].get("right", 1)

                matrix = _beamsplitter_matrix_np(
                    data["properties"]["loss"],
                    data["properties"]["reflectivity"],
                    data["properties"]["tuning"],
                    F,
                    parameters[refractive_index_left_position],
                    parameters[refractive_index_right_position],
                    data["properties"]["alpha"],
                )

                parameter_linking(
                    [
                        data["properties"]["loss"],
                        data["properties"]["reflectivity"],
                        data["properties"]["tuning"],
                        data["properties"]["alpha"],
                    ],
                    indices_to_link,
                    linked_names,
                    linking_function_indices,
                    parameters,
                )
                parameter_names += [f"{node}_loss", f"{node}_reflectivity", f"{node}_tuning", f"{node}_alpha"]
                loss_position = len(parameters) - 4

                function_input_indices = np.array(
                    [
                        [
                            len(parameters) - 4,
                            len(parameters) - 3,
                            len(parameters) - 2,
                            0,
                            refractive_index_left_position,
                            refractive_index_right_position,
                            len(parameters) - 1,
                        ]
                    ]
                )
                signal_function_input_indices = np.array(
                    [
                        [
                            len(parameters) - 4,
                            len(parameters) - 3,
                            len(parameters) - 2,
                            signal_frequency_position,
                            refractive_index_left_position,
                            refractive_index_right_position,
                            len(parameters) - 1,
                        ]
                    ]
                )

                output_column_indices = np.array([0, 1, 0, 1, 1, 2, 1, 2])
                system_matrix_row_indices = component_index + np.array([1, 1, 3, 3, 5, 5, 7, 7])
                system_matrix_column_indices = component_index + np.array([2, 4, 0, 6, 0, 6, 2, 4])

                noise_output_column_indices = np.array([0, 0, 0, 0])
                noise_system_matrix_indices = component_index + np.array([1, 3, 5, 7])

            # Initialize the carrier matrix diagonal block for the surface.
            carrier_matrix[
                component_index : component_index + matrix.shape[0],
                component_index : component_index + matrix.shape[1],
            ] = matrix

            # Carrier update instructions for the surface.
            set_instructions(
                carrier_instructions,
                node,
                function_input_indices=function_input_indices,
                output_column_indices=output_column_indices,
                system_matrix_row_indices=system_matrix_row_indices,
                system_matrix_column_indices=system_matrix_column_indices,
                function_indices=[FUNCTIONS.index(surface)],
            )

            # Signal update instructions: same surface function but sideband-aware frequency.
            set_instructions(
                signal_instructions,
                node,
                function_input_indices=signal_function_input_indices,
                output_column_indices=output_column_indices,
                system_matrix_row_indices=system_matrix_row_indices,
                system_matrix_column_indices=system_matrix_column_indices,
                function_indices=[FUNCTIONS.index(surface)],
                system_size_for_sidebands=system_size,
            )

            # Loss-induced vacuum noise entries for the noise matrix.
            set_instructions(
                noise_instructions,
                node,
                function_input_indices=np.array([[loss_position]]),
                output_column_indices=noise_output_column_indices,
                system_matrix_row_indices=noise_system_matrix_indices,
                system_matrix_column_indices=noise_system_matrix_indices,
                function_indices=[FUNCTIONS.index(loss_quantum_noise)],
                system_size_for_sidebands=system_size,
            )

    # -----------------------------------------------------------------
    # Apply laser field modulations (needs final indices of both lasers and signals).
    # -----------------------------------------------------------------
    for node in laser_to_signals:
        laser_index = matrix_positions[node]

        signal_matrix_indices = []
        function_indices = []

        for signal in laser_to_signals[node]:
            signal_index = matrix_positions[signal]
            signal_matrix_indices += [system_size * 2 + signal_index] * 2

            target_property = setup.nodes[signal]["target_property"]
            if target_property == "amplitude":
                function_indices.extend([FUNCTIONS.index(laser_amplitude_modulation), FUNCTIONS.index(laser_amplitude_modulation)])
            elif target_property == "frequency":
                function_indices.extend([FUNCTIONS.index(laser_frequency_modulation), FUNCTIONS.index(laser_frequency_modulation_lower)])
            else:
                raise ValueError("Target property for laser modulation has to be either amplitude or frequency.")

        laser_signal_number = len(laser_to_signals[node])

        system_matrix_row_indices = np.array([laser_index, laser_index + system_size] * laser_signal_number)
        system_matrix_column_indices = np.array(signal_matrix_indices)

        set_instructions(
            signal_instructions,
            node,
            function_input_indices=np.tile(
                np.array([[signal_frequency_position], [signal_frequency_position]]),
                (laser_signal_number, 1),
            ),
            output_column_indices=np.array([0, 0] * laser_signal_number),
            output_row_indices=np.arange(laser_signal_number * 2),
            system_matrix_row_indices=system_matrix_row_indices,
            system_matrix_column_indices=system_matrix_column_indices,
            function_indices=function_indices,
            carrier_indices=np.array([laser_index, laser_index] * laser_signal_number),
            carrier_row_indices=system_matrix_row_indices,
            carrier_column_indices=system_matrix_column_indices,
        )

    # -----------------------------------------------------------------
    # Free mass coupling entries (needs surface parameters and indices).
    # -----------------------------------------------------------------
    for node in free_masses:
        data = setup.nodes[node]
        try:
            component_index = matrix_positions[data["target"]]
            parameter_position = parameter_positions[data["target"]]
            target_component = setup.nodes[data["target"]]["component"]
        except KeyError:
            raise ValueError("Free mass has to be connected to either a mirror or a beamsplitter.")

        if target_component == "mirror":
            free_mass_index = matrix_positions[node] + system_size

            parameter_linking(
                [data["properties"]["mass"]],
                indices_to_link,
                linked_names,
                linking_function_indices,
                parameters,
            )
            parameter_names += [f"{node}_mass"]

            system_matrix_row_indices = np.array(
                [
                    free_mass_index + 1,
                    free_mass_index,
                    free_mass_index,
                    free_mass_index,
                    free_mass_index,
                    component_index + 1,
                    component_index + 3,
                ]
            )
            system_matrix_column_indices = np.array(
                [
                    free_mass_index,
                    component_index,
                    component_index + 1,
                    component_index + 2,
                    component_index + 3,
                    free_mass_index + 1,
                    free_mass_index + 1,
                ]
            )

            refractive_index_left_position = surfaces_to_refractive_index_parameter_position[data["target"]].get("left", 1)
            refractive_index_right_position = surfaces_to_refractive_index_parameter_position[data["target"]].get("right", 1)

            set_instructions(
                signal_instructions,
                node,
                function_input_indices=np.array(
                    [
                        [signal_frequency_position, len(parameters) - 1, 0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 0, 0, 0],
                        [2, refractive_index_left_position, refractive_index_right_position, 0, 0, 0, 0],
                        [signal_frequency_position, parameter_position + 2, parameter_position + 1, parameter_position, refractive_index_left_position, 2, 0],
                        [signal_frequency_position, parameter_position + 2, parameter_position + 1, parameter_position, refractive_index_left_position, refractive_index_right_position, 2],
                    ]
                ),
                output_row_indices=np.array([0, 1, 1, 2, 2, 3, 4]),
                output_column_indices=np.array([0] * 7),
                system_matrix_row_indices=system_matrix_row_indices,
                system_matrix_column_indices=system_matrix_column_indices,
                function_indices=[
                    FUNCTIONS.index(susceptibility),
                    FUNCTIONS.index(force_calculation_left),
                    FUNCTIONS.index(force_calculation_right),
                    FUNCTIONS.index(corrected_optomechanical_phase_left),
                    FUNCTIONS.index(corrected_optomechanical_phase_right),
                ],
                carrier_indices=np.array(
                    [
                        component_index,
                        component_index + 1,
                        component_index + 2,
                        component_index + 3,
                        component_index,
                        component_index + 2,
                    ]
                ),
                carrier_row_indices=system_matrix_row_indices[1:],
                carrier_column_indices=system_matrix_column_indices[1:],
                system_size_for_sidebands=system_size,
            )

        elif target_component == "beamsplitter":
            free_mass_index = matrix_positions[node] + system_size

            parameter_linking(
                [data["properties"]["mass"]],
                indices_to_link,
                linked_names,
                linking_function_indices,
                parameters,
            )
            parameter_names += [f"{node}_mass"]

            system_matrix_row_indices = np.array(
                [
                    free_mass_index + 1,
                    free_mass_index,
                    free_mass_index,
                    free_mass_index,
                    free_mass_index,
                    free_mass_index,
                    free_mass_index,
                    free_mass_index,
                    free_mass_index,
                    component_index + 1,
                    component_index + 3,
                    component_index + 5,
                    component_index + 7,
                ]
            )
            system_matrix_column_indices = np.array(
                [
                    free_mass_index,
                    component_index,
                    component_index + 1,
                    component_index + 2,
                    component_index + 3,
                    component_index + 4,
                    component_index + 5,
                    component_index + 6,
                    component_index + 7,
                    free_mass_index + 1,
                    free_mass_index + 1,
                    free_mass_index + 1,
                    free_mass_index + 1,
                ]
            )

            refractive_index_left_position = surfaces_to_refractive_index_parameter_position[data["target"]].get("left", 1)
            refractive_index_right_position = surfaces_to_refractive_index_parameter_position[data["target"]].get("right", 1)

            set_instructions(
                signal_instructions,
                node,
                function_input_indices=np.array(
                    [
                        [signal_frequency_position, len(parameters) - 1, 0, 0, 0, 0, 0],
                        [parameter_position + 3, 0, 0, 0, 0, 0, 0],
                        [parameter_position + 3, refractive_index_left_position, refractive_index_right_position, 0, 0, 0, 0],
                        [signal_frequency_position, parameter_position + 2, parameter_position + 1, parameter_position, refractive_index_left_position, parameter_position + 3, 0],
                        [signal_frequency_position, parameter_position + 2, parameter_position + 1, parameter_position, refractive_index_left_position, refractive_index_right_position, parameter_position + 3],
                    ]
                ),
                output_row_indices=np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4]),
                output_column_indices=np.array([0] * 13),
                system_matrix_row_indices=system_matrix_row_indices,
                system_matrix_column_indices=system_matrix_column_indices,
                function_indices=[
                    FUNCTIONS.index(susceptibility),
                    FUNCTIONS.index(force_calculation_left),
                    FUNCTIONS.index(force_calculation_right),
                    FUNCTIONS.index(corrected_optomechanical_phase_left),
                    FUNCTIONS.index(corrected_optomechanical_phase_right),
                ],
                carrier_indices=np.array(
                    [
                        component_index,
                        component_index + 1,
                        component_index + 2,
                        component_index + 3,
                        component_index + 4,
                        component_index + 5,
                        component_index + 6,
                        component_index + 7,
                        component_index + 2,
                        component_index,
                        component_index + 6,
                        component_index + 4,
                    ]
                ),
                carrier_row_indices=system_matrix_row_indices[1:],
                carrier_column_indices=system_matrix_column_indices[1:],
                system_size_for_sidebands=system_size,
            )

    # -----------------------------------------------------------------
    # Edge loop: connect components through spaces and add space instructions.
    # -----------------------------------------------------------------
    for (source, target, data) in setup.edges(data=True):
        source_node = setup.nodes[source]
        target_node = setup.nodes[target]

        if source_node["component"] in MATRIX_SIZES:
            used_ports.add(source + "." + data["source_port"])
        if target_node["component"] in MATRIX_SIZES:
            used_ports.add(target + "." + data["target_port"])

        source_index = matrix_positions[source]
        target_index = matrix_positions[target]

        # Determine input and output indices for source and target ports.
        try:
            source_port_index = PORT_DICTS[setup.nodes[source]["component"]][data["source_port"]]
            source_input_index = source_index + source_port_index
            source_output_index = source_index + source_port_index + 1
        except KeyError:
            # Laser is represented at an output index when connected through a space endpoint.
            source_input_index = source_index - 1
            source_output_index = source_index

        try:
            target_port_index = PORT_DICTS[setup.nodes[target]["component"]][data["target_port"]]
            target_input_index = target_index + target_port_index
            target_output_index = target_index + target_port_index + 1
        except KeyError:
            # Detector is represented at an output index when connected through a space endpoint.
            target_input_index = target_index + 1
            target_output_index = target_index

        # Initialize the carrier matrix with the space propagation entry.
        space_entry = _space_np(F, data["properties"]["length"], data["properties"]["refractive_index"])
        carrier_matrix[source_input_index, target_output_index] = space_entry
        carrier_matrix[target_input_index, source_output_index] = space_entry

        parameter_position = parameter_positions[f"{source}_{target}"]

        set_instructions(
            carrier_instructions,
            f"{source}_{target}",
            function_input_indices=np.array([[0, parameter_position, parameter_position + 1]]),
            output_column_indices=np.array([0, 0]),
            system_matrix_row_indices=np.array([source_input_index, target_input_index]),
            system_matrix_column_indices=np.array([target_output_index, source_output_index]),
            function_indices=[FUNCTIONS.index(space)],
        )

        # Signal run: update both sidebands. Lower sideband uses `space_lower`.
        set_instructions(
            signal_instructions,
            f"{source}_{target}",
            function_input_indices=np.array(
                [
                    [signal_frequency_position, parameter_position, parameter_position + 1],
                    [signal_frequency_position, parameter_position, parameter_position + 1],
                ]
            ),
            output_row_indices=np.array([0, 0, 1, 1]),
            output_column_indices=np.array([0, 0, 0, 0]),
            system_matrix_row_indices=np.array(
                [
                    source_input_index,
                    target_input_index,
                    source_input_index + system_size,
                    target_input_index + system_size,
                ]
            ),
            system_matrix_column_indices=np.array(
                [
                    target_output_index,
                    source_output_index,
                    target_output_index + system_size,
                    source_output_index + system_size,
                ]
            ),
            function_indices=[FUNCTIONS.index(space), FUNCTIONS.index(space_lower)],
        )

        # Space strain signals inject into the signal connector columns.
        if f"{source}_{target}" in space_to_signals:
            signal_matrix_indices = []
            for signal in space_to_signals[f"{source}_{target}"]:
                signal_index = matrix_positions[signal]
                signal_matrix_indices += [system_size * 2 + signal_index] * 4

            space_signal_number = len(space_to_signals[f"{source}_{target}"])

            system_matrix_row_indices = np.array(
                [source_input_index, target_input_index, source_input_index + system_size, target_input_index + system_size]
                * space_signal_number
            )
            system_matrix_column_indices = np.array(signal_matrix_indices)

            set_instructions(
                signal_instructions,
                f"{source}_{target}_modulation",
                function_input_indices=np.tile(
                    np.array(
                        [
                            [signal_frequency_position, parameter_position, parameter_position + 1],
                            [signal_frequency_position, parameter_position, parameter_position + 1],
                        ]
                    ),
                    (space_signal_number, 1),
                ),
                output_column_indices=np.array([0, 0, 0, 0] * space_signal_number),
                output_row_indices=np.repeat(np.arange(space_signal_number * 2), 2),
                system_matrix_row_indices=system_matrix_row_indices,
                system_matrix_column_indices=system_matrix_column_indices,
                function_indices=[FUNCTIONS.index(space_modulation), FUNCTIONS.index(space_modulation_lower)] * space_signal_number,
                carrier_indices=np.array([source_input_index, target_input_index, source_input_index, target_input_index] * space_signal_number),
                carrier_row_indices=system_matrix_row_indices,
                carrier_column_indices=system_matrix_column_indices,
            )

    # Insert carrier matrix into both sideband blocks of the signal matrix.
    signal_matrix[:system_size, :system_size] = carrier_matrix[:, :system_size]
    signal_matrix[system_size:-signal_size, system_size:-signal_size - 1] = carrier_matrix[:, :system_size]

    # -----------------------------------------------------------------
    # Mark unused ports and inject vacuum noise at those ports.
    # -----------------------------------------------------------------
    for node_port in all_ports - used_ports:
        node, port = node_port.split(".")
        try:
            port_offset = PORT_DICTS[setup.nodes[node]["component"]][port]
        except KeyError:
            port_offset = 0

        node_index = matrix_positions[node]
        port_index = node_index + port_offset

        set_instructions(
            noise_instructions,
            node_port,
            function_input_indices=np.array([[0]]),
            output_column_indices=np.array([0]),
            system_matrix_row_indices=np.array([port_index]),
            system_matrix_column_indices=np.array([port_index]),
            function_indices=[FUNCTIONS.index(vacuum_quantum_noise)],
            system_size_for_sidebands=system_size,
        )

    # -----------------------------------------------------------------
    # Build port-to-index lookup for carrier ports.
    # -----------------------------------------------------------------
    for node, data in setup.nodes(data=True):
        component = data.get("component")
        if component not in PORT_DICTS:
            continue

        node_index = matrix_positions[node]
        for port_name, port_offset in PORT_DICTS[component].items():
            port_to_index[f"{node}.{port_name}.in"] = node_index + port_offset
            port_to_index[f"{node}.{port_name}.out"] = node_index + port_offset + 1

    # Convert linked parameter names to parameter index positions.
    linked_indices = [parameter_names.index(name) for name in linked_names]
    if len(linked_indices) == 0:
        linked_indices = [0]
        indices_to_link = [0]
        LINKING_FUNCTIONS.append(lambda x: x)
        linking_function_indices = [0]

    return (
        (carrier_instructions, signal_instructions, noise_instructions, signal_components, parameter_names),
        (
            _device_array([parameters], dtype=complex),
            _device_array(carrier_matrix),
            _device_array(signal_matrix),
            _device_array(noise_matrix, dtype=complex),
            _device_array(noise_selection_vectors, dtype=complex),
            _device_array(qhd_parameter_indices, dtype=int),
            _device_array(qhd_placing_indices, dtype=int),
            _device_array(linked_indices, dtype=int),
            _device_array(linking_function_indices, dtype=int),
            _device_array(indices_to_link, dtype=int),
        ),
        (
            _device_array(detector_indices, dtype=int),
            _device_array(mirror_indices, dtype=int),
            _device_array(beamsplitter_indices, dtype=int),
            _device_array(isolator_indices, dtype=int),
            port_to_index,
        ),
    )


def pairs_to_arrays(
    parameter_instructions: dict,
    signal_instructions: dict,
    noise_instructions: dict,
    signal_components: list,
    parameter_names: list,
    optimization_pairs: list | None = None,
    changing_pairs: list | None = None,
):
    """
    Convert instruction dictionaries into dense index arrays for vectorized JAX execution.

    This function gathers update instructions from:
    - Carrier updates (`parameter_instructions`)
    - Signal updates (`signal_instructions`)
    - Noise updates (`noise_instructions`)

    It then compiles them into stacks of arrays, which downstream code can use to:
    - Gather parameter values by index,
    - Call a batch of component functions,
    - Scatter results into matrices using integer index arrays.

    Parameters
    ----------
    parameter_instructions:
        Carrier update instructions keyed by component identifier.
    signal_instructions:
        Signal update instructions keyed by component identifier.
    noise_instructions:
        Noise update instructions keyed by component identifier.
    signal_components:
        List of component identifiers that belong to the signal solver update set.
    parameter_names:
        List of parameter names aligned with the parameter vector indices.
    optimization_pairs:
        List of pairs `(component, parameter)` or list of lists to represent shared optimization values.
    changing_pairs:
        List of pairs `(component, parameter)` indicating parameters changed over a sweep.

    Returns
    -------
    preparation_arrays:
        `(optimized_parameter_indices, optimization_value_indices,
          carrier_changing_parameter_indices, signal_changing_parameter_indices)`
    carrier_arrays:
        Tuple of arrays needed to update the carrier matrix.
    signal_arrays:
        Tuple of arrays needed to update the signal matrix, including carrier-solution coupling indices.
    noise_arrays:
        Tuple of arrays needed to update the noise matrix.

    Notes
    -----
    - If an instruction set is empty, a dummy update is produced to keep array shapes valid.
    - The signal arrays may include additional arrays that reference carrier solution entries.
    """
    optimized_parameter_indices = []
    optimization_value_indices = []

    carrier_components = []
    carrier_changing_parameter_indices = []
    carrier_arrays = defaultdict(list)

    signal_changing_parameter_indices = []
    signal_arrays = defaultdict(list)

    noise_components = []
    noise_arrays = defaultdict(list)

    if optimization_pairs is None:
        optimization_pairs = []
    if changing_pairs is None:
        changing_pairs = []

    signal_component_number = 0

    # Compile signal update arrays by iterating through all signal instructions.
    for instruction in signal_instructions.values():
        try:
            instruction["function_indices"]
        except KeyError:
            continue

        signal_arrays["function_input_indices"].append(instruction["function_input_indices"])
        signal_arrays["function_indices"].extend(instruction["function_indices"])
        signal_arrays["output_column_indices"].append(instruction["output_column_indices"])

        if "output_row_indices" in instruction:
            signal_arrays["output_row_indices"].append(instruction["output_row_indices"].flatten() + signal_component_number)
            individual_rows = len(np.unique(instruction["output_row_indices"]))
            signal_component_number += individual_rows
        else:
            signal_arrays["output_row_indices"].append(np.ones(len(instruction["output_column_indices"])) * signal_component_number)
            signal_component_number += 1

        signal_arrays["system_matrix_row_indices"].append(instruction["system_matrix_row_indices"])
        signal_arrays["system_matrix_column_indices"].append(instruction["system_matrix_column_indices"])

        if "carrier_indices" in instruction:
            signal_arrays["carrier_indices"].append(instruction["carrier_indices"])
            signal_arrays["carrier_row_indices"].append(instruction["carrier_row_indices"])
            signal_arrays["carrier_column_indices"].append(instruction["carrier_column_indices"])

    def append_information(instructions: dict, arrays: dict, component: str, components: list | None = None):
        """
        Helper to append one component's instruction data to a target arrays dictionary.

        Parameters
        ----------
        instructions:
            Instruction dictionary keyed by component.
        arrays:
            Destination dictionary of lists that will later be stacked into arrays.
        component:
            Component key to append.
        components:
            List used as a "seen" set to avoid adding the same component multiple times.

        Returns
        -------
        None
            This function mutates `arrays` and `components` in-place.
        """
        component_function_input_indices = instructions[component]["function_input_indices"]

        if component in components:
            return None
        components.append(component)

        arrays["function_input_indices"].append(component_function_input_indices)
        arrays["function_indices"].extend(instructions[component]["function_indices"])
        arrays["output_column_indices"].append(instructions[component]["output_column_indices"])
        arrays["output_row_indices"].append(
            np.ones(len(instructions[component]["output_column_indices"])) * (len(components) - 1)
        )
        arrays["system_matrix_row_indices"].append(instructions[component]["system_matrix_row_indices"])
        arrays["system_matrix_column_indices"].append(instructions[component]["system_matrix_column_indices"])

    # Compile noise arrays first (every noise instruction is always relevant).
    for component, instruction in noise_instructions.items():
        append_information(noise_instructions, noise_arrays, component, noise_components)

    # Compile optimized parameter indices and include any carrier-side components that need updating.
    for ix, optimization_pair in enumerate(optimization_pairs):
        if not isinstance(optimization_pair[0], list):
            optimized_component, optimized_parameter = optimization_pair
            try:
                optimized_parameter_indices.append(parameter_names.index(f"{optimized_component}_{optimized_parameter}"))
                optimization_value_indices.append(ix)
            except ValueError:
                pass
        else:
            for optimized_component, optimized_parameter in optimization_pair:
                try:
                    optimized_parameter_indices.append(parameter_names.index(f"{optimized_component}_{optimized_parameter}"))
                    optimization_value_indices.append(ix)
                except ValueError:
                    pass

        if optimized_component not in signal_components and optimized_component in parameter_instructions:
            append_information(parameter_instructions, carrier_arrays, optimized_component, carrier_components)

    # Compile changing parameter indices for sweeps.
    for changing_component, changing_parameter in changing_pairs:
        if changing_component in signal_components:
            signal_changing_parameter_indices.append(parameter_names.index(f"{changing_component}_{changing_parameter}"))
        else:
            carrier_changing_parameter_indices.append(parameter_names.index(f"{changing_component}_{changing_parameter}"))
            append_information(parameter_instructions, carrier_arrays, changing_component, carrier_components)

    def combine_arrays(arrays: dict, signal: bool = False):
        """
        Stack per-component lists into dense arrays. Provide a dummy update if empty.

        Parameters
        ----------
        arrays:
            Dictionary of lists populated by `append_information`.
        signal:
            If True, include carrier-solution coupling indices required by the signal solver.

        Returns
        -------
        tuple
            For carrier and noise: `(function_input_indices, function_indices, output_indices, system_matrix_indices)`
            For signal: plus carrier coupling arrays:
            `(function_input_indices, function_indices, output_indices, system_matrix_indices,
              carrier_solution_indices, carrier_solution_placing_indices)`
        """
        if len(arrays["function_input_indices"]) == 0:
            # Dummy update that leaves matrices unchanged.
            grouped_inputs, grouped_rows, output_indices = _group_by_function(
                np.array([[0, 0, 0]]), [FUNCTIONS.index(dummy_function)], np.array([[0], [0]])
            )
            return (grouped_inputs, grouped_rows, _device_array(output_indices, dtype=int), _device_array([[0], [0]]))

        function_input_indices = np.vstack(arrays["function_input_indices"])
        output_indices = np.stack((np.concatenate(arrays["output_row_indices"]), np.concatenate(arrays["output_column_indices"])))
        system_matrix_indices = np.stack((np.concatenate(arrays["system_matrix_row_indices"]), np.concatenate(arrays["system_matrix_column_indices"])))
        grouped_inputs, grouped_rows, output_indices = _group_by_function(
            function_input_indices, arrays["function_indices"], output_indices
        )

        if signal:
            # Dummy carrier coupling values if none are required.
            carrier_solution_indices = np.array([0])
            carrier_solution_placing_indices = np.array([[0], [-1]])

            if "carrier_indices" in arrays:
                carrier_solution_indices = np.concatenate(arrays["carrier_indices"])
                carrier_solution_placing_indices = np.stack(
                    (np.concatenate(arrays["carrier_row_indices"]), np.concatenate(arrays["carrier_column_indices"]))
                )

            return (
                grouped_inputs,
                grouped_rows,
                _device_array(output_indices, dtype=int),
                _device_array(system_matrix_indices, dtype=int),
                _device_array(carrier_solution_indices, dtype=int),
                _device_array(carrier_solution_placing_indices, dtype=int),
            )

        return (
            grouped_inputs,
            grouped_rows,
            _device_array(output_indices, dtype=int),
            _device_array(system_matrix_indices, dtype=int),
        )

    return (
        (
            _device_array(optimized_parameter_indices, dtype=int),
            _device_array(optimization_value_indices, dtype=int),
            _device_array(carrier_changing_parameter_indices, dtype=int),
            _device_array(signal_changing_parameter_indices, dtype=int),
        ),
        combine_arrays(carrier_arrays),
        combine_arrays(signal_arrays, signal=True),
        combine_arrays(noise_arrays),
    )


def prepare_arrays(
    optimized_parameter_indices,
    optimized_value_indices,
    carrier_changing_parameter_indices,
    signal_changing_parameter_indices,
    changing_values,
    parameters,
):
    """
    Ensure that all index and value arrays are non-empty and have consistent shapes.

    This function inserts "do-nothing" dummy indices and values when a particular feature is unused,
    so downstream vectorized code can run without conditional branches.

    Parameters
    ----------
    optimized_parameter_indices:
        Integer indices into the parameter vector for parameters under optimization.
    optimized_value_indices:
        Integer indices mapping each optimized parameter to an entry in an optimization value vector.
    carrier_changing_parameter_indices:
        Integer indices for parameters changed in the carrier solver sweep.
    signal_changing_parameter_indices:
        Integer indices for parameters changed in the signal solver sweep.
    changing_values:
        Values for the changing parameters. Shape is either `(R,)` or `(V, R)`, where:
        - `V` is number of changing parameters,
        - `R` is number of sweep points.
    parameters:
        Parameter array with shape `(1, N)`.

    Returns
    -------
    tuple
        `(optimized_parameters, optimized_parameter_indices, optimized_value_indices,
          carrier_changing_parameter_indices, carrier_changing_values,
          signal_changing_parameter_indices, signal_changing_values)`

    Raises
    ------
    ValueError
        If both carrier and signal changing indices are non-empty (unsupported combination),
        or if `changing_values` has an incompatible shape.
    """
    optimized_parameters = None

    # Ensure at least one optimized parameter index exists, using a dummy identity update if needed.
    # Index the parameter row on the host: eager JAX indexing would compile XLA programs.
    host_parameters = np.asarray(parameters)

    if len(optimized_parameter_indices) == 0:
        optimized_parameter_indices = _device_array([0])
        optimized_parameters = _device_array(host_parameters[0][[0]])
        optimized_value_indices = _device_array([0])

    # Normalize changing_values to shape (V, R) if provided.
    if changing_values is not None:
        if len(changing_values.shape) == 1:
            changing_values = _reshape_to_row(changing_values)
        # JAX arrays are immutable, so only host (NumPy) inputs need a defensive copy.
        signal_changing_values = changing_values if isinstance(changing_values, jax.Array) else changing_values.copy()

    if len(carrier_changing_parameter_indices) != 0 and len(signal_changing_parameter_indices) != 0:
        raise ValueError("This combination of changing component-parameter pairs is not supported yet.")

    # Carrier changing defaults.
    if len(carrier_changing_parameter_indices) == 0:
        carrier_changing_parameter_indices = _device_array([0])
        carrier_changing_values = _device_array([host_parameters[0][[0]]])
    else:
        carrier_changing_values = changing_values
        if carrier_changing_values.shape[0] != len(carrier_changing_parameter_indices):
            raise ValueError(
                "The changing_values array needs to have shape (number of changing parameters, number of values). "
                "The changing_values array here has shape "
                + str(changing_values.shape)
                + ", but there are "
                + str(len(carrier_changing_parameter_indices))
                + " changing parameters."
            )

    # Signal changing defaults.
    if len(signal_changing_parameter_indices) == 0:
        signal_changing_parameter_indices = _device_array([0])
        signal_changing_values = _device_array([host_parameters[0][[0]]])

    return (
        optimized_parameters,
        optimized_parameter_indices,
        optimized_value_indices,
        carrier_changing_parameter_indices,
        carrier_changing_values,
        signal_changing_parameter_indices,
        signal_changing_values,
    )


# ---------------------------------------------------------------------
# Support dictionaries
# ---------------------------------------------------------------------

MATRIX_SIZES = {
    "mirror": 4,
    "beamsplitter": 8,
    "nothing": 4,
    "directional_beamsplitter": 8,
}
"""
Map from component name to the size of its diagonal submatrix.

These sizes define how many consecutive indices in the carrier system matrix are reserved
for each component type.
"""

PORT_DICTS = {
    "mirror": {"left": 0, "right": 2},
    "beamsplitter": {"left": 0, "right": 4, "bottom": 6, "top": 2},
    "nothing": {"left": 0, "right": 2},
    "directional_beamsplitter": {"left": 0, "right": 4, "bottom": 6, "top": 2},
}
"""
Port offset definitions for each matrix component.

Offsets are counted within the component submatrix, moving counter-clockwise and alternating
between input and output, starting at the left input.

For example for a mirror (size 4):
- left port input is at offset 0, left port output is at offset 1
- right port input is at offset 2, right port output is at offset 3
"""
