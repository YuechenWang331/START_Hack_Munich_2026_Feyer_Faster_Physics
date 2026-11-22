"""Assemble signal and noise coefficients and propagate them in one CPU callback."""
import jax
import numpy as np

from differometor.carrier_solve import _solve_host as solve_carrier
from differometor.host_components import _evaluate as evaluate_components
from differometor.sparse_solve import _forward_host as solve_fields, _backward_host as field_pullback


def _assemble(parameters, base, writes, calls):
    coefficients = evaluate_components(parameters, *calls)
    values = np.broadcast_to(base, (parameters.shape[0], base.size)).copy()
    values[:, writes] = coefficients
    return values


def _write_cotangents(cotangent, writes):
    """Gather derivatives of indexed assignments, including repeated destinations."""
    writes = np.asarray(writes)
    result = cotangent[:, writes]
    # NumPy advanced assignment retains the last source for each destination.
    _, last_from_end = np.unique(writes[::-1], return_index=True)
    if last_from_end.size != writes.size:
        live = np.zeros(writes.size, dtype=bool)
        live[writes.size - 1 - last_from_end] = True
        result[:, ~live] = 0
    return result


def _assembly_pullback(parameters, cotangent, writes, calls):
    coefficients_bar = _write_cotangents(cotangent, writes)
    base_bar = cotangent.sum(axis=0)
    base_bar[writes] = 0
    return evaluate_components(parameters, *calls, coefficients_bar), base_bar


def _carrier_slots(shape, matrix_indices, calls):
    rows, cols = map(np.asarray, matrix_indices)
    rows = np.where(rows < 0, rows + shape[0], rows)
    cols = np.where(cols < 0, cols + shape[1], cols)
    # Match JAX scatter.set: wrap negative indices once and drop out-of-bounds writes.
    valid = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    if not np.all(valid):
        rows, cols = rows[valid], cols[valid]
        groups, (output_rows, output_cols) = calls
        calls = (groups, (np.asarray(output_rows)[valid], np.asarray(output_cols)[valid]))
    return rows * shape[1] + cols, calls


def _carrier_host(parameters, base, matrix_indices, calls):
    parameters, base = np.asarray(parameters), np.asarray(base)
    slots, calls = _carrier_slots(base.shape, matrix_indices, calls)
    augmented = _assemble(parameters, base.ravel(), slots, calls).reshape((-1, *base.shape))
    matrix = augmented[:, :, :-1]
    return solve_carrier(matrix, augmented[:, :, -1]), matrix


def _carrier_pullback(parameters, matrix, carrier, matrix_indices, calls, carrier_bar):
    parameters, matrix, carrier = map(np.asarray, (parameters, matrix, carrier))
    rhs_bar = solve_carrier(matrix, np.asarray(carrier_bar), transpose=True)
    batch, size = carrier.shape
    augmented_bar = np.empty((batch, size, size + 1), dtype=matrix.dtype)
    augmented_bar[:, :, :-1] = -rhs_bar[:, :, None] * carrier[:, None, :]
    augmented_bar[:, :, -1] = rhs_bar
    slots, calls = _carrier_slots((size, size + 1), matrix_indices, calls)
    p_bar, base_bar = _assembly_pullback(parameters, augmented_bar.reshape(batch, -1), slots, calls)
    return p_bar, base_bar.reshape(size, size + 1)


def _carrier_forward(parameters, base, matrix_indices, calls):
    batch, size = parameters.shape[0], base.shape[0]
    shapes = (jax.ShapeDtypeStruct((batch, size), base.dtype),
              jax.ShapeDtypeStruct((batch, size, size), base.dtype))
    return jax.pure_callback(_carrier_host, shapes, parameters, base, matrix_indices, calls,
                             vmap_method="sequential")


@jax.custom_vjp
def assembled_carrier(parameters, base, matrix_indices, calls):
    """Solve the batched carrier system after inserting the current optical coefficients."""
    carrier, _ = _carrier_forward(parameters, base, matrix_indices, calls)
    return carrier


def _carrier_fwd(parameters, base, matrix_indices, calls):
    carrier, matrix = _carrier_forward(parameters, base, matrix_indices, calls)
    return carrier, (parameters, matrix, carrier, matrix_indices, calls)


def _carrier_bwd(residuals, carrier_bar):
    parameters, matrix, *_ = residuals
    size = matrix.shape[-1]
    shapes = (jax.ShapeDtypeStruct(parameters.shape, parameters.dtype),
              jax.ShapeDtypeStruct((size, size + 1), matrix.dtype))
    p_bar, base_bar = jax.pure_callback(_carrier_pullback, shapes, *residuals, carrier_bar,
                                        vmap_method="sequential")
    return p_bar, base_bar, None, None


assembled_carrier.defvjp(_carrier_fwd, _carrier_bwd)


def _forward_host(parameters, coupling, detector_rhs, signal_matrix, noise_matrix,
                  writes, signal_calls, noise_calls):
    parameters, coupling, detector_rhs = map(np.asarray, (parameters, coupling, detector_rhs))
    base, conjugate, system_slots, rows, cols, rhs_slots, rhs_rows = map(np.asarray, signal_matrix)
    noise_base, noise_rows, noise_cols = map(np.asarray, noise_matrix)
    signal_writes, carrier_writes, noise_writes = map(np.asarray, writes)

    values = _assemble(parameters, base, signal_writes, signal_calls)
    unscaled = values[:, carrier_writes]
    values[:, carrier_writes] = -unscaled * coupling
    np.conjugate(values, out=values, where=conjugate)
    system = values[:, system_slots]
    rhs = np.zeros(detector_rhs.shape[:2], dtype=values.dtype)
    rhs[:, rhs_rows] = values[:, rhs_slots]
    signal, weights = solve_fields(system, rhs, detector_rhs, rows, cols)

    # w^H N w, using every stored entry of N, including off-diagonal terms.
    noise_values = _assemble(parameters, noise_base, noise_writes, noise_calls)
    power = np.real(np.sum(
        np.conj(weights[:, noise_rows]) * (noise_values[..., None] * weights[:, noise_cols]),
        axis=1,
    ))
    return signal, power, (weights, system, unscaled, noise_values)


def _power_pullback(weights, noise_values, noise_rows, noise_cols, power_bar):
    left, right = weights[:, noise_rows], weights[:, noise_cols]
    scale = power_bar[:, None, :]
    values_bar = np.sum(scale * np.conj(left) * right, axis=-1)
    weights_bar = np.zeros_like(weights)
    # A row or column may receive several contributions from the covariance.
    np.add.at(weights_bar.swapaxes(0, 1), noise_rows,
              (scale * np.conj(noise_values[..., None] * right)).swapaxes(0, 1))
    np.add.at(weights_bar.swapaxes(0, 1), noise_cols,
              (scale * np.conj(left) * noise_values[..., None]).swapaxes(0, 1))
    return values_bar, weights_bar


def _backward_host(parameters, coupling, signal, saved, signal_layout, noise_layout,
                   writes, signal_calls, noise_calls, signal_bar, power_bar):
    parameters, coupling, signal = map(np.asarray, (parameters, coupling, signal))
    weights, system, unscaled, noise_values = map(np.asarray, saved)
    conjugate, system_slots, rows, cols, rhs_slots, rhs_rows = map(np.asarray, signal_layout)
    noise_rows, noise_cols = map(np.asarray, noise_layout)
    signal_writes, carrier_writes, noise_writes = map(np.asarray, writes)

    noise_bar, weights_bar = _power_pullback(weights, noise_values, noise_rows, noise_cols,
                                            np.asarray(power_bar))
    system_bar, rhs_bar, detector_bar = field_pullback(
        system, rows, cols, signal, weights, np.asarray(signal_bar), weights_bar,
    )
    value_bar = np.empty((system.shape[0], conjugate.size), dtype=system.dtype)
    value_bar[:, system_slots] = system_bar
    value_bar[:, rhs_slots] = rhs_bar[:, rhs_rows]
    np.conjugate(value_bar, out=value_bar, where=conjugate)
    entry_bar = _write_cotangents(value_bar, carrier_writes)
    coupling_bar = -entry_bar * unscaled
    if coupling.shape[0] == 1:
        coupling_bar = coupling_bar.sum(axis=0, keepdims=True)
    value_bar[:, carrier_writes] = -entry_bar * coupling

    parameters_bar, base_bar = _assembly_pullback(parameters, value_bar, signal_writes, signal_calls)
    noise_parameters_bar, noise_base_bar = _assembly_pullback(parameters, noise_bar, noise_writes, noise_calls)
    return parameters_bar + noise_parameters_bar, coupling_bar, detector_bar, base_bar, noise_base_bar


def _forward(parameters, coupling, detector_rhs, signal_matrix, noise_matrix, writes, signal_calls, noise_calls):
    frequency_count, size, detectors = detector_rhs.shape
    complex_array = lambda shape: jax.ShapeDtypeStruct(shape, detector_rhs.dtype)
    signal_shape = complex_array((frequency_count, size))
    power_shape = jax.ShapeDtypeStruct((frequency_count, detectors), detector_rhs.real.dtype)
    saved_shapes = (
        complex_array(detector_rhs.shape),
        complex_array((frequency_count, signal_matrix[2].size)),
        complex_array((frequency_count, writes[1].size)),
        complex_array((frequency_count, noise_matrix[0].size)),
    )
    return jax.pure_callback(
        _forward_host, (signal_shape, power_shape, saved_shapes),
        parameters, coupling, detector_rhs, signal_matrix, noise_matrix, writes, signal_calls, noise_calls,
        vmap_method="sequential",
    )


@jax.custom_vjp
def signal_and_noise(parameters, coupling, detector_rhs, signal_matrix, noise_matrix, writes, signal_calls, noise_calls):
    """Return the signal field and each detector's noise power for the supplied physical inputs."""
    signal, power, _ = _forward(parameters, coupling, detector_rhs, signal_matrix, noise_matrix,
                                writes, signal_calls, noise_calls)
    return signal, power


def _response_fwd(parameters, coupling, detector_rhs, signal_matrix, noise_matrix, writes, signal_calls, noise_calls):
    signal, power, saved = _forward(parameters, coupling, detector_rhs, signal_matrix, noise_matrix,
                                    writes, signal_calls, noise_calls)
    return (signal, power), (parameters, coupling, signal, saved, signal_matrix[1:], noise_matrix[1:],
                              writes, signal_calls, noise_calls)


def _response_bwd(residuals, cotangents):
    parameters, coupling, signal, saved, signal_layout, noise_layout, writes, signal_calls, noise_calls = residuals
    weights, _, _, noise_values = saved
    shapes = (
        jax.ShapeDtypeStruct(parameters.shape, parameters.dtype),
        jax.ShapeDtypeStruct(coupling.shape, coupling.dtype),
        jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        jax.ShapeDtypeStruct(signal_layout[0].shape, signal.dtype),
        jax.ShapeDtypeStruct(noise_values.shape[1:], noise_values.dtype),
    )
    p_bar, c_bar, d_bar, base_bar, noise_bar = jax.pure_callback(
        _backward_host, shapes, *residuals, *cotangents, vmap_method="sequential",
    )
    return (p_bar, c_bar, d_bar, (base_bar, *(None for _ in signal_layout)),
            (noise_bar, *(None for _ in noise_layout)), None, None, None)


signal_and_noise.defvjp(_response_fwd, _response_bwd)
