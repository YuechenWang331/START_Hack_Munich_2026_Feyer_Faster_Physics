"""Signal and adjoint solves with a common structural CSC pattern and implicit AD."""
from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


def make_signal_pattern(augmented, update_indices, carrier_indices):
    """Mark all potential coefficient writes, even if initially they contain zero."""
    mask = np.asarray(augmented) != 0
    mask[tuple(np.asarray(update_indices))] = True
    mask[tuple(np.asarray(carrier_indices))] = True
    coefficients = mask[:, :-1]
    cols, rows = np.nonzero(coefficients.T)
    column_starts = np.searchsorted(cols, np.arange(coefficients.shape[1] + 1))
    return tuple(jnp.asarray(indices, dtype=jnp.int32) for indices in (rows, cols, column_starts))


def _host_solve(values, row_indices, column_starts, rhs, *, transposed):
    rhs = np.asarray(rhs)
    values = np.asarray(values)
    row_indices = np.asarray(row_indices)
    column_starts = np.asarray(column_starts)
    n = rhs.shape[1]
    solutions = np.empty_like(rhs)
    for frequency, frequency_values in enumerate(values):
        matrix = csc_matrix((frequency_values, row_indices, column_starts), shape=(n, n))
        lu = splu(matrix)
        sources = rhs[frequency]
        # First column uses A, the remaining detector columns use Aᴴ.
        # The complex-linear JAX transpose uses Aᵀ and conjugate(A), respectively.
        solutions[frequency, :, 0] = lu.solve(sources[:, 0], trans="T" if transposed else "N")
        if sources.shape[1] > 1:
            if transposed:
                weights = np.conj(lu.solve(np.conj(sources[:, 1:])))
            else:
                weights = lu.solve(sources[:, 1:], trans="H")
            solutions[frequency, :, 1:] = weights
    return solutions


def solve_signal_and_weights(system, signal_rhs, detector_rhs, pattern):
    rows, cols, column_starts = pattern
    values = system[:, rows, cols]
    rhs = jnp.concatenate((signal_rhs[..., None], detector_rhs), axis=-1)

    def matvec(x):
        signal = jnp.zeros_like(x[:, :, 0]).at[:, rows].add(values * x[:, cols, 0])
        detectors = jnp.zeros_like(x[:, :, 1:]).at[:, cols, :].add(
            jnp.conj(values)[..., None] * x[:, rows, 1:]
        )
        return jnp.concatenate((signal[..., None], detectors), axis=-1)

    def apply_lu(_, b, *, transposed):
        return jax.pure_callback(
            partial(_host_solve, transposed=transposed), jax.ShapeDtypeStruct(b.shape, b.dtype),
            lax.stop_gradient(values), rows, column_starts, b, vmap_method="sequential",
        )

    result = lax.custom_linear_solve(
        matvec, rhs, solve=partial(apply_lu, transposed=False),
        transpose_solve=partial(apply_lu, transposed=True),
    )
    return result[:, :, 0], result[:, :, 1:]
