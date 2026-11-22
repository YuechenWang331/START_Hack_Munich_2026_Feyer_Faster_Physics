"""Dense CPU carrier solve with an implicit reverse-mode derivative."""
import jax
import numpy as np


def _solve_host(matrix, rhs, *, transpose=False):
    from scipy.linalg import solve

    matrix = np.asarray(matrix)
    rhs = np.asarray(rhs)
    if transpose:
        matrix = matrix.swapaxes(-1, -2)
    return solve(matrix, rhs[..., None], assume_a="gen", check_finite=False)[..., 0]


def _solve(matrix, rhs, *, transpose=False):
    from functools import partial

    return jax.pure_callback(
        partial(_solve_host, transpose=transpose),
        jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
        matrix,
        rhs,
        vmap_method="sequential",
    )


@jax.custom_vjp
def carrier_solve(matrix, rhs):
    return _solve(matrix, rhs)


def _forward(matrix, rhs):
    solution = _solve(matrix, rhs)
    return solution, (matrix, solution)


def _backward(residuals, solution_bar):
    matrix, solution = residuals
    # JAX's complex cotangents use the bilinear pairing, hence transpose.
    rhs_bar = _solve(matrix, solution_bar, transpose=True)
    matrix_bar = -rhs_bar[..., :, None] * solution[..., None, :]
    return matrix_bar, rhs_bar


carrier_solve.defvjp(_forward, _backward)
