"""
Sparse LU solves of the signal system (Option A).

The signal matrix is ~0.3-0.5 % non-zero. Factorising it densely costs O(N^3) per
frequency; SciPy's SuperLU factorises the same matrices about 40x faster. JAX has no
batched sparse LU on CPU, so the solves run in NumPy/SciPy through `jax.pure_callback`,
and `jax.custom_vjp` supplies their exact reverse-mode derivative (adjoint method).

For every frequency f the forward pass solves, with ONE factorisation of A_f:
    A_f x_f = b_f                (signal fields)
    A_f^H W_f = C_f              (detector weights, one column per noise detector)

Reverse mode (JAX's convention: plain transposes, conj where the map is conjugate-linear):
    b̄ = A^{-T} x̄
    q = A^{-1} conj(W̄),  C̄ = conj(q)
    Ā = -b̄ x^T - conj(W) q^T     (needed only at the non-zero positions of A)
The backward pass factorises A again (a factorisation object cannot be stored on the
AD tape between the separately compiled forward and gradient programs).

Nothing here is precomputed or cached between calls: every evaluation factorises
the matrices of the current parameters.
"""

import numpy as np
import jax
import jax.numpy as jnp


def _csc_layout(rows, columns, size):
    """Permutation of the COO entries into CSC order, plus CSC indices and indptr."""
    order = np.lexsort((rows, columns))
    indices = rows[order].astype(np.int32)
    indptr = np.zeros(size + 1, dtype=np.int32)
    np.cumsum(np.bincount(columns, minlength=size), out=indptr[1:])
    return order, indices, indptr


def _factorise(values, layout, size):
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    order, indices, indptr = layout
    matrix = csc_matrix((values[order], indices, indptr), shape=(size, size))
    return matrix, splu(matrix)


def _refined_solve(matrix, lu, rhs, trans="N"):
    """
    Solve with one step of iterative refinement (residual computed with the exact sparse matrix).

    The signal systems are numerically singular (condition numbers ~1e30). Different but
    equally valid LU orderings (SuperLU vs dense LAPACK) then round differently; one
    refinement step brings the sparse result to within ~1e-9 of the dense one.
    """
    operator = {"N": matrix, "T": matrix.T, "H": matrix.conj().T}[trans]
    solution = lu.solve(rhs, trans=trans)
    return solution + lu.solve(rhs - operator @ solution, trans=trans)


def _forward_host(values, rhs, adjoint_rhs, rows, columns):
    values, rhs, adjoint_rhs = np.asarray(values), np.asarray(rhs), np.asarray(adjoint_rhs)
    rows, columns = np.asarray(rows), np.asarray(columns)
    size = rhs.shape[-1]
    layout = _csc_layout(rows, columns, size)
    solution = np.empty(rhs.shape, dtype=np.complex128)
    weights = np.empty(adjoint_rhs.shape, dtype=np.complex128)
    for frequency in range(values.shape[0]):
        matrix, lu = _factorise(values[frequency], layout, size)
        solution[frequency] = _refined_solve(matrix, lu, rhs[frequency])
        weights[frequency] = _refined_solve(matrix, lu, adjoint_rhs[frequency], trans="H")
    return solution, weights


def _backward_host(values, rows, columns, solution, weights, solution_bar, weights_bar):
    values, solution, weights = np.asarray(values), np.asarray(solution), np.asarray(weights)
    solution_bar, weights_bar = np.asarray(solution_bar), np.asarray(weights_bar)
    rows, columns = np.asarray(rows), np.asarray(columns)
    size = solution.shape[-1]
    layout = _csc_layout(rows, columns, size)
    values_bar = np.empty(values.shape, dtype=np.complex128)
    rhs_bar = np.empty(solution.shape, dtype=np.complex128)
    adjoint_rhs_bar = np.empty(weights.shape, dtype=np.complex128)
    for frequency in range(values.shape[0]):
        matrix, lu = _factorise(values[frequency], layout, size)
        b_bar = _refined_solve(matrix, lu, solution_bar[frequency], trans="T")
        q = _refined_solve(matrix, lu, np.conj(weights_bar[frequency]))
        rhs_bar[frequency] = b_bar
        adjoint_rhs_bar[frequency] = np.conj(q)
        values_bar[frequency] = -b_bar[rows] * solution[frequency][columns] - np.sum(
            np.conj(weights[frequency][rows]) * q[columns], axis=-1
        )
    return values_bar, rhs_bar, adjoint_rhs_bar


def _shape(array):
    return jax.ShapeDtypeStruct(array.shape, jnp.complex128)


def _forward(values, rhs, adjoint_rhs, rows, columns):
    return jax.pure_callback(
        _forward_host,
        (_shape(rhs), _shape(adjoint_rhs)),
        values, rhs, adjoint_rhs, rows, columns,
        vmap_method="sequential",
    )


@jax.custom_vjp
def signal_solves(values, rhs, adjoint_rhs, rows, columns):
    """
    Solve A_f x_f = rhs_f and A_f^H W_f = adjoint_rhs_f for every frequency f.

    Parameters
    ----------
    values : (F, nnz) complex
        Non-zero entries of A_f at (`rows`, `columns`) (unique positions).
    rhs : (F, N) complex
    adjoint_rhs : (F, N, D) complex
    rows, columns : (nnz,) int

    Returns
    -------
    solution : (F, N), weights : (F, N, D)
    """
    return _forward(values, rhs, adjoint_rhs, rows, columns)


def _signal_solves_fwd(values, rhs, adjoint_rhs, rows, columns):
    solution, weights = _forward(values, rhs, adjoint_rhs, rows, columns)
    return (solution, weights), (values, rows, columns, solution, weights)


def _signal_solves_bwd(residuals, cotangents):
    values, rows, columns, solution, weights = residuals
    solution_bar, weights_bar = cotangents
    values_bar, rhs_bar, adjoint_rhs_bar = jax.pure_callback(
        _backward_host,
        (_shape(values), _shape(solution), _shape(weights)),
        values, rows, columns, solution, weights, solution_bar, weights_bar,
        vmap_method="sequential",
    )
    return values_bar, rhs_bar, adjoint_rhs_bar, None, None


signal_solves.defvjp(_signal_solves_fwd, _signal_solves_bwd)
