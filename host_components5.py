"""Experimental batched NumPy kernels for real physical component inputs."""
from types import FunctionType

import jax
import numpy as np

from differometor import components


class _Dual:
    """A batched value and its derivative in each local input coordinate."""

    __array_priority__ = 1000

    def __init__(self, value, derivative):
        self.value = value
        self.derivative = derivative

    def __add__(self, other):
        if isinstance(other, _Dual):
            return _Dual(self.value + other.value, self.derivative + other.derivative)
        return _Dual(self.value + other, self.derivative)

    __radd__ = __add__

    def __neg__(self):
        return _Dual(-self.value, -self.derivative)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, _Dual):
            return _Dual(self.value * other.value,
                         self.derivative * other.value + other.derivative * self.value)
        return _Dual(self.value * other, self.derivative * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, _Dual):
            value = self.value / other.value
            return _Dual(value, (self.derivative - other.derivative * value) / other.value)
        return _Dual(self.value / other, self.derivative / other)

    def __rtruediv__(self, other):
        value = other / self.value
        return _Dual(value, -self.derivative * value / self.value)

    def __pow__(self, power):
        return _Dual(self.value**power, self.derivative * (power * self.value**(power - 1)))


def _unary(function, slope):
    def evaluate(x):
        if isinstance(x, _Dual):
            value = function(x.value)
            return _Dual(value, x.derivative * slope(x.value, value))
        return function(x)
    return staticmethod(evaluate)


def _conjugate(x):
    if isinstance(x, _Dual):
        return _Dual(np.conj(x.value), np.conj(x.derivative))
    return np.conj(x)


class _Coefficients(tuple):
    def __neg__(self):
        return _Coefficients(-value for value in self)


class _ComponentMath:
    """NumPy operations used by the existing scalar component formulas."""

    pi = np.pi
    e = np.e
    complex128 = np.complex128
    log10 = staticmethod(np.log10)  # Only applied to a constant by the component laws.
    sin = _unary(np.sin, lambda x, _: np.cos(x))
    cos = _unary(np.cos, lambda x, _: -np.sin(x))
    exp = _unary(np.exp, lambda _, y: y)
    sqrt = _unary(np.sqrt, lambda _, y: 0.5 / y)
    arcsin = _unary(np.arcsin, lambda x, _: 1 / np.sqrt(1 - x*x))
    sinh = _unary(np.sinh, lambda x, _: np.cosh(x))
    cosh = _unary(np.cosh, lambda x, _: np.sinh(x))
    conjugate = conj = staticmethod(_conjugate)

    @staticmethod
    def radians(x):
        return x * (np.pi / 180)

    @staticmethod
    def array(values, dtype=None):
        # The untyped arrays in the laws are vectors of parameters passed to
        # another law. The typed arrays are the standardized coefficient outputs.
        if dtype is None:
            return tuple(values)
        return _Coefficients(values)


def _numpy_component_functions():
    # Keep one source of truth for the equations and use an isolated namespace
    # for the NumPy backend, without modifying the imported JAX module.
    namespace = dict(vars(components), jnp=_ComponentMath)
    for name, function in vars(components).items():
        if isinstance(function, FunctionType) and function.__module__ == components.__name__:
            namespace[name] = FunctionType(function.__code__, namespace, name, function.__defaults__, function.__closure__)
    return tuple(namespace[function.__name__] for function in components.FUNCTIONS)


_FUNCTIONS = _numpy_component_functions()


def _evaluate(parameters, groups, output_indices, cotangent=None):
    parameters = np.asarray(parameters)
    output_rows, output_columns = map(np.asarray, output_indices)
    batch_size = parameters.shape[0]
    if cotangent is None:
        coefficients = np.empty((batch_size, output_rows.size), dtype=np.complex128)
    else:
        cotangent = np.asarray(cotangent)
        gradient = np.zeros_like(parameters)
        batch_rows = np.arange(batch_size)[:, None, None]

    start = 0
    for function, indices in zip(_FUNCTIONS, groups):
        indices = np.asarray(indices)
        count, width = indices.shape
        if not count:
            continue
        selected = (output_rows >= start) & (output_rows < start + count)
        local_rows = output_rows[selected] - start
        local_columns = output_columns[selected]
        inputs = parameters[:, indices]
        if cotangent is None:
            outputs = function(tuple(np.moveaxis(inputs, -1, 0)))
            values = np.stack([np.broadcast_to(x, (batch_size, count)) for x in outputs], axis=-1)
            coefficients[:, selected] = values[:, local_rows, local_columns]
        else:
            basis = np.eye(width)[:, :, None, None]
            dual_inputs = [_Dual(inputs[..., col], basis[:, col]) for col in range(width)]
            outputs = function(dual_inputs)
            local_gradient = np.zeros((batch_size, count, width), dtype=parameters.dtype)
            for column, output in enumerate(outputs):
                if not isinstance(output, _Dual):
                    continue
                writes = local_columns == column
                rows = local_rows[writes]
                direction = np.broadcast_to(output.derivative, (width, batch_size, count))
                partial = np.moveaxis(direction[:, :, rows], 0, -1)
                # The input is real and JAX complex cotangents are bilinear.
                contributions = np.real(cotangent[:, selected][:, writes, None] * partial)
                np.add.at(local_gradient, (batch_rows, rows[None, :, None], np.arange(width)), contributions)
            np.add.at(gradient, (batch_rows, indices[None, :, :]), local_gradient)
        start += count
    return coefficients if cotangent is None else gradient


def _forward_host(parameters, groups, output_indices):
    return _evaluate(parameters, groups, output_indices)


def _backward_host(parameters, groups, output_indices, cotangent):
    return _evaluate(parameters, groups, output_indices, cotangent)


@jax.custom_vjp
def component_outputs(parameters, groups, output_indices):
    output = jax.ShapeDtypeStruct((parameters.shape[0], output_indices[0].shape[0]), np.complex128)
    return jax.pure_callback(_forward_host, output, parameters, groups, output_indices,
                             vmap_method="sequential")


def _forward(parameters, groups, output_indices):
    return component_outputs(parameters, groups, output_indices), (parameters, groups, output_indices)


def _backward(residuals, cotangent):
    parameters, groups, output_indices = residuals
    output = jax.ShapeDtypeStruct(parameters.shape, parameters.dtype)
    gradient = jax.pure_callback(_backward_host, output, parameters, groups, output_indices, cotangent,
                                 vmap_method="sequential")
    return gradient, None, None


component_outputs.defvjp(_forward, _backward)
