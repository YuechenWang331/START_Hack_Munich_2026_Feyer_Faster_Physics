"""Check CPU callback values and complex VJPs against explicit small dense systems."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from differometor.components import FUNCTIONS, signal_function
from differometor.host_components import component_outputs
from differometor.host_response import assembled_carrier, signal_and_noise


def calls(count):
    # Each call reads an overlapping amplitude/phase pair from the parameter vector.
    groups = [jnp.empty((0, 2), dtype=int) for _ in FUNCTIONS]
    groups[FUNCTIONS.index(signal_function)] = jnp.stack((jnp.arange(count), jnp.arange(count) + 1), axis=1)
    return tuple(groups), (jnp.arange(count), jnp.zeros(count, dtype=int))


class HostResponseTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(47)

    def complex_values(self, shape):
        return jnp.asarray(self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape))

    def compare(self, cpu, reference, args):
        actual, back = jax.vjp(cpu, *args)
        expected, ref_back = jax.vjp(reference, *args)
        cotangents = jax.tree.map(
            lambda x: self.complex_values(x.shape) if jnp.iscomplexobj(x)
            else jnp.asarray(self.rng.normal(size=x.shape)), actual,
        )
        def check(a, b):
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)
        jax.tree.map(check, actual, expected)
        jax.tree.map(check, back(cotangents), ref_back(cotangents))

    def test_carrier_overwrites(self):
        groups = calls(5)
        # Includes a repeated slot, a negative RHS-column index and a dropped write.
        rows, cols = jnp.array([0, 0, 0, 1, 5]), jnp.array([1, 1, -1, 0, 0])
        parameters = jnp.asarray(self.rng.normal(size=(2, 6)))
        base = self.complex_values((3, 4)).at[:, :3].add(6 * jnp.eye(3))

        def reference(p, a):
            coefficients = component_outputs(p, *groups)
            matrices = jax.vmap(lambda values: a.at[rows, cols].set(values))(coefficients)
            return jnp.linalg.solve(matrices[:, :, :-1], matrices[:, :, -1:])[..., 0]

        self.compare(lambda p, a: assembled_carrier(p, a, (rows, cols), groups), reference, (parameters, base))

    def test_signal_and_noise(self):
        n, frequencies = 3, 3
        signal_calls, noise_calls = calls(4), calls(3)
        signal_writes, carrier_writes, noise_writes = map(jnp.asarray, ([1, 2, 1, 6], [2, 6, 2], [1, 1, 3]))
        slots = np.arange(n * (n + 1)).reshape(n, n + 1)
        system_slots, rhs_slots = jnp.asarray(slots[:, :-1].ravel()), jnp.asarray(slots[:, -1])
        rows, cols = jnp.indices((n, n)).reshape(2, -1)
        noise_rows, noise_cols = jnp.array([0, 0, 1, 2, 2, 1]), jnp.array([0, 1, 2, 0, 2, 1])
        conjugate = jnp.asarray([False, True, True, False, False, True, False, True, True, False, True, True])
        layout = (conjugate, system_slots, rows, cols, rhs_slots, jnp.arange(n))
        parameters = jnp.asarray(self.rng.normal(size=(frequencies, 5)))
        base = self.complex_values((n, n + 1)).at[:, :-1].add(7 * jnp.eye(n))
        noise_base = self.complex_values((noise_rows.size,))

        def cpu(p, carrier, detectors, base, noise_base):
            return signal_and_noise(
                p, carrier, detectors, (base, *layout), (noise_base, noise_rows, noise_cols),
                (signal_writes, carrier_writes, noise_writes), signal_calls, noise_calls,
            )

        def reference(p, carrier, detectors, base, noise_base):
            coefficients = component_outputs(p, *signal_calls)
            values = jax.vmap(lambda e: base.at[signal_writes].set(e))(coefficients)
            values = values.at[:, carrier_writes].set(-values[:, carrier_writes] * carrier)
            values = jnp.where(conjugate, jnp.conj(values), values)
            a = values[:, system_slots].reshape(frequencies, n, n)
            signal = jnp.linalg.solve(a, values[:, rhs_slots, None])[..., 0]
            weights = jnp.linalg.solve(jnp.conj(a).swapaxes(-1, -2), detectors)
            noise = component_outputs(p, *noise_calls)
            entries = jax.vmap(lambda e: noise_base.at[noise_writes].set(e))(noise)
            covariance = jax.vmap(lambda e: jnp.zeros_like(a[0]).at[noise_rows, noise_cols].add(e))(entries)
            return signal, (jnp.conj(weights) * (covariance @ weights)).sum(axis=1).real

        for carrier_batch, detectors in ((1, 2), (frequencies, 1)):
            with self.subTest(carrier_batch=carrier_batch, detectors=detectors):
                self.compare(cpu, reference, (
                    parameters, self.complex_values((carrier_batch, carrier_writes.size)),
                    self.complex_values((frequencies, n, detectors)), base.ravel(), noise_base,
                ))


if __name__ == "__main__":
    unittest.main()
