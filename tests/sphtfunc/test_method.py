from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

import jax_healpy as jhp

NSIDE = 8
LMAX = 2 * NSIDE - 1


def transforms(method: str) -> dict[str, Callable[[], object]]:
    """The functions that take a ``method`` argument, as calls without arguments."""
    L = LMAX + 1
    alm = jnp.zeros((L, 2 * L - 1), dtype=complex)
    m = jnp.zeros(12 * NSIDE**2)
    return {
        'alm2map': lambda: jhp.alm2map(alm, NSIDE, lmax=LMAX, method=method),
        'map2alm': lambda: jhp.map2alm(m, lmax=LMAX, method=method),
        'alm2map_spin': lambda: jhp.alm2map_spin([alm, alm], NSIDE, 2, lmax=LMAX, method=method),
        'map2alm_spin': lambda: jhp.map2alm_spin([m, m], 2, lmax=LMAX, method=method),
        'anafast': lambda: jhp.anafast(m, lmax=LMAX, method=method),
        'synfast': lambda: jhp.synfast(jax.random.PRNGKey(0), jnp.ones(L), NSIDE, lmax=LMAX, method=method),
    }


FUNCTIONS = list(transforms('jax'))


@pytest.mark.parametrize('name', FUNCTIONS)
def test_method_jax_healpy_is_rejected(name: str) -> None:
    """s2fft's method 'jax_healpy' cannot be compiled and ignores the spin: it is refused upfront."""
    with pytest.raises(ValueError, match="method='jax_healpy' is not supported"):
        transforms('jax_healpy')[name]()


@pytest.mark.parametrize('name', FUNCTIONS)
@pytest.mark.parametrize('method', ['numpy', 'JAX', ''])
def test_invalid_method(name: str, method: str) -> None:
    with pytest.raises(ValueError, match=f'Invalid method {method!r}'):
        transforms(method)[name]()
