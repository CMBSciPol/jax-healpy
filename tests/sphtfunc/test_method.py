import warnings
from collections.abc import Callable

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_healpy as jhp
from jax_healpy import _ducc0, sphtfunc

from .helpers import requires_ducc0, requires_s2fft

NSIDE = 8
LMAX = 2 * NSIDE - 1
NPIX = 12 * NSIDE**2
NALM = (LMAX + 1) * (LMAX + 2) // 2

# The transforms of ducc0 are compiled much faster than those of s2fft.
FASTEST_METHOD = 'ducc0' if _ducc0.is_available() else 'jax'

# These tests choose the methods themselves: they are run once, not once per default method.
pytestmark = pytest.mark.parametrize('default_method', [FASTEST_METHOD], indirect=True)

# The function itself: during the tests, the default_method fixture replaces it in the module.
select_default_method = sphtfunc._default_method


def transforms(method: str | None) -> dict[str, Callable[[], object]]:
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
        'smoothing': lambda: jhp.smoothing(m, fwhm=0.1, lmax=LMAX, method=method),
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


@pytest.mark.parametrize('name', FUNCTIONS)
def test_method_ducc0_requires_ducc0(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_ducc0, 'is_available', lambda: False)
    with pytest.raises(ImportError, match="Missing optional library 'ducc0'"):
        transforms('ducc0')[name]()


@pytest.mark.parametrize(
    'ducc0_installed, backend, expected',
    [(True, 'cpu', 'ducc0'), (False, 'cpu', 'jax'), (True, 'gpu', 'jax'), (False, 'gpu', 'jax')],
)
def test_default_method(monkeypatch: pytest.MonkeyPatch, ducc0_installed: bool, backend: str, expected: str) -> None:
    """ducc0 is the default on CPU when it is installed, the pure JAX transforms otherwise."""
    monkeypatch.setattr(_ducc0, 'is_available', lambda: ducc0_installed)
    monkeypatch.setattr(jax, 'default_backend', lambda: backend)
    assert select_default_method() == expected


@pytest.mark.parametrize('name', FUNCTIONS)
def test_default_method_is_used(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without method, the functions use the default backend, even if it changed since they were compiled."""
    transforms(None)[name]()

    def default_method() -> str:
        raise RuntimeError('The default method is requested.')

    monkeypatch.setattr(sphtfunc, '_default_method', default_method)
    with pytest.raises(RuntimeError, match='The default method is requested'):
        transforms(None)[name]()
    transforms(FASTEST_METHOD)[name]()


def test_method_can_be_positional() -> None:
    L = LMAX + 1
    alm = jnp.zeros((L, 2 * L - 1), dtype=complex)
    with pytest.raises(ValueError, match="Invalid method 'numpy'"):
        jhp.alm2map(alm, NSIDE, LMAX, None, False, 0.0, None, True, False, 'numpy')


@requires_ducc0
@requires_s2fft
@pytest.mark.parametrize('healpy_ordering', [False, True])
def test_methods_agree(healpy_ordering: bool) -> None:
    """The two backends return the same alms and maps, with the same dtypes, in both alm orderings."""
    nside = NSIDE
    lmax = 3 * nside - 1
    iqu = np.random.RandomState(0).randn(3, hp.nside2npix(nside))

    alms = {
        method: jhp.map2alm(iqu, lmax=lmax, iter=1, healpy_ordering=healpy_ordering, method=method)
        for method in ('jax', 'ducc0')
    }
    assert alms['ducc0'].shape == alms['jax'].shape
    assert alms['ducc0'].dtype == alms['jax'].dtype
    np.testing.assert_allclose(alms['ducc0'], alms['jax'], atol=1e-12, rtol=0)

    maps = {
        method: jhp.alm2map(alms['jax'], nside, lmax=lmax, healpy_ordering=healpy_ordering, method=method)
        for method in ('jax', 'ducc0')
    }
    assert maps['ducc0'].shape == maps['jax'].shape
    assert maps['ducc0'].dtype == maps['jax'].dtype
    np.testing.assert_allclose(maps['ducc0'], maps['jax'], atol=1e-10, rtol=0)


@requires_ducc0
@requires_s2fft
@pytest.mark.parametrize('healpy_ordering', [False, True])
def test_map2alm_transpose_matches_s2fft(healpy_ordering: bool) -> None:
    """jax.linear_transpose gives the same transpose of map2alm with both backends."""
    maps = jnp.asarray(np.random.RandomState(0).randn(NPIX))

    def transposed(method: str) -> jax.Array:
        def func(x):
            return jhp.map2alm(x, lmax=LMAX, iter=1, healpy_ordering=healpy_ordering, method=method)

        rng = np.random.RandomState(1)
        shape = func(maps).shape
        cotangent = jnp.asarray(rng.randn(*shape) + 1j * rng.randn(*shape))
        return jax.linear_transpose(func, maps)(cotangent)[0]

    np.testing.assert_allclose(transposed('ducc0'), transposed('jax'), atol=1e-13, rtol=0)


@requires_ducc0
@requires_s2fft
def test_gradients_match_s2fft() -> None:
    """The derivatives through the transforms are the same with both backends."""
    iqu = jnp.asarray(np.random.RandomState(0).randn(3, NPIX))

    def gradient(method: str) -> jax.Array:
        def loss(x):
            alms = jhp.map2alm(x, lmax=LMAX, iter=1, method=method)
            return jnp.sum(jnp.abs(alms) ** 2) + jnp.sum(jhp.alm2map(alms, NSIDE, lmax=LMAX, method=method) ** 3)

        return jax.grad(loss)(iqu)

    expected = gradient('jax')
    assert jnp.abs(expected).max() > 1e-3
    np.testing.assert_allclose(gradient('ducc0'), expected, atol=1e-12, rtol=0)


@requires_ducc0
@pytest.mark.parametrize('dtype', [np.float32, np.int32])
def test_map2alm_promotes_to_default_precision(dtype: type) -> None:
    """Like those of s2fft, the transforms are computed in the default precision of JAX."""
    maps = jnp.asarray(np.random.RandomState(0).randint(-5, 5, NPIX), dtype=dtype)
    alms = jhp.map2alm(maps, lmax=LMAX, iter=0, healpy_ordering=True, method='ducc0')
    assert alms.dtype == jnp.complex128
    expected = hp.map2alm(np.asarray(maps, dtype=float), lmax=LMAX, iter=0)
    np.testing.assert_allclose(alms, expected, atol=1e-14, rtol=0)

    out = jhp.alm2map(alms.astype(jnp.complex64), NSIDE, lmax=LMAX, healpy_ordering=True, method='ducc0')
    assert out.dtype == jnp.float64


@requires_ducc0
def test_alm2map_wrong_size() -> None:
    with pytest.raises(ValueError, match='The alms have 135 coefficients, instead of 136 for lmax=15'):
        jhp.alm2map(jnp.zeros(NALM - 1, dtype=complex), NSIDE, lmax=LMAX, healpy_ordering=True, method='ducc0')


@requires_ducc0
def test_single_precision_mode() -> None:
    """Without 64-bit precision, the transforms run in single precision."""
    maps = np.random.RandomState(0).randn(NPIX)
    expected = hp.map2alm(maps, lmax=LMAX, iter=0)
    with jax.enable_x64(False), warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # the warning of jax-healpy about 32-bit precision
        alms = jhp.map2alm(jnp.asarray(maps), lmax=LMAX, iter=0, healpy_ordering=True, method='ducc0')
        assert alms.dtype == jnp.complex64
        out = jhp.alm2map(alms, NSIDE, lmax=LMAX, healpy_ordering=True, method='ducc0')
        assert out.dtype == jnp.float32
    np.testing.assert_allclose(alms, expected, atol=1e-6, rtol=0)
    np.testing.assert_allclose(out, hp.alm2map(expected, NSIDE, lmax=LMAX), atol=1e-4, rtol=0)
