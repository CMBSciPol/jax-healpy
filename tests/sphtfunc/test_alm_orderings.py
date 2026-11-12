"""Conversions between the healpy ordering of the alms (m >= 0, flat) and the 2-D one (all m)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_healpy import sphtfunc

from .helpers import METHODS, flm_2d_to_hp, flm_hp_to_2d, hp_indices, requires_s2fft

# These tests do not transform anything: they are run once, not once per default method.
pytestmark = pytest.mark.parametrize('default_method', METHODS[:1], indirect=True)

LMAX = 15
NALM = (LMAX + 1) * (LMAX + 2) // 2


@pytest.fixture(params=['stored', 'computed'])
def alm_indices(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Exercise the indices of the healpy ordering both as constants and as computations."""
    if request.param == 'computed':
        monkeypatch.setattr(sphtfunc, '_STORED_INDICES_LMAX', -1)
    return request.param


@pytest.mark.usefixtures('alm_indices')
@pytest.mark.parametrize('lmax', [*range(20), 95, 191, 767])
def test_alm_hp_indices(lmax: int, x64: bool) -> None:
    """The degrees and the orders of the healpy ordering, in 32-bit and in 64-bit precision."""
    expected_ell, expected_m = hp_indices(lmax + 1)
    ell, m = sphtfunc._alm_hp_indices(lmax)
    assert ell.dtype == m.dtype == np.int32
    np.testing.assert_array_equal(ell, expected_ell)
    np.testing.assert_array_equal(m, expected_m)


def test_alm_hp_indices_are_stored_up_to_a_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The indices are numpy constants for the small lmax, and computed by JAX beyond."""
    monkeypatch.setattr(sphtfunc, '_STORED_INDICES_LMAX', 7)
    assert all(isinstance(indices, np.ndarray) for indices in sphtfunc._alm_hp_indices(7))
    assert all(isinstance(indices, jax.Array) for indices in sphtfunc._alm_hp_indices(8))


@pytest.mark.usefixtures('alm_indices')
@pytest.mark.parametrize('lmax', [0, 1, 2, 7, LMAX])
def test_alm_2d_to_hp(lmax: int) -> None:
    """The conversion to the healpy ordering, for a batch of alms, eagerly and compiled."""
    L = lmax + 1
    rng = np.random.RandomState(lmax)
    alms = rng.randn(2, L, 2 * L - 1) + 1j * rng.randn(2, L, 2 * L - 1)
    expected = np.stack([flm_2d_to_hp(alm, L) for alm in alms])
    np.testing.assert_array_equal(sphtfunc._alm_2d_to_hp(alms, lmax), expected)
    np.testing.assert_array_equal(jax.jit(lambda x: sphtfunc._alm_2d_to_hp(x, lmax))(alms), expected)


@pytest.mark.parametrize('lmax', [0, 1, 2, 7, LMAX])
def test_alm_hp_to_2d(lmax: int) -> None:
    """The conversion to the 2-D ordering, for a batch of alms, eagerly and compiled."""
    L = lmax + 1
    nalm = L * (L + 1) // 2
    rng = np.random.RandomState(lmax)
    alms = rng.randn(2, nalm) + 1j * rng.randn(2, nalm)
    expected = np.stack([flm_hp_to_2d(alm, L) for alm in alms])
    np.testing.assert_array_equal(sphtfunc._alm_hp_to_2d(alms, lmax), expected)
    np.testing.assert_array_equal(jax.jit(lambda x: sphtfunc._alm_hp_to_2d(x, lmax))(alms), expected)


@requires_s2fft
@pytest.mark.usefixtures('alm_indices')
@pytest.mark.parametrize('lmax', [0, 1, 2, 7, LMAX])
def test_alm_orderings_match_s2fft(lmax: int) -> None:
    """The conversions between the alm orderings are those of s2fft."""
    from s2fft.sampling.reindex import flm_2d_to_hp_fast, flm_hp_to_2d_fast

    L = lmax + 1
    rng = np.random.RandomState(lmax)
    alm_2d = jnp.asarray(rng.randn(L, 2 * L - 1) + 1j * rng.randn(L, 2 * L - 1))
    alm_hp = sphtfunc._alm_2d_to_hp(alm_2d, lmax)
    np.testing.assert_array_equal(alm_hp, flm_2d_to_hp_fast(alm_2d, L))
    np.testing.assert_array_equal(sphtfunc._alm_hp_to_2d(alm_hp, lmax), flm_hp_to_2d_fast(alm_hp, L))


@pytest.mark.usefixtures('alm_indices')
def test_alm_2d_to_hp_transpose() -> None:
    """jax.linear_transpose can transpose the conversion to the healpy ordering."""
    L = LMAX + 1
    rng = np.random.RandomState(0)
    alm_2d = jnp.asarray(rng.randn(L, 2 * L - 1) + 1j * rng.randn(L, 2 * L - 1))
    cotangent = jnp.asarray(rng.randn(NALM) + 1j * rng.randn(NALM))

    def func(x):
        return sphtfunc._alm_2d_to_hp(x, LMAX)

    (transposed,) = jax.linear_transpose(func, alm_2d)(cotangent)
    np.testing.assert_allclose(jnp.sum(func(alm_2d) * cotangent), jnp.sum(alm_2d * transposed), rtol=1e-13)


@pytest.mark.parametrize('healpy_ordering', [False, True])
def test_almxfl_batched(healpy_ordering: bool) -> None:
    """almxfl multiplies each alm by the filter at its degree, in both orderings and for several alms."""
    L = LMAX + 1
    rng = np.random.RandomState(0)
    alms_2d = np.stack([flm_hp_to_2d(rng.randn(NALM) + 1j * rng.randn(NALM), L) for _ in range(2)])
    fl = rng.randn(L)
    alms = np.stack([flm_2d_to_hp(alm, L) for alm in alms_2d]) if healpy_ordering else alms_2d

    actual = sphtfunc.almxfl(alms, fl, healpy_ordering=healpy_ordering)

    expected = alms_2d * fl[:, None]
    if healpy_ordering:
        expected = np.stack([flm_2d_to_hp(alm, L) for alm in expected])
    np.testing.assert_allclose(actual, expected, rtol=1e-15)
