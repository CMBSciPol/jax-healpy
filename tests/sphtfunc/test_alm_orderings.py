"""Conversions between the healpy ordering of the alms (m >= 0, flat) and the 2-D one (all m)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_mock import MockerFixture

import jax_healpy as jhp
from jax_healpy import sphtfunc

from .helpers import METHODS, flm_2d_to_hp, flm_hp_to_2d, hp_indices, requires_s2fft

# These tests do not transform anything: they are run once, not once per default method.
pytestmark = pytest.mark.parametrize('default_method', METHODS[:1], indirect=True)

LMAX = 15
NALM = (LMAX + 1) * (LMAX + 2) // 2


@pytest.fixture(params=['stored', 'computed'])
def alm_indices(mocker: MockerFixture, request: pytest.FixtureRequest) -> str:
    """Exercise the indices of the healpy ordering both as constants and as computations."""
    if request.param == 'computed':
        mocker.patch.object(sphtfunc, '_STORED_INDICES_LMAX', -1)
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


def test_alm_hp_indices_are_stored_up_to_a_limit(mocker: MockerFixture) -> None:
    """The indices are numpy constants for the small lmax, and computed by JAX beyond."""
    mocker.patch.object(sphtfunc, '_STORED_INDICES_LMAX', 7)
    assert all(isinstance(indices, np.ndarray) for indices in sphtfunc._alm_hp_indices(7))
    assert all(isinstance(indices, jax.Array) for indices in sphtfunc._alm_hp_indices(8))


@pytest.mark.parametrize('lmax', [1024, 4607])
def test_alm_hp_indices_beyond_the_stored_ones(lmax: int, x64: bool) -> None:
    """The indices that are computed in practice, just beyond the stored ones and at a high lmax.

    At lmax = 4607, the order of 512 coefficients is off by one in 32-bit precision before its
    integer correction, which the lower values of lmax do not exercise.
    """
    assert lmax > sphtfunc._STORED_INDICES_LMAX
    expected_ell, expected_m = hp_indices(lmax + 1)
    ell, m = sphtfunc._alm_hp_indices(lmax)
    assert ell.dtype == m.dtype == np.int32
    np.testing.assert_array_equal(ell, expected_ell)
    np.testing.assert_array_equal(m, expected_m)


def test_alm_hp_indices_int32_limit() -> None:
    """The limit of the 32-bit indices is the last lmax at which their computation fits in 32 bits."""
    limit = sphtfunc._INT32_INDICES_LMAX
    assert (2 * limit + 3) ** 2 <= np.iinfo(np.int32).max < (2 * limit + 5) ** 2


@pytest.fixture
def low_int32_limit(mocker: MockerFixture) -> None:
    """Reach the limit of the 32-bit indices at lmax = 7, with computed indices."""
    mocker.patch.object(sphtfunc, '_STORED_INDICES_LMAX', -1)
    mocker.patch.object(sphtfunc, '_INT32_INDICES_LMAX', 7)


@pytest.mark.usefixtures('low_int32_limit')
def test_alm_hp_indices_beyond_the_int32_limit() -> None:
    """Beyond the limit of the 32-bit indices, they are computed with 64-bit integers."""
    expected_ell, expected_m = hp_indices(LMAX + 1)
    with jax.enable_x64(True):
        assert sphtfunc._alm_hp_indices(7)[0].dtype == np.int32
        ell, m = sphtfunc._alm_hp_indices(LMAX)
    assert ell.dtype == m.dtype == np.int64
    np.testing.assert_array_equal(ell, expected_ell)
    np.testing.assert_array_equal(m, expected_m)


@pytest.mark.usefixtures('low_int32_limit')
def test_alm_hp_indices_beyond_the_int32_limit_without_x64() -> None:
    """Without 64-bit precision, the square root would overflow: the orders are repeated instead."""
    expected_ell, expected_m = hp_indices(LMAX + 1)
    with jax.enable_x64(False):
        ell, m = sphtfunc._alm_hp_indices(LMAX)
    assert ell.dtype == m.dtype == np.int32
    np.testing.assert_array_equal(ell, expected_ell)
    np.testing.assert_array_equal(m, expected_m)


def test_alm_int32_limits() -> None:
    """The limits of the 32-bit indices are the last lmax at which they do not overflow."""
    int32_max = np.iinfo(np.int32).max
    limit = sphtfunc._INT32_ALMS_LMAX
    assert (limit + 1) * (limit + 2) // 2 - 1 <= int32_max < (limit + 2) * (limit + 3) // 2 - 1
    limit = sphtfunc._INT32_FLAT_INDICES_LMAX
    assert (limit + 1) ** 2 - 1 <= int32_max < (limit + 2) ** 2 - 1


@pytest.fixture
def low_int32_alms_limit(mocker: MockerFixture) -> None:
    """Reach the limit of the 32-bit indices of the alms at lmax = 7, with computed indices."""
    mocker.patch.object(sphtfunc, '_STORED_INDICES_LMAX', -1)
    mocker.patch.object(sphtfunc, '_INT32_INDICES_LMAX', 3)
    mocker.patch.object(sphtfunc, '_INT32_ALMS_LMAX', 7)


@pytest.mark.usefixtures('low_int32_alms_limit')
def test_alms_beyond_the_int32_limit_require_x64() -> None:
    """Beyond the limit of the 32-bit indices, 32-bit precision is refused instead of overflowing."""
    alms = jnp.zeros(NALM, dtype=jnp.complex64)
    with jax.enable_x64(False):
        assert sphtfunc._alm_hp_indices(7)[0].dtype == np.int32
        with pytest.raises(ValueError, match='lmax=8 cannot be indexed with 32-bit integers.*enable 64-bit precision'):
            sphtfunc._alm_hp_indices(8)
        with pytest.raises(ValueError, match='lmax=15 cannot be indexed with 32-bit integers'):
            sphtfunc._alm_hp_to_2d(alms, LMAX)


@pytest.mark.usefixtures('low_int32_alms_limit')
def test_alm_hp_to_2d_beyond_the_int32_limit() -> None:
    """Beyond the limit of the 32-bit indices, the conversion uses 64-bit integers."""
    rng = np.random.RandomState(0)
    alms = rng.randn(NALM) + 1j * rng.randn(NALM)
    with jax.enable_x64(True):
        actual = sphtfunc._alm_hp_to_2d(alms, LMAX)
    np.testing.assert_array_equal(actual, flm_hp_to_2d(alms, LMAX + 1))


@pytest.mark.parametrize('x64', [False, True])
def test_alm_2d_to_hp_beyond_the_flat_int32_limit(mocker: MockerFixture, x64: bool) -> None:
    """Beyond the limit of the 32-bit flat indices, the 2-D alms are read with two indices."""
    mocker.patch.object(sphtfunc, '_STORED_INDICES_LMAX', -1)
    mocker.patch.object(sphtfunc, '_INT32_FLAT_INDICES_LMAX', 7)
    L = LMAX + 1
    rng = np.random.RandomState(0)
    alms = (rng.randn(2, L, 2 * L - 1) + 1j * rng.randn(2, L, 2 * L - 1)).astype(np.complex64)
    with jax.enable_x64(x64):
        actual = sphtfunc._alm_2d_to_hp(alms, LMAX)
    np.testing.assert_array_equal(actual, np.stack([flm_2d_to_hp(alm, L) for alm in alms]))


@pytest.mark.usefixtures('low_int32_limit')
def test_almxfl_beyond_the_int32_limit_without_x64() -> None:
    """almxfl in the healpy ordering works without 64-bit precision beyond the square root limit."""
    rng = np.random.RandomState(0)
    alms = (rng.randn(NALM) + 1j * rng.randn(NALM)).astype(np.complex64)
    fl = rng.randn(LMAX + 1).astype(np.float32)
    ell, _ = hp_indices(LMAX + 1)
    with jax.enable_x64(False):
        actual = jhp.almxfl(alms, fl, healpy_ordering=True)
    np.testing.assert_allclose(actual, alms * fl[ell], rtol=1e-6)


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
