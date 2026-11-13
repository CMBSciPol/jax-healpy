"""Tests of the JAX primitives that wrap the transforms of ducc0."""

import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_mock import MockerFixture

pytest.importorskip('ducc0')

from jax_healpy import _ducc0

# These tests do not depend on the default transform backend: they are run once.
pytestmark = pytest.mark.parametrize('default_method', ['ducc0'], indirect=True)

NSIDE = 8
LMAX = 3 * NSIDE - 1
NPIX = 12 * NSIDE**2
NALM = (LMAX + 1) * (LMAX + 2) // 2
NREAL = (LMAX + 1) ** 2
SPINS = [0, 1, 2, 3]


def random_alms(ncomp: int, spin: int = 0, seed: int = 0) -> np.ndarray:
    """Random alms of real fields in the healpy ordering: real for m = 0, zero for l < spin."""
    rng = np.random.RandomState(seed)
    alms = rng.randn(ncomp, NALM) + 1j * rng.randn(ncomp, NALM)
    alms[:, : LMAX + 1] = alms[:, : LMAX + 1].real
    ell = hp.Alm.getlm(LMAX)[0]
    alms[:, ell < spin] = 0
    return alms


def transforms(spin: int) -> tuple[Callable, Callable, jax.Array, jax.Array, np.ndarray]:
    """The two primitives for a spin, and operands for them."""
    ncomp = 1 if spin == 0 else 2
    keywords = {'nside': NSIDE, 'lmax': LMAX, 'spin': spin}
    alms = random_alms(ncomp, spin)
    x = _ducc0.alm_to_real(jnp.asarray(alms), LMAX)
    maps = jnp.asarray(np.random.RandomState(1).randn(ncomp, NPIX))

    def synthesis(x):
        return _ducc0.synthesis(x, **keywords)

    def adjoint_synthesis(maps):
        return _ducc0.adjoint_synthesis(maps, **keywords)

    return synthesis, adjoint_synthesis, x, maps, alms


def test_real_representation() -> None:
    alms = jnp.asarray(random_alms(2))
    x = _ducc0.alm_to_real(alms, LMAX)
    assert x.shape == (2, NREAL)
    assert x.dtype == jnp.float64
    np.testing.assert_allclose(_ducc0.real_to_alm(x, LMAX), alms, rtol=1e-15)
    # all the coefficients have the same weight: the norm is that of the alms over all m
    norm2 = jnp.sum(jnp.abs(alms[:, : LMAX + 1]) ** 2) + 2 * jnp.sum(jnp.abs(alms[:, LMAX + 1 :]) ** 2)
    np.testing.assert_allclose(jnp.sum(x**2), norm2, rtol=1e-14)


@pytest.mark.parametrize('spin', SPINS)
def test_synthesis_matches_healpy(spin: int) -> None:
    synthesis, _, x, _, alms = transforms(spin)
    if spin == 0:
        expected = hp.alm2map(alms[0], NSIDE, lmax=LMAX)[None]
    else:
        expected = np.stack(hp.alm2map_spin(list(alms), NSIDE, spin, LMAX))
    np.testing.assert_allclose(synthesis(x), expected, atol=1e-11, rtol=0)


@pytest.mark.parametrize('spin', SPINS)
def test_adjoint_synthesis_matches_healpy(spin: int) -> None:
    """With the weight of the pixels, the transpose of the synthesis is map2alm without iterations."""
    _, adjoint_synthesis, _, maps, _ = transforms(spin)
    if spin == 0:
        expected = hp.map2alm(np.asarray(maps[0]), lmax=LMAX, iter=0)[None]
    else:
        expected = np.stack(hp.map2alm_spin(list(np.asarray(maps)), spin, lmax=LMAX))
    actual = _ducc0.real_to_alm(4 * np.pi / NPIX * adjoint_synthesis(maps), LMAX)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)


@pytest.mark.parametrize('spin', SPINS)
def test_primitives_are_transposes(spin: int) -> None:
    synthesis, adjoint_synthesis, x, maps, _ = transforms(spin)
    np.testing.assert_allclose(jnp.vdot(synthesis(x), maps), jnp.vdot(x, adjoint_synthesis(maps)), rtol=1e-12)
    np.testing.assert_array_equal(jax.linear_transpose(synthesis, x)(maps)[0], adjoint_synthesis(maps))
    np.testing.assert_array_equal(jax.linear_transpose(adjoint_synthesis, maps)(x)[0], synthesis(x))


@pytest.mark.parametrize('spin', [0, 2])
def test_primitives_jit(spin: int) -> None:
    synthesis, adjoint_synthesis, x, maps, _ = transforms(spin)
    np.testing.assert_array_equal(jax.jit(synthesis)(x), synthesis(x))
    np.testing.assert_array_equal(jax.jit(adjoint_synthesis)(maps), adjoint_synthesis(maps))


@pytest.mark.parametrize('spin', [0, 2])
@pytest.mark.parametrize('sequential', [False, True])
def test_primitives_vmap(mocker: MockerFixture, spin: int, sequential: bool) -> None:
    """Batches are transformed in a single call to ducc0 for small maps, one map at a time for large ones."""
    mocker.patch.object(_ducc0, '_SEQUENTIAL_BATCH_NSIDE', NSIDE if sequential else 2 * NSIDE)
    synthesis, adjoint_synthesis, x, maps, _ = transforms(spin)
    xs = jnp.stack([x, 2 * x, 3 * x])
    expected = synthesis(x)

    np.testing.assert_allclose(jax.vmap(synthesis)(xs)[2], 3 * expected, rtol=1e-13, atol=1e-13)
    nested = jax.jit(jax.vmap(jax.vmap(synthesis)))(jnp.stack([xs, -xs]))
    assert nested.shape == (2, 3, *expected.shape)
    np.testing.assert_allclose(nested[1, 1], -2 * expected, rtol=1e-13, atol=1e-13)
    last = jax.vmap(synthesis, in_axes=2, out_axes=2)(jnp.moveaxis(xs, 0, 2))
    np.testing.assert_allclose(last[..., 1], 2 * expected, rtol=1e-13, atol=1e-13)

    batched = jax.vmap(adjoint_synthesis)(jnp.stack([maps, 2 * maps]))
    np.testing.assert_allclose(batched[1], 2 * adjoint_synthesis(maps), rtol=1e-13, atol=1e-13)
    assert jax.vmap(synthesis)(xs[:0]).shape == (0, *expected.shape)


@pytest.mark.parametrize('spin', [0, 2])
@pytest.mark.parametrize('sequential', [False, True])
def test_primitives_derivatives(mocker: MockerFixture, spin: int, sequential: bool) -> None:
    mocker.patch.object(_ducc0, '_SEQUENTIAL_BATCH_NSIDE', NSIDE if sequential else 2 * NSIDE)
    synthesis, adjoint_synthesis, x, _, _ = transforms(spin)
    normal = adjoint_synthesis(synthesis(x))

    def loss(x):
        return 0.5 * jnp.sum(synthesis(x) ** 2)

    np.testing.assert_array_equal(jax.jvp(synthesis, (x,), (2 * x,))[1], 2 * synthesis(x))
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(x), normal, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(jax.vmap(jax.grad(loss))(jnp.stack([x, 2 * x]))[1], 2 * normal, rtol=1e-13, atol=1e-13)
    # the Hessian of the loss is S^T S
    np.testing.assert_allclose(jax.jvp(jax.grad(loss), (x,), (x,))[1], normal, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('spin', [0, 2])
def test_primitives_single_precision(spin: int) -> None:
    synthesis, adjoint_synthesis, x, maps, _ = transforms(spin)
    out = synthesis(x.astype(jnp.float32))
    assert out.dtype == jnp.float32
    np.testing.assert_allclose(out, synthesis(x), atol=1e-4)
    out = adjoint_synthesis(maps.astype(jnp.float32))
    assert out.dtype == jnp.float32
    np.testing.assert_allclose(out, adjoint_synthesis(maps), atol=1e-3)


@pytest.mark.parametrize(
    'shape, dtype, spin, match',
    [
        ((NREAL,), float, 0, r'shape \(\.\.\., 1, 576\) for spin 0, got \(576,\)'),
        ((2, NREAL), float, 0, r'shape \(\.\.\., 1, 576\) for spin 0, got \(2, 576\)'),
        ((1, NREAL), float, 2, r'shape \(\.\.\., 2, 576\) for spin 2, got \(1, 576\)'),
        ((1, NALM), float, 0, r'shape \(\.\.\., 1, 576\) for spin 0, got \(1, 300\)'),
        ((1, NREAL), complex, 0, 'expects a real operand, got complex128'),
    ],
)
def test_primitives_invalid_operand(shape: tuple[int, ...], dtype: type, spin: int, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        _ducc0.synthesis(jnp.zeros(shape, dtype=dtype), nside=NSIDE, lmax=LMAX, spin=spin)


@pytest.fixture
def thread_count(mocker: MockerFixture) -> Iterator[Callable[[], int]]:
    """The uncached number of threads, without the environment variables that set it."""
    mocker.patch.dict(os.environ)
    os.environ.pop('DUCC0_NUM_THREADS', None)
    os.environ.pop('OMP_NUM_THREADS', None)
    _ducc0._thread_count.cache_clear()
    yield _ducc0._thread_count
    _ducc0._thread_count.cache_clear()


def test_thread_count_is_the_physical_core_count(mocker: MockerFixture, thread_count: Callable[[], int]) -> None:
    mocker.patch.object(_ducc0, '_physical_core_count', return_value=3)
    assert thread_count() == 3


@pytest.mark.parametrize('variable', ['DUCC0_NUM_THREADS', 'OMP_NUM_THREADS'])
def test_thread_count_set_by_the_environment(
    mocker: MockerFixture, thread_count: Callable[[], int], variable: str
) -> None:
    """The environment variables size the pool of ducc0, which is then used whole."""
    mocker.patch.object(_ducc0, '_physical_core_count', return_value=3)
    mocker.patch.dict(os.environ, {variable: '2'})
    assert thread_count() == 0


def test_thread_count_without_topology(mocker: MockerFixture, thread_count: Callable[[], int]) -> None:
    """Without the CPU topology, ducc0 uses all the threads of its pool."""
    mocker.patch.object(_ducc0, '_physical_core_count', return_value=None)
    assert thread_count() == 0


def test_physical_core_count() -> None:
    count = _ducc0._physical_core_count()
    if count is not None:
        assert 1 <= count <= len(os.sched_getaffinity(0))


def test_is_available_with_a_broken_ducc0(mocker: MockerFixture, tmp_path: Path) -> None:
    """A ducc0 that cannot be imported is reported and not used."""
    package = tmp_path / 'ducc0'
    package.mkdir()
    (package / '__init__.py').write_text("raise ImportError('built for another processor')")
    mocker.patch.object(sys, 'path', [str(tmp_path), *sys.path])
    mocker.patch.dict(sys.modules)
    sys.modules.pop('ducc0', None)
    _ducc0.is_available.cache_clear()
    try:
        with pytest.warns(UserWarning, match='ducc0 is installed but cannot be imported.*another processor'):
            assert not _ducc0.is_available()
    finally:
        _ducc0.is_available.cache_clear()
