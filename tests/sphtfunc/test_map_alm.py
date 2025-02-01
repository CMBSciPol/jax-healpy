from pathlib import Path
from typing import Callable

import healpy as hp
import numpy as np
import pytest
from s2fft.sampling.s2_samples import flm_2d_to_hp
from s2fft.sampling.reindex import flm_hp_to_2d_fast
import jax_healpy as jhp

# TODO: '_map2alm_pol',
# '_alm2map_pol',
# 'almxfl',
# 'alm2cl',
# 'synalm',
# 'synfast'

# TODO: test that precompute is faster?     'precompute_temperature_harmonic_transforms',
# 'precompute_polarization_harmonic_transforms',

data_path = Path(jhp.__file__).parent.parent / 'tests/data'


@pytest.fixture(scope='session')
def cla(data_path: Path) -> np.ndarray:
    return hp.read_cl(data_path / 'cl_wmap_band_iqumap_r9_7yr_W_v4_udgraded32_II_lmax64_rmmono_3iter.fits')


@pytest.fixture(scope='session')
def map1(data_path: Path) -> np.ndarray:
    return hp.read_map(data_path / 'wmap_band_iqumap_r9_7yr_V_v4_udgraded32.fits', field=(0, 1, 2))


@pytest.fixture(scope='session')
def map2(data_path: Path) -> np.ndarray:
    return hp.read_map(data_path / 'wmap_band_iqumap_r9_7yr_W_v4_udgraded32.fits', field=(0, 1, 2))


@pytest.fixture(scope='session')
def synthesized_map(cla: np.ndarray) -> np.ndarray:
    nside = 32
    lmax = 64
    fwhm_deg = 7.0
    seed = 12345
    np.random.seed(seed)
    return hp.synfast(
        cla,
        nside,
        lmax=lmax,
        pixwin=False,
        fwhm=np.radians(fwhm_deg),
        new=False,
    )


@pytest.mark.parametrize('fast_non_differentiable', [False, True])
@pytest.mark.parametrize('healpy_ordering', [False, True])
@pytest.mark.parametrize('lmax', [None, 64])
def test_map2alm(
    synthesized_map: np.ndarray, lmax: int | None, healpy_ordering: bool, fast_non_differentiable: bool
) -> None:
    nside = hp.npix2nside(synthesized_map.size)
    actual_flm = jhp.map2alm(
        synthesized_map,
        lmax=lmax,
        iter=0,
        healpy_ordering=healpy_ordering,
        fast_non_differentiable=fast_non_differentiable,
    )

    expected_flm = hp.map2alm(synthesized_map, lmax=lmax, iter=0)
    if not healpy_ordering:
        L = 3 * nside if lmax is None else lmax + 1
        expected_flm = flm_hp_to_2d_fast(expected_flm, L)
    np.testing.assert_allclose(actual_flm, expected_flm, atol=1e-14)


# TODO: Wait for s2fft response from the issue https://github.com/astro-informatics/s2fft/issues/269
# @pytest.mark.parametrize('fast_non_differentiable', [False, True])
# @pytest.mark.parametrize('healpy_ordering', [False, True])
# @pytest.mark.parametrize('lmax', [None, 64])
# def test_map2alm_pol(map1: np.ndarray, lmax: int | None, healpy_ordering: bool, fast_non_differentiable: bool) -> None:

#     nside = hp.npix2nside(map1.shape[1])
#     actual_flm = jnp.array(jhp.map2alm(map1, lmax=lmax, iter=0, pol=True, healpy_ordering=healpy_ordering, fast_non_differentiable=fast_non_differentiable))

#     expected_flm = np.array(hp.map2alm(map1, lmax=lmax, iter=0))
#     if not healpy_ordering:
#         L = 3 * nside if lmax is None else lmax + 1
#         lmax = L - 1

#         assert expected_flm.shape[0] == 3
#         assert expected_flm.shape[1] ==  lmax * (2 * lmax + 1 - lmax) // 2 + lmax + 1


#         # expected_flm = flm_hp_to_2d(expected_flm, L)
#         array_flm = jnp.array(expected_flm)
#         result_flm = jnp.zeros_like(actual_flm)
#         for i in range(array_flm.shape[0]):
#             result_flm = result_flm.at[i,...].set(flm_hp_to_2d_fast(array_flm[i], L))
#         expected_flm = result_flm
#     np.testing.assert_allclose(actual_flm, expected_flm, atol=1e-8)


@pytest.mark.parametrize('lmax', [None, 7])
@pytest.mark.parametrize('healpy_ordering', [False, True])
@pytest.mark.parametrize('fast_non_differentiable', [False, True])
def test_alm2map(
    flm_generator: Callable[[...], np.ndarray], lmax: int | None, healpy_ordering: bool, fast_non_differentiable: bool
) -> None:
    nside = 4
    if lmax is None:
        L = 3 * nside
    else:
        L = lmax + 1
    flm = flm_generator(L=L, spin=0, reality=True, healpy_ordering=healpy_ordering)
    # alm_input = hp.map2alm(map1[0], lmax=L - 1)
    # if not healpy_ordering:
    #     flm = flm_hp_to_2d_fast(alm_input, L)
    # else:
    #     flm = alm_input

    actual_map = jhp.alm2map(
        flm,
        nside,
        lmax=lmax,
        pol=False,
        healpy_ordering=healpy_ordering,
        fast_non_differentiable=fast_non_differentiable,
    ).block_until_ready()

    if not healpy_ordering:
        flm = flm_2d_to_hp(flm, L)
    expected_map = hp.alm2map(flm, nside, lmax=lmax, pol=False)

    np.testing.assert_allclose(actual_map, expected_map, atol=1e-12)

    # TODO: Wait for s2fft response from the issue https://github.com/astro-informatics/s2fft/issues/269
    # @pytest.mark.parametrize('lmax', [None, 7])
    # @pytest.mark.parametrize('healpy_ordering', [False, True])
    # @pytest.mark.parametrize('fast_non_differentiable', [False, True])
    # def test_alm2map_TEB(map1: Callable[[...], np.ndarray], lmax: int | None, healpy_ordering: bool, fast_non_differentiable: bool) -> None:
    #     nside = 4
    #     if lmax is None:
    #         L = 3 * nside
    #     else:
    #         L = lmax + 1

    #     # flm_list = []
    #     # for i in range(3):
    #     #     flm_list.append(flm_generator(L=L, spin=0, reality=True, healpy_ordering=healpy_ordering))

    #     alm_hp = hp.map2alm(map1, lmax=L - 1, pol=True, iter=3)

    #     if not healpy_ordering:
    #         flm_list = []
    #         for i in range(3):
    #             flm_list.append(flm_hp_to_2d_fast(alm_hp[i], L))
    #     else:
    #         flm_list = alm_hp

    #     flm = jnp.array(flm_list)
    #     actual_map = jhp.alm2map(flm, nside, lmax=lmax, pol=True, healpy_ordering=healpy_ordering, fast_non_differentiable=fast_non_differentiable).block_until_ready()

    #     if not healpy_ordering:
    #         flm_list_hp = []
    #         for i in range(3):
    #             flm_list_hp.append(flm_2d_to_hp_fast(flm[i], L))
    #         flm = np.array(flm_list_hp)

    #     expected_map = hp.alm2map(np.array(flm), nside, lmax=lmax, pol=True)

    #     np.testing.assert_allclose(actual_map, expected_map, atol=1e-10)

    # @pytest.mark.parametrize('healpy_ordering', [False, True])
    # @pytest.mark.parametrize('fast_non_differentiable', [False, True])
    # def test_alm2map_batched(flm_generator: Callable[[...], np.ndarray], healpy_ordering: bool, fast_non_differentiable: bool) -> None:
    #     nside = 4
    #     L = 2 * nside
    #     flm0 = flm_generator(L=L, spin=0, reality=True, healpy_ordering=healpy_ordering)
    #     flm = jnp.stack([flm0, flm0])
    # actual_map = jhp.alm2map(
    #     flm,
    #     nside,
    #     lmax=L - 1,
    #     pol=False,
    #     healpy_ordering=healpy_ordering,
    #     fast_non_differentiable=fast_non_differentiable,
    # )


#     if not healpy_ordering:
#         flm0 = flm_2d_to_hp(flm0, L)
#     expected_map0 = hp.alm2map(flm0, nside, lmax=L - 1, pol=False)  # healpy cannot batch alm2map with pol=False
#     expected_map = np.stack([expected_map0, expected_map0])

#     np.testing.assert_allclose(actual_map, expected_map, atol=1e-14)


# @pytest.mark.parametrize('healpy_ordering', [False, True])
# @pytest.mark.parametrize('fast_non_differentiable', [False, True])
# def test_map2alm_batched(synthesized_map: np.ndarray, healpy_ordering: bool, fast_non_differentiable: bool) -> None:
#     nside = hp.npix2nside(synthesized_map.size)
#     L = 2 * nside
#     synthesized_map = jnp.stack([synthesized_map, synthesized_map])
#     actual_flm = jhp.map2alm(synthesized_map, lmax=L - 1, iter=0, pol=False, healpy_ordering=healpy_ordering)

#     expected_flm = hp.map2alm(np.array(synthesized_map), lmax=L - 1, iter=0, pol=False)
#     if not healpy_ordering:
#         expected_flm = jnp.stack([flm_hp_to_2d(expected_flm[0], L), flm_hp_to_2d(expected_flm[1], L)])

#     np.testing.assert_allclose(actual_flm, expected_flm, atol=1e-14)


# def test_alm2map_scalar_error() -> None:
#     with pytest.raises(ValueError):
#         _ = jhp.alm2map(jnp.array(0.0 + 0j), nside=1)


# def test_map2alm_scalar_error() -> None:
#     with pytest.raises(ValueError):
#         _ = jhp.map2alm(jnp.array(0.0), iter=0)


# def test_alm2map_invalid_ndim_error() -> None:
#     alms = jnp.zeros((2, 3), dtype=complex)
#     with pytest.raises(ValueError):
#         _ = jhp.alm2map(alms[None, None, ...], nside=1, pol=False, healpy_ordering=False)


# def test_alm2map_invalid_ndim_healpy_ordering_error() -> None:
#     alms = jnp.zeros(4, dtype=complex)
#     with pytest.raises(ValueError):
#         _ = jhp.alm2map(alms[None, None, ...], nside=1, pol=False, healpy_ordering=True)


# def test_map2alm_invalid_ndim_error() -> None:
#     with pytest.raises(ValueError):
#         _ = jhp.map2alm(jnp.array([[[0.0]]]), iter=0)
