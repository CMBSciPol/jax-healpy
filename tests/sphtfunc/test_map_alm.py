from typing import Callable

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from s2fft.sampling.s2_samples import flm_2d_to_hp, flm_hp_to_2d

import jax_healpy as jhp

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('healpy_ordering', [False, True])
def test_map2alm(synthesized_map: np.ndarray, lmax: int | None, healpy_ordering: bool, nside: int) -> None:
    nside = hp.npix2nside(synthesized_map.size)
    actual_flm = jhp.map2alm(synthesized_map, lmax=lmax, iter=0, healpy_ordering=healpy_ordering)

    expected_flm = hp.map2alm(synthesized_map, lmax=lmax, iter=0)
    if not healpy_ordering:
        L = 3 * nside if lmax is None else lmax + 1
        expected_flm = flm_hp_to_2d(expected_flm, L)
    np.testing.assert_allclose(actual_flm, expected_flm, atol=1e-14)


def test_map2alm_requires_minimum_bandlimit(synthesized_map: np.ndarray, nside: int) -> None:
    """map2alm should surface s2fft limitations."""
    lmax = nside - 1  # violates 2 * nside - 1 requirement
    with pytest.raises(NotImplementedError):
        _ = jhp.map2alm(synthesized_map, lmax=lmax, iter=0, healpy_ordering=False)


# This test in particular is flaky due to alm2map accuracy variations
# even with fixed random seed. Allow reruns to reduce false positives.
@pytest.mark.flaky(reruns=5, reruns_delay=2)
@pytest.mark.parametrize('healpy_ordering', [False, True])
def test_alm2map(
    flm_generator: Callable[[...], np.ndarray], lmax: int | None, healpy_ordering: bool, nside: int
) -> None:
    if lmax is None:
        L = 3 * nside
    else:
        L = lmax + 1
    flm = flm_generator(L=L, spin=0, reality=True, healpy_ordering=healpy_ordering)
    actual_map = jhp.alm2map(flm, nside, lmax=lmax, healpy_ordering=healpy_ordering).real

    if not healpy_ordering:
        flm = flm_2d_to_hp(flm, L)
    expected_map = hp.alm2map(flm, nside, lmax=lmax, pol=False).real

    # alm2map is randomly less accurate sometimes
    np.testing.assert_allclose(actual_map, expected_map, atol=1e-14)


def test_alm2map_pixwin_not_supported(flm_generator: Callable[[...], np.ndarray], nside: int) -> None:
    L = 3 * nside
    flm = flm_generator(L=L, spin=0, reality=True, healpy_ordering=False)
    with pytest.raises(NotImplementedError):
        _ = jhp.alm2map(flm, nside, lmax=L - 1, pixwin=True, healpy_ordering=False)


def test_alm2map_mmax_requires_lmax(flm_generator: Callable[[...], np.ndarray], nside: int) -> None:
    L = 2 * nside
    flm = flm_generator(L=L, spin=0, reality=True, healpy_ordering=False)
    with pytest.raises(NotImplementedError):
        _ = jhp.alm2map(flm, nside, lmax=L - 1, mmax=L - 2, healpy_ordering=False)


@pytest.mark.parametrize('healpy_ordering', [False, True])
def test_alm2map_batched(flm_generator_batched: Callable[[...], np.ndarray], healpy_ordering: bool, nside: int) -> None:
    L = 2 * nside
    flm_generator_1, flm_generator_2 = flm_generator_batched
    flm0 = flm_generator_1(L=L, spin=0, reality=True, healpy_ordering=healpy_ordering)
    flm1 = flm_generator_2(L=L, spin=0, reality=True, healpy_ordering=healpy_ordering)
    flm = jnp.stack([flm0, flm1])  # Slightly different second map
    jhp_map = jhp.alm2map(flm, nside, lmax=L - 1, pol=False, healpy_ordering=healpy_ordering).real

    if not healpy_ordering:
        flm0 = flm_2d_to_hp(flm0, L)
        flm1 = flm_2d_to_hp(flm1, L)
    expected_map_1 = hp.alm2map(flm0, nside, lmax=L - 1, pol=False)  # healpy cannot batch alm2map with pol=False
    expected_map_2 = hp.alm2map(flm1, nside, lmax=L - 1, pol=False)
    expected_map = jnp.stack([expected_map_1, expected_map_2]).real

    np.testing.assert_allclose(jhp_map, expected_map, atol=1e-10, rtol=1e-6)


@pytest.mark.parametrize('healpy_ordering', [False, True])
def test_map2alm_batched(synthesized_map: np.ndarray, healpy_ordering: bool, nside: int) -> None:
    L = 2 * nside
    synthesized_map_1 = synthesized_map
    synthesized_map_2 = synthesized_map + 1e-2  # Slightly different second map
    synthesized_map = jnp.stack([synthesized_map_1, synthesized_map_2])
    actual_flm = jhp.map2alm(synthesized_map, lmax=L - 1, iter=0, pol=False, healpy_ordering=healpy_ordering)

    expected_flm_1 = hp.map2alm(np.array(synthesized_map_1), lmax=L - 1, iter=0, pol=False)
    expected_flm_2 = hp.map2alm(np.array(synthesized_map_2), lmax=L - 1, iter=0, pol=False)
    if not healpy_ordering:
        expected_flm_1, expected_flm_2 = flm_hp_to_2d(expected_flm_1, L), flm_hp_to_2d(expected_flm_2, L)

    expected_flm = jnp.stack([expected_flm_1, expected_flm_2])

    np.testing.assert_allclose(actual_flm, expected_flm, atol=1e-14)


@pytest.mark.parametrize('healpy_ordering', [False, True])
def test_alm2map_invalid_ndim_error(healpy_ordering) -> None:
    alms = jnp.zeros((2, 3), dtype=complex)
    with pytest.raises(ValueError):
        _ = jhp.alm2map(alms[None, None, ...], nside=1, pol=False, healpy_ordering=healpy_ordering)


@pytest.mark.parametrize('iter_val', [2, 3])
def test_map2alm_iter(synthesized_map: np.ndarray, iter_val: int, lmax: int | None) -> None:
    """Test map2alm with different iter values."""
    actual_flm = jhp.map2alm(synthesized_map, lmax=lmax, iter=iter_val, healpy_ordering=True, method='jax')
    expected_flm = hp.map2alm(synthesized_map, lmax=lmax, iter=iter_val)

    # With iterative refinement, results should be close but not identical
    # due to different implementations
    np.testing.assert_allclose(actual_flm, expected_flm, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('healpy_ordering', [False, True])
def test_almxfl(healpy_ordering) -> None:
    """Test almxfl with healpy ordering."""
    lmax = 10
    L = lmax + 1
    nalm = (lmax + 1) * (lmax + 2) // 2

    # Create alm in appropriate format based on healpy_ordering
    if healpy_ordering:
        alm = jnp.ones(nalm, dtype=jnp.complex128)
    else:
        # s2fft 2D format: shape (L, 2*L-1)
        alm = jnp.ones((L, 2 * L - 1), dtype=jnp.complex128)

    fl = jnp.arange(lmax + 1, dtype=jnp.float64)

    result = jhp.almxfl(alm, fl, healpy_ordering=healpy_ordering)

    # Test against healpy result (only for healpy ordering)
    if healpy_ordering:
        hp_result = hp.almxfl(np.array(alm), np.array(fl))
        # For healpy ordering, each alm should be multiplied by fl[l]
        # We'll check a few specific coefficients
        assert result.shape == alm.shape
        # The l=0, m=0 coefficient (index 0) should be multiplied by fl[0]=0
        np.testing.assert_allclose(result[0], 0.0)
        # Test against healpy result
        np.testing.assert_allclose(result, hp_result, atol=1e-14)
    else:
        # For s2fft 2D format, verify shape and basic behavior
        assert result.shape == alm.shape
        # The l=0, m=L-1 coefficient (center of first row) should be multiplied by fl[0]=0
        np.testing.assert_allclose(result[0, L - 1], 0.0)


def test_alm2map_smoothing_fwhm(flm_generator: Callable[[...], np.ndarray]) -> None:
    """Test alm2map with FWHM smoothing."""
    nside = 4
    lmax = 7
    L = lmax + 1
    fwhm = np.radians(10.0)  # 10 degrees

    flm = flm_generator(L=L, spin=0, reality=True, healpy_ordering=False)

    # Apply smoothing via alm2map
    map_smoothed = jhp.alm2map(flm, nside, lmax=lmax, fwhm=fwhm, healpy_ordering=False)

    # Compare with manual smoothing: almxfl + alm2map
    ell = jnp.arange(L)
    sigma = fwhm / jnp.sqrt(8.0 * jnp.log(2.0))
    bl = jnp.exp(-ell * (ell + 1) * sigma**2 / 2.0)
    flm_smoothed = jhp.almxfl(flm, bl, healpy_ordering=False)
    map_smoothed_manual = jhp.alm2map(flm_smoothed, nside, lmax=lmax, healpy_ordering=False)

    np.testing.assert_allclose(map_smoothed, map_smoothed_manual, atol=1e-14)


def test_alm2map_smoothing_sigma(flm_generator: Callable[[...], np.ndarray]) -> None:
    """Test alm2map with sigma smoothing."""
    nside = 4
    lmax = 7
    L = lmax + 1
    sigma = np.radians(5.0)  # 5 degrees

    flm = flm_generator(L=L, spin=0, reality=True, healpy_ordering=False)

    # Apply smoothing via alm2map
    map_smoothed = jhp.alm2map(flm, nside, lmax=lmax, sigma=sigma, healpy_ordering=False)

    # Compare with manual smoothing
    ell = jnp.arange(L)
    bl = jnp.exp(-ell * (ell + 1) * sigma**2 / 2.0)
    flm_smoothed = jhp.almxfl(flm, bl, healpy_ordering=False)
    map_smoothed_manual = jhp.alm2map(flm_smoothed, nside, lmax=lmax, healpy_ordering=False)

    np.testing.assert_allclose(map_smoothed, map_smoothed_manual, atol=1e-14)


# map2alm pol=True (TQU -> TEB) against healpy.
#
# Empirical agreement against hp.map2alm(pol=True) for nside in {32,64,128}, lmax in {default, 2*nside-1, 3*nside-1}:
#   alm_T: max_abs ~ 1e-14 to 1e-13 (scalar spin=0 path, essentially machine precision)
#   alm_E: max_abs ~ 5e-4 to 2e-3  (spin=2 path, inherits s2fft vs healpy spin-2 floor)
#   alm_B: max_abs ~ 3e-4 to 1e-3  (same)
# The E/B floor is the same one validated by test_map2alm_spin_basic (atol=1e-2).
@pytest.mark.parametrize('healpy_ordering', [False, True])
def test_map2alm_pol(synthesized_tqu_map: np.ndarray, lmax: int | None, healpy_ordering: bool, nside: int) -> None:
    alm_T_jax, alm_E_jax, alm_B_jax = jhp.map2alm(
        synthesized_tqu_map, lmax=lmax, iter=0, pol=True, healpy_ordering=healpy_ordering
    )

    alm_T_hp, alm_E_hp, alm_B_hp = hp.map2alm(synthesized_tqu_map, lmax=lmax, iter=0, pol=True)

    if healpy_ordering:
        expected_T, expected_E, expected_B = alm_T_hp, alm_E_hp, alm_B_hp
    else:
        L = (3 * nside) if lmax is None else (lmax + 1)
        expected_T = flm_hp_to_2d(alm_T_hp, L)
        expected_E = flm_hp_to_2d(alm_E_hp, L)
        expected_B = flm_hp_to_2d(alm_B_hp, L)

    np.testing.assert_allclose(alm_T_jax, expected_T, atol=1e-10, rtol=1e-10, err_msg='alm_T')
    np.testing.assert_allclose(alm_E_jax, expected_E, atol=5e-3, rtol=1e-8, err_msg='alm_E')
    np.testing.assert_allclose(alm_B_jax, expected_B, atol=5e-3, rtol=1e-8, err_msg='alm_B')


def test_map2alm_pol_requires_three_maps(synthesized_tqu_map: np.ndarray) -> None:
    """pol=True requires maps with shape (3, npix)."""
    # Only 2 maps
    with pytest.raises(ValueError, match=r'pol=True requires maps with shape \(3, npix\)'):
        jhp.map2alm(synthesized_tqu_map[:2], pol=True)


def test_map2alm_pol_returns_tuple(synthesized_tqu_map: np.ndarray, nside: int) -> None:
    """pol=True returns a 3-tuple of alm arrays."""
    result = jhp.map2alm(synthesized_tqu_map, lmax=3 * nside - 1, iter=0, pol=True, healpy_ordering=False)
    assert isinstance(result, tuple)
    assert len(result) == 3
    L = 3 * nside
    for alm in result:
        assert alm.shape == (L, 2 * L - 1)


@pytest.mark.parametrize('iter_val', [2, 3])
def test_map2alm_pol_iter(synthesized_tqu_map: np.ndarray, iter_val: int) -> None:
    """pol=True with iterative refinement still agrees with healpy."""
    nside = hp.npix2nside(synthesized_tqu_map.shape[-1])
    lmax = 3 * nside - 1
    alm_T_jax, alm_E_jax, alm_B_jax = jhp.map2alm(
        synthesized_tqu_map, lmax=lmax, iter=iter_val, pol=True, healpy_ordering=True
    )
    alm_T_hp, alm_E_hp, alm_B_hp = hp.map2alm(synthesized_tqu_map, lmax=lmax, iter=iter_val, pol=True)

    np.testing.assert_allclose(alm_T_jax, alm_T_hp, atol=1e-10, rtol=1e-10, err_msg='alm_T')
    np.testing.assert_allclose(alm_E_jax, alm_E_hp, atol=5e-3, rtol=1e-8, err_msg='alm_E')
    np.testing.assert_allclose(alm_B_jax, alm_B_hp, atol=5e-3, rtol=1e-8, err_msg='alm_B')


# alm2map pol=True (TEB -> TQU) against healpy.
#
# Input TEB alms are generated via hp.map2alm(pol=True) from synthesized_tqu_map
# (a healpy-synthesized TQU map). Output TQU is compared to hp.alm2map(pol=True).
# Tolerances mirror test_map2alm_pol — spin-0 (T) is essentially machine precision,
# spin-2 (Q, U) inherits the s2fft vs healpy spin-2 floor.
@pytest.mark.parametrize('healpy_ordering', [False, True])
def test_alm2map_pol(synthesized_tqu_map: np.ndarray, lmax: int | None, healpy_ordering: bool, nside: int) -> None:
    alm_T_hp, alm_E_hp, alm_B_hp = hp.map2alm(synthesized_tqu_map, lmax=lmax, iter=0, pol=True)

    if healpy_ordering:
        teb_alms = jnp.stack([jnp.asarray(alm_T_hp), jnp.asarray(alm_E_hp), jnp.asarray(alm_B_hp)])
    else:
        L = (3 * nside) if lmax is None else (lmax + 1)
        teb_alms = jnp.stack(
            [
                jnp.asarray(flm_hp_to_2d(alm_T_hp, L)),
                jnp.asarray(flm_hp_to_2d(alm_E_hp, L)),
                jnp.asarray(flm_hp_to_2d(alm_B_hp, L)),
            ]
        )

    tqu_jax = jhp.alm2map(teb_alms, nside, lmax=lmax, pol=True, healpy_ordering=healpy_ordering)
    tqu_hp = hp.alm2map([alm_T_hp, alm_E_hp, alm_B_hp], nside, lmax=lmax, pol=True)

    np.testing.assert_allclose(tqu_jax[0], tqu_hp[0], atol=1e-10, rtol=1e-10, err_msg='T_map')
    np.testing.assert_allclose(tqu_jax[1], tqu_hp[1], atol=5e-3, rtol=1e-8, err_msg='Q_map')
    np.testing.assert_allclose(tqu_jax[2], tqu_hp[2], atol=5e-3, rtol=1e-8, err_msg='U_map')


def test_alm2map_pol_returns_3xnpix(synthesized_tqu_map: np.ndarray, nside: int) -> None:
    """pol=True returns a (3, npix) array (matching hp.alm2map convention)."""
    lmax = 3 * nside - 1
    alm_T_hp, alm_E_hp, alm_B_hp = hp.map2alm(synthesized_tqu_map, lmax=lmax, iter=0, pol=True)
    teb_alms = jnp.stack([jnp.asarray(alm_T_hp), jnp.asarray(alm_E_hp), jnp.asarray(alm_B_hp)])

    tqu = jhp.alm2map(teb_alms, nside, lmax=lmax, pol=True, healpy_ordering=True)
    assert tqu.shape == (3, hp.nside2npix(nside))


def test_alm2map_pol_requires_three_alms(synthesized_tqu_map: np.ndarray, nside: int) -> None:
    """pol=True rejects inputs whose leading axis is not 3."""
    lmax = 3 * nside - 1
    alm_T_hp, alm_E_hp, _ = hp.map2alm(synthesized_tqu_map, lmax=lmax, iter=0, pol=True)
    teb_alms = jnp.stack([jnp.asarray(alm_T_hp), jnp.asarray(alm_E_hp)])  # only 2 alms

    with pytest.raises(ValueError, match=r'pol=True requires alms with shape \(3, \.\.\.\)'):
        _ = jhp.alm2map(teb_alms, nside, lmax=lmax, pol=True, healpy_ordering=True)


def test_alm2map_pol_smoothing_not_supported(synthesized_tqu_map: np.ndarray, nside: int) -> None:
    """Polarized smoothing (pol=True + fwhm/sigma) is deferred to a follow-up."""
    lmax = 3 * nside - 1
    alm_T_hp, alm_E_hp, alm_B_hp = hp.map2alm(synthesized_tqu_map, lmax=lmax, iter=0, pol=True)
    teb_alms = jnp.stack([jnp.asarray(alm_T_hp), jnp.asarray(alm_E_hp), jnp.asarray(alm_B_hp)])

    with pytest.raises(NotImplementedError, match=r'Polarized smoothing'):
        _ = jhp.alm2map(teb_alms, nside, lmax=lmax, pol=True, fwhm=np.radians(10.0), healpy_ordering=True)
