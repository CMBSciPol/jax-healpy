"""Tests for spherical harmonic transform tools (beam, smoothing, etc.)."""

from collections.abc import Callable

import healpy as hp
import jax
import numpy as np
import pytest
from s2fft.sampling.s2_samples import flm_2d_to_hp

import jax_healpy as jhp

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('fwhm_deg', [5.0, 10.0])
@pytest.mark.parametrize('lmax_val', [64, 128, 256])
def test_gauss_beam(fwhm_deg: float, lmax_val: int) -> None:
    """Test gauss_beam against healpy."""
    fwhm = np.radians(fwhm_deg)

    # Compute beam window with jax_healpy
    beam_jax = jhp.gauss_beam(fwhm, lmax=lmax_val, pol=False)

    # Compute with healpy
    beam_hp = hp.gauss_beam(fwhm, lmax=lmax_val, pol=False)

    # Should match healpy
    np.testing.assert_allclose(beam_jax, beam_hp, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize('fwhm_deg', [5.0, 10.0])
def test_smoothalm(flm_generator: Callable[[...], np.ndarray], fwhm_deg: float, nside: int) -> None:
    """Test smoothalm against healpy (s2fft 2D format only).

    Note: Only testing with healpy_ordering=False because healpy's smoothalm
    implementation differs from its almxfl implementation in ways that make
    direct comparison difficult for healpy-ordered inputs.
    """
    lmax = 2 * nside - 1
    L = lmax + 1
    fwhm = np.radians(fwhm_deg)

    # Generate alm coefficients in s2fft 2D format
    alm = flm_generator(L=L, spin=0, reality=True, healpy_ordering=False)
    hp_alm = flm_2d_to_hp(alm, L)

    # Smooth with healpy
    alm_smooth_hp = hp.smoothalm(hp_alm, fwhm=fwhm, pol=False)

    # Smooth with jax_healpy
    alm_smooth_jax = jhp.smoothalm(alm, fwhm=fwhm, pol=False, healpy_ordering=False)

    # Convert jax result to healpy ordering for comparison
    alm_smooth_jax = flm_2d_to_hp(alm_smooth_jax, L)

    # Should match healpy
    np.testing.assert_allclose(alm_smooth_jax, alm_smooth_hp, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize('fwhm_deg', [5.0, 10.0])
def test_smoothing(synthesized_map: np.ndarray, fwhm_deg: float) -> None:
    """Test smoothing against healpy."""
    fwhm = np.radians(fwhm_deg)

    # Smooth with healpy
    map_smooth_hp = hp.smoothing(synthesized_map, fwhm=fwhm, pol=False, verbose=False)

    # Smooth with jax_healpy
    map_smooth_jax = jhp.smoothing(synthesized_map, fwhm=fwhm, pol=False)

    # Should be close to healpy (some differences due to s2fft vs healpy backend)
    np.testing.assert_allclose(map_smooth_jax, map_smooth_hp, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize('fwhm_deg', [5.0, 10.0])
@pytest.mark.parametrize('lmax_val', [64, 128])
def test_gauss_beam_pol(fwhm_deg: float, lmax_val: int) -> None:
    """gauss_beam(pol=True) returns the (lmax+1, 4) [T, E, B, TE] beam matching healpy."""
    fwhm = float(np.radians(fwhm_deg))
    beam_jax = np.asarray(jhp.gauss_beam(fwhm, lmax=lmax_val, pol=True))
    beam_hp = hp.gauss_beam(fwhm, lmax=lmax_val, pol=True)
    assert beam_jax.shape == (lmax_val + 1, 4)
    np.testing.assert_allclose(beam_jax, beam_hp, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize('fwhm_deg', [5.0, 10.0])
def test_smoothalm_pol(synthesized_tqu_map: np.ndarray, fwhm_deg: float) -> None:
    """smoothalm(pol=True) applies the spin-2 beam to E/B, matching healpy exactly."""
    nside = hp.npix2nside(synthesized_tqu_map.shape[-1])
    lmax = 3 * nside - 1
    fwhm = float(np.radians(fwhm_deg))
    alm_T, alm_E, alm_B = hp.map2alm(synthesized_tqu_map, lmax=lmax, iter=0, pol=True)

    sm_hp = hp.smoothalm([alm_T.copy(), alm_E.copy(), alm_B.copy()], fwhm=fwhm, pol=True, inplace=False)
    sm_jax = np.asarray(jhp.smoothalm(np.stack([alm_T, alm_E, alm_B]), fwhm=fwhm, pol=True, healpy_ordering=True))

    np.testing.assert_allclose(sm_jax, np.asarray(sm_hp), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('fwhm_deg', [5.0])
def test_smoothing_pol(synthesized_tqu_map: np.ndarray, fwhm_deg: float) -> None:
    """smoothing(pol=True) on I,Q,U matches healpy element-wise (T exact; Q/U at the s2fft floor)."""
    nside = hp.npix2nside(synthesized_tqu_map.shape[-1])
    fwhm = float(np.radians(fwhm_deg))
    sm_hp = hp.smoothing(synthesized_tqu_map, fwhm=fwhm, pol=True, verbose=False)
    sm_jax = np.asarray(jhp.smoothing(synthesized_tqu_map, fwhm=fwhm, pol=True, iter=3))

    assert sm_jax.shape == (3, hp.nside2npix(nside))
    np.testing.assert_allclose(sm_jax[0], sm_hp[0], atol=1e-10, rtol=1e-10, err_msg='T')
    np.testing.assert_allclose(sm_jax[1], sm_hp[1], atol=5e-3, rtol=1e-6, err_msg='Q')
    np.testing.assert_allclose(sm_jax[2], sm_hp[2], atol=5e-3, rtol=1e-6, err_msg='U')


def test_pixwin_not_implemented() -> None:
    """pixwin is a stub that raises until pixel-window data files are integrated."""
    with pytest.raises(NotImplementedError):
        jhp.pixwin(64)


def test_precompute_harmonic_transforms_smoke() -> None:
    """The precompute helpers return s2fft recursion coefficients (not yet consumed by the
    transforms, but exported, so at least exercise the code path)."""
    temp = jhp.precompute_temperature_harmonic_transforms(32, lmax=63)
    assert isinstance(temp, (list, tuple)) and len(temp) > 0
    p2, pm2 = jhp.precompute_polarization_harmonic_transforms(32, lmax=63)
    assert isinstance(p2, (list, tuple)) and len(p2) > 0
    assert isinstance(pm2, (list, tuple)) and len(pm2) > 0


# --- smoothing remembers and restores masked pixels (UNSEEN/non-finite) ------------------


def test_smoothing_restores_unseen_mask(synthesized_map: np.ndarray, nside: int) -> None:
    """smoothing zeros UNSEEN before the transform and restores UNSEEN on output, like healpy."""
    fwhm = np.radians(10.0)
    lmax = 2 * nside - 1
    idx = [5, 100, synthesized_map.size - 1]
    masked = synthesized_map.copy()
    masked[idx] = jhp.UNSEEN

    out_jax = np.asarray(jhp.smoothing(masked, fwhm=fwhm, pol=False, iter=0, lmax=lmax))
    out_hp = hp.smoothing(masked, fwhm=fwhm, pol=False, iter=0, lmax=lmax)

    # Masked positions restored to UNSEEN, identical set to healpy.
    np.testing.assert_array_equal(out_jax == jhp.UNSEEN, out_hp == hp.UNSEEN)
    assert np.all(out_jax[idx] == jhp.UNSEEN)
    good = out_hp != hp.UNSEEN
    np.testing.assert_allclose(out_jax[good], out_hp[good], rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize('badval', [np.nan, np.inf, -np.inf])
def test_smoothing_restores_nonfinite_as_unseen(synthesized_map: np.ndarray, badval: float, nside: int) -> None:
    """Non-finite input pixels (extension beyond healpy) are restored as UNSEEN."""
    fwhm = np.radians(10.0)
    lmax = 2 * nside - 1
    masked = synthesized_map.copy()
    masked[7] = badval

    out = np.asarray(jhp.smoothing(masked, fwhm=fwhm, pol=False, iter=0, lmax=lmax))
    assert out[7] == jhp.UNSEEN
    good = np.ones(out.size, dtype=bool)
    good[7] = False
    assert np.all(np.isfinite(out[good]))


def test_smoothing_pol_independent_masks(synthesized_tqu_map: np.ndarray, nside: int) -> None:
    """Each of I, Q, U restores its own mask positions independently."""
    fwhm = np.radians(10.0)
    lmax = 2 * nside - 1
    iqu = np.asarray(synthesized_tqu_map).copy()
    iqu[0, 1] = jhp.UNSEEN
    iqu[1, 2] = jhp.UNSEEN
    iqu[2, 3] = np.nan

    out = np.asarray(jhp.smoothing(iqu, fwhm=fwhm, pol=True, iter=0, lmax=lmax))
    assert out.shape == iqu.shape
    assert out[0, 1] == jhp.UNSEEN and out[1, 2] == jhp.UNSEEN and out[2, 3] == jhp.UNSEEN
    # A mask on one component must not bleed into the others.
    assert out[0, 2] != jhp.UNSEEN and out[1, 1] != jhp.UNSEEN and out[2, 1] != jhp.UNSEEN


def test_smoothing_clean_map_unchanged_behavior(synthesized_map: np.ndarray, nside: int) -> None:
    """With no masked pixels, the masking logic is a no-op (output stays all-finite)."""
    fwhm = np.radians(10.0)
    lmax = 2 * nside - 1
    out = np.asarray(jhp.smoothing(synthesized_map, fwhm=fwhm, pol=False, iter=0, lmax=lmax))
    assert np.all(np.isfinite(out))
    assert not np.any(out == jhp.UNSEEN)
