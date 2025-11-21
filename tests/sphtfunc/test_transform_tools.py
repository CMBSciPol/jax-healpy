"""Tests for spherical harmonic transform tools (beam, smoothing, etc.)."""

from typing import Callable

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
    map_smooth_jax = jhp.smoothing(synthesized_map, fwhm=fwhm, pol=False, verbose=False)

    # Should be close to healpy (some differences due to s2fft vs healpy backend)
    np.testing.assert_allclose(map_smooth_jax, map_smooth_hp, rtol=1e-6, atol=1e-10)
