from collections.abc import Callable

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest
from s2fft.sampling.reindex import flm_hp_to_2d_fast

import jax_healpy as jhp


def test_map2alm_spin_iter(nside: int, lmax: int) -> None:
    """map2alm_spin exposes an iter option (unlike healpy); it runs and changes the result."""
    if lmax is None:
        lmax = 3 * nside - 1
    npix = hp.nside2npix(nside)
    rng = np.random.RandomState(7)
    q = rng.randn(npix)
    u = rng.randn(npix)

    e0, b0 = jhp.map2alm_spin([q, u], spin=2, lmax=lmax, iter=0)
    e3, b3 = jhp.map2alm_spin([q, u], spin=2, lmax=lmax, iter=3)

    assert np.all(np.isfinite(np.asarray(e3))) and np.all(np.isfinite(np.asarray(b3)))
    assert not np.allclose(np.asarray(e0), np.asarray(e3))

    # iter=0 must match healpy's (non-iterated) map2alm_spin, in the default s2fft layout
    # (atol is the documented spin-2 alm floor ~0.01, so a real E/B error would be caught).
    # nside=16 is a coarse-resolution smoke case: the s2fft<->healpy spin-transform
    # difference is intrinsically larger there (measured ~0.019), so it gets a looser
    # atol while nside>=32 keeps the tight ~0.01 floor.
    atol = 4e-2 if nside < 32 else 1e-2
    eh, bh = hp.map2alm_spin([q, u], 2, lmax=lmax)
    L = lmax + 1
    np.testing.assert_allclose(np.asarray(e0), flm_hp_to_2d_fast(eh, L), atol=atol, rtol=1e-8)
    np.testing.assert_allclose(np.asarray(b0), flm_hp_to_2d_fast(bh, L), atol=atol, rtol=1e-8)


@pytest.mark.parametrize('healpy_ordering', [True, False])
def test_map2alm_spin_basic(
    flm_generator: Callable[[...], np.ndarray], nside: int, lmax: int, healpy_ordering: bool
) -> None:
    """Test basic map2alm_spin with spin=2 against healpy."""
    if lmax is None:
        lmax = 3 * nside - 1
    L = lmax + 1
    spin = 2
    npix = hp.nside2npix(nside)

    # Generate random Q and U maps (not from alm2map_spin to avoid roundtrip issues)
    np.random.seed(42)
    map1 = np.random.randn(npix)
    map2 = np.random.randn(npix)

    # Apply spin transform with jax_healpy
    result_jax = jhp.map2alm_spin([map1, map2], spin=spin, lmax=lmax, healpy_ordering=healpy_ordering)

    # Apply spin transform with healpy
    result_hp = hp.map2alm_spin([map1, map2], spin, lmax=lmax)

    # Compare results
    assert isinstance(result_jax, list)
    assert len(result_jax) == 2

    # nside=16 is a coarse-resolution smoke case: the s2fft<->healpy spin-transform floor
    # is intrinsically larger there (measured MSE ~3e-6, max-abs ~0.025), so loosen its
    # tolerances. nside>=32 keeps the tight values validated below.
    mse_tol = 1e-5 if nside < 32 else 1e-6
    atol = 4e-2 if nside < 32 else 1e-2

    # Check against healpy
    if healpy_ordering:
        # Both should be 1D healpy format
        nalm = hp.Alm.getsize(lmax)
        assert result_jax[0].shape == (nalm,)
        assert result_jax[1].shape == (nalm,)
        # Tolerances based on validation (max errors: ~0.007 abs, ~0.066 rel)
        # Using random maps instead of alm2map_spin output to avoid roundtrip incompatibilities
        MSE_Q = np.mean(np.abs(result_jax[0] - result_hp[0]) ** 2)
        MSE_U = np.mean(np.abs(result_jax[1] - result_hp[1]) ** 2)
        assert MSE_Q < mse_tol, f'MSE Q too high: {MSE_Q}'
        assert MSE_U < mse_tol, f'MSE U too high: {MSE_U}'
        np.testing.assert_allclose(result_jax[0], result_hp[0], atol=atol, rtol=1e-8)
        np.testing.assert_allclose(result_jax[1], result_hp[1], atol=atol, rtol=1e-8)
    else:
        # jax result should be 2D s2fft format
        assert result_jax[0].shape == (L, 2 * L - 1)
        assert result_jax[1].shape == (L, 2 * L - 1)

        # Convert healpy result to s2fft format for comparison
        result_hp_2d_0 = flm_hp_to_2d_fast(result_hp[0], L)
        result_hp_2d_1 = flm_hp_to_2d_fast(result_hp[1], L)
        # Tolerances based on validation (max errors: ~0.007 abs, ~0.066 rel)
        # Using random maps instead of alm2map_spin output to avoid roundtrip incompatibilities
        MSE_Q = np.mean(np.abs(result_jax[0] - result_hp_2d_0) ** 2)
        MSE_U = np.mean(np.abs(result_jax[1] - result_hp_2d_1) ** 2)
        assert MSE_Q < mse_tol, f'MSE Q too high: {MSE_Q}'
        assert MSE_U < mse_tol, f'MSE U too high: {MSE_U}'
        np.testing.assert_allclose(result_jax[0], result_hp_2d_0, atol=atol, rtol=1e-8)
        np.testing.assert_allclose(result_jax[1], result_hp_2d_1, atol=atol, rtol=1e-8)


@pytest.mark.parametrize('healpy_ordering', [True, False])
def test_alm2map_spin_basic(flm_generator: Callable[[...], np.ndarray], nside: int, lmax: int, healpy_ordering) -> None:
    """Test basic alm2map_spin with spin=2 against healpy.

    Uses a roundtrip approach: generate random Q/U maps, convert to alms with healpy,
    then convert back to maps with both healpy and jax_healpy and compare.
    This avoids issues with s2fft-generated alms that may not match healpy conventions.
    """
    if lmax is None:
        lmax = 3 * nside - 1
    L = lmax + 1
    spin = 2
    npix = hp.nside2npix(nside)

    # Generate random Q and U maps (approach similar to test_map2alm_spin_basic)
    np.random.seed(42)
    map_Q = np.random.randn(npix)
    map_U = np.random.randn(npix)

    # Convert to alms using healpy to get proper test data
    alm_E_hp, alm_B_hp = hp.map2alm_spin([map_Q, map_U], spin, lmax=lmax)

    # Convert alms to test format if needed
    if healpy_ordering:
        alm_E_test = alm_E_hp
        alm_B_test = alm_B_hp
    else:
        # Convert to s2fft 2D format for jax_healpy
        alm_E_test = flm_hp_to_2d_fast(alm_E_hp, L)
        alm_B_test = flm_hp_to_2d_fast(alm_B_hp, L)

    # Apply inverse spin transform with jax_healpy
    result_jax = jhp.alm2map_spin(
        [alm_E_test, alm_B_test], nside, spin=spin, lmax=lmax, healpy_ordering=healpy_ordering
    )

    # Apply inverse spin transform with healpy
    result_hp = hp.alm2map_spin([alm_E_hp, alm_B_hp], nside, spin=spin, lmax=lmax)

    # Compare results using MSE metric
    # Note: Tolerance accounts for numerical roundtrip error through healpy's map2alm_spin
    # MSE is more robust than pointwise comparison for handling near-zero pixels
    # nside=16 is a coarse-resolution smoke case with an intrinsically larger floor
    # (measured MSE ~2.1e-4, max-abs ~0.16); nside>=32 keeps the tight values below.
    mse_tol = 5e-4 if nside < 32 else 1e-4
    atol = 0.25 if nside < 32 else 0.15
    MSE_Q = np.mean(np.abs(result_jax[0] - result_hp[0]) ** 2)
    MSE_U = np.mean(np.abs(result_jax[1] - result_hp[1]) ** 2)
    assert MSE_Q < mse_tol, f'MSE Q too high: {MSE_Q}'
    assert MSE_U < mse_tol, f'MSE U too high: {MSE_U}'
    # Pointwise atol sits just above the measured map-error floor: the single-transform
    # s2fft alm2map_spin differs from healpy's C++ by up to ~0.13 (spin 2) at the worst
    # nside/lmax (measured), so atol=0.15 (~1.15x floor) is meaningful, not slack.
    np.testing.assert_allclose(result_jax[0], result_hp[0], atol=atol, rtol=1e-8)
    np.testing.assert_allclose(result_jax[1], result_hp[1], atol=atol, rtol=1e-8)


def test_map2alm_spin_validation(nside: int, lmax: int) -> None:
    """Test that map2alm_spin validates input correctly."""
    if lmax is None:
        lmax = 3 * nside - 1
    npix = hp.nside2npix(nside)

    # For spin != 0, should require list of 2 maps
    with pytest.raises(ValueError, match='must be a list/tuple of 2 arrays'):
        jhp.map2alm_spin(np.random.randn(npix), spin=2, lmax=lmax)

    # Wrong number of maps
    with pytest.raises(ValueError, match='must be a list/tuple of 2 arrays'):
        jhp.map2alm_spin([np.random.randn(npix)], spin=2, lmax=lmax)


def test_alm2map_spin_validation(nside: int, lmax: int) -> None:
    """Test that alm2map_spin validates input correctly."""
    if lmax is None:
        lmax = 3 * nside - 1
    L = lmax + 1

    # For spin != 0, should require list of 2 alms
    with pytest.raises(ValueError, match='must be a list/tuple of 2 arrays'):
        jhp.alm2map_spin(jnp.zeros((L, 2 * L - 1), dtype=complex), nside, spin=2, lmax=lmax, healpy_ordering=False)

    # Wrong number of alms
    with pytest.raises(ValueError, match='must be a list/tuple of 2 arrays'):
        jhp.alm2map_spin([jnp.zeros((L, 2 * L - 1), dtype=complex)], nside, spin=2, lmax=lmax, healpy_ordering=False)


@pytest.mark.parametrize('spin', [1, 2, 3])
@pytest.mark.parametrize('healpy_ordering', [True, False])
def test_map2alm_spin_different_spins(spin: int, nside: int, healpy_ordering: bool) -> None:
    """map2alm_spin with different spins matches healpy on both returned alms."""
    lmax = 3 * nside - 1
    L = lmax + 1
    npix = hp.nside2npix(nside)

    # Two real maps to feed the spin transform (same approach as test_map2alm_spin_basic).
    np.random.seed(42 + spin)
    map1 = np.random.randn(npix)
    map2 = np.random.randn(npix)

    result_jax = jhp.map2alm_spin([map1, map2], spin=spin, lmax=lmax, healpy_ordering=healpy_ordering)
    result_hp = hp.map2alm_spin([map1, map2], spin, lmax=lmax)

    assert isinstance(result_jax, list)
    assert len(result_jax) == 2

    # Reference in the same layout as the jax output.
    if healpy_ordering:
        nalm = hp.Alm.getsize(lmax)
        assert result_jax[0].shape == (nalm,)
        ref0, ref1 = np.asarray(result_hp[0]), np.asarray(result_hp[1])
    else:
        assert result_jax[0].shape == (L, 2 * L - 1)
        ref0 = flm_hp_to_2d_fast(result_hp[0], L)
        ref1 = flm_hp_to_2d_fast(result_hp[1], L)

    # alm comparison: measured max-abs floor across these spins/nsides is ~0.009, so
    # atol=1.5e-2 (~1.7x floor) is tight yet robust; MSE floor measured ~1e-7.
    # nside=16 is coarser (max-abs floor ~0.018), so it gets a looser atol; the MSE bound
    # (1e-4) already holds at every nside. nside>=32 keeps the tight 1.5e-2.
    atol = 4e-2 if nside < 32 else 1.5e-2
    for jax_alm, ref in ((result_jax[0], ref0), (result_jax[1], ref1)):
        jax_alm = np.asarray(jax_alm)
        assert np.mean(np.abs(jax_alm - ref) ** 2) < 1e-4
        np.testing.assert_allclose(jax_alm, ref, atol=atol, rtol=1e-8)


@pytest.mark.parametrize('spin', [1, 2, 3])
@pytest.mark.parametrize('healpy_ordering', [True, False])
def test_alm2map_spin_different_spins(
    flm_generator: Callable[[...], np.ndarray], spin: int, nside: int, lmax: int, healpy_ordering: bool
) -> None:
    """Test alm2map_spin with different spin values against healpy.

    Uses a roundtrip approach: generate random Q/U maps, convert to alms with healpy,
    then convert back to maps with both healpy and jax_healpy and compare.
    """
    if lmax is None:
        lmax = 3 * nside - 1
    L = lmax + 1
    npix = hp.nside2npix(nside)

    # Generate random Q and U maps
    np.random.seed(42 + spin)  # Different seed per spin for variety
    map_Q = np.random.randn(npix)
    map_U = np.random.randn(npix)

    # Convert to alms using healpy to get proper test data
    alm_E_hp, alm_B_hp = hp.map2alm_spin([map_Q, map_U], spin, lmax=lmax)

    # Convert alms to test format if needed
    if healpy_ordering:
        alm_E_test = alm_E_hp
        alm_B_test = alm_B_hp
    else:
        # Convert to s2fft 2D format for jax_healpy
        alm_E_test = flm_hp_to_2d_fast(alm_E_hp, L)
        alm_B_test = flm_hp_to_2d_fast(alm_B_hp, L)

    # Apply inverse spin transform with jax_healpy
    result_jax = jhp.alm2map_spin(
        [alm_E_test, alm_B_test], nside, spin=spin, lmax=lmax, healpy_ordering=healpy_ordering
    )

    # Apply inverse spin transform with healpy
    result_hp = hp.alm2map_spin([alm_E_hp, alm_B_hp], nside, spin=spin, lmax=lmax)

    # Should return list of 2 maps
    assert isinstance(result_jax, list)
    assert len(result_jax) == 2
    assert result_jax[0].shape == (npix,)
    assert result_jax[1].shape == (npix,)

    # Compare results using MSE metric. Higher spin has a larger numerical floor.
    # Pointwise atol sits just above the measured map-error floor of the single-transform
    # s2fft alm2map_spin vs healpy (worst at nside=32, lmax=3*nside-1): ~0.13 for spin<=2
    # and ~0.17 for spin=3. The nside>=32 values below are ~1.15x that floor -- meaningful,
    # not slack. nside=16 is a coarse-resolution smoke case with an intrinsically larger
    # floor (measured MSE up to ~5.8e-4, max-abs up to ~0.26), so it gets looser bounds.
    if nside < 32:
        mse_threshold = 5e-4 if spin <= 2 else 1e-3
        atol_threshold = 0.25 if spin <= 2 else 0.35
    else:
        mse_threshold = 1e-4 if spin <= 2 else 2e-4
        atol_threshold = 0.15 if spin <= 2 else 0.20

    MSE_Q = np.mean(np.abs(result_jax[0] - result_hp[0]) ** 2)
    MSE_U = np.mean(np.abs(result_jax[1] - result_hp[1]) ** 2)
    assert MSE_Q < mse_threshold, f'MSE Q too high for spin={spin}: {MSE_Q}'
    assert MSE_U < mse_threshold, f'MSE U too high for spin={spin}: {MSE_U}'

    np.testing.assert_allclose(result_jax[0], result_hp[0], atol=atol_threshold, rtol=1e-2)
    np.testing.assert_allclose(result_jax[1], result_hp[1], atol=atol_threshold, rtol=1e-2)
