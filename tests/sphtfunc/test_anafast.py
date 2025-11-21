import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from s2fft.sampling.reindex import flm_hp_to_2d_fast

import jax_healpy as jhp

jax.config.update('jax_enable_x64', True)


def test_anafast_basic(synthesized_map: np.ndarray, lmax: int) -> None:
    """Test basic anafast auto-spectrum computation."""
    cl_jax = jhp.anafast(synthesized_map, lmax=lmax, iter=0, pol=False)
    cl_healpy = hp.anafast(synthesized_map, lmax=lmax, iter=0, pol=False)

    # Power spectra should match closely
    np.testing.assert_allclose(cl_jax, cl_healpy, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('iter', [2, 3])
def test_anafast_with_iter(synthesized_map: np.ndarray, lmax: int, iter: int) -> None:
    """Test anafast with iterative refinement."""
    cl_iter3_healpy = hp.anafast(synthesized_map, lmax=lmax, iter=iter, pol=False)
    cl_iter3_jax = jhp.anafast(synthesized_map, lmax=lmax, iter=iter, pol=False)

    # Both should be reasonably close (iter improves accuracy but not drastically for well-behaved maps)
    np.testing.assert_allclose(cl_iter3_healpy, cl_iter3_jax, atol=1e-10, rtol=1e-10)


def test_anafast_cross_spectrum(synthesized_map: np.ndarray, lmax: int) -> None:
    """Test anafast cross-spectrum computation."""
    # Create two correlated maps (same map + noise)
    np.random.seed(42)
    noise = np.random.randn(synthesized_map.size) * 0.1
    map2 = synthesized_map + noise

    cl_cross_jax = jhp.anafast(synthesized_map, map2, lmax=lmax, iter=0, pol=False)
    cl_cross_healpy = hp.anafast(synthesized_map, map2, lmax=lmax, iter=0, pol=False)

    np.testing.assert_allclose(cl_cross_jax, cl_cross_healpy, atol=1e-10, rtol=1e-10)


def test_anafast_with_alm_return(synthesized_map: np.ndarray, lmax: int) -> None:
    """Test anafast with alm return option."""
    # With alm=True, should return both cl and alm
    result = jhp.anafast(synthesized_map, lmax=lmax, iter=0, alm=True, pol=False)
    results_hp = hp.anafast(synthesized_map, lmax=lmax, iter=0, alm=True, pol=False)
    assert isinstance(result, tuple)
    assert len(result) == 2

    cl_hp, alm_hp = results_hp
    resolved_lmax = cl_hp.shape[0] - 1
    L = resolved_lmax + 1

    cl, alm = result
    assert cl.shape == (L,)
    assert alm.shape == (L, 2 * L - 1)  # s2fft 2D format

    alm_hp_2d = flm_hp_to_2d_fast(jnp.asarray(alm_hp), L)
    np.testing.assert_allclose(cl, cl_hp, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(alm, alm_hp_2d, atol=1e-10, rtol=1e-10)


def test_anafast_cross_with_alm_return(synthesized_map: np.ndarray, lmax: int) -> None:
    """Test anafast cross-spectrum with alm return option."""
    np.random.seed(42)
    map2 = synthesized_map + np.random.randn(synthesized_map.size) * 0.1

    result = jhp.anafast(synthesized_map, map2, lmax=lmax, iter=0, alm=True, pol=False)
    results_hp = hp.anafast(synthesized_map, map2, lmax=lmax, iter=0, alm=True, pol=False)
    assert isinstance(result, tuple)
    assert len(result) == 3

    cl_hp, alm1_hp, alm2_hp = results_hp
    resolved_lmax = cl_hp.shape[0] - 1
    L = resolved_lmax + 1

    cl, alm1, alm2 = result
    assert cl.shape == (L,)
    assert alm1.shape == (L, 2 * L - 1)
    assert alm2.shape == (L, 2 * L - 1)

    alm1_hp_2d = flm_hp_to_2d_fast(jnp.asarray(alm1_hp), L)
    alm2_hp_2d = flm_hp_to_2d_fast(jnp.asarray(alm2_hp), L)
    np.testing.assert_allclose(cl, cl_hp, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(alm1, alm1_hp_2d, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(alm2, alm2_hp_2d, atol=1e-10, rtol=1e-10)


def test_synfast_basic(cla: np.ndarray, nside: int) -> None:
    """Test basic synfast map generation."""
    lmax = 3 * nside - 1

    seed = 12345
    map_jax = jhp.synfast(jax.random.PRNGKey(seed), cla, nside, lmax=lmax, pol=False)
    rng_state = np.random.get_state()
    np.random.seed(seed)
    map_healpy = hp.synfast(cla, nside, lmax=lmax, pol=False)
    np.random.set_state(rng_state)

    # Check output shape
    npix = hp.nside2npix(nside)
    assert map_jax.shape == (npix,)

    # Check that map has reasonable statistics
    # Mean should be close to zero
    assert jnp.abs(jnp.mean(map_jax)) < 1e-2

    # Variance should be reasonable (related to C_l)
    assert jnp.std(map_jax) > 0

    print(f'Mean JAX: {jnp.mean(map_jax)}, Mean Healpy: {np.mean(map_healpy)}')
    print(f'Std JAX: {jnp.std(map_jax)}, Std Healpy: {np.std(map_healpy)}')
    mean_diff = jnp.abs(jnp.mean(map_jax) - np.mean(map_healpy))
    mean_rtol = mean_diff / (np.abs(np.mean(map_healpy)) + 1e-20)
    print(f'atol diff mean: {mean_diff} rtol diff mean: {mean_rtol}')
    std_diff = jnp.abs(jnp.std(map_jax) - np.std(map_healpy))
    std_rtol = std_diff / (np.abs(np.std(map_healpy)) + 1e-20)
    print(f'atol diff std: {std_diff} rtol diff std: {std_rtol}')

    # Compare summary statistics with healpy map (stochastic realizations won't match sample-wise)
    assert np.isclose(np.mean(map_jax), np.mean(map_healpy), atol=1e-5)
    assert np.isclose(np.std(map_jax), np.std(map_healpy), atol=1e-2)


def test_synfast_with_smoothing(cla: np.ndarray, nside: int) -> None:
    """Test synfast with Gaussian smoothing."""
    fwhm = np.radians(5.0)  # 5 degrees
    seed = 12345

    lmax = 3 * nside - 1
    map_smooth = jhp.synfast(jax.random.PRNGKey(seed), cla, nside, lmax=lmax, fwhm=fwhm, pol=False)
    map_no_smooth = jhp.synfast(jax.random.PRNGKey(seed), cla, nside, lmax=lmax, fwhm=0.0, pol=False)

    # Smoothed map should have lower variance at high frequencies
    # This is hard to test directly, but at least check they're different
    assert not jnp.allclose(map_smooth, map_no_smooth)
    assert jnp.std(map_smooth) < jnp.std(map_no_smooth)


def test_synfast_roundtrip(nside: int) -> None:
    """Test synfast -> anafast roundtrip recovers input spectrum."""
    seed = 12345

    lmax = 3 * nside - 1

    # Generate a synthetic power spectrum with the correct lmax
    ell = jnp.arange(lmax + 1)
    cl_input = 1.0 / (ell + 10) ** 2

    # Generate map from this spectrum
    map_synth = jhp.synfast(jax.random.PRNGKey(seed), cl_input, nside, lmax=lmax, pol=False)

    # Recover spectrum
    cl_recovered = jhp.anafast(map_synth, lmax=lmax, iter=3, pol=False)

    target_cl = cl_input

    rel_error = jnp.abs(cl_recovered - target_cl) / (target_cl + 1e-20)
    a_error = jnp.abs(cl_recovered - target_cl)

    # Most multipoles should have reasonable accuracy (within ~20% for single realization)
    # Exclude l<2 which can be noisy
    # Absolute error threshold accounts for cosmic variance in single realizations
    print(f'jnp.median(rel_error[2:]) = {jnp.median(rel_error[2:])}')
    print(f'jnp.median(a_error[2:]) = {jnp.median(a_error[2:])}')
    assert jnp.median(rel_error[2:]) < 0.2
    assert jnp.median(a_error[2:]) < 1e-3


def test_synfast_different_seeds() -> None:
    """Test that synfast with different seeds produces different maps."""
    nside = 16
    lmax = 31
    ell = jnp.arange(lmax + 1)
    cl = 1.0 / (ell + 10) ** 2

    map1 = jhp.synfast(jax.random.PRNGKey(42), cl, nside, lmax=lmax, pol=False)
    map2 = jhp.synfast(jax.random.PRNGKey(43), cl, nside, lmax=lmax, pol=False)

    # Maps should be different
    assert not jnp.allclose(map1, map2)


# Parameter validation tests for anafast
def test_anafast_nspec() -> None:
    """Test that anafast nspec parameter works correctly."""
    nside = 16
    npix = hp.nside2npix(nside)
    lmax = 31
    np.random.seed(42)
    map_data = np.random.randn(npix)

    # Get full spectrum
    cl_full = jhp.anafast(map_data, lmax=lmax, pol=False)

    # Get only first 4 spectra
    cl_partial = jhp.anafast(map_data, lmax=lmax, nspec=4, pol=False)

    # Should match the first 4 elements
    assert cl_partial.shape == (4,)
    np.testing.assert_allclose(cl_partial, cl_full[:4], atol=1e-15, rtol=1e-15)


def test_anafast_pol_raises() -> None:
    """Test that anafast raises NotImplementedError for pol=True."""
    nside = 16
    npix = hp.nside2npix(nside)
    map_data = np.random.randn(npix)

    with pytest.raises(NotImplementedError, match='pol=True.*is not supported'):
        jhp.anafast(map_data, pol=True)


def test_anafast_use_weights_raises() -> None:
    """Test that anafast raises NotImplementedError for use_weights=True."""
    nside = 16
    npix = hp.nside2npix(nside)
    map_data = np.random.randn(npix)

    with pytest.raises(NotImplementedError, match='use_weights is not supported'):
        jhp.anafast(map_data, use_weights=True, pol=False)


def test_anafast_datapath_raises() -> None:
    """Test that anafast raises NotImplementedError for datapath."""
    nside = 16
    npix = hp.nside2npix(nside)
    map_data = np.random.randn(npix)

    with pytest.raises(NotImplementedError, match='datapath is not supported'):
        jhp.anafast(map_data, datapath='/some/path', pol=False)


def test_anafast_gal_cut_raises() -> None:
    """Test that anafast raises NotImplementedError for gal_cut != 0."""
    nside = 16
    npix = hp.nside2npix(nside)
    map_data = np.random.randn(npix)

    with pytest.raises(NotImplementedError, match='gal_cut is not supported'):
        jhp.anafast(map_data, gal_cut=10, pol=False)


def test_anafast_use_pixel_weights_raises() -> None:
    """Test that anafast raises NotImplementedError for use_pixel_weights=True."""
    nside = 16
    npix = hp.nside2npix(nside)
    map_data = np.random.randn(npix)

    with pytest.raises(NotImplementedError, match='use_pixel_weights is not supported'):
        jhp.anafast(map_data, use_pixel_weights=True, pol=False)


# Parameter validation tests for synfast
def test_synfast_pol_raises() -> None:
    """Test that synfast raises NotImplementedError for pol=True."""
    nside = 16
    lmax = 31
    ell = jnp.arange(lmax + 1)
    cl = 1.0 / (ell + 10) ** 2

    with pytest.raises(NotImplementedError, match='pol=True.*is not supported'):
        jhp.synfast(jax.random.PRNGKey(42), cl, nside, pol=True)


def test_synfast_pixwin_raises() -> None:
    """Test that synfast raises NotImplementedError for pixwin=True."""
    nside = 16
    lmax = 31
    ell = jnp.arange(lmax + 1)
    cl = 1.0 / (ell + 10) ** 2

    with pytest.raises(NotImplementedError, match='pixwin=True is not supported'):
        jhp.synfast(jax.random.PRNGKey(42), cl, nside, pixwin=True, pol=False)


def test_synfast_alm_return(cla: np.ndarray, nside: int) -> None:
    """Test synfast with alm return option."""
    seed = 12345

    lmax = 3 * nside - 1
    # With alm=True, should return both map and alm
    result = jhp.synfast(jax.random.PRNGKey(seed), cla, nside, lmax=lmax, alm=True, pol=False)
    assert isinstance(result, tuple)
    assert len(result) == 2

    map_synth, alm = result
    npix = hp.nside2npix(nside)
    L = lmax + 1

    # Check shapes
    assert map_synth.shape == (npix,)
    assert alm.shape == (L, 2 * L - 1)  # s2fft 2D format

    # Verify that transforming the alm back gives us the same map
    map_from_alm = jhp.alm2map(alm, nside, lmax=lmax, healpy_ordering=False)
    np.testing.assert_allclose(map_synth, map_from_alm, atol=1e-10, rtol=1e-10)


def test_synfast_with_sigma(cla: np.ndarray, nside: int) -> None:
    """Test synfast with sigma parameter (alternative to fwhm)."""
    seed = 12345
    sigma = np.radians(2.0)  # 2 degrees sigma
    lmax = 3 * nside - 1
    # Generate map with sigma
    map_sigma = jhp.synfast(jax.random.PRNGKey(seed), cla, nside, lmax=lmax, sigma=sigma, pol=False)

    # Generate map without smoothing
    map_no_smooth = jhp.synfast(jax.random.PRNGKey(seed), cla, nside, lmax=lmax, pol=False)

    # Smoothed map should have lower variance
    assert jnp.std(map_sigma) < jnp.std(map_no_smooth)

    # Maps should be different
    assert not jnp.allclose(map_sigma, map_no_smooth)


def test_synfast_lmax_under_resolution_raises(cla: np.ndarray) -> None:
    """Ensure synfast raises before calling s2fft when lmax is below 2*nside-1."""
    nside = 64
    too_low_lmax = 2 * nside - 2

    with pytest.raises(ValueError, match='lmax >='):
        jhp.synfast(jax.random.PRNGKey(0), cla, nside, lmax=too_low_lmax, pol=False)


# Tests for alm2cl
def test_alm2cl_auto_spectrum(synthesized_map: np.ndarray, lmax: int) -> None:
    """Test alm2cl auto-spectrum against healpy."""
    # Get alm from map
    alm_hp = hp.map2alm(synthesized_map, lmax=lmax, iter=0)
    alm_jax = jhp.map2alm(synthesized_map, lmax=lmax, iter=0, pol=False, healpy_ordering=True)

    # Compute auto-spectrum
    cl_hp = hp.alm2cl(alm_hp)
    cl_jax = jhp.alm2cl(alm_jax)

    # Should match closely
    np.testing.assert_allclose(cl_jax, cl_hp, atol=1e-10, rtol=1e-10)


def test_alm2cl_cross_spectrum(synthesized_map: np.ndarray, lmax: int) -> None:
    """Test alm2cl cross-spectrum against healpy."""
    np.random.seed(42)
    noise = np.random.randn(synthesized_map.size) * 0.1
    map2 = synthesized_map + noise

    # Get alms
    alm1_hp = hp.map2alm(synthesized_map, lmax=lmax, iter=0)
    alm2_hp = hp.map2alm(map2, lmax=lmax, iter=0)

    alm1_jax = jhp.map2alm(synthesized_map, lmax=lmax, iter=0, pol=False, healpy_ordering=True)
    alm2_jax = jhp.map2alm(map2, lmax=lmax, iter=0, pol=False, healpy_ordering=True)

    # Compute cross-spectrum
    cl_hp = hp.alm2cl(alm1_hp, alm2_hp)
    cl_jax = jhp.alm2cl(alm1_jax, alm2_jax)

    # Should match closely
    np.testing.assert_allclose(cl_jax, cl_hp, atol=1e-10, rtol=1e-10)


def test_alm2cl_multiple_alms() -> None:
    """Test alm2cl with multiple alms producing n(n+1)/2 spectra."""
    nside = 16
    lmax = 31
    npix = hp.nside2npix(nside)

    # Create 3 test maps
    np.random.seed(42)
    maps = [np.random.randn(npix) for _ in range(3)]

    # Get alms
    alms_hp = [hp.map2alm(m, lmax=lmax, iter=0) for m in maps]
    alms_jax = [jhp.map2alm(m, lmax=lmax, iter=0, pol=False, healpy_ordering=True) for m in maps]

    # Compute all spectra with healpy
    cls_hp = hp.alm2cl(alms_hp)

    # Compute with jax-healpy
    cls_jax = jhp.alm2cl(alms_jax)

    # Should return 6 spectra (3*4/2)
    assert isinstance(cls_jax, tuple)
    assert len(cls_jax) == 6

    # Each spectrum should match
    for i, (cl_jax, cl_hp) in enumerate(zip(cls_jax, cls_hp)):
        np.testing.assert_allclose(cl_jax, cl_hp, atol=1e-10, rtol=1e-10, err_msg=f'Spectrum {i} mismatch')


def test_alm2cl_nspec() -> None:
    """Test alm2cl nspec parameter."""
    nside = 16
    lmax = 31
    npix = hp.nside2npix(nside)

    np.random.seed(42)
    maps = [np.random.randn(npix) for _ in range(3)]
    alms_jax = [jhp.map2alm(m, lmax=lmax, iter=0, pol=False, healpy_ordering=True) for m in maps]

    # Get first 3 spectra
    cls_jax = jhp.alm2cl(alms_jax, nspec=3)

    assert isinstance(cls_jax, tuple)
    assert len(cls_jax) == 3


# Tests for synalm
def test_synalm_roundtrip() -> None:
    """Test synalm -> alm2cl roundtrip recovers input spectrum."""
    lmax = 64
    ell = jnp.arange(lmax + 1)
    cl_input = 1.0 / (ell + 10) ** 2

    # Generate alm from spectrum
    key = jax.random.PRNGKey(42)
    alm = jhp.synalm(key, cl_input, lmax=lmax)

    # Recover spectrum
    cl_recovered = jhp.alm2cl(alm)

    # Should be similar (within cosmic variance for single realization)
    rel_error = jnp.abs(cl_recovered - cl_input) / (cl_input + 1e-20)

    # Most multipoles should be reasonably close
    assert jnp.median(rel_error[2:]) < 0.5


def test_synalm_vs_healpy_roundtrip() -> None:
    """Compare synalm roundtrip against healpy (statistical test)."""
    lmax = 64
    n_realizations = 50
    ell = jnp.arange(lmax + 1)
    cl_input = 1.0 / (ell + 10) ** 2

    # Multiple realizations with jax-healpy
    cls_jax = []
    for seed in range(n_realizations):
        key = jax.random.PRNGKey(seed)
        alm = jhp.synalm(key, cl_input, lmax=lmax)
        cl = jhp.alm2cl(alm)
        cls_jax.append(np.array(cl))

    # Multiple realizations with healpy
    cls_hp = []
    for seed in range(n_realizations):
        np.random.seed(seed)
        alm = hp.synalm(cl_input, lmax=lmax)
        cl = hp.alm2cl(alm)
        cls_hp.append(cl)

    # Compare mean spectra (should be similar statistically)
    mean_cl_jax = np.mean(cls_jax, axis=0)
    mean_cl_hp = np.mean(cls_hp, axis=0)

    # Mean should recover input reasonably well
    rel_error = np.abs(mean_cl_jax - mean_cl_hp) / (cl_input + 1e-20)
    assert np.median(rel_error[2:]) < 0.1  # Statistical consistency


def test_synalm_different_seeds() -> None:
    """Test that synalm with different seeds produces different alms."""
    lmax = 64
    ell = jnp.arange(lmax + 1)
    cl = 1.0 / (ell + 10) ** 2

    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(43)

    alm1 = jhp.synalm(key1, cl, lmax=lmax)
    alm2 = jhp.synalm(key2, cl, lmax=lmax)

    # Alms should be different
    assert not jnp.allclose(alm1, alm2)
