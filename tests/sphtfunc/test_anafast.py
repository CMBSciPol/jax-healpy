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


def test_anafast_nspec(synthesized_map: np.ndarray, lmax: int | None) -> None:
    """Test that anafast nspec parameter works correctly."""
    cl_full = jhp.anafast(synthesized_map, lmax=lmax, pol=False)
    cl_partial = jhp.anafast(synthesized_map, lmax=lmax, nspec=4, pol=False)

    # nspec keeps the first 4 multipoles of the single spectrum (matches healpy slicing).
    assert cl_partial.shape == (4,)
    np.testing.assert_allclose(cl_partial, cl_full[:4], atol=1e-15, rtol=1e-15)


def test_anafast_use_weights_raises(synthesized_map: np.ndarray) -> None:
    """Test that anafast raises NotImplementedError for use_weights=True."""
    with pytest.raises(NotImplementedError, match='use_weights is not supported'):
        jhp.anafast(synthesized_map, use_weights=True, pol=False)


def test_anafast_datapath_raises(synthesized_map: np.ndarray) -> None:
    """Test that anafast raises NotImplementedError for datapath."""
    with pytest.raises(NotImplementedError, match='datapath is not supported'):
        jhp.anafast(synthesized_map, datapath='/some/path', pol=False)


def test_anafast_gal_cut_raises(synthesized_map: np.ndarray) -> None:
    """Test that anafast raises NotImplementedError for gal_cut != 0."""
    with pytest.raises(NotImplementedError, match='gal_cut is not supported'):
        jhp.anafast(synthesized_map, gal_cut=10, pol=False)


def test_anafast_use_pixel_weights_raises(synthesized_map: np.ndarray) -> None:
    """Test that anafast raises NotImplementedError for use_pixel_weights=True."""
    with pytest.raises(NotImplementedError, match='use_pixel_weights is not supported'):
        jhp.anafast(synthesized_map, use_pixel_weights=True, pol=False)


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


def test_synfast_roundtrip(nside: int, power_law_cl) -> None:
    """synfast alms -> map -> map2alm recovers the alms element-wise (with iteration).

    At lmax = 2*nside - 1 the field is band-limited well inside the healpix Nyquist
    limit, so map2alm o alm2map converges to the identity: with enough iterations the
    recovered alms match the synfast alms to ~machine precision. (At lmax = 3*nside - 1
    aliasing near the Nyquist edge prevents convergence, so this uses the supported
    minimum band-limit.)
    """
    lmax = 2 * nside - 1
    cl_input = power_law_cl

    # synfast returns the s2fft-layout alms it drew along with the map.
    map_synth, alms = jhp.synfast(jax.random.PRNGKey(12345), cl_input, nside, lmax=lmax, pol=False, alm=True)

    # Invert the map back to alms; iteration drives the roundtrip to machine precision.
    alms_recovered = jhp.map2alm(np.asarray(map_synth), lmax=lmax, iter=10, pol=False, healpy_ordering=False)
    np.testing.assert_allclose(np.asarray(alms_recovered), np.asarray(alms), atol=1e-8, rtol=1e-8)

    # anafast's spectrum of the map must equal alm2cl of the drawn alms (self-consistency).
    # Note: a single realization's empirical spectrum differs from the *input* C_l by cosmic
    # variance, so the reference is the spectrum of the alms that produced the map, not cl_input.
    cl_recov = jhp.anafast(np.asarray(map_synth), lmax=lmax, iter=10, pol=False)
    np.testing.assert_allclose(
        np.asarray(cl_recov), np.asarray(jhp.alm2cl(alms, healpy_ordering=False)), atol=1e-8, rtol=1e-8
    )


def test_synfast_different_seeds(power_law_cl) -> None:
    """Test that synfast with different seeds produces different maps."""
    nside = 16
    lmax = 31
    cl = power_law_cl

    map1 = jhp.synfast(jax.random.PRNGKey(42), cl, nside, lmax=lmax, pol=False)
    map2 = jhp.synfast(jax.random.PRNGKey(43), cl, nside, lmax=lmax, pol=False)

    # Maps should be different
    assert not jnp.allclose(map1, map2)


# Polarization (pol=True) tests for synfast / synalm
def test_synfast_pol_shape_and_consistency(pol_nside, pol_lmax, synthetic_teb) -> None:
    """synfast pol=True returns TQU maps consistent with alm2map(pol=True)."""
    cls = synthetic_teb

    map_synth, alms = jhp.synfast(jax.random.PRNGKey(0), cls, pol_nside, lmax=pol_lmax, new=True, pol=True, alm=True)

    npix = hp.nside2npix(pol_nside)
    assert map_synth.shape == (3, npix)
    assert alms.shape == (3, pol_lmax + 1, 2 * pol_lmax + 1)
    assert np.all(np.isfinite(np.asarray(map_synth)))

    # The returned map must be exactly alm2map(pol=True) of the returned alms.
    map_from_alm = jhp.alm2map(alms, pol_nside, lmax=pol_lmax, pol=True, healpy_ordering=False)
    np.testing.assert_allclose(np.asarray(map_synth), np.asarray(map_from_alm), atol=1e-12)


def test_synfast_pol_new_vs_old_ordering(pol_nside, pol_lmax, synthetic_teb) -> None:
    """new=True (TT,EE,BB,TE) and new=False (TT,TE,EE,BB) must agree for one key."""
    tt, ee, bb, te = synthetic_teb
    cls_new = np.array([tt, ee, bb, te])  # diagonal order
    cls_old = np.array([tt, te, ee, bb])  # row order

    key = jax.random.PRNGKey(11)
    map_new = jhp.synfast(key, cls_new, pol_nside, lmax=pol_lmax, new=True, pol=True)
    map_old = jhp.synfast(key, cls_old, pol_nside, lmax=pol_lmax, new=False, pol=True)
    np.testing.assert_allclose(np.asarray(map_new), np.asarray(map_old), atol=1e-12)


def test_synfast_pol_recovers_input_spectra(pol_nside, pol_lmax, synthetic_teb) -> None:
    """Ensemble-averaged anafast of synfast pol maps recovers the input TEB spectra."""
    tt, ee, bb, te = synthetic_teb
    cls = np.array([tt, ee, bb, te])
    ell = np.arange(pol_lmax + 1)
    m = ell >= 2

    n_real = 80
    acc = np.zeros((6, pol_lmax + 1))
    for i in range(n_real):
        map_synth = jhp.synfast(jax.random.PRNGKey(7000 + i), cls, pol_nside, lmax=pol_lmax, new=True, pol=True)
        acc += np.asarray(jhp.anafast(np.asarray(map_synth), lmax=pol_lmax, pol=True))  # TT,EE,BB,TE,EB,TB
    acc /= n_real

    # Auto- and TE cross-spectra recover the input (healpy synfast contract).
    for idx, ref in [(0, tt), (1, ee), (2, bb), (3, te)]:
        rel = np.median(np.abs(acc[idx][m] - ref[m]) / (np.abs(ref[m]) + 1e-30))
        assert rel < 0.2, f'spectrum {idx}: median rel error {rel}'
    # EB, TB have no input correlation -> consistent with zero.
    bound = 0.1 * np.mean(np.sqrt(tt[m] * ee[m]))
    assert np.mean(np.abs(acc[4][m])) < bound  # EB
    assert np.mean(np.abs(acc[5][m])) < bound  # TB


def test_synfast_pol_false_recovers_input_spectra(pol_nside, pol_lmax, synthetic_teb) -> None:
    """pol=False with several spectra transforms the correlated alms as independent
    spin-0 maps; the recovered auto- and TE cross-spectra still match the input.

    Physical counterpart to the pol=True recovery test: here the 3 returned maps are
    the T, E, B fields themselves (not Q, U), so a scalar anafast on each recovers
    TT/EE/BB and a scalar cross of the T and E maps recovers TE.
    """
    tt, ee, bb, te = synthetic_teb
    cls = np.array([tt, ee, bb, te])
    ell = np.arange(pol_lmax + 1)
    m = ell >= 2

    n_real = 80
    acc_tt = np.zeros(pol_lmax + 1)
    acc_ee = np.zeros(pol_lmax + 1)
    acc_bb = np.zeros(pol_lmax + 1)
    acc_te = np.zeros(pol_lmax + 1)
    for i in range(n_real):
        maps = np.asarray(jhp.synfast(jax.random.PRNGKey(8000 + i), cls, pol_nside, lmax=pol_lmax, new=True, pol=False))
        assert maps.shape == (3, hp.nside2npix(pol_nside))
        t_map, e_map, b_map = maps
        acc_tt += np.asarray(jhp.anafast(t_map, lmax=pol_lmax, pol=False))
        acc_ee += np.asarray(jhp.anafast(e_map, lmax=pol_lmax, pol=False))
        acc_bb += np.asarray(jhp.anafast(b_map, lmax=pol_lmax, pol=False))
        acc_te += np.asarray(jhp.anafast(t_map, e_map, lmax=pol_lmax, pol=False))
    acc_tt /= n_real
    acc_ee /= n_real
    acc_bb /= n_real
    acc_te /= n_real

    for rec, ref in [(acc_tt, tt), (acc_ee, ee), (acc_bb, bb), (acc_te, te)]:
        rel = np.median(np.abs(rec[m] - ref[m]) / (np.abs(ref[m]) + 1e-30))
        assert rel < 0.2, f'median rel error {rel}'


def test_synalm_pol_recovers_input_spectra(pol_lmax, synthetic_teb) -> None:
    """Ensemble-averaged alm2cl of synalm pol coefficients recovers the input spectra."""
    tt, ee, bb, te = synthetic_teb
    cls = np.array([tt, ee, bb, te])
    ell = np.arange(pol_lmax + 1)
    m = ell >= 2

    n_real = 300
    acc = {k: np.zeros(pol_lmax + 1) for k in ('TT', 'EE', 'BB', 'TE', 'TB', 'EB')}
    for i in range(n_real):
        alm = jhp.synalm(jax.random.PRNGKey(3000 + i), cls, lmax=pol_lmax, new=True, healpy_ordering=False)
        assert alm.shape == (3, pol_lmax + 1, 2 * pol_lmax + 1)
        acc['TT'] += np.asarray(jhp.alm2cl(alm[0], healpy_ordering=False))
        acc['EE'] += np.asarray(jhp.alm2cl(alm[1], healpy_ordering=False))
        acc['BB'] += np.asarray(jhp.alm2cl(alm[2], healpy_ordering=False))
        acc['TE'] += np.asarray(jhp.alm2cl(alm[0], alm[1], healpy_ordering=False))
        acc['TB'] += np.asarray(jhp.alm2cl(alm[0], alm[2], healpy_ordering=False))
        acc['EB'] += np.asarray(jhp.alm2cl(alm[1], alm[2], healpy_ordering=False))
    for k in acc:
        acc[k] /= n_real

    for name, ref in [('TT', tt), ('EE', ee), ('BB', bb), ('TE', te)]:
        rel = np.median(np.abs(acc[name][m] - ref[m]) / (np.abs(ref[m]) + 1e-30))
        assert rel < 0.2, f'{name}: median rel error {rel}'
    bound = 0.1 * np.mean(np.sqrt(tt[m] * ee[m]))
    assert np.mean(np.abs(acc['TB'][m])) < bound
    assert np.mean(np.abs(acc['EB'][m])) < bound


def test_synalm_pol_six_cl_and_none_match_four_cl(pol_lmax, synthetic_teb) -> None:
    """6-cl input (TT,EE,BB,TE,EB,TB) with None cross-terms matches the 4-cl promotion.

    Exercises the n(n+1)/2 branch of synalm's covariance build, the list-with-None
    path of _as_spectra_list, and _new_to_old_spectra_order for n=3.
    """
    tt, ee, bb, te = synthetic_teb
    cls_four = np.array([tt, ee, bb, te])  # new order, promoted internally to EB=TB=0
    cls_six = [tt, ee, bb, te, None, None]  # new diagonal order with EB, TB omitted

    key = jax.random.PRNGKey(5)
    alm_four = jhp.synalm(key, cls_four, lmax=pol_lmax, new=True, healpy_ordering=False)
    alm_six = jhp.synalm(key, cls_six, lmax=pol_lmax, new=True, healpy_ordering=False)
    np.testing.assert_allclose(np.asarray(alm_four), np.asarray(alm_six), atol=1e-12)


def test_synfast_mmax_raises() -> None:
    """synfast raises NotImplementedError when mmax != lmax."""
    nside = 16
    lmax = 2 * nside - 1
    ell = jnp.arange(lmax + 1)
    cl = 1.0 / (ell + 10) ** 2

    with pytest.raises(NotImplementedError, match='mmax'):
        jhp.synfast(jax.random.PRNGKey(42), cl, nside, lmax=lmax, mmax=lmax - 1, pol=False)


def test_synalm_mmax_raises() -> None:
    """synalm raises NotImplementedError when mmax != lmax."""
    lmax = 31
    ell = jnp.arange(lmax + 1)
    cl = 1.0 / (ell + 10) ** 2

    with pytest.raises(NotImplementedError, match='mmax'):
        jhp.synalm(jax.random.PRNGKey(42), cl, lmax=lmax, mmax=lmax - 1)


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
    cl_jax = jhp.alm2cl(alm_jax, healpy_ordering=True)

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
    cl_jax = jhp.alm2cl(alm1_jax, alm2_jax, healpy_ordering=True)

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
    cls_jax = jhp.alm2cl(alms_jax, healpy_ordering=True)

    # Should return 6 spectra (3*4/2) as a stacked array, matching healpy's ndarray return.
    assert np.asarray(cls_jax).shape[0] == 6

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
    cls_jax = jhp.alm2cl(alms_jax, nspec=3, healpy_ordering=True)

    assert np.asarray(cls_jax).shape[0] == 3


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


# anafast pol=True against healpy.
#
# Empirical agreement against hp.anafast(pol=True) for nside in {32,64,128}, lmax in {default, 2*nside-1, 3*nside-1}:
#   TT: max_abs ~ 1e-17            (T goes through spin=0, matches to machine precision)
#   EE: max_abs ~ 6e-6   rel ~5e-14 (spin=2 alms averaged over m; relative agreement is machine precision)
#   BB: max_abs ~ 3e-6   rel ~1e-13
#   TE: max_abs ~ 2e-5   rel ~2e-13
#   EB: max_abs ~ 5e-6   rel ~2e-12
#   TB: max_abs ~ 2e-5   rel ~2e-12
# Per-spectrum atol is set ~5x above the worst observed max_abs across all configurations.
_POL_SPECTRA_NAMES = ('TT', 'EE', 'BB', 'TE', 'EB', 'TB')
_POL_ATOL = {'TT': 1e-14, 'EE': 3e-5, 'BB': 3e-5, 'TE': 1e-4, 'EB': 3e-5, 'TB': 1e-4}
_POL_RTOL = {'TT': 1e-12, 'EE': 1e-10, 'BB': 1e-10, 'TE': 1e-10, 'EB': 1e-10, 'TB': 1e-10}


def _assert_pol_cl_match(cl_jax: np.ndarray, cl_hp: np.ndarray) -> None:
    assert cl_jax.shape == cl_hp.shape == (6, cl_hp.shape[1])
    for i, name in enumerate(_POL_SPECTRA_NAMES):
        np.testing.assert_allclose(
            cl_jax[i], cl_hp[i], atol=_POL_ATOL[name], rtol=_POL_RTOL[name], err_msg=f'{name} spectrum'
        )


def test_anafast_pol_auto(synthesized_tqu_map: np.ndarray, lmax: int | None) -> None:
    cl_jax = np.asarray(jhp.anafast(synthesized_tqu_map, lmax=lmax, iter=0, pol=True))
    cl_hp = hp.anafast(synthesized_tqu_map, lmax=lmax, iter=0, pol=True)
    _assert_pol_cl_match(cl_jax, cl_hp)


def test_anafast_pol_cross(synthesized_tqu_map: np.ndarray, lmax: int | None) -> None:
    np.random.seed(7)
    tqu2 = synthesized_tqu_map + 0.1 * np.random.randn(*synthesized_tqu_map.shape)

    cl_jax = np.asarray(jhp.anafast(synthesized_tqu_map, tqu2, lmax=lmax, iter=0, pol=True))
    cl_hp = hp.anafast(synthesized_tqu_map, tqu2, lmax=lmax, iter=0, pol=True)
    _assert_pol_cl_match(cl_jax, cl_hp)


def test_anafast_pol_nspec(synthesized_tqu_map: np.ndarray) -> None:
    nside = hp.npix2nside(synthesized_tqu_map.shape[-1])
    lmax = 3 * nside - 1

    cl_full = jhp.anafast(synthesized_tqu_map, lmax=lmax, iter=0, pol=True)
    cl_partial = jhp.anafast(synthesized_tqu_map, lmax=lmax, iter=0, pol=True, nspec=4)

    assert cl_full.shape == (6, lmax + 1)
    assert cl_partial.shape == (4, lmax + 1)
    np.testing.assert_array_equal(cl_partial, cl_full[:4])


def test_anafast_pol_with_alm_return(synthesized_tqu_map: np.ndarray, lmax: int | None) -> None:
    """alm=True returns (cl, alm_TEB) with alm_TEB of shape (3, L, 2L-1)."""
    nside = hp.npix2nside(synthesized_tqu_map.shape[-1])
    target_lmax = (3 * nside - 1) if lmax is None else lmax
    L = target_lmax + 1

    cl, alm_teb = jhp.anafast(synthesized_tqu_map, lmax=lmax, iter=0, pol=True, alm=True)

    assert cl.shape == (6, L)
    assert alm_teb.shape == (3, L, 2 * L - 1)

    # The returned alms must match jhp.map2alm(pol=True) in the same ordering.
    alm_ref = jhp.map2alm(synthesized_tqu_map, lmax=lmax, iter=0, pol=True, healpy_ordering=False)
    assert alm_teb.shape == alm_ref.shape
    np.testing.assert_allclose(np.asarray(alm_teb), np.asarray(alm_ref), atol=1e-14, rtol=1e-14)

    # And cl must match the non-alm return path exactly.
    cl_only = jhp.anafast(synthesized_tqu_map, lmax=lmax, iter=0, pol=True)
    np.testing.assert_array_equal(cl, cl_only)


def test_anafast_pol_cross_with_alm_return(synthesized_tqu_map: np.ndarray) -> None:
    nside = hp.npix2nside(synthesized_tqu_map.shape[-1])
    lmax = 3 * nside - 1
    L = lmax + 1

    np.random.seed(7)
    tqu2 = synthesized_tqu_map + 0.1 * np.random.randn(*synthesized_tqu_map.shape)

    cl, alm1, alm2 = jhp.anafast(synthesized_tqu_map, tqu2, lmax=lmax, iter=0, pol=True, alm=True)

    assert cl.shape == (6, L)
    assert alm1.shape == (3, L, 2 * L - 1)
    assert alm2.shape == (3, L, 2 * L - 1)


def test_anafast_pol_qu_only_matches_healpy_iqu(synthesized_tqu_map: np.ndarray, lmax: int | None) -> None:
    """anafast on Q, U only (2 maps) returns (EE, BB, EB), matching healpy's IQU spectra.

    Deviation from healpy (which requires I, Q, U). E/B depend only on Q, U, so EE, BB, EB
    match healpy's IQU values at indices 1, 2, 4 of (TT, EE, BB, TE, EB, TB).
    """
    nside = hp.npix2nside(synthesized_tqu_map.shape[-1])
    target_lmax = (3 * nside - 1) if lmax is None else lmax

    cl_qu = np.asarray(jhp.anafast(synthesized_tqu_map[1:], lmax=lmax, iter=0, pol=True))  # (3, L): EE, BB, EB
    cl_iqu_hp = hp.anafast(synthesized_tqu_map, lmax=lmax, iter=0, pol=True)  # (6, L)

    assert cl_qu.shape == (3, target_lmax + 1)
    expected = np.stack([cl_iqu_hp[1], cl_iqu_hp[2], cl_iqu_hp[4]])  # EE, BB, EB
    np.testing.assert_allclose(cl_qu, expected, atol=1e-4, rtol=1e-10)


def test_anafast_pol_invalid_count_raises(synthesized_tqu_map: np.ndarray) -> None:
    """pol=True accepts (3, npix) or (2, npix); a 1-D map is scalar; other shapes raise."""
    # A single (1-D) map is treated as a scalar field (matches healpy): returns the
    # scalar auto-spectrum instead of raising, even with pol=True.
    nside = hp.npix2nside(synthesized_tqu_map.shape[-1])
    lmax = 2 * nside - 1
    cl_scalar = jhp.anafast(synthesized_tqu_map[0], lmax=lmax, iter=0, pol=True)
    cl_hp = hp.anafast(synthesized_tqu_map[0], lmax=lmax, iter=0, pol=True)
    assert cl_scalar.shape == (lmax + 1,)
    np.testing.assert_allclose(np.asarray(cl_scalar), cl_hp, atol=1e-10, rtol=1e-10)
    # 4 components is not valid.
    four = np.concatenate([synthesized_tqu_map, synthesized_tqu_map[:1]], axis=0)
    with pytest.raises(ValueError, match=r'pol=True requires map1 with shape'):
        jhp.anafast(four, pol=True)
    # Cross-spectrum: map2 must have the same component count as map1.
    with pytest.raises(ValueError, match=r'map2 with the same shape'):
        jhp.anafast(synthesized_tqu_map[1:], synthesized_tqu_map, pol=True)  # QU vs IQU
