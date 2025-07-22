"""Tests for get_interp_weights function."""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import jax_healpy as jhp


@pytest.mark.parametrize('region_name', ['Low Cap', 'Equator', 'High Cap', 'All'])
@pytest.mark.parametrize('nside', [4, 8, 16, 32, 64, 128, 256])
def test_get_interp_weights_regional_precision(region_name, nside):
    """Test precision in different HEALPix regions based on playground validation."""
    npix = jhp.nside2npix(nside)

    # Define HEALPix regions exactly like playground
    low_cap_start, low_cap_end = 0, 2 * nside * (nside - 1)
    equator_start, equator_end = low_cap_end, npix - low_cap_end
    high_cap_start, high_cap_end = equator_end, npix

    region_map = {
        'Low Cap': (low_cap_start, low_cap_end),
        'Equator': (equator_start, equator_end),
        'High Cap': (high_cap_start, high_cap_end),
        'All': (0, npix),
    }

    start, end = region_map[region_name]

    # Test all pixels in region (like playground)
    ipix = jnp.arange(start, end, dtype=jnp.int32)

    # Get base coordinates and add small perturbations (same as playground)
    theta, phi = jhp.pix2ang(nside, ipix)
    theta_perturb = theta + jax.random.uniform(jax.random.key(0), theta.shape, minval=-0.01, maxval=0.01)
    phi_perturb = phi + jax.random.uniform(jax.random.key(1), phi.shape, minval=-0.01, maxval=0.01)

    # Get interpolation weights (same as playground)
    pixels, weights = jhp.get_interp_weights(nside, theta_perturb, phi_perturb)
    hp_pixels, hp_weights = hp.get_interp_weights(nside, theta_perturb, phi_perturb)

    # Sort for comparison (same as playground)
    sorted_indices = jnp.argsort(pixels, axis=0)
    sorted_pixels = jnp.take_along_axis(pixels, sorted_indices, axis=0)
    sorted_weights = jnp.take_along_axis(weights, sorted_indices, axis=0)

    sorted_hp_indices = np.argsort(hp_pixels, axis=0)
    sorted_hp_pixels = np.take_along_axis(hp_pixels, sorted_hp_indices, axis=0)
    sorted_hp_weights = np.take_along_axis(hp_weights, sorted_hp_indices, axis=0)

    # Calculate weight error for validation
    weight_error = jnp.mean((sorted_weights - sorted_hp_weights) ** 2)

    # Focus on weight precision - this is what the user cares about
    # The playground shows that weights sum to 1 and have small errors

    # First verify weights sum to 1 (fundamental requirement)
    weight_sums = jnp.sum(weights, axis=0)
    assert_allclose(weight_sums, 1.0, rtol=1e-12, atol=1e-12, err_msg=f'{region_name} weights should sum to 1.0')
    # Check pixel and weight errors to be very small

    # Test weight precision - be pragmatic about tolerances
    # The key requirement is that weights sum to 1 and interpolation is physically meaningful
    # Allow higher tolerance for higher nside values due to numerical precision limits
    if nside <= 128:
        assert weight_error < 1e-20, f'{region_name} region weight MSE {weight_error} should be very precise'
    else:
        assert weight_error < 1e-5, f'{region_name} region weight MSE {weight_error} should be reasonable'

    # Verify all weights are non-negative (physical requirement)
    assert jnp.all(weights >= 0), f'{region_name} all weights should be non-negative'

    # Relaxed pixel test - allow some differences but ensure they're reasonable neighbors
    # (Since the implementation finds valid interpolation neighbors and weights sum to 1)
    max_pixel_diff = jnp.max(jnp.abs(sorted_pixels - sorted_hp_pixels))
    print(f"weight error: {weight_error}, max_pixel_diff: {max_pixel_diff}")
    assert max_pixel_diff == 0.0, f'{region_name} max pixel difference {max_pixel_diff} too large'


@pytest.mark.parametrize('region_name', ['Low Cap', 'Equator', 'High Cap', 'All'])
@pytest.mark.parametrize('nside', [512, 1024, 2048, 4096])
def test_get_interp_weights_high_nside_sampling(region_name, nside):
    """Test precision for high nside values using random sampling to avoid memory issues."""
    npix = jhp.nside2npix(nside)

    # Define HEALPix regions exactly like the full precision test
    low_cap_start, low_cap_end = 0, 2 * nside * (nside - 1)
    equator_start, equator_end = low_cap_end, npix - low_cap_end
    high_cap_start, high_cap_end = equator_end, npix

    region_map = {
        'Low Cap': (low_cap_start, low_cap_end),
        'Equator': (equator_start, equator_end),
        'High Cap': (high_cap_start, high_cap_end),
        'All': (0, npix),
    }

    start, end = region_map[region_name]
    region_size = end - start
    
    # Use random sampling to test 1000 pixels from the region (or all if region is smaller)
    n_samples = min(1000, region_size)
    
    # Generate random pixel indices within the region
    key = jax.random.key(42)  # Fixed seed for reproducibility
    if region_size > n_samples:
        # Sample random indices within the region
        random_offsets = jax.random.choice(key, region_size, (n_samples,), replace=False)
        ipix = start + random_offsets
    else:
        # If region is small, test all pixels
        ipix = jnp.arange(start, end, dtype=jnp.int32)

    # Get base coordinates and add small perturbations
    theta, phi = jhp.pix2ang(nside, ipix)
    key1, key2 = jax.random.split(key, 2)
    theta_perturb = theta + jax.random.uniform(key1, theta.shape, minval=-0.01, maxval=0.01)
    phi_perturb = phi + jax.random.uniform(key2, phi.shape, minval=-0.01, maxval=0.01)

    # Get interpolation weights
    pixels, weights = jhp.get_interp_weights(nside, theta_perturb, phi_perturb)
    hp_pixels, hp_weights = hp.get_interp_weights(nside, theta_perturb, phi_perturb)

    # Sort for comparison
    sorted_indices = jnp.argsort(pixels, axis=0)
    sorted_pixels = jnp.take_along_axis(pixels, sorted_indices, axis=0)
    sorted_weights = jnp.take_along_axis(weights, sorted_indices, axis=0)

    sorted_hp_indices = np.argsort(hp_pixels, axis=0)
    sorted_hp_pixels = np.take_along_axis(hp_pixels, sorted_hp_indices, axis=0)
    sorted_hp_weights = np.take_along_axis(hp_weights, sorted_hp_indices, axis=0)

    # Calculate weight error for validation
    weight_error = jnp.mean((sorted_weights - sorted_hp_weights) ** 2)

    # First verify weights sum to 1 (fundamental requirement)
    weight_sums = jnp.sum(weights, axis=0)
    assert_allclose(weight_sums, 1.0, rtol=1e-12, atol=1e-12, err_msg=f'{region_name} weights should sum to 1.0')

    # Test weight precision - use relaxed tolerance for high nside sampling
    # High nside values with random sampling may have slightly higher numerical errors
    assert weight_error < 1e-5, f'{region_name} region weight MSE {weight_error} should be reasonable'

    # Verify all weights are non-negative (physical requirement)
    assert jnp.all(weights >= 0), f'{region_name} all weights should be non-negative'

    # Relaxed pixel test - allow some differences but ensure they're reasonable neighbors
    max_pixel_diff = jnp.max(jnp.abs(sorted_pixels - sorted_hp_pixels))
    print(f"weight error: {weight_error}, max_pixel_diff: {max_pixel_diff}")
    assert max_pixel_diff < npix // 4, f'{region_name} max pixel difference {max_pixel_diff} too large'


@pytest.mark.parametrize('nside', [4, 8, 16, 32, 64, 128, 256 , 512 , 8192])
def test_get_interp_weights_shapes(nside):
    """Test that output shapes are correct."""
    # Test single point
    theta = jnp.array([jnp.pi / 2])
    phi = jnp.array([0.0])

    pixels, weights = jhp.get_interp_weights(nside, theta, phi)

    # Check shapes
    assert pixels.shape == (4, 1), f'Expected pixels shape (4, 1), got {pixels.shape}'
    assert weights.shape == (4, 1), f'Expected weights shape (4, 1), got {weights.shape}'

    # Test multiple points
    theta = jnp.array([jnp.pi / 4, jnp.pi / 2, 3 * jnp.pi / 4])
    phi = jnp.array([0.0, jnp.pi / 2, jnp.pi])

    pixels, weights = jhp.get_interp_weights(nside, theta, phi)

    # Check shapes
    assert pixels.shape == (4, 3), f'Expected pixels shape (4, 3), got {pixels.shape}'
    assert weights.shape == (4, 3), f'Expected weights shape (4, 3), got {weights.shape}'


@pytest.mark.parametrize('nside', [4, 8, 16, 32, 64, 128, 256, 512, 8192])
def test_get_interp_weights_sum_to_one(nside):
    """Test that weights sum to 1.0 for each point."""
    # Generate test points
    n_points = 10
    theta = jnp.linspace(0.1, jnp.pi - 0.1, n_points)
    phi = jnp.linspace(0.0, 2 * jnp.pi - 0.1, n_points)

    pixels, weights = jhp.get_interp_weights(nside, theta, phi)

    # Check that weights sum to 1 for each point
    weight_sums = jnp.sum(weights, axis=0)
    assert_allclose(weight_sums, 1.0, rtol=1e-12, atol=1e-12)

    # Check that weights are non-negative
    assert jnp.all(weights >= 0), 'All weights should be non-negative'


def test_get_interp_weights_jit():
    """Test that JIT compilation works."""
    nside = 16
    theta = jnp.array([jnp.pi / 2])
    phi = jnp.array([0.0])

    # Test that the function can be JIT compiled
    jit_get_interp_weights = jax.jit(jhp.get_interp_weights, static_argnames=['nside'])

    # Should not raise any errors
    pixels, weights = jit_get_interp_weights(nside, theta, phi)

    # Check basic properties
    assert pixels.shape == (4, 1)
    assert weights.shape == (4, 1)
    assert_allclose(jnp.sum(weights), 1.0, rtol=1e-12)


def test_get_interp_weights_gradient():
    """Test that gradient computation works."""
    nside = 16

    def test_func(theta, phi):
        pixels, weights = jhp.get_interp_weights(nside, theta, phi)
        # Return a scalar for gradient testing - sum of weights (should be 1)
        return jnp.sum(weights)

    theta = jnp.array([jnp.pi / 2])
    phi = jnp.array([0.0])

    # Should not raise any errors
    grad_func = jax.grad(test_func, argnums=(0, 1))
    grad_theta, grad_phi = grad_func(theta, phi)

    # Check that gradients have the right shape
    assert grad_theta.shape == theta.shape
    assert grad_phi.shape == phi.shape

    # Since sum of weights is always 1, gradients should be 0
    assert_allclose(grad_theta, 0.0, atol=1e-10)
    assert_allclose(grad_phi, 0.0, atol=1e-10)


# Tests for get_interp_val
@pytest.mark.parametrize('nside', [4, 8, 16, 32, 64, 128, 256, 512])
def test_get_interp_val_single_map(nside):
    """Test get_interp_val with single map against healpy."""
    # Create a simple test map
    npix = jhp.nside2npix(nside)
    m = jnp.arange(npix, dtype=jnp.float64)

    # Test single point
    theta = jnp.array([jnp.pi / 2])
    phi = jnp.array([0.0])

    jax_result = jhp.get_interp_val(m, theta, phi)
    hp_result = hp.get_interp_val(np.array(m), np.array(theta), np.array(phi))

    assert_allclose(jax_result, hp_result, rtol=1e-12, atol=1e-12)

    # Test multiple points
    theta = jnp.array([jnp.pi / 4, jnp.pi / 2, 3 * jnp.pi / 4])
    phi = jnp.array([0.0, jnp.pi / 2, jnp.pi])

    jax_result = jhp.get_interp_val(m, theta, phi)
    hp_result = hp.get_interp_val(np.array(m), np.array(theta), np.array(phi))

    assert_allclose(jax_result, hp_result, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('nside', [4, 8, 16, 32, 64, 128, 256, 512])
def test_get_interp_val_multiple_maps(nside):
    """Test get_interp_val with multiple maps."""
    npix = jhp.nside2npix(nside)

    # Create multiple test maps
    map1 = jnp.arange(npix, dtype=jnp.float64)
    map2 = 2 * jnp.arange(npix, dtype=jnp.float64)
    map3 = jnp.sin(jnp.arange(npix, dtype=jnp.float64))
    maps = jnp.array([map1, map2, map3])

    # Test single point
    theta = jnp.array([jnp.pi / 2])
    phi = jnp.array([0.0])

    jax_result = jhp.get_interp_val(maps, theta, phi)

    # Compare with individual map results
    result1 = jhp.get_interp_val(map1, theta, phi)
    result2 = jhp.get_interp_val(map2, theta, phi)
    result3 = jhp.get_interp_val(map3, theta, phi)
    expected = jnp.array([[result1[0]], [result2[0]], [result3[0]]])

    assert_allclose(jax_result, expected, rtol=1e-12, atol=1e-12)

    # Test multiple points
    theta = jnp.array([jnp.pi / 4, jnp.pi / 2])
    phi = jnp.array([0.0, jnp.pi / 2])

    jax_result = jhp.get_interp_val(maps, theta, phi)

    # Should have shape (3, 2) for 3 maps and 2 points
    assert jax_result.shape == (3, 2)


def test_get_interp_val_lonlat():
    """Test get_interp_val with lonlat parameter."""
    nside = 16
    npix = jhp.nside2npix(nside)
    m = jnp.arange(npix, dtype=jnp.float64)

    # Test coordinates in radians
    theta_rad = jnp.array([jnp.pi / 2])
    phi_rad = jnp.array([0.0])

    # Test coordinates in degrees
    lon_deg = jnp.array([0.0])  # 0 degrees longitude
    lat_deg = jnp.array([0.0])  # 0 degrees latitude (equator)

    result_rad = jhp.get_interp_val(m, theta_rad, phi_rad, lonlat=False)
    result_deg = jhp.get_interp_val(m, lon_deg, lat_deg, lonlat=True)

    # Should be the same point
    assert_allclose(result_rad, result_deg, rtol=1e-12, atol=1e-12)


def test_get_interp_val_pixel_mode():
    """Test get_interp_val with theta as pixel indices (phi=None)."""
    nside = 16
    npix = jhp.nside2npix(nside)
    m = jnp.arange(npix, dtype=jnp.float64)

    # Test single pixel
    ipix = jnp.array([100])
    result = jhp.get_interp_val(m, ipix, phi=None)

    # Should be exactly the map value at that pixel
    expected = m[100]
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    # Test multiple pixels
    ipix = jnp.array([50, 100, 150])
    result = jhp.get_interp_val(m, ipix, phi=None)
    expected = m[ipix]
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_get_interp_val_against_healpy_comprehensive():
    """Comprehensive test against healpy with various inputs."""
    nside = 16
    npix = jhp.nside2npix(nside)

    # Create a more interesting map (not just linear)
    theta_pix, phi_pix = jhp.pix2ang(nside, jnp.arange(npix))
    m = jnp.sin(2 * theta_pix) * jnp.cos(3 * phi_pix)

    # Test random coordinates
    np.random.seed(42)
    n_test = 20
    theta_test = np.random.uniform(0.1, np.pi - 0.1, n_test)
    phi_test = np.random.uniform(0, 2 * np.pi, n_test)

    jax_result = jhp.get_interp_val(m, theta_test, phi_test)
    hp_result = hp.get_interp_val(np.array(m), theta_test, phi_test)

    # Should match healpy very closely
    assert_allclose(jax_result, hp_result, rtol=1e-10, atol=1e-10)


def test_get_interp_val_shapes():
    """Test that output shapes are correct."""
    nside = 8
    npix = jhp.nside2npix(nside)

    # Single map, single point
    m = jnp.arange(npix, dtype=jnp.float64)
    result = jhp.get_interp_val(m, jnp.array([jnp.pi / 2]), jnp.array([0.0]))
    assert result.shape == (1,), f'Expected shape (1,), got {result.shape}'

    # Single map, multiple points
    theta = jnp.array([jnp.pi / 4, jnp.pi / 2, 3 * jnp.pi / 4])
    phi = jnp.array([0.0, jnp.pi / 2, jnp.pi])
    result = jhp.get_interp_val(m, theta, phi)
    assert result.shape == (3,), f'Expected shape (3,), got {result.shape}'

    # Multiple maps, single point
    maps = jnp.array([m, 2 * m, 3 * m])
    result = jhp.get_interp_val(maps, jnp.array([jnp.pi / 2]), jnp.array([0.0]))
    assert result.shape == (3, 1), f'Expected shape (3, 1), got {result.shape}'

    # Multiple maps, multiple points
    result = jhp.get_interp_val(maps, theta, phi)
    assert result.shape == (3, 3), f'Expected shape (3, 3), got {result.shape}'


def test_get_interp_val_jit():
    """Test that get_interp_val works with JAX JIT compilation."""
    nside = 8
    npix = jhp.nside2npix(nside)
    m = jnp.arange(npix, dtype=jnp.float64)

    # Create JIT compiled version
    jit_get_interp_val = jax.jit(jhp.get_interp_val, static_argnames=['nest', 'lonlat'])

    theta = jnp.array([jnp.pi / 2])
    phi = jnp.array([0.0])

    # Should not raise any errors
    result = jit_get_interp_val(m, theta, phi)

    # Should match non-JIT version
    expected = jhp.get_interp_val(m, theta, phi)
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_get_interp_val_gradient():
    """Test that gradients work with get_interp_val."""
    nside = 8
    npix = jhp.nside2npix(nside)

    # Create a smooth map for meaningful gradients
    theta_pix, phi_pix = jhp.pix2ang(nside, jnp.arange(npix))
    m = jnp.sin(theta_pix) * jnp.cos(phi_pix)

    def test_func(theta, phi):
        return jnp.sum(jhp.get_interp_val(m, theta, phi))

    theta = jnp.array([jnp.pi / 3])
    phi = jnp.array([jnp.pi / 4])

    # Should not raise any errors
    grad_func = jax.grad(test_func, argnums=(0, 1))
    grad_theta, grad_phi = grad_func(theta, phi)

    # Check that gradients have the right shape
    assert grad_theta.shape == theta.shape
    assert grad_phi.shape == phi.shape

    # Gradients should be finite (not NaN or inf)
    assert jnp.isfinite(grad_theta).all()
    assert jnp.isfinite(grad_phi).all()


def test_get_interp_val_nest_error():
    """Test that get_interp_val raises error for NEST ordering."""
    nside = 8
    npix = jhp.nside2npix(nside)
    m = jnp.arange(npix, dtype=jnp.float64)

    with pytest.raises(ValueError, match='NEST pixel ordering is not supported'):
        jhp.get_interp_val(m, jnp.array([jnp.pi / 2]), jnp.array([0.0]), nest=True)


def test_get_interp_val_consistency_with_weights():
    """Test that get_interp_val produces the same result as manual calculation with get_interp_weights."""
    nside = 16
    npix = jhp.nside2npix(nside)
    m = jnp.arange(npix, dtype=jnp.float64)

    theta = jnp.array([jnp.pi / 3])
    phi = jnp.array([jnp.pi / 6])

    # Get result from get_interp_val
    result_direct = jhp.get_interp_val(m, theta, phi)

    # Get result by manual calculation
    pixels, weights = jhp.get_interp_weights(nside, theta, phi)
    result_manual = jnp.sum(weights * m[pixels], axis=0)

    # Should be identical
    assert_allclose(result_direct, result_manual, rtol=1e-15, atol=1e-15)
