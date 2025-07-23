# This file is part of jax-healpy.
# Copyright (C) 2024 CNRS / SciPol developers
#
# jax-healpy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# jax-healpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with jax-healpy. If not, see <https://www.gnu.org/licenses/>.

"""Comprehensive tests for get_all_neighbours function."""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import jax_healpy as jhp


@pytest.mark.parametrize('region_name', ['Low Cap', 'Equator', 'High Cap', 'All'])
@pytest.mark.parametrize('nside', [4, 8, 16, 32, 64, 128, 256])
def test_get_all_neighbours_comprehensive_precision(region_name, nside):
    """Test precision in different HEALPix regions with comprehensive pixel coverage."""
    npix = jhp.nside2npix(nside)

    # Define HEALPix regions exactly like test_interp.py
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
    if start >= end:
        pytest.skip(f'Empty region {region_name} for nside={nside}')

    # Test ALL pixels in region (comprehensive coverage)
    ipix = jnp.arange(start, end, dtype=jnp.int32)

    # Test 1: Pixel mode (direct pixel indices)
    jhp_neighbors_pixel = jhp.get_all_neighbours(nside, ipix, nest=False)
    hp_neighbors_pixel = hp.get_all_neighbours(nside, np.array(ipix), nest=False)

    # Exact matching required for pixel mode
    assert_array_equal(
        jhp_neighbors_pixel, hp_neighbors_pixel, f'{region_name} region nside {nside}: pixel mode must match exactly'
    )

    # Test 2: Angular mode with small perturbations (robustness test)
    # Convert pixels to angular coordinates
    theta, phi = jhp.pix2ang(nside, ipix)

    # Add small perturbations (smaller than pixel size to test robustness)
    # Ensure perturbations keep theta in valid range [0, π]
    key1, key2 = jax.random.split(jax.random.key(42), 2)
    theta_perturbation = jax.random.uniform(key1, theta.shape, minval=-0.005, maxval=0.005)
    phi_perturbation = jax.random.uniform(key2, phi.shape, minval=-0.005, maxval=0.005)

    # Clamp theta to valid range [0, π]
    theta_perturb = jnp.clip(theta + theta_perturbation, 0.0, jnp.pi)
    phi_perturb = phi + phi_perturbation

    # Get neighbors using perturbed angular coordinates
    jhp_neighbors_angular = jhp.get_all_neighbours(nside, theta_perturb, phi_perturb, nest=False)
    hp_neighbors_angular = hp.get_all_neighbours(nside, np.array(theta_perturb), np.array(phi_perturb), nest=False)

    # Exact matching required even with perturbations
    assert_array_equal(
        jhp_neighbors_angular,
        hp_neighbors_angular,
        f'{region_name} region nside {nside}: angular mode with perturbations must match exactly',
    )

    # Test 3: Consistency between pixel and exact angular modes
    theta_exact, phi_exact = jhp.pix2ang(nside, ipix)
    jhp_neighbors_exact = jhp.get_all_neighbours(nside, theta_exact, phi_exact, nest=False)

    # Pixel mode and exact angular mode should give identical results
    assert_array_equal(
        jhp_neighbors_pixel,
        jhp_neighbors_exact,
        f'{region_name} region nside {nside}: pixel and exact angular modes must be consistent',
    )


@pytest.mark.slow
@pytest.mark.parametrize('region_name', ['Low Cap', 'Equator', 'High Cap', 'All'])
@pytest.mark.parametrize('nside', [512, 1024, 2048, 4096, 8192])
def test_get_all_neighbours_high_nside_sampling(region_name, nside):
    """Test precision for high nside values using random sampling to avoid memory issues."""
    npix = jhp.nside2npix(nside)

    # Define HEALPix regions exactly like the comprehensive test
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
    if start >= end:
        pytest.skip(f'Empty region {region_name} for nside={nside}')

    region_size = end - start

    # Use random sampling to test substantial number of pixels from the region
    n_samples = min(5000, region_size)  # Test up to 5000 pixels per region

    # Generate random pixel indices within the region
    key = jax.random.key(11)  # Fixed seed for reproducibility
    if region_size > n_samples:
        # Sample random indices within the region
        random_offsets = jax.random.choice(key, region_size, (n_samples,), replace=False)
        ipix = start + random_offsets
    else:
        # If region is small, test all pixels
        ipix = jnp.arange(start, end, dtype=jnp.int32)

    # Test pixel mode
    jhp_neighbors = jhp.get_all_neighbours(nside, ipix, nest=False)
    hp_neighbors = hp.get_all_neighbours(nside, np.array(ipix), nest=False)

    # Exact matching required
    assert_array_equal(
        jhp_neighbors, hp_neighbors, f'{region_name} region nside {nside}: high nside sampling must match exactly'
    )

    # Test angular mode with perturbations
    theta, phi = jhp.pix2ang(nside, ipix)
    key1, key2 = jax.random.split(key, 2)
    theta_perturbation = jax.random.uniform(key1, theta.shape, minval=-0.003, maxval=0.003)
    phi_perturbation = jax.random.uniform(key2, phi.shape, minval=-0.003, maxval=0.003)

    # Clamp theta to valid range [0, π]
    theta_perturb = jnp.clip(theta + theta_perturbation, 0.0, jnp.pi)
    phi_perturb = phi + phi_perturbation

    jhp_neighbors_angular = jhp.get_all_neighbours(nside, theta_perturb, phi_perturb, nest=False)
    hp_neighbors_angular = hp.get_all_neighbours(nside, np.array(theta_perturb), np.array(phi_perturb), nest=False)

    assert_array_equal(
        jhp_neighbors_angular,
        hp_neighbors_angular,
        f'{region_name} region nside {nside}: high nside angular mode must match exactly',
    )


@pytest.mark.parametrize('nside', [4, 16, 64])
@pytest.mark.parametrize('nest', [False])  # Only test RING mode since pix2ang NEST not implemented
def test_get_all_neighbours_angular_coordinates(nside, nest):
    """Test get_all_neighbours with various angular coordinate inputs."""
    npix = jhp.nside2npix(nside)

    # Test a selection of pixels across different regions
    test_pixels = jnp.array([0, npix // 4, npix // 2, 3 * npix // 4, npix - 1], dtype=jnp.int32)

    # Test 1: Exact angular coordinates (should match pixel mode)
    theta, phi = jhp.pix2ang(nside, test_pixels, nest=nest)
    jhp_neighbors_pixel = jhp.get_all_neighbours(nside, test_pixels, nest=nest)
    jhp_neighbors_angular = jhp.get_all_neighbours(nside, theta, phi, nest=nest)

    assert_array_equal(
        jhp_neighbors_pixel,
        jhp_neighbors_angular,
        f'nside {nside} nest={nest}: pixel and exact angular modes must match',
    )

    # Test 2: Small perturbations
    key1, key2 = jax.random.split(jax.random.key(123), 2)
    theta_perturbation = jax.random.uniform(key1, theta.shape, minval=-0.005, maxval=0.005)
    phi_perturbation = jax.random.uniform(key2, phi.shape, minval=-0.005, maxval=0.005)

    # Clamp theta to valid range [0, π]
    theta_perturb = jnp.clip(theta + theta_perturbation, 0.0, jnp.pi)
    phi_perturb = phi + phi_perturbation

    jhp_neighbors_perturb = jhp.get_all_neighbours(nside, theta_perturb, phi_perturb, nest=nest)
    hp_neighbors_perturb = hp.get_all_neighbours(nside, np.array(theta_perturb), np.array(phi_perturb), nest=nest)

    assert_array_equal(
        jhp_neighbors_perturb,
        hp_neighbors_perturb,
        f'nside {nside} nest={nest}: perturbed angular coordinates must match healpy',
    )


@pytest.mark.parametrize('nside', [4, 16, 64])
def test_get_all_neighbours_lonlat_mode(nside):
    """Test get_all_neighbours with longitude/latitude inputs in degrees."""
    npix = jhp.nside2npix(nside)
    test_pixels = jnp.array([0, npix // 4, npix // 2, 3 * npix // 4, npix - 1], dtype=jnp.int32)

    # Convert to longitude/latitude in degrees
    theta, phi = jhp.pix2ang(nside, test_pixels, nest=False)
    lat_deg = 90.0 - jnp.rad2deg(theta)  # Convert colatitude to latitude
    lon_deg = jnp.rad2deg(phi)

    # Test lonlat mode
    jhp_neighbors_lonlat = jhp.get_all_neighbours(nside, lon_deg, lat_deg, nest=False, lonlat=True)
    jhp_neighbors_pixel = jhp.get_all_neighbours(nside, test_pixels, nest=False)

    assert_array_equal(jhp_neighbors_lonlat, jhp_neighbors_pixel, f'nside {nside}: lonlat mode must match pixel mode')

    # Test with small perturbations in degrees
    key1, key2 = jax.random.split(jax.random.key(456), 2)
    lon_perturb = lon_deg + jax.random.uniform(key1, lon_deg.shape, minval=-0.5, maxval=0.5)
    lat_perturb = lat_deg + jax.random.uniform(key2, lat_deg.shape, minval=-0.5, maxval=0.5)

    jhp_neighbors_lonlat_perturb = jhp.get_all_neighbours(nside, lon_perturb, lat_perturb, nest=False, lonlat=True)

    # Convert back to theta/phi for healpy comparison
    theta_from_lonlat = jnp.deg2rad(90.0 - lat_perturb)
    phi_from_lonlat = jnp.deg2rad(lon_perturb)
    hp_neighbors_lonlat = hp.get_all_neighbours(
        nside, np.array(theta_from_lonlat), np.array(phi_from_lonlat), nest=False
    )

    assert_array_equal(
        jhp_neighbors_lonlat_perturb, hp_neighbors_lonlat, f'nside {nside}: perturbed lonlat mode must match healpy'
    )


@pytest.mark.parametrize('nside', [4, 16, 64])
@pytest.mark.parametrize('nest', [False])  # Only test RING mode since pix2ang NEST not implemented
def test_get_all_neighbours_jit(nside, nest):
    """Test get_all_neighbours works correctly under JIT compilation."""
    npix = jhp.nside2npix(nside)
    test_pixels = jnp.array([0, npix // 4, npix // 2, npix - 1], dtype=jnp.int32)

    # Non-JIT version
    neighbors_no_jit = jhp.get_all_neighbours(nside, test_pixels, nest=nest)

    # JIT compiled version
    jit_get_neighbors = jax.jit(jhp.get_all_neighbours, static_argnames=['nside', 'nest'])
    neighbors_jit = jit_get_neighbors(nside, test_pixels, nest=nest)

    assert_array_equal(
        neighbors_no_jit, neighbors_jit, f'nside {nside} nest={nest}: JIT and non-JIT results must match'
    )

    # Test angular coordinates with JIT
    theta, phi = jhp.pix2ang(nside, test_pixels, nest=nest)
    neighbors_angular_no_jit = jhp.get_all_neighbours(nside, theta, phi, nest=nest)

    jit_get_neighbors_angular = jax.jit(jhp.get_all_neighbours, static_argnames=['nside', 'nest'])
    neighbors_angular_jit = jit_get_neighbors_angular(nside, theta, phi, nest=nest)

    assert_array_equal(
        neighbors_angular_no_jit,
        neighbors_angular_jit,
        f'nside {nside} nest={nest}: JIT angular mode must match non-JIT',
    )


@pytest.mark.parametrize('nside', [4, 16])
def test_get_all_neighbours_grad(nside):
    """Test gradient computation through get_all_neighbours for angular inputs."""
    # Use a single pixel for gradient testing
    test_pixel = jnp.array([nside**2], dtype=jnp.int32)  # Middle-ish pixel
    theta, phi = jhp.pix2ang(nside, test_pixel, nest=False)

    def neighbors_sum(theta_val, phi_val):
        """Sum of valid neighbors (for gradient testing)."""
        neighbors = jhp.get_all_neighbours(nside, theta_val, phi_val, nest=False)
        # Sum only valid neighbors (not -1)
        valid_neighbors = jnp.where(neighbors >= 0, neighbors, 0)
        return jnp.sum(valid_neighbors, dtype=jnp.float32)

    # Compute gradients
    grad_fn = jax.grad(neighbors_sum, argnums=(0, 1))
    theta_grad, phi_grad = grad_fn(theta[0], phi[0])

    # Gradients should be finite (not NaN or infinite)
    assert jnp.isfinite(theta_grad), f'nside {nside}: theta gradient must be finite'
    assert jnp.isfinite(phi_grad), f'nside {nside}: phi gradient must be finite'


@pytest.mark.parametrize('nside', [4, 16, 64])
def test_get_all_neighbours_nest_vs_ring(nside):
    """Test consistency between NEST and RING ordering schemes."""
    npix = jhp.nside2npix(nside)
    test_pixels_ring = jnp.array([0, npix // 4, npix // 2, npix - 1], dtype=jnp.int32)

    # Convert RING pixels to NEST
    test_pixels_nest = jhp.ring2nest(nside, test_pixels_ring)

    # Get neighbors in both schemes
    neighbors_ring = jhp.get_all_neighbours(nside, test_pixels_ring, nest=False)
    neighbors_nest = jhp.get_all_neighbours(nside, test_pixels_nest, nest=True)

    # Convert RING neighbors to NEST for comparison
    neighbors_ring_as_nest = jnp.where(neighbors_ring >= 0, jhp.ring2nest(nside, neighbors_ring), -1)

    assert_array_equal(
        neighbors_nest, neighbors_ring_as_nest, f'nside {nside}: NEST and RING schemes must be consistent'
    )

    # Test with healpy for verification
    hp_neighbors_ring = hp.get_all_neighbours(nside, np.array(test_pixels_ring), nest=False)
    hp_neighbors_nest = hp.get_all_neighbours(nside, np.array(test_pixels_nest), nest=True)

    assert_array_equal(neighbors_ring, hp_neighbors_ring, f'nside {nside}: RING mode must match healpy')
    assert_array_equal(neighbors_nest, hp_neighbors_nest, f'nside {nside}: NEST mode must match healpy')


@pytest.mark.parametrize('nside', [4, 8, 16])
def test_get_all_neighbours_pixel_vs_angular_consistency(nside):
    """Test consistency between pixel indices and exact angular coordinates."""
    npix = jhp.nside2npix(nside)

    # Test with various pixels including boundary cases
    test_pixels = jnp.array(
        [
            0,
            1,
            2,
            3,  # First few pixels (pole region)
            npix // 2 - 1,
            npix // 2,
            npix // 2 + 1,  # Around middle
            npix - 4,
            npix - 3,
            npix - 2,
            npix - 1,  # Last few pixels (other pole)
        ],
        dtype=jnp.int32,
    )

    # Test RING mode only (NEST mode pix2ang not implemented)
    for nest in [False]:
        # Get neighbors using pixel indices
        neighbors_pixel = jhp.get_all_neighbours(nside, test_pixels, nest=nest)

        # Get exact angular coordinates and compute neighbors
        theta, phi = jhp.pix2ang(nside, test_pixels, nest=nest)
        neighbors_angular = jhp.get_all_neighbours(nside, theta, phi, nest=nest)

        # Must be exactly equal
        assert_array_equal(
            neighbors_pixel,
            neighbors_angular,
            f'nside {nside} nest={nest}: pixel and angular modes must be perfectly consistent',
        )

        # Verify against healpy
        hp_neighbors = hp.get_all_neighbours(nside, np.array(test_pixels), nest=nest)
        assert_array_equal(neighbors_pixel, hp_neighbors, f'nside {nside} nest={nest}: must match healpy exactly')

    # Test NEST mode separately without pix2ang (pixel mode only)
    for nest in [True]:
        # Get neighbors using pixel indices
        neighbors_pixel = jhp.get_all_neighbours(nside, test_pixels, nest=nest)

        # Verify against healpy
        hp_neighbors = hp.get_all_neighbours(nside, np.array(test_pixels), nest=nest)
        assert_array_equal(neighbors_pixel, hp_neighbors, f'nside {nside} nest={nest}: must match healpy exactly')


@pytest.mark.parametrize('nside', [4, 8, 16, 32, 64])
@pytest.mark.parametrize('nest', [False, True])
def test_get_all_neighbours_get_center_pixel_mode(nside, nest):
    """Test get_center parameter functionality with pixel mode input.

    This test validates that:
    1. get_center=False (default) maintains perfect healpy compatibility
    2. get_center=True returns 9 pixels (center + 8 neighbors) in correct order
    3. The center pixel is correctly identified as the first element
    4. Neighbor pixels match between the two modes
    """
    # Test with a variety of pixels across different regions
    npix = jhp.nside2npix(nside)
    test_pixels = np.array([0, npix // 4, npix // 2, 3 * npix // 4, npix - 1])

    # Test get_center=False (default behavior, should match healpy)
    neighbors_false = jhp.get_all_neighbours(nside, test_pixels, nest=nest, get_center=False)
    hp_neighbors = hp.get_all_neighbours(nside, test_pixels, nest=nest)
    assert_array_equal(neighbors_false, hp_neighbors, 'get_center=False must match healpy exactly')
    assert neighbors_false.shape == (8, len(test_pixels)), 'get_center=False should return 8 neighbors'

    # Test get_center=True (should return 9 pixels: center + 8 neighbors)
    neighbors_true = jhp.get_all_neighbours(nside, test_pixels, nest=nest, get_center=True)
    assert neighbors_true.shape == (9, len(test_pixels)), 'get_center=True should return 9 pixels'

    # First pixel should be the center pixel
    assert_array_equal(neighbors_true[0], test_pixels, 'First pixel should be center pixel')

    # Remaining 8 pixels should match the regular neighbors
    assert_array_equal(neighbors_true[1:], neighbors_false, 'Neighbors should match get_center=False result')


@pytest.mark.parametrize('nside', [4, 16, 64])
def test_get_all_neighbours_get_center_angular_mode(nside):
    """Test get_center parameter functionality with angular coordinate input.

    This test validates the get_center feature works correctly when input is provided
    as (theta, phi) angular coordinates rather than pixel indices, ensuring:
    1. Backward compatibility with healpy for get_center=False
    2. Correct center pixel identification from angular coordinates
    3. Consistent neighbor ordering and shape changes for get_center=True
    """
    # Test with angular coordinates across different regions
    test_pixels = np.array([0, jhp.nside2npix(nside) // 4, jhp.nside2npix(nside) // 2, jhp.nside2npix(nside) - 1])
    theta, phi = jhp.pix2ang(nside, test_pixels, nest=False)

    # Test get_center=False
    neighbors_false = jhp.get_all_neighbours(nside, theta, phi, nest=False, get_center=False)
    hp_neighbors = hp.get_all_neighbours(nside, test_pixels, nest=False)
    assert_array_equal(neighbors_false, hp_neighbors, 'Angular mode get_center=False must match healpy')

    # Test get_center=True
    neighbors_true = jhp.get_all_neighbours(nside, theta, phi, nest=False, get_center=True)
    assert neighbors_true.shape == (9, len(test_pixels)), 'Angular mode should return 9 pixels'

    # First pixel should be the center pixel
    assert_array_equal(neighbors_true[0], test_pixels, 'First pixel should be center pixel')

    # Remaining 8 pixels should match the regular neighbors
    assert_array_equal(neighbors_true[1:], neighbors_false, 'Neighbors should match get_center=False result')


@pytest.mark.parametrize('nside', [4, 16])
def test_get_all_neighbours_get_center_lonlat_basic(nside):
    """Test get_center=True functionality with basic longitude/latitude input."""
    # Use a single well-behaved coordinate
    lon = np.array([0.0])
    lat = np.array([0.0])  # Equator

    # Test get_center=False (should return 8 neighbors)
    neighbors_false = jhp.get_all_neighbours(nside, lat, lon, nest=False, lonlat=True, get_center=False)
    assert neighbors_false.shape == (8, 1), 'get_center=False should return 8 neighbors'

    # Test get_center=True (should return 9 pixels: center + 8 neighbors)
    neighbors_true = jhp.get_all_neighbours(nside, lat, lon, nest=False, lonlat=True, get_center=True)
    assert neighbors_true.shape == (9, 1), 'get_center=True should return 9 pixels'

    # Center pixel should be at index 0
    center_pixel = jhp.ang2pix(nside, lat, lon, nest=False, lonlat=True)
    assert neighbors_true[0] == center_pixel, 'First pixel should be center pixel'

    # Remaining 8 pixels should match the regular neighbors
    assert_array_equal(neighbors_true[1:], neighbors_false, 'Neighbors should match get_center=False result')


@pytest.mark.parametrize('nside', [4, 16])
@pytest.mark.parametrize('nest', [False, True])
def test_get_all_neighbours_get_center_jit_compatibility(nside, nest):
    """Test that get_center parameter works with JIT compilation."""
    test_pixels = np.array([0, jhp.nside2npix(nside) // 2, jhp.nside2npix(nside) - 1])

    # Test JIT compilation works by calling the already JIT-ed function
    # (get_all_neighbours is already decorated with jit)
    neighbors_false_jit = jhp.get_all_neighbours(nside, test_pixels, nest=nest, get_center=False)
    neighbors_true_jit = jhp.get_all_neighbours(nside, test_pixels, nest=nest, get_center=True)

    # Verify shapes and relationships work correctly with JIT
    assert neighbors_false_jit.shape == (8, len(test_pixels)), 'JIT get_center=False should return correct shape'
    assert neighbors_true_jit.shape == (9, len(test_pixels)), 'JIT get_center=True should return correct shape'
    assert_array_equal(neighbors_true_jit[0], test_pixels, 'JIT center pixels must be correct')
    assert_array_equal(neighbors_true_jit[1:], neighbors_false_jit, 'JIT neighbors must match')

    # Test multiple calls work correctly (JIT caching)
    neighbors_false_jit2 = jhp.get_all_neighbours(nside, test_pixels, nest=nest, get_center=False)
    neighbors_true_jit2 = jhp.get_all_neighbours(nside, test_pixels, nest=nest, get_center=True)

    assert_array_equal(neighbors_false_jit, neighbors_false_jit2, 'JIT cached calls must be consistent')
    assert_array_equal(neighbors_true_jit, neighbors_true_jit2, 'JIT cached calls must be consistent')


def test_get_all_neighbours_get_center_single_pixel():
    """Test get_center functionality with single pixel input."""
    nside = 16
    ipix = 100

    # Test scalar input
    neighbors_false = jhp.get_all_neighbours(nside, ipix, nest=False, get_center=False)
    neighbors_true = jhp.get_all_neighbours(nside, ipix, nest=False, get_center=True)

    # Verify shapes for scalar input
    assert neighbors_false.shape == (8,), 'Scalar input get_center=False should return (8,)'
    assert neighbors_true.shape == (9,), 'Scalar input get_center=True should return (9,)'

    # Verify center pixel
    assert neighbors_true[0] == ipix, 'First element should be center pixel'

    # Verify neighbors match
    assert_array_equal(neighbors_true[1:], neighbors_false, 'Neighbors should match')

    # Verify against healpy
    hp_neighbors = hp.get_all_neighbours(nside, ipix, nest=False)
    assert_array_equal(neighbors_false, hp_neighbors, 'Must match healpy exactly')


@pytest.mark.parametrize('nside', [4, 16])
def test_get_all_neighbours_get_center_comprehensive_consistency(nside):
    """Test comprehensive consistency between get_center modes across input types.

    This is a critical test that validates the get_center functionality works
    consistently across different input modes (pixel indices vs angular coordinates)
    and maintains internal consistency between get_center=False and get_center=True.

    Key validations:
    1. Pixel mode and angular mode give identical results for both get_center values
    2. Center pixels are correctly identified across all input modes
    3. Neighbor relationships are preserved between modes
    4. Shape transformations work correctly for both scalar and array inputs
    """
    # Test pixels from different regions
    npix = jhp.nside2npix(nside)
    test_pixels = np.array([0, npix // 8, npix // 4, npix // 2, 3 * npix // 4, npix - 1])

    # Get neighbors with both modes
    neighbors_false = jhp.get_all_neighbours(nside, test_pixels, nest=False, get_center=False)
    neighbors_true = jhp.get_all_neighbours(nside, test_pixels, nest=False, get_center=True)

    # Test with angular coordinates
    theta, phi = jhp.pix2ang(nside, test_pixels, nest=False)
    neighbors_angular_false = jhp.get_all_neighbours(nside, theta, phi, nest=False, get_center=False)
    neighbors_angular_true = jhp.get_all_neighbours(nside, theta, phi, nest=False, get_center=True)

    # All methods should be consistent
    assert_array_equal(
        neighbors_false, neighbors_angular_false, 'Pixel and angular modes must match (get_center=False)'
    )
    assert_array_equal(neighbors_true, neighbors_angular_true, 'Pixel and angular modes must match (get_center=True)')

    # Center pixels should be correct
    assert_array_equal(neighbors_true[0], test_pixels, 'Center pixels must be correct')
    assert_array_equal(neighbors_angular_true[0], test_pixels, 'Angular mode center pixels must be correct')

    # Neighbors should match between modes
    assert_array_equal(neighbors_true[1:], neighbors_false, 'Neighbors must match between get_center modes')
    assert_array_equal(
        neighbors_angular_true[1:], neighbors_angular_false, 'Angular neighbors must match between get_center modes'
    )
