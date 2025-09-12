import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_healpy as jhp
from jax_healpy._query_disc import _query_disc_bruteforce


def test_basic_functionality():
    """Test basic functionality of query_disc."""
    nside = 16
    vec = [1.0, 0.0, 0.0]  # Point on equator
    radius = 0.1  # ~5.7 degrees

    # Test basic call
    ipix = jhp.query_disc(nside, vec, radius)
    assert len(ipix) == jhp.nside2npix(nside), 'Should return fixed-size array'

    # Count valid pixels (those < npix)
    npix = jhp.nside2npix(nside)
    valid_pixels = ipix[ipix < npix]
    assert len(valid_pixels) > 0, 'Should find some pixels'

    # Test with different parameters
    ipix_inclusive = jhp.query_disc(nside, vec, radius, inclusive=True)
    valid_pixels_inclusive = ipix_inclusive[ipix_inclusive < npix]
    assert len(valid_pixels_inclusive) >= len(valid_pixels), 'Inclusive should return more or equal pixels'


def test_map_indexing_example():
    """Test the map indexing example provided by user."""
    nside = 16

    # JAX-healpy version
    jhp_map = jnp.zeros((jhp.nside2npix(nside),), dtype=jnp.float32)
    central_pix = jhp.ang2pix(nside, np.pi / 2, 0.0, lonlat=False)
    central_vec = jhp.pix2vec(nside, central_pix)
    disc = jhp.query_disc(nside, central_vec, np.pi / 4, inclusive=True)
    jhp_map = jhp_map.at[disc].set(1.0)

    # healpy version
    hp_map = np.zeros((hp.nside2npix(nside),), dtype=np.float32)
    central_pix_hp = hp.ang2pix(nside, np.pi / 2, 0.0, lonlat=False)
    central_vec_hp = hp.pix2vec(nside, central_pix_hp)
    disc_hp = hp.query_disc(nside, central_vec_hp, np.pi / 4, inclusive=True)
    hp_map[disc_hp] = 1.0

    # Compare the maps - they should be reasonably similar
    # Allow for some differences in inclusive mode due to algorithm differences
    jhp_marked = np.where(jhp_map > 0.5)[0]
    hp_marked = np.where(hp_map > 0.5)[0]

    # Check overlap ratio rather than exact match for inclusive mode
    overlap = len(set(jhp_marked) & set(hp_marked))
    total_unique = len(set(jhp_marked) | set(hp_marked))
    overlap_ratio = overlap / total_unique if total_unique > 0 else 1.0
    assert overlap_ratio > 0.7, f'Insufficient map overlap: {overlap_ratio:.2f}'


@pytest.mark.parametrize(
    'nside,vec,radius,inclusive,description',
    [
        (16, [1.0, 0.0, 0.0], 0.1, False, 'Small disc on equator'),
        (16, [0.0, 0.0, 1.0], 0.2, False, 'Small disc at north pole'),
        (16, [0.0, 0.0, -1.0], 0.15, False, 'Small disc at south pole'),
        (32, [1.0, 0.0, 0.0], 0.05, False, 'Very small disc'),
        (8, [0.5, 0.5, 0.707], 0.3, False, 'Medium disc at arbitrary position'),
        (16, [1.0, 0.0, 0.0], 0.1, True, 'Small disc inclusive mode'),
        (16, [0.0, 0.0, 1.0], 0.2, True, 'Pole disc inclusive mode'),
    ],
)
def test_against_healpy_comprehensive(nside, vec, radius, inclusive, description):
    """Comprehensive test against healpy implementation."""
    # JAX-healpy result
    jax_result = jhp.query_disc(nside, vec, radius, inclusive=inclusive)

    # healpy result
    healpy_result = hp.query_disc(nside, vec, radius, inclusive=inclusive)

    # Extract valid pixels from jax result
    npix = jhp.nside2npix(nside)
    jax_valid = jax_result[jax_result < npix]

    # Sort both results for comparison
    jax_sorted = np.sort(jax_valid)
    healpy_sorted = np.sort(healpy_result)

    # Compare results - allow for some differences in inclusive mode
    if inclusive:
        # For inclusive mode, check that results are reasonably close
        overlap = len(set(jax_sorted) & set(healpy_sorted))
        total_unique = len(set(jax_sorted) | set(healpy_sorted))
        overlap_ratio = overlap / total_unique if total_unique > 0 else 1.0
        assert overlap_ratio > 0.7, f'Insufficient overlap for {description}: {overlap_ratio:.2f}'
    else:
        # For non-inclusive mode, expect exact match
        np.testing.assert_array_equal(jax_sorted, healpy_sorted, err_msg=f'Failed for: {description}')


def test_edge_cases():
    """Test edge cases and special conditions."""
    nside = 16
    vec = [1.0, 0.0, 0.0]
    npix = jhp.nside2npix(nside)

    # Test with zero radius
    ipix = jhp.query_disc(nside, vec, 0.0)
    valid_pixels = ipix[ipix < npix]
    # Zero radius correctly returns no pixels (both JAX-healpy and healpy)
    assert len(valid_pixels) == 0, 'Zero radius should return no pixels'

    # Test with very large radius (full sphere)
    ipix = jhp.query_disc(nside, vec, np.pi)
    valid_pixels = ipix[ipix < npix]
    assert len(valid_pixels) == npix, 'Full sphere should return all pixels'

    # Test with non-unit vector (should be normalized)
    vec_unnorm = [2.0, 0.0, 0.0]
    ipix1 = jhp.query_disc(nside, vec, 0.1)
    ipix2 = jhp.query_disc(nside, vec_unnorm, 0.1)
    np.testing.assert_array_equal(ipix1, ipix2, 'Non-unit vector should be normalized')


def test_input_validation():
    """Test input validation and error conditions."""
    nside = 16
    vec = [1.0, 0.0, 0.0]

    # Test zero vector (should be handled gracefully)
    ipix = jhp.query_disc(nside, [0.0, 0.0, 0.0], 0.1)
    assert len(ipix) == jhp.nside2npix(nside), 'Zero vector should be handled'

    # Test negative radius (should be clipped)
    ipix = jhp.query_disc(nside, vec, -0.1)
    assert len(ipix) == jhp.nside2npix(nside), 'Negative radius should be clipped'


def test_jit_compatibility():
    """Test that the function is JIT-compatible."""
    import jax

    nside = 16
    vec = jnp.array([1.0, 0.0, 0.0])
    radius = 0.1

    # JIT compile the function with nside as static argument
    jit_query_disc = jax.jit(lambda n, v, r: jhp.query_disc(n, v, r), static_argnums=(0,))

    # Test that it works
    result = jit_query_disc(nside, vec, radius)
    assert len(result) == jhp.nside2npix(nside), 'JIT-compiled function should work'

    # Test that multiple calls work (compilation caching)
    result2 = jit_query_disc(nside, vec, radius)
    np.testing.assert_array_equal(result, result2, 'JIT-compiled function should be deterministic')


def test_differentiability():
    """Test that the function is differentiable w.r.t. vec and radius."""

    nside = 8  # Small nside for faster computation
    vec = jnp.array([1.0, 0.0, 0.0])
    radius = 0.2

    # Test gradient w.r.t. vec
    def loss_vec(v):
        # Use JIT with static nside for performance
        jit_query = jax.jit(lambda n, v, r: jhp.query_disc(n, v, r), static_argnums=(0,))
        disc = jit_query(nside, v, radius)
        npix = jhp.nside2npix(nside)
        valid_count = jnp.sum(disc < npix)
        return valid_count.astype(jnp.float32)

    grad_vec = jax.grad(loss_vec)(vec)
    assert grad_vec.shape == (3,), 'Gradient w.r.t. vec should have shape (3,)'

    # Test gradient w.r.t. radius
    def loss_radius(r):
        # Use JIT with static nside for performance
        jit_query = jax.jit(lambda n, v, r: jhp.query_disc(n, v, r), static_argnums=(0,))
        disc = jit_query(nside, vec, r)
        npix = jhp.nside2npix(nside)
        valid_count = jnp.sum(disc < npix)
        return valid_count.astype(jnp.float32)

    grad_radius = jax.grad(loss_radius)(radius)
    assert isinstance(grad_radius, jnp.ndarray), 'Gradient w.r.t. radius should be computed'


@pytest.mark.parametrize('radius', [0.01, 0.1, 0.5, 1.0])
def test_fixed_size_output(radius):
    """Test that output has fixed size with npix sentinel values."""
    nside = 16
    vec = [1.0, 0.0, 0.0]
    npix = jhp.nside2npix(nside)

    result = jhp.query_disc(nside, vec, radius)
    assert len(result) == npix, f'Output should have fixed size {npix}'

    # Check that sentinel values are npix
    invalid_mask = result == npix
    valid_mask = result < npix
    assert np.all(result[invalid_mask] == npix), 'Invalid pixels should have value npix'
    assert np.all(result[valid_mask] < npix), 'Valid pixels should have value < npix'


def test_batch_functionality():
    """Test batched input functionality with multiple disc centers."""
    nside = 16
    npix = jhp.nside2npix(nside)
    radius = 0.1
    max_length = 100

    # Test batch of 3 vectors
    vecs = jnp.array(
        [
            [1.0, 0.0, 0.0],  # equator
            [0.0, 1.0, 0.0],  # equator, 90 degrees rotated
            [0.0, 0.0, 1.0],  # north pole
        ]
    ).T  # Shape (3, 3)

    # Test batch query
    batch_result = jhp.query_disc(nside, vecs, radius, max_length=max_length)
    assert batch_result.shape == (3, max_length), f'Expected (3, {max_length}), got {batch_result.shape}'

    # Test individual queries for comparison
    for i in range(3):
        single_vec = vecs[:, i]
        single_result = jhp.query_disc(nside, single_vec, radius, max_length=max_length)

        # Get valid pixels from both
        batch_valid = batch_result[i][batch_result[i] < npix]
        single_valid = single_result[single_result < npix]

        # Should have same valid pixels (sorted)
        np.testing.assert_array_equal(
            np.sort(batch_valid),
            np.sort(single_valid),
            err_msg=f"Batch result for vector {i} doesn't match single result",
        )

    # Test that invalid pixels are marked as npix
    invalid_mask = batch_result == npix
    valid_mask = batch_result < npix
    assert np.all(batch_result[invalid_mask] == npix), 'Invalid pixels should be npix'
    assert np.all(batch_result[valid_mask] < npix), 'Valid pixels should be < npix'

    # Test with single vector (should squeeze back)
    single_vec = vecs[:, 0]
    single_result = jhp.query_disc(nside, single_vec, radius, max_length=max_length)
    assert single_result.shape == (max_length,), f'Single vector should return ({max_length},)'

    # Compare with first batch element
    batch_first = batch_result[0]
    np.testing.assert_array_equal(single_result, batch_first, 'Single vector result should match first batch element')


def test_batch_edge_cases():
    """Test edge cases for batch functionality."""
    nside = 8  # Small for faster testing
    npix = jhp.nside2npix(nside)

    # Test with very small max_length (truncation)
    vecs = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    max_length = 5

    result = jhp.query_disc(nside, vecs, 0.5, max_length=max_length)  # Large radius
    assert result.shape == (2, max_length), f'Expected (2, {max_length})'

    # Should have valid pixels followed by npix padding
    for i in range(2):
        row = result[i]
        valid_pixels = row[row < npix]
        assert len(valid_pixels) <= max_length, 'Should not exceed max_length valid pixels'

    # Test with max_length larger than typical disc size
    large_max_length = npix // 2
    result2 = jhp.query_disc(nside, vecs, 0.1, max_length=large_max_length)  # Small radius
    assert result2.shape == (2, large_max_length), f'Expected (2, {large_max_length})'

    # Most entries should be npix (padding)
    for i in range(2):
        row = result2[i]
        valid_pixels = row[row < npix]
        padding_pixels = row[row == npix]
        assert len(valid_pixels) + len(padding_pixels) == large_max_length
        assert len(valid_pixels) < large_max_length  # Should have some padding


def test_estimate_disc_pixel_count():
    """Test the estimate_disc_pixel_count function covering all edge cases."""
    nside = 16
    npix = jhp.nside2npix(nside)

    # Test normal case
    radius = 0.1
    estimated = jhp.estimate_disc_pixel_count(nside, radius)
    assert isinstance(estimated, int), 'Should return integer'
    assert estimated > 0, 'Should return positive count for positive radius'

    # Test zero radius
    estimated_zero = jhp.estimate_disc_pixel_count(nside, 0.0)
    assert estimated_zero == 0, 'Zero radius should return zero pixels'

    # Test full sphere case (radius >= π)
    estimated_full = jhp.estimate_disc_pixel_count(nside, np.pi)
    assert estimated_full == npix, 'Full sphere radius should return exact npix'

    # Test radius larger than π (should be clipped)
    estimated_large = jhp.estimate_disc_pixel_count(nside, 2 * np.pi)
    assert estimated_large == npix, 'Radius > π should be clipped to return npix'

    # Test negative radius (should be clipped to 0)
    estimated_neg = jhp.estimate_disc_pixel_count(nside, -0.1)
    assert estimated_neg == 0, 'Negative radius should be clipped to 0'

    # Test analytical approximation formula
    radius = np.pi / 2  # Quarter sphere
    expected_analytical = int(jnp.ceil(6 * nside**2 * (1 - jnp.cos(radius))))
    estimated_analytical = jhp.estimate_disc_pixel_count(nside, radius)
    assert estimated_analytical == expected_analytical, 'Should match analytical formula'

    # Test with very small radius
    small_radius = 1e-6
    estimated_small = jhp.estimate_disc_pixel_count(nside, small_radius)
    assert estimated_small >= 0, 'Very small radius should return non-negative count'

    # Test monotonicity: larger radius should give larger or equal count
    radius1 = 0.1
    radius2 = 0.2
    count1 = jhp.estimate_disc_pixel_count(nside, radius1)
    count2 = jhp.estimate_disc_pixel_count(nside, radius2)
    assert count2 >= count1, 'Larger radius should give larger or equal pixel count'


def test_estimate_disc_radius():
    """Test the estimate_disc_radius function with comprehensive parameter testing."""
    nside = 16
    npix = jhp.nside2npix(nside)

    # Test normal case
    pixel_count = 100
    estimated_radius = jhp.estimate_disc_radius(nside, pixel_count)
    assert isinstance(estimated_radius, (float, jnp.ndarray)), 'Should return float or JAX array'
    estimated_radius_val = float(estimated_radius)
    assert 0 <= estimated_radius_val <= np.pi, 'Radius should be in [0, π] range'

    # Test full sphere case (pixel_count >= npix)
    full_radius = jhp.estimate_disc_radius(nside, npix)
    assert float(full_radius) == np.pi, 'Full npix should return π'

    # Test pixel_count larger than npix
    large_radius = jhp.estimate_disc_radius(nside, npix + 100)
    assert float(large_radius) == np.pi, 'Pixel count > npix should return π'

    # Test zero pixel count
    zero_radius = jhp.estimate_disc_radius(nside, 0)
    assert float(zero_radius) == 0.0, 'Zero pixel count should return 0.0 radius'

    # Test negative pixel count
    neg_radius = jhp.estimate_disc_radius(nside, -10)
    assert float(neg_radius) == 0.0, 'Negative pixel count should return 0.0 radius'

    # Test inverse relationship with estimate_disc_pixel_count
    original_radius = 0.2
    estimated_count = jhp.estimate_disc_pixel_count(nside, original_radius)
    back_radius = jhp.estimate_disc_radius(nside, estimated_count)
    # Allow small numerical differences
    assert abs(float(back_radius) - original_radius) < 0.01, 'Inverse relationship should hold approximately'

    # Test edge case where cos_term needs clipping
    # Very large pixel count that would give cos_term < -1
    large_count = 10 * npix
    clipped_radius = jhp.estimate_disc_radius(nside, large_count)
    assert float(clipped_radius) == np.pi, 'Very large pixel count should be clipped'

    # Test monotonicity: larger pixel count should give larger radius
    count1 = 50
    count2 = 150
    radius1 = jhp.estimate_disc_radius(nside, count1)
    radius2 = jhp.estimate_disc_radius(nside, count2)
    assert float(radius2) >= float(radius1), 'Larger pixel count should give larger or equal radius'


def test_query_disc_additional_edge_cases():
    """Test additional edge cases to achieve 100% coverage."""
    nside = 16
    vec = [1.0, 0.0, 0.0]
    radius = 0.1
    npix = jhp.nside2npix(nside)

    # Test max_length=None case (should default to npix)
    result_none = jhp.query_disc(nside, vec, radius, max_length=None)
    assert len(result_none) == npix, 'max_length=None should return full npix length'

    # Compare with explicit max_length=npix
    result_explicit = jhp.query_disc(nside, vec, radius, max_length=npix)
    np.testing.assert_array_equal(result_none, result_explicit, 'max_length=None should equal max_length=npix')

    # Test case that triggers line 499 - specific geometric condition
    # Use parameters that create a ring with ysq <= 0 (no geometric intersection)
    # This happens when the disc is very small and doesn't intersect certain rings
    small_nside = 4
    small_vec = [0.0, 0.0, 1.0]  # North pole
    very_small_radius = 1e-8  # Extremely small radius

    result_small = jhp.query_disc(small_nside, small_vec, very_small_radius)
    small_npix = jhp.nside2npix(small_nside)
    assert len(result_small) == small_npix, 'Should handle very small radius case'

    # Test specific case that triggers edge condition in ring processing
    # Use configuration that creates specific geometric edge case
    edge_vec = [0.5, 0.5, 0.707]  # Arbitrary position
    edge_radius = 0.01  # Very small radius
    edge_nside = 8

    edge_result = jhp.query_disc(edge_nside, edge_vec, edge_radius)
    edge_npix = jhp.nside2npix(edge_nside)
    assert len(edge_result) == edge_npix, 'Should handle edge geometric case'

    # Test with vector that needs normalization to trigger specific code paths
    unnorm_vec = [3.0, 4.0, 0.0]  # Non-unit vector
    norm_result = jhp.query_disc(nside, unnorm_vec, radius)

    # Should be equivalent to normalized vector
    normalized_vec = np.array(unnorm_vec) / np.linalg.norm(unnorm_vec)
    normalized_result = jhp.query_disc(nside, normalized_vec, radius)
    np.testing.assert_array_equal(norm_result, normalized_result, 'Non-unit vectors should be normalized')


def test_query_disc_bruteforce():
    """Test the deprecated _query_disc_bruteforce function."""
    nside = 8  # Small nside for faster testing
    vec = [1.0, 0.0, 0.0]
    radius = 0.2
    npix = jhp.nside2npix(nside)

    # Test basic functionality
    bf_result = _query_disc_bruteforce(nside, vec, radius, inclusive=False, fact=4, max_length=None)
    assert len(bf_result) == npix, 'Brute force should return npix length by default'

    # Test with explicit max_length
    max_len = 50
    bf_result_limited = _query_disc_bruteforce(nside, vec, radius, inclusive=False, fact=4, max_length=max_len)
    assert len(bf_result_limited) == max_len, f'Should return max_length={max_len}'

    # Test inclusive mode
    bf_inclusive = _query_disc_bruteforce(nside, vec, radius, inclusive=True, fact=4, max_length=None)
    bf_non_inclusive = _query_disc_bruteforce(nside, vec, radius, inclusive=False, fact=4, max_length=None)

    # Count valid pixels
    bf_inc_valid = bf_inclusive[bf_inclusive < npix]
    bf_non_inc_valid = bf_non_inclusive[bf_non_inclusive < npix]
    assert len(bf_inc_valid) >= len(bf_non_inc_valid), 'Inclusive should have more or equal pixels'

    # Test batch functionality
    vecs_batch = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    bf_batch = _query_disc_bruteforce(nside, vecs_batch, radius, inclusive=False, fact=4, max_length=max_len)
    assert bf_batch.shape == (2, max_len), f'Batch should return (2, {max_len})'

    # Test single vector case (should squeeze)
    bf_single = _query_disc_bruteforce(nside, vec, radius, inclusive=False, fact=4, max_length=max_len)
    assert bf_single.shape == (max_len,), f'Single vector should return ({max_len},)'

    # Test zero vector handling
    zero_vec = [0.0, 0.0, 0.0]
    bf_zero = _query_disc_bruteforce(nside, zero_vec, radius, inclusive=False, fact=4, max_length=max_len)
    assert len(bf_zero) == max_len, 'Should handle zero vector'

    # Test radius clipping
    bf_neg_radius = _query_disc_bruteforce(nside, vec, -0.1, inclusive=False, fact=4, max_length=max_len)
    bf_zero_radius = _query_disc_bruteforce(nside, vec, 0.0, inclusive=False, fact=4, max_length=max_len)
    np.testing.assert_array_equal(bf_neg_radius, bf_zero_radius, 'Negative radius should be clipped to 0')

    # Test large radius (full sphere)
    bf_full = _query_disc_bruteforce(nside, vec, np.pi, inclusive=False, fact=4, max_length=max_len)
    bf_valid_full = bf_full[bf_full < npix]
    assert len(bf_valid_full) == max_len, 'Full sphere should fill max_length with valid pixels'

    # Test truncation warning trigger (use large radius and small max_length to trigger warning)
    small_max = 5
    large_radius = 0.8  # Large disc
    # This should trigger the truncation warning
    bf_truncated = _query_disc_bruteforce(nside, vec, large_radius, inclusive=False, fact=4, max_length=small_max)
    assert len(bf_truncated) == small_max, 'Should return requested max_length even with truncation'

    # Verify that sentinel values are npix
    invalid_mask = bf_truncated == npix
    valid_mask = bf_truncated < npix
    assert np.all(bf_truncated[invalid_mask] == npix), 'Invalid pixels should be npix'
    assert np.all(bf_truncated[valid_mask] < npix), 'Valid pixels should be < npix'


def test_ring_single_internal_functions():
    """Test internal functions of _query_disc_ring_single for edge cases."""
    from jax_healpy._query_disc import _query_disc_ring_single

    # Test line 247: max_length = npix when max_length is None in _query_disc_ring_single
    nside = 4  # Small for faster testing
    vec = jnp.array([1.0, 0.0, 0.0])
    radius = 0.1
    npix = 12 * nside * nside

    # Call _query_disc_ring_single directly with max_length=None to trigger line 247
    result_none = _query_disc_ring_single(nside, vec, radius, inclusive=False, fact=4, max_length=None)
    assert len(result_none) == npix, 'max_length=None should default to npix in _query_disc_ring_single'

    # Test line 499: no_ring_pixels() function - need specific geometric configuration
    # This occurs when a ring has no intersection with the disc
    # Use a very small disc at a specific position that causes some rings to have no intersection

    # Configuration that creates rings with no intersection
    small_nside = 2  # Very small nside
    pole_vec = jnp.array([0.0, 0.0, 1.0])  # North pole
    tiny_radius = 1e-10  # Extremely tiny radius

    # This should trigger the no_ring_pixels case for most rings
    result_tiny = _query_disc_ring_single(small_nside, pole_vec, tiny_radius, inclusive=False, fact=4, max_length=None)
    small_npix = 12 * small_nside * small_nside
    assert len(result_tiny) == small_npix, 'Should handle case with no ring intersection'

    # Additional test: try different geometric configuration to ensure line 499 coverage
    # Use equatorial position with very small radius
    equatorial_vec = jnp.array([1.0, 0.0, 0.0])
    result_equatorial = _query_disc_ring_single(
        small_nside, equatorial_vec, tiny_radius, inclusive=False, fact=4, max_length=None
    )
    assert len(result_equatorial) == small_npix, 'Should handle equatorial case with no ring intersection'
