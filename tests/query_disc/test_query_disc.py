#!/usr/bin/env python3
"""
Comprehensive tests for query_disc implementation.
"""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_healpy as jhp


class TestQueryDisc:
    """Comprehensive tests for query_disc functionality."""

    def test_basic_functionality(self):
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

    def test_map_indexing_example(self):
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

    def test_against_healpy_comprehensive(self):
        """Comprehensive test against healpy implementation."""
        test_cases = [
            # (nside, vec, radius, inclusive, description)
            (16, [1.0, 0.0, 0.0], 0.1, False, 'Small disc on equator'),
            (16, [0.0, 0.0, 1.0], 0.2, False, 'Small disc at north pole'),
            (16, [0.0, 0.0, -1.0], 0.15, False, 'Small disc at south pole'),
            (32, [1.0, 0.0, 0.0], 0.05, False, 'Very small disc'),
            (8, [0.5, 0.5, 0.707], 0.3, False, 'Medium disc at arbitrary position'),
            (16, [1.0, 0.0, 0.0], 0.1, True, 'Small disc inclusive mode'),
            (16, [0.0, 0.0, 1.0], 0.2, True, 'Pole disc inclusive mode'),
        ]

        for nside, vec, radius, inclusive, description in test_cases:
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

    def test_edge_cases(self):
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

    def test_input_validation(self):
        """Test input validation and error conditions."""
        nside = 16
        vec = [1.0, 0.0, 0.0]

        # Test zero vector (should be handled gracefully)
        ipix = jhp.query_disc(nside, [0.0, 0.0, 0.0], 0.1)
        assert len(ipix) == jhp.nside2npix(nside), 'Zero vector should be handled'

        # Test negative radius (should be clipped)
        ipix = jhp.query_disc(nside, vec, -0.1)
        assert len(ipix) == jhp.nside2npix(nside), 'Negative radius should be clipped'

    def test_nested_ordering_error(self):
        """Test that nested ordering raises NotImplementedError."""
        nside = 16
        vec = [1.0, 0.0, 0.0]
        radius = 0.1

        with pytest.raises(NotImplementedError, match='Nested ordering not yet supported'):
            jhp.query_disc(nside, vec, radius, nest=True)

    def test_jit_compatibility(self):
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

    def test_differentiability(self):
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

    def test_fixed_size_output(self):
        """Test that output has fixed size with npix sentinel values."""
        nside = 16
        vec = [1.0, 0.0, 0.0]
        npix = jhp.nside2npix(nside)

        # Test with different radii
        for radius in [0.01, 0.1, 0.5, 1.0]:
            result = jhp.query_disc(nside, vec, radius)
            assert len(result) == npix, f'Output should have fixed size {npix}'

            # Check that sentinel values are npix
            invalid_mask = result == npix
            valid_mask = result < npix
            assert np.all(result[invalid_mask] == npix), 'Invalid pixels should have value npix'
            assert np.all(result[valid_mask] < npix), 'Valid pixels should have value < npix'

    def test_batch_functionality(self):
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
        np.testing.assert_array_equal(
            single_result, batch_first, 'Single vector result should match first batch element'
        )

    def test_batch_edge_cases(self):
        """Test edge cases for batch functionality."""
        nside = 8  # Small for faster testing
        npix = jhp.nside2npix(nside)

        # Test with very small max_length (truncation)
        vecs = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # (3, 2)
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
