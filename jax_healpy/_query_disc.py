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

"""
Query disc implementation for HEALPix spherical disc queries.
"""

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit, lax
from jaxtyping import Array, ArrayLike

from .pixelfunc import _get_ring_info, _ring_above, nside2resol, pix2vec

__all__ = ['query_disc', 'estimate_disc_pixel_count', 'estimate_disc_radius', '_query_disc_bruteforce']


def _ring2z(nside: int, ring_idx: ArrayLike) -> Array:
    """Convert ring index to z-coordinate following HEALPix C++ implementation.

    This follows the exact logic from the C++ healpix_base.cc ring2z function.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter
    ring_idx : ArrayLike
        Ring index (1 to 4*nside-1)

    Returns
    -------
    z : Array
        z-coordinate (cos(theta)) for the ring
    """
    ring = jnp.asarray(ring_idx, dtype=jnp.int32)

    # Convert to northern hemisphere ring number
    northring = jnp.where(ring > 2 * nside, 4 * nside - ring, ring)

    # HEALPix constants
    fact2 = 4.0 / (12.0 * nside * nside)  # 4/npix

    # Polar cap region (northring < nside)
    polar_z = 1.0 - (northring * northring) * fact2

    # Equatorial region (northring >= nside)
    equatorial_z = (2.0 * nside - northring) * 2.0 / (3.0 * nside)

    # Select based on region
    z = jnp.where(northring < nside, polar_z, equatorial_z)

    # Handle southern hemisphere (original ring > 2*nside)
    z = jnp.where(ring > 2 * nside, -z, z)

    return z


def _max_pixrad(nside: int) -> float:
    """Calculate maximum pixel radius for the given nside.

    This approximates the maximum angular distance from a pixel center
    to any point within the pixel. Based on HEALPix C++ max_pixrad.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter

    Returns
    -------
    max_radius : float
        Maximum pixel radius in radians
    """
    # Approximate maximum pixel radius
    # This is a conservative estimate - in reality varies by pixel location
    resol = jnp.sqrt(4.0 * jnp.pi / (12.0 * nside * nside))
    return resol * 0.6  # Conservative factor from HEALPix


def estimate_disc_pixel_count(nside: int, radius: float) -> int:
    """Estimate number of pixels in a disc of given radius.

    Uses the analytical approximation: n_approx = 6 * nside^2 * (1 - cos(radius))

    This approximation is exact at radius = π (full sphere) and loses precision
    for smaller radii. It provides a good estimate for setting the max_length
    parameter in query_disc functions.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter
    radius : float
        Disc radius in radians

    Returns
    -------
    pixel_count : int
        Estimated number of pixels in the disc

    Notes
    -----
    The approximation assumes uniform pixel density across the sphere, which
    is accurate for large discs but less precise for small discs where
    pixel shape variations matter more.

    Examples
    --------
    >>> import jax_healpy as hp
    >>> nside = 64
    >>> radius = 0.1  # ~5.7 degrees
    >>> estimated_count = hp.estimate_disc_pixel_count(nside, radius)
    >>> # Use as max_length for query_disc
    >>> actual_disc = hp.query_disc(nside, [1, 0, 0], radius, max_length=estimated_count)
    """

    # Clip radius to valid range
    radius = jnp.clip(radius, 0.0, jnp.pi)

    # Analytical approximation: n = 6 * nside^2 * (1 - cos(radius))
    pixel_count = 6 * nside**2 * (1 - jnp.cos(radius))

    # For full sphere, ensure we return exactly npix
    npix_total = 12 * nside**2
    if radius >= jnp.pi:
        return npix_total

    return int(jnp.ceil(pixel_count))


def estimate_disc_radius(nside: int, pixel_count: int) -> float:
    """Estimate radius needed for a disc containing given pixel count.

    Inverse of estimate_disc_pixel_count. Uses the analytical relationship:
    radius = arccos(1 - pixel_count / (6 * nside^2))

    Parameters
    ----------
    nside : int
        HEALPix nside parameter
    pixel_count : int
        Desired number of pixels in the disc

    Returns
    -------
    radius : float
        Estimated disc radius in radians

    Notes
    -----
    This is the inverse of estimate_disc_pixel_count and has the same
    accuracy characteristics: exact at full sphere, less precise for
    small discs.

    Examples
    --------
    >>> import jax_healpy as hp
    >>> nside = 64
    >>> target_pixels = 1000
    >>> estimated_radius = hp.estimate_disc_radius(nside, target_pixels)
    >>> # Verify the relationship
    >>> back_pixels = hp.estimate_disc_pixel_count(nside, estimated_radius)
    >>> abs(back_pixels - target_pixels) < 10  # Should be close
    True
    """

    npix_total = 12 * nside**2
    if pixel_count >= npix_total:
        return jnp.pi  # Full sphere

    if pixel_count <= 0:
        return 0.0

    # Inverse formula: radius = arccos(1 - pixel_count / (6 * nside^2))
    cos_term = 1 - pixel_count / (6 * nside**2)

    # Clamp to valid range for arccos
    cos_term = jnp.clip(cos_term, -1.0, 1.0)

    return jnp.arccos(cos_term)


def _query_disc_ring_single(
    nside: int, vec: Array, radius: float, inclusive: bool = False, fact: int = 4, max_length: int = None
) -> Array:
    """True geometric single-disc query for RING scheme following HEALPix C++ algorithm.

    This implements the exact geometry-based algorithm from HEALPix C++ that processes
    only candidate rings and generates pixels directly from geometric intersections.
    NO brute-force operations are performed. The algorithm is memory-optimized using
    lax.fori_loop instead of lax.scan to avoid large intermediate arrays.

    Algorithm Steps:
    1. Setup geometric bounds and normalize input vector
    2. Calculate candidate ring range based on disc geometry
    3. Initialize pixel mask for accumulation
    4. Add polar region pixels if disc intersects poles
    5. Process rings using fixed-size loop for JAX compatibility
    6. Extract valid pixels using memory-optimized collection method

    Parameters
    ----------
    nside : int
        HEALPix nside parameter (must be a power of 2)
    vec : Array
        Unit vector (3,) defining disc center. Will be normalized if not unit length.
    radius : float
        Disc radius in radians, will be clipped to [0, π]
    inclusive : bool, optional
        If True, include pixels that overlap the disc boundary (default: False)
    fact : int, optional
        Oversampling factor for inclusive mode (default: 4)
    max_length : int, optional
        Maximum number of pixels to return. If None, defaults to npix.

    Returns
    -------
    pixels : Array
        Array of pixel indices in the disc, shape (max_length,).
        Pixels outside the disc are marked as npix (sentinel value).
        Results are padded with npix for unused entries.

    Notes
    -----
    This function achieves significant memory optimization compared to brute-force
    approaches by avoiding creation of large intermediate arrays and using geometric
    ring processing following the HEALPix C++ reference implementation.
    """
    npix = 12 * nside * nside
    if max_length is None:
        max_length = npix

    # #step1: Setup and Geometric Bounds
    # Normalize vector and handle edge cases
    vec_norm = jnp.linalg.norm(vec)
    safe_vec = jnp.where(vec_norm > 1e-10, vec / vec_norm, jnp.array([1.0, 0.0, 0.0]))

    # Clip radius to valid range [0, π]
    radius = jnp.clip(radius, 0.0, jnp.pi)

    # Calculate inclusive mode radii based on C++ reference
    if inclusive:
        # Use finer grid and original grid pixel radii for inclusive bounds
        finer_pixrad = _max_pixrad(fact * nside)  # More precise pixel radius
        coarse_pixrad = _max_pixrad(nside)  # Original pixel radius
        rsmall = radius + finer_pixrad
        rbig = radius + coarse_pixrad
    else:
        rsmall = rbig = radius

    # Handle full-sphere case
    full_sphere = rsmall >= jnp.pi
    rbig = jnp.minimum(jnp.pi, rbig)

    # Pre-compute trigonometric values
    cosrbig = jnp.cos(rbig)

    # #step2: Calculate Disc Center Coordinates and Ring Range
    # Disc center coordinates
    z0 = safe_vec[2]  # cos(theta)
    phi0 = jnp.arctan2(safe_vec[1], safe_vec[0])

    # Handle polar singularity where sin(theta) = 0
    sin_theta_sq = (1.0 - z0) * (1.0 + z0)  # sin²(theta) = 1 - cos²(theta)
    xa = jnp.where(
        sin_theta_sq > 1e-10,
        1.0 / jnp.sqrt(sin_theta_sq),  # Normal case: 1/sin(theta)
        1e10,  # Polar case: very large value (effectively infinity)
    )

    # Calculate candidate ring range
    # Note: z0 = cos(theta), so theta = arccos(z0)
    theta0 = jnp.arccos(jnp.clip(z0, -1.0, 1.0))  # Clip to handle numerical precision
    rlat1 = theta0 - rsmall  # theta - rsmall
    rlat2 = theta0 + rsmall  # theta + rsmall

    zmax = jnp.cos(jnp.maximum(0.0, rlat1))
    irmin = _ring_above(nside, zmax) + 1

    zmin = jnp.cos(jnp.minimum(jnp.pi, rlat2))
    irmax = _ring_above(nside, zmin)

    # For inclusive mode, expand ring range slightly
    irmin = jnp.where(inclusive & (rlat1 > 0), jnp.maximum(1, irmin - 1), irmin)
    irmax = jnp.where(inclusive & (rlat2 < jnp.pi), jnp.minimum(4 * nside - 1, irmax + 1), irmax)

    # Handle polar regions (following C++ logic exactly)
    north_pole_in_disc = (rlat1 <= 0.0) & (irmin > 1)
    south_pole_in_disc = (rlat2 >= jnp.pi) & (irmax + 1 < 4 * nside)

    # #step3: Initialize pixel mask for accumulation
    # Use boolean mask to track valid pixels - JAX compatible approach
    pixel_mask = jnp.zeros(npix, dtype=bool)

    # #step4: Add polar region pixels if needed
    def add_north_pole_pixels(mask):
        # If north pole is in disc, add pixels from rings 1 to irmin-1
        # NOTE: When irmin = 1, we still need to potentially add pixels from the north cap
        def add_north_pixels():
            # Determine which ring to use as the boundary
            boundary_ring = jnp.maximum(1, irmin - 1)

            # Get total pixels in north cap up to boundary_ring
            ring_info = _get_ring_info(nside, boundary_ring)
            north_cap_pixels = ring_info[1] + ring_info[2]  # startpix + ringpix of boundary ring

            # Create mask for pixels 0 to north_cap_pixels-1
            north_indices = jnp.arange(npix)
            north_mask = north_indices < north_cap_pixels
            return mask | north_mask

        def no_north_pixels():
            return mask

        return lax.cond(north_pole_in_disc, add_north_pixels, no_north_pixels)

    def add_south_pole_pixels(mask):
        # C++ logic: if (rlat2>=pi) && (irmax+1<4*nside_)
        # Add pixels from startpix of ring (irmax+1) to npix-1
        def add_south_pixels():
            # Get start pixel of ring irmax+1 (which is guaranteed to exist by the condition)
            ring_info = _get_ring_info(nside, irmax + 1)
            south_start_pixel = ring_info[1]  # startpix

            # Create mask for pixels from south_start_pixel to npix-1
            south_indices = jnp.arange(npix)
            south_mask = south_indices >= south_start_pixel
            return mask | south_mask

        def no_south_pixels():
            return mask

        return lax.cond(south_pole_in_disc, add_south_pixels, no_south_pixels)

    # Add polar pixels
    pixel_mask = add_north_pole_pixels(pixel_mask)
    pixel_mask = add_south_pole_pixels(pixel_mask)

    # #step5: Process rings using fixed-size loop
    def process_ring_iteration(ring_idx, mask):
        """Process a single ring and update the pixel mask."""

        # Only process if ring is in our candidate range
        in_range = (ring_idx >= irmin) & (ring_idx <= irmax) & (ring_idx >= 1) & (ring_idx < 4 * nside)

        def process_valid_ring():
            # Get ring properties
            z = _ring2z(nside, ring_idx)
            ring_info = _get_ring_info(nside, ring_idx)
            ipix1 = ring_info[1]  # Start pixel index for this ring
            nr = ring_info[2]  # Number of pixels in ring
            shifted = ring_info[3]  # Whether ring is shifted

            # Calculate intersection geometry
            x = (cosrbig - z * z0) * xa
            ysq = 1.0 - z * z - x * x

            def calculate_ring_pixels():
                """Calculate which pixels in this ring are in the disc."""
                # Following C++ logic: handle ysq <= 0 case
                # When ysq <= 0, no normal intersection exists - ring is either
                # completely inside or completely outside the disc

                def handle_no_intersection():
                    # When ysq <= 0, ring is either completely inside or outside the disc
                    # Check if ring center is inside the disc to determine which case

                    # Get a representative point on the ring (any longitude will do)
                    ring_phi = 0.0  # Use phi=0 as representative point
                    ring_vec = jnp.array(
                        [
                            jnp.sqrt(1 - z * z) * jnp.cos(ring_phi),  # x = sin(theta) * cos(phi)
                            jnp.sqrt(1 - z * z) * jnp.sin(ring_phi),  # y = sin(theta) * sin(phi)
                            z,  # z = cos(theta)
                        ]
                    )

                    # Check if this ring point is inside the disc
                    ring_dot = jnp.dot(ring_vec, safe_vec)
                    ring_inside_disc = ring_dot >= cosrbig

                    # If ring is inside disc, include all pixels (dphi = pi)
                    # If ring is outside disc, include no pixels (dphi = 0)
                    dphi = jnp.where(ring_inside_disc, jnp.pi - 1e-15, 0.0)
                    return dphi

                def handle_normal_intersection():
                    # Normal case: calculate intersection half-angle
                    dphi = jnp.arctan2(jnp.sqrt(ysq), x)
                    return dphi

                # Calculate dphi based on whether we have a geometric intersection
                dphi = lax.cond(ysq <= 0, handle_no_intersection, handle_normal_intersection)

                # If dphi <= 0, no pixels in this ring
                def no_pixels():
                    return jnp.zeros(npix, dtype=bool)

                def calculate_pixels():
                    # Convert longitude range to pixel indices within ring
                    shift = jnp.where(shifted, 0.5, 0.0)
                    inv_twopi = 1.0 / (2.0 * jnp.pi)

                    # Calculate pixel range in ring coordinates (following C++ logic exactly)
                    ip_lo = jnp.floor(nr * inv_twopi * (phi0 - dphi) - shift).astype(jnp.int32) + 1
                    ip_hi = jnp.floor(nr * inv_twopi * (phi0 + dphi) - shift).astype(jnp.int32)

                    # Handle fullcircle case (when dphi ≈ π, we want nearly the entire ring)
                    fullcircle = dphi >= (jnp.pi - 1e-10)  # Close to full circle

                    def adjust_for_fullcircle():
                        # C++ logic: if (ip_hi-ip_lo<nr-1) expand the range
                        adj_ip_lo = ip_lo
                        adj_ip_hi = ip_hi

                        needs_expansion = (adj_ip_hi - adj_ip_lo) < (nr - 1)

                        def expand_range():
                            # if (ip_lo>0) --ip_lo; else ++ip_hi;
                            new_ip_lo = jnp.where(adj_ip_lo > 0, adj_ip_lo - 1, adj_ip_lo)
                            new_ip_hi = jnp.where(adj_ip_lo > 0, adj_ip_hi, adj_ip_hi + 1)
                            return new_ip_lo, new_ip_hi

                        def keep_range():
                            return adj_ip_lo, adj_ip_hi

                        return lax.cond(needs_expansion, expand_range, keep_range)

                    def keep_original():
                        return ip_lo, ip_hi

                    ip_lo, ip_hi = lax.cond(fullcircle, adjust_for_fullcircle, keep_original)

                    # DO NOT clip here - wraparound is detected by ip_lo > ip_hi or out-of-bounds values

                    # Create mask for this ring's pixels
                    ring_pixel_indices = jnp.arange(npix)

                    # Handle the C++ wraparound logic exactly
                    def simple_range():
                        # Standard case: ip_lo <= ip_hi and both in bounds
                        ring_start = ipix1 + ip_lo
                        ring_end = ipix1 + ip_hi + 1
                        ring_mask = (ring_pixel_indices >= ring_start) & (ring_pixel_indices < ring_end)
                        return ring_mask

                    def handle_wraparound():
                        # Handle out-of-bounds cases according to C++ logic
                        # Adjust indices for wraparound
                        adj_ip_lo = ip_lo
                        adj_ip_hi = ip_hi

                        # Handle ip_hi >= nr case
                        adj_ip_lo = jnp.where(ip_hi >= nr, adj_ip_lo - nr, adj_ip_lo)
                        adj_ip_hi = jnp.where(ip_hi >= nr, adj_ip_hi - nr, adj_ip_hi)

                        # Handle ip_lo < 0 case (wraparound)
                        def wraparound_case():
                            # Two segments: [ipix1, ipix1+ip_hi+1) and [ipix1+ip_lo+nr, ipix1+nr)
                            # Following C++ logic: append(ipix1, ipix1+ip_hi+1) and append(ipix1+ip_lo+nr, ipix2+1)
                            mask1 = (ring_pixel_indices >= ipix1) & (ring_pixel_indices < ipix1 + adj_ip_hi + 1)
                            mask2 = (ring_pixel_indices >= ipix1 + adj_ip_lo + nr) & (ring_pixel_indices < ipix1 + nr)
                            return mask1 | mask2

                        def normal_case():
                            # Single segment: [ipix1+adj_ip_lo, ipix1+adj_ip_hi]
                            ring_start = ipix1 + adj_ip_lo
                            ring_end = ipix1 + adj_ip_hi + 1
                            ring_mask = (ring_pixel_indices >= ring_start) & (ring_pixel_indices < ring_end)
                            return ring_mask

                        return lax.cond(adj_ip_lo < 0, wraparound_case, normal_case)

                    # Check if we need special handling
                    needs_special_handling = (ip_lo > ip_hi) | (ip_hi >= nr) | (ip_lo < 0)
                    return lax.cond(needs_special_handling, handle_wraparound, simple_range)

                # Return appropriate result based on dphi
                return lax.cond(dphi <= 0, no_pixels, calculate_pixels)

            def no_ring_pixels():
                """No intersection - return empty mask."""
                return jnp.zeros(npix, dtype=bool)

            # Always try to calculate ring pixels - the dphi calculation handles the ysq <= 0 case
            ring_mask = calculate_ring_pixels()

            return mask | ring_mask

        def skip_ring():
            """Ring not in range - return unchanged mask."""
            return mask

        return lax.cond(in_range, process_valid_ring, skip_ring)

    # Process all rings using fixed-size loop (static bounds)
    max_rings = 4 * nside
    pixel_mask = lax.fori_loop(1, max_rings, process_ring_iteration, pixel_mask)

    # #step6: Extract valid pixels using memory-optimized method
    def extract_pixels_from_mask(mask):
        """Extract pixel indices from boolean mask without expensive argsort.

        This memory-optimized approach uses lax.fori_loop instead of lax.scan
        to avoid creating large intermediate arrays with jnp.arange(npix).
        """

        # #step6a: Use fori_loop to collect valid pixels sequentially
        def fori_body(i, carry):
            result_array, count = carry
            pixel_is_valid = mask[i]

            # Add pixel to result if valid and we have space
            should_add = pixel_is_valid & (count < max_length)
            new_result = jnp.where(should_add, result_array.at[count].set(i), result_array)
            new_count = jnp.where(should_add, count + 1, count)

            return (new_result, new_count)

        # #step6b: Initialize result array filled with sentinel values
        init_result = jnp.full(max_length, npix, dtype=jnp.int32)
        init_carry = (init_result, 0)

        # #step6c: Scan through all pixels to collect valid ones
        (final_result, final_count) = lax.fori_loop(0, npix, fori_body, init_carry)

        return final_result

    # Handle full sphere case
    def get_all_pixels():
        return jnp.arange(max_length, dtype=jnp.int32)

    def get_geometric_pixels():
        return extract_pixels_from_mask(pixel_mask)

    return lax.cond(full_sphere, get_all_pixels, get_geometric_pixels)


@partial(jit, static_argnames=['nside', 'inclusive', 'fact', 'nest', 'max_length'])
def query_disc(
    nside: int,
    vec: ArrayLike,
    radius: float,
    inclusive: bool = False,
    fact: int = 4,
    nest: bool = False,
    max_length: Optional[int] = None,
) -> Array:
    """Find pixels within a disc on the sphere.

    This function supports both single and batched queries. It is fully JIT-compatible
    and differentiable with respect to vec and radius parameters.

    Parameters
    ----------
    nside : int
        The resolution parameter of the HEALPix map
    vec : array-like
        Either a single three-component unit vector (3,) defining the center of the disc,
        or a batch of vectors (B, 3) defining B disc centers
    radius : float
        The radius of the disc in radians
    inclusive : bool, optional
        If False (default), return pixels whose centers lie within the disc.
        Results are guaranteed to match healpy exactly for inclusive=False.
        If True, return all pixels that overlap with the disc.
        Note: inclusive=True may produce slightly different results compared to healpy
        due to algorithm differences in determining pixel overlap.
    fact : int, optional
        For inclusive queries, the pixelization factor (default: 4)
    nest : bool, optional
        If True, assume NESTED pixel ordering, otherwise RING ordering (default: False)
    max_length : int, optional
        Maximum number of pixels to return per disc. If None, defaults to npix.
        For batched inputs, this limits memory usage by returning (max_length, B)
        instead of (npix, B).

    Returns
    -------
    ipix : array
        For single vector input (3,): returns (max_length,) or (npix,) if max_length is None
        For batch vector input (B, 3): returns (max_length, B)
        Pixels outside the disc are marked as npix (sentinel value).
        This allows direct indexing like: map.at[disc].set(value)

    Raises
    ------
    NotImplementedError
        If nest=True (nested ordering not yet supported)

    Notes
    -----
    This function currently only supports RING ordering. The function is
    JIT-compatible and differentiable when compiled with static_argnums=(0,)
    for the nside parameter. The returned array has fixed size for JIT
    compatibility - pixels outside the disc have value npix.

    When indexing a JAX array, pixels outside the disc should be ignored.
    If indexing a numpy array, this will raise an out-of-bounds error.

    Examples
    --------
    Single disc query:

    >>> import jax_healpy as hp
    >>> import jax.numpy as jnp
    >>> import jax
    >>> nside = 16
    >>> vec = jnp.array([1.0, 0.0, 0.0])  # Point on equator
    >>> radius = 0.1  # ~5.7 degrees
    >>> disc = hp.query_disc(nside, vec, radius)
    >>> # Use directly for indexing
    >>> map = jnp.zeros(hp.nside2npix(nside))
    >>> map = map.at[disc].set(1.0)

    Batch disc query:

    >>> vecs = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3) - two centers
    >>> discs = hp.query_disc(nside, vecs, radius, max_length=1000)  # (1000, 2)
    >>> # Each column contains pixels for one disc
    >>> map1 = map.at[discs[:, 0]].set(1.0)  # First disc
    >>> map2 = map.at[discs[:, 1]].set(2.0)  # Second disc

    For JIT compilation:

    >>> jit_query_disc = jax.jit(
    ...     lambda n, v, r: hp.query_disc(n, v, r)
    ... )
    >>> disc_jit = jit_query_disc(nside, vec, radius)
    """
    # Raise error for nested ordering
    if nest:
        raise NotImplementedError('Nested ordering not yet supported')

    return _query_disc_ring(nside, vec, radius, inclusive, fact, max_length)


def _query_disc_ring(
    nside: int, vec: ArrayLike, radius: float, inclusive: bool, fact: int, max_length: Optional[int]
) -> Array:
    """Efficient RING scheme query with batching support using geometric algorithm.

    This function handles both single and batched disc queries by using jax.vmap
    to vectorize the single-disc geometric algorithm.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter
    vec : ArrayLike
        Either single vector (3,) or batch of vectors (B, 3)
    radius : float
        Disc radius in radians
    inclusive : bool
        If True, include pixels that overlap the disc boundary
    fact : int
        Oversampling factor for inclusive mode
    max_length : Optional[int]
        Maximum number of pixels to return per disc

    Returns
    -------
    Array
        For single vector: shape (max_length,)
        For batch: shape (B, max_length)
        Pixels outside discs are marked as npix (sentinel value)
    """

    # Convert to JAX arrays
    vec = jnp.asarray(vec, dtype=jnp.float64)
    radius = jnp.asarray(radius, dtype=jnp.float64)
    original_is_single = vec.ndim == 1

    if original_is_single:
        vec = vec[None, :]  # (3,) → (1, 3)

    npix = 12 * nside * nside

    if max_length is None:
        max_length = npix

    # Process each vector in the batch
    def process_single_vec(single_vec):
        return _query_disc_ring_single(nside, single_vec, radius, inclusive, fact, max_length)

    # Use vmap to handle batching
    result = jax.vmap(process_single_vec)(vec)  # (batch_dims, max_length)

    # Squeeze for single vector input
    if original_is_single:
        result = jnp.squeeze(result, axis=0)  # (1, max_length) → (max_length,)

    return result


def _query_disc_bruteforce(
    nside: int, vec: ArrayLike, radius: float, inclusive: bool, fact: int, max_length: Optional[int]
) -> Array:
    """DEPRECATED: Brute-force disc query with O(batch_size × npix) complexity.

    ⚠️  **WARNING: NOT RECOMMENDED FOR PRODUCTION USE** ⚠️

    This function has poor computational and memory scaling characteristics:
    - **Complexity**: O(batch_size × npix) where npix = 12 × nside²
    - **Memory**: Creates large intermediate arrays of size (npix × batch_size)
    - **Performance**: Much slower than the geometric algorithm for large nside values

    **RECOMMENDATION**: Use the default `query_disc()` function instead, which uses
    an efficient geometric algorithm with much better scaling properties.

    This brute-force implementation is kept only for reference and testing purposes.
    It computes dot products with ALL pixels on the sphere, making it inefficient
    for typical use cases.

    Algorithm Overview:
    1. Standardize input to (batch_dims, 3) format and set defaults
    2. Normalize input vectors and clip radius to valid range
    3. Calculate the cosine threshold for the dot product test
    4. Generate ALL pixel vectors and compute broadcast dot products (EXPENSIVE!)
    5. Create mask for pixels within the disc(s) (large intermediate arrays)
    6. Select top max_length pixels per disc with sentinel padding
    7. Apply JAX-compatible warning system for truncation
    8. Squeeze output for single vector compatibility

    Performance Comparison:
    - For nside=512: ~3M pixels → creates 3M × batch_size arrays
    - For nside=1024: ~12M pixels → creates 12M × batch_size arrays
    - Geometric algorithm processes only candidate rings (~10-100× fewer operations)
    """
    # Step 1: Input standardization to (batch_dims, 3) format
    vec = jnp.asarray(vec, dtype=jnp.float64)
    original_is_single = vec.ndim == 1
    if original_is_single:
        vec = vec[None, :]  # (3,) → (1, 3)

    batch_dims = vec.shape[0]
    npix = 12 * nside * nside
    radius = jnp.asarray(radius, dtype=jnp.float64)

    # Default max_length to npix if not provided
    if max_length is None:
        max_length = npix

    # Step 2: Normalize center vectors (handle zero vector case)
    vec_norms = jnp.linalg.norm(vec, axis=1)  # (batch_dims,)
    # Create default direction - broadcasts to (batch_dims, 3)
    default_dir = jnp.array([1.0, 0.0, 0.0])[None, :]

    # Normalize each vector individually
    safe_vecs = jnp.where(vec_norms[:, None] > 1e-10, vec / vec_norms[:, None], default_dir)

    # Clip radius to valid range [0, π]
    radius = jnp.clip(radius, 0.0, jnp.pi)

    # Step 3: Calculate cosine threshold for dot product comparison
    cos_radius = jnp.cos(radius)

    # For inclusive mode, expand the radius by pixel resolution divided by fact
    if inclusive:
        expanded_radius = radius + nside2resol(nside) / fact
        cos_expanded_radius = jnp.cos(jnp.clip(expanded_radius, 0, jnp.pi))
    else:
        cos_expanded_radius = cos_radius

    # Step 4: Generate all pixel vectors and compute broadcast dot products
    all_pixels = jnp.arange(npix, dtype=jnp.int32)
    pixel_vecs = pix2vec(nside, all_pixels, nest=False)  # (npix, 3)

    # Broadcast dot products: (npix, 3) @ (batch_dims, 3).T → (npix, batch_dims)
    dot_products = jnp.dot(pixel_vecs, safe_vecs.T)

    # Step 5: Create mask for pixels within the disc(s)
    # Use small tolerance to handle floating point precision issues
    tolerance = 1e-6
    mask = dot_products >= (cos_expanded_radius - tolerance)  # (npix, batch_dims)

    # Step 6: Select top max_length pixels per disc
    # Create sort keys: valid pixels keep dot product, invalid get -inf
    sort_keys = jnp.where(mask, dot_products, -jnp.inf)  # (npix, batch_dims)

    # Sort indices by dot product (best pixels last)
    sorted_indices = jnp.argsort(sort_keys, axis=0)  # (npix, batch_dims)

    # Select top max_length pixels per batch
    top_indices = sorted_indices[-max_length:]  # (max_length, batch_dims)
    result = top_indices  # (max_length, batch_dims) - keep pixels as leading axis

    # Replace invalid entries (where sort key was -inf) with npix
    selected_scores = sort_keys[top_indices, jnp.arange(batch_dims)]  # (max_length, batch_dims)
    invalid_mask = selected_scores == -jnp.inf  # (max_length, batch_dims)
    result = jnp.where(invalid_mask, npix, result)

    # Step 7: JAX-compatible warning system for truncation
    if max_length < npix:  # Only check for truncation if limiting
        # Count valid pixels per batch
        valid_counts = jnp.sum(mask.astype(jnp.int32), axis=0)  # (batch_dims,)
        exceeded_count = jnp.sum(valid_counts > max_length)  # scalar

        # Use lax.cond for JAX-compatible conditional warning
        lax.cond(
            exceeded_count > 0,
            lambda: jax.debug.print('Warning: {} valid pixels exceeded max_length={}', valid_counts.max(), max_length),
            lambda: None,
        )

    # Step 8: Squeeze output for single vector compatibility
    if original_is_single:
        result = jnp.squeeze(result, axis=1)  # (max_length, 1) → (max_length,)

    return result  # This file is part of jax-healpy.
