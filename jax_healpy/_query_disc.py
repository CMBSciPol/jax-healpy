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

from .pixelfunc import pix2vec

__all__ = ['query_disc']


def _compute_resolution(nside: int) -> Array:
    """JAX-compatible computation of pixel resolution in radians.

    This computes the approximate pixel size as the square root of the pixel area.
    This is a gross approximation given the different pixel shapes in HEALPix.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter

    Returns
    -------
    resol : Array
        Approximate pixel size in radians
    """
    npix = 12 * nside * nside  # Total number of pixels
    pixarea = 4 * jnp.pi / npix  # Pixel area = total sphere area / npix
    return jnp.sqrt(pixarea)  # Resolution = sqrt(area)


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
        or a batch of vectors (3, B) defining B disc centers
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
        For batched inputs, this limits memory usage by returning (B, max_length)
        instead of (B, npix).

    Returns
    -------
    ipix : array
        For single vector input (3,): returns (max_length,) or (npix,) if max_length is None
        For batch vector input (3, B): returns (B, max_length)
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

    >>> vecs = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # (3, 2) - two centers
    >>> discs = hp.query_disc(nside, vecs, radius, max_length=1000)  # (2, 1000)
    >>> # Each row contains pixels for one disc
    >>> map1 = map.at[discs[0]].set(1.0)  # First disc
    >>> map2 = map.at[discs[1]].set(2.0)  # Second disc

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
    """JIT-compatible unified implementation of disc query supporting both single and batch inputs.

    Algorithm Overview:
    1. Standardize input to (3, batch_dims) format and set defaults
    2. Normalize input vectors and clip radius to valid range
    3. Calculate the cosine threshold for the dot product test
    4. Generate all pixel vectors and compute broadcast dot products
    5. Create mask for pixels within the disc(s)
    6. Select top max_length pixels per disc with sentinel padding
    7. Apply JAX-compatible warning system for truncation
    8. Squeeze output for single vector compatibility
    """
    # Step 1: Input standardization to (3, batch_dims) format
    vec = jnp.asarray(vec, dtype=jnp.float64)
    original_is_single = vec.ndim == 1
    if original_is_single:
        vec = vec[:, None]  # (3,) → (3, 1)

    batch_dims = vec.shape[1]
    npix = 12 * nside * nside
    radius = jnp.asarray(radius, dtype=jnp.float64)

    # Default max_length to npix if not provided
    if max_length is None:
        max_length = npix

    # Step 2: Normalize center vectors (handle zero vector case)
    vec_norms = jnp.linalg.norm(vec, axis=0)  # (batch_dims,)
    # Create default direction - broadcasts to (3, batch_dims)
    default_dir = jnp.array([1.0, 0.0, 0.0])[:, None]

    # Normalize each vector individually
    safe_vecs = jnp.where(vec_norms[None, :] > 1e-10, vec / vec_norms[None, :], default_dir)

    # Clip radius to valid range [0, π]
    radius = jnp.clip(radius, 0.0, jnp.pi)

    # Step 3: Calculate cosine threshold for dot product comparison
    cos_radius = jnp.cos(radius)

    # For inclusive mode, expand the radius by pixel resolution divided by fact
    if inclusive:
        expanded_radius = radius + _compute_resolution(nside) / fact
        cos_expanded_radius = jnp.cos(jnp.clip(expanded_radius, 0, jnp.pi))
    else:
        cos_expanded_radius = cos_radius

    # Step 4: Generate all pixel vectors and compute broadcast dot products
    all_pixels = jnp.arange(npix, dtype=jnp.int32)
    pixel_vecs = pix2vec(nside, all_pixels, nest=False)  # (npix, 3)

    # Broadcast dot products: (npix, 3) @ (3, batch_dims) → (npix, batch_dims)
    dot_products = jnp.dot(pixel_vecs, safe_vecs)

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
    result = top_indices.T  # (batch_dims, max_length)

    # Replace invalid entries (where sort key was -inf) with npix
    selected_scores = sort_keys[top_indices, jnp.arange(batch_dims)]  # (max_length, batch_dims)
    invalid_mask = selected_scores.T == -jnp.inf  # (batch_dims, max_length)
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
        result = jnp.squeeze(result, axis=0)  # (1, max_length) → (max_length,)

    return result
