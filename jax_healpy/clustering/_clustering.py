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

from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import jax_healpy as jhp

from ..pixelfunc import UNSEEN
from ._kmeans import kmeans_sample


def call_back_check(n_regions: Array, max_centroids: None) -> None:
    """Check if the number of regions exceeds the maximum centroids.

    Args:
        n_regions (Array): Number of regions requested.
        max_centroids (None): Maximum allowed centroids.

    Raises:
        RuntimeError: If n_regions exceeds max_centroids.
    """
    if max_centroids is not None:
        if n_regions > max_centroids:
            raise RuntimeError("""
            In function [get_clusters] in the comp_sep module:
            Number of regions (n_regions) is greater than max_centroids.
            Either:
            - Increase max_centroids.
            - Set max_centroids to None, but n_regions will have
              to be static and can no longer be a tracer.
            """)


@partial(jax.jit, static_argnums=(2))
def get_cutout_from_mask(ful_map: Array, indices: Array, axis: int = 0) -> Array:
    """Extract a cutout from a full map using given indices.

    Args:
        ful_map (Array): The full HEALPix map.
        indices (Array): Indices for the cutout.
        axis (int, optional): Axis along which to apply the cutout. Defaults to 0.

    Returns:
        Array: The cutout map.

    Example:

        >>> mask = np.load("GAL20.npy")
        >>> indices, = jnp.where(mask == 1)
        >>> full_map = random.normal(random.key(0), shape=(jhp.nside2npix(64),))
        >>> cutout = get_cutout_from_mask(full_map, indices)
        >>> print(cutout.shape)
    """
    return jax.tree.map(lambda x: jnp.take(x, indices, axis=axis), ful_map)


@partial(jax.jit, static_argnums=(2, 3))
def combine_masks(cutouts: list[Array], indices: list[Array], nside: int, axis: int = 0) -> Array:
    if len(cutouts) != len(indices):
        raise ValueError(' The number of cutouts and indices must match.')
    structure = jax.tree.structure(cutouts[0])
    for cutout in cutouts[1:]:
        if jax.tree.structure(cutout) != structure:
            raise ValueError('All cutouts must have the same structure.')

    npix = 12 * nside**2
    full_shape = list(jax.tree.leaves(cutouts)[0].shape)
    full_shape[axis] = npix
    map_ids = jax.tree.map(lambda x: jnp.full(full_shape, UNSEEN), cutouts[0])

    for cutout, indices in zip(cutouts, indices):
        patch_slice = [slice(None)] * len(jax.tree.leaves(cutout)[0].shape)
        patch_slice[axis] = indices
        patch_slice = tuple(patch_slice)
        map_ids = jax.tree.map(lambda maps, lbl: maps.at[patch_slice].set(lbl), map_ids, cutout)

    return map_ids


@partial(jax.jit, static_argnums=(2, 3))
def from_cutout_to_fullmap(labels: Array, indices: Array, nside: int, axis: int = 0) -> Array:
    """
    Reconstruct the full map from a cutout by inserting values along a specified axis.

    Args:
        labels (Array): The cutout array, shape [..., npatch, ...].
        indices (Array): The pixel indices for the cutout.
        nside (int): HEALPix NSIDE.
        axis (int): The axis in `labels` that corresponds to the patch dimension (to be expanded to npix).

    Returns:
        Array: Full map array with shape like `labels`, but with `npatch` → `npix` along the specified axis.
    Example:

        >>> mask = np.load("GAL20.npy")
        >>> indices, = jnp.where(mask == 1)
        >>> full_map = random.normal(random.key(0), shape=(jhp.nside2npix(64),))
        >>> cutout = get_cutout_from_mask(full_map, indices)
        >>> reconstructed = from_cutout_to_fullmap(cutout, indices, nside=64)
        >>> print(jnp.array_equal(reconstructed, full_map))
    """
    npix = 12 * nside**2

    def insert_fn(lbl):
        full_shape = list(lbl.shape)
        full_shape[axis] = npix
        base = jnp.full(full_shape, UNSEEN)
        slicing = [slice(None)] * lbl.ndim
        slicing[axis] = indices
        slicing = tuple(slicing)
        return base.at[slicing].set(lbl)

    return jax.tree.map(insert_fn, labels)


def get_clusters(
    mask: Array,
    indices: Array,
    n_regions: int,
    key: PRNGKeyArray,
    max_centroids: None = None,
    unassigned: float = UNSEEN,
    initial_sample_size: int = 3,
) -> Array:
    """Cluster pixels of a HEALPix map into regions using KMeans.

    Args:
        mask (Array): HEALPix mask.
        indices (Array): Indices of valid pixels.
        n_regions (int): Number of regions to cluster into.
        key (PRNGKeyArray): JAX random key.
        max_centroids (None, optional): Maximum allowed centroids. Defaults to None.
        unassigned (float, optional): Value for unassigned pixels. Defaults to jhp.UNSEEN.
        initial_sample_size (int, optional): Initial sample size for KMeans. Defaults to 3.
            It is used to initialize the centroids.
            The sample size is initial_sample_size * n_regions.

    Returns:
        Array: Map with clustered region labels.

    Raises:
        RuntimeError: If n_regions exceeds max_centroids when provided.
        TracerBoolConversionError: If n_regions is a tracer and max_centroids is None.

    Example:
        >>> import numpy as np
        >>> from jax import numpy as jnp, random
        >>> import jax_healpy as jhp

        # Load mask and identify valid pixels
        >>> mask = np.load("GAL20.npy")
        >>> indices, = jnp.where(mask == 1)
        >>> key = random.key(0)

        # Perform clustering
        >>> clustered_map = get_clusters(mask, indices, n_regions=5, key=key)
        >>> print(jnp.unique(clustered_map))
        [0 1 2 3 4]

        # Error example when max_centroids constraint is violated
        >>> try:
        ...     clustered_map = get_clusters(mask, indices, n_regions=15, key=key, max_centroids=10)
        ... except RuntimeError as e:
        ...     print(e)
    """
    jax.debug.callback(call_back_check, n_regions, max_centroids)

    npix = mask.size
    nside = jhp.npix2nside(npix)
    ipix = jnp.arange(npix)
    ra, dec = jhp.pix2ang(nside, ipix, lonlat=True)
    ra_dec = jnp.stack([ra[indices], dec[indices]], axis=-1)
    km = kmeans_sample(
        key,
        ra_dec,
        n_regions,
        max_centroids=max_centroids,
        maxiter=100,
        tol=1.0e-5,
        initial_sample_size=initial_sample_size,
    )
    map_ids = jnp.full(npix, unassigned)
    return map_ids.at[ipix[indices]].set(km.labels)


@partial(jax.jit, static_argnums=(2,))
def normalize_by_first_occurrence(arr: Array, n_regions: int, max_centroids: int) -> Array:
    """
    Normalize an array by mapping each unique value to the index of its first occurrence,
    preserving order up to `n_regions` values.

    Any value not among the first `n_regions` unique elements (determined by order of
    appearance) is clipped to fit within `[0, n_regions - 1]`, or set to `UNSEEN` if
    originally marked as such.

    This is useful after clustering or segmentation tasks to ensure label indices are
    contiguous, compact, and order-consistent for downstream processing.

    Args:
        arr: Integer array (1D or ND) containing raw labels, including possible `UNSEEN` markers.
        n_regions: Maximum number of regions (unique labels) to preserve. Others are clipped.
        max_centroids: Maximum number of unique labels expected (must be static for JIT).

    Returns:
        An array of same shape as `arr`, where each label is replaced by its first-seen index,
        or `UNSEEN` if it was originally marked or beyond `n_regions`.

    Example:
        >>> arr = jnp.array([UNSEEN, UNSEEN, 5, 5, 5, 2, 3, 3, 8])
        >>> normalize_by_first_occurrence(arr, 4, 10)
        Array([UNSEEN, UNSEEN, 0, 0, 0, 1, 2, 2, 3])
    """
    arr_unseen = jnp.concatenate([jnp.array([UNSEEN]), arr])

    unique_vals, first_idxs = jnp.unique(arr_unseen, size=max_centroids + 1, return_index=True)
    order = jnp.argsort(first_idxs)
    unique_by_first = unique_vals[order]
    matches = arr_unseen[..., None] == unique_by_first
    idxs = jnp.argmax(matches, axis=-1)
    no_match = ~jnp.any(matches, axis=-1)
    normalized = jnp.where(no_match, UNSEEN, idxs)
    normalized = normalized[1:]
    normalized = jnp.where(
        arr == UNSEEN, UNSEEN, jnp.clip(normalized - (max_centroids - n_regions) - 1, 0, n_regions - 1)
    )
    return normalized


def shuffle_labels(arr: Array) -> Array:
    """
    Randomly reassigns label indices using a NumPy-based permutation.

    Assumes that input labels are normalized integers in [0, N), with possible `hp.UNSEEN`
    entries. The function produces a random bijective mapping of present labels, preserving
    shape and leaving `hp.UNSEEN` values unchanged.

    This is intended for visualization purposes — shuffling label indices can reduce
    misleading visual patterns (e.g., color clumping) in plots such as `mollview`, making
    class structure easier to interpret.

    Args:
        arr: Integer array of label indices, e.g., [0, 0, 1, 2, hp.UNSEEN].

    Returns:
        A NumPy array of the same shape as `arr`, with valid labels randomly permuted.
        `hp.UNSEEN` entries are left unchanged.

    Example:
        >>> arr = np.array([0, 0, 1, 1, 2, hp.UNSEEN])
        >>> shuffle_labels(arr)
        array([2, 2, 0, 0, 1, hp.UNSEEN])  # result will vary
    """
    unique_vals = np.unique(arr[arr != UNSEEN])
    shuffled_vals = np.random.permutation(unique_vals)

    mapping = dict(zip(unique_vals, shuffled_vals))
    shuffled_arr = np.vectorize(lambda x: mapping.get(x, UNSEEN))(arr)
    return shuffled_arr.astype(np.float64)
