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


import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import jax_healpy as jhp

from ..pixelfunc import UNSEEN
from ._kmeans import kmeans_sample


def call_back_check(n_regions: Array, max_centroids: None) -> None:
    """Check if the number of regions exceeds the maximum centroids.

    Parameters
    ----------
    n_regions : Array
        Number of regions requested.
    max_centroids : int or None
        Maximum allowed centroids.

    Raises
    ------
    RuntimeError
        If `n_regions` exceeds `max_centroids`.
    """
    if max_centroids is not None and n_regions > max_centroids:
        raise RuntimeError("""
            In function [get_clusters] in the comp_sep module:
            Number of regions (n_regions) is greater than max_centroids.
            Either:
            - Increase max_centroids.
            - Set max_centroids to None, but n_regions will have
              to be static and can no longer be a tracer.
            """)


@jax.jit(static_argnums=(2))
def get_cutout_from_mask(ful_map: Array, indices: Array, axis: int = 0) -> Array:
    """Extract a cutout from a full map using given indices.

    Parameters
    ----------
    ful_map : Array or pytree of Arrays
        The full HEALPix map(s).
    indices : Array
        Pixel indices of the cutout.
    axis : int, optional
        Axis along which to apply the cutout. Default: 0.

    Returns
    -------
    Array or pytree of Arrays
        The cutout map(s).

    See Also
    --------
    get_fullmap_from_cutout : The inverse operation.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> nside = 16
    >>> theta, phi = jhp.pix2ang(nside, jnp.arange(jhp.nside2npix(nside)))
    >>> mask = jnp.abs(theta - jnp.pi / 2) > jnp.deg2rad(20)  # galactic cut
    >>> (indices,) = jnp.where(mask)
    >>> full_map = jax.random.normal(jax.random.key(0), (jhp.nside2npix(nside),))
    >>> cutout = get_cutout_from_mask(full_map, indices)
    >>> cutout.shape
    (1984,)
    """
    return jax.tree.map(lambda x: jnp.take(x, indices, axis=axis), ful_map)


@jax.jit(static_argnums=(2, 3))
def combine_masks(cutouts: list[Array], indices: list[Array], nside: int, axis: int = 0) -> Array:
    """Combine several cutouts into a full map.

    Parameters
    ----------
    cutouts : list of Arrays or of pytrees of Arrays
        The cutouts, which must all have the same pytree structure.
    indices : list of Arrays
        The pixel indices of each cutout.
    nside : int
        HEALPix nside of the full map.
    axis : int, optional
        The axis of the cutouts that corresponds to the pixel dimension. Default: 0.

    Returns
    -------
    Array or pytree of Arrays
        Full map(s), with `UNSEEN` in the pixels not covered by any cutout. Where cutouts
        overlap, the last one wins.

    Raises
    ------
    ValueError
        If the numbers of cutouts and indices differ, or if the cutouts do not have the same
        structure.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> nside = 16
    >>> theta, phi = jhp.pix2ang(nside, jnp.arange(jhp.nside2npix(nside)))
    >>> mask = jnp.abs(theta - jnp.pi / 2) > jnp.deg2rad(20)  # galactic cut
    >>> (indices,) = jnp.where(mask)
    >>> full_map = jax.random.normal(jax.random.key(0), (jhp.nside2npix(nside),))
    >>> north, south = indices[:992], indices[992:]
    >>> cutouts = [get_cutout_from_mask(full_map, north), get_cutout_from_mask(full_map, south)]
    >>> combined = combine_masks(cutouts, [north, south], nside)
    >>> bool(jnp.array_equal(combined[indices], full_map[indices]))
    True
    """
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

    for cutout, patch_indices in zip(cutouts, indices):
        patch_slice = [slice(None)] * len(jax.tree.leaves(cutout)[0].shape)
        patch_slice[axis] = patch_indices
        patch_slice = tuple(patch_slice)
        map_ids = jax.tree.map(lambda maps, lbl: maps.at[patch_slice].set(lbl), map_ids, cutout)  # noqa: B023

    return map_ids


@jax.jit(static_argnums=(2, 3))
def get_fullmap_from_cutout(labels: Array, indices: Array, nside: int, axis: int = 0) -> Array:
    """Reconstruct the full map from a cutout by inserting values along a specified axis.

    Parameters
    ----------
    labels : Array or pytree of Arrays
        The cutout array(s), of shape `(..., npatch, ...)`.
    indices : Array
        The pixel indices of the cutout.
    nside : int
        HEALPix nside of the full map.
    axis : int, optional
        The axis of `labels` that corresponds to the patch dimension, to be expanded to
        `npix`. Default: 0.

    Returns
    -------
    Array or pytree of Arrays
        Full map(s) with the shape of `labels`, but with `npatch` replaced by `npix` along
        `axis`. The pixels outside the cutout are set to `UNSEEN`.

    See Also
    --------
    get_cutout_from_mask : The inverse operation.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> nside = 16
    >>> theta, phi = jhp.pix2ang(nside, jnp.arange(jhp.nside2npix(nside)))
    >>> mask = jnp.abs(theta - jnp.pi / 2) > jnp.deg2rad(20)  # galactic cut
    >>> (indices,) = jnp.where(mask)
    >>> full_map = jax.random.normal(jax.random.key(0), (jhp.nside2npix(nside),))
    >>> cutout = get_cutout_from_mask(full_map, indices)
    >>> reconstructed = get_fullmap_from_cutout(cutout, indices, nside=nside)
    >>> bool(jnp.array_equal(reconstructed[mask], full_map[mask]))
    True
    >>> bool(jnp.all(reconstructed[~mask] == jhp.UNSEEN))
    True
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


def find_kmeans_clusters(
    mask: Array,
    indices: Array,
    n_regions: int,
    key: PRNGKeyArray,
    max_centroids: int | None = None,
    unassigned: float = UNSEEN,
    initial_sample_size: int = 3,
) -> Array:
    """Cluster pixels of a HEALPix map into regions using KMeans.

    Parameters
    ----------
    mask : Array
        HEALPix mask. Only its size is used, to determine the map nside.
    indices : Array
        Indices of the valid pixels, which are the pixels to be clustered.
    n_regions : int
        Number of regions to cluster into.
    key : PRNGKeyArray
        JAX random key.
    max_centroids : int, optional
        Maximum number of centroids. If provided, `n_regions` can be a tracer, so that the
        function can be jitted for different numbers of regions without recompilation.
    unassigned : float, optional
        Value of the pixels that are not in `indices`. Default: `UNSEEN`.
    initial_sample_size : int, optional
        The centroids are initialized from a random sample of
        `initial_sample_size * n_regions` pixels. Default: 3.

    Returns
    -------
    Array
        Map of the region labels, from 0 to `n_regions - 1`, and `unassigned` outside
        `indices`.

    Raises
    ------
    RuntimeError
        If `n_regions` exceeds `max_centroids` when the latter is provided.
    TracerBoolConversionError
        If `n_regions` is a tracer and `max_centroids` is None.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> nside = 16
    >>> theta, phi = jhp.pix2ang(nside, jnp.arange(jhp.nside2npix(nside)))
    >>> mask = jnp.abs(theta - jnp.pi / 2) > jnp.deg2rad(20)  # galactic cut
    >>> (indices,) = jnp.where(mask)
    >>> labels = find_kmeans_clusters(mask, indices, n_regions=5, key=jax.random.key(0))
    >>> print(jnp.unique(labels[indices]))
    [0. 1. 2. 3. 4.]
    >>> bool(jnp.all(labels[~mask] == jhp.UNSEEN))
    True
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


@jax.jit(static_argnums=(2,))
def normalize_by_first_occurrence(arr: Array, n_regions: int, max_centroids: int) -> Array:
    """Relabel an array by the order of first occurrence of its values.

    Each unique value is replaced by the rank of its first appearance, so that the labels
    are contiguous, compact and order-consistent. Values beyond the first `n_regions`
    unique values are clipped to `[0, n_regions - 1]`, and `UNSEEN` values are kept.

    This is useful after clustering or segmentation tasks, for downstream processing.

    Parameters
    ----------
    arr : Array
        Array of raw labels, possibly containing `UNSEEN` values.
    n_regions : int
        Maximum number of regions (unique labels) to preserve. Others are clipped.
    max_centroids : int
        Maximum number of unique labels expected. It must be static under `jax.jit`.

    Returns
    -------
    Array
        Array of the same shape as `arr`, where each label is replaced by the rank of its
        first occurrence, and `UNSEEN` values are kept.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax_healpy import UNSEEN
    >>> arr = jnp.array([UNSEEN, UNSEEN, 5, 5, 5, 2, 3, 3, 8])
    >>> labels = normalize_by_first_occurrence(arr, 4, 10)
    >>> print(labels[2:])
    [0. 0. 0. 1. 2. 2. 3.]
    >>> bool(jnp.all(labels[:2] == UNSEEN))
    True
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
    """Randomly reassign label indices using a NumPy-based permutation.

    The input labels are assumed to be normalized integers in `[0, N)`, with possible
    `UNSEEN` entries. The function applies a random bijective mapping to the labels,
    preserving the shape and leaving the `UNSEEN` values unchanged.

    This is intended for visualization purposes: shuffling label indices can reduce
    misleading visual patterns (e.g., color clumping) in plots such as `mollview`, making
    the class structure easier to interpret.

    Parameters
    ----------
    arr : Array
        Array of label indices, e.g. `[0, 0, 1, 2, UNSEEN]`. It cannot be a tracer.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as `arr`, with valid labels randomly permuted and `UNSEEN`
        entries unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> from jax_healpy import UNSEEN
    >>> arr = np.array([0, 0, 1, 1, 2, UNSEEN])
    >>> shuffle_labels(arr)  # doctest: +SKIP
    array([ 2.0000e+00,  2.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00, -1.6375e+30])
    """
    try:
        arr = np.asarray(arr)
    except jax.errors.TracerArrayConversionError:
        raise ValueError('shuffle_labels is not compatible with JIT.') from None

    unique_vals = np.unique(arr[arr != UNSEEN])
    shuffled_vals = np.random.permutation(unique_vals)

    mapping = dict(zip(unique_vals, shuffled_vals))
    shuffled_arr = np.vectorize(lambda x: mapping.get(x, UNSEEN))(arr)
    return shuffled_arr.astype(arr.dtype)
