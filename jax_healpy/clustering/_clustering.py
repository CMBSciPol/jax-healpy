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
    """Extract the pixels of a cutout from a map, or from each map of a pytree.

    The cutout holds the values of the map at the pixels `indices`, along the axis `axis`:
    it is `jnp.take(ful_map, indices, axis=axis)`, applied to each leaf of a pytree of maps.

    Parameters
    ----------
    ful_map : jax.Array or pytree of jax.Array
        The map, or a pytree of maps, of `npix` pixels along `axis`.
    indices : jax.Array
        The indices of the pixels of the cutout.
    axis : int, optional
        The axis of the pixels.

    Returns
    -------
    jax.Array or pytree of jax.Array
        The cutout, of `len(indices)` pixels along `axis`, or the pytree of the cutouts.

    See Also
    --------
    - [`get_fullmap_from_cutout`][jax_healpy.clustering.get_fullmap_from_cutout]: the inverse operation.
    - [`combine_masks`][jax_healpy.clustering.combine_masks]: the combination of several cutouts into a map.

    Examples
    --------
    The cutout of the pixels more than 20 degrees away from the galactic plane:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> from jax_healpy.clustering import get_cutout_from_mask
    >>> nside = 16
    >>> theta, phi = jhp.pix2ang(nside, jnp.arange(jhp.nside2npix(nside)))
    >>> mask = jnp.abs(theta - jnp.pi / 2) > jnp.deg2rad(20)
    >>> (indices,) = jnp.where(mask)
    >>> full_map = jax.random.normal(jax.random.key(0), (jhp.nside2npix(nside),))
    >>> get_cutout_from_mask(full_map, indices).shape
    (1984,)
    """
    return jax.tree.map(lambda x: jnp.take(x, indices, axis=axis), ful_map)


@jax.jit(static_argnums=(2, 3))
def combine_masks(cutouts: list[Array], indices: list[Array], nside: int, axis: int = 0) -> Array:
    """Combine cutouts into a map, or into a pytree of maps.

    Each cutout fills the pixels of its indices along the axis `axis`, and the pixels that no
    cutout covers are set to [`UNSEEN`][jax_healpy.UNSEEN]. Where cutouts overlap, the last one
    prevails. A pytree of cutouts gives a pytree of maps, leaf by leaf.

    Parameters
    ----------
    cutouts : list[jax.Array] or list of pytrees of jax.Array
        The cutouts, which have the same pytree structure.
    indices : list[jax.Array]
        The indices of the pixels of each cutout.
    nside : int
        The resolution of the map.
    axis : int, optional
        The axis of the pixels in the cutouts.

    Returns
    -------
    jax.Array or pytree of jax.Array
        The map, of `npix` pixels along `axis`, or the pytree of the maps.

    Raises
    ------
    ValueError
        If the numbers of cutouts and of index arrays differ, or if the cutouts do not have the
        same pytree structure.

    See Also
    --------
    - [`get_cutout_from_mask`][jax_healpy.clustering.get_cutout_from_mask]: the extraction of a cutout from a map.
    - [`get_fullmap_from_cutout`][jax_healpy.clustering.get_fullmap_from_cutout]: the map of a single cutout.

    Examples
    --------
    The map of the cutouts of the northern and southern regions more than 20 degrees away from
    the galactic plane:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> from jax_healpy.clustering import combine_masks, get_cutout_from_mask
    >>> nside = 16
    >>> theta, phi = jhp.pix2ang(nside, jnp.arange(jhp.nside2npix(nside)))
    >>> mask = jnp.abs(theta - jnp.pi / 2) > jnp.deg2rad(20)
    >>> (indices,) = jnp.where(mask)
    >>> full_map = jax.random.normal(jax.random.key(0), (jhp.nside2npix(nside),))
    >>> north, south = indices[:992], indices[992:]
    >>> cutouts = [get_cutout_from_mask(full_map, north), get_cutout_from_mask(full_map, south)]
    >>> combined = combine_masks(cutouts, [north, south], nside)
    >>> bool(jnp.array_equal(combined[indices], full_map[indices]))
    True
    >>> bool(jnp.all(combined[~mask] == jhp.UNSEEN))
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
    """Insert a cutout into a map, or each cutout of a pytree into a map.

    The cutout fills the pixels `indices` along the axis `axis`, and the other pixels are set to
    [`UNSEEN`][jax_healpy.UNSEEN].

    Parameters
    ----------
    labels : jax.Array or pytree of jax.Array
        The cutout, of `len(indices)` pixels along `axis`, or a pytree of cutouts.
    indices : jax.Array
        The indices of the pixels of the cutout.
    nside : int
        The resolution of the map.
    axis : int, optional
        The axis of the pixels in the cutout.

    Returns
    -------
    jax.Array or pytree of jax.Array
        The map, of the shape of `labels` but with `npix` pixels along `axis`, or the pytree of
        the maps.

    See Also
    --------
    - [`get_cutout_from_mask`][jax_healpy.clustering.get_cutout_from_mask]: the inverse operation.
    - [`combine_masks`][jax_healpy.clustering.combine_masks]: the map of several cutouts.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> from jax_healpy.clustering import get_cutout_from_mask, get_fullmap_from_cutout
    >>> nside = 16
    >>> theta, phi = jhp.pix2ang(nside, jnp.arange(jhp.nside2npix(nside)))
    >>> mask = jnp.abs(theta - jnp.pi / 2) > jnp.deg2rad(20)
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
    r"""Cluster the pixels of a map into regions, by the k-means algorithm on the sphere.

    Each pixel of `indices` is assigned to the nearest of `n_regions` centroids, at the angular
    distance

    $$\gamma = \arccos\left(\hat{\mathbf{n}}_p \cdot \hat{\mathbf{n}}_c\right)$$

    between the direction $\hat{\mathbf{n}}_p$ of the pixel and that $\hat{\mathbf{n}}_c$ of
    the centroid. The iterations of the k-means algorithm alternate this assignment and the
    update of each centroid to the direction of the mean of the unit vectors of its pixels. They
    stop when the mean distance of the pixels to their centroids decreases by less than a
    relative $10^{-5}$, or after 100 iterations.

    The centroids are drawn at random among the pixels, then fitted to a random sample of
    $\max\left(2\sqrt{N}, s\,n\right)$ pixels, where $N$ is the number of pixels of `indices`,
    $s$ is `initial_sample_size` and $n$ is `max_centroids`, or `n_regions` if it is None. The
    centroids of the sample start the iterations on all the pixels.

    Parameters
    ----------
    mask : jax.Array
        A map, whose size sets the resolution: its values are not used.
    indices : jax.Array
        The indices of the pixels to cluster, in the RING scheme.
    n_regions : int
        The number of regions. It can be traced, for instance by `jax.jit`, if `max_centroids`
        is given.
    key : PRNGKeyArray
        The JAX PRNG key that draws the initial centroids and the sample.
    max_centroids : int, optional
        The maximum number of regions. If given, `n_regions` can be traced, so that a compiled
        function clusters into any number of regions up to `max_centroids`. If None,
        `n_regions` must be a Python integer.
    unassigned : float, optional
        The value of the pixels that are not in `indices`.
    initial_sample_size : int, optional
        The number $s$ of pixels per centroid in the random sample.

    Returns
    -------
    jax.Array
        The map of the labels of the regions, from 0 to `n_regions - 1`, and `unassigned` outside
        `indices`. The labels are numbered in an arbitrary order:
        [`normalize_by_first_occurrence`][jax_healpy.clustering.normalize_by_first_occurrence]
        renumbers them.

    Raises
    ------
    RuntimeError
        If `max_centroids` is given and `n_regions` exceeds it.
    ValueError
        If the random sample has more pixels than `indices`.
    jax.errors.TracerBoolConversionError
        If `n_regions` is traced and `max_centroids` is None.

    Examples
    --------
    Five regions more than 20 degrees away from the galactic plane:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> from jax_healpy.clustering import find_kmeans_clusters
    >>> nside = 16
    >>> theta, phi = jhp.pix2ang(nside, jnp.arange(jhp.nside2npix(nside)))
    >>> mask = jnp.abs(theta - jnp.pi / 2) > jnp.deg2rad(20)
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
    r"""Renumber labels by the order of their first occurrence.

    The labels of a clustering, such as those of
    [`find_kmeans_clusters`][jax_healpy.clustering.find_kmeans_clusters], are numbered in an
    arbitrary order. This function renumbers them by their first occurrence in `arr`: the first
    label met becomes 0, the second one 1, and so on, so that two clusterings that differ only
    by the numbering of their labels get the same labels. The [`UNSEEN`][jax_healpy.UNSEEN]
    values are kept.

    !!! warning
        The renumbering is exact only if `arr` has `n_regions` distinct labels besides
        `UNSEEN`, as the labels of a clustering into `n_regions` regions. With $k$ distinct
        labels, the ranks of their first occurrences are shifted by $n - k$, where $n$ is
        `n_regions`, and clipped to $[0, n - 1]$: if $k > n$, the first labels are merged into
        0 (#44).

    Parameters
    ----------
    arr : jax.Array
        The labels, a one-dimensional array that can contain `UNSEEN` values.
    n_regions : int
        The number of distinct labels besides `UNSEEN`. It can be traced, for instance by
        `jax.jit`.
    max_centroids : int
        An upper bound of the number of distinct labels. It is a static argument.

    Returns
    -------
    jax.Array
        The renumbered labels, of the shape of `arr`, and `UNSEEN` where `arr` is.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax_healpy import UNSEEN
    >>> from jax_healpy.clustering import normalize_by_first_occurrence
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
