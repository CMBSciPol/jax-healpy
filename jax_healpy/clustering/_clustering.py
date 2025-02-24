from functools import partial

import jax
from jax import numpy as jnp
from jaxtyping import Array

import jax_healpy as jhp

from ._kmeans import kmeans_sample

PRNGKey = Array


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


@partial(jax.jit, static_argnums=(2))
def from_cutout_to_fullmap(labels: Array, indices: Array, nside: int) -> Array:
    """Reconstruct the full map from a cutout.

    Args:
        labels (Array): The cutout map labels.
        indices (Array): Indices where the cutout labels should be placed.
        nside (int): HEALPix nside parameter.

    Returns:
        Array: The reconstructed full map.

    Example:

        >>> mask = np.load("GAL20.npy")
        >>> indices, = jnp.where(mask == 1)
        >>> full_map = random.normal(random.key(0), shape=(jhp.nside2npix(64),))
        >>> cutout = get_cutout_from_mask(full_map, indices)
        >>> reconstructed = from_cutout_to_fullmap(cutout, indices, nside=64)
        >>> print(jnp.array_equal(reconstructed, full_map))
    """
    npix = 12 * nside**2
    map_ids = jax.tree.map(lambda x: jnp.full(npix, jhp.UNSEEN), labels)
    return jax.tree.map(lambda maps, lbl: maps.at[indices].set(lbl), map_ids, labels)


def get_clusters(
    mask: Array,
    indices: Array,
    n_regions: int,
    key: PRNGKey,
    max_centroids: None = None,
    unassigned: float = jhp.UNSEEN,
) -> Array:
    """Cluster pixels of a HEALPix map into regions using KMeans.

    Args:
        mask (Array): HEALPix mask.
        indices (Array): Indices of valid pixels.
        n_regions (int): Number of regions to cluster into.
        key (PRNGKey): JAX random key.
        max_centroids (None, optional): Maximum allowed centroids. Defaults to None.
        unassigned (float, optional): Value for unassigned pixels. Defaults to jhp.UNSEEN.

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
    km = kmeans_sample(key, ra_dec, n_regions, max_centroids=max_centroids, maxiter=100, tol=1.0e-5)
    map_ids = jnp.full(npix, unassigned)
    return map_ids.at[ipix[indices]].set(km.labels)
