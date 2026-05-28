Clustering Functions
====================

This module provides advanced clustering algorithms and utilities for astronomical data analysis, including K-means clustering and mask manipulation functions.

.. automodule:: jax_healpy.clustering
   :members:
   :undoc-members:
   :show-inheritance:

K-means Clustering
------------------

.. autofunction:: jax_healpy.find_kmeans_clusters

   Cluster pixels of a HEALPix map into regions using KMeans.

   Parameters
   ----------
   mask : Array
       HEALPix mask.
   indices : Array
       Indices of valid pixels.
   n_regions : int
       Number of regions to cluster into.
   key : PRNGKeyArray
       JAX random key.
   max_centroids : int, optional
       Maximum allowed centroids. Defaults to None.
   unassigned : float, optional
       Value for unassigned pixels. Defaults to hp.UNSEEN.
   initial_sample_size : int, optional
       Initial sample size for KMeans. Defaults to 3.

   Returns
   -------
   Array
       Map with clustered region labels.

Mask and Map Utilities
----------------------

Functions for manipulating masks and extracting map regions:

.. autofunction:: jax_healpy.get_cutout_from_mask

   Extract a cutout from a full map using given indices.

   Parameters
   ----------
   ful_map : Array
       The full HEALPix map.
   indices : Array
       Indices for the cutout.
   axis : int, optional
       Axis along which to apply the cutout. Defaults to 0.

   Returns
   -------
   Array
       The cutout map.

.. autofunction:: jax_healpy.get_fullmap_from_cutout

   Reconstruct the full map from a cutout by inserting values along a specified axis.

   Parameters
   ----------
   labels : Array
       The cutout array.
   indices : Array
       The pixel indices for the cutout.
   nside : int
       HEALPix NSIDE.
   axis : int, optional
       The axis in `labels` that corresponds to the patch dimension.

   Returns
   -------
   Array
       Full map array with data inserted.

.. autofunction:: jax_healpy.combine_masks

   Combine multiple cutouts/masks into a single map.

   Parameters
   ----------
   cutouts : list[Array]
       List of cutouts to combine.
   indices : list[Array]
       List of indices corresponding to each cutout.
   nside : int
       HEALPix NSIDE.
   axis : int, optional
       Axis along which to combine. Defaults to 0.

   Returns
   -------
   Array
       Combined map.

Label Utilities
---------------

Functions for manipulating cluster labels:

.. autofunction:: jax_healpy.normalize_by_first_occurrence

   Normalize cluster labels by order of first occurrence.

   Parameters
   ----------
   arr : Array
       Integer array containing raw labels.
   n_regions : int
       Maximum number of regions to preserve.
   max_centroids : int
       Maximum number of unique labels expected.

   Returns
   -------
   Array
       Normalized labels.

Examples
--------

K-means clustering on the sphere:

.. code-block:: python

   import jax.numpy as jnp
   import jax_healpy as hp
   import jax

   # Create a test mask and indices
   nside = 64
   npix = hp.nside2npix(nside)
   mask = jnp.ones(npix)
   indices = jnp.arange(npix)

   # Random key
   key = jax.random.PRNGKey(0)

   # Perform clustering
   clustered_map = hp.find_kmeans_clusters(
       mask,
       indices,
       n_regions=5,
       key=key,
       max_centroids=10
   )

Working with Cutouts:

.. code-block:: python

   # Extract cutout
   cutout = hp.get_cutout_from_mask(clustered_map, indices)

   # Reconstruct full map
   full_map = hp.get_fullmap_from_cutout(cutout, indices, nside=nside)
