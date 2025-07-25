Clustering Functions
====================

This module provides advanced clustering algorithms and utilities for astronomical data analysis, including K-means clustering and mask manipulation functions.

.. automodule:: jax_healpy.clustering
   :members:
   :undoc-members:
   :show-inheritance:

K-means Clustering
------------------

.. autoclass:: jax_healpy.KMeans
   :members:
   :undoc-members:
   :show-inheritance:

   JAX-based implementation of K-means clustering algorithm optimized for HEALPix data.

   This class provides a high-performance implementation of K-means clustering
   with support for GPU acceleration and automatic differentiation.

.. autofunction:: jax_healpy.kmeans_sample

   Perform K-means clustering on sample data.

   Parameters
   ----------
   data : array_like
       Input data array of shape (n_samples, n_features)
   n_clusters : int
       Number of clusters
   max_iter : int, optional
       Maximum number of iterations (default: 100)
   tol : float, optional
       Convergence tolerance (default: 1e-4)
   init : str or array_like, optional
       Initialization method ('random', 'k-means++', or array of initial centroids)
   random_state : int, optional
       Random seed for reproducible results

   Returns
   -------
   centroids : array_like
       Final cluster centroids
   labels : array_like
       Cluster labels for each sample
   inertia : float
       Sum of squared distances to centroids

Mask and Map Utilities
----------------------

Functions for manipulating masks and extracting map regions:

.. autofunction:: jax_healpy.get_clusters

   Extract connected clusters from a binary mask.

   Identifies connected regions in a HEALPix mask and assigns cluster labels.

   Parameters
   ----------
   mask : array_like
       Binary mask array (True for valid pixels)
   connectivity : str, optional
       Connectivity definition ('face', 'edge', or 'vertex')
   min_size : int, optional
       Minimum cluster size (smaller clusters are ignored)

   Returns
   -------
   labels : array_like
       Cluster labels (0 for background, 1+ for clusters)
   n_clusters : int
       Number of identified clusters

.. autofunction:: jax_healpy.get_cutout_from_mask

   Extract a cutout region from a HEALPix map based on a mask.

   Parameters
   ----------
   healpix_map : array_like
       Input HEALPix map
   mask : array_like
       Binary mask defining the cutout region
   buffer_size : int, optional
       Additional pixels to include around the mask boundary

   Returns
   -------
   cutout : array_like
       Extracted cutout data
   cutout_mask : array_like
       Mask for the cutout region
   pixel_indices : array_like
       Original pixel indices for the cutout

.. autofunction:: jax_healpy.from_cutout_to_fullmap

   Insert cutout data back into a full HEALPix map.

   Parameters
   ----------
   cutout : array_like
       Cutout data to insert
   pixel_indices : array_like
       Original pixel indices from get_cutout_from_mask
   nside : int
       HEALPix resolution parameter for the full map
   fill_value : float, optional
       Value for pixels not in the cutout (default: 0)

   Returns
   -------
   full_map : array_like
       Full HEALPix map with cutout data inserted

.. autofunction:: jax_healpy.combine_masks

   Combine multiple binary masks using logical operations.

   Parameters
   ----------
   masks : list of array_like
       List of binary masks to combine
   operation : str, optional
       Logical operation ('and', 'or', 'xor') (default: 'or')

   Returns
   -------
   combined_mask : array_like
       Combined binary mask

Label Utilities
---------------

Functions for manipulating cluster labels:

.. autofunction:: jax_healpy.normalize_by_first_occurrence

   Normalize cluster labels by order of first occurrence.

   Relabels clusters so that the first encountered cluster gets label 1,
   the second gets label 2, etc.

   Parameters
   ----------
   labels : array_like
       Input cluster labels

   Returns
   -------
   normalized_labels : array_like
       Normalized cluster labels

.. autofunction:: jax_healpy.shuffle_labels

   Randomly shuffle cluster labels while preserving cluster structure.

   Parameters
   ----------
   labels : array_like
       Input cluster labels
   random_state : int, optional
       Random seed for reproducible results

   Returns
   -------
   shuffled_labels : array_like
       Labels with shuffled cluster identities

Examples
--------

Basic K-means clustering:

.. code-block:: python

   import jax.numpy as jnp
   import jax_healpy as hp

   # Generate sample data
   data = jnp.random.normal(0, 1, (1000, 3))

   # Perform K-means clustering
   centroids, labels, inertia = hp.kmeans_sample(data, n_clusters=5)

   print(f"Final inertia: {inertia}")
   print(f"Cluster sizes: {jnp.bincount(labels)}")

Using the KMeans class:

.. code-block:: python

   # Initialize K-means object
   kmeans = hp.KMeans(n_clusters=3, max_iter=100, tol=1e-4)

   # Fit the model
   kmeans.fit(data)

   # Get predictions for new data
   new_data = jnp.random.normal(0, 1, (100, 3))
   predictions = kmeans.predict(new_data)

Working with HEALPix masks:

.. code-block:: python

   # Create a test mask
   nside = 64
   npix = hp.nside2npix(nside)

   # Generate random clusters
   mask = jnp.random.random(npix) > 0.8

   # Find connected clusters
   cluster_labels, n_clusters = hp.get_clusters(mask, min_size=10)

   print(f"Found {n_clusters} clusters")

Extracting map cutouts:

.. code-block:: python

   # Create test map and mask
   test_map = jnp.random.normal(0, 1, npix)
   region_mask = cluster_labels == 1  # Focus on cluster 1

   # Extract cutout
   cutout, cutout_mask, indices = hp.get_cutout_from_mask(
       test_map, region_mask, buffer_size=5
   )

   # Process cutout data
   processed_cutout = cutout * 2.0  # Example processing

   # Insert back into full map
   result_map = hp.from_cutout_to_fullmap(
       processed_cutout, indices, nside, fill_value=hp.UNSEEN
   )

Performance Tips
----------------

- K-means clustering benefits significantly from GPU acceleration
- Use JIT compilation for repeated clustering operations:

.. code-block:: python

   @jax.jit
   def fast_kmeans(data, n_clusters):
       return hp.kmeans_sample(data, n_clusters)

- For large datasets, consider using mini-batch K-means or data sampling
- Mask operations are vectorized and GPU-accelerated automatically
