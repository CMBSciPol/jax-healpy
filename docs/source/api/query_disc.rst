Query Disc Functions
====================

This module provides functions for querying pixels within circular regions on the sphere, essential for astronomical applications involving point sources, aperture photometry, and local analysis.

.. automodule:: jax_healpy._query_disc
   :members:
   :undoc-members:
   :show-inheritance:

Disc Query Functions
--------------------

.. autofunction:: jax_healpy.query_disc

Disc Estimation Functions
-------------------------

.. autofunction:: jax_healpy.estimate_disc_pixel_count

   Estimate the number of pixels within a circular disc on the sphere.

.. autofunction:: jax_healpy.estimate_disc_radius

   Estimate the angular radius needed to contain a given number of pixels.

   Find all pixels within a circular region on the sphere.

   This function identifies all HEALPix pixels whose centers lie within
   a circular region defined by a center direction and angular radius.

   Parameters
   ----------
   nside : int
       HEALPix resolution parameter
   vec : array_like
       Unit vector pointing to the disc center, shape (3,)
   radius : float
       Angular radius of the disc in radians
   inclusive : bool, optional
       If True, include pixels that are partially within the disc.
       If False, only include pixels whose centers are within the disc.
       Default: False
   fact : int, optional
       Factor for subdividing pixels when inclusive=True. Higher values
       give more accurate results but are slower. Default: 4
   nest : bool, optional
       If True, return pixel indices in NESTED ordering.
       If False, return in RING ordering. Default: False

   Returns
   -------
   pixels : array_like
       Array of pixel indices within the specified disc

   Examples
   --------

   Basic disc query:

   .. code-block:: python

      import jax.numpy as jnp
      import jax_healpy as hp

      # Define disc center (North pole)
      center_vec = jnp.array([0.0, 0.0, 1.0])

      # Query pixels within 10 degree radius
      radius = jnp.radians(10.0)  # Convert to radians
      nside = 64

      pixels = hp.query_disc(nside, center_vec, radius)
      print(f"Found {len(pixels)} pixels in disc")

   Using angular coordinates:

   .. code-block:: python

      # Convert from theta, phi to unit vector
      theta, phi = jnp.radians(45.0), jnp.radians(30.0)  # 45° from pole, 30° azimuth
      center_vec = hp.ang2vec(theta, phi)

      # Query with inclusive boundary
      pixels = hp.query_disc(nside, center_vec, radius, inclusive=True)

   Batch processing multiple discs:

   .. code-block:: python

      # Define multiple disc centers
      centers = jnp.array([
          [0.0, 0.0, 1.0],    # North pole
          [1.0, 0.0, 0.0],    # X-axis
          [0.0, 1.0, 0.0]     # Y-axis
      ])

      # Query each disc
      all_pixels = []
      for center in centers:
          pixels = hp.query_disc(nside, center, radius)
          all_pixels.append(pixels)

Mathematical Background
-----------------------

The query_disc function finds pixels whose centers satisfy:

.. math::

   \cos(\text{radius}) \leq \vec{v}_{\text{center}} \cdot \vec{v}_{\text{pixel}}

where :math:`\vec{v}_{\text{center}}` is the disc center unit vector and
:math:`\vec{v}_{\text{pixel}}` is the pixel center unit vector.

For inclusive queries, the function also considers pixels that are partially
within the disc boundary by subdividing pixel boundaries and checking if
any subdivision points fall within the disc.

Applications
------------

Common use cases for disc queries include:

**Point Source Analysis**

.. code-block:: python

   # Find pixels around a known source position
   source_ra, source_dec = 83.6331, 22.0145  # Crab Nebula (degrees)

   # Convert to HEALPix coordinates
   theta = jnp.radians(90.0 - source_dec)  # Convert Dec to colatitude
   phi = jnp.radians(source_ra)
   source_vec = hp.ang2vec(theta, phi)

   # Query 1-degree radius around source
   aperture_radius = jnp.radians(1.0)
   source_pixels = hp.query_disc(nside, source_vec, aperture_radius)

**Aperture Photometry**

.. code-block:: python

   # Extract flux within aperture
   sky_map = jnp.random.normal(0, 1, hp.nside2npix(nside))

   aperture_pixels = hp.query_disc(nside, source_vec, aperture_radius)
   aperture_flux = jnp.sum(sky_map[aperture_pixels])

   print(f"Total flux in aperture: {aperture_flux}")

**Local Statistics**

.. code-block:: python

   # Compute local statistics around multiple positions
   positions = jnp.array([
       hp.ang2vec(jnp.radians(30), jnp.radians(0)),
       hp.ang2vec(jnp.radians(60), jnp.radians(90)),
       hp.ang2vec(jnp.radians(90), jnp.radians(180))
   ])

   local_stats = []
   for pos in positions:
       pixels = hp.query_disc(nside, pos, aperture_radius)
       local_mean = jnp.mean(sky_map[pixels])
       local_std = jnp.std(sky_map[pixels])
       local_stats.append((local_mean, local_std))

**Masking and Exclusion**

.. code-block:: python

   # Create exclusion mask around bright sources
   bright_sources = [
       hp.ang2vec(jnp.radians(45), jnp.radians(0)),
       hp.ang2vec(jnp.radians(135), jnp.radians(90))
   ]

   exclusion_mask = jnp.ones(hp.nside2npix(nside))
   exclusion_radius = jnp.radians(5.0)  # 5-degree exclusion

   for source in bright_sources:
       excluded_pixels = hp.query_disc(nside, source, exclusion_radius)
       exclusion_mask = exclusion_mask.at[excluded_pixels].set(0)

Performance Considerations
--------------------------

- Query disc is optimized for GPU execution when JAX is configured with CUDA/ROCm
- Performance scales with disc size and resolution (nside)
- For many small discs, consider batching operations:

.. code-block:: python

   @jax.jit
   def batch_query_disc(nside, centers, radius):
       # Vectorized implementation for multiple discs
       return jax.vmap(lambda c: hp.query_disc(nside, c, radius))(centers)

- The `inclusive` parameter significantly affects computation time
- For very large discs (> 60°), consider alternative approaches

Accuracy Notes
--------------

- Standard queries (inclusive=False) are exact for pixel centers
- Inclusive queries provide better boundary accuracy at computational cost
- Higher `fact` values in inclusive mode improve accuracy but slow computation
- For critical applications, validate results against known geometric cases

See Also
--------

- :func:`jax_healpy.ang2vec` : Convert angular coordinates to unit vectors
- :func:`jax_healpy.pix2vec` : Get unit vectors for pixel centers
- :func:`jax_healpy.get_interp_weights` : For interpolation within regions
