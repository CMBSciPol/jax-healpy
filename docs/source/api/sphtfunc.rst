Spherical Harmonic Functions (sphtfunc)
=======================================

This module provides functions for spherical harmonic transforms, enabling conversion between pixel-domain maps and spherical harmonic coefficients.

.. note::
   These functions require the `s2fft` package to be installed. Install with:

   .. code-block:: bash

      pip install jax-healpy[recommended]

.. automodule:: jax_healpy.sphtfunc
   :members:
   :undoc-members:
   :show-inheritance:

Spherical Harmonic Transforms
-----------------------------

Core functions for forward and inverse spherical harmonic transforms:

.. autofunction:: jax_healpy.map2alm

   Convert a HEALPix map to spherical harmonic coefficients.

   This function performs the forward spherical harmonic transform, converting
   a map defined on the HEALPix pixelization into spherical harmonic coefficients
   :math:`a_{\ell m}`.

   Parameters
   ----------
   m : array_like
       Input HEALPix map
   lmax : int, optional
       Maximum multipole moment. If not specified, lmax = 3*nside - 1
   mmax : int, optional
       Maximum azimuthal quantum number. If not specified, mmax = lmax
   iter : int, optional
       Number of iterations for the iterative map2alm algorithm
   use_weights : bool, optional
       Whether to use ring weights for improved accuracy

   Returns
   -------
   alm : array_like
       Complex spherical harmonic coefficients

.. autofunction:: jax_healpy.alm2map

   Convert spherical harmonic coefficients to a HEALPix map.

   This function performs the inverse spherical harmonic transform, converting
   spherical harmonic coefficients :math:`a_{\ell m}` into a map defined on
   the HEALPix pixelization.

   Parameters
   ----------
   alm : array_like
       Input spherical harmonic coefficients
   nside : int
       HEALPix resolution parameter
   lmax : int, optional
       Maximum multipole moment
   mmax : int, optional
       Maximum azimuthal quantum number
   pixwin : bool, optional
       Whether to apply pixel window function correction

   Returns
   -------
   m : array_like
       Output HEALPix map

Examples
--------

Basic spherical harmonic transform:

.. code-block:: python

   import jax.numpy as jnp
   import jax_healpy as hp

   # Create a test map
   nside = 64
   npix = hp.nside2npix(nside)
   test_map = jnp.random.normal(0, 1, npix)

   # Forward transform: map -> alm
   alm = hp.map2alm(test_map, lmax=128)

   # Inverse transform: alm -> map
   reconstructed_map = hp.alm2map(alm, nside=nside)

   # Check reconstruction accuracy
   rms_error = jnp.sqrt(jnp.mean((test_map - reconstructed_map)**2))
   print(f"RMS reconstruction error: {rms_error}")

Working with specific multipole ranges:

.. code-block:: python

   # Transform with specific lmax and mmax
   lmax, mmax = 64, 32
   alm = hp.map2alm(test_map, lmax=lmax, mmax=mmax)

   # Reconstruct with the same parameters
   reconstructed = hp.alm2map(alm, nside=nside, lmax=lmax, mmax=mmax)

Batch processing multiple maps:

.. code-block:: python

   # Process multiple maps simultaneously
   batch_maps = jnp.random.normal(0, 1, (10, npix))

   # Vectorized transform
   batch_alm = jax.vmap(lambda m: hp.map2alm(m, lmax=64))(batch_maps)

   # Vectorized inverse transform
   batch_reconstructed = jax.vmap(lambda a: hp.alm2map(a, nside=nside))(batch_alm)

Mathematical Background
-----------------------

The spherical harmonic transform decomposes a function :math:`f(\theta, \phi)`
on the sphere into spherical harmonic coefficients:

.. math::

   a_{\ell m} = \int f(\theta, \phi) Y_{\ell m}^*(\theta, \phi) d\Omega

where :math:`Y_{\ell m}(\theta, \phi)` are the spherical harmonic basis functions.

The inverse transform reconstructs the function:

.. math::

   f(\theta, \phi) = \sum_{\ell=0}^{\ell_{max}} \sum_{m=-\ell}^{\ell} a_{\ell m} Y_{\ell m}(\theta, \phi)

HEALPix pixelization provides an efficient discretization of the sphere that
enables fast spherical harmonic transforms while maintaining good accuracy
for most astronomical applications.

Performance Notes
-----------------

- Transforms are automatically JIT-compiled for optimal performance
- GPU acceleration is supported when JAX is configured with CUDA/ROCm
- Memory usage scales as O(npix) for maps and O(lmax * mmax) for coefficients
- For large transforms, consider using batch processing to manage memory usage
