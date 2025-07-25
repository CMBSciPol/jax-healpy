Pixel Functions (pixelfunc)
===========================

This module provides functions related to HEALPix pixelization scheme, including coordinate conversions, interpolation, and map manipulation functions.

.. automodule:: jax_healpy.pixelfunc
   :members:
   :undoc-members:
   :show-inheritance:

Coordinate Conversions
----------------------

Functions for converting between different coordinate representations:

.. autofunction:: jax_healpy.pix2ang
.. autofunction:: jax_healpy.ang2pix
.. autofunction:: jax_healpy.pix2vec
.. autofunction:: jax_healpy.vec2pix
.. autofunction:: jax_healpy.ang2vec
.. autofunction:: jax_healpy.vec2ang

Pixel Coordinates
-----------------

Functions for working with pixel coordinates within HEALPix faces:

.. autofunction:: jax_healpy.pix2xyf
.. autofunction:: jax_healpy.xyf2pix

Scheme Conversions
------------------

Functions for converting between RING and NESTED pixelization schemes:

.. autofunction:: jax_healpy.nest2ring
.. autofunction:: jax_healpy.ring2nest
.. autofunction:: jax_healpy.reorder

Interpolation
-------------

Functions for interpolating values on the sphere:

.. autofunction:: jax_healpy.get_interp_weights
.. autofunction:: jax_healpy.get_interp_val

HEALPix Parameters
------------------

Functions for working with HEALPix resolution parameters:

.. autofunction:: jax_healpy.nside2npix
.. autofunction:: jax_healpy.npix2nside
.. autofunction:: jax_healpy.nside2order
.. autofunction:: jax_healpy.order2nside
.. autofunction:: jax_healpy.order2npix
.. autofunction:: jax_healpy.npix2order
.. autofunction:: jax_healpy.nside2resol
.. autofunction:: jax_healpy.nside2pixarea

Utility Functions
-----------------

Helper functions for validation and map properties:

.. autofunction:: jax_healpy.isnsideok
.. autofunction:: jax_healpy.isnpixok
.. autofunction:: jax_healpy.maptype

Constants
---------

.. autodata:: jax_healpy.UNSEEN
   :annotation: = 1.6375e+30

   Sentinel value used to mark invalid or missing pixels in HEALPix maps.
