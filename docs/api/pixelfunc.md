# Pixel Functions

Functions related to the HEALPix pixelization scheme: coordinate conversions, pixel ordering
schemes, resolution parameters, interpolation and neighbours.

## Coordinate Conversions

::: jax_healpy.pixelfunc.pix2ang
::: jax_healpy.pixelfunc.ang2pix
::: jax_healpy.pixelfunc.pix2vec
::: jax_healpy.pixelfunc.vec2pix
::: jax_healpy.pixelfunc.ang2vec
::: jax_healpy.pixelfunc.vec2ang
::: jax_healpy.pixelfunc.pix2loc
::: jax_healpy.pixelfunc.loc2pix

## Pixel Coordinates

Coordinates of the pixels within the 12 base faces.

::: jax_healpy.pixelfunc.pix2xyf
::: jax_healpy.pixelfunc.xyf2pix

## Pixel Ordering Schemes

Conversions between the RING and NESTED pixel ordering schemes.

::: jax_healpy.pixelfunc.nest2ring
::: jax_healpy.pixelfunc.ring2nest
::: jax_healpy.pixelfunc.reorder

## Map Resolution

::: jax_healpy.pixelfunc.ud_grade

## Interpolation

::: jax_healpy.pixelfunc.get_interp_weights
::: jax_healpy.pixelfunc.get_interp_val
::: jax_healpy.pixelfunc.InterpCenters

## Neighbours

::: jax_healpy.pixelfunc.get_all_neighbours

## Resolution Parameters

::: jax_healpy.pixelfunc.nside2npix
::: jax_healpy.pixelfunc.npix2nside
::: jax_healpy.pixelfunc.nside2order
::: jax_healpy.pixelfunc.order2nside
::: jax_healpy.pixelfunc.order2npix
::: jax_healpy.pixelfunc.npix2order
::: jax_healpy.pixelfunc.nside2resol
::: jax_healpy.pixelfunc.nside2pixarea
::: jax_healpy.pixelfunc.get_nside

## Validation

::: jax_healpy.pixelfunc.isnsideok
::: jax_healpy.pixelfunc.isnpixok
::: jax_healpy.pixelfunc.maptype
::: jax_healpy.pixelfunc.mask_bad

## Constants

::: jax_healpy.pixelfunc.UNSEEN
