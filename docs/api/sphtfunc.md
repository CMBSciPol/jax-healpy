# Spherical Harmonic Functions

Spherical harmonic transforms between HEALPix maps and spherical harmonic coefficients
\(a_{\ell m}\), power spectra and beam smoothing.

!!! note
    These functions require the `recommended` extra dependencies:
    `pip install "jax-healpy[recommended]"`.

## Transforms

::: jax_healpy.sphtfunc.map2alm
::: jax_healpy.sphtfunc.alm2map
::: jax_healpy.sphtfunc.map2alm_spin
::: jax_healpy.sphtfunc.alm2map_spin

## Precomputations

::: jax_healpy.sphtfunc.precompute_temperature_harmonic_transforms
::: jax_healpy.sphtfunc.precompute_polarization_harmonic_transforms

## Power Spectra

::: jax_healpy.sphtfunc.anafast
::: jax_healpy.sphtfunc.alm2cl
::: jax_healpy.sphtfunc.synalm
::: jax_healpy.sphtfunc.synfast

## Smoothing and Beams

::: jax_healpy.sphtfunc.smoothing
::: jax_healpy.sphtfunc.smoothalm
::: jax_healpy.sphtfunc.almxfl
::: jax_healpy.sphtfunc.gauss_beam
::: jax_healpy.sphtfunc.pixwin

## Masks

::: jax_healpy.sphtfunc.bad_pixel_mask
