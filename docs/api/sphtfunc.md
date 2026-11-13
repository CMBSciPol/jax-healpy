# Spherical Harmonic Functions

Spherical harmonic transforms between HEALPix maps and spherical harmonic coefficients
\(a_{\ell m}\), power spectra and beam smoothing.

!!! note
    These functions require the `recommended` extra dependencies:
    `pip install "jax-healpy[recommended]"`.

## Transform backends

The transforms are computed by [s2fft](https://github.com/astro-informatics/s2fft), with the
backend selected by the `method` argument of the functions:

| `method` | Fourier transforms along the rings | Runs on |
|---|---|---|
| `'jax'` (default) | in JAX, one per ring | CPU and GPU |
| `'jax_cuda'` | by the CUDA kernel of s2fft | NVIDIA GPU |

The two backends return the same results to rounding errors. `'jax_cuda'` shortens the compilation
of the transforms on GPU: it requires s2fft to be
[built from source with CUDA](https://github.com/astro-informatics/s2fft#cuda-extension-support).

The maps to transform must be real: complex maps raise a `TypeError`.

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
