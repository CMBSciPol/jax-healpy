# Spherical Harmonic Functions

Spherical harmonic transforms between HEALPix maps and spherical harmonic coefficients
\(a_{\ell m}\), power spectra and beam smoothing.

!!! note
    These functions require the `recommended` extra dependencies:
    `pip install "jax-healpy[recommended]"`.

## Transform backends

The transforms are computed by one of two libraries, selected by the `method` argument of the
functions:

| `method` | Library | Runs on |
|---|---|---|
| `'ducc0'` | [ducc0](https://pypi.org/project/ducc0/), in C++ | CPU, on all the cores |
| `'jax'` | [s2fft](https://github.com/astro-informatics/s2fft), in JAX | CPU and GPU |
| `'jax_cuda'` | s2fft, with its CUDA kernel for the Fourier transforms along the rings | NVIDIA GPU |

With any of them, the functions can be compiled by `jax.jit`, batched by `jax.vmap` and
differentiated, and they return the same results to rounding errors.

By default, `'ducc0'` is used when the computations run on CPU and ducc0 is installed, and `'jax'`
otherwise. On CPU, the transforms of ducc0 are two to three orders of magnitude faster than the
JAX ones, which run on a single core, and they are compiled in a fraction of a second.

ducc0 uses all the cores that are available to the process, unless the environment variable
`DUCC0_NUM_THREADS` or, if it is not set, `OMP_NUM_THREADS` gives a lower number of threads.

`'jax_cuda'` shortens the compilation of the transforms on GPU. It requires s2fft to be
[built from source with CUDA](https://github.com/astro-informatics/s2fft#cuda-extension-support).

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
