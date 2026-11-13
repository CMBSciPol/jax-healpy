# Spherical Harmonic Functions

Spherical harmonic transforms between HEALPix maps and spherical harmonic coefficients
\(a_{\ell m}\), power spectra and beam smoothing.

!!! note
    The transforms require a library that is installed by an extra dependency: ducc0 on CPU
    (`pip install "jax-healpy[cpu]"`) or s2fft on GPU (`pip install "jax-healpy[gpu]"`). See the
    [installation](../user-guide/installation.md#spherical-harmonic-transforms).

## Transform backends

The transforms are computed by one of two libraries, selected by the `method` argument of the
functions:

| `method` | Library | Extra | Runs on |
|---|---|---|---|
| `'ducc0'` | [ducc0](https://pypi.org/project/ducc0/), in C++ | `cpu` | CPU, on all the physical cores |
| `'jax'` | [s2fft](https://github.com/astro-informatics/s2fft), in JAX | `gpu` | CPU and GPU |
| `'jax_cuda'` | s2fft, with its CUDA kernel for the Fourier transforms along the rings | `gpu` | NVIDIA GPU |

With any of them, the functions can be compiled by `jax.jit`, batched by `jax.vmap` and
differentiated, and they return the same results to rounding errors. The transforms of s2fft
require `lmax >= 2 * nside - 1` and raise a `NotImplementedError` below, whereas those of ducc0,
like those of healpy, accept any `lmax`.

By default, `'ducc0'` is used when the computations run on CPU and ducc0 is installed, and `'jax'`
otherwise. The device is the default one of JAX, which `jax.default_device` can change: on a
machine with a GPU, the transforms computed within `with jax.default_device(jax.devices('cpu')[0])`
use ducc0. A ducc0 that is installed but cannot be imported is reported by a warning and not used. On CPU, the transforms of ducc0 are two to three orders of magnitude faster than the
JAX ones, which run on a single core, and they are compiled in a fraction of a second: when the
transforms of s2fft are used on CPU because ducc0 is missing, a warning recommends to install it.
It is not emitted when `method='jax'` is requested.

The functions that do not transform anything, such as `alm2cl`, `almxfl`, `synalm` or `gauss_beam`,
need none of these libraries.

ducc0 uses one thread per physical core available to the process: the hardware threads of a core
(simultaneous multithreading, or hyper-threading) compete for its resources and make the transforms
up to twice slower. On Linux, the physical cores are read from the CPU topology; elsewhere, ducc0
uses all the logical CPUs. The environment variable `DUCC0_NUM_THREADS` or, if it is not set,
`OMP_NUM_THREADS` sets the number of threads instead.

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
