# What jax-healpy adds

healpy computes on NumPy arrays, on the CPU. jax-healpy computes the same quantities with JAX
functions, which JAX can compile, batch, differentiate and run on GPU, and it provides some
functionality that healpy lacks. What it does not provide, and how it departs from healpy, is
listed in [Differences from healpy](differences-from-healpy.md).

The examples of this page share this setup:

```python
from functools import partial

import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np

jax.config.update('jax_enable_x64', True)

nside = 64
lmax = 3 * nside - 1
cl = 1 / (jnp.arange(lmax + 1) + 10.0) ** 2
m = jhp.synfast(jax.random.key(0), cl, nside)
```

## Compilation

The functions of jax-healpy are compiled by XLA at their first call, for the shapes of the
arrays and the values of the static arguments, such as `nside`. A function that chains several
of them can be compiled as a whole with `jax.jit`, which fuses their operations:

```python
@partial(jax.jit, static_argnames=['nside', 'lmax'])
def smoothed_spectrum(m, nside, lmax):
    smoothed = jhp.smoothing(m, fwhm=np.radians(1), lmax=lmax)
    return jhp.anafast(smoothed, lmax=lmax)


spectrum = smoothed_spectrum(m, nside, lmax)
```

[Performance tips](performance-tips.md) shows the cost of the compilation and how to declare
the static arguments.

## Batching

`jax.vmap` maps a function over a batch dimension, for example to draw and analyze many
realizations at once:

```python
keys = jax.random.split(jax.random.key(0), 100)
maps = jax.vmap(jhp.synfast, in_axes=(0, None, None))(keys, cl, nside)  # (100, npix)
spectra = jax.vmap(jhp.anafast)(maps)  # (100, lmax + 1)
```

Some functions also accept a batch directly:

- the pixel functions broadcast over their array arguments;
- `map2alm` and `alm2map` transform a batch of maps or alms stacked along a leading axis:
  `jhp.map2alm(maps, lmax=lmax, pol=False)` for scalar maps, or an array of shape
  `(batch, 3, npix)` for polarized maps;
- [`query_disc`][jax_healpy.query_disc] takes a batch of disc centers, of shape `(B, 3)`.

## Differentiation

The spherical harmonic transforms, the power spectra, the smoothing, the interpolation and the
simulations are differentiable with `jax.grad`, in forward and reverse mode. For instance, the
gradient of a misfit in pixel space with respect to the alms:

```python
alm = jhp.synalm(jax.random.key(1), cl)


def misfit(alm):
    return jnp.sum((jhp.alm2map(alm, nside, lmax=lmax) - m) ** 2)


gradient = jax.grad(misfit)(alm)
```

The [inference of the tensor-to-scalar ratio](inference.md) puts these gradients to work: it
samples the posterior with the NUTS sampler of NumPyro, which follows the gradient of the
likelihood, and differentiates the spectrum of a simulated map with respect to $r$.

For a fixed mask of bad pixels, `map2alm` is a linear function of the map: `jax.linear_transpose`
gives its adjoint, for use as a linear operator in a solver:

```python
(adjoint,) = jax.linear_transpose(lambda m: jhp.map2alm(m, lmax=lmax, iter=0), m)(alm)
```

## CPU and GPU

The same code runs on CPU and on GPU: the computations run on the default device of JAX,
which is the GPU when JAX is installed with GPU support (see [Installation](installation.md)).

- **Pixel functions.** Benchmarked on 10 million points against healpy, they run about 10
  times faster on CPU and up to 1000 times faster on GPU (Intel Xeon Gold 2648, NVIDIA Tesla
  V100; see [Performance](../index.md#performance)).
- **Spherical harmonic transforms.** They are computed by ducc0 on CPU, on all the physical
  cores, and by s2fft on GPU (see
  [Transform backends](../api/sphtfunc.md#transform-backends)). In 64-bit precision, they agree
  with healpy to about $10^{-12}$ or better. They are not part of the benchmark against healpy.

## Additional functionality

- **Clustering.** [`find_kmeans_clusters`][jax_healpy.clustering.find_kmeans_clusters] divides
  the pixels of a mask into regions, and the other functions of
  [`jax_healpy.clustering`](../api/clustering.md) extract the pixels of a mask from a map, put
  them back into a full map and relabel the regions (see [Clustering](clustering.md)).
- **Masks.** [`bad_pixel_mask`][jax_healpy.bad_pixel_mask] flags the `UNSEEN` and non-finite
  pixels of a map, for the `bad_mask` argument of `map2alm` and `map2alm_spin`.
- **Polarization without temperature.** `map2alm`, `alm2map` and `anafast` accept the $Q, U$
  maps or the $E, B$ alms alone (see [Polarization](polarization.md)).
- **Disc sizes.** [`estimate_disc_pixel_count`][jax_healpy.estimate_disc_pixel_count] and
  [`estimate_disc_radius`][jax_healpy.estimate_disc_radius] relate the radius of a disc to its
  number of pixels, to size the output of `query_disc`.
- **Pixel geometry.** [`pix2loc`][jax_healpy.pix2loc] and [`loc2pix`][jax_healpy.loc2pix]
  convert between pixels and the cosine and sine of the colatitude, without trigonometric
  functions. `get_interp_weights(..., with_centers=True)` also returns the centers of the four
  interpolation neighbours, and `get_all_neighbours(..., get_center=True)` the pixel itself.
