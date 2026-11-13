![jax-healpy logo](assets/jax-healpy-logo.png#only-light){ width="600" .centered }
![jax-healpy logo](assets/jax-healpy-logo-dark.png#only-dark){ width="600" .centered }

# HEALPix with JAX

[![PyPI version](https://img.shields.io/pypi/v/jax-healpy)](https://pypi.org/project/jax-healpy/)
[![Python versions](https://img.shields.io/pypi/pyversions/jax-healpy)](https://pypi.org/project/jax-healpy/)
[![License](https://img.shields.io/pypi/l/jax-healpy)](https://github.com/CMBSciPol/jax-healpy/blob/main/LICENSE)

**jax-healpy** is a JAX-based implementation of the HEALPix (Hierarchical Equal Area isoLatitude
Pixelization) functions, running on CPUs and GPUs. It provides JAX-native versions of popular healpy functions, enabling automatic differentiation,
just-in-time compilation and seamless integration with modern scientific computing workflows.

!!! warning
    This project is in beta stage. APIs may change and some features are still under development.

## Key Features

🚀 **GPU acceleration**: XLA compilation on CPUs and GPUs, with pixel functions up to 1000 times
faster than healpy on GPU (see [Performance](#performance))

🔄 **Automatic differentiation**: forward and reverse-mode automatic differentiation

🧩 **Vectorized operations**: efficient batch processing of HEALPix operations

🔁 **HEALPix compatibility**: drop-in replacement for many healpy functions

🌐 **Spherical harmonics**: differentiable spherical harmonic transforms through ducc0 on CPU
and s2fft on GPU, which agree with healpy to $10^{-12}$

🗺️ **Clustering tools**: clustering of HEALPix maps into regions

## Quick Start

```python
import jax
import jax.numpy as jnp
import jax_healpy as hp

# Enable 64-bit precision (recommended)
jax.config.update('jax_enable_x64', True)

# Convert pixel indices to sky coordinates and back
nside = 64
pixels = jnp.arange(hp.nside2npix(nside))
theta, phi = hp.pix2ang(nside, pixels)
recovered_pixels = hp.ang2pix(nside, theta, phi)

# Spherical harmonic transforms (requires the `cpu` or the `gpu` extra dependencies)
skymap = jax.random.normal(jax.random.key(0), pixels.shape)
alm = hp.map2alm(skymap, lmax=2 * nside)
smoothed_map = hp.alm2map(alm, nside=nside, lmax=2 * nside)

# Differentiate and vectorize: derivative of the power of simulated maps with respect to
# the amplitude of their power spectrum, for 8 random realizations at once
cl = 1 / (jnp.arange(3 * nside) + 10.0) ** 2


def total_power(amplitude, key):
    cmb_map = hp.synfast(key, amplitude * cl, nside)
    return jnp.sum(hp.anafast(cmb_map))


keys = jax.random.split(jax.random.key(0), 8)
gradients = jax.vmap(jax.grad(total_power), in_axes=(None, 0))(1.0, keys)
```

The [inference](user-guide/inference.md) page puts these tools to work on the measurement of
the tensor-to-scalar ratio of the CMB, with the NUTS sampler of NumPyro.

## Performance

The pixel functions, benchmarked on 10 million points against healpy, run about 10 times
faster on CPU and up to 1000 times faster on GPU (Intel Xeon Gold 2648, NVIDIA Tesla V100;
times relative to jax-healpy on GPU in 32-bit precision):

![Benchmark of the pixel functions](benchmarks/chart-default-n10000000.png#only-light){ width="600" .centered }
![Benchmark of the pixel functions](benchmarks/chart-darkbackground-n10000000.png#only-dark){ width="600" .centered }

The spherical harmonic transforms, computed by ducc0 on CPU and by s2fft on GPU, are not part
of this benchmark.

## Documentation Contents

- **User Guide**: [Installation](user-guide/installation.md),
  [Getting started](user-guide/getting-started.md),
  [Migrating from healpy](user-guide/migrating-from-healpy.md),
  [Basic usage](user-guide/basic-usage.md),
  [Pixelization](user-guide/pixelization.md), [Polarization](user-guide/polarization.md),
  [Inference of the tensor-to-scalar ratio](user-guide/inference.md),
  [Clustering](user-guide/clustering.md),
  [Performance tips](user-guide/performance-tips.md)
- **API Reference**: [Pixel functions](api/pixelfunc.md),
  [Disc queries](api/query-disc.md), [Spherical harmonics](api/sphtfunc.md),
  [Clustering](api/clustering.md)
- [**Changelog**](changelog.md)
- [**Contributing**](contributing.md)

## Acknowledgements

[![Logos of SciPol, the European Union, the ERC and the CNRS](assets/scipol-banner.png#only-light){ width="500" }](https://scipol.in2p3.fr/)
[![Logos of SciPol, the European Union, the ERC and the CNRS](assets/scipol-banner-dark.png#only-dark){ width="500" }](https://scipol.in2p3.fr/)

jax-healpy is developed within the [SciPol](https://scipol.in2p3.fr/) project.
