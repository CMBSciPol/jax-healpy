![jax-healpy logo](assets/jax-healpy-logo.png#only-light){ width="600" .centered }
![jax-healpy logo](assets/jax-healpy-logo-dark.png#only-dark){ width="600" .centered }

# HEALPix with JAX

[![PyPI version](https://img.shields.io/pypi/v/jax-healpy)](https://pypi.org/project/jax-healpy/)
[![Python versions](https://img.shields.io/pypi/pyversions/jax-healpy)](https://pypi.org/project/jax-healpy/)
[![License](https://img.shields.io/pypi/l/jax-healpy)](https://github.com/CMBSciPol/jax-healpy/blob/main/LICENSE)

**jax-healpy** is a JAX-based implementation of the HEALPix (Hierarchical Equal Area isoLatitude
Pixelization) functions, designed for high-performance scientific computing on CPUs and GPUs. It provides JAX-native versions of popular healpy functions, enabling automatic differentiation,
just-in-time compilation and seamless integration with modern scientific computing workflows.

!!! warning
    This project is in beta stage. APIs may change and some features are still under development.

## Key Features

🚀 **GPU acceleration**: XLA compilation for high-performance computing on CPUs and GPUs

🔄 **Automatic differentiation**: forward and reverse-mode automatic differentiation

🧩 **Vectorized operations**: efficient batch processing of HEALPix operations

🔁 **HEALPix compatibility**: drop-in replacement for many healpy functions

🌐 **Spherical harmonics**: spherical harmonic transforms through ducc0 on CPU and s2fft on GPU

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
```

## Documentation Contents

- **User Guide**: [Installation](user-guide/installation.md),
  [Getting started](user-guide/getting-started.md), [Basic usage](user-guide/basic-usage.md),
  [Pixelization](user-guide/pixelization.md), [Polarization](user-guide/polarization.md),
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
