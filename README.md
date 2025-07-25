# jax-healpy

**jax-healpy**: A JAX-based implementation of HEALPix functions for high-performance scientific computing.

This project provides a comprehensive JAX-native implementation of HEALPix (Hierarchical Equal Area isoLatitude Pixelization) functions, designed for modern scientific computing with GPU acceleration, automatic differentiation, and seamless integration with machine learning workflows.

[![Documentation Status](https://readthedocs.org/projects/jax-healpy/badge/?version=latest)](https://jax-healpy.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/jax-healpy.svg)](https://badge.fury.io/py/jax-healpy)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **⚠️ WARNING: BETA STAGE** - This project is in active development. APIs may change and some features are still under development.

## Key Features

- **🚀 GPU Acceleration**: Leverage JAX's XLA compilation for high-performance computing on CPUs and GPUs
- **🔄 Automatic Differentiation**: Full support for forward and reverse-mode automatic differentiation
- **📊 Vectorized Operations**: Efficient batch processing of HEALPix operations
- **🔧 HEALPix Compatibility**: Drop-in replacement for many healpy functions
- **🌐 Spherical Harmonics**: Integration with s2fft for spherical harmonic transforms
- **🎯 Clustering Tools**: Advanced clustering algorithms for astronomical data analysis

## Installation

### Prerequisites

First, install JAX following the [official documentation](https://jax.readthedocs.io/en/latest/installation.html) for your target architecture (CPU/GPU).

### Install jax-healpy

Install via PyPI:

```bash
pip install jax-healpy
```

For spherical harmonics functionality, install with recommended dependencies:

```bash
pip install jax-healpy[recommended]
```

### Development Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/pchanial/jax-healpy.git
cd jax-healpy
pip install -e .
```

## Quick Start

```python
import jax.numpy as jnp
import jax_healpy as hp

# Create a HEALPix map
nside = 64
npix = hp.nside2npix(nside)

# Convert pixel indices to sky coordinates
pixels = jnp.arange(npix)
theta, phi = hp.pix2ang(nside, pixels)

# Convert sky coordinates back to pixels
recovered_pixels = hp.ang2pix(nside, theta, phi, nest=False)

# Spherical harmonics transform (requires s2fft)
alm = hp.map2alm(skymap, lmax=128)
reconstructed_map = hp.alm2map(alm, nside=nside)
```

## Performance Benchmarks

Execution time measured on high-performance computing systems:

**Test System:**
- CPU: Intel(R) Xeon(R) Gold 2648 @ 2.50GHz
- GPU: NVIDIA Tesla V100-SXM2-16GB

![Performance Benchmark](/docs/benchmarks/chart-darkbackground-n10000000.png)

jax-healpy demonstrates significant performance improvements, especially for GPU-accelerated workloads and batch operations.

## Documentation

Complete documentation is available at [jax-healpy.readthedocs.io](https://jax-healpy.readthedocs.io/)

- [Getting Started Guide](https://jax-healpy.readthedocs.io/en/latest/getting_started.html)
- [API Reference](https://jax-healpy.readthedocs.io/en/latest/api/)
- [Examples and Tutorials](https://jax-healpy.readthedocs.io/en/latest/examples/)

## Development

### Setting up Development Environment

Install development dependencies:

```bash
pip install -e .[test]
```

### Running Tests

Execute the test suite:

```bash
pytest
```

### Code Quality

This project uses pre-commit hooks for code quality:

```bash
pip install pre-commit
pre-commit install
```

## High-Performance Computing

### Environment Setup

For HPC systems, load required modules:

```bash
module load python/3.10
python -m venv venv
source venv/bin/activate
pip install jax-healpy
```

### GPU Support

Ensure JAX is properly configured for your GPU architecture. See the [JAX GPU installation guide](https://jax.readthedocs.io/en/latest/installation.html#gpu) for details.

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://jax-healpy.readthedocs.io/en/latest/development/contributing.html) for details on:

- Setting up the development environment
- Code style and testing requirements
- Submitting pull requests
- Reporting issues

## Citation

If you use jax-healpy in your research, please cite:

```bibtex
@software{jax_healpy,
  author = {Chanial, Pierre and Biquard, Simon and Kabalan, Wassim},
  title = {jax-healpy: JAX-based HEALPix implementation},
  url = {https://github.com/pchanial/jax-healpy},
  year = {2024}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [JAX](https://github.com/google/jax) for high-performance computing
- Compatible with [HEALPix](https://healpix.jpl.nasa.gov/) pixelization scheme
- Integrates with [s2fft](https://github.com/astro-informatics/s2fft) for spherical harmonics
- Inspired by the original [healpy](https://github.com/healpy/healpy) package
