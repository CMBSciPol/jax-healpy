# Installation

This page provides detailed installation instructions for jax-healpy.

## Prerequisites

jax-healpy requires Python 3.11 or later, and JAX. If JAX is not already installed,
`pip install jax-healpy` installs its CPU version. To run on a GPU or a TPU, install JAX as
described in its [installation guide](https://docs.jax.dev/en/latest/installation.html).

## Installation

Wheels are available on PyPI:

```bash
pip install jax-healpy
```

### Spherical Harmonic Transforms

The spherical harmonic transforms are computed by an additional library, which depends on the device
on which the computations run. Each one is installed by an extra dependency:

| Extra | Library | Transforms |
|---|---|---|
| `cpu` | [ducc0](https://pypi.org/project/ducc0/), in C++ | on CPU, on all the physical cores |
| `gpu` | [s2fft](https://github.com/astro-informatics/s2fft), in JAX | on GPU |

```bash
# On CPU
pip install "jax-healpy[cpu]"

# On GPU, here with NVIDIA CUDA 13 support
pip install "jax-healpy[gpu]" "jax[cuda13]"
```

The extra `gpu` does not install the GPU version of JAX: it is selected by the extra of JAX, here
`jax[cuda13]`, as described in its [installation guide](https://docs.jax.dev/en/latest/installation.html).

!!! warning "The extra `gpu` currently requires s2fft from GitHub"
    s2fft 1.5.0, its latest release, computes the transforms of polarization on HEALPix maps with errors
    of a few percent, which [s2fft#387](https://github.com/astro-informatics/s2fft/pull/387) fixes. The
    extra `gpu` requires a more recent version, which is not released yet. Meanwhile, the commands that
    request this extra fail with `No matching distribution found for s2fft>1.5.0`, unless s2fft is
    installed from GitHub in the same command:

    ```bash
    pip install "jax-healpy[gpu]" "jax[cuda13]" "s2fft @ git+https://github.com/astro-informatics/s2fft"
    ```

    s2fft is then built from its sources, which requires git and a C++ compiler.

The transforms of s2fft also run on CPU, but they are two to three orders of magnitude slower there than
those of ducc0: without ducc0, jax-healpy uses them and emits a warning. See the
[transform backends](../api/sphtfunc.md#transform-backends).

## Developer Installation

Contributions are welcome. To set up a development environment in which the tests can be run and the
documentation can be built, clone the repository and use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/CMBSciPol/jax-healpy.git
cd jax-healpy
uv sync --all-extras

# With NVIDIA CUDA 13 support
uv sync --all-extras --group cuda
```

See the [contributing guide](../contributing.md) for more details.

## Verifying Installation

Test your installation:

```python
import jax
import jax.numpy as jnp
import jax_healpy as hp

print(f'JAX version: {jax.__version__}')
print(f'JAX backend: {jax.default_backend()}')
print(f'Available devices: {jax.devices()}')

# Test basic functionality
nside = 32
pixels = jnp.arange(100)
theta, phi = hp.pix2ang(nside, pixels)
print(f'Successfully converted {len(pixels)} pixels to coordinates')
```

## Troubleshooting

### JAX Not Found

```
ModuleNotFoundError: No module named 'jax'
```

**Solution**: Install JAX according to your hardware configuration (see the [prerequisites](#prerequisites)).

### GPU Not Detected

```python
print(jax.devices())  # Should show GPU devices
```

**Solution**: Ensure that the NVIDIA driver is properly installed and that JAX was installed with GPU support,
as described in its [installation guide](https://docs.jax.dev/en/latest/installation.html).

### Out of Memory Errors

See the memory management section of the [performance tips](performance-tips.md).

### Import Errors with ducc0 or s2fft

If the spherical harmonic transforms raise an `ImportError`, install the extra dependency of the device
on which they run:

```bash
# On CPU
pip install "jax-healpy[cpu]"

# On GPU
pip install "jax-healpy[gpu]"
```

The extra `recommended` of the previous versions no longer exists: it is replaced by these two.

### No Matching Distribution Found for s2fft

```
ERROR: No matching distribution found for s2fft>1.5.0
```

**Solution**: the extra `gpu` requires a version of s2fft that is not released yet. Install s2fft from
GitHub in the same command, as described in the section on the
[spherical harmonic transforms](#spherical-harmonic-transforms).

### Getting Help

If you encounter issues:

1. Search the existing [GitHub issues](https://github.com/CMBSciPol/jax-healpy/issues)
2. Create a new issue with:
   - Your system configuration
   - Python and JAX versions
   - Complete error message
   - Minimal reproduction example
