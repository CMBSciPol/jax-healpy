# Installation

This page provides detailed installation instructions for jax-healpy.

## Prerequisites

### Python Version

jax-healpy requires Python 3.11 or later.

### JAX Installation

jax-healpy depends on JAX, whose installation depends on your target hardware. If JAX is not already
installed, `pip install jax-healpy` installs the CPU version. To run on accelerators, install JAX with
the appropriate extra, as described in the [official JAX documentation](https://docs.jax.dev/en/latest/installation.html).
For example:

```bash
# NVIDIA GPU, CUDA 13
pip install "jax[cuda13]"

# NVIDIA GPU, CUDA 12
pip install "jax[cuda12]"

# Google Cloud TPU
pip install "jax[tpu]"
```

## Installation

Wheels are available on PyPI:

```bash
pip install jax-healpy
```

JAX and jax-healpy can be installed in a single command, for instance with NVIDIA CUDA 13 support:

```bash
pip install jax-healpy "jax[cuda13]"
```

For the spherical harmonics functionalities, install the extra dependency `recommended`:

```bash
pip install "jax-healpy[recommended]"
```

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

See the [contributing guide](../development/contributing.md) for more details.

## High-Performance Computing Systems

### General HPC Setup

Many HPC systems require module loading:

```bash
module load python/3.12
module load cuda/13.0  # If using GPU
python -m venv .venv
source .venv/bin/activate
pip install "jax-healpy[recommended]" "jax[cuda13]"
```

### SLURM Job Scripts

Example SLURM script for GPU jobs:

```bash
#!/bin/bash
#SBATCH --job-name=jax-healpy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

module load python/3.12
module load cuda/13.0
source .venv/bin/activate

python your_script.py
```

### Jean Zay (France)

```bash
module load python/3.12.7
module load cuda/12.8.0
python -m venv .venv
source .venv/bin/activate
pip install "jax-healpy[recommended]" "jax[cuda12]"
```

### NERSC Perlmutter

```bash
module load python
module load cuda
python -m venv .venv
source .venv/bin/activate
pip install "jax-healpy[recommended]" "jax[cuda12]"
```

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

**Solution**: Install JAX according to your hardware configuration (see above).

### GPU Not Detected

```python
print(jax.devices())  # Should show GPU devices
```

**Solution**: Ensure that the NVIDIA driver is properly installed and that JAX was installed with GPU support,
for instance with `pip install "jax[cuda13]"`.

### Out of Memory Errors

See the memory management section of the [performance tips](performance_tips.md).

### Import Errors with s2fft

If the spherical harmonics functions fail, install the `recommended` extra dependency:

```bash
pip install "jax-healpy[recommended]"
```

### Getting Help

If you encounter issues:

1. Search the existing [GitHub issues](https://github.com/CMBSciPol/jax-healpy/issues)
2. Create a new issue with:
   - Your system configuration
   - Python and JAX versions
   - Complete error message
   - Minimal reproduction example
