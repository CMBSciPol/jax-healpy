# Installation

This page provides detailed installation instructions for jax-healpy.

## Prerequisites

### JAX Installation

jax-healpy requires JAX to be installed first. The installation method depends on your target hardware:

#### CPU Only

```bash
pip install jax
```

#### GPU Support (NVIDIA)

For CUDA 12:
```bash
pip install jax[cuda12]
```

For CUDA 11:
```bash
pip install jax[cuda11]
```

#### GPU Support (AMD)

```bash
pip install jax[rocm]
```

#### TPU Support

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

For detailed JAX installation instructions, see the [official JAX documentation](https://jax.readthedocs.io/en/latest/installation.html).

### Python Version

jax-healpy requires Python 3.8 or later.

## Installing jax-healpy

### From PyPI (Recommended)

```bash
pip install jax-healpy
```

### With Optional Dependencies

For spherical harmonics functionality (requires s2fft):

```bash
pip install jax-healpy[recommended]
```

For development and testing:

```bash
pip install jax-healpy[test]
```

### From Source

#### Latest Release

```bash
pip install git+https://github.com/pchanial/jax-healpy.git
```

#### Development Version

```bash
git clone https://github.com/pchanial/jax-healpy.git
cd jax-healpy
pip install -e .
```

#### Development with All Dependencies

```bash
git clone https://github.com/pchanial/jax-healpy.git
cd jax-healpy
pip install -e .[test,recommended]
```

## Virtual Environment Setup

We recommend using a virtual environment to avoid dependency conflicts:

### Using venv

```bash
python -m venv jax-healpy-env
source jax-healpy-env/bin/activate  # On Windows: jax-healpy-env\Scripts\activate
pip install jax-healpy
```

### Using conda

```bash
conda create -n jax-healpy python=3.10
conda activate jax-healpy
pip install jax-healpy
```

## High-Performance Computing Systems

### General HPC Setup

Many HPC systems require module loading:

```bash
module load python/3.10
module load cuda/12.0  # If using GPU
python -m venv venv
source venv/bin/activate
pip install jax[cuda12] jax-healpy
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

module load python/3.10
module load cuda/12.0
source venv/bin/activate

python your_script.py
```

### Common HPC Systems

#### Jean Zay (France)

```bash
module load python/3.10.4
module load cuda/11.2
python -m venv venv
source venv/bin/activate
pip install jax[cuda11] jax-healpy
```

#### NERSC Perlmutter

```bash
module load python
module load cuda
python -m venv venv
source venv/bin/activate
pip install jax[cuda12] jax-healpy
```

## Verifying Installation

Test your installation:

```python
import jax
import jax.numpy as jnp
import jax_healpy as hp

print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")

# Test basic functionality
nside = 32
npix = hp.nside2npix(nside)
pixels = jnp.arange(100)
theta, phi = hp.pix2ang(nside, pixels)
print(f"Successfully converted {len(pixels)} pixels to coordinates")
```

## Troubleshooting

### Common Issues

#### JAX Not Found

```
ImportError: No module named 'jax'
```

**Solution**: Install JAX first according to your hardware configuration.

#### GPU Not Detected

```python
print(jax.devices())  # Should show GPU devices
```

**Solution**: Ensure CUDA/ROCm is properly installed and JAX was installed with GPU support.

#### Out of Memory Errors

For large computations, consider:

```python
# Use 32-bit precision
jax.config.update("jax_enable_x64", False)

# Process in smaller batches
def batch_process(data, batch_size=1000):
    # Your processing logic here
    pass
```

#### Import Errors with s2fft

If spherical harmonics functions fail:

```bash
pip install s2fft
```

### Getting Help

If you encounter issues:

1. Check the [troubleshooting section](../development/contributing.html#troubleshooting)
2. Search existing [GitHub issues](https://github.com/pchanial/jax-healpy/issues)
3. Create a new issue with:
   - Your system configuration
   - Python and JAX versions
   - Complete error message
   - Minimal reproduction example

## Performance Tips

### Compilation

Functions are compiled on first use. For best performance:

```python
@jax.jit
def my_healpix_function(nside, data):
    # Your code here
    return result

# First call compiles the function
result = my_healpix_function(64, test_data)

# Subsequent calls are fast
result = my_healpix_function(64, other_data)
```

### Memory Management

For memory-intensive operations:

```python
# Clear JAX compilation cache if needed
jax.clear_caches()

# Use smaller data types when possible
data = data.astype(jnp.float32)
```

### Batch Processing

Process multiple maps efficiently:

```python
# Vectorize over batch dimension
batch_function = jax.vmap(hp.pix2ang, in_axes=(None, 0))
batch_results = batch_function(nside, batch_pixels)
```
