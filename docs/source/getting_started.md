# Getting Started

Welcome to jax-healpy! This guide will help you get up and running with JAX-based HEALPix functions.

## What is jax-healpy?

jax-healpy is a JAX-native implementation of HEALPix (Hierarchical Equal Area isoLatitude Pixelization) functions designed for high-performance scientific computing. It provides:

- GPU acceleration through JAX's XLA compilation
- Automatic differentiation for optimization and inference
- Vectorized operations for batch processing
- Drop-in compatibility with many healpy functions

## Prerequisites

Before installing jax-healpy, you need to install JAX for your target architecture:

### CPU Installation

```bash
pip install jax
```

### GPU Installation (NVIDIA)

For CUDA support:

```bash
pip install jax[cuda12]  # For CUDA 12
# or
pip install jax[cuda11]  # For CUDA 11
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for detailed instructions.

## Installation

### From PyPI

```bash
pip install jax-healpy
```

### With Optional Dependencies

For spherical harmonics functionality:

```bash
pip install jax-healpy[recommended]
```

### Development Installation

```bash
git clone https://github.com/pchanial/jax-healpy.git
cd jax-healpy
pip install -e .[test]
```

## Basic Usage

### Import and Setup

```python
import jax
import jax.numpy as jnp
import jax_healpy as hp

# Enable 64-bit precision (recommended for astronomical calculations)
jax.config.update("jax_enable_x64", True)
```

### HEALPix Basics

```python
# Set up a HEALPix map
nside = 64
npix = hp.nside2npix(nside)
print(f"Number of pixels: {npix}")

# Create some test data
pixels = jnp.arange(npix)
```

### Coordinate Conversions

```python
# Convert pixel indices to angular coordinates
theta, phi = hp.pix2ang(nside, pixels)

# Convert back to pixels
recovered_pixels = hp.ang2pix(nside, theta, phi, nest=False)

# Verify the conversion
print(f"Conversion successful: {jnp.allclose(pixels, recovered_pixels)}")
```

### Vector Operations

```python
# Convert pixels to 3D unit vectors
vectors = hp.pix2vec(nside, pixels)

# Convert vectors back to pixels
recovered_pixels_vec = hp.vec2pix(nside, vectors[0], vectors[1], vectors[2])

print(f"Vector conversion successful: {jnp.allclose(pixels, recovered_pixels_vec)}")
```

### Coordinate System Conversions

```python
# Convert between RING and NESTED schemes
ring_pixels = jnp.arange(100)  # First 100 pixels in RING scheme
nest_pixels = hp.ring2nest(nside, ring_pixels)
back_to_ring = hp.nest2ring(nside, nest_pixels)

print(f"Scheme conversion successful: {jnp.allclose(ring_pixels, back_to_ring)}")
```

## Working with Maps

### Map Interpolation

```python
# Create a simple test map
test_map = jnp.sin(theta) * jnp.cos(phi)

# Get interpolation weights for arbitrary coordinates
theta_interp = jnp.array([0.5, 1.0, 1.5])
phi_interp = jnp.array([0.0, 1.0, 2.0])

weights, pixels = hp.get_interp_weights(nside, theta_interp, phi_interp)

# Interpolate values
interpolated_values = hp.get_interp_val(test_map, theta_interp, phi_interp, nside)
```

### Spherical Harmonics (requires s2fft)

```python
# This requires the 'recommended' dependencies
try:
    # Forward transform: map to spherical harmonic coefficients
    alm = hp.map2alm(test_map, lmax=64)

    # Inverse transform: coefficients back to map
    reconstructed_map = hp.alm2map(alm, nside=nside)

    print(f"SHT round-trip error: {jnp.mean(jnp.abs(test_map - reconstructed_map))}")
except ImportError:
    print("s2fft not installed - spherical harmonics not available")
```

## Performance Benefits

### GPU Acceleration

```python
# Operations automatically run on GPU if available
print(f"JAX backend: {jax.default_backend()}")

# Compile functions for maximum performance
@jax.jit
def fast_coordinate_conversion(nside, pixels):
    theta, phi = hp.pix2ang(nside, pixels)
    return hp.ang2pix(nside, theta, phi)

# Time the compiled function
import time
pixels = jnp.arange(hp.nside2npix(128))

start = time.time()
result = fast_coordinate_conversion(128, pixels)
result.block_until_ready()  # Wait for GPU computation
end = time.time()

print(f"Conversion time: {end - start:.4f} seconds")
```

### Vectorized Operations

```python
# Process multiple maps simultaneously
batch_size = 10
nside = 32
npix = hp.nside2npix(nside)

# Create batch of test maps
batch_maps = jnp.random.normal(0, 1, (batch_size, npix))

# Vectorized pixel-to-angle conversion for all maps
pixels = jnp.arange(npix)
theta_batch, phi_batch = jax.vmap(lambda m: hp.pix2ang(nside, pixels))(batch_maps)

print(f"Processed {batch_size} maps with {npix} pixels each")
```

## Next Steps

Now that you have jax-healpy working, explore:

- [User Guide](user_guide/pixelization.md) for detailed function documentation
- [Examples](examples/basic_usage.md) for more complex use cases
- [API Reference](api/pixelfunc.md) for complete function listings
- [Benchmarks](examples/benchmarks.md) to see performance comparisons

## Common Issues

### Memory Usage

For large maps, consider using 32-bit precision:

```python
jax.config.update("jax_enable_x64", False)
```

### GPU Memory

For very large operations, you may need to batch your computations:

```python
def process_in_batches(data, batch_size=10000):
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        result = your_function(batch)
        results.append(result)
    return jnp.concatenate(results)
```
