# Getting Started

Welcome to jax-healpy! This guide will help you get up and running with JAX-based HEALPix functions.

## Basic Usage

### Import and Setup

```python
import jax
import jax.numpy as jnp
import jax_healpy as hp

# Enable 64-bit precision (recommended, see the note below)
jax.config.update('jax_enable_x64', True)
```

```{note}
jax-healpy follows JAX's precision setting and does not enable 64-bit precision on import. With the
default 32-bit precision, the results diverge from healpy even at moderate resolution (e.g. `nside = 256`).
64-bit precision should be enabled before the first array operation.
```

### HEALPix Basics

```python
# Set up a HEALPix map
nside = 64
npix = hp.nside2npix(nside)
print(f'Number of pixels: {npix}')

# Create some test data
pixels = jnp.arange(npix)
```

### Coordinate Conversions

```python
# Convert pixel indices to angular coordinates
theta, phi = hp.pix2ang(nside, pixels)

# Convert back to pixels
recovered_pixels = hp.ang2pix(nside, theta, phi)

# Verify the conversion
print(f'Conversion successful: {jnp.array_equal(pixels, recovered_pixels)}')
```

### Vector Operations

```python
# Convert pixels to 3D unit vectors, of shape (npix, 3)
vectors = hp.pix2vec(nside, pixels)

# Convert vectors back to pixels
recovered_pixels_vec = hp.vec2pix(nside, vectors[:, 0], vectors[:, 1], vectors[:, 2])

print(f'Vector conversion successful: {jnp.array_equal(pixels, recovered_pixels_vec)}')
```

### Pixel Ordering Schemes

```python
# Convert between RING and NESTED schemes
ring_pixels = jnp.arange(100)  # First 100 pixels in RING scheme
nest_pixels = hp.ring2nest(nside, ring_pixels)
back_to_ring = hp.nest2ring(nside, nest_pixels)

print(f'Scheme conversion successful: {jnp.array_equal(ring_pixels, back_to_ring)}')
```

## Working with Maps

### Map Interpolation

```python
# Create a simple test map
test_map = jnp.sin(theta) * jnp.cos(phi)

# Get the interpolation pixels and weights for arbitrary coordinates
theta_interp = jnp.array([0.5, 1.0, 1.5])
phi_interp = jnp.array([0.0, 1.0, 2.0])
neighbours, weights = hp.get_interp_weights(nside, theta_interp, phi_interp)

# Interpolate values
interpolated_values = hp.get_interp_val(test_map, theta_interp, phi_interp)
```

### Spherical Harmonics

The spherical harmonic transforms require the `recommended` extra dependencies.

```python
lmax = 2 * nside

# Forward transform: map to spherical harmonic coefficients
alm = hp.map2alm(test_map, lmax=lmax)

# Inverse transform: coefficients back to map
reconstructed_map = hp.alm2map(alm, nside=nside, lmax=lmax)

print(f'SHT round-trip error: {jnp.max(jnp.abs(test_map - reconstructed_map))}')
```
