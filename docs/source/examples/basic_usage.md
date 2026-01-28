# Basic Usage Examples

This page provides practical examples of using `jax-healpy` for common tasks. Whether you are new to HEALPix or just transitioning to JAX, these examples will help you get started.

## Getting Started

First, let's import the necessary libraries. We recommend enabling 64-bit precision for astronomical calculations to avoid numerical issues.

```python
import jax
import jax.numpy as jnp
import jax_healpy as hp

# Enable 64-bit precision (recommended for better accuracy with coordinates)
jax.config.update("jax_enable_x64", True)

# Check your JAX backend (CPU or GPU)
print(f"Running on: {jax.default_backend()}")
```

## Basic HEALPix Operations

### Understanding Resolution (NSIDE)

The resolution of a HEALPix map is defined by the `nside` parameter, which must be a power of 2 (e.g., 1, 2, 4, 8, ...).

```python
# Choose a resolution
nside = 32

# Calculate properties
npix = hp.nside2npix(nside)
resolution = hp.nside2resol(nside, arcmin=True)

print(f"NSIDE: {nside}")
print(f"Total pixels: {npix}")
print(f"Resolution: {resolution:.2f} arcminutes")
```

### Coordinate Conversions

You can easily convert between pixel indices and angular coordinates (theta, phi).

```python
# specific pixels we want to look at
pixels = jnp.array([0, 100, 1000])

# Convert pixels -> (theta, phi)
theta, phi = hp.pix2ang(nside, pixels)

# Convert (theta, phi) -> pixels
pixels_recovered = hp.ang2pix(nside, theta, phi)

print("Pixel Conversions:")
for p, t, f in zip(pixels, theta, phi):
    print(f"  Pixel {p} is at theta={t:.2f}, phi={f:.2f} rad")
```

### Pixel Ordering Schemes

HEALPix has two ordering schemes:
- **RING**: Pixels are ordered in rings of constant latitude. (Default)
- **NESTED**: Pixels are ordered hierarchically.

```python
# Convert RING index to NESTED index
ring_idx = jnp.array([10, 20, 30])
nest_idx = hp.ring2nest(nside, ring_idx)

print(f"RING indices: {ring_idx}")
print(f"NEST indices: {nest_idx}")
```

## Creating Maps

A HEALPix map is simply a 1D array of length `npix` (or `12 * nside**2`).

```python
# Create a map where pixel value depends on its latitude (theta)
all_pixels = jnp.arange(npix)
theta, phi = hp.pix2ang(nside, all_pixels)

# Create a dipole pattern
dipole_map = jnp.cos(theta)

print(f"Created map with {len(dipole_map)} pixels")
print(f"Map min: {jnp.min(dipole_map):.3f}, max: {jnp.max(dipole_map):.3f}")
```

## Interpolation

Sometimes you need to find the map value at an arbitrary location, not just at the pixel centers.

```python
# Define some arbitrary points on the sphere
theta_points = jnp.array([1.0, 1.5, 2.0]) # Radians
phi_points = jnp.array([0.5, 1.0, 1.5])   # Radians

# Interpolate the dipole map at these points
values = hp.get_interp_val(dipole_map, theta_points, phi_points, nside)

print(f"Interpolated values: {values}")
```

## Spherical Harmonics

If you have `s2fft` installed, you can perform spherical harmonic transforms.

```python
try:
    # 1. Map -> Alms (Analysis)
    lmax = 3 * nside - 1
    alm = hp.map2alm(dipole_map, lmax=lmax)

    # 2. Alms -> Map (Synthesis)
    reconstructed_map = hp.alm2map(alm, nside=nside)

    print("Spherical harmonic transform successful!")

except ImportError:
    print("Please install 's2fft' to use spherical harmonic functions.")
    # pip install jax-healpy[recommended]
```

## Finding Pixels in a Region (Query Disc)

You can find all pixels within a certain radius of a direction. This is useful for analyzing specific patches of the sky.

```python
# Define a center point (e.g., pointing direction)
center_vec = hp.ang2vec(theta=jnp.pi/2, phi=0.0) # On the equator

# Define a radius (e.g., 10 degrees)
radius = jnp.radians(10.0)

# Find all pixels within this disc
disc_pixels = hp.query_disc(nside, center_vec, radius)

print(f"Found {len(disc_pixels)} pixels within 10 degrees of the equator point.")
```

## Leveraging JAX Features (JIT & Vmap)

Since `jax-healpy` is written in JAX, you can use `jax.jit` to compile functions for speed and `jax.vmap` to vectorize operations.

### Faster Execution with JIT

```python
import time

@jax.jit
def fast_conversion(theta, phi):
    return hp.ang2pix(nside, theta, phi)

# Run once to compile
dummy_theta, dummy_phi = jnp.zeros(1), jnp.zeros(1)
fast_conversion(dummy_theta, dummy_phi)

# Benchmark
start = time.time()
pixels = fast_conversion(theta, phi) # Processing entire map
pixels.block_until_ready()
print(f"Conversion time: {time.time() - start:.5f} seconds")
```

### Batch Processing with Vmap

If you have multiple maps (e.g., simulations), you can process them all at once.

```python
# Create a batch of 10 random maps
batch_maps = jax.random.normal(jax.random.PRNGKey(42), (10, npix))

# Define a function to process one map
def process_map(m):
    return jnp.mean(m)

# Vectorize it to handle the batch
batch_means = jax.vmap(process_map)(batch_maps)

print(f"Mean values of 10 maps: {batch_means}")
```

---

**Next Steps:**
- Check out [Clustering Examples](clustering.md) for advanced mask manipulation.
