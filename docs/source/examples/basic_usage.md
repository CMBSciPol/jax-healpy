# Basic Usage Examples

This page provides practical examples of using jax-healpy for common tasks.

## Getting Started

```python
import jax
import jax.numpy as jnp
import jax_healpy as hp

# Enable 64-bit precision for better accuracy
jax.config.update("jax_enable_x64", True)

# Check your JAX backend
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
```

## Basic HEALPix Operations

### Working with Resolution Parameters

```python
# Choose resolution
nside = 64
print(f"NSIDE: {nside}")
print(f"Number of pixels: {hp.nside2npix(nside)}")
print(f"Pixel resolution: {hp.nside2resol(nside, arcmin=True):.2f} arcmin")
print(f"Pixel area: {hp.nside2pixarea(nside) * (180/jnp.pi)**2:.4f} deg²")
```

### Coordinate Conversions

```python
# Convert between pixel indices and coordinates
nside = 32
pixels = jnp.array([0, 100, 1000, 5000])

# Pixels to angular coordinates
theta, phi = hp.pix2ang(nside, pixels)
print("Pixel to angle conversion:")
for i, (p, t, f) in enumerate(zip(pixels, theta, phi)):
    print(f"  Pixel {p}: θ={jnp.degrees(t):.2f}°, φ={jnp.degrees(f):.2f}°")

# Convert back to pixels
pixels_back = hp.ang2pix(nside, theta, phi)
print(f"Round-trip accuracy: {jnp.allclose(pixels, pixels_back)}")

# Work with unit vectors
vectors = hp.pix2vec(nside, pixels)
print(f"Unit vectors shape: {vectors.shape}")  # (3, 4)

# Verify unit length
lengths = jnp.linalg.norm(vectors, axis=0)
print(f"Vector lengths: {lengths}")  # Should all be 1.0
```

### Scheme Conversions

```python
# Convert between RING and NESTED orderings
nside = 16
ring_pixels = jnp.arange(20)  # First 20 pixels in RING scheme

# Convert to NESTED
nest_pixels = hp.ring2nest(nside, ring_pixels)
print(f"RING pixels: {ring_pixels}")
print(f"NEST pixels: {nest_pixels}")

# Convert back
ring_back = hp.nest2ring(nside, nest_pixels)
print(f"Round-trip successful: {jnp.array_equal(ring_pixels, ring_back)}")

# Reorder entire maps
npix = hp.nside2npix(nside)
ring_map = jnp.arange(npix, dtype=float)  # Map with pixel indices as values

nest_map = hp.reorder(ring_map, r2n=True)   # RING → NESTED
ring_restored = hp.reorder(nest_map, n2r=True)  # NESTED → RING

print(f"Map reordering successful: {jnp.allclose(ring_map, ring_restored)}")
```

## Creating and Manipulating Maps

### Generating Test Maps

```python
nside = 64
npix = hp.nside2npix(nside)

# Get coordinates for all pixels
all_pixels = jnp.arange(npix)
theta, phi = hp.pix2ang(nside, all_pixels)

# Create various test maps
gaussian_map = jnp.exp(-((theta - jnp.pi/2)**2 + (phi - jnp.pi)**2) / 0.1)
dipole_map = jnp.cos(theta)
spiral_map = jnp.sin(3*phi) * jnp.sin(theta)
random_map = jax.random.normal(jax.random.PRNGKey(42), (npix,))

print(f"Created maps with {npix} pixels")
print(f"Gaussian map range: [{jnp.min(gaussian_map):.3f}, {jnp.max(gaussian_map):.3f}]")
print(f"Dipole map range: [{jnp.min(dipole_map):.3f}, {jnp.max(dipole_map):.3f}]")
```

### Map Statistics

```python
def map_statistics(map_data, name="Map"):
    """Calculate basic statistics for a HEALPix map."""
    mean = jnp.mean(map_data)
    std = jnp.std(map_data)
    min_val, max_val = jnp.min(map_data), jnp.max(map_data)

    print(f"{name} statistics:")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std:  {std:.6f}")
    print(f"  Min:  {min_val:.6f}")
    print(f"  Max:  {max_val:.6f}")
    print(f"  RMS:  {jnp.sqrt(jnp.mean(map_data**2)):.6f}")

# Analyze our test maps
map_statistics(gaussian_map, "Gaussian")
map_statistics(dipole_map, "Dipole")
map_statistics(random_map, "Random")
```

## Interpolation

### Basic Interpolation

```python
# Create a smooth test map
nside = 32
npix = hp.nside2npix(nside)
pixels = jnp.arange(npix)
theta_pix, phi_pix = hp.pix2ang(nside, pixels)

# Smooth function on the sphere
test_map = jnp.sin(2*theta_pix) * jnp.cos(3*phi_pix)

# Interpolate at arbitrary points
theta_interp = jnp.array([0.5, 1.0, 1.5, 2.0])
phi_interp = jnp.array([0.0, 1.0, 2.0, 3.0])

# Get interpolated values
interp_values = hp.get_interp_val(test_map, theta_interp, phi_interp, nside)

print("Interpolation example:")
for i, (t, p, v) in enumerate(zip(theta_interp, phi_interp, interp_values)):
    print(f"  Point {i}: θ={t:.2f}, φ={p:.2f} → value={v:.4f}")

# Get interpolation weights (for understanding the process)
weights, pixel_indices = hp.get_interp_weights(nside, theta_interp, phi_interp)
print(f"Interpolation weights shape: {weights.shape}")  # (4, 4)
print(f"Pixel indices shape: {pixel_indices.shape}")    # (4, 4)
```

### High-Resolution Interpolation

```python
# Create high-resolution grid for smooth interpolation
nside_highres = 128
n_interp = 50

# Create interpolation grid
theta_grid = jnp.linspace(0.1, jnp.pi-0.1, n_interp)
phi_grid = jnp.linspace(0, 2*jnp.pi, n_interp)
theta_mesh, phi_mesh = jnp.meshgrid(theta_grid, phi_grid)

# Flatten for interpolation
theta_flat = theta_mesh.flatten()
phi_flat = phi_mesh.flatten()

# Interpolate test map onto high-resolution grid
interp_map_flat = hp.get_interp_val(test_map, theta_flat, phi_flat, nside)
interp_map_grid = interp_map_flat.reshape(n_interp, n_interp)

print(f"Interpolated {len(theta_flat)} points")
print(f"Interpolated map shape: {interp_map_grid.shape}")
```

## Spherical Harmonics

### Basic Transforms

```python
# Note: Requires s2fft package
try:
    # Create test map with known spherical harmonic content
    nside = 64
    npix = hp.nside2npix(nside)
    pixels = jnp.arange(npix)
    theta, phi = hp.pix2ang(nside, pixels)

    # Dipole (l=1, m=0) and quadrupole (l=2, m=0) components
    test_map = jnp.cos(theta) + 0.5 * (3*jnp.cos(theta)**2 - 1)

    # Forward transform: map → alm
    lmax = 64
    alm = hp.map2alm(test_map, lmax=lmax)
    print(f"Generated {len(alm)} spherical harmonic coefficients")

    # Inverse transform: alm → map
    reconstructed_map = hp.alm2map(alm, nside=nside)

    # Check reconstruction quality
    diff = test_map - reconstructed_map
    rms_error = jnp.sqrt(jnp.mean(diff**2))
    max_error = jnp.max(jnp.abs(diff))

    print(f"Reconstruction quality:")
    print(f"  RMS error: {rms_error:.2e}")
    print(f"  Max error: {max_error:.2e}")
    print(f"  Relative RMS: {rms_error/jnp.std(test_map):.2e}")

except ImportError:
    print("s2fft not available - skipping spherical harmonics examples")
    print("Install with: pip install jax-healpy[recommended]")
```

### Power Spectrum Analysis

```python
try:
    # Create random map with known power spectrum
    nside = 32
    npix = hp.nside2npix(nside)

    # Generate random map
    random_map = jax.random.normal(jax.random.PRNGKey(123), (npix,))

    # Transform to harmonic space
    lmax = 64
    alm = hp.map2alm(random_map, lmax=lmax)

    # Calculate power spectrum (simplified)
    # Note: This is a basic example - proper power spectrum calculation
    # requires more sophisticated indexing and normalization
    power_approx = jnp.abs(alm)**2

    print(f"Power spectrum statistics:")
    print(f"  Mean power: {jnp.mean(power_approx):.2e}")
    print(f"  Max power: {jnp.max(power_approx):.2e}")

except ImportError:
    print("Spherical harmonics require s2fft package")
```

## Query Disc Operations

### Point Source Analysis

```python
# Simulate point sources on the sky
nside = 64
npix = hp.nside2npix(nside)

# Create background map
background = jax.random.normal(jax.random.PRNGKey(1), (npix,)) * 0.1

# Add point sources
source_positions = [
    (jnp.pi/4, 0.0),      # 45° from pole, 0° azimuth
    (jnp.pi/2, jnp.pi),   # Equator, 180° azimuth
    (3*jnp.pi/4, jnp.pi/2) # 135° from pole, 90° azimuth
]

source_map = background.copy()
for theta_s, phi_s in source_positions:
    # Find pixel for source
    source_pixel = hp.ang2pix(nside, theta_s, phi_s)
    source_map = source_map.at[source_pixel].add(10.0)  # Strong source

print(f"Added {len(source_positions)} point sources")

# Query discs around sources
aperture_radius = jnp.radians(5.0)  # 5-degree radius

for i, (theta_s, phi_s) in enumerate(source_positions):
    # Convert to unit vector
    source_vec = hp.ang2vec(theta_s, phi_s)

    # Find pixels in aperture
    aperture_pixels = hp.query_disc(nside, source_vec, aperture_radius)

    # Calculate aperture photometry
    aperture_flux = jnp.sum(source_map[aperture_pixels])
    n_pixels = len(aperture_pixels)

    print(f"Source {i+1}:")
    print(f"  Position: θ={jnp.degrees(theta_s):.1f}°, φ={jnp.degrees(phi_s):.1f}°")
    print(f"  Aperture pixels: {n_pixels}")
    print(f"  Total flux: {aperture_flux:.2f}")
    print(f"  Mean flux per pixel: {aperture_flux/n_pixels:.3f}")
```

### Sky Coverage Analysis

```python
# Analyze sky coverage for a survey
nside = 32
npix = hp.nside2npix(nside)

# Define survey pointing strategy (simplified)
n_pointings = 100
pointing_theta = jax.random.uniform(
    jax.random.PRNGKey(10), (n_pointings,), minval=0, maxval=jnp.pi
)
pointing_phi = jax.random.uniform(
    jax.random.PRNGKey(11), (n_pointings,), minval=0, maxval=2*jnp.pi
)

# Field of view radius
fov_radius = jnp.radians(2.0)  # 2 degrees

# Create coverage map
coverage_map = jnp.zeros(npix)

for theta_p, phi_p in zip(pointing_theta, pointing_phi):
    # Convert to unit vector
    pointing_vec = hp.ang2vec(theta_p, phi_p)

    # Find pixels in field of view
    fov_pixels = hp.query_disc(nside, pointing_vec, fov_radius)

    # Increment coverage
    coverage_map = coverage_map.at[fov_pixels].add(1)

# Analyze coverage
total_area_sr = 4 * jnp.pi  # Full sky in steradians
pixel_area_sr = hp.nside2pixarea(nside)
covered_pixels = jnp.sum(coverage_map > 0)
coverage_fraction = covered_pixels * pixel_area_sr / total_area_sr

print(f"Survey coverage analysis:")
print(f"  Total pointings: {n_pointings}")
print(f"  Field of view: {jnp.degrees(fov_radius):.1f}°")
print(f"  Covered pixels: {covered_pixels}/{npix}")
print(f"  Sky coverage: {coverage_fraction:.1%}")
print(f"  Mean coverage depth: {jnp.mean(coverage_map[coverage_map > 0]):.1f}")
print(f"  Max coverage depth: {int(jnp.max(coverage_map))}")
```

## Performance Optimization

### JIT Compilation

```python
# Compile functions for better performance
@jax.jit
def optimized_coordinate_conversion(nside, n_coords):
    """Fast coordinate conversion with JIT compilation."""
    # Generate random coordinates
    key1, key2 = jax.random.split(jax.random.PRNGKey(42))
    theta = jax.random.uniform(key1, (n_coords,), minval=0, maxval=jnp.pi)
    phi = jax.random.uniform(key2, (n_coords,), minval=0, maxval=2*jnp.pi)

    # Convert to pixels and back
    pixels = hp.ang2pix(nside, theta, phi)
    theta_back, phi_back = hp.pix2ang(nside, pixels)

    # Calculate accuracy
    theta_error = jnp.abs(theta - theta_back)
    phi_error = jnp.abs(phi - phi_back)

    return jnp.max(theta_error), jnp.max(phi_error)

# Benchmark compilation benefit
import time

nside, n_coords = 64, 100000

# First call includes compilation time
start = time.time()
theta_err, phi_err = optimized_coordinate_conversion(nside, n_coords)
compile_time = time.time() - start

# Second call is just execution
start = time.time()
theta_err, phi_err = optimized_coordinate_conversion(nside, n_coords)
execution_time = time.time() - start

print(f"JIT compilation benchmark:")
print(f"  Coordinates processed: {n_coords}")
print(f"  First call (with compilation): {compile_time:.3f}s")
print(f"  Second call (execution only): {execution_time:.3f}s")
print(f"  Speedup factor: {compile_time/execution_time:.1f}x")
print(f"  Max coordinate errors: θ={theta_err:.2e}, φ={phi_err:.2e}")
```

### Vectorized Operations

```python
# Process multiple maps simultaneously
n_maps = 10
nside = 32
npix = hp.nside2npix(nside)

# Create batch of maps
batch_maps = jax.random.normal(jax.random.PRNGKey(100), (n_maps, npix))

# Vectorized coordinate conversion
@jax.jit
def batch_process_maps(maps):
    """Process multiple maps in parallel."""
    # Get coordinates for all pixels
    pixels = jnp.arange(npix)

    # Vectorize over map dimension
    def process_single_map(single_map):
        # Example processing: calculate map statistics
        mean = jnp.mean(single_map)
        std = jnp.std(single_map)
        return jnp.array([mean, std])

    # Apply to all maps simultaneously
    return jax.vmap(process_single_map)(maps)

# Process all maps at once
start = time.time()
batch_stats = batch_process_maps(batch_maps)
batch_time = time.time() - start

print(f"Batch processing:")
print(f"  Maps processed: {n_maps}")
print(f"  Batch processing time: {batch_time:.4f}s")
print(f"  Time per map: {batch_time/n_maps:.6f}s")
print(f"  Statistics shape: {batch_stats.shape}")  # (n_maps, 2)

# Show some results
for i in range(min(3, n_maps)):
    mean, std = batch_stats[i]
    print(f"  Map {i}: mean={mean:.4f}, std={std:.4f}")
```

This covers the essential usage patterns for jax-healpy. For more advanced examples, see the [clustering analysis](clustering_analysis.md) and [benchmarks](benchmarks.md) sections.
