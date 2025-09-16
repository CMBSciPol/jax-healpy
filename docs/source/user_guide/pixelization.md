# HEALPix Pixelization

This guide covers the HEALPix pixelization scheme and how to work with it using jax-healpy.

## Introduction to HEALPix

HEALPix (Hierarchical Equal Area isoLatitude Pixelization) is a pixelization scheme for the sphere that provides:

- **Equal area pixels**: All pixels have the same area on the sphere
- **Hierarchical structure**: Pixels can be subdivided recursively
- **Isolatitude rings**: Pixels are arranged in rings of constant latitude
- **Efficient spherical harmonic transforms**: Optimized for fast SHT computation

## Resolution Parameters

HEALPix uses several parameters to define the resolution:

### NSIDE Parameter

The fundamental resolution parameter is `nside`, which must be a power of 2:

```python
import jax_healpy as hp

# Valid nside values: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ...
nside = 64

# Calculate total number of pixels
npix = hp.nside2npix(nside)
print(f"NSIDE {nside} has {npix} pixels")  # 49152 pixels
```

### Related Parameters

```python
# Convert between different resolution parameters
nside = 64
order = hp.nside2order(nside)  # order = 6 (since 2^6 = 64)
npix = hp.nside2npix(nside)    # npix = 49152

# Convert back
nside_recovered = hp.order2nside(order)
nside_from_npix = hp.npix2nside(npix)

print(f"Original nside: {nside}")
print(f"From order: {nside_recovered}")
print(f"From npix: {nside_from_npix}")
```

### Angular Resolution

```python
# Calculate angular resolution
resol_arcmin = hp.nside2resol(nside, arcmin=True)
resol_rad = hp.nside2resol(nside, arcmin=False)

print(f"Resolution at nside={nside}:")
print(f"  {resol_arcmin:.2f} arcminutes")
print(f"  {resol_rad:.6f} radians")

# Pixel area
pixarea_sr = hp.nside2pixarea(nside)  # steradians
pixarea_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600  # square arcminutes

print(f"Pixel area: {pixarea_sr:.2e} steradians")
print(f"Pixel area: {pixarea_arcmin2:.2f} square arcminutes")
```

## Coordinate Systems

HEALPix uses spherical coordinates with specific conventions:

### Angular Coordinates

```python
import jax.numpy as jnp

# Spherical coordinates: (theta, phi)
# theta: colatitude [0, π] (0 = North pole, π = South pole)
# phi: azimuth [0, 2π] (0 = +X axis in equatorial plane)

# Examples of common directions
north_pole = (0.0, 0.0)                    # theta=0, phi arbitrary
south_pole = (jnp.pi, 0.0)                 # theta=π, phi arbitrary
equator_x = (jnp.pi/2, 0.0)               # +X axis
equator_y = (jnp.pi/2, jnp.pi/2)          # +Y axis
equator_minus_x = (jnp.pi/2, jnp.pi)      # -X axis

# Convert specific coordinates
ra_deg, dec_deg = 83.6331, 22.0145  # Crab Nebula in degrees

# Convert RA/Dec to HEALPix theta/phi
theta = jnp.radians(90.0 - dec_deg)  # Colatitude = 90° - declination
phi = jnp.radians(ra_deg)             # Azimuth = right ascension

print(f"Crab Nebula: theta={theta:.4f}, phi={phi:.4f} radians")
```

### Unit Vectors

HEALPix also works with 3-dimensional unit vectors:

```python
# Convert angles to unit vectors
theta, phi = jnp.pi/3, jnp.pi/4  # 60° colatitude, 45° azimuth
unit_vec = hp.ang2vec(theta, phi)

print(f"Unit vector: {unit_vec}")
print(f"Length: {jnp.linalg.norm(unit_vec):.6f}")  # Should be 1.0

# Convert back to angles
theta_back, phi_back = hp.vec2ang(unit_vec[0], unit_vec[1], unit_vec[2])
print(f"Recovered angles: theta={theta_back:.4f}, phi={phi_back:.4f}")
```

## Pixel Indexing Schemes

HEALPix supports two pixel ordering schemes:

### RING Ordering (Default)

Pixels are numbered in rings of constant latitude:

```python
nside = 4
pixels_ring = jnp.arange(12)  # First 12 pixels

# Convert to coordinates
theta_ring, phi_ring = hp.pix2ang(nside, pixels_ring, nest=False)

print("RING ordering:")
for i, (t, p) in enumerate(zip(theta_ring, phi_ring)):
    print(f"Pixel {i:2d}: theta={t:.3f}, phi={p:.3f}")
```

### NESTED Ordering

Pixels follow a hierarchical tree structure:

```python
# Same pixels in NESTED ordering
theta_nest, phi_nest = hp.pix2ang(nside, pixels_ring, nest=True)

print("\nNESTED ordering:")
for i, (t, p) in enumerate(zip(theta_nest, phi_nest)):
    print(f"Pixel {i:2d}: theta={t:.3f}, phi={p:.3f}")
```

### Converting Between Schemes

```python
# Convert RING to NESTED
ring_pixels = jnp.arange(100)
nest_pixels = hp.ring2nest(nside, ring_pixels)

# Convert NESTED to RING
ring_pixels_back = hp.nest2ring(nside, nest_pixels)

# Verify round-trip conversion
print(f"Round-trip successful: {jnp.allclose(ring_pixels, ring_pixels_back)}")

# Reorder entire maps
ring_map = jnp.random.normal(0, 1, hp.nside2npix(nside))
nest_map = hp.reorder(ring_map, r2n=True)   # RING to NESTED
ring_map_back = hp.reorder(nest_map, n2r=True)  # NESTED to RING

print(f"Map reordering successful: {jnp.allclose(ring_map, ring_map_back)}")
```

## Coordinate Conversions

### Basic Conversions

```python
nside = 32
npix = hp.nside2npix(nside)

# Generate all pixel indices
all_pixels = jnp.arange(npix)

# Convert pixels to coordinates
theta_all, phi_all = hp.pix2ang(nside, all_pixels)
vectors_all = hp.pix2vec(nside, all_pixels)

# Convert coordinates back to pixels
pixels_from_ang = hp.ang2pix(nside, theta_all, phi_all)
pixels_from_vec = hp.vec2pix(nside, vectors_all[0], vectors_all[1], vectors_all[2])

# Verify conversions
print(f"Angle conversion accuracy: {jnp.allclose(all_pixels, pixels_from_ang)}")
print(f"Vector conversion accuracy: {jnp.allclose(all_pixels, pixels_from_vec)}")
```

### Face Coordinates

HEALPix internally uses coordinates within 12 base faces:

```python
# Convert pixels to face coordinates
pixels = jnp.arange(24)  # First 24 pixels
x, y, face = hp.pix2xyf(nside, pixels)

print("Face coordinates:")
for i, (xi, yi, fi) in enumerate(zip(x, y, face)):
    print(f"Pixel {i:2d}: x={xi:.3f}, y={yi:.3f}, face={fi}")

# Convert back to pixels
pixels_back = hp.xyf2pix(nside, x, y, face)
print(f"\nFace coordinate round-trip: {jnp.allclose(pixels, pixels_back)}")
```

## Working with Real Data

### Astronomical Coordinates

```python
# Convert from equatorial coordinates (RA/Dec)
def radec_to_healpix(ra_deg, dec_deg, nside, nest=False):
    """Convert RA/Dec coordinates to HEALPix pixel indices."""
    theta = jnp.radians(90.0 - dec_deg)  # Convert Dec to colatitude
    phi = jnp.radians(ra_deg)            # RA is already azimuth
    return hp.ang2pix(nside, theta, phi, nest=nest)

# Example: Messier catalog objects
messier_coords = [
    ("M1 (Crab Nebula)", 83.6331, 22.0145),
    ("M31 (Andromeda)", 10.6847, 41.2689),
    ("M42 (Orion Nebula)", 83.8221, -5.3911)
]

nside = 128
for name, ra, dec in messier_coords:
    pixel = radec_to_healpix(ra, dec, nside)
    print(f"{name}: RA={ra:.2f}°, Dec={dec:.2f}° → Pixel {pixel}")
```

### Map Resolution Considerations

```python
# Choose appropriate nside for your application
def recommend_nside(angular_scale_arcmin):
    """Recommend nside for a given angular scale."""
    # Rule of thumb: pixel size should be ~1/3 of smallest feature
    desired_pixel_size = angular_scale_arcmin / 3.0

    # Find appropriate nside
    for nside in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        pixel_size = hp.nside2resol(nside, arcmin=True)
        if pixel_size <= desired_pixel_size:
            return nside
    return 1024  # Maximum commonly used resolution

# Examples
scales = [60, 30, 10, 5, 1]  # arcminutes
for scale in scales:
    recommended = recommend_nside(scale)
    actual_resolution = hp.nside2resol(recommended, arcmin=True)
    print(f"Feature scale {scale}': recommended nside={recommended} "
          f"(pixel size {actual_resolution:.2f}')")
```

## Performance Tips

### Vectorized Operations

```python
# Process many coordinates at once
n_coords = 100000
theta_batch = jnp.random.uniform(0, jnp.pi, n_coords)
phi_batch = jnp.random.uniform(0, 2*jnp.pi, n_coords)

# Vectorized conversion (efficient)
pixels_batch = hp.ang2pix(nside, theta_batch, phi_batch)

# This is much faster than:
# pixels_slow = [hp.ang2pix(nside, t, p) for t, p in zip(theta_batch, phi_batch)]
```

### JIT Compilation

```python
import jax

# Compile coordinate conversion for speed
@jax.jit
def fast_coordinate_pipeline(nside, n_random):
    # Generate random coordinates
    theta = jax.random.uniform(jax.random.PRNGKey(42), (n_random,),
                               minval=0, maxval=jnp.pi)
    phi = jax.random.uniform(jax.random.PRNGKey(43), (n_random,),
                            minval=0, maxval=2*jnp.pi)

    # Convert to pixels and back
    pixels = hp.ang2pix(nside, theta, phi)
    theta_back, phi_back = hp.pix2ang(nside, pixels)

    return theta_back, phi_back

# First call compiles, subsequent calls are fast
result = fast_coordinate_pipeline(64, 10000)
```

## Validation and Error Checking

```python
# Validate nside values
test_nsides = [1, 3, 64, 127, 128]
for nside in test_nsides:
    is_valid = hp.isnsideok(nside)
    print(f"nside={nside}: {'valid' if is_valid else 'invalid'}")

# Validate pixel indices
nside = 64
npix = hp.nside2npix(nside)
test_pixels = [0, npix//2, npix-1, npix, -1]

for pixel in test_pixels:
    is_valid = hp.isnpixok(nside, pixel)
    print(f"nside={nside}, pixel={pixel}: {'valid' if is_valid else 'invalid'}")

# Check map properties
test_map = jnp.random.normal(0, 1, npix)
map_type = hp.maptype(test_map)
print(f"Map type: {map_type}")  # Should be 'random' or similar
```
