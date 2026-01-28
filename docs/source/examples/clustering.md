# Clustering and Mask Utilities

This guide demonstrates how to use `jax-healpy` for clustering analysis and mask manipulation on HEALPix maps. These tools are particularly useful for isolating regions of interest, such as galaxy clusters or specific sky patches.

## Setup

First, let's import the necessary libraries and configure JAX.

```python
import jax
import jax.numpy as jnp
import numpy as np
import jax_healpy as hp
from jax_healpy.clustering import (
    find_kmeans_clusters,
    get_cutout_from_mask,
    get_fullmap_from_cutout,
    combine_masks,
    normalize_by_first_occurrence
)

# Enable 64-bit precision for better accuracy
jax.config.update("jax_enable_x64", True)
```

## K-Means Clustering on the Sphere

You can cluster pixels on the sphere using K-Means. This is useful for dividing the sky into regions or finding groups of sources.

```python
# Create a sample mask (e.g., from a galaxy catalog or intensity threshold)
nside = 64
npix = hp.nside2npix(nside)
indices = jnp.arange(npix)

# Simulate some "valid" pixels (e.g., where we have data)
# In a real scenario, this would come from your data mask
key = jax.random.PRNGKey(0)
mask = jax.random.bernoulli(key, p=0.1, shape=(npix,))
valid_indices = jnp.where(mask)[0]

print(f"Number of valid pixels: {len(valid_indices)}")

# Perform K-Means clustering
# We want to divide the valid pixels into 5 regions
n_regions = 5
clustered_map = find_kmeans_clusters(
    mask, 
    valid_indices, 
    n_regions=n_regions, 
    key=key,
    max_centroids=10 # Maximum buffer for centroids (useful for JIT)
)

print(f"Unique labels in map: {jnp.unique(clustered_map)}")
```

### Normalizing Labels

After clustering, you might want to normalize the labels so they are contiguous and ordered by their first appearance.

```python
# Normalize labels
normalized_map = normalize_by_first_occurrence(
    clustered_map, 
    n_regions=n_regions, 
    max_centroids=10
)

print(f"Normalized labels: {jnp.unique(normalized_map)}")
```

## Working with Cutouts

When working with specific regions, it's often more efficient to extract just the pixels of interest (cutouts) rather than processing the full sky map.

### Extracting a Cutout

```python
# Generate a random full-sky map
full_map = jax.random.normal(key, (npix,))

# Extract values only for our valid pixels
cutout = get_cutout_from_mask(full_map, valid_indices)

print(f"Full map shape: {full_map.shape}")
print(f"Cutout shape: {cutout.shape}")
```

### Reconstructing the Full Map

You can insert the processed cutout back into a full-sky map.

```python
# Suppose we modified the cutout (e.g., smoothed it, or applied some filter)
processed_cutout = cutout * 2.0

# Put it back into the full map structure
# Pixels not in the cutout will be filled with hp.UNSEEN
reconstructed_map = get_fullmap_from_cutout(
    processed_cutout, 
    valid_indices, 
    nside=nside
)

# Check a pixel
idx = valid_indices[0]
print(f"Original value: {full_map[idx]:.4f}")
print(f"Processed value: {reconstructed_map[idx]:.4f}") # Should be ~2x
```

## Combining Masks

You can combine multiple masks or cutouts into a single map structure. This is useful when you have multiple patches and want to assemble them.

```python
# Create two different cutouts (regions)
indices1 = jnp.array([1, 2, 3])
indices2 = jnp.array([10, 11, 12])

cutout1 = jnp.array([100., 200., 300.]) # Data for region 1
cutout2 = jnp.array([1000., 2000., 3000.]) # Data for region 2

# Combine them into a single map
# Note: You need to provide the indices for each cutout
combined_map = combine_masks(
    [cutout1, cutout2], 
    [indices1, indices2], 
    nside=nside
)

# Check values
print(f"Pixel 1 value: {combined_map[1]}")
print(f"Pixel 10 value: {combined_map[10]}")
print(f"Pixel 50 value (unseen): {combined_map[50]}") # Should be UNSEEN
```

## Advanced: Performance Tips

- **JIT Compatibility**: Most functions in this module are JIT-compatible.
    - `find_kmeans_clusters` can be JIT-compiled. If `n_regions` is dynamic (a tracer), you **must** provide `max_centroids` as a static argument to define the maximum buffer size. If `n_regions` is static, `max_centroids` is optional.
    - **Exception**: `shuffle_labels` uses NumPy for randomization and is **not** JIT-compatible. It is primarily intended for visualization (e.g., to make contiguous clusters distinct in plots) rather than performance-critical loops.

- **Static Arguments**: When JIT-compiling, ensure arguments like `nside` are marked as static if they affect array shapes.

