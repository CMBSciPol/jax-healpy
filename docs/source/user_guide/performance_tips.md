# Performance Tips

## Compilation

jax-healpy functions are JIT-compiled on their first call for a given set of input shapes, dtypes and
static arguments (such as `nside`). Wrapping your own pipelines with `jax.jit` lets XLA fuse the
operations. The resolution parameter `nside` determines the array shapes, so it must be declared
static:

```python
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax_healpy as hp


@partial(jax.jit, static_argnames='nside')
def roundtrip(nside, pixels):
    theta, phi = hp.pix2ang(nside, pixels)
    return hp.ang2pix(nside, theta, phi)


nside = 128
pixels = jnp.arange(hp.nside2npix(nside))

# The first call triggers the compilation
roundtrip(nside, pixels).block_until_ready()

# Subsequent calls with the same shapes and nside reuse the compiled function
start = time.perf_counter()
roundtrip(nside, pixels).block_until_ready()
print(f'Round-trip time: {time.perf_counter() - start:.4f} seconds')
```

JAX dispatches computations asynchronously, so call `block_until_ready()` before stopping a timer.
Each new value of `nside` triggers a new compilation.

## GPU Acceleration

The operations run on the default JAX device, which is the GPU if JAX has been installed with GPU support:

```python
print(f'JAX backend: {jax.default_backend()}')
print(f'JAX devices: {jax.devices()}')
```

## Batch Processing

Most functions broadcast over their array arguments. For more complex pipelines, `jax.vmap` vectorizes
a function over a batch dimension:

```python
# Interpolate a batch of maps at the same locations
batch_size = 10
npix = hp.nside2npix(nside)
batch_maps = jax.random.normal(jax.random.key(0), (batch_size, npix))
theta = jnp.array([0.5, 1.0, 1.5])
phi = jnp.array([0.0, 1.0, 2.0])

values = jax.vmap(hp.get_interp_val, in_axes=(0, None, None))(batch_maps, theta, phi)
print(values.shape)  # (10, 3)
```

## Memory Management

For very large operations that do not fit in the device memory, process the data in chunks:

```python
def ang2pix_in_chunks(nside, theta, phi, chunk_size=1_000_000):
    results = []
    for start in range(0, theta.size, chunk_size):
        stop = start + chunk_size
        results.append(hp.ang2pix(nside, theta[start:stop], phi[start:stop]))
    return jnp.concatenate(results)
```

By default, JAX preallocates 75% of the GPU memory on its first operation. This behavior can be changed
through [environment variables](https://docs.jax.dev/en/latest/gpu_memory_allocation.html), for
example `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

Using 32-bit precision halves the memory footprint, but the results then diverge from healpy even at
moderate resolution, so 64-bit precision is recommended (see [Getting Started](getting_started.md)).
