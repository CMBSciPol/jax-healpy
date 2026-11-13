# Migrating from healpy

jax-healpy follows the API and the conventions of healpy: most code runs after replacing
`import healpy as hp` by `import jax_healpy as hp`. This page lists the differences, most
of which come from JAX: compilation, random numbers and arrays of fixed shapes.

## Precision

JAX computes in 32-bit precision by default. Enable 64-bit precision at the start of the
program, otherwise jax-healpy warns at the first call:

```python
import jax

jax.config.update('jax_enable_x64', True)
```

In 64-bit precision, the spherical harmonic transforms agree with healpy to about
$10^{-12}$ or better. In 32-bit precision, those of ducc0, the default on CPU, agree to
about $10^{-7}$ at all resolutions. Those of s2fft reach relative differences of $10^{-4}$
to $10^{-3}$ at `nside = 64`, which grow with `nside`, especially for the $B$ modes (see
[Agreement with healpy](polarization.md#agreement-with-healpy)).

The spherical harmonic transforms require the `cpu` extra on CPU, which installs ducc0, or the
`gpu` one on GPU, which installs s2fft (see
[Installation](installation.md#spherical-harmonic-transforms)).

## Arrays and compilation

- The functions return JAX arrays and accept NumPy arrays. healpy's plotting functions,
  such as `mollview`, accept JAX arrays directly.
- The functions are compiled for each value of the arguments that set the shapes of the
  arrays, which must be Python or NumPy numbers, not JAX arrays: `nside`, `lmax`, `mmax`,
  `iter`, `pol`, `spin`... When you compile your own functions with `jax.jit`, declare
  them static (see [Performance tips](performance-tips.md)). Otherwise, JAX raises
  `ValueError: Non-hashable static arguments are not supported`.
- The beam width `fwhm` (or `sigma`) is also static: pass a Python or NumPy float, such as
  `np.radians(1)`, not a JAX scalar such as `jnp.radians(1)`. It cannot be differentiated.
- [`query_disc`][jax_healpy.query_disc] returns an array of fixed size, padded with the
  value `npix`, since the number of pixels in the disc is not known at compilation time.
  Keep the pixels below `npix`, or use `jnp.zeros(npix).at[pixels].set(1, mode='drop')`,
  which ignores the padding.
- The pixel functions require `nside` to be a power of 2, also in the RING scheme.

## Random numbers

[`synalm`][jax_healpy.synalm] and [`synfast`][jax_healpy.synfast] take a JAX PRNG key as
first argument, instead of relying on the global state of NumPy:

```python
key = jax.random.key(0)
cmb_map = hp.synfast(key, cl, nside)
```

The realizations differ from those of healpy for the same seed, but follow the same
statistics. Split the key to draw several independent realizations, or draw a batch of
them at once with `jax.vmap`.

## Spherical harmonic coefficients

- **Layout.** By default, the alms are stored in the layout of s2fft: a two-dimensional
  array of shape `(lmax + 1, 2 * lmax + 1)`, indexed by $\ell$ and $m + \ell_\mathrm{max}$,
  which includes the negative $m$. Pass `healpy_ordering=True` to read and write the
  one-dimensional layout of healpy, for example to exchange alms with healpy.
- **Not implemented:** `mmax != lmax`, and the pixel window (`pixwin=True`) in `alm2map`
  and `synfast`.
- **Band limit.** The transforms of ducc0, the default on CPU, accept any `lmax`, like those
  of healpy. Those of s2fft require `lmax >= 2 * nside - 1`.

## Polarization

`map2alm` and `alm2map` also accept a stack of two maps or alms, interpreted as $Q, U$ or
$E, B$, where healpy requires the temperature too, and `anafast` then returns the
$EE, BB, EB$ spectra (see [Polarization](polarization.md)).

## Masked pixels

healpy replaces the `UNSEEN` pixels by zeros before a transform. jax-healpy never inspects
the pixel values, so that `map2alm` stays a linear function of the map: pass the mask of
the bad pixels explicitly.

```python
alm = hp.map2alm(m, bad_mask=hp.bad_pixel_mask(m))
cl = hp.anafast(jnp.where(hp.bad_pixel_mask(m), 0, m))
```

## Not available

- **Visualization**: use healpy's `mollview`, `gnomview`, `projscatter`...
- **Input and output**: read and write the maps with healpy's `read_map` and `write_map`,
  or with astropy.
- **Other functions**: rotations, polygon and strip queries, dipole fitting... The
  [API reference](../api/pixelfunc.md) lists the available functions.
