# Differences from healpy

jax-healpy is used alongside healpy. It implements in JAX the functions of healpy that compute,
pixel functions and spherical harmonic transforms, so that they can be compiled, batched,
differentiated and run on GPU. healpy keeps the input and output and the visualization:

```python
import healpy as hp
import jax
import jax_healpy as jhp
import numpy as np

jax.config.update('jax_enable_x64', True)

m = hp.read_map('map.fits', dtype=np.float64)  # input: healpy
smoothed = jhp.smoothing(m, fwhm=np.radians(1))  # computation: jax-healpy
hp.mollview(smoothed)  # visualization: healpy
```

The functions of jax-healpy accept NumPy arrays and return JAX arrays, which the functions of
healpy, such as `mollview` or `write_map`, accept directly. JAX rejects the big-endian arrays
that `read_map` returns by default: pass `dtype=np.float64` (or `np.float32`) to `read_map`, as
above, to get native arrays.

jax-healpy follows the API and the conventions of healpy: most computations run after replacing
`import healpy as hp` by `import jax_healpy as hp`. This page lists the differences: the
functionality that jax-healpy lacks, its intentional deviations from the API of healpy, and the
behaviors that come from JAX. What jax-healpy brings in addition to healpy is in
[What jax-healpy adds](what-jax-healpy-adds.md).

## Missing functionality

The functions of healpy that jax-healpy does not provide, and the arguments that it does not
accept, are either:

- **out of scope**: they are left to healpy;
- **not implemented yet**: they could be added, and contributions are welcome.

The [API reference](../api/pixelfunc.md) lists the available functions.

| Kind | Functions and arguments of healpy | Status |
|---|---|---|
| Visualization | `mollview`, `gnomview`, `cartview`, `orthview`, `azeqview`, `projview`, `mollzoom`, `projplot`, `projscatter`, `projtext`, `newprojplot`, `graticule`, `delgraticules`, `set_g_clim`, `line_integral_convolution` | out of scope |
| Input and output | `read_map`, `write_map`, `read_alm`, `write_alm`, `read_cl`, `write_cl`, `sphtfunc.load_sample_spectra` | out of scope |
| Masked arrays | `ma`, `pixelfunc.is_ma`, `pixelfunc.accept_ma`, `pixelfunc.ma_to_array`: JAX has no masked arrays, see [Masked pixels](#masked-pixels) | out of scope |
| In-place operations | the `inplace` argument of `alm2map`, `almxfl` and `smoothalm`: JAX arrays are immutable | out of scope |
| Utilities of healpy itself | `disable_warnings`, `enable_warnings`, `deprecated`, `check_max_nside`, the `verbose` argument | out of scope |
| Rotations | `Rotator`, `rotate_alm`, and the functions of `healpy.rotator`, such as `angdist`, `euler_matrix_new` or `rotateVector` | not implemented yet |
| Pixel queries and geometry | `query_polygon`, `query_strip`, `boundaries`, `max_pixrad`, `pix2ring`, `ringinfo`, `dist2holes`, `hotspots`; `nest=True` in [`query_disc`][jax_healpy.query_disc]; the `buff` and `return_ranges` arguments of `query_disc`; the `latauto` and `latbounce` arguments of `ang2pix` | not implemented yet |
| Monopole and dipole | `fit_monopole`, `fit_dipole`, `remove_monopole`, `remove_dipole` | not implemented yet |
| Spherical harmonics | `alm2map_der1`, `map2alm_lsq`, `resize_alm`, `harmonic_ud_grade`, `beam2bl`, `bl2beam`, `blm_gauss`, `effective_resolution_fwhm`, the `Alm` class; `mmax != lmax`; [`pixwin`][jax_healpy.pixwin], which raises a `NotImplementedError`, and `pixwin=True` in `alm2map` and `synfast`; `nest=True` in `smoothing`; the weights of `map2alm`, `anafast` and `smoothing` (`use_weights`, `use_pixel_weights`, `datapath`) and `gal_cut` | not implemented yet |
| Others | `dir2vec` and `vec2dir` (see [`ang2vec`][jax_healpy.ang2vec] and [`vec2ang`][jax_healpy.vec2ang]), `get_map_size`, `mask_good` (see [`mask_bad`][jax_healpy.mask_bad]), `pixelfunc.get_min_valid_nside`, `pixelfunc.lonlat2thetaphi`, `pixelfunc.thetaphi2lonlat` | not implemented yet |

## Intentional API deviations

### Random numbers

[`synalm`][jax_healpy.synalm] and [`synfast`][jax_healpy.synfast] take a JAX PRNG key as
first argument, instead of relying on the global state of NumPy:

```python
key = jax.random.key(0)
cmb_map = jhp.synfast(key, cl, nside)
```

The realizations differ from those of healpy for the same seed, but follow the same
statistics. Split the key to draw several independent realizations, or draw a batch of
them at once with `jax.vmap`.

### Layout of the spherical harmonic coefficients

By default, the alms are stored in the layout of s2fft: a two-dimensional array of shape
`(lmax + 1, 2 * lmax + 1)`, indexed by $\ell$ and $m + \ell_\mathrm{max}$, which includes the
negative $m$. Pass `healpy_ordering=True` to read and write the one-dimensional layout of
healpy, for example to exchange alms with healpy.

### Disc queries

[`query_disc`][jax_healpy.query_disc] returns an array of fixed size, padded with the value
`npix`, since the number of pixels in the disc is not known at compilation time. Keep the
pixels below `npix`, or use `jnp.zeros(npix).at[pixels].set(1, mode='drop')`, which ignores the
padding. The size is `npix` by default: the `max_length` argument bounds it, and
[`estimate_disc_pixel_count`][jax_healpy.estimate_disc_pixel_count] estimates the number of
pixels in a disc. A batch of disc centers, of shape `(B, 3)`, returns an array of shape
`(B, max_length)`.

### Masked pixels

healpy replaces the `UNSEEN` pixels by zeros before a transform. jax-healpy's transforms never
inspect the pixel values, so that `map2alm` stays a linear function of the map: pass the mask
of the bad pixels explicitly.

```python
alm = jhp.map2alm(m, bad_mask=jhp.bad_pixel_mask(m))
cl = jhp.anafast(jnp.where(jhp.bad_pixel_mask(m), 0, m))
```

### Polarization

`map2alm` and `alm2map` also accept a stack of two maps or alms, interpreted as $Q, U$ or
$E, B$, where healpy requires the temperature too, and `anafast` then returns the
$EE, BB, EB$ spectra (see [Polarization](polarization.md)).

### Resolution

The pixel functions require `nside` to be a power of 2, also in the RING scheme.

### Backends of the spherical harmonic transforms

The spherical harmonic transforms require the `cpu` extra on CPU, which installs ducc0, or the
`gpu` one on GPU, which installs s2fft (see
[Installation](installation.md#spherical-harmonic-transforms)). The `method` argument selects
the backend (see [Transform backends](../api/sphtfunc.md#transform-backends)). The transforms
of ducc0, the default on CPU, accept any `lmax`, like those of healpy. Those of s2fft require
`lmax >= 2 * nside - 1`.

### Additional arguments

Some functions accept arguments that healpy lacks: `healpy_ordering` and `method` in the
spherical harmonic functions, `bad_mask` in `map2alm` and `map2alm_spin`, `get_center` in
`get_all_neighbours`, `with_centers` in `get_interp_weights` and `process_by_chunks` in
`reorder`.

## Behaviors that come from JAX

### Precision

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

The pixel indices are 32-bit integers up to `nside = 8192`, and 64-bit integers above, which
require 64-bit precision.

### Compilation and static arguments

The functions are compiled for each value of the arguments that set the shapes of the
arrays, which must be Python or NumPy numbers, not JAX arrays: `nside`, `lmax`, `mmax`,
`iter`, `pol`, `spin`... When you compile your own functions with `jax.jit`, declare
them static (see [Performance tips](performance-tips.md)). Otherwise, JAX raises
`ValueError: Non-hashable static arguments are not supported`.

The beam width `fwhm` (or `sigma`) is also static: pass a Python or NumPy float, such as
`np.radians(1)`, not a JAX scalar such as `jnp.radians(1)`. It cannot be differentiated.

### Fixed shapes

The shape of an array is known at compilation time: a function whose output size depends on
the values of its inputs returns a padded array of fixed size, as
[`query_disc`](#disc-queries) does.

### PRNG keys

JAX has no global random state: the functions that draw random numbers take a PRNG key (see
[Random numbers](#random-numbers)).
