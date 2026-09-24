# Changelog

All notable changes to jax-healpy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8] - 2026-09-24

### Added

- `pix2loc` and `loc2pix`: pixel ↔ (cos θ, sin θ, φ) conversions, without trigonometric round trips (#22)
- NEST ordering for `ang2pix`, `pix2ang`, `vec2pix`, `pix2vec`, `loc2pix` and `pix2loc` (#23)

### Fixed

- `ang2pix` and `vec2pix` use sin θ near the poles (|z| > 0.99), as Healpix C++ does, so near-pole directions at high nside no longer land in the wrong pixel (#22)

## [0.7.1] - 2026-08-12

### Added

- `bad_pixel_mask`, exported at top level (#20)
- `get_interp_weights(..., with_centers=True)` also returns the neighbours' centers as an `InterpCenters` pytree (#21)

### Changed

- **Breaking:** `map2alm` and `map2alm_spin` take a `bad_mask` argument instead of inspecting pixel values, which keeps them linear; maps with `UNSEEN` or non-finite pixels must be masked explicitly (#20)
- `mask_bad` uses healpy's tolerant comparison and is exported at top level (#14)
- `smoothing` restores masked pixels to `UNSEEN` in its output (#14)

## [0.7] - 2026-06-15

### Added

- Spherical harmonic transforms, including polarisation: `alm2map_spin`, `map2alm_spin`, `alm2cl`, `anafast`, `synalm`, `synfast`, `almxfl`, `smoothalm`, `smoothing`, `gauss_beam`, `pixwin` (#4)
- `precompute_temperature_harmonic_transforms` and `precompute_polarization_harmonic_transforms` (#4)
- Clustering examples in the documentation (#1)

### Changed

- **Breaking:** importing jax-healpy no longer enables 64-bit precision; a warning is emitted instead (#5)
- **Breaking:** drop Python 3.10, require JAX 0.10+ (#6)
- `ang2vec`, `vec2ang`, `pix2vec` and `get_all_neighbours` preserve batch dimensions (#2)
- Pixel indices are `int32` for nside ≤ 8192 (#5)
- Build with hatchling and hatch-vcs (#8)
- Unsupported arguments raise `NotImplementedError` (#10)

### Removed

- No-op `verbose` and `inplace` arguments of the spherical harmonic functions (#4)

### Fixed

- Importing jax-healpy no longer initialises the JAX backend (#7)

## [0.6] - 2025-10-10

### Fixed

- `CITATION.cff` metadata

## [0.5] - 2025-09-24

### Added

- `get_all_neighbours`
- `ud_grade`
- `get_nside`
- `estimate_disc_pixel_count` and `estimate_disc_radius`
- `UNSEEN` exported at top level

### Changed

- **Breaking:** clustering functions move to the `jax_healpy.clustering` subpackage; `get_clusters` is renamed `find_kmeans_clusters` and `from_cutout_to_fullmap` is renamed `get_fullmap_from_cutout`
- `query_disc` uses less memory

## [0.4] - 2025-07-25

### Fixed

- Documentation links and version

## [0.3] - 2025-07-25

### Added

- `get_interp_weights` and `get_interp_val`: bilinear interpolation for RING ordering
- `query_disc`
- K-means clustering and mask utilities: `KMeans`, `kmeans_sample`, `get_clusters`, `get_cutout_from_mask`, `from_cutout_to_fullmap`, `combine_masks`, `normalize_by_first_occurrence`, `shuffle_labels`
- Documentation on ReadTheDocs
- Citation file and license

## [0.2.1] - 2025-02-03

### Fixed

- Setuptools version for the release build

## [0.2] - 2025-02-03

### Added

- `map2alm` and `alm2map` (spin 0) through s2fft, with batched maps
- `ring2nest`, `nest2ring` and `reorder`
- Benchmarks against healpy
- Release workflow

### Changed

- s2fft is an optional dependency

## [0.1] - 2023-10-04

Initial tagged release.

### Added

- `pix2ang`, `ang2pix`, `pix2vec`, `vec2pix`, `ang2vec` and `vec2ang`
- `nside2npix`, `npix2nside`, `nside2order`, `order2nside`, `order2npix`, `npix2order`, `nside2resol` and `nside2pixarea`
- `isnsideok`, `isnpixok` and `maptype`

[unreleased]: https://github.com/CMBSciPol/jax-healpy/compare/v0.8...HEAD
[0.8]: https://github.com/CMBSciPol/jax-healpy/compare/v0.7.1...v0.8
[0.7.1]: https://github.com/CMBSciPol/jax-healpy/compare/v0.7...v0.7.1
[0.7]: https://github.com/CMBSciPol/jax-healpy/compare/v0.6...v0.7
[0.6]: https://github.com/CMBSciPol/jax-healpy/compare/v0.5...v0.6
[0.5]: https://github.com/CMBSciPol/jax-healpy/compare/v0.4...v0.5
[0.4]: https://github.com/CMBSciPol/jax-healpy/compare/v0.3...v0.4
[0.3]: https://github.com/CMBSciPol/jax-healpy/compare/v0.2.1...v0.3
[0.2.1]: https://github.com/CMBSciPol/jax-healpy/compare/v0.2...v0.2.1
[0.2]: https://github.com/CMBSciPol/jax-healpy/compare/v0.1...v0.2
[0.1]: https://github.com/CMBSciPol/jax-healpy/releases/tag/v0.1
