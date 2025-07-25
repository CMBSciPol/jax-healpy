# Changelog

All notable changes to jax-healpy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation with ReadTheDocs integration
- User guide for HEALPix pixelization concepts
- API reference with detailed function documentation
- Installation guide with HPC system instructions
- Contributing guidelines for developers

### Changed
- Improved README with better structure and examples
- Enhanced project metadata and PyPI badges

### Fixed
- Documentation build configuration for ReadTheDocs

## [0.1.0] - 2024-XX-XX

### Added
- Initial implementation of core HEALPix functions
- Pixel coordinate conversion functions:
  - `pix2ang`, `ang2pix`: Pixel ↔ angular coordinate conversion
  - `pix2vec`, `vec2pix`: Pixel ↔ unit vector conversion
  - `ang2vec`, `vec2ang`: Angular ↔ unit vector conversion
  - `pix2xyf`, `xyf2pix`: Pixel ↔ face coordinate conversion
- HEALPix scheme conversions:
  - `ring2nest`, `nest2ring`: Convert between RING and NESTED ordering
  - `reorder`: Reorder entire maps between schemes
- Resolution parameter functions:
  - `nside2npix`, `npix2nside`: Convert resolution ↔ pixel count
  - `nside2order`, `order2nside`: Convert resolution ↔ order parameter
  - `order2npix`, `npix2order`: Convert order ↔ pixel count
  - `nside2resol`, `nside2pixarea`: Calculate angular resolution and pixel area
- Map interpolation functions:
  - `get_interp_weights`: Get interpolation weights for arbitrary coordinates
  - `get_interp_val`: Interpolate map values at arbitrary coordinates
- Spherical harmonic transforms (requires s2fft):
  - `map2alm`: Forward spherical harmonic transform
  - `alm2map`: Inverse spherical harmonic transform
- Query disc functionality:
  - `query_disc`: Find pixels within circular regions on sphere
- Clustering algorithms:
  - `KMeans`: JAX-based K-means clustering implementation
  - `kmeans_sample`: Simplified K-means clustering function
  - `get_clusters`: Find connected clusters in binary masks
  - `get_cutout_from_mask`: Extract map cutouts based on masks
  - `from_cutout_to_fullmap`: Insert cutout data back into full maps
- Mask manipulation utilities:
  - `combine_masks`: Combine multiple binary masks
  - `normalize_by_first_occurrence`: Normalize cluster labels
  - `shuffle_labels`: Randomly shuffle cluster identities
- Utility functions:
  - `isnsideok`, `isnpixok`: Validate HEALPix parameters
  - `maptype`: Determine map data type and properties
- Constants:
  - `UNSEEN`: Sentinel value for invalid/missing pixels
- JAX integration features:
  - GPU acceleration support
  - Automatic differentiation compatibility
  - JIT compilation optimization
  - Vectorized batch processing
- Comprehensive test suite with pytest
- Benchmarking framework comparing performance to healpy
- Development tools:
  - Pre-commit hooks for code quality
  - Ruff linting and formatting
  - Type checking with mypy
  - Coverage reporting

### Dependencies
- JAX: Core computational framework
- JAXtyping: Type annotations for JAX arrays
- s2fft (optional): Spherical harmonic transforms
- healpy (test only): Reference implementation for testing

### Documentation
- Basic README with installation and usage examples
- Docstrings following NumPy format
- Type hints for all public functions
- Mathematical background for key algorithms

### Performance
- Significant speedups on GPU hardware compared to healpy
- Optimized batch processing for multiple maps/coordinates
- Memory-efficient implementations for large-scale computations
- JIT compilation for optimal runtime performance

### Testing
- Comprehensive test coverage for all core functions
- Accuracy validation against healpy reference implementation
- Performance benchmarking suite
- Edge case and error condition testing
- Continuous integration setup

### Known Limitations
- Beta software: APIs may change in future versions
- Limited HEALPix function coverage compared to healpy
- Spherical harmonics require additional s2fft dependency
- Some advanced healpy features not yet implemented

---

## Development Notes

### Version Numbering
- **Major version** (X.y.z): Breaking API changes, major new features
- **Minor version** (x.Y.z): New features, backwards-compatible changes
- **Patch version** (x.y.Z): Bug fixes, documentation updates

### Release Process
1. Update version in `pyproject.toml`
2. Update changelog with release notes
3. Create git tag with version number
4. Build and upload to PyPI
5. Update documentation on ReadTheDocs

### Contributing
See [Contributing Guide](contributing.md) for details on:
- Development setup and workflow
- Code style guidelines
- Testing requirements
- Documentation standards
- Pull request process

### Acknowledgments
This project builds on the excellent work of:
- The original [HEALPix](https://healpix.jpl.nasa.gov/) team
- The [healpy](https://healpy.readthedocs.io/) developers
- The [JAX](https://jax.readthedocs.io/) team at Google
- The [s2fft](https://astro-informatics.github.io/s2fft/) developers
