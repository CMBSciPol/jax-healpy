# Contributing to jax-healpy

We welcome contributions to jax-healpy! This guide will help you get started with contributing to the project.

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- JAX (CPU or GPU version)

### Setting Up Your Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-username/jax-healpy.git
   cd jax-healpy
   ```

2. **Install the development dependencies** with [uv](https://docs.astral.sh/uv/):
   ```bash
   uv sync --all-extras
   ```

3. **Install the git hooks** with [prek](https://github.com/j178/prek), one of the development dependencies:
   ```bash
   uv run prek install
   ```

## Development Workflow

### Making Changes

1. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** following our coding standards (see below)

3. **Add tests** for your changes:
   ```bash
   # Add tests in the appropriate test file under tests/
   # Run tests to ensure they pass
   pytest tests/
   ```

4. **Update documentation** if needed:
   ```bash
   # Update docstrings, README, or the user guide notebooks
   # Build the site locally into site/ to check the formatting
   uv run --group docs sh docs/build.sh
   ```

   The user guide pages are Jupyter notebooks under `docs/user-guide/`. They are committed
   without their outputs, executed as regression tests by `pytest`, and converted to Markdown
   pages by `docs/build.sh` before the site is built by [Zensical](https://zensical.org).
   To preview the site with live reload while editing a Markdown page, run `uv run zensical serve`
   after a first build.

5. **Commit your changes**, adding the files by path and following [Conventional Commits](https://www.conventionalcommits.org):
   ```bash
   git add jax_healpy/pixelfunc.py tests/pixelfunc/test_ang_pix.py
   git commit -m "feat(pixelfunc): support the NEST ordering in ang2pix"
   ```

### Running Tests

We use pytest for testing:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/pixelfunc/test_ang_pix.py

# Run with coverage
pytest --cov=jax_healpy

# Run only fast tests (skip slow benchmarks)
pytest -m "not slow"
```

### Code Quality

We use several tools to maintain code quality:

```bash
uv run prek install

# The git hooks run ruff and other QA tools automatically at each commit
git commit -m "fix(sphtfunc): infer lmax from the size of the alms"

# To run the QA tools on the whole repository:
uv run prek run --all-files
```

## Coding Standards

### Code Style

- Follow PEP 8 with line length of 120 characters
- Use single quotes for strings (configured in ruff)
- Use meaningful variable and function names
- Add type hints for function signatures

### Example Function

```python
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def example_function(nside: int, coordinates: Float[Array, 'n 2'], nest: bool = False) -> Float[Array, 'n']:
    """
    Brief description of what the function does.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter
    coordinates : array_like
        Angular coordinates (theta, phi) in radians, shape (n, 2)
    nest : bool, optional
        Whether to use NESTED ordering (default: False)

    Returns
    -------
    result : array_like
        Description of return value, shape (n,)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> coords = jnp.array([[0.0, 0.0], [jnp.pi/2, 0.0]])
    >>> result = example_function(64, coords)
    """
    # Implementation here
    pass
```

### Docstring Guidelines

- Use NumPy docstring format
- Include clear parameter descriptions with types
- Provide usage examples
- Document any mathematical background when relevant
- Include references to papers or algorithms when appropriate

### Testing Guidelines

- Write tests for all new functions
- Include edge cases and error conditions
- Use descriptive test names
- Add docstrings to test functions explaining what they test

Example test:

```python
import jax.numpy as jnp
import pytest
from jax_healpy import example_function


def test_example_function_basic():
    """Test basic functionality of example_function."""
    nside = 64
    coords = jnp.array([[0.0, 0.0], [jnp.pi / 2, 0.0]])

    result = example_function(nside, coords)

    assert result.shape == (2,)
    assert jnp.all(jnp.isfinite(result))


def test_example_function_invalid_nside():
    """Test that invalid nside raises appropriate error."""
    with pytest.raises(ValueError, match='nside must be'):
        example_function(3, jnp.array([[0.0, 0.0]]))
```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

- Python version and JAX version
- Operating system and hardware (CPU/GPU)
- Minimal code example that reproduces the issue
- Expected vs. actual behavior
- Error messages and stack traces

### Feature Requests

For new features:

- Describe the use case and motivation
- Provide examples of the desired API
- Consider if the feature fits the project scope
- Be willing to help implement or test the feature

### Documentation Improvements

- Fix typos and clarify unclear sections
- Add examples and tutorials
- Improve API documentation
- Update installation instructions

### Performance Improvements

- Profile code to identify bottlenecks
- Provide benchmarks showing improvement
- Ensure accuracy is maintained
- Consider memory usage implications

## Specific Areas for Contribution

### High Priority

- **Additional HEALPix functions**: Implement missing healpy functions
- **Performance optimization**: Improve GPU utilization and memory usage
- **Documentation**: Add tutorials and examples
- **Testing**: Increase test coverage and add benchmarks

### Medium Priority

- **Visualization tools**: Functions for plotting HEALPix maps
- **I/O functions**: Reading/writing FITS files
- **Integration**: Better integration with astronomical libraries
- **Error handling**: Improve error messages and validation

### Advanced

- **New algorithms**: Implement novel HEALPix-based algorithms
- **Distributed computing**: Support for multi-GPU/multi-node processing
- **Automatic differentiation**: Explore AD applications in astronomy
- **Machine learning**: HEALPix-aware neural network layers

## Pull Request Process

1. **Ensure tests pass**: All existing and new tests must pass
2. **Update documentation**: Include docstrings and user guides
3. **Add changelog entry**: Describe your changes in the appropriate section
4. **Title the pull request** as its commit, following Conventional Commits: `<type>[(<scope>)]: <description>`
5. **Keep a single commit**: amend it at each revision (`git commit --amend`) and push it with `git push --force-with-lease`
6. **Link the issue** that the pull request closes: `Closes #<number>`
7. **Request review**: Tag relevant maintainers for review

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Breaking change

## Testing
- [ ] Added tests for new functionality
- [ ] All tests pass locally
- [ ] No significant performance regression

## Documentation
- [ ] Updated docstrings
- [ ] Updated user guide (if applicable)
- [ ] Added examples (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Changes are backwards compatible (or breaking changes documented)
```

## Release Process

Releases are handled by maintainers following semantic versioning:

- **Patch releases** (x.y.Z): Bug fixes, documentation updates
- **Minor releases** (x.Y.z): New features, backwards-compatible changes
- **Major releases** (X.y.z): Breaking changes, major new features

## Getting Help

- **Documentation**: Check the [user guide](https://jax-healpy.readthedocs.io/en/latest/user-guide/getting-started/) and [API reference](https://jax-healpy.readthedocs.io/en/latest/api/pixelfunc/)
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Issues**: Report bugs and request features via GitHub Issues
- **Email**: Contact maintainers for sensitive issues

## Community Guidelines

- Be respectful and inclusive
- Help others learn and contribute
- Follow the Code of Conduct
- Credit others' work appropriately
- Focus on constructive feedback

## Recognition

Contributors are recognized in:

- Release notes and changelog
- GitHub contributors list
- Academic papers (for significant contributions)
- Project documentation

## Troubleshooting Development Issues

### Common Problems

**JAX Installation Issues**:
```bash
# Install JAX with NVIDIA GPU support (CUDA 13)
uv sync --all-extras --group cuda
```

**Import Errors**:
```bash
# Resynchronize the development environment
uv sync --all-extras
```

**Test Failures**:
```bash
uv run pytest tests/
```

**Documentation Build Issues**:
```bash
uv run --group docs sh docs/build.sh
```

### Environment Debugging

```python
# Check your setup
import jax
import jax_healpy as hp

print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')
print(f'jax-healpy version: {hp.__version__ if hasattr(hp, "__version__") else "dev"}')

# Test basic functionality
nside = 32
npix = hp.nside2npix(nside)
print(f'Basic test: nside={nside} → npix={npix}')
```

Thank you for contributing to jax-healpy! 🚀
