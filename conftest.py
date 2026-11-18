"""Configuration shared by the tests of `tests/` and the doctests of `jax_healpy/`."""

import importlib.util
import os

import pytest

# The tests are run in parallel by pytest-xdist: ducc0 gets a single thread in each process, instead of
# all the cores. It reads this variable when it is first imported.
os.environ.setdefault('DUCC0_NUM_THREADS', '1')


@pytest.fixture(scope='session', autouse=True)
def default_x64():
    """Default the whole test session to 64-bit precision.

    jax-healpy does not enable x64 on import, but most tests check numerical accuracy against
    healpy at float64 tolerances, and the examples of the docstrings show the results of a
    program that enabled it, as the documentation recommends. Tests that exercise both
    precisions use the parametrized ``x64`` fixture of ``tests/conftest.py``, whose context
    manager overrides this default within the test.
    """
    import jax

    with jax.enable_x64(True):
        yield


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip the doctests that need an optional library which is not installed.

    As in ``tests/sphtfunc``, the doctests of the spherical harmonic functions are skipped when
    neither ducc0 nor s2fft is installed, and those of the functions decorated with
    ``requires_s2fft`` when s2fft is not. The wrappers of this decorator all share the code
    object of its inner function.
    """
    if not isinstance(item, pytest.DoctestItem):
        return
    from jax_healpy.sphtfunc import _s2fft_is_available, requires_s2fft

    if _s2fft_is_available():
        return
    if item.dtest.name.startswith('jax_healpy.sphtfunc.') and importlib.util.find_spec('ducc0') is None:
        pytest.skip('neither ducc0 nor s2fft is installed')
    func = item.dtest.globs.get(item.dtest.name.rsplit('.', 1)[-1])
    if getattr(func, '__code__', None) is requires_s2fft(lambda: None).__code__:
        pytest.skip('s2fft is not installed')
