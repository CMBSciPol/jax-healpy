from pathlib import Path

import jax
import numpy as np
import pytest


@pytest.fixture(scope='session', autouse=True)
def default_x64():
    """Default the whole test session to 64-bit precision.

    jax-healpy no longer enables x64 on import, but most tests check numerical
    accuracy against healpy at float64 tolerances. Tests that exercise both
    precisions use the parametrized ``x64`` fixture, whose context manager
    overrides this default within the test.
    """
    with jax.enable_x64(True):
        yield


@pytest.fixture(params=[False, True], ids=['x32', 'x64'])
def x64(request: pytest.FixtureRequest):
    """Run the requesting test under both 32-bit and 64-bit JAX precision.

    Use only where 32- vs 64-bit behavior genuinely differs (e.g. integer pixel
    dtype / overflow), not for float64 accuracy assertions.
    """
    with jax.enable_x64(request.param):
        yield request.param


@pytest.fixture(scope='session')
def numpy_rng() -> np.random.RandomState:
    seed = 0
    return np.random.RandomState(seed)


@pytest.fixture(scope='session')
def data_path() -> Path:
    return Path(__file__).parent / 'data'
