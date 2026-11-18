import gc
from pathlib import Path

import pytest


@pytest.fixture(scope='module', autouse=True)
def clear_jax_caches():
    """Drop the programs compiled by JAX after each test module.

    JAX keeps in memory every program it compiles, until the end of the process: over the whole
    suite, each xdist worker would grow by several GB. A program that a later module needs again
    is compiled again, or read back from the persistent compilation cache when
    ``JAX_COMPILATION_CACHE_DIR`` is set, as in the CI.
    """
    yield
    import jax

    jax.clear_caches()
    gc.collect()


@pytest.fixture(params=[False, True], ids=['x32', 'x64'])
def x64(request: pytest.FixtureRequest):
    """Run the requesting test under both 32-bit and 64-bit JAX precision.

    Use only where 32- vs 64-bit behavior genuinely differs (e.g. integer pixel
    dtype / overflow), not for float64 accuracy assertions.
    """
    import jax

    with jax.enable_x64(request.param):
        yield request.param


@pytest.fixture(scope='session')
def data_path() -> Path:
    return Path(__file__).parent / 'data'
