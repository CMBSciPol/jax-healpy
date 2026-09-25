import os
from pathlib import Path

import pytest


@pytest.fixture(scope='session', autouse=True)
def default_x64():
    """Default the whole test session to 64-bit precision.

    jax-healpy no longer enables x64 on import, but most tests check numerical
    accuracy against healpy at float64 tolerances. Tests that exercise both
    precisions use the parametrized ``x64`` fixture, whose context manager
    overrides this default within the test.
    """
    import jax

    with jax.enable_x64(True):
        yield


@pytest.fixture(scope='session', autouse=True)
def compilation_cache():
    """Persist XLA executables across runs, so a re-run pays for tracing but not compilation.

    Most of the suite's wall time is XLA compilation of the s2fft transforms, and the same
    executables come back run after run. JAX keys each entry on the HLO and on the
    jaxlib/backend version, so a stale entry is a miss, not a wrong answer. The defaults skip
    anything that compiles in under a second or is under a few kilobytes, hence the thresholds.

    Set ``JAX_HEALPY_TEST_NO_COMPILATION_CACHE`` to run against cold compilation instead.
    """
    import jax

    if os.environ.get('JAX_HEALPY_TEST_NO_COMPILATION_CACHE'):
        return
    cache_dir = os.environ.get('JAX_HEALPY_TEST_COMPILATION_CACHE_DIR')
    jax.config.update(
        'jax_compilation_cache_dir',
        cache_dir or str(Path(__file__).parents[1] / '.pytest_cache' / 'jax'),
    )
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)


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
