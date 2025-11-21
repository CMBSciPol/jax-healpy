from pathlib import Path

import healpy as hp
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


@pytest.fixture(scope='session', params=[32, 64, 128])
def nside(request):
    return request.param


@pytest.fixture(scope='session', params=[None, 2, 3])
def lmax(request, nside: int):
    if request.param is None:
        return None  # Let the function use default (3*nside-1)
    return request.param * nside - 1  # Multiply factor by nside


@pytest.fixture(scope='session')
def cla(data_path: Path) -> np.ndarray:
    return hp.read_cl(data_path / 'cl_wmap_band_iqumap_r9_7yr_W_v4_udgraded32_II_lmax64_rmmono_3iter.fits')


@pytest.fixture(scope='session')
def synthesized_map(cla: np.ndarray, nside: int) -> np.ndarray:
    lmax = 3 * nside - 1
    fwhm_deg = 7.0
    seed = 12345
    # Save current random state to avoid polluting global state
    old_state = np.random.get_state()
    np.random.seed(seed)
    result = hp.synfast(
        cla,
        nside,
        lmax=lmax,
        pixwin=False,
        fwhm=np.radians(fwhm_deg),
        new=False,
    )
    # Restore previous random state
    np.random.set_state(old_state)
    return result
