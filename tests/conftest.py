from pathlib import Path

import healpy as hp
import numpy as np
import pytest


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


@pytest.fixture(scope='session')
def cla_tqu(cla: np.ndarray) -> np.ndarray:
    # (TT, EE, BB, TE) in healpy new=True ordering. EE/BB/TE are scaled from TT to
    # produce a reproducible non-trivial TEB input; TE is kept well below sqrt(TT*EE)
    # so the covariance stays positive-definite.
    cl_TT = cla
    cl_EE = 0.1 * cl_TT
    cl_BB = 0.01 * cl_TT
    cl_TE = 0.05 * cl_TT
    return np.array([cl_TT, cl_EE, cl_BB, cl_TE])


@pytest.fixture(scope='session')
def synthesized_tqu_map(cla_tqu: np.ndarray, nside: int) -> np.ndarray:
    lmax = 3 * nside - 1
    seed = 12345
    old_state = np.random.get_state()
    np.random.seed(seed)
    result = hp.synfast(
        cla_tqu,
        nside,
        lmax=lmax,
        pixwin=False,
        fwhm=0.0,
        pol=True,
        new=True,
    )
    np.random.set_state(old_state)
    return np.asarray(result)  # shape (3, npix)
