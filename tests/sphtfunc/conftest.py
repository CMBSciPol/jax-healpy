from collections.abc import Callable
from pathlib import Path
from typing import Any

import healpy as hp
import jax
import numpy as np
import pytest

s2fft = pytest.importorskip('s2fft')

from s2fft.sampling.s2_samples import flm_2d_to_hp  # noqa: E402
from s2fft.utils import signal_generator  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_jax_state():
    """Clear JAX caches before every sphtfunc transform test.

    The s2fft-backed transforms accumulate many compiled XLA executables across a
    session, and stale compiled state nondeterministically corrupts otherwise-correct
    results -- NaNs in near-zero high-ell spin coefficients, and a constant monopole
    offset in scalar ``alm2map`` -- so tests pass in isolation but fail when run after
    other transform tests. This affects scalar transforms too (not only pol/spin), and
    clearing only some tests leaves a bad intermediate cache state that corrupts the
    next one; clearing before every test -- the remedy documented in CLAUDE.md --
    makes the suite order-independent. 64-bit precision is re-asserted because s2fft
    accuracy degrades badly without it. This recompiles per test, so the suite is
    slower but correct.
    """
    jax.config.update('jax_enable_x64', True)
    jax.clear_caches()
    yield


# JIT time for s2fft is very slow, so drop 128
@pytest.fixture(scope='session', params=[32, 64])
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


@pytest.fixture(scope='session', params=['big_E', 'big_B', 'equal'])
def eb_balance(request) -> str:
    """E/B amplitude balance for the polarized TEB fixtures.

    Three cases so the polarization tests actually constrain *both* E and B: with a
    single E-dominant spectrum a B-only error (e.g. a sign or normalization bug) is
    invisible because B is too small for the tolerances to bite. ``big_B`` puts B at
    the amplitude E normally has, and ``equal`` makes them comparable.
    """
    return request.param


@pytest.fixture(scope='session')
def cla_tqu(cla: np.ndarray, eb_balance: str) -> np.ndarray:
    # (TT, EE, BB, TE) in healpy new=True ordering, scaled from TT. The largest of EE/BB
    # is kept at 0.1*TT (the level E had originally) so the existing comparison tolerances
    # remain valid for correct code, while B becomes a real signal in big_B/equal. TE is
    # kept below sqrt(TT*EE) so the TT/EE covariance stays positive-definite.
    cl_TT = cla
    if eb_balance == 'big_E':  # E dominant, B negligible (the original balance)
        cl_EE, cl_BB, cl_TE = 0.1 * cl_TT, 0.01 * cl_TT, 0.05 * cl_TT
    elif eb_balance == 'big_B':  # B dominant, E small
        cl_EE, cl_BB, cl_TE = 0.01 * cl_TT, 0.1 * cl_TT, 0.01 * cl_TT
    else:  # 'equal' -> E == B
        cl_EE, cl_BB, cl_TE = 0.05 * cl_TT, 0.05 * cl_TT, 0.02 * cl_TT
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


@pytest.fixture
def pol_nside() -> int:
    """Small, fixed nside for the polarization/ensemble tests (keeps them fast)."""
    return 16


@pytest.fixture
def pol_lmax(pol_nside: int) -> int:
    """Supported s2fft band-limit for ``pol_nside`` (2 * nside - 1)."""
    return 2 * pol_nside - 1


@pytest.fixture
def power_law_cl(pol_lmax):
    """Factory for a simple band-limited power-law C_l, ``1 / (ell + 10)^2``."""
    ell = np.arange(pol_lmax + 1)
    return 1.0 / (ell + 10.0) ** 2


@pytest.fixture
def synthetic_teb(pol_lmax):
    """Factory for reproducible TT, EE, BB, TE spectra (healpy 'new' ordering).

    Monopole/dipole are zero (as for the CMB) so the low-ell covariance blocks
    are singular, exercising the matrix-square-root path. TE is kept well below
    sqrt(TT*EE) so the field covariance stays positive semi-definite.
    """
    ell = np.arange(pol_lmax + 1)
    tt = np.zeros(pol_lmax + 1)
    tt[2:] = 1.0 / (ell[2:] + 10.0) ** 2
    return np.array([tt, 0.1 * tt, 0.01 * tt, 0.05 * tt])


@pytest.fixture
def flm_generator() -> Callable[[...], np.ndarray]:
    # Own RNG seeded per test so the generated coefficients are reproducible and
    # independent of test ordering/selection (a shared session RNG would make the
    # data depend on what ran before).
    rng = np.random.RandomState(0)

    def generate_flm(L: int, healpy_ordering: bool = False, **keywords: Any) -> np.ndarray:
        flm = signal_generator.generate_flm(rng, L, **keywords)
        if healpy_ordering:
            flm = flm_2d_to_hp(flm, L)
        return flm

    return generate_flm


@pytest.fixture
def flm_generator_batched() -> Callable[[...], np.ndarray]:
    # Import s2fft (and indirectly numpy) locally to avoid
    # `RuntimeWarning: numpy.ndarray size changed` when importing at module level
    # import s2fft as s2f
    # from s2fft.utils import signal_generator
    numpy_rng_1 = np.random.RandomState(1)
    numpy_rng_2 = np.random.RandomState(2)

    def generate_flm_1(L: int, healpy_ordering: bool = False, **keywords: Any) -> np.ndarray:
        flm = signal_generator.generate_flm(numpy_rng_1, L, **keywords)
        if healpy_ordering:
            flm = flm_2d_to_hp(flm, L)
        return flm

    def generate_flm_2(L: int, healpy_ordering: bool = False, **keywords: Any) -> np.ndarray:
        flm = signal_generator.generate_flm(numpy_rng_2, L, **keywords)
        if healpy_ordering:
            flm = flm_2d_to_hp(flm, L)
        return flm

    return generate_flm_1, generate_flm_2
