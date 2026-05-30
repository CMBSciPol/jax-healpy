from pathlib import Path

import healpy as hp
import numpy as np
import pytest
from s2fft.sampling.reindex import flm_hp_to_2d_fast

import jax_healpy as jhp

# TODO: test that precompute is faster?     'precompute_temperature_harmonic_transforms',
# 'precompute_polarization_harmonic_transforms',


@pytest.fixture(scope='session')
def cla(data_path: Path) -> np.ndarray:
    return hp.read_cl(data_path / 'cl_wmap_band_iqumap_r9_7yr_W_v4_udgraded32_II_lmax64_rmmono_3iter.fits')


@pytest.fixture(scope='session')
def map1(data_path: Path) -> np.ndarray:
    return hp.read_map(data_path / 'wmap_band_iqumap_r9_7yr_V_v4_udgraded32.fits', field=(0, 1, 2))


@pytest.fixture(scope='session')
def map2(data_path: Path) -> np.ndarray:
    return hp.read_map(data_path / 'wmap_band_iqumap_r9_7yr_W_v4_udgraded32.fits', field=(0, 1, 2))


@pytest.fixture(scope='session')
def synthesized_map(cla: np.ndarray) -> np.ndarray:
    nside = 32
    lmax = 64
    fwhm_deg = 7.0
    seed = 12345
    np.random.seed(seed)
    return hp.synfast(
        cla,
        nside,
        lmax=lmax,
        pixwin=False,
        fwhm=np.radians(fwhm_deg),
        new=False,
    )


@pytest.mark.parametrize('healpy_ordering', [False, True])
@pytest.mark.parametrize('lmax', [None, 64])
@pytest.mark.parametrize('lmax_out', [None, 32])
@pytest.mark.parametrize('nspec', [None, 1, 4, 6])
def test_alm2cl_T(
    synthesized_map: np.ndarray, lmax: int | None, lmax_out: int | None, nspec: int | None, healpy_ordering: bool
) -> None:
    flm_to_test = hp.map2alm(synthesized_map, lmax=lmax, iter=0)

    # Mirror the exact lmax_out/nspec arguments so the comparison is meaningful.
    c_ells_expected = hp.alm2cl(flm_to_test, lmax=lmax, lmax_out=lmax_out, nspec=nspec)

    # Resolve L from the alm size when lmax is not provided.
    resolved_lmax = lmax if lmax is not None else hp.Alm.getlmax(flm_to_test.size)

    if not healpy_ordering:
        flm_to_test = flm_hp_to_2d_fast(flm_to_test, resolved_lmax + 1)

    actual_c_ells = jhp.alm2cl(flm_to_test, lmax=lmax, lmax_out=lmax_out, nspec=nspec, healpy_ordering=healpy_ordering)
    np.testing.assert_allclose(actual_c_ells, c_ells_expected, atol=1e-14)
