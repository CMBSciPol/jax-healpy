"""Random alms, conversions between their orderings and markers for the tests of the transforms.

The numpy functions reproduce those of s2fft that the tests used to rely on, with the same random
draws, so that the tests also run when only ducc0 is installed:

- ``generate_flm``: ``s2fft.utils.signal_generator.generate_flm``
- ``flm_hp_to_2d``: ``s2fft.sampling.s2_samples.flm_hp_to_2d``
- ``flm_2d_to_hp``: ``s2fft.sampling.s2_samples.flm_2d_to_hp``
"""

import importlib.util

import numpy as np
import pytest

DUCC0_IS_INSTALLED = importlib.util.find_spec('ducc0') is not None
S2FFT_IS_INSTALLED = importlib.util.find_spec('s2fft') is not None

requires_ducc0 = pytest.mark.skipif(not DUCC0_IS_INSTALLED, reason='ducc0 is not installed')
requires_s2fft = pytest.mark.skipif(not S2FFT_IS_INSTALLED, reason='s2fft is not installed')

# The transform backends of the libraries that are installed
METHODS = [method for method, installed in [('jax', S2FFT_IS_INSTALLED), ('ducc0', DUCC0_IS_INSTALLED)] if installed]


def hp_indices(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the degree and the order of each alm of the healpy ordering, for the band-limit L = lmax + 1.

    healpy stores the alms of m >= 0 order by order: (0, 0), (1, 0), ..., (L-1, 0), (1, 1), ..., (L-1, L-1).
    """
    ell = np.concatenate([np.arange(m, L) for m in range(L)])
    m = np.concatenate([np.full(L - m, m) for m in range(L)])
    return ell, m


def flm_hp_to_2d(flm_hp: np.ndarray, L: int) -> np.ndarray:
    """Convert the alms of a real field from the healpy ordering to the 2-D one, of shape (L, 2L-1).

    The coefficients of m < 0 are those of a real field: a_{l,-m} = (-1)^m conj(a_{l,m}).
    """
    flm_hp = np.asarray(flm_hp)
    if flm_hp.ndim != 1:
        raise ValueError('The alms in the healpy ordering are not flat.')
    ell, m = hp_indices(L)
    flm_2d = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    flm_2d[ell, L - 1 - m] = np.where(m % 2 == 0, 1, -1) * np.conj(flm_hp)
    flm_2d[ell, L - 1 + m] = flm_hp
    return flm_2d


def flm_2d_to_hp(flm_2d: np.ndarray, L: int) -> np.ndarray:
    """Convert alms from the 2-D ordering to the healpy one, which drops the coefficients of m < 0."""
    ell, m = hp_indices(L)
    return np.asarray(flm_2d)[ell, L - 1 + m].astype(np.complex128)


def generate_flm(rng: np.random.RandomState, L: int, spin: int = 0, reality: bool = False) -> np.ndarray:
    """Draw random alms in the 2-D ordering, from a standard normal distribution.

    The coefficients of m = 0 are real and those of ell < abs(spin) are zero. With ``reality``, the
    coefficients of m < 0 are those of a real field; otherwise, they are drawn independently.
    """
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    min_ell = abs(spin)
    flm[min_ell:L, L - 1] = rng.standard_normal(L - min_ell)
    # The coefficients of m > 0, ordered by increasing degree then by increasing order
    ell, m = np.tril_indices(L, k=-1)
    m = m + 1
    ell, m = ell[ell >= min_ell], m[ell >= min_ell]
    flm[ell, L - 1 + m] = rng.standard_normal(m.size) + 1j * rng.standard_normal(m.size)
    if reality:
        flm[ell, L - 1 - m] = np.where(m % 2 == 0, 1, -1) * flm[ell, L - 1 + m].conj()
    else:
        flm[ell, L - 1 - m] = rng.standard_normal(m.size) + 1j * rng.standard_normal(m.size)
    return flm
