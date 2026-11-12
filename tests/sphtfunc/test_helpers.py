"""The numpy helpers of the tests are checked against healpy and against the s2fft functions they replace."""

import healpy as hp
import numpy as np
import pytest

from .helpers import METHODS, flm_2d_to_hp, flm_hp_to_2d, generate_flm, hp_indices, requires_s2fft

# These tests do not transform anything: they are run once, not once per default method.
pytestmark = pytest.mark.parametrize('default_method', METHODS[:1], indirect=True)

BAND_LIMITS = [1, 2, 3, 8, 33]


@pytest.mark.parametrize('L', BAND_LIMITS)
def test_hp_indices(L: int) -> None:
    expected_ell, expected_m = hp.Alm.getlm(L - 1)
    ell, m = hp_indices(L)
    np.testing.assert_array_equal(ell, expected_ell)
    np.testing.assert_array_equal(m, expected_m)


@pytest.mark.parametrize('L', BAND_LIMITS)
def test_orderings_roundtrip(L: int) -> None:
    """The alms of a real field are recovered after a conversion to the healpy ordering and back."""
    flm = generate_flm(np.random.RandomState(L), L, reality=True)
    flm_hp = flm_2d_to_hp(flm, L)
    assert flm_hp.shape == (hp.Alm.getsize(L - 1),)
    ell, m = hp.Alm.getlm(L - 1)
    np.testing.assert_array_equal(flm_hp, flm[ell, L - 1 + m])
    np.testing.assert_array_equal(flm_hp_to_2d(flm_hp, L), flm)


@requires_s2fft
@pytest.mark.parametrize('L', BAND_LIMITS)
def test_orderings_match_s2fft(L: int) -> None:
    from s2fft.sampling import s2_samples

    flm = generate_flm(np.random.RandomState(L), L, reality=True)
    flm_hp = flm_2d_to_hp(flm, L)
    np.testing.assert_array_equal(flm_hp, s2_samples.flm_2d_to_hp(flm, L))
    np.testing.assert_array_equal(flm_hp_to_2d(flm_hp, L), s2_samples.flm_hp_to_2d(flm_hp, L))


@requires_s2fft
@pytest.mark.parametrize('L', BAND_LIMITS)
@pytest.mark.parametrize('spin', [0, 1, -2])
@pytest.mark.parametrize('reality', [False, True])
def test_generate_flm_matches_s2fft(L: int, spin: int, reality: bool) -> None:
    """The random alms are those that s2fft draws from the same generator."""
    from s2fft.utils import signal_generator

    if abs(spin) > L:
        pytest.skip('The band-limit is too small for this spin.')
    expected = signal_generator.generate_flm(np.random.RandomState(L), L, spin=spin, reality=reality)
    actual = generate_flm(np.random.RandomState(L), L, spin=spin, reality=reality)
    np.testing.assert_array_equal(actual, expected)
