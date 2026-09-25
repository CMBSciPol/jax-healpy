import jax.numpy as jnp
import numpy as np
import pytest

import jax_healpy as hp


@pytest.mark.parametrize(
    'nside, ok_ring, ok_nest',
    [
        (1, True, True),
        (16, True, True),
        (13, True, False),
        (16.0, True, True),
        (2**29, True, True),
        (np.int32(8), True, True),
        (np.array(8), True, True),
        (jnp.array(8), True, True),
        (0, False, False),
        (-16, False, False),
        (16.5, False, False),
        (2**30, False, False),
        (np.inf, False, False),
        (np.nan, False, False),
        (True, False, False),
    ],
)
def test_isnsideok_scalar(nside, ok_ring: bool, ok_nest: bool) -> None:
    assert hp.isnsideok(nside) is ok_ring
    assert hp.isnsideok(nside, nest=True) is ok_nest


def test_isnsideok_array() -> None:
    nside = [1, 2, 3, 16.0, 16.5, -4, 0, np.inf, np.nan, 2**40]
    np.testing.assert_array_equal(
        hp.isnsideok(nside), [True, True, True, True, False, False, False, False, False, False]
    )
    np.testing.assert_array_equal(
        hp.isnsideok(nside, nest=True), [True, True, False, True, False, False, False, False, False, False]
    )


@pytest.mark.parametrize(
    'npix, ok',
    [(12, True), (768, True), (12.0, True), (1002, False), (0, False), (-12, False), (np.inf, False), (np.nan, False)],
)
def test_isnpixok_scalar(npix, ok: bool) -> None:
    assert hp.isnpixok(npix) is ok


def test_isnpixok_array() -> None:
    np.testing.assert_array_equal(hp.isnpixok([12, 768, 1002, 0, -12]), [True, True, False, False, False])


@pytest.mark.parametrize('nside', [0, -1, 2.5, np.nan, True])
def test_nside2npix_rejects_invalid_nside(nside) -> None:
    with pytest.raises(ValueError, match='not a valid nside'):
        hp.nside2npix(nside)


def test_nside2npix_npix2nside_roundtrip() -> None:
    for nside in [1, 3, 7, 16, 2**29]:
        assert hp.npix2nside(hp.nside2npix(nside)) == nside


def test_npix2nside_rejects_empty_map() -> None:
    with pytest.raises(ValueError, match='Wrong pixel number'):
        hp.npix2nside(0)
