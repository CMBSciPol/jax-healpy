import healpy
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import jax_healpy as hp


@pytest.mark.parametrize('nest', [False, True])
@pytest.mark.parametrize('nside', [1, 16, 256, 8388608])
def test_pix2loc_matches_healpy(nside: int, nest: bool) -> None:
    npix = hp.nside2npix(nside)
    pixels = np.unique(np.linspace(0, npix - 1, 1000).astype(np.int64))
    z, sin_theta, phi = hp.pix2loc(nside, pixels, nest=nest)
    # sin(theta) from the vector, which keeps full precision near both poles
    x, y, expected_z = healpy.pix2vec(nside, pixels, nest=nest)
    assert_allclose(z, expected_z, rtol=1e-15, atol=1e-15)
    assert_allclose(sin_theta, np.hypot(x, y), rtol=1e-14)
    assert_allclose(phi, healpy.pix2ang(nside, pixels, nest=nest)[1], rtol=1e-15)


@pytest.mark.parametrize('nest', [False, True])
@pytest.mark.parametrize('nside', [1, 16, 256, 8388608])
def test_loc2pix_pix2loc_roundtrip(nside: int, nest: bool) -> None:
    npix = hp.nside2npix(nside)
    pixels = np.unique(np.linspace(0, npix - 1, 1000).astype(np.int64))
    assert_array_equal(hp.loc2pix(nside, *hp.pix2loc(nside, pixels, nest=nest), nest=nest), pixels)


@pytest.mark.parametrize('nest', [False, True])
@pytest.mark.parametrize('nside', [1, 16, 256, 8388608])
def test_loc2pix_matches_healpy(nside: int, nest: bool) -> None:
    rng = np.random.default_rng(0)
    # include directions very close to the poles, where sin(theta) matters
    theta = np.concatenate(
        [rng.uniform(0, np.pi, 1000), rng.uniform(0, 1e-6, 10000), np.pi - rng.uniform(0, 1e-6, 10000)]
    )
    phi = rng.uniform(0, 2 * np.pi, theta.size)
    actual = hp.loc2pix(nside, jnp.cos(theta), jnp.sin(theta), phi, nest=nest)
    assert_array_equal(actual, healpy.ang2pix(nside, theta, phi, nest=nest))
