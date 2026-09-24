import healpy
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import jax_healpy as hp


@pytest.mark.parametrize('nside', [1, 16, 256, 8388608])
def test_pix2loc_matches_healpy(nside: int) -> None:
    npix = hp.nside2npix(nside)
    pixels = np.unique(np.linspace(0, npix - 1, 1000).astype(np.int64))
    z, sin_theta, phi = hp.pix2loc(nside, pixels)
    # sin(theta) from the vector, which keeps full precision near both poles
    x, y, expected_z = healpy.pix2vec(nside, pixels)
    assert_allclose(z, expected_z, rtol=1e-15, atol=1e-15)
    assert_allclose(sin_theta, np.hypot(x, y), rtol=1e-14)
    assert_allclose(phi, healpy.pix2ang(nside, pixels)[1], rtol=1e-15)


@pytest.mark.parametrize('nside', [1, 16, 256, 8388608])
def test_loc2pix_pix2loc_roundtrip(nside: int) -> None:
    npix = hp.nside2npix(nside)
    pixels = np.unique(np.linspace(0, npix - 1, 1000).astype(np.int64))
    assert_array_equal(hp.loc2pix(nside, *hp.pix2loc(nside, pixels)), pixels)


@pytest.mark.parametrize('nside', [1, 16, 256, 8388608])
def test_loc2pix_matches_healpy(nside: int) -> None:
    rng = np.random.default_rng(0)
    # include directions very close to the poles, where sin(theta) matters
    theta = np.concatenate(
        [rng.uniform(0, np.pi, 1000), rng.uniform(0, 1e-6, 10000), np.pi - rng.uniform(0, 1e-6, 10000)]
    )
    phi = rng.uniform(0, 2 * np.pi, theta.size)
    actual = hp.loc2pix(nside, jnp.cos(theta), jnp.sin(theta), phi)
    assert_array_equal(actual, healpy.ang2pix(nside, theta, phi))


def test_nest_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        hp.pix2loc(16, 0, nest=True)
    with pytest.raises(NotImplementedError):
        hp.loc2pix(16, 1.0, 0.0, 0.0, nest=True)
