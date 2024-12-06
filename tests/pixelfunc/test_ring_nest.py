import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import jax_healpy as hp


@pytest.mark.parametrize('order', range(30))
@pytest.mark.parametrize('nest', [True, False])
def test_pix_to_xyf_to_pix(order: int, nest: bool) -> None:
    nside = hp.order2nside(order)
    npix = hp.nside2npix(nside)
    maxpix = 1_000
    if npix <= maxpix:
        # up to nside=4, test all pixels
        pix = jnp.arange(npix)
    else:
        # otherwise only test a random subset
        # not using jr.choice(replace=False) because it would be slow for large npix
        # it is not a problem if some pixels are tested twice
        pix = jr.randint(jr.key(1234), shape=(maxpix,), minval=0, maxval=npix)
    xyf = hp.pix2xyf(nside, pix, nest=nest)
    ppix = hp.xyf2pix(nside, *xyf, nest=nest)
    assert_array_equal(ppix, pix)


@pytest.mark.parametrize(
    'nside, x, y, f, expected_pix',
    [
        (16, 8, 8, 4, 1440),
        (16, [8, 8, 8, 15, 0], [8, 8, 7, 15, 0], [4, 0, 5, 0, 8], [1440, 427, 1520, 0, 3068]),
    ],
)
def test_xyf2pix(nside: int, x: int, y: int, f: int, expected_pix: int):
    assert_array_equal(hp.xyf2pix(nside, x, y, f), expected_pix)


@pytest.mark.parametrize(
    'nside, ipix, expected_xyf',
    [
        (16, 1440, ([8], [8], [4])),
        (16, [1440, 427, 1520, 0, 3068], ([8, 8, 8, 15, 0], [8, 8, 7, 15, 0], [4, 0, 5, 0, 8])),
        pytest.param(
            (1, 2, 4, 8),
            11,
            ([0, 1, 3, 7], [0, 0, 2, 6], [11, 3, 3, 3]),
            marks=pytest.mark.xfail(reason='nside must be an int'),
        ),
    ],
)
def test_pix2xyf(nside, ipix, expected_xyf):
    x, y, f = hp.pix2xyf(nside, ipix)
    assert_array_equal(x, expected_xyf[0])
    assert_array_equal(y, expected_xyf[1])
    assert_array_equal(f, expected_xyf[2])


@pytest.mark.parametrize(
    'nside, ipix_ring, expected_ipix_nest',
    [
        (16, 1504, 1130),
        (2, np.arange(10), [3, 7, 11, 15, 2, 1, 6, 5, 10, 9]),
        pytest.param(
            [1, 2, 4, 8],
            11,
            [11, 13, 61, 253],
            marks=pytest.mark.xfail(reason='nside must be an int'),
        ),
    ],
)
def test_ring2nest(nside: int, ipix_ring: int, expected_ipix_nest: int):
    assert_array_equal(hp.ring2nest(nside, ipix_ring), expected_ipix_nest)


@pytest.mark.parametrize(
    'nside, ipix_nest, expected_ipix_ring',
    [
        (16, 1130, 1504),
        (2, np.arange(10), [13, 5, 4, 0, 15, 7, 6, 1, 17, 9]),
        pytest.param(
            [1, 2, 4, 8],
            11,
            [11, 2, 12, 211],
            marks=pytest.mark.xfail(reason='nside must be an int'),
        ),
    ],
)
def test_nest2ring(nside: int, ipix_nest: int, expected_ipix_ring: int):
    assert_array_equal(hp.nest2ring(nside, ipix_nest), expected_ipix_ring)
