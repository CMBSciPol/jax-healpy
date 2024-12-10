import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import jax_healpy as hp
from jax_healpy.pixelfunc import MAX_NSIDE


@pytest.mark.parametrize(
    'nside, x, y, expected_fpix',
    [
        (1, 0, 0, 0),
        (2, 0, 0, 0),
        (2, 1, 1, 3),
        (4, 1, 2, 9),
        (4, 0, 3, 10),
        (4, 3, 0, 5),
        (MAX_NSIDE, MAX_NSIDE - 1, MAX_NSIDE - 1, MAX_NSIDE**2 - 1),
    ],
)
def test_xy_to_fpix(nside, x, y, expected_fpix):
    fpix = hp.pixelfunc._xy2fpix(nside, x, y)
    assert_array_equal(fpix, expected_fpix)


@pytest.mark.parametrize(
    'nside, fpix, expected_x, expected_y',
    [
        (1, 0, 0, 0),
        (2, 0, 0, 0),
        (2, 3, 1, 1),
        (4, 9, 1, 2),
        (4, 10, 0, 3),
        (4, 5, 3, 0),
        (MAX_NSIDE, MAX_NSIDE**2 - 1, MAX_NSIDE - 1, MAX_NSIDE - 1),
    ],
)
def test_fpix_to_xy(nside, fpix, expected_x, expected_y):
    x, y = hp.pixelfunc._fpix2xy(nside, fpix)
    assert_array_equal(x, expected_x)
    assert_array_equal(y, expected_y)


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


@pytest.mark.parametrize('func_name', ['ring2nest', 'nest2ring'])
def test_ring_nest_dtypes(func_name: str):
    small_nside = 512
    large_nside = 16384
    pix32 = jnp.zeros(10, dtype=jnp.int32)
    pix64 = jnp.zeros(10, dtype=jnp.int64)
    func = getattr(hp, func_name)
    assert func(small_nside, pix32).dtype == jnp.int32  # input is int32, so should be output
    assert func(small_nside, pix64).dtype == jnp.int64  # input is int64, so should be output
    assert func(large_nside, pix32).dtype == jnp.int64  # nside is large so output is int64
    assert func(large_nside, pix64).dtype == jnp.int64  # nside is large so output is int64


@pytest.mark.parametrize(
    'order', list(range(29)) + [pytest.param(30, marks=pytest.mark.xfail(reason='overflow somewhere?'))]
)
def test_roundtrip_max_pixel(order):
    nside = hp.order2nside(order)
    npix = hp.nside2npix(nside)
    max_pix = npix - 1
    assert hp.ring2nest(nside, hp.nest2ring(nside, max_pix)) == max_pix
    assert hp.nest2ring(nside, hp.ring2nest(nside, max_pix)) == max_pix


@pytest.mark.parametrize('r2n, n2r', [(True, False), (False, True)])
def test_reorder_base_pixels(r2n, n2r):
    map_in = jnp.arange(12)
    map_out = hp.reorder(map_in, r2n=r2n, n2r=n2r)
    assert_array_equal(map_out, map_in)


def test_r2n():
    map_in = jnp.arange(48)
    map_out = hp.reorder(map_in, r2n=True)
    assert_array_equal(map_out, [13,  5,  4,  0, 15,  7,  6,  1, 17,  9,  8,  2, 19, 11, 10,  3, 28,
                                 20, 27, 12, 30, 22, 21, 14, 32, 24, 23, 16, 34, 26, 25, 18, 44, 37,
                                 36, 29, 45, 39, 38, 31, 46, 41, 40, 33, 47, 43, 42, 35])  # fmt: skip


def test_n2r():
    map_in = jnp.arange(48)
    map_out = hp.reorder(map_in, n2r=True)
    assert_array_equal(map_out, [ 3,  7, 11, 15,  2,  1,  6,  5, 10,  9, 14, 13, 19,  0, 23,  4, 27,
                                  8, 31, 12, 17, 22, 21, 26, 25, 30, 29, 18, 16, 35, 20, 39, 24, 43,
                                 28, 47, 34, 33, 38, 37, 42, 41, 46, 45, 32, 36, 40, 44])  # fmt: skip


@pytest.mark.parametrize('r2n, n2r', [(True, False), (False, True)])
@pytest.mark.parametrize('shape', [(12,), (3, 12), (2, 3, 12)])
def test_reorder_shape(shape: tuple[int], r2n: bool, n2r: bool):
    map_out = hp.reorder(jnp.zeros(shape), r2n=r2n, n2r=n2r)
    assert map_out.shape == shape


@pytest.mark.parametrize('order', range(8))
@pytest.mark.parametrize('batch', [(), (1,)])
def test_reorder_roundtrip(order: int, batch: tuple[int]):
    npix = hp.order2npix(order)
    map_in = jr.uniform(jr.key(1143), batch + (npix,))
    assert_array_equal(hp.reorder(hp.reorder(map_in, r2n=True), n2r=True), map_in)
