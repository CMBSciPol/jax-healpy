# from .._query_disc import boundaries
# from .._pixelfunc import ringinfo, pix2ring, isnsideok
import jax.numpy as jnp
import numpy as np
import pytest

import jax_healpy as hp


def test_nside2npix() -> None:
    assert hp.nside2npix(512) == 3145728
    assert hp.nside2npix(1024) == 12582912


def test_nside2npix_array() -> None:
    assert (hp.nside2npix(np.array([512, 1024])) == np.array([3145728, 12582912])).all()


def test_npix2nside() -> None:
    assert hp.npix2nside(3145728) == 512
    assert hp.npix2nside(12582912) == 1024


@pytest.mark.parametrize('order', range(30))
def test_nside2order_scalar(order: int) -> None:
    assert hp.nside2order(2**order) == order


@pytest.mark.parametrize('order', range(30))
def test_order2nside_scalar(order: int) -> None:
    assert hp.order2nside(order) == 2**order


def test_order2nside_array() -> None:
    o = np.arange(30)
    assert (hp.order2nside(o) == 2**o).all()


def test_nside2resol() -> None:
    np.testing.assert_array_almost_equal(hp.nside2resol(512, arcmin=True), 6.87097282363)
    np.testing.assert_array_almost_equal(hp.nside2resol(1024, arcmin=True), 3.43548641181)


def test_nside2resol_array() -> None:
    np.testing.assert_array_almost_equal(
        hp.nside2resol(np.array([512, 1024]), arcmin=True),
        np.array([6.87097282363, 3.43548641181]),
    )


def test_nside2pixarea() -> None:
    np.testing.assert_array_almost_equal(hp.nside2pixarea(512), 3.9947416351188569e-06)


def test_nside2pixarea_array() -> None:
    np.testing.assert_array_almost_equal(
        hp.nside2pixarea(np.array([512, 512])),
        np.array([3.9947416351188569e-06, 3.9947416351188569e-06]),
    )


@pytest.mark.xfail(reason='max_pixrad not implemented')
def test_max_pixrad() -> None:
    np.testing.assert_array_almost_equal(hp.max_pixrad(512), 2.0870552355e-03)
    np.testing.assert_array_almost_equal(hp.max_pixrad(512, degrees=True), np.rad2deg(2.0870552355e-03))


def test_vec2pix_lonlat(lon0: float, lat0: float) -> None:
    # Need to decrease the precision of the check because deg not radians
    vec = hp.ang2vec(lon0, lat0, lonlat=True)
    lon1, lat1 = hp.vec2ang(vec, lonlat=True)
    np.testing.assert_array_almost_equal(lon1, lon0, decimal=5)
    np.testing.assert_array_almost_equal(lat1, lat0, decimal=5)


def test_get_interp_val_lonlat(theta0: float, phi0: float, lon0: float, lat0: float) -> None:
    m = jnp.arange(12.0)
    val0 = hp.get_interp_val(m, theta0, phi0)
    val1 = hp.get_interp_val(m, lon0, lat0, lonlat=True)
    np.testing.assert_array_almost_equal(val0, val1)


def test_get_interp_weights() -> None:
    p0, w0 = (np.array([0, 1, 4, 5]), np.array([1.0, 0.0, 0.0, 0.0]))

    # phi not specified, theta assumed to be pixel
    p1, w1 = hp.get_interp_weights(1, 0)
    np.testing.assert_array_almost_equal(p0, p1)
    np.testing.assert_array_almost_equal(w0, w1)

    # If phi is not specified, lonlat should do nothing
    p1, w1 = hp.get_interp_weights(1, 0, lonlat=True)
    np.testing.assert_array_almost_equal(p0, p1)
    np.testing.assert_array_almost_equal(w0, w1)

    p0, w0 = (np.array([1, 2, 3, 0]), np.array([0.25, 0.25, 0.25, 0.25]))

    p1, w1 = hp.get_interp_weights(1, 0, 0)
    np.testing.assert_array_almost_equal(p0, p1)
    np.testing.assert_array_almost_equal(w0, w1)

    p1, w1 = hp.get_interp_weights(1, 0, 90, lonlat=True)
    np.testing.assert_array_almost_equal(p0, p1)
    np.testing.assert_array_almost_equal(w0, w1)


@pytest.mark.xfail(reason='get_all_neighbours not implemented')
def test_get_all_neighbours() -> None:
    ipix0 = np.array([8, 4, 0, -1, 1, 6, 9, -1])
    ipix1 = hp.get_all_neighbours(1, np.pi / 2, np.pi / 2)
    ipix2 = hp.get_all_neighbours(1, 90, 0, lonlat=True)
    np.testing.assert_array_almost_equal(ipix0, ipix1)
    np.testing.assert_array_almost_equal(ipix0, ipix2)


@pytest.mark.xfail(reason='fit_dipole not implemented')
def test_fit_dipole() -> None:
    nside = 32
    npix = hp.nside2npix(nside)
    d = [0.3, 0.5, 0.2]
    vec = np.transpose(hp.pix2vec(nside, np.arange(npix)))
    signal = np.dot(vec, d)
    mono, dipole = hp.fit_dipole(signal)
    np.testing.assert_array_almost_equal(mono, 0.0)
    np.testing.assert_array_almost_equal(d[0], dipole[0])
    np.testing.assert_array_almost_equal(d[1], dipole[1])
    np.testing.assert_array_almost_equal(d[2], dipole[2])


@pytest.mark.xfail(reason='boundaries not implemented')
@pytest.mark.parametrize('lg_nside', range(1, 5))
@pytest.mark.parametrize('res', range(1, 50, 7))
@pytest.mark.parametrize('nest', [False, True])
def test_boundaries(lg_nside: int, res: int, nest: bool) -> None:
    """Test whether the boundary shapes look sane"""
    nside = 1 << lg_nside
    for pix in range(hp.nside2npix(nside)):
        num = 4 * res  # Expected number of points
        points = hp.boundaries(nside, pix, res, nest=nest)
        assert points.shape == (3, num)
        dist = np.linalg.norm(points[:, : num - 1] - points[:, 1:])  # distance between points
        assert (dist != 0).all()
        dmin = np.min(dist)
        dmax = np.max(dist)
        assert dmax / dmin <= 2.0


@pytest.mark.xfail(reason='pix2ring not implemented')
@pytest.mark.parametrize('lg_nside', range(1, 5))
@pytest.mark.parametrize('nest', [False, True])
def test_ring(lg_nside: int, nest: bool) -> None:
    nside = 1 << lg_nside
    numPix = hp.nside2npix(nside)
    numRings = 4 * nside - 1  # Expected number of rings
    pix = np.arange(numPix, dtype=np.int64)
    ring = hp.pix2ring(nside, pix, nest=nest)
    assert pix.shape == ring.shape
    assert len(set(ring)) == numRings
    if not nest:
        first = ring[: numPix - 1]
        second = ring[1:]
        assert np.logical_or(first == second, first == second - 1).all()


@pytest.mark.xfail(reason='ud_grade not implemented')
def test_accept_ma_allows_only_keywords() -> None:
    """Test whether the accept_ma wrapper accepts calls using only keywords."""
    ma = np.zeros(12 * 16**2)
    hp.ud_grade(map_in=ma, nside_out=32)


def test_isnsideok() -> None:
    assert hp.isnsideok(nside=1, nest=False)
    assert hp.isnsideok(nside=16, nest=True)

    assert not hp.isnsideok(nside=-16, nest=True)
    assert not hp.isnsideok(nside=-16, nest=False)
    assert not hp.isnsideok(nside=13, nest=True)
