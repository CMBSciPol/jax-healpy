import numpy as np
import pytest
from numpy.testing import assert_allclose

import jax_healpy as hp


@pytest.mark.parametrize(
    'z, iphi',
    [
        (1, 0),
        (0.999, 0),
        (0.999, 1),
        (0.999, 2),
        (0.999, 3),
        (0.999, 4),
        (0.98, 0),
        (0.98, 1),
        (0.98, 2),
        (0.98, 3),
        (0.98, 4),
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (-0.98, 0),
        (-0.98, 1),
        (-0.98, 2),
        (-0.98, 3),
        (-0.98, 4),
        (-0.999, 0),
        (-0.999, 1),
        (-0.999, 2),
        (-0.999, 3),
        (-0.999, 4),
        (-1, 0),
        (-1, 1),
        (-1, 2),
        (-1, 3),
        (-1, 4),
    ],
)
@pytest.mark.parametrize('lonlat', [False, True])
def test_ang2vec2ang(z: float, iphi: int, lonlat: bool) -> None:
    # test that we get the same results as healpy
    theta = np.arccos(z)
    phi = iphi / 5 * 2 * np.pi
    actual_theta, actual_phi = hp.vec2ang(hp.ang2vec(theta, phi, lonlat=lonlat), lonlat=lonlat)
    assert_allclose(actual_theta, theta, rtol=1e-14, atol=1e-15)
    assert_allclose(actual_phi, phi, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize('lonlat', [False, True])
def test_ang2vec_array(lonlat: bool) -> None:
    theta = np.array([np.pi / 4, 3 * np.pi / 4])
    phi = np.array([np.pi / 2, 3 * np.pi / 2])
    vec0 = hp.ang2vec(theta[0], phi[0], lonlat=lonlat)
    assert vec0.shape == (3,)
    vec1 = hp.ang2vec(theta[1], phi[1], lonlat=lonlat)
    vec = hp.ang2vec(theta, phi, lonlat=lonlat)
    assert vec.shape == (2, 3)
    assert_allclose(vec, np.array([vec0, vec1]), rtol=1e-14)


@pytest.mark.parametrize('lonlat', [False, True])
def test_vec2ang_array(lonlat: bool) -> None:
    vec = np.array([[1, 2, 3], [-1, 2, -1]])
    theta0, phi0 = hp.vec2ang(vec[0], lonlat=lonlat)
    assert theta0.shape == ()
    assert phi0.shape == ()
    theta1, phi1 = hp.vec2ang(vec[1], lonlat=lonlat)
    theta, phi = hp.vec2ang(vec, lonlat=lonlat)
    assert theta.shape == (2,)
    assert phi.shape == (2,)
    assert_allclose(theta, np.array([theta0, theta1]), rtol=1e-14)
    assert_allclose(phi, np.array([phi0, phi1]), rtol=1e-14)


def test_vec2ang_multidim_batch():
    """vec2ang should preserve arbitrary batch dimensions."""
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(3, 2, 3))  # batch shape (3, 2), last dim is xyz
    theta, phi = hp.vec2ang(vecs)
    assert theta.shape == (3, 2)
    assert phi.shape == (3, 2)
    # Check against flattened computation
    theta_flat, phi_flat = hp.vec2ang(vecs.reshape(-1, 3))
    assert_allclose(theta.ravel(), theta_flat, rtol=1e-14)
    assert_allclose(phi.ravel(), phi_flat, rtol=1e-14)


def test_ang2vec_multidim_batch():
    """ang2vec should produce (*batch, 3) output for multi-dim inputs."""
    theta = np.array([[0.5, 1.0], [1.5, 2.0], [2.5, 0.3]])  # shape (3, 2)
    phi = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    vec = hp.ang2vec(theta, phi)
    assert vec.shape == (3, 2, 3)
    # Verify element-wise consistency
    for i in range(3):
        for j in range(2):
            vec_ij = hp.ang2vec(theta[i, j], phi[i, j])
            assert_allclose(vec[i, j], vec_ij, rtol=1e-14)


def test_pix2vec_multidim_batch():
    """pix2vec should produce (*batch, 3) output for multi-dim pixel arrays."""
    nside = 8
    pixels = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])  # shape (3, 4)
    vec = hp.pix2vec(nside, pixels)
    assert vec.shape == (3, 4, 3)
    # Verify against flat computation
    vec_flat = hp.pix2vec(nside, pixels.ravel())
    assert_allclose(vec.reshape(-1, 3), vec_flat, rtol=1e-14)


def test_vec2ang_ang2vec_roundtrip_multidim():
    """Round-trip vec2ang(ang2vec(theta, phi)) should preserve multi-dim shapes."""
    theta = np.array([[0.5, 1.0], [1.5, 2.0]])  # (2, 2)
    phi = np.array([[0.1, 1.1], [2.1, 3.1]])
    vec = hp.ang2vec(theta, phi)
    assert vec.shape == (2, 2, 3)
    theta_rt, phi_rt = hp.vec2ang(vec)
    assert theta_rt.shape == (2, 2)
    assert phi_rt.shape == (2, 2)
    assert_allclose(theta_rt, theta, rtol=1e-14, atol=1e-15)
    assert_allclose(phi_rt, phi, rtol=1e-14, atol=1e-15)


def test_get_all_neighbours_multidim_batch():
    """get_all_neighbours should produce (8, *batch) for multi-dim theta/phi."""
    nside = 8
    theta = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 0.3]])  # (2, 3)
    phi = np.array([[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]])
    neighbors = hp.get_all_neighbours(nside, theta, phi)
    assert neighbors.shape == (8, 2, 3)
    # Verify against flat computation
    neighbors_flat = hp.get_all_neighbours(nside, theta.ravel(), phi.ravel())
    assert_allclose(neighbors.reshape(8, -1), neighbors_flat, rtol=1e-14)
