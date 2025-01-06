import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import jax_healpy as hp


@pytest.mark.parametrize(
    'nside, theta, phi, lonlat, expected_p, expected_w',
    [
        (1, 0, None, False, [0, 1, 4, 5], [1.0, 0.0, 0.0, 0.0]),  # phi not specified, theta assumed to be pixel
        (1, 0, None, True, [0, 1, 4, 5], [1.0, 0.0, 0.0, 0.0]),  # lonlat should do nothing because phi is None
        (1, 0, 0, False, [1, 2, 3, 0], [0.25, 0.25, 0.25, 0.25]),  # North pole (co-latitude 0, longitude 0)
        (1, 0, 90, True, [1, 2, 3, 0], [0.25, 0.25, 0.25, 0.25]),  # North pole (longitude 0°, latitude 90°)
        (
            1,
            [0, np.pi / 2 + 1e-15],
            0,
            False,
            [[1, 4], [2, 5], [3, 11], [0, 8]],
            [[0.25, 1.0], [0.25, 0], [0.25, 0], [0.25, 0]],
        ),
    ],
)
@pytest.mark.parametrize('nest', [False, pytest.param(True, marks=pytest.mark.xfail(reason='NEST not implemented'))])
def test_get_interp_weights(nside, theta, phi, nest, lonlat, expected_p, expected_w):
    p, w = hp.get_interp_weights(nside, theta, phi, nest=nest, lonlat=lonlat)
    assert_array_equal(p, expected_p)
    assert_allclose(w, np.array(expected_w), atol=1e-14)
