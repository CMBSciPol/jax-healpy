# This file is part of jax-healpy.
# Copyright (C) 2024 CNRS / SciPol developers
#
# jax-healpy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# jax-healpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with jax-healpy. If not, see <https://www.gnu.org/licenses/>.

"""Functions of the HEALPix pixelization.

They convert between the pixels and the positions on the sphere, between the RING and the NESTED
orderings and between the resolution parameters; they interpolate maps, find the neighbours of
pixels and change the resolution of maps.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from jaxtyping import Array, ArrayLike

from ._precision import warn_if_not_x64

__all__ = [
    'pix2ang',
    'ang2pix',
    'pix2xyf',
    'xyf2pix',
    'pix2vec',
    'vec2pix',
    'ang2vec',
    'vec2ang',
    'pix2loc',
    'loc2pix',
    'get_interp_weights',
    'get_interp_val',
    'get_all_neighbours',
    'InterpCenters',
    # 'max_pixrad',
    'nest2ring',
    'ring2nest',
    'reorder',
    'ud_grade',
    'UNSEEN',
    # 'mask_good',
    'mask_bad',
    # 'ma',
    # 'fit_dipole',
    # 'remove_dipole',
    # 'fit_monopole',
    # 'remove_monopole',
    'nside2npix',
    'npix2nside',
    'nside2order',
    'order2nside',
    'order2npix',
    'npix2order',
    'nside2resol',
    'nside2pixarea',
    'isnsideok',
    'isnpixok',
    # 'get_map_size',
    # 'get_min_valid_nside',
    'get_nside',
    'maptype',
    # 'ma_to_array',
]

# We are using 64-bit integer types.
# nside > 2**29 requires extended integer types.
MAX_NSIDE = 1 << 29
UNSEEN = -1.6375e30
r"""Value of the pixels that are not observed, $-1.6375 \times 10^{30}$, as in healpy."""

# HEALPix neighbor finding constants
# These constants implement the exact neighbor-finding algorithm from the original
# HEALPix C++ library (healpix_base.cc) for face boundary transitions

# 8-element offset arrays for x and y directions (SW, W, NW, N, NE, E, SE, S)
# These define the relative positions of the 8 neighbors around any pixel
_NB_XOFFSET = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
_NB_YOFFSET = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)

# Face boundary lookup table (9x12) - handles face transitions for neighbors
# This lookup table maps (nbnum, face) -> new_face when neighbors cross face boundaries
# nbnum encodes the boundary crossing direction, face is the original face (0-11)
# Based on original HEALPix C++ implementation's neighbor finding algorithm
_NB_FACEARRAY = np.array(
    [
        [8, 9, 10, 11, -1, -1, -1, -1, 10, 11, 8, 9],  # S
        [5, 6, 7, 4, 8, 9, 10, 11, 9, 10, 11, 8],  # SE
        [-1, -1, -1, -1, 5, 6, 7, 4, -1, -1, -1, -1],  # E
        [4, 5, 6, 7, 11, 8, 9, 10, 11, 8, 9, 10],  # SW
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # center
        [1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4],  # NE
        [-1, -1, -1, -1, 7, 4, 5, 6, -1, -1, -1, -1],  # W
        [3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7],  # NW
        [2, 3, 0, 1, -1, -1, -1, -1, 0, 1, 2, 3],  # N
    ],
    dtype=np.int32,
)

# Coordinate transformation bits (9x3) - handles x/y swapping across face boundaries
# This lookup table provides bit flags for coordinate transformations when crossing faces
# Bit 1: flip x coordinate, Bit 2: flip y coordinate, Bit 4: swap x and y coordinates
# Index by (nbnum, face >> 2) to get transformation bits for the boundary crossing
_NB_SWAPARRAY = np.array(
    [
        [0, 0, 3],  # S
        [0, 0, 6],  # SE
        [0, 0, 0],  # E
        [0, 0, 5],  # SW
        [0, 0, 0],  # center
        [5, 0, 0],  # NE
        [0, 0, 0],  # W
        [6, 0, 0],  # NW
        [3, 0, 0],  # N
    ],
    dtype=np.int32,
)


def check_theta_valid(theta):
    """
    JIT Compatible check_theta_valid
    Raises exception if theta is not within 0 and pi
    """
    invalid_theta = ~((theta >= 0).all() & (theta <= np.pi + 1e-5).all())

    def _raise_invalid_theta(invalid_theta):
        if invalid_theta:
            raise ValueError('THETA is out of range [0,pi]')

    jax.debug.callback(_raise_invalid_theta, invalid_theta)


def check_nside(nside: int, nest: bool = False) -> None:
    """Raises exception is nside is not valid"""
    warn_if_not_x64()
    if not np.all(isnsideok(nside, nest=nest)):
        requirement = 'a power of 2' if nest else 'a positive integer'
        raise ValueError(f'{nside} is not a valid nside parameter (must be {requirement}, less than 2**30)')


def _pixel_dtype_for(nside: int) -> jnp.dtype:
    """Returns the appropriate dtype for a pixel number given nside"""
    # nside=8192 (2**13) is the largest valid (power-of-2) nside whose pixel
    # indices stay within int32; nside=16384 already overflows in ring/nest
    # conversions, so promote to int64 from there on.
    if nside <= 8192:
        return jnp.int32
    return jnp.int64


def isnsideok(nside: int, nest: bool = False) -> bool:
    """Return whether `nside` is a valid resolution parameter.

    A valid `nside` is a positive integer up to $2^{29}$, and a power of 2 for the NESTED ordering.

    !!! note "Difference from healpy"
        A boolean is not a valid `nside`, and an infinite or NaN value gives `False` instead of
        raising an exception.

    Parameters
    ----------
    nside : int or array-like
        The values to test.
    nest : bool, optional
        If True, also require a power of 2, as the NESTED ordering does.

    Returns
    -------
    bool or numpy.ndarray
        Whether each value is valid: a bool for a scalar, a boolean array otherwise.

    See Also
    --------
    isnpixok

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.isnsideok(13, nest=True)
    False

    >>> hp.isnsideok(13, nest=False)
    True

    >>> hp.isnsideok(32)
    True

    >>> hp.isnsideok([1, 2, 3, 4, 8, 16], nest=True)
    array([ True,  True, False,  True,  True,  True])
    """
    nside = np.asarray(nside)
    is_nside_ok = (
        (nside.dtype != bool) & np.isfinite(nside) & (nside == np.floor(nside)) & (nside > 0) & (nside <= MAX_NSIDE)
    )
    if nest:
        # invalid values are zeroed before the cast to an integer, so that it cannot overflow
        int_nside = np.where(is_nside_ok, nside, 0).astype(np.int64)
        # we use standard bithacks from http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
        is_nside_ok &= (int_nside & (int_nside - 1)) == 0
    return _bool_or_array(is_nside_ok)


def isnpixok(npix: int) -> bool:
    r"""Return whether `npix` is the number of pixels of a map, $12\,n_\mathrm{side}^2$.

    !!! note "Difference from healpy"
        Zero and infinite values are not valid numbers of pixels.

    Parameters
    ----------
    npix : int or array-like
        The values to test.

    Returns
    -------
    bool or numpy.ndarray
        Whether each value is valid: a bool for a scalar, a boolean array otherwise.

    See Also
    --------
    isnsideok

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.isnpixok(12)
    True

    >>> hp.isnpixok(768)
    True

    >>> hp.isnpixok([12, 768, 1002])
    array([ True,  True, False])
    """
    npix = np.asarray(npix)
    # sqrt is only taken on positive sizes, so that it raises no warning on negative ones
    nside = np.sqrt(np.where(npix > 0, npix, 0) / 12.0)
    return _bool_or_array(np.isfinite(npix) & (npix > 0) & (nside == np.floor(nside)))


def _bool_or_array(mask: np.ndarray) -> bool | np.ndarray:
    """Returns a Python bool for a scalar input, and the boolean array otherwise"""
    return bool(mask) if mask.ndim == 0 else mask


def nside2npix(nside: int) -> int:
    r"""Return the number of pixels of a map, $N_\mathrm{pix} = 12\,n_\mathrm{side}^2$.

    !!! note "Difference from healpy"
        An invalid `nside`, such as 0, raises a `ValueError`, whereas healpy returns
        $12\,n_\mathrm{side}^2$ for any input.

    Parameters
    ----------
    nside : int
        The resolution parameter, a positive integer.

    Returns
    -------
    int
        The number of pixels.

    Raises
    ------
    ValueError
        If `nside` is not valid (see [`isnsideok`][]).

    See Also
    --------
    npix2nside

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.nside2npix(8)
    768

    >>> all(hp.nside2npix(nside) == 12 * nside**2 for nside in [2**n for n in range(12)])
    True

    >>> hp.nside2npix(7)
    588

    >>> hp.nside2npix(0)
    Traceback (most recent call last):
        ...
    ValueError: 0 is not a valid nside parameter (must be a positive integer, less than 2**30)
    """
    check_nside(nside)
    return 12 * nside * nside


def npix2nside(npix: int) -> int:
    r"""Return the resolution parameter of a map of `npix` pixels, $n_\mathrm{side} = \sqrt{N_\mathrm{pix} / 12}$.

    !!! note "Difference from healpy"
        An empty map, `npix = 0`, raises a `ValueError` instead of giving `nside = 0`.

    Parameters
    ----------
    npix : int
        The number of pixels.

    Returns
    -------
    int
        The resolution parameter.

    Raises
    ------
    ValueError
        If `npix` is not the number of pixels of a map (see [`isnpixok`][]).

    See Also
    --------
    nside2npix, get_nside

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.npix2nside(768)
    8

    >>> all(hp.npix2nside(12 * nside**2) == nside for nside in [2**n for n in range(12)])
    True

    >>> hp.npix2nside(1000)
    Traceback (most recent call last):
        ...
    ValueError: Wrong pixel number (it is not 12*nside**2)
    """
    if not isnpixok(npix):
        raise ValueError('Wrong pixel number (it is not 12*nside**2)')
    return int(np.sqrt(npix / 12.0))


def nside2order(nside: int) -> int:
    r"""Return the resolution order $k$ of `nside`, such that $n_\mathrm{side} = 2^k$.

    Parameters
    ----------
    nside : int
        The resolution parameter, a power of 2 up to $2^{29}$.

    Returns
    -------
    int
        The resolution order.

    Raises
    ------
    ValueError
        If `nside` is not a power of 2 up to $2^{29}$.

    See Also
    --------
    order2nside

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.nside2order(128)
    7

    >>> all(hp.nside2order(2**o) == o for o in range(30))
    True

    >>> hp.nside2order(7)
    Traceback (most recent call last):
        ...
    ValueError: 7 is not a valid nside parameter (must be a power of 2, less than 2**30)
    """
    check_nside(nside, nest=True)
    return len(f'{nside:b}') - 1


def order2nside(order: int) -> int:
    r"""Return the resolution parameter $n_\mathrm{side} = 2^k$ of the resolution order $k$.

    Parameters
    ----------
    order : int or array-like
        The resolution order, from 0 to 29.

    Returns
    -------
    int or numpy.ndarray
        The resolution parameter.

    Raises
    ------
    ValueError
        If `order` is larger than 29.

    See Also
    --------
    nside2order, order2npix

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> hp.order2nside(7)
    128

    >>> print(hp.order2nside(np.arange(8)))
    [  1   2   4   8  16  32  64 128]

    >>> hp.order2nside(31)
    Traceback (most recent call last):
        ...
    ValueError: 2147483648 is not a valid nside parameter (must be a power of 2, less than 2**30)
    """
    nside = 1 << order
    check_nside(nside, nest=True)
    return nside


def order2npix(order: int) -> int:
    r"""Return the number of pixels $N_\mathrm{pix} = 12 \cdot 4^k$ of the resolution order $k$.

    Parameters
    ----------
    order : int or array-like
        The resolution order, from 0 to 29.

    Returns
    -------
    int or numpy.ndarray
        The number of pixels.

    Raises
    ------
    ValueError
        If `order` is larger than 29.

    See Also
    --------
    npix2order, order2nside

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> hp.order2npix(7)
    196608

    >>> print(hp.order2npix(np.arange(8)))
    [    12     48    192    768   3072  12288  49152 196608]

    >>> hp.order2npix(31)
    Traceback (most recent call last):
        ...
    ValueError: 2147483648 is not a valid nside parameter (must be a power of 2, less than 2**30)
    """
    nside = order2nside(order)
    npix = nside2npix(nside)
    return npix


def npix2order(npix: int) -> int:
    r"""Return the resolution order $k$ of a map of $N_\mathrm{pix} = 12 \cdot 4^k$ pixels.

    Parameters
    ----------
    npix : int
        The number of pixels.

    Returns
    -------
    int
        The resolution order.

    Raises
    ------
    ValueError
        If `npix` is not the number of pixels of a map whose `nside` is a power of 2.

    See Also
    --------
    order2npix, npix2nside

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.npix2order(768)
    3

    >>> all(hp.npix2order(12 * 4**order) == order for order in range(12))
    True

    >>> hp.npix2order(1000)
    Traceback (most recent call last):
        ...
    ValueError: Wrong pixel number (it is not 12*nside**2)
    """
    nside = npix2nside(npix)
    order = nside2order(nside)
    return order


def nside2resol(nside: int, arcmin=False) -> float:
    r"""Return the approximate size of the pixels, the square root of their area.

    All the pixels have the same area, $\Omega_\mathrm{pix} = 4\pi / N_\mathrm{pix}$, but not the
    same shape: $\sqrt{\Omega_\mathrm{pix}}$ is only an approximation of their size.

    Parameters
    ----------
    nside : int
        The resolution parameter, a positive integer.
    arcmin : bool, optional
        If True, return the size in arcminutes, otherwise in radians.

    Returns
    -------
    float
        The approximate size of the pixels.

    Raises
    ------
    ValueError
        If `nside` is not valid (see [`isnsideok`][]).

    See Also
    --------
    nside2pixarea

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.nside2resol(128, arcmin=True)
    np.float64(27.483891294539248)

    >>> hp.nside2resol(256)
    np.float64(0.003997369952915971)

    >>> hp.nside2resol(7)
    np.float64(0.1461895297066412)
    """
    resol = np.sqrt(nside2pixarea(nside))

    if arcmin:
        resol = np.rad2deg(resol) * 60

    return resol


def nside2pixarea(nside: int, degrees=False) -> float:
    r"""Return the area of the pixels, $\Omega_\mathrm{pix} = 4\pi / N_\mathrm{pix}$.

    Parameters
    ----------
    nside : int
        The resolution parameter, a positive integer.
    degrees : bool, optional
        If True, return the area in square degrees, otherwise in steradians.

    Returns
    -------
    float
        The area of the pixels.

    Raises
    ------
    ValueError
        If `nside` is not valid (see [`isnsideok`][]).

    See Also
    --------
    nside2resol

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.nside2pixarea(128, degrees=True)
    np.float64(0.2098234113027917)

    >>> hp.nside2pixarea(256)
    1.5978966540475428e-05

    >>> hp.nside2pixarea(7)
    0.021371378595848933
    """

    pixarea = 4 * np.pi / nside2npix(nside)

    if degrees:
        pixarea = np.rad2deg(np.rad2deg(pixarea))

    return pixarea


def _lonlat2thetaphi(lon: ArrayLike, lat: ArrayLike):
    """Transform longitude and latitude (deg) into co-latitude and longitude (rad)

    Parameters
    ----------
    lon : int or array-like
      Longitude in degrees
    lat : int or array-like
      Latitude in degrees

    Returns
    -------
    theta, phi : float, scalar or array-like
      The co-latitude and longitude in radians
    """
    return np.pi / 2 - jnp.radians(lat), jnp.radians(lon)


def _thetaphi2lonlat(theta, phi):
    """Transform co-latitude and longitude (rad) into longitude and latitude (deg)

    Parameters
    ----------
    theta : int or array-like
      Co-latitude in radians
    phi : int or array-like
      Longitude in radians

    Returns
    -------
    lon, lat : float, scalar or array-like
      The longitude and latitude in degrees
    """
    return jnp.degrees(phi), 90.0 - jnp.degrees(theta)


def maptype(m):
    """Return the number of maps in `m`: 0 for a single map, $n$ for a sequence of $n$ maps.

    Parameters
    ----------
    m : array-like or sequence of array-like
        A map, or a sequence of maps of the same size.

    Returns
    -------
    int
        0 if `m` is a single map, the number of maps otherwise.

    Raises
    ------
    TypeError
        If `m` is a scalar or is empty, if its maps have different sizes, or if their size is not
        the number of pixels of a map (see [`isnpixok`][]).

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> hp.maptype(np.arange(12))
    0
    >>> hp.maptype([np.arange(12), np.arange(12)])
    2
    """
    if not hasattr(m, '__len__'):
        raise TypeError('input map is a scalar')
    if len(m) == 0:
        raise TypeError('input map has length zero')

    try:
        npix = len(m[0])
    except TypeError:
        npix = None

    if npix is not None:
        for mm in m[1:]:
            if len(mm) != npix:
                raise TypeError('input maps have different npix')
        if isnpixok(len(m[0])):
            return len(m)
        else:
            raise TypeError('bad number of pixels')
    else:
        if isnpixok(len(m)):
            return 0
        else:
            raise TypeError('bad number of pixels')


@jit(static_argnames=['nside', 'nest', 'lonlat'])
def ang2pix(
    nside: int,
    theta: ArrayLike,
    phi: ArrayLike,
    nest: bool = False,
    lonlat: bool = False,
) -> Array:
    r"""Return the pixels that contain the points of angular coordinates $(\theta, \phi)$.

    !!! note "Difference from healpy"
        A colatitude outside $[0, \pi]$ gives the pixel -1, instead of raising an exception.
        `nside` is an integer, not an array, and the other arguments are arrays or scalars, not
        lists.

    Parameters
    ----------
    nside : int
        The resolution parameter, a power of 2 for the NESTED ordering.
    theta, phi : float or array-like
        The colatitude $\theta \in [0, \pi]$, from the North pole, and the longitude $\phi$, in
        radians. With `lonlat=True`, the longitude and the latitude, in degrees.
    nest : bool, optional
        If True, use the NESTED ordering, otherwise the RING ordering.
    lonlat : bool, optional
        If True, `theta` and `phi` are the longitude and the latitude, in degrees.

    Returns
    -------
    Array
        The pixels, with the broadcast shape of `theta` and `phi`.

    See Also
    --------
    pix2ang, vec2pix, loc2pix

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> from jax.numpy import pi as π
    >>> hp.ang2pix(16, π/2, 0)
    Array(1440, dtype=int32)

    The points on the boundary between two pixels, such as $(\pi/2, \pi/2)$, are moved off it by
    $10^{-15}$, so that the pixel they fall in does not depend on the rounding:

    >>> print(hp.ang2pix(16, np.array([π/2, π/4, π/2, 0, π]), np.array([0., π/4, π/2 + 1e-15, 0, 0])))
    [1440  427 1520    0 3068]

    >>> print(hp.ang2pix(16, π/2, np.array([0, π/2 + 1e-15])))
    [1440 1520]
    """
    # check_theta_valid(theta)
    if lonlat:
        theta, phi = _lonlat2thetaphi(theta, phi)

    pixels = loc2pix(nside, jnp.cos(theta), jnp.sin(theta), phi, nest=nest)
    return jnp.where((theta < 0) | (theta > np.pi + 1e-5), -1, pixels)


@jit(static_argnames=['nside', 'nest'])
def loc2pix(nside: int, z: ArrayLike, sin_theta: ArrayLike, phi: ArrayLike, nest: bool = False) -> Array:
    r"""Return the pixels that contain the points of coordinates $(z, \sin\theta, \phi)$.

    Same as [`ang2pix`][], but the colatitude $\theta$ is given by its cosine $z$ and its sine,
    which spares the trigonometric functions when they are already at hand, after a rotation for
    example. It is the inverse of [`pix2loc`][]. healpy has no such function.

    Parameters
    ----------
    nside : int
        The resolution parameter, a power of 2 for the NESTED ordering.
    z : float or array-like
        The cosine of the colatitude, in $[-1, 1]$.
    sin_theta : float or array-like
        The sine of the colatitude, in $[0, 1]$. It is only used near the poles, where $|z| > 0.99$
        and where it is more accurate than $\sqrt{1 - z^2}$.
    phi : float or array-like
        The longitude, in radians.
    nest : bool, optional
        If True, use the NESTED ordering, otherwise the RING ordering.

    Returns
    -------
    Array
        The pixels, with the broadcast shape of `z`, `sin_theta` and `phi`.

    See Also
    --------
    pix2loc, ang2pix, vec2pix

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> hp.loc2pix(16, 0.0, 1.0, 0.0)
    Array(1504, dtype=int32)

    >>> print(hp.loc2pix(16, *hp.pix2loc(16, np.array([1440, 427, 1520, 0, 3068]))))
    [1440  427 1520    0 3068]
    """
    check_nside(nside, nest=nest)
    if nest:
        return _loc2pix_nest(nside, z, sin_theta, phi)
    return _zphi2pix_ring(nside, z, sin_theta, phi)


def _zphi2pix_ring(nside: int, z: ArrayLike, sin_theta: ArrayLike, phi: ArrayLike) -> Array:
    tt = jnp.mod(2 * phi / np.pi, 4)
    ipix = jnp.where(
        jnp.abs(z) <= 2 / 3,
        _zphi2pix_equatorial_region_ring(nside, z, sin_theta, tt),
        _zphi2pix_polar_caps_ring(nside, z, sin_theta, tt),
    )
    return ipix


def _zphi2pix_equatorial_region_ring(nside: int, z: ArrayLike, sin_theta: float, tt: ArrayLike) -> Array:
    dt = _pixel_dtype_for(nside)
    ncap = 2 * nside * (nside - 1)
    nl4 = 4 * nside
    jp = (nside * (0.5 + tt - 0.75 * z)).astype(dt)
    jm = (nside * (0.5 + tt + 0.75 * z)).astype(dt)
    ir = nside + 1 + jp - jm
    kshift = 1 - ir & 1  # ir even -> 1, odd -> 0
    t1 = jp + jm - nside + kshift + 1 + nl4 + nl4
    ip = (t1 >> 1) & (nl4 - 1)
    pix = ncap + (ir - 1) * nl4 + ip
    return pix


def _zphi2pix_polar_caps_ring(nside: int, z: ArrayLike, sin_theta: ArrayLike, tt: ArrayLike) -> Array:
    dt = _pixel_dtype_for(nside)
    npixel = nside2npix(nside)
    tp = tt - jnp.floor(tt)
    tmp = _polar_cap_distance(nside, z, sin_theta)
    jp = (tp * tmp).astype(dt)
    jm = ((1.0 - tp) * tmp).astype(dt)
    ir = jp + jm + 1
    ip = (tt * ir).astype(dt)
    return jnp.where(z > 0, 2 * ir * (ir - 1) + ip, npixel - 2 * ir * (ir + 1) + ip)


def _polar_cap_distance(nside: int, z: ArrayLike, sin_theta: ArrayLike) -> Array:
    """Distance to the closest pole, in units where the polar cap boundary is at nside"""
    # near the poles, sin(theta) keeps the precision that 1 - |z| loses (as in Healpix C++)
    abs_z = jnp.abs(z)
    return nside * jnp.where(
        abs_z > 0.99,
        sin_theta / jnp.sqrt((1.0 + abs_z) / 3.0),
        jnp.sqrt(3.0 * (1.0 - abs_z)),
    )


def _loc2pix_nest(nside: int, z: ArrayLike, sin_theta: ArrayLike, phi: ArrayLike) -> Array:
    """Convert (z, sin_theta, phi) to a pixel number in NESTED ordering (Healpix C++ loc2pix)"""
    dt = _pixel_dtype_for(nside)
    order = nside2order(nside)
    tt = jnp.mod(2 * phi / np.pi, 4)

    # equatorial region: face from the indices of the ascending and descending edge lines
    temp1 = nside * (0.5 + tt)
    temp2 = nside * (z * 0.75)
    jp = (temp1 - temp2).astype(dt)
    jm = (temp1 + temp2).astype(dt)
    ifp = jp >> order
    ifm = jm >> order
    face_eq = jnp.where(ifp == ifm, ifp | 4, jnp.where(ifp < ifm, ifp, ifm + 8))
    ix_eq = jm & (nside - 1)
    iy_eq = nside - (jp & (nside - 1)) - 1

    # polar caps: one face per quarter of longitude
    ntt = jnp.minimum(3, tt.astype(dt))
    tp = tt - ntt
    tmp = _polar_cap_distance(nside, z, sin_theta)
    # clip points too close to the face boundary
    jp_cap = jnp.minimum((tp * tmp).astype(dt), nside - 1)
    jm_cap = jnp.minimum(((1.0 - tp) * tmp).astype(dt), nside - 1)
    north = z >= 0
    face_cap = jnp.where(north, ntt, ntt + 8)
    ix_cap = jnp.where(north, nside - jm_cap - 1, jp_cap)
    iy_cap = jnp.where(north, nside - jp_cap - 1, jm_cap)

    equatorial = jnp.abs(z) <= 2 / 3
    ix = jnp.where(equatorial, ix_eq, ix_cap)
    iy = jnp.where(equatorial, iy_eq, iy_cap)
    face = jnp.where(equatorial, face_eq, face_cap)
    return _xyf2pix_nest(nside, ix, iy, face)


@jit(static_argnames=['nside', 'nest', 'lonlat'])
def pix2ang(nside: int, ipix: ArrayLike, nest: bool = False, lonlat: bool = False) -> tuple[Array, Array]:
    r"""Return the angular coordinates $(\theta, \phi)$ of the centers of pixels.

    !!! note "Difference from healpy"
        `nside` is an integer, not an array, and the other arguments are arrays or scalars, not
        lists.

    Parameters
    ----------
    nside : int
        The resolution parameter, a power of 2 for the NESTED ordering.
    ipix : int or array-like
        The pixels.
    nest : bool, optional
        If True, use the NESTED ordering, otherwise the RING ordering.
    lonlat : bool, optional
        If True, return the longitude and the latitude, in degrees.

    Returns
    -------
    theta, phi : Array
        The colatitude $\theta \in [0, \pi]$, from the North pole, and the longitude
        $\phi \in [0, 2\pi)$, in radians, with the shape of `ipix`. With `lonlat=True`, the
        longitude and the latitude, in degrees.

    See Also
    --------
    ang2pix, pix2vec, pix2loc

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> theta, phi = hp.pix2ang(16, 1440)
    >>> print(theta, phi)
    1.5291175943723188 0.0

    >>> theta, phi = hp.pix2ang(16, np.array([1440, 427, 1520, 0, 3068]))
    >>> print(theta)
    [1.52911759 0.78550497 1.57079633 0.05103658 3.09055608]
    >>> print(phi)
    [0.         0.78539816 1.61988371 0.78539816 0.78539816]

    >>> lon, lat = hp.pix2ang(8, 11, lonlat=True)
    >>> print(lon, lat)
    337.5 78.28414760510762
    """

    z, sin_theta, phi = pix2loc(nside, ipix, nest=nest)
    # both are accurate, so arctan2 is well conditioned everywhere, near the poles included
    theta = jnp.arctan2(sin_theta, z)

    if lonlat:
        return _thetaphi2lonlat(theta, phi)
    return theta, phi


def _pix2i_ring(nside: int, pixels: ArrayLike) -> Array:
    npixel = nside2npix(nside)
    ncap = 2 * nside * (nside - 1)
    equatorial = (pixels >= ncap) & (pixels < npixel - ncap)
    return jnp.where(equatorial, _pix2i_equatorial_region_ring(nside, pixels), _pix2i_cap_ring(nside, pixels))


def _pix2i_cap_ring(nside: int, pixels: ArrayLike) -> Array:
    """Ring index counted from the closest pole, for pixels in the polar caps"""
    # both caps share the same formula, on the pixel index counted from their own pole,
    # so select its argument first and evaluate a single square root
    npixel = nside2npix(nside)
    ncap = 2 * nside * (nside - 1)
    arg = jnp.where(pixels < ncap, 1 + 2 * pixels, 2 * (npixel - pixels) - 1)
    # cast to float before sqrt so it follows the x64 flag regardless of pixels dtype
    return (1 + jnp.sqrt(arg.astype(float)).astype(int)) >> 1


def _pix2i_equatorial_region_ring(nside: int, pixels: ArrayLike) -> Array:
    ncap = 2 * nside * (nside - 1)
    ip = pixels - ncap
    order = nside2order(nside)
    #   I tmp = (order_>=0) ? ip>>(order_+2) : ip/nl4;
    tmp = ip >> (order + 2)
    return tmp + nside


def _ring2z(
    nside: int, ring_from_pole: Array, ring_from_north: Array, north_cap: Array, south_cap: Array
) -> tuple[Array, Array]:
    """Cosine and sine of the co-latitude of a ring.

    The polar caps use the ring index counted from the closest pole, the equatorial region
    the ring index counted from the North pole.
    """
    npixel = nside2npix(nside)
    # cast to float so the result follows the x64 flag regardless of the pixel dtype
    fnr = ring_from_pole.astype(float)
    abs_one_minus_z = fnr * fnr * 4 / npixel
    z = jnp.where(
        north_cap,
        1 - abs_one_minus_z,
        jnp.where(south_cap, abs_one_minus_z - 1, (2 * nside - ring_from_north).astype(float) * 2 / 3 / nside),
    )
    # near the poles, 1 - |z| is known exactly, and keeps the precision that 1 - z**2 loses
    sin_theta = jnp.sqrt(
        jnp.where(
            jnp.abs(z) > 0.99,
            abs_one_minus_z * (2 - abs_one_minus_z),
            (1 - z) * (1 + z),
        )
    )
    return z, sin_theta


def _pix2phi_ring(nside: int, iring: ArrayLike, pixels: ArrayLike) -> Array:
    npixel = nside2npix(nside)
    ncap = 2 * nside * (nside - 1)
    # pixel centers sit half a pixel off longitude zero, except on unshifted equatorial rings
    fodd = ((iring + nside) & 1) * 0.5 + 0.5  # iring + nside odd -> 1 else 0.5
    # Each region divides by its own quarter-ring pixel count (iring in the caps, nside on the
    # equator). Dividing once by a selected count instead makes XLA on CPU materialize
    # intermediates, which slows pix2vec down by about 40%.
    return jnp.where(
        pixels < ncap,
        (_pix2iphi_north_cap_ring(nside, iring, pixels) - 0.5) * np.pi / 2 / iring,
        jnp.where(
            pixels < npixel - ncap,
            (_pix2iphi_equatorial_region_ring(nside, iring, pixels) - fodd) * np.pi / 2 / nside,
            (_pix2iphi_south_cap_ring(nside, iring, pixels) - 0.5) * np.pi / 2 / iring,
        ),
    )


@jit(static_argnames=['nside', 'nest'])
def xyf2pix(nside: int, x: ArrayLike, y: ArrayLike, face: ArrayLike, nest: bool = False) -> Array:
    r"""Return the pixels of coordinates $(x, y)$ within the base face `face`.

    The sphere is divided into 12 base faces, numbered from 0 to 11, and each face into
    $n_\mathrm{side} \times n_\mathrm{side}$ pixels, of coordinates $x, y \in [0, n_\mathrm{side})$.

    !!! note "Difference from healpy"
        `nside` is an integer, not an array, and the other arguments are arrays or scalars, not
        lists.

    Parameters
    ----------
    nside : int
        The resolution parameter, a power of 2.
    x, y : int or array-like
        The coordinates of the pixels within their face, in $[0, n_\mathrm{side})$.
    face : int or array-like
        The base face of the pixels, from 0 to 11.
    nest : bool, optional
        If True, use the NESTED ordering, otherwise the RING ordering.

    Returns
    -------
    Array
        The pixels, with the broadcast shape of `x`, `y` and `face`.

    See Also
    --------
    pix2xyf

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> print(hp.xyf2pix(16, 8, 8, 4))
    1440

    >>> print(hp.xyf2pix(16, np.array([8, 8, 8, 15, 0]), np.array([8, 8, 7, 15, 0]), np.array([4, 0, 5, 0, 8])))
    [1440  427 1520    0 3068]
    """
    check_nside(nside, nest=nest)
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    face = jnp.asarray(face)
    if nest:
        return _xyf2pix_nest(nside, x, y, face)
    else:
        return _xyf2pix_ring(nside, x, y, face)


def _xyf2pix_nest(nside: int, ix: Array, iy: Array, fnum: Array) -> Array:
    """Convert (x, y, face) to pixel number in NESTED ordering"""
    fpix = _xy2fpix(nside, ix, iy)
    nested_pixel = fnum * nside**2 + fpix
    return nested_pixel


def _xy2fpix(nside: int, ix: Array, iy: Array) -> Array:
    """Convert (x, y) coordinates to a pixel index inside a face"""
    # fpix = (ix & 0b1) << 0 | (iy & 0b1) << 1 | (ix & 0b10) << 1 | (iy & 0b10) << 2 | ...

    def combine_bits(i, val):
        val |= (ix & (1 << i)) << i
        val |= (iy & (1 << i)) << (i + 1)
        return val

    # ix and iy are always less than nside, so there is no need to extract more bits than this
    length = (nside - 1).bit_length()

    # we use a native for loop because it was slightly faster than lax.fori_loop with unroll=True
    fpix = jnp.zeros_like(ix)
    for i in range(length):
        fpix = combine_bits(i, fpix)
    return fpix


# The two lookup tables of the HEALPix C++ library, indexed by face number, are computed
# arithmetically: a gather from a table does not fuse with the surrounding elementwise
# operations on CPU.


def _jrll(face_num: Array) -> Array:
    """Ring index of the south corner of each face, in units of nside (0 = North pole).

    Equivalent to indexing ``[2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]``.
    """
    return 2 + (face_num >> 2)


def _jpll(face_num: Array) -> Array:
    """Longitude index of the south corner of each face, in units of pi/4 (0 = longitude zero).

    Equivalent to indexing ``[1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7]``.
    """
    return 2 * (face_num & 3) + 1 - ((face_num >> 2) & 1)


def _xyf2pix_ring(nside: int, ix: Array, iy: Array, face_num: Array) -> Array:
    """Convert (x, y, face) to a pixel number in RING ordering"""
    # ring index of the pixel center
    jr = (_jrll(face_num) * nside) - ix - iy - 1

    ringpix = _npix_on_ring(nside, jr)
    startpix = _start_pixel_ring(nside, jr)
    kshift = 1 - _ring_shifted(nside, jr)

    # pixel number in the ring
    jp = (_jpll(face_num) * ringpix // 4 + ix - iy + 1 + kshift) // 2
    jp = jnp.where(jp < 1, jp + 4 * nside, jp)

    return startpix - 1 + jp


def _start_pixel_ring(nside: int, i_ring: ArrayLike) -> Array:
    """Get the first pixel number of a ring"""
    # work in northern hemisphere
    i_north = _northern_ring(nside, i_ring)
    ringpix = _npix_on_ring(nside, i_ring)
    ncap = 2 * nside * (nside - 1)
    npix = nside2npix(nside)
    startpix = jnp.where(
        i_north < nside,
        2 * i_north * (i_north - 1),  # north polar cap
        ncap + (i_north - nside) * 4 * nside,  # north equatorial belt
    )
    # flip results if in southern hemisphere
    startpix = jnp.where(i_ring != i_north, npix - startpix - ringpix, startpix)
    return startpix


def _npix_on_ring(nside: int, i_ring: ArrayLike) -> Array:
    """Get the number of pixels on a ring"""
    i_north = _northern_ring(nside, i_ring)
    ringpix = jnp.where(
        i_north < nside,
        4 * i_north,  # rings in the polar cap have 4*i pixels
        4 * nside,  # rings in the equatorial region have 4*nside pixels
    )
    return ringpix


def _ring_shifted(nside: int, i_ring: ArrayLike) -> Array:
    """Check if a ring is shifted"""
    i_north = _northern_ring(nside, i_ring)
    shifted = jnp.where(
        i_north < nside,
        True,
        (i_north - nside) & 1 == 0,
    )
    return shifted


def _northern_ring(nside: int, i_ring: ArrayLike) -> Array:
    i_north = jnp.where(i_ring > 2 * nside, 4 * nside - i_ring, i_ring)
    return i_north


def _ring_above(nside: int, cos_theta: ArrayLike) -> Array:
    """Find the ring index just north of a point with given cos(theta).

    This follows the exact HEALPix C++ implementation:

    if (az <= twothird) // equatorial region
        return I(nside_*(2-1.5*z));
    I iring = I(nside_*sqrt(3*(1-az)));
    return (z>0) ? iring : 4*nside_-iring-1;
    """
    z = cos_theta
    az = jnp.abs(z)
    twothird = 2.0 / 3.0

    # Equatorial region
    equatorial_ring = nside * (2.0 - 1.5 * z)
    # Stop gradient: Ring indices are discrete selectors, don't affect interpolation math
    equatorial_ring = lax.stop_gradient(jnp.floor(equatorial_ring).astype(jnp.int32))

    # Polar caps
    iring = nside * jnp.sqrt(3.0 * (1.0 - az))
    # Stop gradient: Ring indices are discrete selectors, don't affect interpolation math
    iring = lax.stop_gradient(jnp.floor(iring).astype(jnp.int32))
    polar_ring = jnp.where(z > 0, iring, 4 * nside - iring - 1)

    # Choose based on z value
    ring_idx = jnp.where(az <= twothird, equatorial_ring, polar_ring)

    return ring_idx


def _get_ring_info(nside: int, ring_idx: ArrayLike) -> tuple[Array, Array, Array, Array]:
    """Get ring properties following HEALPix C++ get_ring_info2 exactly.

    Returns: theta, startpix, ringpix, shifted
    """
    # Co-latitude: the cosine and sine are selected per region (and hemisphere) before the
    # single arctan2, so only one transcendental is evaluated per ring instead of one per region
    costheta, sintheta = _get_ring_costheta_sintheta(nside, ring_idx)
    theta = jnp.arctan2(sintheta, costheta)

    startpix = _start_pixel_ring(nside, ring_idx)
    ringpix = _npix_on_ring(nside, ring_idx)
    # Convert shifted boolean to float (0.0 or 0.5)
    shift = jnp.where(_ring_shifted(nside, ring_idx), 0.5, 0.0)

    return theta, startpix, ringpix, shift


def _get_ring_costheta_sintheta(nside: int, ring_idx: ArrayLike) -> tuple[Array, Array]:
    """Get the cosine and sine of a ring's co-latitude, without forming the angle.

    `_get_ring_info` derives its angle from these through `arctan2`. Recovering the
    cosine and sine from that angle instead would round twice and lose accuracy near the
    poles, which matters in float32.

    Parameters
    ----------
    nside : int
        The healpix nside parameter.
    ring_idx : ArrayLike
        Ring index, 1 to 4*nside-1.

    Returns
    -------
    tuple[Array, Array]
        cos(theta) and sin(theta) of the ring. The sine is non-negative.
    """
    ring = ring_idx

    fact1 = 2.0 / (3.0 * nside)
    fact2 = 4.0 / (12.0 * nside * nside)

    # Northern hemisphere equivalent ring
    northring = _northern_ring(nside, ring)

    # Polar cap region (northring < nside): both are exact rearrangements of the
    # HEALPix ring definition, with no cancellation in sin near the pole.
    polar_tmp = northring * northring * fact2
    polar_costheta = 1.0 - polar_tmp
    polar_sin2 = polar_tmp * (2.0 - polar_tmp)

    # Equatorial region (northring >= nside): |z| <= 2/3, so sqrt(1 - z**2) is well
    # conditioned everywhere in this branch. The clamp is for polar rings, where this
    # branch is discarded but still evaluated, and would otherwise be negative.
    equatorial_costheta = (2.0 * nside - northring) * fact1
    equatorial_sin2 = jnp.maximum(1.0 - equatorial_costheta * equatorial_costheta, 0.0)

    # The squared sine is selected before the square root, so a single one is evaluated
    is_polar = northring < nside
    costheta = jnp.where(is_polar, polar_costheta, equatorial_costheta)
    sintheta = jnp.sqrt(jnp.where(is_polar, polar_sin2, equatorial_sin2))

    # Southern hemisphere correction: theta -> pi - theta flips the cosine, keeps the sine
    costheta = jnp.where(northring != ring, -costheta, costheta)

    return costheta, sintheta


@jit(static_argnames=['nside', 'nest'])
def pix2xyf(nside: int, ipix: ArrayLike, nest: bool = False) -> tuple[Array, Array, Array]:
    r"""Return the coordinates $(x, y)$ of pixels within their base face, and the face.

    The sphere is divided into 12 base faces, numbered from 0 to 11, and each face into
    $n_\mathrm{side} \times n_\mathrm{side}$ pixels, of coordinates $x, y \in [0, n_\mathrm{side})$.

    !!! note "Difference from healpy"
        `nside` is an integer, not an array, and the other arguments are arrays or scalars, not
        lists.

    Parameters
    ----------
    nside : int
        The resolution parameter, a power of 2.
    ipix : int or array-like
        The pixels.
    nest : bool, optional
        If True, use the NESTED ordering, otherwise the RING ordering.

    Returns
    -------
    x, y : Array
        The coordinates of the pixels within their face, with the shape of `ipix`.
    face : Array
        The base face of the pixels, with the shape of `ipix`.

    See Also
    --------
    xyf2pix

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> x, y, face = hp.pix2xyf(16, 1440)
    >>> print(x, y, face)
    8 8 4

    >>> x, y, face = hp.pix2xyf(16, np.array([1440, 427, 1520, 0, 3068]))
    >>> print(x, y, face)
    [ 8  8  8 15  0] [ 8  8  7 15  0] [4 0 5 0 8]
    """
    check_nside(nside, nest=nest)
    ipix = jnp.asarray(ipix)
    if nest:
        return _pix2xyf_nest(nside, ipix)
    else:
        return _pix2xyf_ring(nside, ipix)


def _pix2xyf_nest(nside: int, pix: Array) -> tuple[Array, Array, Array]:
    """Convert a pixel number in NESTED ordering to (x, y, face)"""
    fnum, fpix = jnp.divmod(pix, nside**2)
    ix, iy = _fpix2xy(nside, fpix)
    return ix, iy, fnum


def _fpix2xy(nside: int, fpix: Array) -> tuple[Array, Array]:
    """Convert a pixel index inside a face into (x, y) coordinates.

    Pixel indices inside the face must be less than nside**2.
    """
    # x = (pix & 0b1) >> 0 | (pix & 0b100) >> 1 | (pix & 0b10000) >> 2 | ...
    # y = (pix & 0b10) >> 1 | (pix & 0b1000) >> 2 | (pix & 0b100000) >> 3 | ...

    def extract_bits(i, carry):
        x, y = carry
        x |= (fpix & (1 << (2 * i))) >> i
        y |= (fpix & (1 << (2 * i + 1))) >> (i + 1)
        return x, y

    # imagine that nside = 2 ** ord (nside must be a power of 2 in nested ordering)
    # the maximum value of pix is nside**2 - 1, which fits on 2 * ord bits
    # because we extract 2 bits at a time, we need to loop ord times
    # and ord is the bit length of nside - 1
    length = (nside - 1).bit_length()

    # we use a native for loop because it was slightly faster than lax.fori_loop with unroll=True
    x, y = jnp.zeros_like(fpix), jnp.zeros_like(fpix)
    for i in range(length):
        x, y = extract_bits(i, (x, y))
    return x, y


def _pix2xyf_ring(nside: int, pix: Array) -> tuple[Array, Array, Array]:
    """Convert a pixel number in RING ordering to (x, y, face)"""
    ncap = 2 * nside * (nside - 1)
    npix = nside2npix(nside)
    nl2 = 2 * nside  # number of pixels in a latitude circle

    # TODO(simon): remove this cast when https://github.com/CMBSciPol/jax-healpy/issues/4 is fixed
    iring = _pix2i_ring(nside, pix).astype(_pixel_dtype_for(nside))
    iphi = _pix2iphi_ring(nside, iring, pix)
    nr = _npix_on_ring(nside, iring) // 4
    kshift = 1 - _ring_shifted(nside, iring)

    ire = iring - nside + 1
    irm = nl2 + 2 - ire
    ifm = (iphi - ire // 2 + nside - 1) // nside
    ifp = (iphi - irm // 2 + nside - 1) // nside

    # in the polar caps, the face is the quarter of the ring holding the pixel: (iphi - 1) // nr,
    # which lies in [0, 3], counted with comparisons since an integer division by an array
    # does not vectorize on CPU
    quarter = (iphi > nr).astype(iphi.dtype) + (iphi > 2 * nr) + (iphi > 3 * nr)
    face_num = jnp.where(
        pix < ncap,
        quarter,  # north polar cap
        jnp.where(
            pix < (npix - ncap),
            jnp.where(ifp == ifm, ifp | 4, jnp.where(ifp < ifm, ifp, ifm + 8)),
            8 + quarter,  # south polar cap
        ),
    )

    iring_for_irt = jnp.where(
        pix < (npix - ncap),
        iring,  # north polar cap and equatorial region
        4 * nside - iring,  # south polar cap
    )  # ring number counted from North pole or South pole

    irt = iring_for_irt - (_jrll(face_num) * nside) + 1
    ipt = 2 * iphi - _jpll(face_num) * nr - kshift - 1
    ipt -= jnp.where(ipt >= nl2, 8 * nside, 0)

    ix = (ipt - irt) // 2
    iy = (-ipt - irt) // 2

    return ix, iy, face_num


def _pix2iphi_ring(nside: int, iring: Array, pixels: Array) -> Array:
    npixel = nside2npix(nside)
    ncap = 2 * nside * (nside - 1)
    iphi = jnp.where(
        pixels < ncap,
        _pix2iphi_north_cap_ring(nside, iring, pixels),
        jnp.where(
            pixels < npixel - ncap,
            _pix2iphi_equatorial_region_ring(nside, iring, pixels),
            _pix2iphi_south_cap_ring(nside, iring, pixels),
        ),
    )
    return iphi


def _pix2iphi_north_cap_ring(nside: int, iring: Array, pixels: Array) -> Array:
    iphi = pixels + 1 - 2 * iring * (iring - 1)
    return iphi


def _pix2iphi_equatorial_region_ring(nside: int, iring: Array, pixels: Array) -> Array:
    iphi = pixels + 2 * nside * (nside + 1) - 4 * nside * iring + 1
    return iphi


def _pix2iphi_south_cap_ring(nside: int, iring: Array, pixels: Array) -> Array:
    npixel = nside2npix(nside)
    iphi = 4 * iring + 1 - (npixel - pixels - 2 * iring * (iring - 1))
    return iphi


@jit(static_argnames=['nside', 'nest'])
def vec2pix(nside: int, x: ArrayLike, y: ArrayLike, z: ArrayLike, nest: bool = False) -> Array:
    """Return the pixels that contain the directions of the vectors $(x, y, z)$.

    The vectors need not be normalized.

    !!! note "Difference from healpy"
        `nside` is an integer, not an array, and the other arguments are arrays or scalars, not
        lists.

    Parameters
    ----------
    nside : int
        The resolution parameter, a power of 2 for the NESTED ordering.
    x, y, z : float or array-like
        The coordinates of the vectors.
    nest : bool, optional
        If True, use the NESTED ordering, otherwise the RING ordering.

    Returns
    -------
    Array
        The pixels, with the broadcast shape of `x`, `y` and `z`.

    See Also
    --------
    pix2vec, ang2pix

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> print(hp.vec2pix(16, 1, 0, 0))
    1504

    >>> print(hp.vec2pix(16, np.array([1, 0]), np.array([0, 1]), np.array([0, 0])))
    [1504 1520]
    """
    dnorm = 1 / jnp.sqrt(x**2 + y**2 + z**2)
    return loc2pix(nside, z * dnorm, jnp.sqrt(x**2 + y**2) * dnorm, jnp.arctan2(y, x), nest=nest)


@jit(static_argnames=['nside', 'nest'])
def pix2vec(nside: int, ipix: ArrayLike, nest: bool = False) -> Array:
    """Return the unit vectors of the centers of pixels.

    !!! note "Difference from healpy"
        The vectors are one array of shape `(*ipix.shape, 3)`, whereas healpy returns a tuple of
        three arrays `(x, y, z)`. For a scalar or one-dimensional `ipix`, the transpose,
        `hp.pix2vec(nside, ipix).T`, has the layout of healpy. `nside` is an integer, not an array,
        and `ipix` an array or a scalar, not a list.

    Parameters
    ----------
    nside : int
        The resolution parameter, a power of 2 for the NESTED ordering.
    ipix : int or array-like
        The pixels.
    nest : bool, optional
        If True, use the NESTED ordering, otherwise the RING ordering.

    Returns
    -------
    Array
        The unit vectors, of shape `(*ipix.shape, 3)`.

    See Also
    --------
    vec2pix, pix2ang

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> print(hp.pix2vec(16, 1504))
    [0.99879546 0.04906767 0.        ]

    >>> print(hp.pix2vec(16, np.array([1440, 427])))
    [[0.99913157 0.         0.04166667]
     [0.5000534  0.5000534  0.70703125]]
    """
    z, sin_theta, phi = pix2loc(nside, ipix, nest=nest)
    return jnp.stack([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), z], axis=-1)


@jit(static_argnames=['nside', 'nest'])
def pix2loc(nside: int, ipix: ArrayLike, nest: bool = False) -> tuple[Array, Array, Array]:
    r"""Return the coordinates $(z, \sin\theta, \phi)$ of the centers of pixels.

    Same as [`pix2ang`][], but the colatitude $\theta$ is given by its cosine $z$ and its sine,
    which the pixelization gives without trigonometric functions. It is the inverse of
    [`loc2pix`][]. healpy has no such function.

    Parameters
    ----------
    nside : int
        The resolution parameter, a power of 2 for the NESTED ordering.
    ipix : int or array-like
        The pixels.
    nest : bool, optional
        If True, use the NESTED ordering, otherwise the RING ordering.

    Returns
    -------
    z, sin_theta, phi : Array
        The cosine and the sine of the colatitude, and the longitude in radians, with the shape
        of `ipix`. Near the poles, $\sin\theta$ is computed from $1 - |z|$, which keeps its
        precision.

    See Also
    --------
    loc2pix, pix2ang, pix2vec

    Examples
    --------
    >>> import jax_healpy as hp
    >>> z, sin_theta, phi = hp.pix2loc(16, 1440)
    >>> print(f'{z:.8f} {sin_theta:.8f} {phi:.8f}')
    0.04166667 0.99913157 0.00000000
    """
    check_nside(nside, nest=nest)
    if nest:
        return _pix2loc_nest(nside, ipix)
    return _pix2loc_ring(nside, ipix)


def _pix2loc_ring(nside: int, pixels: ArrayLike) -> tuple[Array, Array, Array]:
    npixel = nside2npix(nside)
    ncap = 2 * nside * (nside - 1)
    # in the polar caps, iring is counted from the closest pole
    iring = _pix2i_ring(nside, pixels)
    z, sin_theta = _ring2z(nside, iring, iring, pixels < ncap, pixels >= npixel - ncap)
    phi = _pix2phi_ring(nside, iring, pixels)
    return z, sin_theta, phi


def _pix2loc_nest(nside: int, pixels: ArrayLike) -> tuple[Array, Array, Array]:
    """Convert a pixel number in NESTED ordering to (z, sin_theta, phi) (Healpix C++ pix2loc)"""
    ix, iy, face_num = _pix2xyf_nest(nside, jnp.asarray(pixels).astype(_pixel_dtype_for(nside)))

    # ring index of the pixel center, counted from the North pole
    jr = _jrll(face_num) * nside - ix - iy - 1
    north_cap = jr < nside
    south_cap = jr > 3 * nside
    # number of pixels in a quarter of the ring, which in the polar caps is also the ring
    # index counted from the closest pole
    nr = jnp.where(north_cap, jr, jnp.where(south_cap, 4 * nside - jr, nside))

    z, sin_theta = _ring2z(nside, nr, jr, north_cap, south_cap)

    # pixel index in the ring, counted from longitude zero, in units of half pixels
    kk = _jpll(face_num) * nr + ix - iy
    kk = jnp.where(kk < 0, kk + 8 * nr, kk)
    phi = kk.astype(float) * np.pi / 4 / nr.astype(float)
    return z, sin_theta, phi


@jit(static_argnames=['lonlat'])
def ang2vec(theta: ArrayLike, phi: ArrayLike, lonlat: bool = False) -> Array:
    r"""Return the unit vectors of the angular coordinates $(\theta, \phi)$.

    !!! note "Difference from healpy"
        A colatitude outside $[0, \pi]$ gives a vector of NaN, instead of raising an exception.

    Parameters
    ----------
    theta, phi : float or array-like
        The colatitude $\theta \in [0, \pi]$, from the North pole, and the longitude $\phi$, in
        radians. With `lonlat=True`, the longitude and the latitude, in degrees.
    lonlat : bool, optional
        If True, `theta` and `phi` are the longitude and the latitude, in degrees.

    Returns
    -------
    Array
        The unit vectors, of shape `(*shape, 3)`, where `shape` is the broadcast shape of `theta`
        and `phi`.

    See Also
    --------
    vec2ang, pix2vec

    Examples
    --------
    >>> import jax_healpy as hp
    >>> print(hp.ang2vec(0.0, 0.0))
    [0. 0. 1.]
    """
    if lonlat:
        theta, phi = _lonlat2thetaphi(theta, phi)

    theta = jnp.where((theta < 0) | (theta > np.pi + 1e-5), np.nan, theta)
    sin_theta = jnp.sin(theta)
    x = sin_theta * jnp.cos(phi)
    y = sin_theta * jnp.sin(phi)
    z = jnp.cos(theta)
    return jnp.stack([x, y, z], axis=-1)


@jit(static_argnames=['lonlat'])
def vec2ang(vectors: ArrayLike, lonlat: bool = False) -> tuple[Array, Array]:
    r"""Return the angular coordinates $(\theta, \phi)$ of the directions of vectors.

    The vectors need not be normalized.

    Parameters
    ----------
    vectors : array-like
        The vectors, of shape `(*batch, 3)`.
    lonlat : bool, optional
        If True, return the longitude and the latitude, in degrees.

    Returns
    -------
    theta, phi : Array
        The colatitude $\theta \in [0, \pi]$, from the North pole, and the longitude
        $\phi \in [0, 2\pi)$, in radians, of shape `batch`. With `lonlat=True`, the longitude and
        the latitude, in degrees.

    Raises
    ------
    ValueError
        If the last dimension of `vectors` is not 3.

    See Also
    --------
    ang2vec, vec2pix

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> theta, phi = hp.vec2ang(np.array([0.0, 2.0, 0.0]))
    >>> print(theta, phi)
    1.5707963267948966 1.5707963267948966
    """
    vectors = jnp.asarray(vectors)
    # Enforce documented shape (*batch, 3) to avoid silently ignoring components
    if vectors.ndim == 0 or vectors.shape[-1] != 3:
        raise ValueError(f'`vec2ang` expects `vectors` with shape (*batch, 3); got array with shape {vectors.shape}')
    dnorm = jnp.sqrt(vectors[..., 0] ** 2 + vectors[..., 1] ** 2 + vectors[..., 2] ** 2)
    theta = jnp.arccos(vectors[..., 2] / dnorm)
    phi = jnp.arctan2(vectors[..., 1], vectors[..., 0])
    phi = jnp.where(phi < 0, phi + 2 * np.pi, phi)
    if lonlat:
        return _thetaphi2lonlat(theta, phi)
    return theta, phi


@jit(static_argnames=['nside'])
def ring2nest(nside: int, ipix: ArrayLike) -> Array:
    """Convert pixels from the RING ordering to the NESTED ordering.

    !!! note "Difference from healpy"
        `nside` is an integer, not an array, and the other arguments are arrays or scalars, not
        lists.

    Parameters
    ----------
    nside : int
        The resolution parameter, a power of 2.
    ipix : int or array-like
        The pixels in the RING ordering.

    Returns
    -------
    Array
        The pixels in the NESTED ordering, with the shape of `ipix`.

    See Also
    --------
    nest2ring, reorder

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> print(hp.ring2nest(16, 1504))
    1130

    >>> print(hp.ring2nest(2, np.arange(10)))
    [ 3  7 11 15  2  1  6  5 10  9]
    """
    check_nside(nside, nest=True)
    ipix = jnp.asarray(ipix)
    # promote to int64 only if nside requires it
    ipix = ipix.astype(jnp.promote_types(ipix.dtype, _pixel_dtype_for(nside)))
    xyf = _pix2xyf_ring(nside, ipix)
    ipix_nest = _xyf2pix_nest(nside, *xyf)
    return ipix_nest


@jit(static_argnames=['nside'])
def nest2ring(nside: int, ipix: ArrayLike) -> Array:
    """Convert pixels from the NESTED ordering to the RING ordering.

    !!! note "Difference from healpy"
        `nside` is an integer, not an array, and the other arguments are arrays or scalars, not
        lists.

    Parameters
    ----------
    nside : int
        The resolution parameter, a power of 2.
    ipix : int or array-like
        The pixels in the NESTED ordering.

    Returns
    -------
    Array
        The pixels in the RING ordering, with the shape of `ipix`.

    See Also
    --------
    ring2nest, reorder

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> print(hp.nest2ring(16, 1130))
    1504

    >>> print(hp.nest2ring(2, np.arange(10)))
    [13  5  4  0 15  7  6  1 17  9]
    """
    check_nside(nside, nest=True)
    ipix = jnp.asarray(ipix)
    # promote to int64 only if nside requires it
    ipix = ipix.astype(jnp.promote_types(ipix.dtype, _pixel_dtype_for(nside)))
    xyf = _pix2xyf_nest(nside, ipix)
    ipix_ring = _xyf2pix_ring(nside, *xyf)
    return ipix_ring


@jit(static_argnames=['inp', 'out', 'r2n', 'n2r', 'process_by_chunks'])
def reorder(
    map_in: ArrayLike,
    inp: str | None = None,
    out: str | None = None,
    r2n: bool = False,
    n2r: bool = False,
    process_by_chunks: bool = False,
) -> Array:
    """Reorder maps from the RING ordering to the NESTED ordering, or back.

    !!! note "Difference from healpy"
        Masked arrays are not supported. The maps are reordered in one go, unless
        `process_by_chunks` is True.

    Parameters
    ----------
    map_in : array-like
        The maps, of shape `(..., npix)`.
    inp, out : {'RING', 'NESTED'}, optional
        The orderings of the input and of the output. Only their first four letters count,
        whatever their case: `'NEST'` is the NESTED ordering.
    r2n : bool, optional
        If True, reorder from RING to NESTED, whatever `inp` and `out`.
    n2r : bool, optional
        If True, reorder from NESTED to RING, whatever `inp` and `out`.
    process_by_chunks : bool, optional
        If True, and if `nside > 128`, reorder the pixels in 24 chunks, one after the other, as
        healpy does, which needs less memory.

    Returns
    -------
    Array
        The reordered maps, with the shape of `map_in`.

    Raises
    ------
    ValueError
        If `map_in` is a scalar, if its number of pixels is not that of a map whose `nside` is a
        power of 2, if `r2n` and `n2r` are both True, or if an ordering is neither RING nor NESTED.

    See Also
    --------
    nest2ring, ring2nest

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> print(hp.reorder(np.arange(48), r2n=True))
    [13  5  4  0 15  7  6  1 17  9  8  2 19 11 10  3 28 20 27 12 30 22 21 14
     32 24 23 16 34 26 25 18 44 37 36 29 45 39 38 31 46 41 40 33 47 43 42 35]

    >>> maps = np.stack([np.arange(48), 100 + np.arange(48)])
    >>> print(hp.reorder(maps, r2n=True)[:, :4])
    [[ 13   5   4   0]
     [113 105 104 100]]
    """
    # Check input map(s)
    map_in = jnp.asarray(map_in)
    if map_in.ndim == 0:
        raise ValueError('Input map can not be a scalar')
    npix = map_in.shape[-1]
    nside = npix2nside(npix)
    # npix2nside already fails on bad number of pixels
    # but in nested ordering we must also ensure that nside is power of 2
    check_nside(nside, nest=True)

    # Check input parameters
    if r2n and n2r:
        raise ValueError('r2n and n2r cannot be used simultaneously')
    if r2n:
        inp, out = 'RING', 'NEST'
    if n2r:
        inp, out = 'NEST', 'RING'
    inp, out = str(inp).upper()[:4], str(out).upper()[:4]
    if not {inp, out}.issubset({'RING', 'NEST'}):
        raise ValueError('inp and out must be either RING or NEST')
    if inp == out:
        return map_in

    # Perform the conversion, which is just a reordering of the pixels
    def _reorder(ipix):
        if inp == 'RING':
            ipix_reordered = nest2ring(nside, ipix)
        else:
            ipix_reordered = ring2nest(nside, ipix)
        return map_in[..., ipix_reordered]

    if not process_by_chunks:
        ipix_full = jnp.arange(npix, dtype=_pixel_dtype_for(nside))
        return _reorder(ipix_full)

    # To reduce memory requirements, process the map in chunks
    chunk_size = npix // 24 if nside > 128 else npix
    n_chunks = npix // chunk_size

    def body(i, map_out):
        # interval bounds must be static, so we shift the values afterwards
        ipix_chunk = jnp.arange(chunk_size, dtype=_pixel_dtype_for(nside)) + i * chunk_size
        return map_out.at[..., ipix_chunk].set(_reorder(ipix_chunk))

    map_out = lax.fori_loop(0, n_chunks, body, jnp.empty_like(map_in))
    return map_out


class InterpCenters(NamedTuple):
    r"""Centers of the four pixels returned by [`get_interp_weights`][] with `with_centers=True`.

    The four pixels lie on two rings: rows 0 and 1 of `pixels` on the first, rows 2 and 3 on the
    second. The colatitude therefore takes two values per point, not four. It is given by its
    cosine and its sine, the form that the geometry of the rings gives, and that the parallel
    transport of spin fields needs.

    As a `NamedTuple`, it is a JAX pytree, which crosses the boundaries of `jax.jit` and is batched
    by `jax.vmap`, and it unpacks as a tuple `(z, sth, phi)`.

    Attributes
    ----------
    z : Array
        The cosine of the colatitude of the two rings, of shape `(2, *shape)`.
    sth : Array
        The sine of the colatitude of the two rings, nonnegative, of shape `(2, *shape)`.
    phi : Array
        The longitude of the four pixels, in radians, in $[0, 2\pi)$, of shape `(4, *shape)`,
        row for row as `pixels`.
    """

    z: Array
    sth: Array
    phi: Array


@jit(static_argnames=['nside', 'nest', 'lonlat', 'with_centers'])
def get_interp_weights(
    nside: int,
    theta: ArrayLike,
    phi: ArrayLike | None = None,
    nest: bool = False,
    lonlat: bool = False,
    with_centers: bool = False,
) -> tuple[Array, Array] | tuple[Array, Array, InterpCenters]:
    r"""Return the four pixels and the weights of the bilinear interpolation at points of the sphere.

    The four pixels are the nearest to the point on the two rings of pixel centers that surround
    it, two on each ring. The interpolation of a map $m$ is $\sum_i w_i\, m[p_i]$ (see
    [`get_interp_val`][]).

    !!! note "Difference from healpy"
        - The NESTED ordering is not supported: `nest=True` raises a `ValueError`.
        - The four pixels may come in another order than in healpy: sorting them, and the weights
          with them, gives those of healpy (see the notes).
        - `with_centers` and [`InterpCenters`][] are specific to jax-healpy.
        - `nside` is an integer, not an array, and the other arguments are arrays or scalars, not
          lists.

    Parameters
    ----------
    nside : int
        The resolution parameter.
    theta : float or array-like
        The colatitude, in radians, or with `lonlat=True` the longitude, in degrees. If `phi` is
        None, the pixels whose centers are the points.
    phi : float or array-like, optional
        The longitude, in radians, or with `lonlat=True` the latitude, in degrees.
    nest : bool, optional
        Must be False: the NESTED ordering is not supported.
    lonlat : bool, optional
        If True, `theta` and `phi` are the longitude and the latitude, in degrees.
    with_centers : bool, optional
        If True, also return the centers of the four pixels.

    Returns
    -------
    pixels : Array
        The four pixels, of shape `(4, *shape)`, where `shape` is the broadcast shape of `theta`
        and `phi`. Rows 0 and 1 are on the first ring, rows 2 and 3 on the second.
    weights : Array
        The interpolation weights, of shape `(4, *shape)`. They sum to 1.
    centers : InterpCenters
        Only if `with_centers` is True: the centers of `pixels`, row for row. The colatitude takes
        one value per ring, so that `centers.z` and `centers.sth` have the shape `(2, *shape)`, and
        `centers.phi` the shape `(4, *shape)`: `centers.z[jnp.array([0, 0, 1, 1])]` gives one
        value per pixel. The colatitude is given by its cosine and its sine, as the geometry of
        the rings gives it: the angle would be rounded twice.

    Raises
    ------
    ValueError
        If `nest` is True.

    See Also
    --------
    get_interp_val, InterpCenters

    Notes
    -----
    **Order of the pixels.** Sorting the pixels, and the weights with them, gives those of healpy:

    ```python
    order = jnp.argsort(pixels, axis=0)
    pixels = jnp.take_along_axis(pixels, order, axis=0)
    weights = jnp.take_along_axis(weights, order, axis=0)
    ```

    **Agreement with healpy.** On 100 000 random points, in 64-bit precision, the four pixels are
    those of healpy, and the weights differ from those of healpy by at most
    $1.2 \times 10^{-15}\, n_\mathrm{side}$, for `nside` from 16 to 4096. The difference grows
    with `nside` because the weight along the colatitude is divided by the distance between the
    two rings, about $1 / n_\mathrm{side}$.

    **The centers do not change the weights.** `pixels` and `weights` are the same, to the bit,
    whether `with_centers` is True or False. The weights and the ring indices go through a
    `lax.optimization_barrier`, so that XLA cannot fuse their computation differently when the
    centers also use them: a one-ulp change would be amplified by about `nside`.

    **Differentiation.** The weights depend continuously on the coordinates of the point: their
    gradient is that of the interpolation, and the gradient of their sum is zero. The pixels and
    their centers are piecewise constant: their gradient is zero.

    Examples
    --------
    Gradient of an interpolation with respect to the coordinates of the point:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as hp
    >>> nside = 4
    >>> m = jnp.arange(hp.nside2npix(nside), dtype=float)
    >>> def interpolate_map(m, theta, phi):
    ...     pixels, weights = hp.get_interp_weights(nside, theta, phi)
    ...     return jnp.sum(weights * m[pixels], axis=0)
    >>> grad_theta, grad_phi = jax.grad(interpolate_map, argnums=(1, 2))(m, 1.0, 0.5)

    The gradient agrees with a finite difference:

    >>> eps = 1e-6
    >>> finite_difference = (interpolate_map(m, 1.0 + eps, 0.5) - interpolate_map(m, 1.0 - eps, 0.5)) / (2 * eps)
    >>> bool(jnp.isclose(grad_theta, finite_difference))
    True
    """
    check_nside(nside, nest=nest)

    if nest:
        raise ValueError('NEST pixel ordering is not supported. Only RING ordering is supported.')

    # Handle different input modes
    if phi is None:
        # theta contains pixel indices, convert to (theta, phi) coordinates
        theta_coords, phi_coords = pix2ang(nside, theta, nest=False, lonlat=False)
    else:
        theta_coords, phi_coords = jnp.asarray(theta), jnp.asarray(phi)
        if lonlat:
            theta_coords, phi_coords = _lonlat2thetaphi(theta_coords, phi_coords)

    # Call the RING implementation
    return _get_interp_weights_ring(nside, theta_coords, phi_coords, with_centers)


def _get_interp_weights_ring(
    nside: int, theta_coords: Array, phi_coords: Array, with_centers: bool = False
) -> tuple[Array, Array] | tuple[Array, Array, InterpCenters]:
    """
    Bilinear interpolation for RING ordering.

    The work is split in two stages. The first computes everything that needs a
    transcendental (the ring above the target, the co-latitude of the two rings) and
    reduces it to a small per-sample state: the ring index, the index of the western
    neighbour on each ring, and the four finished weights. The second rebuilds the
    pixels, and the centers when asked for, from that state with integer arithmetic.
    Outputs are assembled with `jnp.stack`, which XLA lowers to concatenate kernels
    that read the state once; the fewer arrays the first stage hands over, the less
    memory traffic, which is what bounds this function on GPU in single precision.

    **Gradient compatibility.**
    This function is fully compatible with JAX's automatic differentiation system.
    The implementation carefully separates discrete operations (pixel selection) from
    continuous operations (weight computation):

    - Discrete pixel indices use `lax.stop_gradient()` to prevent gradient flow
      through non-differentiable operations like `jnp.floor().astype(int)`
    - Weight computations use continuous mathematical operations that preserve gradients
    - Final weight normalization enforces the constraint sum(weights) = 1.0, ensuring
      that gradients of the weight sum are exactly zero

    The stop_gradient usage is mathematically sound because:
    1. Pixel indices are discrete selectors that don't affect the interpolation mathematics
    2. Weight values depend continuously on input coordinates within each pixel region
    3. The fundamental constraint sum(weights) = 1.0 must hold regardless of pixel selection
    """
    # First stage: ring geometry, longitude grid position and weights
    z = jnp.cos(theta_coords)
    ir1 = _ring_above(nside, z)

    # Inside ring 1 (or the last ring) the target is surrounded by a single ring and the
    # pole. The clamped ring indices then collapse onto that ring.
    is_north_pole = ir1 == 0
    is_south_pole = ir1 == 4 * nside - 1
    ir1_safe = jnp.maximum(ir1, 1)
    ir2_safe = jnp.minimum(ir1 + 1, 4 * nside - 1)

    theta1, _, nr1, shift1 = _get_ring_info(nside, ir1_safe)
    theta2, _, nr2, shift2 = _get_ring_info(nside, ir2_safe)
    if 4 * nside < theta_coords.size:
        # Fewer rings than samples: evaluate the co-latitude once per ring and look it up,
        # rather than two arctan2 per sample. Same expression, so the same bits; the
        # arctan2 above is then dead and removed by XLA.
        ring_theta, _, _, _ = _get_ring_info(nside, jnp.arange(1, 4 * nside, dtype=ir1_safe.dtype))
        theta1 = ring_theta[ir1_safe - 1]
        theta2 = ring_theta[ir2_safe - 1]

    # Phi interpolation position, in units of pixels along each ring
    phi1_norm = phi_coords / (2.0 * jnp.pi / nr1) - shift1
    phi2_norm = phi_coords / (2.0 * jnp.pi / nr2) - shift2

    # Stop gradient: Floor operations are non-differentiable and only used for indexing
    floor1 = lax.stop_gradient(jnp.floor(phi1_norm))
    floor2 = lax.stop_gradient(jnp.floor(phi2_norm))

    # Fractional parts: the weights keep the unit gradient with respect to phi. Wrapping
    # the position onto the ring (a float remainder) is left to the integer indices, as
    # it does not change the fractional part.
    w_phi1 = phi1_norm - floor1
    w_phi2 = phi2_norm - floor2

    # Index of the western neighbour on each ring, wrapped onto [0, nr). The ring sizes
    # are positive, so a truncated remainder needs a single sign fix, cheaper than the
    # general floored `%`.
    rem1 = lax.rem(floor1.astype(jnp.int32), nr1)
    rem2 = lax.rem(floor2.astype(jnp.int32), nr2)
    i1_1 = jnp.where(rem1 < 0, rem1 + nr1, rem1)
    i1_2 = jnp.where(rem2 < 0, rem2 + nr2, rem2)

    # Theta interpolation weight. Near a pole the missing ring is replaced by the pole
    # itself, at theta = 0 or pi. Numerator and denominator are selected first so that
    # a single division is evaluated rather than one per branch.
    w_theta = jnp.where(is_north_pole, theta_coords, theta_coords - theta1) / jnp.where(
        is_north_pole, theta2, jnp.where(is_south_pole, jnp.pi - theta1, theta2 - theta1)
    )

    # Weights. Near a pole, the share of the missing ring goes in equal parts to the
    # four pixels of the cap ring.
    w1_phi = 1.0 - w_phi1
    w2_phi = w_phi1
    w3_phi = 1.0 - w_phi2
    w4_phi = w_phi2
    north_factor = (1.0 - w_theta) * 0.25
    south_factor = w_theta * 0.25
    w1 = jnp.where(
        is_north_pole,
        north_factor,
        jnp.where(is_south_pole, w1_phi * (1.0 - w_theta) + south_factor, w1_phi * (1.0 - w_theta)),
    )
    w2 = jnp.where(
        is_north_pole,
        north_factor,
        jnp.where(is_south_pole, w2_phi * (1.0 - w_theta) + south_factor, w2_phi * (1.0 - w_theta)),
    )
    w3 = jnp.where(
        is_north_pole, w3_phi * w_theta + north_factor, jnp.where(is_south_pole, south_factor, w3_phi * w_theta)
    )
    w4 = jnp.where(
        is_north_pole, w4_phi * w_theta + north_factor, jnp.where(is_south_pole, south_factor, w4_phi * w_theta)
    )

    # Clamp weights to ensure non-negativity (handles floating point precision issues)
    w1 = jnp.maximum(w1, 0.0)
    w2 = jnp.maximum(w2, 0.0)
    w3 = jnp.maximum(w3, 0.0)
    w4 = jnp.maximum(w4, 0.0)

    # Ensure weights sum to exactly 1.0 for gradient consistency
    # This enforces the mathematical constraint sum(weights) = 1.0, making gradients
    # of the sum exactly zero while preserving gradients of individual weights.
    # The sum is spelled out elementwise rather than reduced over a stacked axis: XLA
    # otherwise emits a reduction fusion that recomputes the whole weight graph.
    weight_sum = w1 + w2 + w3 + w4
    state = (ir1, i1_1, i1_2, w1 / weight_sum, w2 / weight_sum, w3 / weight_sum, w4 / weight_sum)

    # Pass the state through a barrier, with or without centers. The centers share
    # subexpressions with the weights, and XLA fuses those shared nodes differently once
    # they have a second consumer. The co-latitude weight divides by theta2 - theta1
    # ~ 1/nside, so a one-ulp change there is amplified by about nside, which would make
    # the weights depend on whether centers were requested. The barrier is also needed
    # without centers: since JAX 0.12, XLA otherwise fuses the weights into the stack
    # below with a different rounding than when they are materialized, and that fusion
    # is also about twice as slow on CPU.
    state = lax.optimization_barrier(state)
    ir1, i1_1, i1_2, w1, w2, w3, w4 = state
    weights = jnp.stack([w1, w2, w3, w4])

    # Second stage: pixel indices, from the ring index and the in-ring indices
    ir1 = lax.stop_gradient(ir1)
    is_north_pole = ir1 == 0
    is_south_pole = ir1 == 4 * nside - 1
    ir1_safe = jnp.maximum(ir1, 1)
    ir2_safe = jnp.minimum(ir1 + 1, 4 * nside - 1)
    sp1, nr1 = _start_pixel_ring(nside, ir1_safe), _npix_on_ring(nside, ir1_safe)
    sp2, nr2 = _start_pixel_ring(nside, ir2_safe), _npix_on_ring(nside, ir2_safe)

    # Eastern neighbours
    i2_1 = jnp.where(i1_1 == nr1 - 1, 0, i1_1 + 1)
    i2_2 = jnp.where(i1_2 == nr2 - 1, 0, i1_2 + 1)

    pixels_ring2_1 = sp2 + i1_2
    pixels_ring2_2 = sp2 + i2_2

    # Near a pole the missing ring is replaced by the cap ring pixels across the pole
    npix_total = 12 * nside * nside
    pixels_ring1_1 = jnp.where(is_north_pole, (pixels_ring2_1 + 2) & 3, sp1 + i1_1)
    pixels_ring1_2 = jnp.where(is_north_pole, (pixels_ring2_2 + 2) & 3, sp1 + i2_1)
    pixels_ring2_1 = jnp.where(is_south_pole, ((pixels_ring1_1 + 2) & 3) + npix_total - 4, pixels_ring2_1)
    pixels_ring2_2 = jnp.where(is_south_pole, ((pixels_ring1_2 + 2) & 3) + npix_total - 4, pixels_ring2_2)

    # Stop gradient: Pixel indices are discrete array selectors, not part of interpolation math
    pixels = lax.stop_gradient(jnp.stack([pixels_ring1_1, pixels_ring1_2, pixels_ring2_1, pixels_ring2_2]))

    if not with_centers:
        return pixels, weights

    # Co-latitude of the two rings. Safe in every branch: inside a cap the clamps above
    # collapse ir1_safe and ir2_safe onto the same ring, which is the ring the four
    # replacement pixels lie on, so the two entries simply coincide.
    z1, sth1 = _get_ring_costheta_sintheta(nside, ir1_safe)
    z2, sth2 = _get_ring_costheta_sintheta(nside, ir2_safe)

    # Longitude, read back from the final pixel indices. They are materialized through
    # a barrier: otherwise XLA recomputes them here and has to keep the eastern
    # neighbour indices around for both uses. The ring grid is only valid where the pole
    # branches did not replace the pixels. The four cap pixels sit at (k + 1/2) * pi/2,
    # and their index modulo 4 is k in both caps.
    pixels_c = lax.optimization_barrier(pixels)
    shift1 = jnp.where(_ring_shifted(nside, ir1_safe), 0.5, 0.0)
    shift2 = jnp.where(_ring_shifted(nside, ir2_safe), 0.5, 0.0)

    def longitude(pixel, start, npix_ring, shift, in_cap):
        on_grid = (pixel - start + shift) * (2.0 * jnp.pi / npix_ring)
        return jnp.where(in_cap, ((pixel & 3) + 0.5) * (0.5 * jnp.pi), on_grid)

    phi_c = jnp.stack(
        [
            longitude(pixels_c[0], sp1, nr1, shift1, is_north_pole),
            longitude(pixels_c[1], sp1, nr1, shift1, is_north_pole),
            longitude(pixels_c[2], sp2, nr2, shift2, is_south_pole),
            longitude(pixels_c[3], sp2, nr2, shift2, is_south_pole),
        ]
    )

    # Stop gradient: centers are fixed points of the HEALPix grid, piecewise constant in
    # the input coordinates. Keeping them constant is also what leaves an interpolation
    # operator built on them linear, hence transposable, in the map it interpolates.
    dtype = weights.dtype
    centers = lax.stop_gradient(
        InterpCenters(
            z=jnp.stack([z1, z2]).astype(dtype),
            sth=jnp.stack([sth1, sth2]).astype(dtype),
            phi=phi_c.astype(dtype),
        )
    )

    return pixels, weights, centers


@jit(static_argnames=['nest', 'lonlat'])
def get_interp_val(
    m: ArrayLike, theta: ArrayLike, phi: ArrayLike | None = None, nest: bool = False, lonlat: bool = False
) -> Array:
    r"""Interpolate maps at points of the sphere, bilinearly from the four nearest pixels.

    The interpolated value is $\sum_i w_i\, m[p_i]$, with the pixels $p_i$ and the weights $w_i$
    of [`get_interp_weights`][]. It is valid only for scalar (spin 0) fields: each map is
    interpolated on its own, without the parallel transport of the neighbours to the point, so
    that interpolating $Q$ and $U$ mixes them.

    !!! note "Difference from healpy"
        The NESTED ordering is not supported: `nest=True` raises a `ValueError`.

    Parameters
    ----------
    m : array-like
        The maps, in the RING ordering, of shape `(npix,)` or `(nmaps, npix)`.
    theta : float or array-like
        The colatitude, in radians, or with `lonlat=True` the longitude, in degrees. If `phi` is
        None, the pixels whose centers are the points.
    phi : float or array-like, optional
        The longitude, in radians, or with `lonlat=True` the latitude, in degrees.
    nest : bool, optional
        Must be False: the NESTED ordering is not supported.
    lonlat : bool, optional
        If True, `theta` and `phi` are the longitude and the latitude, in degrees.

    Returns
    -------
    Array
        The interpolated values, of shape `shape` for a single map and `(nmaps, *shape)`
        otherwise, where `shape` is the broadcast shape of `theta` and `phi`.

    Raises
    ------
    ValueError
        If `nest` is True.

    See Also
    --------
    get_interp_weights

    Examples
    --------
    At the center of a pixel, the interpolation is the value of the pixel:

    >>> import jax.numpy as jnp
    >>> import jax_healpy as hp
    >>> m = jnp.arange(12.0)
    >>> hp.get_interp_val(m, jnp.pi/2, 0.0)
    Array(4., dtype=float64)

    >>> theta = jnp.array([jnp.pi/4, jnp.pi/2])
    >>> phi = jnp.array([0.0, jnp.pi/2])
    >>> hp.get_interp_val(m, theta, phi)
    Array([1.5, 5. ], dtype=float64)

    >>> maps = jnp.array([jnp.arange(12.0), 2 * jnp.arange(12.0)])
    >>> hp.get_interp_val(maps, jnp.pi/2, 0.0)
    Array([4., 8.], dtype=float64)
    """
    if nest:
        raise ValueError('NEST pixel ordering is not supported. Only RING ordering is supported.')

    # Convert inputs to JAX arrays
    m = jnp.asarray(m)
    theta = jnp.asarray(theta)
    if phi is not None:
        phi = jnp.asarray(phi)

    # Determine nside from map size
    nside = npix2nside(m.shape[-1])

    # Handle multiple maps vs single map
    single_map = m.ndim == 1
    if single_map:
        map_data = m[jnp.newaxis, :]  # Add map dimension
    else:
        map_data = m

    # Get interpolation weights and pixels
    pixels, weights = get_interp_weights(nside, theta, phi, nest=nest, lonlat=lonlat)

    # Perform interpolation: sum(weights * map_values[pixels])
    # pixels shape: (4, ...) where ... is broadcast shape of theta, phi
    # weights shape: (4, ...)
    # map_data shape: (nmaps, npix)

    # Extract map values at interpolation pixels
    # map_values shape: (nmaps, 4, ...)
    map_values = map_data[..., pixels]  # Broadcasting: (nmaps, npix)[..., (4, ...)] -> (nmaps, 4, ...)

    # Compute weighted sum along the pixel dimension (axis=1)
    # result shape: (nmaps, ...)
    result = jnp.sum(weights[jnp.newaxis, ...] * map_values, axis=1)

    # If input was single map, remove the map dimension
    if single_map:
        result = result[0]

    return result


@jit(static_argnames=['nside', 'nest', 'lonlat', 'get_center'])
def get_all_neighbours(
    nside: int,
    theta: ArrayLike,
    phi: ArrayLike | None = None,
    nest: bool = False,
    lonlat: bool = False,
    get_center: bool = False,
) -> Array:
    """Return the 8 neighbours of pixels, or of the pixels that contain points of the sphere.

    The neighbours come in the order SW, W, NW, N, NE, E, SE, S. Some pixels at the corners of
    the base faces have only 7 neighbours: the missing one is -1.

    !!! note "Difference from healpy"
        `get_center=True`, which also returns the pixel itself, is specific to jax-healpy. `nside`
        is an integer, not an array, and the other arguments are arrays or scalars, not lists.

    Parameters
    ----------
    nside : int
        The resolution parameter, a power of 2.
    theta : int, float or array-like
        The pixels if `phi` is None. Otherwise, the colatitude of the points, in radians, or with
        `lonlat=True` their longitude, in degrees.
    phi : float or array-like, optional
        The longitude of the points, in radians, or with `lonlat=True` their latitude, in degrees.
    nest : bool, optional
        If True, use the NESTED ordering, otherwise the RING ordering.
    lonlat : bool, optional
        If True, `theta` and `phi` are the longitude and the latitude, in degrees.
    get_center : bool, optional
        If True, return the pixel itself before its neighbours.

    Returns
    -------
    Array
        The neighbours, of shape `(8, *shape)`, where `shape` is the shape of the pixels. With
        `get_center=True`, the pixel and its neighbours, of shape `(9, *shape)`.

    See Also
    --------
    get_interp_weights

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax_healpy as hp
    >>> print(hp.get_all_neighbours(1, 4))
    [11  7  3 -1  0  5  8 -1]

    >>> print(hp.get_all_neighbours(1, 4, get_center=True))
    [ 4 11  7  3 -1  0  5  8 -1]

    The neighbours of the pixel that contains a point:

    >>> print(hp.get_all_neighbours(1, jnp.pi/2, jnp.pi/2, get_center=True))
    [ 5  8  4  0 -1  1  6  9 -1]
    """
    # Validate inputs
    check_nside(nside, nest=nest)
    theta = jnp.asarray(theta)

    # Handle the two API modes: pixel indices vs angular coordinates
    if phi is None:
        # theta contains pixel indices
        ipix = theta.astype(_pixel_dtype_for(nside))
        input_shape = ipix.shape
    else:
        # theta, phi contain angular coordinates - convert to pixels
        phi = jnp.asarray(phi)
        if lonlat:
            theta, phi = _lonlat2thetaphi(theta, phi)

        # Ensure theta and phi can be broadcast together
        theta_bc, phi_bc = jnp.broadcast_arrays(theta, phi)
        input_shape = theta_bc.shape

        # Convert angular coordinates to pixel indices (preserves batch shape)
        ipix = ang2pix(nside, theta_bc, phi_bc, nest=nest)

    # Convert pixels to (x, y, face) coordinates (all element-wise, preserves batch shape)
    ix, iy, face_num = pix2xyf(nside, ipix, nest=nest)

    # Vectorized neighbor finding for all pixels — output shape (8, *input_shape)
    neighbors = _get_all_neighbors_xyf(nside, ix, iy, face_num, nest=nest)

    # Conditionally include center pixel based on get_center parameter
    if get_center:
        # Add center pixels as first element: [CENTER, SW, W, NW, N, NE, E, SE, S]
        # Combine center + neighbors: shape (9, *input_shape)
        result = jnp.concatenate([ipix[None, ...], neighbors], axis=0)
    else:
        result = neighbors

    if input_shape == ():
        return result.squeeze()
    return result


def _get_all_neighbors_xyf(nside: int, ix: Array, iy: Array, face_num: Array, nest: bool = False) -> Array:
    """Vectorized neighbor finding in (x, y, face) coordinates.

    This is the core neighbor-finding algorithm that handles face boundary crossings
    using the original HEALPix neighbor-finding methodology. It applies the 8-directional
    offsets to find potential neighbors, then handles cases where neighbors cross face
    boundaries using lookup tables and coordinate transformations.

    The algorithm follows these steps:
    1. Apply 8-directional offsets (_NB_XOFFSET, _NB_YOFFSET) to get neighbor coordinates
    2. Check which neighbors remain within the current face (valid range [0, nside-1])
    3. For neighbors that cross face boundaries, apply face transition logic using
       lookup tables (_NB_FACEARRAY, _NB_SWAPARRAY) based on original C++ implementation

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter (must be power of 2)
    ix, iy : Array
        Face-local x, y coordinates of pixels (shape: (*batch,))
        Valid range: [0, nside-1] for pixels within face
    face_num : Array
        Face numbers of pixels (shape: (*batch,))
        Valid range: [0, 11] for HEALPix faces
    nest : bool, optional
        Whether to use NESTED ordering scheme. Default is False (RING ordering).

    Returns
    -------
    neighbors : Array
        Neighbor pixel indices for each input pixel. Shape: (8, *batch)
        Neighbors in order: [SW, W, NW, N, NE, E, SE, S]
        Non-existent neighbors (at map boundaries) are marked with -1.

    Notes
    -----
    This function implements the exact neighbor-finding logic from the original
    HEALPix C++ library, ensuring bit-for-bit compatibility with healpy results.
    """
    batch_shape = ix.shape
    batch_ndim = len(batch_shape)

    # Initialize output array for neighbors
    neighbors = jnp.full((8,) + batch_shape, -1, dtype=_pixel_dtype_for(nside))

    # Apply 8-direction offsets to get neighbor coordinates
    # Reshape offsets to (8, 1, 1, ...) for broadcasting with (*batch,)
    offset_shape = (-1,) + (1,) * batch_ndim
    neighbor_ix = ix[None, ...] + _NB_XOFFSET.reshape(offset_shape)  # Shape: (8, *batch)
    neighbor_iy = iy[None, ...] + _NB_YOFFSET.reshape(offset_shape)  # Shape: (8, *batch)
    neighbor_face = jnp.broadcast_to(face_num[None, ...], (8,) + batch_shape)  # Shape: (8, *batch)

    # Check which neighbors are within the current face (no boundary crossing)
    # Valid range is [0, nside-1] for both ix and iy
    within_face = (
        (neighbor_ix >= 0) & (neighbor_ix < nside) & (neighbor_iy >= 0) & (neighbor_iy < nside)
    )  # Shape: (8, *batch)

    # For neighbors within face, convert directly to pixels
    valid_mask = within_face
    valid_neighbors = xyf2pix(nside, neighbor_ix, neighbor_iy, neighbor_face, nest=nest)  # Shape: (8, N)
    neighbors = jnp.where(valid_mask, valid_neighbors, neighbors)

    # Handle boundary crossings for neighbors outside current face
    boundary_mask = ~within_face  # Shape: (8, N)

    # For boundary pixels, we need to use the lookup tables
    # This is complex due to the face transition logic - we'll implement a simplified version
    # that handles the most common boundary cases

    # Apply face boundary corrections using lookup tables
    corrected_neighbors = _handle_face_boundaries(nside, neighbor_ix, neighbor_iy, face_num, nest)

    # Use corrected neighbors where we have boundary crossings
    neighbors = jnp.where(boundary_mask, corrected_neighbors, neighbors)

    return neighbors


def _handle_face_boundaries(
    nside: int, neighbor_ix: Array, neighbor_iy: Array, original_face: Array, nest: bool
) -> Array:
    """Handle neighbor pixels that cross face boundaries.

    This implements the exact HEALPix face transition logic using lookup tables,
    based on the original C++ implementation in healpix_base.cc. When a neighbor
    coordinate falls outside the current face boundaries, this function determines
    the correct face and applies coordinate transformations.

    The algorithm follows these steps for each boundary crossing:
    1. Detect boundary crossing condition (x < 0, x >= nside, y < 0, y >= nside)
    2. Calculate nbnum index encoding the crossing direction
    3. Look up new face using _NB_FACEARRAY[nbnum, original_face]
    4. Apply coordinate corrections (wrap coordinates to valid range)
    5. Apply bit-based transformations using _NB_SWAPARRAY (flip x, flip y, swap x/y)
    6. Convert corrected (x, y, face) back to pixel indices

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter
    neighbor_ix, neighbor_iy : Array
        Neighbor coordinates that may be outside face boundaries. Shape: (8, *batch)
    original_face : Array
        Original face numbers of input pixels. Shape: (*batch,)
    nest : bool
        Whether to use NESTED ordering

    Returns
    -------
    corrected_neighbors : Array
        Corrected neighbor pixel indices. Shape: (8, *batch)
        Returns -1 for invalid neighbors (outside map boundaries)

    Notes
    -----
    This function is a direct translation of the original HEALPix C++ neighbor
    finding algorithm, ensuring exact compatibility with healpy. The lookup tables
    (_NB_FACEARRAY, _NB_SWAPARRAY) encode the complex geometric relationships
    between HEALPix faces and handle all 12 face transitions correctly.
    """
    batch_shape = original_face.shape

    # Initialize result with invalid neighbors
    result = jnp.full((8,) + batch_shape, -1, dtype=_pixel_dtype_for(nside))

    # Process each neighbor direction individually
    for direction_idx in range(8):
        # Get coordinates for this direction across all pixels
        ix = neighbor_ix[direction_idx]  # Shape: (*batch,)
        iy = neighbor_iy[direction_idx]  # Shape: (*batch,)
        orig_face = original_face  # Shape: (*batch,)

        # Check boundary conditions - exact replication of original algorithm
        x_low = ix < 0
        x_high = ix >= nside
        y_low = iy < 0
        y_high = iy >= nside

        # Any pixel crossing face boundary
        boundary_crossing = x_low | x_high | y_low | y_high

        # Initialize corrected coordinates with original values
        corrected_ix = ix
        corrected_iy = iy

        # Apply boundary corrections exactly as in original C++ code
        # First handle x boundary crossings
        corrected_ix = jnp.where(x_low, corrected_ix + nside, corrected_ix)
        corrected_ix = jnp.where(x_high, corrected_ix - nside, corrected_ix)

        # Then handle y boundary crossings
        corrected_iy = jnp.where(y_low, corrected_iy + nside, corrected_iy)
        corrected_iy = jnp.where(y_high, corrected_iy - nside, corrected_iy)

        # Calculate nbnum index for lookup tables (matches original C++ logic)
        nbnum = 4  # Start with center case
        nbnum = jnp.where(x_low, nbnum - 1, nbnum)
        nbnum = jnp.where(x_high, nbnum + 1, nbnum)
        nbnum = jnp.where(y_low, nbnum - 3, nbnum)
        nbnum = jnp.where(y_high, nbnum + 3, nbnum)

        # Look up new face using the face array (vectorized)
        # Use advanced indexing to get new faces for each pixel
        new_face = jnp.asarray(_NB_FACEARRAY)[nbnum, orig_face]

        # Only process pixels that actually cross boundaries and have valid new faces
        valid_crossing = boundary_crossing & (new_face >= 0) & (new_face < 12)

        # Apply coordinate transformations using swap array bits
        # Get swap bits for face transitions (vectorized)
        swap_bits = jnp.asarray(_NB_SWAPARRAY)[nbnum, orig_face >> 2]

        # Apply bit transformations exactly as in original C++
        # Bit 1: Flip x coordinate
        flip_x = (swap_bits & 1) != 0
        corrected_ix = jnp.where(valid_crossing & flip_x, nside - corrected_ix - 1, corrected_ix)

        # Bit 2: Flip y coordinate
        flip_y = (swap_bits & 2) != 0
        corrected_iy = jnp.where(valid_crossing & flip_y, nside - corrected_iy - 1, corrected_iy)

        # Bit 4: Swap x and y coordinates
        swap_xy = (swap_bits & 4) != 0
        new_x = jnp.where(valid_crossing & swap_xy, corrected_iy, corrected_ix)
        new_y = jnp.where(valid_crossing & swap_xy, corrected_ix, corrected_iy)
        corrected_ix = new_x
        corrected_iy = new_y

        # Use new face for valid crossings, original face otherwise
        corrected_face = jnp.where(valid_crossing, new_face, orig_face)

        # Convert to pixel indices
        neighbor_pixels = xyf2pix(nside, corrected_ix, corrected_iy, corrected_face, nest=nest)

        # Update result for this direction - only valid crossings get neighbor pixels
        result = result.at[direction_idx].set(jnp.where(valid_crossing, neighbor_pixels, -1))

    return result


# Note: Removed _get_adjacent_face - using lookup table directly in _handle_face_boundaries


def get_nside(m: ArrayLike) -> int:
    """Return the resolution parameter of a map, from its number of pixels.

    Parameters
    ----------
    m : array-like
        A map, of shape `(npix,)`, or maps, of shape `(nmaps, npix)`.

    Returns
    -------
    int
        The resolution parameter.

    Raises
    ------
    ValueError
        If `m` has more than two dimensions, or if its number of pixels is not that of a map
        (see [`isnpixok`][]).

    See Also
    --------
    npix2nside

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> hp.get_nside(np.zeros(768))
    8
    """
    m = jnp.asarray(m)
    if m.ndim == 1:
        npix = len(m)
    elif m.ndim == 2:
        npix = m.shape[-1]  # Last dimension should be pixels
    else:
        raise ValueError(f'Map must be 1D or 2D, got shape {m.shape}')

    return npix2nside(npix)


def mask_bad(m: ArrayLike, badval: float = UNSEEN, rtol: float = 1.0e-5, atol: float = 1.0e-8) -> Array:
    r"""Return a boolean mask, True where the map is close to `badval`.

    A pixel is bad where $|m - \mathrm{badval}| \leq \mathrm{atol} + \mathrm{rtol}\,
    |\mathrm{badval}|$. A NaN pixel is not bad.

    Parameters
    ----------
    m : array-like
        The map.
    badval : float, optional
        The value of the bad pixels, [`UNSEEN`][jax_healpy.pixelfunc.UNSEEN] by default.
    rtol, atol : float, optional
        The relative and absolute tolerances.

    Returns
    -------
    Array
        The mask, True where `m` is close to `badval`, with the shape of `m`.

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> print(hp.mask_bad(np.array([1.0, hp.UNSEEN, np.nan])))
    [False  True False]
    """
    m = jnp.asarray(m)
    atol = abs(atol)
    rtol = abs(rtol)
    return jnp.abs(m - badval) <= atol + rtol * abs(badval)


@jit(static_argnames=['nside_out', 'pess', 'order_in', 'order_out', 'power', 'dtype'])
def ud_grade(
    map_in: ArrayLike,
    nside_out: int,
    pess: bool = False,
    order_in: str = 'RING',
    order_out: str | None = None,
    power: float | None = None,
    dtype: type | None = None,
) -> Array:
    r"""Change the resolution of maps.

    Upgrading gives each pixel of the output the value of the pixel of the input that contains it.
    Degrading gives each pixel of the output the mean of the pixels of the input that it contains,
    ignoring the [`UNSEEN`][jax_healpy.pixelfunc.UNSEEN] pixels.

    This change of resolution creates artifacts in the power spectra: when possible, changing the
    band limit of the spherical harmonic coefficients is preferable.

    Parameters
    ----------
    map_in : array-like
        The maps, of shape `(npix,)` or `(nmaps, npix)`.
    nside_out : int
        The resolution parameter of the output, a power of 2.
    pess : bool, optional
        When degrading, if True, a pixel of the output is UNSEEN if any of the pixels of the input
        that it contains is UNSEEN; if False, only if all of them are.
    order_in : {'RING', 'NESTED'}, optional
        The ordering of the input maps.
    order_out : {'RING', 'NESTED'}, optional
        The ordering of the output maps, `order_in` by default.
    power : float, optional
        If given, the values of the output are multiplied by
        $(n_\mathrm{side,out} / n_\mathrm{side,in})^\mathrm{power}$. With `power=-2`, the sum of
        the map is kept, as for a map of hits.
    dtype : data type, optional
        The data type of the output, that of the input by default.

    Returns
    -------
    Array
        The maps at the resolution `nside_out`, of shape `(12 * nside_out**2,)` or
        `(nmaps, 12 * nside_out**2)`.

    Raises
    ------
    ValueError
        If `nside_out`, or the resolution parameter of the input maps, is not a power of 2.

    Examples
    --------
    In the NESTED ordering, the four pixels contained in a pixel of a lower resolution are
    consecutive:

    >>> import numpy as np
    >>> import jax_healpy as hp
    >>> print(hp.ud_grade(np.arange(48.0), 1, order_in='NESTED'))
    [ 1.5  5.5  9.5 13.5 17.5 21.5 25.5 29.5 33.5 37.5 41.5 45.5]

    >>> print(hp.ud_grade(np.arange(12.0), 2, order_in='NESTED')[:8])
    [0. 0. 0. 0. 1. 1. 1. 1.]
    """
    # Early validation to provide clear error messages
    # udgrade requires power-of-2 nside values regardless of ordering scheme
    if not isnsideok(nside_out, nest=True):
        raise ValueError(
            f'{nside_out} is not a valid nside parameter for udgrade (must be a power of 2, less than 2**30)'
        )

    # Convert input to JAX array and handle map format
    map_in = jnp.asarray(map_in)
    is_single_map = map_in.ndim == 1

    # Ensure we work with 2D array (n_maps, npix)
    if is_single_map:
        maps = map_in[None, :]  # Add map dimension
    else:
        maps = map_in

    # Get input nside and validate
    nside_in = get_nside(maps[0])  # Use first map to get nside
    if not isnsideok(nside_in, nest=True):
        raise ValueError(
            f'{nside_in} is not a valid nside parameter for udgrade (must be a power of 2, less than 2**30)'
        )

    # Determine output ordering
    if order_out is None:
        order_out = order_in

    # Call the core implementation
    return _ud_grade_core(maps, nside_in, nside_out, pess, order_in, order_out, power, dtype, is_single_map)


def _ud_grade_core(
    maps: Array,
    nside_in: int,
    nside_out: int,
    pess: bool,
    order_in: str,
    order_out: str,
    power: float,
    dtype: type,
    is_single_map: bool,
) -> Array:
    """Core udgrade implementation for processing multiple maps."""
    npix_in = nside2npix(nside_in)
    npix_out = nside2npix(nside_out)

    # Determine output dtype
    if dtype is not None:
        output_dtype = dtype
    else:
        output_dtype = maps.dtype

    # Step 1: Convert to NESTED if needed (reorder handles batch dimension)
    if order_in == 'RING':
        maps = reorder(maps, r2n=True)

    # Step 2: Core resolution change in NESTED scheme
    if nside_out == nside_in:
        # No change needed
        result = maps
    elif nside_out > nside_in:
        # UPGRADE: replicate parent pixels to children
        rat2 = npix_out // npix_in

        # Apply power scaling if specified
        if power is not None:
            ratio = (jnp.float32(nside_out) / jnp.float32(nside_in)) ** jnp.float32(power)
        else:
            ratio = 1.0

        # Replicate each pixel value to its children using broadcasting
        # maps shape: (n_maps, npix_in)
        # fact shape: (rat2,)
        # outer product: (n_maps, npix_in, rat2) -> reshape to (n_maps, npix_out)
        fact = jnp.ones(rat2, dtype=output_dtype) * ratio
        # Use broadcasting: maps[..., :, None] * fact[None, None, :]
        expanded_maps = maps[..., :, None] * fact  # (n_maps, npix_in, rat2)
        result = expanded_maps.reshape(maps.shape[0], npix_out)

    else:
        # DEGRADE: average children pixels to parent
        rat2 = npix_in // npix_out

        # Reshape to group children pixels: (n_maps, npix_out, rat2)
        reshaped_maps = maps.reshape(maps.shape[0], npix_out, rat2)

        # Create mask for valid pixels (not UNSEEN and finite)
        goods = ~(mask_bad(reshaped_maps) | (~jnp.isfinite(reshaped_maps)))

        # Sum valid pixels along children axis
        map_sum = jnp.sum(reshaped_maps * goods, axis=-1)  # (n_maps, npix_out)
        n_good = jnp.sum(goods, axis=-1)  # (n_maps, npix_out)

        # Determine which output pixels should be UNSEEN
        if pess:
            # Pessimistic: mark UNSEEN if ANY child is bad
            badout = n_good != rat2
        else:
            # Optimistic: mark UNSEEN only if ALL children are bad
            badout = n_good == 0

        # Apply power scaling if specified
        if power is not None:
            ratio = (jnp.float32(nside_out) / jnp.float32(nside_in)) ** jnp.float32(power)
            n_good = n_good / ratio

        # Calculate averages for pixels with valid children
        result = jnp.where(
            n_good > 0,
            map_sum / n_good,
            0.0,  # Temporary value, will be set to UNSEEN below
        )

        # Set UNSEEN pixels
        result = jnp.where(badout, UNSEEN, result)

    # Step 3: Convert back to desired output ordering (reorder handles batch dimension)
    if order_out == 'RING' and order_in == 'NESTED':
        result = reorder(result, n2r=True)
    elif order_out == 'NESTED' and order_in == 'RING':
        # Map was converted to NESTED in step 1, keep it
        pass
    elif order_out == order_in == 'RING':
        # Convert back since we changed it in step 1
        result = reorder(result, n2r=True)

    # Apply output dtype
    result = result.astype(output_dtype)

    if is_single_map:
        return result[0]  # Remove the added map dimension
    else:
        return result
