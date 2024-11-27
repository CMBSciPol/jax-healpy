from functools import partial, wraps
from typing import Callable, ParamSpec, TypeVar

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

try:
    from s2fft.recursions.price_mcewen import generate_precomputes_jax
    from s2fft.sampling.reindex import flm_2d_to_hp_fast, flm_hp_to_2d_fast
    from s2fft.transforms import spherical
except ImportError:
    pass

from jax_healpy import npix2nside

__all__ = [
    'alm2map',
    'map2alm',
]

Param = ParamSpec('Param')
ReturnType = TypeVar('ReturnType')


def requires_s2fft(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    try:
        import s2fft  # noqa

        return func
    except ImportError:
        pass

    @wraps(func)
    def deferred_func(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        msg = "Missing optional library 's2fft', part of the 'recommended' dependency group."
        raise ImportError(msg)

    return deferred_func


@partial(
    jax.jit,
    static_argnames=[
        'nside',
        'lmax',
        'mmax',
        'pixwin',
        'fwhm',
        'sigma',
        'pol',
        'inplace',
        'verbose',
        'healpy_ordering',
    ],
)
@requires_s2fft
def alm2map(
    alms: ArrayLike,
    nside: int,
    lmax=None,
    mmax=None,
    pixwin=False,
    fwhm=0.0,
    sigma=None,
    pol=True,
    inplace=False,
    verbose=True,
    healpy_ordering: bool = False,
):
    """Computes a Healpix map given the alm.

    The alm are given as a complex array. You can specify lmax
    and mmax, or they will be computed from array size (assuming
    lmax==mmax).

    Parameters
    ----------
    alms : complex, array or sequence of arrays
      A complex array or a sequence of complex arrays.
      Each array must have a size of the form: mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1
    nside : int, scalar
      The nside of the output map.
    lmax : None or int, scalar, optional
      Explicitly define lmax (needed if mmax!=lmax)
    mmax : None or int, scalar, optional
      Explicitly define mmax (needed if mmax!=lmax)
    pixwin : bool, optional
      Smooth the alm using the pixel window functions. Default: False.
    fwhm : float, scalar, optional
      The fwhm of the Gaussian used to smooth the map (applied on alm)
      [in radians]
    sigma : float, scalar, optional
      The sigma of the Gaussian used to smooth the map (applied on alm)
      [in radians]
    pol : bool, optional
      If True, assumes input alms are TEB. Output will be TQU maps.
      (input must be 1 or 3 alms)
      If False, apply spin 0 harmonic transform to each alm.
      (input can be any number of alms)
      If there is only one input alm, it has no effect. Default: True.
    inplace : bool, optional
      If True, input alms may be modified by pixel window function and beam
      smoothing (if alm(s) are complex128 contiguous arrays).
      Otherwise, input alms are not modified. A copy is made if needed to
      apply beam smoothing or pixel window.
    healpy ordering : bool, optional
      True if the input alms follow the healpy ordering. By default, the s2fft
      ordering is assumed.

    Returns
    -------
    maps : array or list of arrays
      A Healpix map in RING scheme at nside or a list of T,Q,U maps (if
      polarized input)

    Notes
    -----
    Running map2alm then alm2map will not return exactly the same map if the discretized field you construct on the
    sphere is not band-limited (for example, if you have a map containing pixel-based noise rather than beam-smoothed
    noise). If you need a band-limited map, you have to start with random numbers in lm space and transform these via
    alm2map. With such an input, the accuracy of map2alm->alm2map should be quite good, depending on your choices
    of lmax, mmax and nside (for some typical values, see e.g., section 5.1 of https://arxiv.org/pdf/1010.2084).
    """
    if mmax is not None:
        raise NotImplementedError('Specifying mmax is not implemented.')
    if pixwin:
        raise NotImplementedError('Specifying pixwin is not implemented.')
    if fwhm != 0:
        raise NotImplementedError('Specifying fwhm is not implemented.')
    if sigma is not None:
        raise NotImplementedError('Specifying sigma is not implemented.')
    alms = jnp.asarray(alms)
    if alms.ndim == 0:
        raise ValueError('Input alms must have at least one dimension.')
    expected_ndim = 1 if healpy_ordering else 2
    if alms.ndim > expected_ndim + 1 + pol:
        raise ValueError('Input alms have too many dimensions.')
    if alms.ndim == expected_ndim + 1 + pol:
        return jax.vmap(alm2map, in_axes=(0,) + 10 * (None,))(
            alms,
            nside,
            lmax,
            mmax,
            pixwin,
            fwhm,
            sigma,
            pol,
            inplace,
            False,
            healpy_ordering,
        )
    if alms.ndim > expected_ndim:
        # only happens if pol=True
        raise NotImplementedError('TEB alms are not implemented.')

    if lmax is None:
        L = 3 * nside
    else:
        L = lmax + 1

    if healpy_ordering:
        alms = flm_hp_to_2d_fast(alms, L)

    sampling = 'healpix'
    method = 'jax'
    spmd = False

    precomps = generate_precomputes_jax(L, 0, sampling, nside, False)
    f = spherical.inverse(
        alms,
        L,
        spin=0,
        nside=nside,
        sampling=sampling,
        method=method,
        reality=True,
        precomps=precomps,
        spmd=spmd,
    )
    return f


@partial(
    jax.jit,
    static_argnames=[
        'lmax',
        'mmax',
        'iter',
        'pol',
        'use_weights',
        'datapath',
        'gal_cut',
        'use_pixel_weights',
        'verbose',
        'healpy_ordering',
    ],
)
@requires_s2fft
def map2alm(
    maps,
    lmax=None,
    mmax=None,
    iter=3,
    pol=True,
    use_weights=False,
    datapath=None,
    gal_cut=0,
    use_pixel_weights=False,
    verbose=True,
    healpy_ordering: bool = False,
):
    """Computes the alm of a Healpix map. The input maps must all be
    in ring ordering.

    For recommendations about how to set `lmax`, `iter`, and weights, see the
    `Anafast documentation <https://healpix.sourceforge.io/html/fac_anafast.htm>`_

    Pixel values are weighted before applying the transform:

    * when you don't specify any weights, the uniform weight value 4*pi/n_pix is used
    * with ring weights enabled (use_weights=True), pixels in every ring
      are weighted with a uniform value similar to the one above, ring weights are
      included in healpy
    * with pixel weights (use_pixel_weights=True), every pixel gets an individual weight

    Pixel weights provide the most accurate transform, so you should always use them if
    possible. However they are not included in healpy and will be automatically downloaded
    and cached in ~/.astropy the first time you compute a trasform at a specific nside.

    If datapath is specified, healpy will first check that local folder before downloading
    the weights.
    The easiest way to setup the folder is to clone the healpy-data repository:

    git clone --depth 1 https://github.com/healpy/healpy-data
    cd healpy-data
    bash download_weights_8192.sh

    and set datapath to the root of the repository.

    Parameters
    ----------
    maps : array-like, shape (Npix,) or (n, Npix)
      The input map or a list of n input maps. Must be in ring ordering.
    lmax : int, scalar, optional
      Maximum l of the power spectrum. Default: 3*nside-1
    mmax : int, scalar, optional
      Maximum m of the alm. Default: lmax
    iter : int, scalar, optional
      Number of iteration (default: 3)
    pol : bool, optional
      If True, assumes input maps are TQU. Output will be TEB alm's.
      (input must be 1 or 3 maps)
      If False, apply spin 0 harmonic transform to each map.
      (input can be any number of maps)
      If there is only one input map, it has no effect. Default: True.
    use_weights: bool, scalar, optional
      If True, use the ring weighting. Default: False.
    datapath : None or str, optional
      If given, the directory where to find the pixel weights.
      See in the docstring above details on how to set it up.
    gal_cut : float [degrees]
      pixels at latitude in [-gal_cut;+gal_cut] are not taken into account
    use_pixel_weights: bool, optional
      If True, use pixel by pixel weighting, healpy will automatically download the weights, if needed
    verbose : bool, optional
      Deprecated, has not effect.
    healpy_ordering : bool, optional
      By default, we follow the s2fft ordering for the alms. To use healpy
      ordering, set it to True.

    Returns
    -------
    alms : array or tuple of array
      alm or a tuple of 3 alm (almT, almE, almB) if polarized input.

    Notes
    -----
    The pixels which have the special `UNSEEN` value are replaced by zeros
    before spherical harmonic transform. They are converted back to `UNSEEN`
    value, so that the input maps are not modified. Each map have its own,
    independent mask.
    """
    if mmax is not None:
        raise NotImplementedError('Specifying mmax is not implemented.')
    if iter != 0:
        raise NotImplementedError('Specifying iter > 0 is not implemented')
    if use_weights:
        raise NotImplementedError('Specifying use_weights is not implemented.')
    if datapath is not None:
        raise NotImplementedError('Specifying datapath is not implemented.')
    if gal_cut != 0:
        raise NotImplementedError('Specifying gal_cut is not implemented.')
    if use_pixel_weights:
        raise NotImplementedError('Specifying use_pixel_weights is not implemented.')
    if maps.ndim == 0:
        raise ValueError('The input map must have at least one dimension.')
    if maps.ndim > 2:
        raise ValueError('The input map has too many dimensions.')
    if maps.ndim > 1:
        if pol:
            raise NotImplementedError('TQU maps are not implemented.')
        return jax.vmap(map2alm, in_axes=(0,) + 10 * (None,))(
            maps,
            lmax,
            mmax,
            iter,
            pol,
            use_weights,
            datapath,
            gal_cut,
            use_pixel_weights,
            False,
            healpy_ordering,
        )

    maps = jnp.asarray(maps)
    nside = npix2nside(maps.shape[-1])
    if lmax is None:
        L = 3 * nside
    else:
        L = lmax + 1

    sampling = 'healpix'
    method = 'jax'
    spmd = False

    precomps = generate_precomputes_jax(L, 0, sampling, nside, True)
    flm = spherical.forward(
        maps,
        L,
        spin=0,
        nside=nside,
        sampling=sampling,
        method=method,
        reality=True,
        precomps=precomps,
        spmd=spmd,
    )

    if healpy_ordering:
        return flm_2d_to_hp_fast(flm, L)

    return flm
