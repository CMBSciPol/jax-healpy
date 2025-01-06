from functools import partial, wraps
from typing import Callable, ParamSpec, TypeVar

import healpy as hp
import jax
import jax.lax as jlax
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
    'precompute_temperature_harmonic_transforms',
    'precompute_polarization_harmonic_transforms',
    '_map2alm_pol',
    '_alm2map_pol',
    'almxfl',
    'alm2cl',
    'synalm',
    'synfast'
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

@requires_s2fft
def _inverse(
    alms: ArrayLike, 
    L: int, 
    spin: int, 
    nside: int, 
    sampling: str, 
    reality: bool, 
    precomps: list, 
    fast_non_differentiable: bool = False,
    spmd: bool = False
    ): 
    """ 
    Wrapper over the spherical.inverse s2fft function, inverse spin-spherical harmonic transform 
    to obtain pixelized map from spherical harmonic coefficients. 

    Parameters
    ----------
    alms: complex, array or sequence of arrays
      A complex array or a sequence of complex arrays.
      Each array must have a size of the form: mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1 
    L: int 
      Harmonic band-limit, according to s2fft convention 
    spin: int 
      Harmonic spin 
    nside: int, scalar
      The nside of the output map.
    sampling: str
      Sampling scheme, taken to be "healpix"
    reality: bool
      If the signal in the pixelized maps is real or not. If so, s2fft is able to reduce computational costs. 
    precomps: list[np.ndarray]
      Precomputed list of recursion coefficients. At most of length, which is a minimal memory overhead.
    fast_non_differentiable: bool
      If the SHT is done in a faster jittable way, but not JAX-differentiable. 
    spmd: bool
      Use the spmd option of s2fft, to map over multiple devies.  
    
    Returns
    -------
    Pixelized maps of the provided nside, containing the signal given by the alms

    Notes
    -------
    This function uses the jax_healpy method of s2fft to compute the inverse spin-spherical harmonic transform
    on CPU, and jax on GPU. 
    """ 

    inverse_s2fft = lambda flm, method: spherical.inverse(
      flm=flm,
      L=L,
      spin=spin,
      nside=nside,
      sampling=sampling,
      method=method,
      reality=reality,
      precomps=precomps,
      spmd=spmd
    )

    if fast_non_differentiable:
      lmax = L - 1 

      if alms.ndim == 3:
        nstokes = alms.shape[0]
      else:
        nstokes = 1

      # Wrapper for alm2map, to prepare the pure callback of JAX
      def wrapper_alm2map(alm_, lmax=lmax, nside=nside):
          alm_np = jax.tree.map(np.asarray, alm_)
          return hp.alm2map(alm_np, nside, lmax=lmax).reshape((nstokes, 12 * nside**2))

      @partial(jax.jit, static_argnums=(1, 2))
      def pure_call_alm2map(alm_, lmax, nside):
        alm_healpy = flm_2d_to_hp_fast(alm_, lmax)
        shape_output = (nstokes, 12 * nside**2)
        return jax.pure_callback(wrapper_alm2map, jax.ShapeDtypeStruct(shape_output, np.float64), alm_).squeeze()
      inverse_cpu = partial(pure_call_alm2map, nside=nside, lmax=lmax)
      inverse_gpu = partial(inverse_s2fft, method='cuda')  
    else:
      inverse_cpu = partial(inverse_s2fft, method='jax')
      inverse_gpu = partial(inverse_s2fft, method='jax')

    return jlax.platform_dependent(alms, cpu=inverse_cpu, cuda=inverse_gpu) 

@requires_s2fft
def _forward(
    maps: ArrayLike, 
    L: int, 
    spin: int, 
    nside: int, 
    sampling: str, 
    reality: bool, 
    precomps: list, 
    iter: int = 3,
    fast_non_differentiable: bool = False,
    spmd: bool = False
    ): 
    """ 
    Wrapper over the spherical.forward s2fft function, forward spin-spherical harmonic transform 
    to obtain spherical harmonic coefficients from a pixelized map. 

    Parameters
    ----------
    map: complex, array or sequence of arrays
      Pixelized maps of the provided nside
    L: int 
      Harmonic band-limit, according to s2fft convention 
    spin: int 
      Harmonic spin 
    nside: int, scalar
      The nside of the output map.
    sampling: str
      Sampling scheme, taken to be "healpix"
    reality: bool
      If the signal in the pixelized maps is real or not. If so, s2fft is able to reduce computational costs. 
    precomps: list[np.ndarray]
      Precomputed list of recursion coefficients. At most of length, which is a minimal memory overhead.
    iter: int, scalar, optional
      Number of spherical harmonics iteration for regularisation, only relevant if fast_non_differentiable and on CPU (default: 3)
    fast_non_differentiable: bool
      If the SHT is done in a faster jittable way, but not JAX-differentiable.
    spmd: bool
      Use the spmd option of s2fft, to map over multiple devies.  
    
    Returns
    -------
    Spherical harmonic coefficients 

    Notes
    -------
    This function uses the jax_healpy method of s2fft to compute the inverse spin-spherical harmonic transform
    on CPU, and jax on GPU. 
    """ 

    forward_s2fft = lambda maps, method: spherical.forward(
      f=maps,
      L=L,
      spin=spin,
      nside=nside,
      sampling=sampling,
      method=method,
      reality=reality,
      precomps=precomps,
      spmd=spmd
    )

    if fast_non_differentiable:
      lmax = L - 1 

      if alms.ndim == 3:
        nstokes = alms.shape[0]
      else:
        nstokes = 1

      forward_cpu = partial(pure_call_alm2map, nside=nside, lmax=lmax)
      forward_gpu = partial(forward_s2fft, method='cuda')  

      # Wrapper for map2alm, to prepare the pure callback of JAX
      def wrapper_map2alm(maps_, lmax=lmax, iter=iter, nside=nside):
          maps_np = jax.tree.map(np.asarray, maps_).reshape((nstokes, 12 * nside**2))
          alm_T, alm_E, alm_B = hp.map2alm(maps_np, lmax=lmax, iter=iter)
          return np.array([alm_T, alm_E, alm_B]).reshape((nstokes, (lmax + 1) * (lmax // 2 + 1)))

      # Pure call back of map2alm, to be used with JAX for JIT compilation
      @partial(jax.jit, static_argnums=(1, 2))
      def pure_call_map2alm(maps_, lmax, nside):
          shape_output = (nstokes, (lmax + 1) * (lmax // 2 + 1))
          alm_hp = jax.pure_callback(
              wrapper_map2alm,
              jax.ShapeDtypeStruct(shape_output, np.complex128),
              maps_.ravel(),
          ).squeeze()
          return flm_hp_to_2d_fast(alm_hp, lmax)

    else:
      forward_cpu = partial(forward_s2fft, method='jax')
      forward_gpu = partial(forward_s2fft, method='jax')

    return jlax.platform_dependent(maps, cpu=forward_cpu, cuda=forward_gpu) 


def precompute_temperature_harmonic_transforms(
    nside: int, 
    lmax: int = None, 
    sampling: str = "healpix", 
    pix2harm: bool = False
    ): 
    """
    Pre-compute recursion coefficient for s2fft functions when they are used with spin = 0, 
    so for transforms involving the intensity maps or harmonic coefficients. Only relevant if the "jax" method is used. 

    Parameters
    ----------
    nside: int, scalar
      nside of the output map.
    lmax: int, scalar
      maximum multipole for spherical harmonic computations
    sampling: str
      Sampling scheme, taken to be "healpix"
    pix2harm: bool
      if coefficients are computed for a forward operation (map2alm) or an inverse operation (alm2map), default False
    
    Returns
    -------
    List of pre-computed coefficients for spin = 0 stored as np.ndarray
    """

    if lmax is None:
      L = 3 * nside
    else:
      L = lmax + 1 

    return generate_precomputes_jax(L, spin=0, sampling=sampling, nside=nside, forward=pix2harm) 
    
def precompute_polarization_harmonic_transforms(nside: int, 
    lmax: int = None, 
    sampling: str = "healpix", 
    pix2harm: bool = False
    ): 
    """
    Pre-compute recursion coefficient for s2fft functions when they are used with spin = 2 or -2, 
    so for transforms involving the polarization maps or harmonic coefficients. Only relevant if the "jax" method is used. 

    Parameters
    ----------
    nside: int, scalar
      nside of the output map.
    lmax: int, scalar
      maximum multipole for spherical harmonic computations
    sampling: str
      Sampling scheme, taken to be "healpix"
    pix2harm: bool
      if coefficients are computed for a forward operation (map2alm) or an inverse operation (alm2map), default False
    
    Returns
    -------
    Tuple of the two list of pre-computed coefficients, respectively for spin = 2 and spin = -2, stored as np.ndarray
    """
    if lmax is None:
      L = 3 * nside
    else:
      L = lmax + 1 

    precomps_plus2 = generate_precomputes_jax(L, spin=2, sampling=sampling, nside=nside, forward=pix2harm)
    precomps_minus2 = generate_precomputes_jax(L, spin=-2, sampling=sampling, nside=nside, forward=pix2harm)
    return precomps_plus2, precomps_minus2 


@partial(
    jax.jit,
    static_argnames=[
        'nside',
        'lmax',
        'mmax',
        'pixwin',
        'fwhm',
        'sigma',
        'inplace',
        'verbose',
        'healpy_ordering',
    ],
)
@requires_s2fft
def _alm2map_pol(
    alms: ArrayLike,
    nside: int,
    lmax: int = None,
    mmax: int = None,
    pixwin: bool = False,
    fwhm: float = 0.0,
    sigma: float = None,
    pol: bool = True,
    inplace: bool = False,
    verbose: bool = True,
    healpy_ordering: bool = False, 
    precomps_polar: list = None, 
    fast_non_differentiable: bool = False
):
    """Computes a Healpix map given the alm.

    The alm are given as a complex array. You can specify lmax
    and mmax, or they will be computed from array size (assuming
    lmax==mmax).

    Parameters
    ----------
    alms: complex, array or sequence of arrays
      A complex array or a sequence of complex arrays.
      Each array must have a size of the form: mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1
    nside: int, scalar
      The nside of the output map.
    lmax: None or int, scalar, optional
      Explicitly define lmax (needed if mmax!=lmax)
    mmax: None or int, scalar, optional
      Explicitly define mmax (needed if mmax!=lmax)
    pixwin: bool, optional
      Smooth the alm using the pixel window functions. Default: False.
    fwhm: float, scalar, optional
      The fwhm of the Gaussian used to smooth the map (applied on alm)
      [in radians]
    sigma: float, scalar, optional
      The sigma of the Gaussian used to smooth the map (applied on alm)
      [in radians]
    pol: bool, optional
      If True, assumes input alms are TEB. Output will be TQU maps.
      (input must be 1 or 3 alms)
      If False, apply spin 0 harmonic transform to each alm.
      (input can be any number of alms)
      If there is only one input alm, it has no effect. Default: True.
    inplace: bool, optional
      If True, input alms may be modified by pixel window function and beam
      smoothing (if alm(s) are complex128 contiguous arrays).
      Otherwise, input alms are not modified. A copy is made if needed to
      apply beam smoothing or pixel window.
    healpy ordering: bool, optional
      True if the input alms follow the healpy ordering. By default, the s2fft
      ordering is assumed.
    precomps_polar: list of np.ndarray, optional
      Precomputed coefficients for the forward or inverse harmonic transform.
    fast_non_differentiable: bool, optional
      If the SHT is done in a faster jittable way, but not JAX-differentiable.

    Returns
    -------
    maps: array or list of arrays
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
            alms, nside, lmax, mmax, pixwin, fwhm, sigma, pol, inplace, False, healpy_ordering
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

    spmd = False

    if precomps_polar is not None:
      precomps_plus2 = precomps_polar[0] 
      precomps_minus2 = precomps_polar[1] 
    else:
      precomps_plus2, precomps_minus2 = None, None 

    map_plus2 = _inverse(
      -(alms[...,0,:]+1j*alms[...,1,:]),
      L,
      spin=2,
      nside=nside,
      sampling='healpix',
      reality=True,
      precomps=precomps_plus2,
      fast_non_differentiable=fast_non_differentiable,
      spmd=spmd,
      )
    map_minus2 = _inverse(
      -(alms[...,0,:]-1j*alms[...,1,:]),
      L,
      spin=-2,
      nside=nside,
      sampling='healpix',
      reality=True,
      precomps=precomps_minus2,
      fast_non_differentiable=fast_non_differentiable,
      spmd=spmd,
      )
    
    map_Q = (map_plus2 + map_minus2) /2
    map_U = -1j*(map_plus2 - map_minus2) /2
  
    return jnp.concatenate([map_Q, map_U], axis=-2)

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
    lmax: int = None,
    mmax: int = None,
    pixwin: bool = False,
    fwhm: float = 0.0,
    sigma: float = None,
    pol: bool = True,
    inplace: bool = False,
    verbose: bool = True,
    healpy_ordering: bool = False,
    precomps = None,
    fast_non_differentiable: bool = False
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

    if lmax is None:
        L = 3 * nside
    else:
        L = lmax + 1

    alms_temperature = alms
    if pol:
      alms_temperature = alms.at[...,0,:].get()

      maps_polarization = _alm2map_pol(
        alms.at[..., 1:,:].get(), 
        nside=nside, 
        lmax=lmax, 
        mmax=mmax, 
        pixwin=pixwin, 
        fwhm=fwhm, 
        sigma=sigma,
        inplace=inplace, 
        healpy_ordering=healpy_ordering
        )
    
    if healpy_ordering:
        alms_temperature = flm_hp_to_2d_fast(alms_temperature, L)

    spmd = False

    map_temperature = _inverse(
        alms_temperature,
        L,
        spin=0,
        nside=nside,
        sampling='healpix',
        reality=True,
        precomps=precomps,
        fast_non_differentiable=fast_non_differentiable,
        spmd=spmd,
    )

    if pol:
      return jnp.concatenate([map_temperature, maps_polarization], axis=-2)
    return map_temperature



@partial(
    jax.jit,
    static_argnames=[
        'lmax',
        'mmax',
        'iter',
        'use_weights',
        'datapath',
        'gal_cut',
        'use_pixel_weights',
        'verbose',
        'healpy_ordering',
    ],
)
@requires_s2fft
def _map2alm_pol(
    maps: ArrayLike,
    lmax: int = None,
    mmax: int = None,
    iter: int = 0,
    use_weights: bool = False,
    datapath: str = None,
    gal_cut: float = 0,
    use_pixel_weights: bool = False,
    verbose: bool = True,
    healpy_ordering: bool = False, 
    precomps_polar: list = None, 
    fast_non_differentiable: bool = False
):
    """Computes the alm of a Healpix map. The input polarization maps must all be
    in ring ordering. 

    Maps are assumed to be given as polarization maps, indexed with Q, U Stokes parameters  
    in the second last dimension, and pixel as last dimension. 

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
    maps: array-like, shape (Npix,) or (n, Npix)
      The input map or a list of n input maps. Must be in ring ordering.
    lmax: int, scalar, optional
      Maximum l of the power spectrum. Default: 3*nside-1
    mmax: int, scalar, optional
      Maximum m of the alm. Default: lmax
    iter: int, scalar, optional
      Number of iteration (default: 3)
    use_weights: bool, scalar, optional
      If True, use the ring weighting. Default: False.
    datapath: None or str, optional
      If given, the directory where to find the pixel weights.
      See in the docstring above details on how to set it up.
    gal_cut: float [degrees]
      pixels at latitude in [-gal_cut;+gal_cut] are not taken into account
    use_pixel_weights: bool, optional
      If True, use pixel by pixel weighting, healpy will automatically download the weights, if needed
    verbose: bool, optional
      Deprecated, has not effect.
    healpy_ordering: bool, optional
      By default, we follow the s2fft ordering for the alms. To use healpy
      ordering, set it to True.
    precomps_polar: list of np.ndarray, optional
      Precomputed coefficients for the forward harmonic transform.
    fast_non_differentiable: bool, optional
      If the SHT is done in a faster jittable way, but not JAX-differentiable.

    Returns
    -------
    alms: array or tuple of array
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
    if maps.shape[-2] != 2:
        raise ValueError('Input maps must have 2 Stokes parameters, Q and U.')
    

    maps = jnp.asarray(maps)
    nside = npix2nside(maps.shape[-1])
    if lmax is None:
        L = 3 * nside
    else:
        L = lmax + 1

    spmd = False

    if precomps_polar is not None: 
          precomps_plus2 = precomps_polar[0] 
          precomps_minus2 = precomps_polar[1] 
    else: 
      precomps_plus2, precomps_minus2 = None, None 
    
    flm_spin_plus2 = _forward(
              maps[...,0,:]+1j*maps[...,1,:],
              L,
              spin=2,
              nside=nside,
              sampling='healpix',
              reality=True,
              precomps=precomps_plus2,
              spmd=spmd,
              fast_non_differentiable=fast_non_differentiable,
          )
    flm_spin_minus2 = _forward(
              maps[...,0,:]-1j*maps[...,1,:],
              L,
              spin=-2,
              nside=nside,
              sampling='healpix',
              reality=True,
              precomps=precomps_minus2,
              spmd=spmd,
              fast_non_differentiable=fast_non_differentiable,
          )

    flm_E = -(flm_spin_plus2 + flm_spin_minus2) /2
    flm_B = 1j*(flm_spin_plus2 - flm_spin_minus2) /2

    if healpy_ordering:
        return flm_2d_to_hp_fast(flm_E, L), flm_2d_to_hp_fast(flm_B, L)
    return flm_E, flm_B

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
    maps: ArrayLike,
    lmax: int = None,
    mmax: int = None,
    iter: int = 0,
    pol: bool = True,
    use_weights: bool = False,
    datapath: str = None,
    gal_cut: int = 0,
    use_pixel_weights: bool = False,
    verbose: bool = True,
    healpy_ordering: bool = False,
    precomps: list[ArrayLike] = None,
    fast_non_differentiable: bool = False
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
    precomps : list of np.ndarray
      Precomputed list of recursion coefficients
    fast_non_differentiable : bool
      If the SHT is done in a faster jittable way, but not JAX-differentiable.

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
    if (maps.ndim == 2 and not pol) or (maps.ndim == 3 and pol):
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
            precomps,
        )
    elif maps.ndim > 3:
        raise ValueError('The input map has too many dimensions.')

    maps = jnp.asarray(maps)
    nside = npix2nside(maps.shape[-1])
    if lmax is None:
        L = 3 * nside
    else:
        L = lmax + 1

    spmd = False

    if precomps is not None and not pol: 
      precomps_0 = precomps
    else: 
      precomps_0 = None
    
    if maps.ndim == 1:
      pol = False

    maps_temperature = maps
    if pol:
      maps_temperature = maps.at[...,0,:].get()

      if precomps is not None and pol and len(precomps) == 3: 
          precomps_0 = precomps[0]
          precomps_plus2 = precomps[1] 
          precomps_minus2 = precomps[2] 
      else: 
        precomps_plus2, precomps_minus2 = None, None 

      flm_E, flm_B = _map2alm_pol(
        maps.at[..., 1:,:].get(), 
        lmax=lmax, 
        mmax=mmax, 
        use_weights=use_weights, 
        datapath=datapath, 
        gal_cut=gal_cut, 
        use_pixel_weights=use_pixel_weights, 
        healpy_ordering=healpy_ordering,
        precomps_polar=precomps_polar
        )

    flm = _forward(
        maps_temperature,
        L,
        spin=0,
        nside=nside,
        sampling='healpix',
        reality=True,
        precomps=precomps_0,
        spmd=spmd,
        fast_non_differentiable=fast_non_differentiable,
    )


    if healpy_ordering:
        if pol:
              return flm_2d_to_hp_fast(flm, L), flm_E, flm_B
        return flm_2d_to_hp_fast(flm, L)

    if pol:
          return flm, flm_E, flm_B
    return flm


@partial(
    jax.jit,
    static_argnames=[
        'mmax',
        'inplace',
        'healpy_ordering',
        'lmax'
    ],
)
def almxfl(
  alm: ArrayLike, 
  fl: ArrayLike, 
  mmax: int = None, 
  inplace: bool = False, 
  healpy_ordering: bool = False,
  lmax: int = None
  ):
    """Multiply alm by a function of l. The function is assumed
    to be zero where not defined.

    Parameters
    ----------
    alm: array
      The alm to multiply
    fl: array
      The function (at l=0..fl.size-1) by which alm must be multiplied.
    mmax: None or int, optional
      The maximum m defining the alm layout. Default: lmax.
    inplace: bool, optional
      If True, modify the given alm, otherwise make a copy before multiplying.
    healpy_ordering: bool, optional
      By default, we follow the s2fft ordering for the alms. To use healpy
      ordering, set it to True.
    lmax: int, optional
      If healpy_ordering is True, then lmax is needed

    Returns
    -------
    alm: array
      The modified alm, either a new array or a reference to input alm,
      if inplace is True.
    """

    if inplace:
        raise NotImplementedError('Specifying inplace is not implemented.')

    if healpy_ordering:
        if lmax is None:
          raise ValueError('lmax is needed when healpy_ordering is True')

        # Identifying the m indices of a set of alms according to Healpy convention
        all_m_idx = jax.vmap(lambda m_idx: m_idx * (2 * lmax + 1 - m_idx) // 2)(jnp.arange(lmax + 1))

        def func_scan(carry, ell):
          """
          For a given ell, returns the alms convolved with the covariance matrix fl for all m
          """
          _alm_carry = carry
          mask_m = jnp.where(jnp.arange(lmax + 1) <= ell, fl[...,ell], 1)
          _alm_carry = _alm_carry.at[all_m_idx + ell].set(_alm_carry[all_m_idx + ell] * mask_m)
          return _alm_carry, ell

        alms_output, _ = jax.lax.scan(func_scan, alm, jnp.arange(lmax + 1))
        return alms_output
    
    return jnp.einsum('...lm, ...l -> ...lm', alm, fl)


@partial(
    jax.jit,
    static_argnames=[
        'lmax',
        'mmax',
        'lmax_out',
        'nspec',
        'healpy_ordering'
    ],
)
def alm2cl(
  alms: ArrayLike, 
  alms2: ArrayLike = None, 
  lmax: int = None, 
  mmax: int = None, 
  lmax_out: int = None, 
  nspec: int = None,
  healpy_ordering: bool = False,
):
    """Computes (cross-)spectra from alm(s). If alm2 is given, cross-spectra between
    alm and alm2 are computed. If alm (and alm2 if provided) contains n alm,
    then n(n+1)/2 auto and cross-spectra are returned.

    Parameters
    ----------
    alm: complex, array or sequence of arrays
      The alm from which to compute the power spectrum. If n>=2 arrays are given,
      computes both auto- and cross-spectra.
    alms2: complex, array or sequence of 3 arrays, optional
      If provided, computes cross-spectra between alm and alm2.
      Default: alm2=alm, so auto-spectra are computed.
    lmax: None or int, optional
      The maximum l of the input alm. Default: computed from size of alm
      and mmax_in
    mmax: None or int, optional
      The maximum m of the input alm. Default: assume mmax_in = lmax_in
    lmax_out: None or int, optional
      The maximum l of the returned spectra. By default: the lmax of the given
      alm(s).
    healpy_ordering: bool, optional
      By default, we follow the s2fft ordering for the alms. To use healpy
      ordering, set it to True.

    Returns
    -------
    cl: array or tuple of n(n+1)/2 arrays
      the spectrum <*alm* x *alm2*> if *alm* (and *alm2*) is one alm, or
      the auto- and cross-spectra <*alm*[i] x *alm2*[j]> if alm (and alm2)
      contains more than one spectra.
      If more than one spectrum is returned, they are ordered by diagonal.
      For example, if *alm* is almT, almE, almB, then the returned spectra are:
      TT, EE, BB, TE, EB, TB.
    """

    if lmax is None and healpy_ordering:
      raise ValueError('lmax must be provided with alm2cl if using healpy_ordering')

    alms = jnp.asarray(alms)
    if healpy_ordering: 
      alms = flm_hp_to_2d_fast(alms, lmax + 1)
    if alms2 is None:
      alms2 = alms
    else: 
      if healpy_ordering: 
        alms2 = flm_hp_to_2d_fast(alms2, lmax + 1)
      alms2 = jnp.asarray(alms2)

      if alms.shape != alms2.shape:
        raise ValueError('alm and alm2 must have the same shape')

    if not healpy_ordering:
      if alms.ndim == 2:
        n_stokes = 1
      elif alms.ndim == 3:
        n_stokes = alms.shape[0]
      else:
        raise ValueError('The input alms have a wrong dimension')
    else:
      if alms.ndim == 1:
        n_stokes = 1
      elif alms.ndim == 2:
        n_stokes = alms.shape[0]
      else:
        raise ValueError('The input alms have too many dimensions')

    if lmax is None:
      lmax = alms.shape[-2] - 1       
    
    if lmax_out is None:
      lmax_out = lmax
    
    if mmax is None:
      mmax = lmax



    if nspec is None:
      nspec = (n_stokes*(n_stokes+1)) // 2 

    get_cl = lambda _alms1, _alms2: (jnp.sum((_alms1.real*_alms2.real + _alms1.imag*_alms2.imag)[...,:], axis=-1)/(2*jnp.arange(lmax+1) +1))[...,:lmax_out+1]

    auto_cl = get_cl(
      alms, 
      alms2, 
      )
    if n_stokes == 1:
      return auto_cl

    cross_cl = jnp.roll(
      get_cl(
        alms, 
        jnp.roll(alms2, 1, axis=0)
        ),
      shift=-1, 
      axis=0)
    cross_cl_revert = jnp.roll(
      get_cl(
        alms, 
        jnp.roll(alms2, 1, axis=0)
        ),
      shift=1, 
      axis=0)[::-1]
    #TODO: For cross-cl include (TE + ET)/2, (TB + BT)/2, (EB + BE)/2?
    #TODO: TO RECHECK!!!
    
    return jnp.vstack([auto_cl, (cross_cl+cross_cl_revert)/2]).at[:nspec].get()

  
@partial(
    jax.jit,
    static_argnames=[
        'nspec',
        'lmax',
        'mmax',
        'iter',
        'alm',
        'pol',
        'use_weights',
        'datapath',
        'gal_cut',
        'use_pixel_weights',
    ],
)
def anafast(
    map1,
    map2: ArrayLike = None,
    nspec: int = None,
    lmax: int = None,
    mmax: int = None,
    iter: int = 3,
    alm: bool = False,
    pol: bool = True,
    only_pol: bool = False,
    use_weights: bool = False,
    datapath: str = None,
    gal_cut: float = 0,
    use_pixel_weights: bool = False,
    healpy_ordering: bool = False,
):
    """Computes the power spectrum of a Healpix map, or the cross-spectrum
    between two maps if *map2* is given.
    No removal of monopole or dipole is performed. The input maps must be
    in ring-ordering.
    Spherical harmonics transforms in HEALPix are always on the full sky,
    if the map is masked, those pixels are set to 0. It is recommended to
    remove monopole from the map before running `anafast` to reduce
    boundary effects.

    For recommendations about how to set `lmax`, `iter`, and weights, see the
    `Anafast documentation <https://healpix.sourceforge.io/html/fac_anafast.htm>`_

    Parameters
    ----------
    map1: float, array-like shape (Npix,), (2, Npix) or (3, Npix)
      Either an array representing a map, a sequence of 3 arrays
      representing I, Q, U maps, or Q, U maps if only_pol is True. 
      Must be in ring ordering.
    map2: float, array-like shape (Npix,), (2, Npix) or (3, Npix)
      Either an array representing a map, a sequence of 3 arrays
      representing I, Q, U maps, or Q, U maps if only_pol is True. 
      Must be in ring ordering.
    nspec: None or int, optional
      The number of spectra to return. If None, returns all, otherwise
      returns cls[:nspec]
    lmax: int, scalar, optional
      Maximum l of the power spectrum (default: 3*nside-1)
    mmax: int, scalar, optional
      Maximum m of the alm (default: lmax)
    iter: int, scalar, optional
      Number of iteration (default: 3)
    alm: bool, scalar, optional
      If True, returns both cl and alm, otherwise only cl is returned
    pol: bool, optional
      If True, assumes input maps are TQU. Output will be TEB cl's and
      correlations (input must be 1 or 3 maps).
      If False, maps are assumed to be described by spin 0 spherical harmonics.
      (input can be any number of maps)
      If there is only one input map, it has no effect. Default: True.
    only_pol: bool, optional
      If True, consider maps are only given only return the polarization spectra (EE, BB, TE, EB, TB).
    datapath: None or str, optional
      If given, the directory where to find the weights data.
      See the docstring of `map2alm` for details on how to set it up
    gal_cut: float [degrees]
      pixels at latitude in [-gal_cut;+gal_cut] are not taken into account
    use_pixel_weights: bool, optional
      If True, use pixel by pixel weighting, healpy will automatically download the weights, if needed
      See the map2alm docs for details about weighting

    Returns
    -------
    res: array or sequence of arrays
      If *alm* is False, returns cl or a list of cl's (TT, EE, BB, TE, EB, TB for
      polarized input map)
      Otherwise, returns a tuple (cl, alm), where cl is as above and
      alm is the spherical harmonic transform or a list of almT, almE, almB
      for polarized input
    
    Notes
    -------
    The alms will be returned according to s2fft ordering 
    """
    if map2 is not None and map1.shape != map2.shape:
      raise ValueError('map1 and map2 must have the same shape')
    
    if lmax is None and healpy_ordering:
      raise ValueError('lmax must be provided with alm2cl if using healpy_ordering')
    
    if datapath is not None:
      raise NotImplementedError('Specifying datapath is not implemented.')
    if use_pixel_weights:
      raise NotImplementedError('Specifying use_pixel_weights is not implemented.')

    if lmax is None:
      lmax = alms.shape[-2] - 1
    
    if mmax is None:
      mmax = lmax

    if nspec is None:
      if map1.ndim == 1:
        n_stokes = 1
      else:
        n_stokes = map1.shape[0]

      nspec = n_stokes**2 // 2 + n_stokes // 2 + n_stokes % 2

    if only_pol:
      map2alm_func = _map2alm_pol
      if map1.shape[-2] != 2:
        raise ValueError('Input maps must have 2 Stokes parameters, Q and U.')
    else:
      map2alm_func = map2alm

    alm1 = jnp.asarray(
      map2alm_func(
        map1, 
        lmax=lmax, 
        mmax=mmax, 
        iter=iter, 
        pol=pol,
        use_weights=use_weights, 
        datapath=datapath, 
        gal_cut=gal_cut, 
        use_pixel_weights=use_pixel_weights
        )
      )

    if map2 is None:
      alm2 = alm1
    else:
      alm2 = jnp.asarray(
        map2alm_func(
          map2, 
          lmax=lmax, 
          mmax=mmax, 
          iter=iter, 
          pol=pol, 
          use_weights=use_weights, 
          datapath=datapath, 
          gal_cut=gal_cut, 
          use_pixel_weights=use_pixel_weights
          )
        )

    cls_res = alm2cl(
      alm1, 
      alm2, 
      lmax=lmax, 
      mmax=mmax, 
      lmax_out=lmax, 
      nspec=nspec, 
      healpy_ordering=False
      )

    if alm:
      if map2 is not None:
        return cls_res, alm1, alm2
      return cls_res, alm1
    return cls_res


@partial(
    jax.jit,
    static_argnames=[
      'lmax',
      'mmax',
      'new',
      'verbose'
    ],
)
def synalm(
  cls: ArrayLike, 
  lmax: int = None, 
  mmax: int = None,
  seed: int = 0, 
  new: bool = False, 
  verbose: bool = True
  ):
    """Generate a set of alm given cl.
    The cl are given as a float array. Corresponding alm are generated.
    If lmax is None, it is assumed lmax=cl.size-1
    If mmax is None, it is assumed mmax=lmax.

    Parameters
    ----------
    cls: float, array or tuple of arrays
      Either one cl (1D array) or a tuple of either 4 cl
      or of n*(n+1)/2 cl.
      Some of the cl may be None, implying no
      cross-correlation. See *new* parameter.
    lmax: int, scalar, optional
      The lmax (if None or <0, the largest size-1 of cls)
    mmax: int, scalar, optional
      The mmax (if None or <0, =lmax)
    seed: int, scalar, optional
      The seed for the random number generator
    new: bool, optional
      If True, use the new ordering of cl's, ie by diagonal
      (e.g. TT, EE, BB, TE, EB, TB or TT, EE, BB, TE if 4 cl as input).
      If False, use the old ordering, ie by row
      (e.g. TT, TE, TB, EE, EB, BB or TT, TE, EE, BB if 4 cl as input).

    Returns
    -------
    alms: array or list of arrays
      the generated alm if one spectrum is given, or a list of n alms
      (with n(n+1)/2 the number of input cl, or n=3 if there are 4 input cl).

    Notes
    -----
    We don't plan to change the default order anymore, that would break old
    code in a way difficult to debug.
    """

    if new==True:
        # From TT, EE, BB, TE, EB, TB to TT, TE, TB, EE, EB, BB
        new_order = jnp.array([0, 3, 5, 1, 4, 2])

        cls = cls[new_order]
    
    if lmax is None:
        if cls.ndim == 1:
          lmax = cls.size - 1
        else:
          lmax = cls[0].size - 1
    
    if mmax is None:
        mmax = lmax

    if cls.ndim == 1:
      n_stokes == 1
      cls = cls[None,...]
    elif cls.ndim == 4:
      n_stokes == 3
    else:
      n_stokes == jnp.int16(-.5 + jnp.sqrt(.25 + 2*cls.shape[0]))

    random_keys = jax.random.split(jax.random.PRNGKey(seed), lmax + 1)

    def get_map_alms(ell_seed):
      ell, seed = ell_seed
      matrix_triangular = jnp.zeros((n_stokes, n_stokes), dtype=jnp.float64)
      matrix_triangular = matrix_triangular.at[jnp.tril_indices(n_stokes)].set(cls[...,ell]) 

      cholesky_decomposition = jnp.linalg.cholesky(jnp.maximum(matrix_triangular,matrix_triangular.T))

      mask_m = jnp.where(jnp.arange(mmax*2-1) <= 2*ell+1, 1, 0)
      random = jax.random.normal(seed, (n_stokes, mmax*2-1), dtype=jnp.complex64) * mask_m

      return jnp.roll(jnp.einsum('sk,km->sm',cholesky_decomposition, random), shift=lmax+ell, axis=1)

    all_alms = jax.vmap(get_map_alms, in_axes=(0, 0))(jnp.arange(lmax + 1), random_keys)

    return all_alms

@partial(
    jax.jit,
    static_argnames=[
      'nside',
      'lmax',
      'mmax',
      'alm',
      'pol',
      'pixwin',
      'fwhm',
      'sigma',
      'new',
      'verbose'
    ],
)
def synfast(
    cls: ArrayLike,
    nside: int,
    lmax: int = None,
    mmax: int = None,
    alm: bool = False,
    pol: bool = True,
    pixwin: bool = False,
    fwhm: float = 0.0,
    sigma: float = None,
    new: bool = False,
    verbose: bool = True,
):
    """Create a map(s) from cl(s).

    You can choose a random seed using `numpy.random.seed(SEEDVALUE)`
    before calling `synfast`.

    Parameters
    ----------
    cls: array or tuple of array
      A cl or a list of cl (either 4 or 6, see:func:`synalm`)
    nside: int, scalar
      The nside of the output map(s)
    lmax: int, scalar, optional
      Maximum l for alm. Default: min of 3*nside-1 or length of the cls - 1
    mmax: int, scalar, optional
      Maximum m for alm.
    alm: bool, scalar, optional
      If True, return also alm(s). Default: False.
    pol: bool, optional
      If True, assumes input cls are TEB and correlation. Output will be TQU maps.
      (input must be 1, 4 or 6 cl's)
      If False, fields are assumed to be described by spin 0 spherical harmonics.
      (input can be any number of cl's)
      If there is only one input cl, it has no effect. Default: True.
    pixwin: bool, scalar, optional
      If True, convolve the alm by the pixel window function. Default: False.
    fwhm: float, scalar, optional
      The fwhm of the Gaussian used to smooth the map (applied on alm)
      [in radians]
    sigma: float, scalar, optional
      The sigma of the Gaussian used to smooth the map (applied on alm)
      [in radians]
    new: bool, optional
      If True, use the new ordering of cl's, ie by diagonal
      (e.g. TT, EE, BB, TE, EB, TB or TT, EE, BB, TE if 4 cl as input).
      If False, use the old ordering, ie by row
      (e.g. TT, TE, TB, EE, EB, BB or TT, TE, EE, BB if 4 cl as input).

    Returns
    -------
    maps: array or tuple of arrays
      The output map (possibly list of maps if polarized input).
      or, if alm is True, a tuple of (map,alm)
      (alm possibly a list of alm if polarized input)

    Notes
    -----
    We don't plan to change the default order anymore, that would break old
    code in a way difficult to debug.
    """
    if jnp.log(nside) / jnp.log(2) % 1 != 0:
        raise ValueError('nside must be a power of 2')

    if cls.ndim == 1:
      cls_lmax = cls.size - 1
    else:
      cls_lmax = cls[0].size - 1
    
    if lmax is None or lmax < 0:
        lmax = jnp.min(cls_lmax, 3 * nside - 1)
    
    alms = synalm(cls, lmax=lmax, mmax=mmax, new=new)
    
    maps = alm2map(
        alms,
        nside,
        lmax=lmax,
        mmax=mmax,
        pixwin=pixwin,
        pol=pol,
        fwhm=fwhm,
        sigma=sigma,
        inplace=True,
    )
    
    if alm:
        return jnp.asarray(maps), jnp.asarray(alms)
    return jnp.asarray(maps)


@partial(
    jax.jit,
    static_argnames=[
        'mmax',
        'inplace',
        'healpy_ordering',
        'lmax'
    ],
)
def almxfl(
  alm: ArrayLike, 
  fl: ArrayLike, 
  mmax: int = None, 
  inplace: bool = False, 
  healpy_ordering: bool = False,
  lmax: int = None
  ):
    """Multiply alm by a function of l. The function is assumed
    to be zero where not defined.

    Parameters
    ----------
    alm: array
      The alm to multiply
    fl: array
      The function (at l=0..fl.size-1) by which alm must be multiplied.
    mmax: None or int, optional
      The maximum m defining the alm layout. Default: lmax.
    inplace: bool, optional
      If True, modify the given alm, otherwise make a copy before multiplying.
    healpy_ordering: bool, optional
      By default, we follow the s2fft ordering for the alms. To use healpy
      ordering, set it to True.
    lmax: int, optional
      If healpy_ordering is True, then lmax is needed

    Returns
    -------
    alm: array
      The modified alm, either a new array or a reference to input alm,
      if inplace is True.
    """

    if inplace:
        raise NotImplementedError('Specifying inplace is not implemented.')

    if healpy_ordering:
        if lmax is None:
          raise ValueError('lmax is needed when healpy_ordering is True')

        # Identifying the m indices of a set of alms according to Healpy convention
        all_m_idx = jax.vmap(lambda m_idx: m_idx * (2 * lmax + 1 - m_idx) // 2)(jnp.arange(lmax + 1))

        def func_scan(carry, ell):
          """
          For a given ell, returns the alms convolved with the covariance matrix fl for all m
          """
          _alm_carry = carry
          mask_m = jnp.where(jnp.arange(lmax + 1) <= ell, fl[...,ell], 1)
          _alm_carry = _alm_carry.at[all_m_idx + ell].set(_alm_carry[all_m_idx + ell] * mask_m)
          return _alm_carry, ell

        alms_output, _ = jax.lax.scan(func_scan, alm, jnp.arange(lmax + 1))
        return alms_output
    
    return jnp.einsum('...lm, ...l -> ...lm', alm, fl)


@partial(
    jax.jit,
    static_argnames=[
        'lmax',
        'mmax',
        'lmax_out',
        'nspec',
        'healpy_ordering'
    ],
)
def alm2cl(
  alms: ArrayLike, 
  alms2: ArrayLike = None, 
  lmax: int = None, 
  mmax: int = None, 
  lmax_out: int = None, 
  nspec: int = None,
  healpy_ordering: bool = False,
):
    """Computes (cross-)spectra from alm(s). If alm2 is given, cross-spectra between
    alm and alm2 are computed. If alm (and alm2 if provided) contains n alm,
    then n(n+1)/2 auto and cross-spectra are returned.

    Parameters
    ----------
    alm: complex, array or sequence of arrays
      The alm from which to compute the power spectrum. If n>=2 arrays are given,
      computes both auto- and cross-spectra.
    alms2: complex, array or sequence of 3 arrays, optional
      If provided, computes cross-spectra between alm and alm2.
      Default: alm2=alm, so auto-spectra are computed.
    lmax: None or int, optional
      The maximum l of the input alm. Default: computed from size of alm
      and mmax_in
    mmax: None or int, optional
      The maximum m of the input alm. Default: assume mmax_in = lmax_in
    lmax_out: None or int, optional
      The maximum l of the returned spectra. By default: the lmax of the given
      alm(s).
    healpy_ordering: bool, optional
      By default, we follow the s2fft ordering for the alms. To use healpy
      ordering, set it to True.

    Returns
    -------
    cl: array or tuple of n(n+1)/2 arrays
      the spectrum <*alm* x *alm2*> if *alm* (and *alm2*) is one alm, or
      the auto- and cross-spectra <*alm*[i] x *alm2*[j]> if alm (and alm2)
      contains more than one spectra.
      If more than one spectrum is returned, they are ordered by diagonal.
      For example, if *alm* is almT, almE, almB, then the returned spectra are:
      TT, EE, BB, TE, EB, TB.
    """

    if lmax is None and healpy_ordering:
      raise ValueError('lmax must be provided with alm2cl if using healpy_ordering')

    alms = jnp.asarray(alms)
    if healpy_ordering: 
      alms = flm_hp_to_2d_fast(alms, lmax + 1)
    if alms2 is None:
      alms2 = alms
    else: 
      if healpy_ordering: 
        alms2 = flm_hp_to_2d_fast(alms2, lmax + 1)
      alms2 = jnp.asarray(alms2)

      if alms.shape != alms2.shape:
        raise ValueError('alm and alm2 must have the same shape')

    if not healpy_ordering:
      if alms.ndim == 2:
        n_stokes = 1
      elif alms.ndim == 3:
        n_stokes = alms.shape[0]
      else:
        raise ValueError('The input alms have a wrong dimension')
    else:
      if alms.ndim == 1:
        n_stokes = 1
      elif alms.ndim == 2:
        n_stokes = alms.shape[0]
      else:
        raise ValueError('The input alms have too many dimensions')

    if lmax is None:
      lmax = alms.shape[-2] - 1       
    
    if lmax_out is None:
      lmax_out = lmax
    
    if mmax is None:
      mmax = lmax



    if nspec is None:
      nspec = (n_stokes*(n_stokes+1)) // 2 

    
      

    get_cl = lambda _alms1, _alms2: (jnp.sum((_alms1.real*_alms2.real + _alms1.imag*_alms2.imag)[...,:], axis=-1)/(2*jnp.arange(lmax+1) +1))[...,:lmax_out+1]

    auto_cl = get_cl(
      alms, 
      alms2, 
      )
    if n_stokes == 1:
      return auto_cl

    cross_cl = jnp.roll(
      get_cl(
        alms, 
        jnp.roll(alms2, 1, axis=0)
        ),
      shift=-1, 
      axis=0)
    
    return jnp.vstack([auto_cl, cross_cl]).at[:nspec].get()

  
@partial(
    jax.jit,
    static_argnames=[
        'nspec',
        'lmax',
        'mmax',
        'iter',
        'alm',
        'pol',
        'use_weights',
        'datapath',
        'gal_cut',
        'use_pixel_weights',
    ],
)
def anafast(
    map1,
    map2: ArrayLike = None,
    nspec: int = None,
    lmax: int = None,
    mmax: int = None,
    iter: int = 3,
    alm: bool = False,
    pol: bool = True,
    only_pol: bool = False,
    use_weights: bool = False,
    datapath: str = None,
    gal_cut: float = 0,
    use_pixel_weights: bool = False,
    healpy_ordering: bool = False,
):
    """Computes the power spectrum of a Healpix map, or the cross-spectrum
    between two maps if *map2* is given.
    No removal of monopole or dipole is performed. The input maps must be
    in ring-ordering.
    Spherical harmonics transforms in HEALPix are always on the full sky,
    if the map is masked, those pixels are set to 0. It is recommended to
    remove monopole from the map before running `anafast` to reduce
    boundary effects.

    For recommendations about how to set `lmax`, `iter`, and weights, see the
    `Anafast documentation <https://healpix.sourceforge.io/html/fac_anafast.htm>`_

    Parameters
    ----------
    map1: float, array-like shape (Npix,), (2, Npix) or (3, Npix)
      Either an array representing a map, a sequence of 3 arrays
      representing I, Q, U maps, or Q, U maps if only_pol is True. 
      Must be in ring ordering.
    map2: float, array-like shape (Npix,), (2, Npix) or (3, Npix)
      Either an array representing a map, a sequence of 3 arrays
      representing I, Q, U maps, or Q, U maps if only_pol is True. 
      Must be in ring ordering.
    nspec: None or int, optional
      The number of spectra to return. If None, returns all, otherwise
      returns cls[:nspec]
    lmax: int, scalar, optional
      Maximum l of the power spectrum (default: 3*nside-1)
    mmax: int, scalar, optional
      Maximum m of the alm (default: lmax)
    iter: int, scalar, optional
      Number of iteration (default: 3)
    alm: bool, scalar, optional
      If True, returns both cl and alm, otherwise only cl is returned
    pol: bool, optional
      If True, assumes input maps are TQU. Output will be TEB cl's and
      correlations (input must be 1 or 3 maps).
      If False, maps are assumed to be described by spin 0 spherical harmonics.
      (input can be any number of maps)
      If there is only one input map, it has no effect. Default: True.
    only_pol: bool, optional
      If True, consider maps are only given only return the polarization spectra (EE, BB, TE, EB, TB).
    datapath: None or str, optional
      If given, the directory where to find the weights data.
      See the docstring of `map2alm` for details on how to set it up
    gal_cut: float [degrees]
      pixels at latitude in [-gal_cut;+gal_cut] are not taken into account
    use_pixel_weights: bool, optional
      If True, use pixel by pixel weighting, healpy will automatically download the weights, if needed
      See the map2alm docs for details about weighting

    Returns
    -------
    res: array or sequence of arrays
      If *alm* is False, returns cl or a list of cl's (TT, EE, BB, TE, EB, TB for
      polarized input map)
      Otherwise, returns a tuple (cl, alm), where cl is as above and
      alm is the spherical harmonic transform or a list of almT, almE, almB
      for polarized input
    
    Notes
    -------
    The alms will be returned according to s2fft ordering 
    """
    if map2 is not None and map1.shape != map2.shape:
      raise ValueError('map1 and map2 must have the same shape')
    
    if lmax is None and healpy_ordering:
      raise ValueError('lmax must be provided with alm2cl if using healpy_ordering')
    
    if datapath is not None:
      raise NotImplementedError('Specifying datapath is not implemented.')
    if use_pixel_weights:
      raise NotImplementedError('Specifying use_pixel_weights is not implemented.')

    if lmax is None:
      lmax = alms.shape[-2] - 1
    
    if mmax is None:
      mmax = lmax

    if nspec is None:
      if map1.ndim == 1:
        n_stokes = 1
      else:
        n_stokes = map1.shape[0]

      nspec = n_stokes**2 // 2 + n_stokes // 2 + n_stokes % 2

    if only_pol:
      map2alm_func = _map2alm_pol
      if map1.shape[-2] != 2:
        raise ValueError('Input maps must have 2 Stokes parameters, Q and U.')
    else:
      map2alm_func = map2alm

    alm1 = jnp.asarray(
      map2alm_func(
        map1, 
        lmax=lmax, 
        mmax=mmax, 
        iter=iter, 
        pol=pol,
        use_weights=use_weights, 
        datapath=datapath, 
        gal_cut=gal_cut, 
        use_pixel_weights=use_pixel_weights
        )
      )

    if map2 is None:
      alm2 = alm1
    else:
      alm2 = jnp.asarray(
        map2alm_func(
          map2, 
          lmax=lmax, 
          mmax=mmax, 
          iter=iter, 
          pol=pol, 
          use_weights=use_weights, 
          datapath=datapath, 
          gal_cut=gal_cut, 
          use_pixel_weights=use_pixel_weights
          )
        )

    cls_res = alm2cl(
      alm1, 
      alm2, 
      lmax=lmax, 
      mmax=mmax, 
      lmax_out=lmax, 
      nspec=nspec, 
      healpy_ordering=False
      )

    if alm:
      if map2 is not None:
        return cls_res, alm1, alm2
      return cls_res, alm1
    return cls_res


@partial(
    jax.jit,
    static_argnames=[
      'lmax',
      'mmax',
      'new',
      'verbose'
    ],
)
def synalm(
  cls: ArrayLike, 
  lmax: int = None, 
  mmax: int = None,
  seed: int = 0, 
  new: bool = False, 
  verbose: bool = True
  ):
    """Generate a set of alm given cl.
    The cl are given as a float array. Corresponding alm are generated.
    If lmax is None, it is assumed lmax=cl.size-1
    If mmax is None, it is assumed mmax=lmax.

    Parameters
    ----------
    cls: float, array or tuple of arrays
      Either one cl (1D array) or a tuple of either 4 cl
      or of n*(n+1)/2 cl.
      Some of the cl may be None, implying no
      cross-correlation. See *new* parameter.
    lmax: int, scalar, optional
      The lmax (if None or <0, the largest size-1 of cls)
    mmax: int, scalar, optional
      The mmax (if None or <0, =lmax)
    seed: int, scalar, optional
      The seed for the random number generator
    new: bool, optional
      If True, use the new ordering of cl's, ie by diagonal
      (e.g. TT, EE, BB, TE, EB, TB or TT, EE, BB, TE if 4 cl as input).
      If False, use the old ordering, ie by row
      (e.g. TT, TE, TB, EE, EB, BB or TT, TE, EE, BB if 4 cl as input).

    Returns
    -------
    alms: array or list of arrays
      the generated alm if one spectrum is given, or a list of n alms
      (with n(n+1)/2 the number of input cl, or n=3 if there are 4 input cl).

    Notes
    -----
    We don't plan to change the default order anymore, that would break old
    code in a way difficult to debug.
    """

    if new==True:
        # From TT, EE, BB, TE, EB, TB to TT, TE, TB, EE, EB, BB
        new_order = jnp.array([0, 3, 5, 1, 4, 2])

        cls = cls[new_order]
    
    if lmax is None:
        if cls.ndim == 1:
          lmax = cls.size - 1
        else:
          lmax = cls[0].size - 1
    
    if mmax is None:
        mmax = lmax

    if cls.ndim == 1:
      n_stokes == 1
      cls = cls[None,...]
    elif cls.ndim == 4:
      n_stokes == 3
    else:
      n_stokes == jnp.int16(-.5 + jnp.sqrt(.25 + 2*cls.shape[0]))

    random_keys = jax.random.split(jax.random.PRNGKey(seed), lmax + 1)

    def get_map_alms(ell_seed):
      ell, seed = ell_seed
      matrix_triangular = jnp.zeros((n_stokes, n_stokes), dtype=jnp.float64)
      matrix_triangular = matrix_triangular.at[jnp.tril_indices(n_stokes)].set(cls[...,ell]) 

      cholesky_decomposition = jnp.linalg.cholesky(jnp.maximum(matrix_triangular,matrix_triangular.T))

      mask_m = jnp.where(jnp.arange(mmax*2-1) <= 2*ell+1, 1, 0)
      random = jax.random.normal(seed, (n_stokes, mmax*2-1), dtype=jnp.complex64) * mask_m

      return jnp.roll(jnp.einsum('sk,km->sm',cholesky_decomposition, random), shift=lmax+ell, axis=1)

    all_alms = jax.vmap(get_map_alms, in_axes=(0, 0))(jnp.arange(lmax + 1), random_keys)

    return all_alms

@partial(
    jax.jit,
    static_argnames=[
      'nside',
      'lmax',
      'mmax',
      'alm',
      'pol',
      'pixwin',
      'fwhm',
      'sigma',
      'new',
      'verbose'
    ],
)
def synfast(
    cls: ArrayLike,
    nside: int,
    lmax: int = None,
    mmax: int = None,
    alm: bool = False,
    pol: bool = True,
    pixwin: bool = False,
    fwhm: float = 0.0,
    sigma: float = None,
    new: bool = False,
    verbose: bool = True,
):
    """Create a map(s) from cl(s).

    You can choose a random seed using `numpy.random.seed(SEEDVALUE)`
    before calling `synfast`.

    Parameters
    ----------
    cls: array or tuple of array
      A cl or a list of cl (either 4 or 6, see:func:`synalm`)
    nside: int, scalar
      The nside of the output map(s)
    lmax: int, scalar, optional
      Maximum l for alm. Default: min of 3*nside-1 or length of the cls - 1
    mmax: int, scalar, optional
      Maximum m for alm.
    alm: bool, scalar, optional
      If True, return also alm(s). Default: False.
    pol: bool, optional
      If True, assumes input cls are TEB and correlation. Output will be TQU maps.
      (input must be 1, 4 or 6 cl's)
      If False, fields are assumed to be described by spin 0 spherical harmonics.
      (input can be any number of cl's)
      If there is only one input cl, it has no effect. Default: True.
    pixwin: bool, scalar, optional
      If True, convolve the alm by the pixel window function. Default: False.
    fwhm: float, scalar, optional
      The fwhm of the Gaussian used to smooth the map (applied on alm)
      [in radians]
    sigma: float, scalar, optional
      The sigma of the Gaussian used to smooth the map (applied on alm)
      [in radians]
    new: bool, optional
      If True, use the new ordering of cl's, ie by diagonal
      (e.g. TT, EE, BB, TE, EB, TB or TT, EE, BB, TE if 4 cl as input).
      If False, use the old ordering, ie by row
      (e.g. TT, TE, TB, EE, EB, BB or TT, TE, EE, BB if 4 cl as input).

    Returns
    -------
    maps: array or tuple of arrays
      The output map (possibly list of maps if polarized input).
      or, if alm is True, a tuple of (map,alm)
      (alm possibly a list of alm if polarized input)

    Notes
    -----
    We don't plan to change the default order anymore, that would break old
    code in a way difficult to debug.
    """
    if jnp.log(nside) / jnp.log(2) % 1 != 0:
        raise ValueError('nside must be a power of 2')

    if cls.ndim == 1:
      cls_lmax = cls.size - 1
    else:
      cls_lmax = cls[0].size - 1
    
    if lmax is None or lmax < 0:
        lmax = jnp.min(cls_lmax, 3 * nside - 1)
    
    alms = synalm(cls, lmax=lmax, mmax=mmax, new=new)
    
    maps = alm2map(
        alms,
        nside,
        lmax=lmax,
        mmax=mmax,
        pixwin=pixwin,
        pol=pol,
        fwhm=fwhm,
        sigma=sigma,
        inplace=True,
    )
    
    if alm:
        return jnp.asarray(maps), jnp.asarray(alms)
    return jnp.asarray(maps)
