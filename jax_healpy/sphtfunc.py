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

import warnings
from functools import partial, wraps
from typing import Callable, ParamSpec, TypeVar

import healpy as hp
import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.extend
import jax.lax as jlax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import PRNGKeyArray

try:
    from s2fft.recursions.price_mcewen import generate_precomputes_jax
    from s2fft.sampling.reindex import flm_2d_to_hp_fast, flm_hp_to_2d_fast
    from s2fft.transforms import spherical
except ImportError:
    pass

from jax_healpy import npix2nside

__all__ = [
    'alm2cl',
    'alm2map',
    'alm2map_spin',
    'almxfl',
    'anafast',
    'gauss_beam',
    'map2alm',
    'map2alm_spin',
    'pixwin',
    'smoothalm',
    'smoothing',
    'synalm',
    'synfast',
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
    spmd: bool = False,
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
        spmd=spmd,
    )

    if fast_non_differentiable:
        lmax = L - 1

        if alms.ndim == 3:
            nstokes = alms.shape[0]
        else:
            nstokes = 1

        npix = 12 * nside**2

        # Wrapper for alm2map, to prepare the pure callback of JAX
        def wrapper_alm2map(alm_, lmax=lmax, nside=nside):
            alm_np = jax.tree.map(np.asarray, alm_).squeeze()
            if nstokes == 2:
                alm_np_extended = np.vstack([np.zeros_like(alm_np[0]), alm_np])
            else:
                alm_np_extended = alm_np
            map_output = hp.alm2map(alm_np_extended, nside, lmax=lmax)
            if len(map_output.shape) == 2:
                return map_output[-nstokes:, ...]
            return map_output.reshape((nstokes, npix))

        @partial(jax.jit, static_argnums=(1, 2))
        def pure_call_alm2map(alm_, lmax, nside):
            alm_healpy = jax.vmap(partial(flm_2d_to_hp_fast, L=L), in_axes=(0,))(
                alm_.reshape((nstokes,) + tuple(alm_.shape[-2:]))
            )
            shape_output = (nstokes, npix)
            return jax.pure_callback(
                wrapper_alm2map, jax.ShapeDtypeStruct(shape_output, np.float64), alm_healpy
            ).squeeze()

        inverse_cpu = partial(pure_call_alm2map, nside=nside, lmax=lmax)

        # inverse_gpu = partial(inverse_s2fft, method='cuda')
        def inverse_gpu(alms):
            if 'gpu' in jax.extend.backend.get_backend().platform:
                return inverse_s2fft(alms, method='cuda')
            else:
                return pure_call_alm2map(alms, nside=nside, lmax=lmax)

    else:
        inverse_cpu = partial(inverse_s2fft, method='jax')
        inverse_gpu = partial(inverse_s2fft, method='jax')

    return jlax.platform_dependent(alms, cpu=inverse_cpu, cuda=inverse_gpu, default=inverse_cpu)


@requires_s2fft
def _forward(
    maps: ArrayLike,
    L: int,
    shape_alms: int,
    spin: int,
    nside: int,
    sampling: str,
    reality: bool,
    precomps: list,
    iter: int = 0,
    fast_non_differentiable: bool = False,
    spmd: bool = False,
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
      Number of spherical harmonics iteration for regularisation, only relevant if fast_non_differentiable and on CPU (default: 0)
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
        spmd=spmd,
        iter=iter,
    )

    if fast_non_differentiable:
        lmax = L - 1

        if maps.squeeze().ndim == 2:
            nstokes = maps.squeeze().shape[0]
        else:
            nstokes = 1

        # Wrapper for map2alm, to prepare the pure callback of JAX
        def wrapper_map2alm(maps_, lmax=lmax, iter=iter, nside=nside):
            maps_np = jax.tree.map(np.asarray, maps_).reshape((nstokes, 12 * nside**2))
            if nstokes == 2:
                maps_np_extended = np.vstack([np.zeros_like(maps_np[0]), maps_np])
            else:
                maps_np_extended = maps_np
            alms = np.array(hp.map2alm(maps_np_extended, lmax=lmax, iter=iter))
            if len(alms.shape) == 2:
                return alms[-nstokes:, ...]
            return alms.reshape((nstokes, shape_alms))

        # Pure call back of map2alm, to be used with JAX for JIT compilation
        @partial(jax.jit, static_argnums=(1, 2))
        def pure_call_map2alm(maps_, lmax, nside):
            shape_output = (nstokes, shape_alms)
            alm_hp = jax.pure_callback(
                wrapper_map2alm,
                jax.ShapeDtypeStruct(shape_output, np.complex128),
                maps_.ravel(),
            )  #
            return (jax.vmap(partial(flm_hp_to_2d_fast, L=lmax + 1), in_axes=(0,))(alm_hp)).squeeze()
            # return flm_hp_to_2d_fast(alm_hp, lmax+1)

        forward_cpu = partial(pure_call_map2alm, nside=nside, lmax=lmax)

        # forward_gpu = partial(forward_s2fft, method='cuda')
        def forward_gpu(alms):
            if 'gpu' in jax.extend.backend.get_backend().platform:
                result = forward_s2fft(alms, method='cuda')
            else:
                result = pure_call_map2alm(alms, nside=nside, lmax=lmax)
            return result

    else:
        forward_cpu = partial(forward_s2fft, method='jax')
        forward_gpu = partial(forward_s2fft, method='jax')

    return jlax.platform_dependent(maps, cpu=forward_cpu, cuda=forward_gpu, default=forward_cpu)


def precompute_temperature_harmonic_transforms(
    nside: int, lmax: int = None, sampling: str = 'healpix', pix2harm: bool = False
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


def precompute_polarization_harmonic_transforms(
    nside: int, lmax: int = None, sampling: str = 'healpix', pix2harm: bool = False
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
        'fast_non_differentiable',
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
    inplace: bool = False,
    verbose: bool = True,
    healpy_ordering: bool = False,
    precomps_polar: list = None,
    fast_non_differentiable: bool = False,
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

    # if mmax is not None:
    #     raise NotImplementedError('Specifying mmax is not implemented.')
    if pixwin:
        raise NotImplementedError('Specifying pixwin is not implemented.')
    if fwhm != 0:
        raise NotImplementedError('Specifying fwhm is not implemented.')
    if sigma is not None:
        raise NotImplementedError('Specifying sigma is not implemented.')
    alms = jnp.asarray(alms)
    if alms.ndim == 0:
        raise ValueError('Input alms must have at least one dimension.')
    expected_ndim = 2 if healpy_ordering else 3
    if alms.ndim > expected_ndim + 1:
        raise ValueError('Input alms have too many dimensions.')
    if alms.ndim == expected_ndim + 1:
        return jax.vmap(_alm2map_pol, in_axes=(0,) + 10 * (None,))(
            alms, nside, lmax, mmax, pixwin, fwhm, sigma, inplace, healpy_ordering
        )
    # if alms.ndim > expected_ndim:
    # only happens if pol=True
    # raise NotImplementedError('TEB alms are not implemented.')

    if lmax is None:
        L = 3 * nside
        lmax = L - 1
    else:
        L = lmax + 1

    if mmax is None:
        mmax = lmax

    if healpy_ordering:
        alms = jax.vmap(partial(flm_hp_to_2d_fast, L=L), in_axes=(0,))(alms)

    spmd = False

    if precomps_polar is not None:
        precomps_plus2 = precomps_polar[0]
        precomps_minus2 = precomps_polar[1]
    else:
        precomps_plus2, precomps_minus2 = None, None

    if not fast_non_differentiable or 'gpu' in jax.extend.backend.get_backend().platform:
        map_plus2 = _inverse(
            -(alms[0, ...] + 1j * alms[1, ...]),
            L,
            spin=2,
            nside=nside,
            sampling='healpix',
            reality=False,
            precomps=precomps_plus2,
            fast_non_differentiable=fast_non_differentiable,
            spmd=spmd,
        )
        map_minus2 = _inverse(
            -(alms[0, ...] - 1j * alms[1, ...]),
            L,
            spin=-2,
            nside=nside,
            sampling='healpix',
            reality=False,
            precomps=precomps_minus2,
            fast_non_differentiable=fast_non_differentiable,
            spmd=spmd,
        )

        map_Q = (map_plus2 + map_minus2) / 2
        map_U = -1j * (map_plus2 - map_minus2) / 2
    else:
        map_QU = _inverse(
            alms,
            L,
            spin=None,
            nside=nside,
            sampling='healpix',
            reality=False,
            precomps=precomps_plus2,
            fast_non_differentiable=fast_non_differentiable,
            spmd=spmd,
        )
        map_Q = map_QU[0]
        map_U = map_QU[1]

    return jnp.vstack([map_Q, map_U])


def _compute_beam_window(lmax: int, fwhm: float = 0.0, sigma: float | None = None) -> ArrayLike:
    """Return Gaussian beam factors up to lmax."""
    if sigma is None:
        # Use jnp.where to handle fwhm=0 case in a JAX-compatible way
        sigma = jnp.where(fwhm == 0.0, 0.0, fwhm / jnp.sqrt(8.0 * jnp.log(2.0)))

    ell = jnp.arange(lmax + 1)
    # When sigma=0, exp returns 1.0 (no smoothing)
    return jnp.exp(-ell * (ell + 1) * sigma**2 / 2.0)


def _compute_cl(alms: ArrayLike, L: int, alms2: ArrayLike | None = None) -> ArrayLike:
    """Compute C_l from one or two alm grids in s2fft layout."""
    m_vals = jnp.arange(-L + 1, L)
    ell_vals = jnp.arange(L)
    ell_grid, m_grid = jnp.meshgrid(ell_vals, m_vals, indexing='ij')
    valid_mask = jnp.abs(m_grid) <= ell_grid

    alm_prod = jnp.abs(alms) ** 2 if alms2 is None else alms * jnp.conj(alms2)
    alm_prod_masked = alm_prod * valid_mask
    cl = alm_prod_masked.sum(axis=1) / (2 * ell_vals + 1)
    cl = cl.at[0].set(alm_prod_masked[0, L - 1])
    # Cross-spectra should be real-valued
    if alms2 is not None:
        cl = jnp.real(cl)
    return cl


def _generate_random_alm(cl: ArrayLike, lmax: int, mmax: int | None, prng_key: PRNGKeyArray) -> ArrayLike:
    """Draw random alm with healpy reality convention."""
    if mmax is None:
        mmax = lmax

    L = lmax + 1
    cl = jnp.asarray(cl)
    if cl.shape[0] < L:
        cl = jnp.pad(cl, (0, L - cl.shape[0]), constant_values=0)
    elif cl.shape[0] > L:
        cl = cl[:L]

    key_real, key_imag = jax.random.split(prng_key)
    rand_real = jax.random.normal(key_real, shape=(L, 2 * L - 1), dtype=jnp.float64)
    rand_imag = jax.random.normal(key_imag, shape=(L, 2 * L - 1), dtype=jnp.float64)

    m_vals = jnp.arange(-L + 1, L)
    ell_vals = jnp.arange(L)
    ell_grid, m_grid = jnp.meshgrid(ell_vals, m_vals, indexing='ij')
    valid_mask = (jnp.abs(m_grid) <= ell_grid) & (jnp.abs(m_grid) <= mmax)

    cl_grid = jnp.broadcast_to(cl[:, None], (L, 2 * L - 1))
    scale = jnp.sqrt(cl_grid / 2.0)
    scale = jnp.where(m_grid == 0, jnp.sqrt(cl_grid), scale)

    alms = (rand_real + 1j * rand_imag) * scale * valid_mask
    alms = alms.at[:, L - 1].set(alms[:, L - 1].real)

    left_half = alms[:, : L - 1]
    right_half = alms[:, L:]
    m_positive = jnp.arange(1, L)
    phase = (-1) ** m_positive
    conj_right_flipped = jnp.flip(jnp.conj(right_half), axis=1) * phase[None, :]

    ell_grid_left, m_neg_grid = jnp.meshgrid(jnp.arange(L), jnp.arange(-(L - 1), 0), indexing='ij')
    valid_left = jnp.abs(m_neg_grid) <= ell_grid_left
    left_half = conj_right_flipped * valid_left

    return jnp.concatenate([left_half, alms[:, L - 1 : L], right_half], axis=1)


def _alm2map_core(
    alms: ArrayLike,
    nside: int,
    lmax: int,
    mmax: int | None,
    method: str,
    spin: int,
    healpy_ordering: bool,
) -> ArrayLike:
    """Core alm2map implementation supporting spin-weighted transforms.

    Parameters
    ----------
    alms : ArrayLike
        Spherical harmonic coefficients
    nside : int
        Output map nside
    lmax : int
        Maximum l (note: L = lmax + 1)
    mmax : int, optional
        Maximum m
    method : str
        s2fft method ('jax', 'jax_healpy', 'jax_cuda')
    spin : int
        Spin weight (0 for scalar, 2 for polarization, etc.)
    healpy_ordering : bool
        Whether input alms use healpy ordering

    Returns
    -------
    ArrayLike
        Output map(s)
    """
    if mmax is not None and mmax != lmax:
        raise NotImplementedError('Specifying mmax != lmax is not yet implemented.')

    L = lmax + 1

    # Convert from healpy ordering if needed
    if healpy_ordering:
        if spin != 0:
            # For spin transforms, convert list of 2 alms to s2fft format
            alms = [flm_hp_to_2d_fast(alm, L) for alm in alms]
        else:
            alms = flm_hp_to_2d_fast(alms, L)

    sampling = 'healpix'
    spmd = False

    # For spin transforms, use single-transform approach
    # Reference: healpy alm2map_spin convention
    if spin != 0:
        # Input: [alm_plus, alm_minus] where for spin=2: alm_plus=E_lm, alm_minus=B_lm
        # Output: [Q_map, U_map] where Q and U are real-valued Stokes parameters
        alm_plus = alms[0]
        alm_minus = alms[1]

        # Construct spin-s harmonic coefficients
        # For spin=2 with E and B modes: _s a_lm = -(E_lm + i B_lm)
        alm_spin = -(alm_plus + 1j * alm_minus)

        # Generate precomputes for spin-s transform
        precomps = generate_precomputes_jax(L, spin, sampling, nside, False)

        # Single spin-s inverse transform
        # This yields the complex spin-s field: f = Q + iU (for spin=2)
        f = spherical.inverse(
            alm_spin,
            L,
            spin=spin,
            nside=nside,
            sampling=sampling,
            method=method,
            reality=False,  # Complex output
            precomps=precomps,
            spmd=spmd,
        )

        # Extract Q and U as real and imaginary parts
        q_map = jnp.real(f)
        u_map = jnp.imag(f)

        # Return real-valued maps
        return [q_map, u_map]
    else:
        # Scalar transform (spin=0)
        alms_complex = alms

        # Generate precomputes
        precomps = generate_precomputes_jax(L, spin, sampling, nside, False)

        # Call s2fft inverse transform
        f_complex = spherical.inverse(
            alms_complex,
            L,
            spin=spin,
            nside=nside,
            sampling=sampling,
            method=method,
            reality=True,  # Real output for scalar
            precomps=precomps,
            spmd=spmd,
        )

        return f_complex


def _map2alm_core(
    maps: ArrayLike,
    lmax: int,
    mmax: int | None,
    iter: int,
    method: str,
    spin: int,
) -> ArrayLike:
    """Core map2alm implementation supporting spin-weighted transforms.

    Parameters
    ----------
    maps : ArrayLike
        Input map(s)
    lmax : int
        Maximum l (note: L = lmax + 1)
    mmax : int, optional
        Maximum m
    iter : int
        Number of iterative refinement iterations
    method : str
        s2fft method ('jax', 'jax_healpy', 'jax_cuda')
    spin : int
        Spin weight (0 for scalar, 2 for polarization, etc.)
    Returns
    -------
    ArrayLike
        Output spherical harmonic coefficients
    """
    if mmax is not None and mmax != lmax:
        raise NotImplementedError('Specifying mmax != lmax is not yet implemented.')

    # For spin transforms, use dual-transform approach
    # Reference: healpy alm2map_spin/map2alm_spin convention
    if spin != 0:
        # Input: [Q_map, U_map] where Q and U are real-valued Stokes parameters
        # Output: [alm_E, alm_B] where E and B are complex-valued mode coefficients
        q_map = jnp.asarray(maps[0])
        u_map = jnp.asarray(maps[1])

        nside = npix2nside(q_map.shape[-1])
        L = lmax + 1

        # Construct spin-weighted maps for +spin and -spin transforms
        spin_map_pos = q_map + 1j * u_map  # _s S = Q + iU
        spin_map_neg = q_map - 1j * u_map  # _-s S = Q - iU

        sampling = 'healpix'
        spmd = False

        # Generate precomputes for both transforms
        precomps_pos = generate_precomputes_jax(L, spin, sampling, nside, False)
        precomps_neg = generate_precomputes_jax(L, -spin, sampling, nside, False)

        # Forward transforms for +spin and -spin
        flm_pos = spherical.forward(
            spin_map_pos,
            L,
            spin=spin,
            nside=nside,
            sampling=sampling,
            method=method,
            reality=False,  # Complex input
            precomps=precomps_pos,
            spmd=spmd,
            iter=iter,
        )

        flm_neg = spherical.forward(
            spin_map_neg,
            L,
            spin=-spin,
            nside=nside,
            sampling=sampling,
            method=method,
            reality=False,  # Complex input
            precomps=precomps_neg,
            spmd=spmd,
            iter=iter,
        )

        # Combine to get E and B mode coefficients
        # E and B are extracted using the HEALPix convention
        phase = (-1) ** spin
        alm_E = -0.5 * (flm_pos + phase * flm_neg)
        alm_B = 0.5j * (flm_pos - phase * flm_neg)

        return [alm_E, alm_B]
    else:
        # Scalar transform (spin=0)
        maps_complex = jnp.asarray(maps)
        nside = npix2nside(maps_complex.shape[-1])
        L = lmax + 1

        sampling = 'healpix'
        spmd = False

        precomps = generate_precomputes_jax(L, spin, sampling, nside, True)

        flm_complex = spherical.forward(
            maps_complex,
            L,
            spin=spin,
            nside=nside,
            sampling=sampling,
            method=method,
            reality=True,  # Real scalar field
            precomps=precomps,
            spmd=spmd,
            iter=iter,
        )

        return flm_complex


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
        'method',
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
    pol: bool = False,
    inplace: bool | None = None,
    verbose=True,
    healpy_ordering: bool = False,
    method: str = 'jax',
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
      Polarized TEB inputs are not supported yet. Must remain False.
    inplace : bool, optional
      Ignored in the JAX backend. Any truthy value will emit a warning.
    healpy ordering : bool, optional
      True if the input alms follow the healpy ordering. By default, the s2fft
      ordering is assumed.
    method : str, optional
      Transform backend ('jax', 'jax_healpy', 'jax_cuda'). Default: 'jax'.

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
    alms = jnp.asarray(alms)
    if alms.ndim == 0:
        raise ValueError('Input alms must have at least one dimension.')
    expected_ndim = 1 if healpy_ordering else 2
    if alms.ndim > expected_ndim + 1 + pol:
        raise ValueError('Input alms have too many dimensions.')

    if pol:
        raise NotImplementedError('Polarized alm2map (pol=True) is not implemented yet.')

    if inplace not in (None, False):
        warnings.warn('alm2map ignores inplace=True under JAX; arrays are immutable.', UserWarning)
    inplace_flag = False

    if pixwin:
        raise NotImplementedError('Pixel-window smoothing is not implemented; set pixwin=False.')

    # Handle batched input
    if alms.ndim == expected_ndim + 1 + pol:
        return jax.vmap(alm2map, in_axes=(0,) + 11 * (None,))(
            alms,
            nside,
            lmax,
            mmax,
            pixwin,
            fwhm,
            sigma,
            pol,
            inplace_flag,
            False,
            healpy_ordering,
            method,
        )

    if alms.ndim > expected_ndim:
        # only happens if pol=True
        raise NotImplementedError('TEB alms are not implemented.')

    if lmax is None:
        L = 3 * nside
        lmax = L - 1
    else:
        L = lmax + 1

    if mmax is not None:
        if lmax is None or mmax != lmax:
            raise NotImplementedError('Specifying mmax != lmax (or without lmax) is not implemented.')

    # Apply smoothing if requested
    if fwhm != 0 or sigma is not None:
        # Need to smooth the alms before transforming
        # Create filter function
        fl = jnp.ones(L)
        fl = fl * _compute_beam_window(L - 1, fwhm, sigma)

        # Apply filter
        alms_smoothed = almxfl(alms, fl, mmax=mmax, inplace=inplace_flag, healpy_ordering=healpy_ordering)
        alms = alms_smoothed

    # Call core function
    f = _alm2map_core(
        alms=alms, nside=nside, lmax=L - 1, mmax=mmax, method=method, spin=0, healpy_ordering=healpy_ordering
    )

    return f

    if pol:
        # return jnp.concatenate([map_temperature, maps_polarization], axis=-2)
        return jnp.vstack([map_temperature, maps_polarization])
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
        'fast_non_differentiable',
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
    fast_non_differentiable: bool = False,
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
      Number of iteration (default: 0)
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
    # if mmax is not None:
    #     raise NotImplementedError('Specifying mmax is not implemented.')
    # if iter != 0:
    #     raise NotImplementedError('Specifying iter > 0 is not implemented')
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
        lmax = L - 1
    else:
        L = lmax + 1

    if mmax is None:
        mmax = lmax

    spmd = False

    if precomps_polar is not None:
        precomps_plus2 = precomps_polar[0]
        precomps_minus2 = precomps_polar[1]
    else:
        precomps_plus2, precomps_minus2 = None, None

    shape_alms = mmax * (2 * lmax + 1 - mmax) // 2 + lmax + 1

    if not fast_non_differentiable or 'gpu' in jax.extend.backend.get_backend().platform:
        flm_spin_plus2 = _forward(
            maps[..., 0, :] + 1j * maps[..., 1, :],
            L,
            shape_alms=shape_alms,
            spin=2,
            nside=nside,
            sampling='healpix',
            reality=False,
            precomps=precomps_plus2,
            iter=iter,
            spmd=spmd,
            fast_non_differentiable=fast_non_differentiable,
        )
        flm_spin_minus2 = _forward(
            maps[..., 0, :] - 1j * maps[..., 1, :],
            L,
            shape_alms=shape_alms,
            spin=-2,
            nside=nside,
            sampling='healpix',
            reality=False,
            precomps=precomps_minus2,
            iter=iter,
            spmd=spmd,
            fast_non_differentiable=fast_non_differentiable,
        )

        flm_E = -(flm_spin_plus2 + flm_spin_minus2) / 2
        flm_B = 1j * (flm_spin_plus2 - flm_spin_minus2) / 2

    else:
        flm_EB = _forward(
            maps,
            L,
            spin=None,
            shape_alms=shape_alms,
            nside=nside,
            sampling='healpix',
            reality=True,
            precomps=precomps_plus2,
            iter=iter,
            spmd=spmd,
            fast_non_differentiable=True,
        )
        flm_E, flm_B = flm_EB[..., 0, :, :], flm_EB[..., 1, :, :]

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
        'method',
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
    method: str = 'jax',
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
      Number of iteration (default: 0)
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
    method : str, optional
      Transform backend ('jax', 'jax_healpy', 'jax_cuda'). Default: 'jax'.

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

    # Handle batched input
    if maps.ndim > 1:
        if pol:
            raise NotImplementedError('TQU maps are not implemented.')
        return jax.vmap(map2alm, in_axes=(0,) + 11 * (None,))(
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
            method,
        )
    elif maps.ndim > 3:
        raise ValueError('The input map has too many dimensions.')

    maps = jnp.asarray(maps)
    nside = npix2nside(maps.shape[-1])
    target_L = 3 * nside if lmax is None else lmax + 1
    if target_L < 2 * nside:
        raise NotImplementedError('map2alm requires lmax >= 2*nside - 1 for s2fft transforms.')
    target_lmax = target_L - 1

    if mmax is not None and mmax != target_lmax:
        raise NotImplementedError('Specifying mmax != lmax is not implemented.')

    # Call core function
    flm = _map2alm_core(
        maps=maps,
        lmax=target_L - 1,
        mmax=mmax,
        iter=iter,
        method=method,
        spin=0,
    )


    if healpy_ordering:
        flm = flm_2d_to_hp_fast(flm, target_L)

    if pol:
        return flm, flm_E, flm_B
    return flm


@partial(jax.jit, static_argnames=['mmax', 'inplace', 'healpy_ordering'])
def almxfl(alm: ArrayLike, fl: ArrayLike, mmax: int | None = None, inplace: bool = False, healpy_ordering: bool = True):
    """Multiply alm by a filter function fl.

    Parameters
    ----------
    alm : array-like
        The input alm, shape (nalm,) for healpy ordering or (L, 2L-1) for s2fft ordering
    fl : array-like
        The filter function, shape (lmax+1,). The function is applied as:
        alm_out[l,m] = alm[l,m] * fl[l]
    mmax : int, optional
        Maximum m. If None, assumes mmax=lmax
    inplace : bool, optional
        If True, modify alm in place (currently ignored for JAX arrays)
    healpy_ordering : bool, optional
        If True, alm uses healpy 1D format. If False, uses s2fft 2D format.

    Returns
    -------
    array-like
        Filtered alm, same shape and format as input

    Notes
    -----
    This function is commonly used for:
    - Beam convolution: fl = exp(-l(l+1)*sigma^2/2)
    - Pixel window correction: fl = pixel_window[l]
    - Arbitrary filtering: fl = filter_function[l]

    Examples
    --------
    Apply a Gaussian beam with FWHM of 5 arcmin:

    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> lmax = 64
    >>> alm = jnp.zeros((lmax+1, 2*lmax+1), dtype=complex)  # s2fft format
    >>> fwhm_rad = jnp.radians(5.0 / 60.0)
    >>> sigma = fwhm_rad / jnp.sqrt(8 * jnp.log(2))
    >>> ell = jnp.arange(lmax + 1)
    >>> bl = jnp.exp(-ell * (ell + 1) * sigma**2 / 2.0)
    >>> alm_smoothed = jhp.almxfl(alm, bl, healpy_ordering=False)
    """
    alm = jnp.asarray(alm)
    fl = jnp.asarray(fl)

    # Determine lmax from fl
    lmax = fl.shape[0] - 1
    L = lmax + 1

    if healpy_ordering:
        # Convert to s2fft 2D format for easier manipulation
        alm_2d = flm_hp_to_2d_fast(alm, L)

        # Apply filter
        fl_2d = fl[:, None]  # Broadcast over m dimension
        alm_filtered = alm_2d * fl_2d

        # Convert back to healpy format
        return flm_2d_to_hp_fast(alm_filtered, L)
    else:
        # s2fft 2D format: directly broadcast fl over m dimension
        fl_2d = fl[:, None]
        return alm * fl_2d


@partial(jax.jit, static_argnames=['lmax', 'pol'])
def gauss_beam(fwhm: float, lmax: int = 512, pol: bool = False) -> ArrayLike:
    """Compute Gaussian beam window function.

    Generates the spherical harmonic transform of an axisymmetric Gaussian beam.
    The beam window function B_l represents how an instrumental beam modifies
    observed power spectra: C_l^observed = C_l^true × B_l².

    Parameters
    ----------
    fwhm : float
        Full Width at Half Maximum of the Gaussian beam in radians (required)
    lmax : int, optional
        Maximum multipole moment l. Default: 512
    pol : bool, optional
        If True, returns polarization beam components.
        Not supported yet. Raises NotImplementedError if True. Default: False

    Returns
    -------
    beam : array-like
        Beam window function, shape (lmax+1,)
        Values represent B_l = exp(-l(l+1)*sigma²/2) where sigma = fwhm/sqrt(8*ln(2))

    Notes
    -----
    This function is commonly used in CMB analysis to model the smoothing effect
    of telescope beams on sky maps. The Gaussian beam is characterized by:
    B_l = exp(-l(l+1)*sigma²/2), where sigma = fwhm / sqrt(8*ln(2))

    Examples
    --------
    Create a beam window for a 5 arcminute FWHM beam:

    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> fwhm_arcmin = 5.0
    >>> fwhm_rad = jnp.radians(fwhm_arcmin / 60.0)
    >>> beam = jhp.gauss_beam(fwhm_rad, lmax=2048)
    """
    if pol:
        raise NotImplementedError('pol=True (polarization beam components) is not supported yet')

    return _compute_beam_window(lmax, fwhm=fwhm, sigma=None)


@partial(jax.jit, static_argnames=['lmax', 'mmax', 'lmax_out', 'nspec', 'healpy_ordering'])
def alm2cl(
    alms1: ArrayLike | list[ArrayLike],
    alms2: ArrayLike | list[ArrayLike] | None = None,
    lmax: int | None = None,
    mmax: int | None = None,
    lmax_out: int | None = None,
    nspec: int | None = None,
    healpy_ordering: bool = True,
) -> ArrayLike | tuple[ArrayLike, ...]:
    """Compute power spectra from spherical harmonic coefficients.

    Calculates auto-spectra and cross-spectra from one or more sets of alm coefficients.
    For n input alm arrays, produces n(n+1)/2 spectra containing all auto- and cross-spectra.

    Parameters
    ----------
    alms1 : array-like or list of arrays
        Spherical harmonic coefficients. Can be:
        - Single array for one alm
        - List/tuple of arrays for multiple alms (produces n(n+1)/2 spectra)
    alms2 : array-like or list of arrays, optional
        Second set of alm coefficients for cross-spectra. If None, computes auto-spectra.
    lmax : int, optional
        Maximum l value of input alm. If None, inferred from array size.
    mmax : int, optional
        Maximum m value of input alm. Default: lmax
    lmax_out : int, optional
        Maximum l value in output spectra. If None, uses lmax from input.
        Note: Currently must equal lmax if provided.
    nspec : int, optional
        Number of spectra to return. If None, returns all.
    healpy_ordering : bool, optional
        If True, input alms use healpy 1D format. If False, uses s2fft 2D format.
        Default: True

    Returns
    -------
    cl : array-like or tuple of arrays
        Power spectrum/spectra. Shape (lmax+1,) for single spectrum.
        For multiple alms: tuple of n(n+1)/2 spectra in order: (11, 12, 22, 13, 23, 33, ...)

    Notes
    -----
    The power spectrum is computed as:
        C_l = (1 / (2*l + 1)) * sum_m |a_lm|^2  (auto-spectrum)
        C_l = (1 / (2*l + 1)) * sum_m a_lm * conj(a'_lm)  (cross-spectrum)

    For multiple input alms, the function computes all unique auto- and cross-spectra.
    The ordering follows healpy convention: spectra are returned in diagonal-major order.

    Examples
    --------
    Compute auto-spectrum from a single alm:

    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> alm = jnp.array([...])  # Your alm coefficients
    >>> cl = jhp.alm2cl(alm)

    Compute cross-spectrum between two alms:

    >>> cl_cross = jhp.alm2cl(alm1, alm2)

    Compute all spectra from multiple alms:

    >>> alms = [alm1, alm2, alm3]
    >>> cl_tuple = jhp.alm2cl(alms)  # Returns (cl11, cl12, cl22, cl13, cl23, cl33)
    """
    # Check if lmax_out is provided and different from lmax
    if lmax_out is not None and lmax is not None and lmax_out != lmax:
        raise ValueError(f'lmax_out ({lmax_out}) must equal lmax ({lmax}) if both provided')

    # Handle multiple alms case
    if isinstance(alms1, (list, tuple)):
        alms1_list = [jnp.asarray(a) for a in alms1]
        n_alms = len(alms1_list)

        # Infer lmax from first alm if not provided
        if lmax is None:
            if healpy_ordering:
                nalm = alms1_list[0].shape[0]
                lmax = int((-1 + np.sqrt(1 + 8 * (nalm - 1))) / 2)
            else:
                lmax = alms1_list[0].shape[0] - 1

        L = lmax + 1

        # Convert to s2fft format if needed
        if healpy_ordering:
            alms1_list = [flm_hp_to_2d_fast(a, L) for a in alms1_list]

        # Compute all n(n+1)/2 spectra
        # HealPy order: all auto-spectra first, then cross-spectra grouped by separation
        spectra = []

        # First, all auto-spectra (i, i)
        for i in range(n_alms):
            cl = _compute_cl(alms1_list[i], L, None)
            spectra.append(cl)

        # Then, cross-spectra grouped by separation (j-i): (0,1), (1,2), ..., (0,2), (1,3), ...
        for separation in range(1, n_alms):
            for i in range(n_alms - separation):
                j = i + separation
                cl = _compute_cl(alms1_list[i], L, alms1_list[j])
                spectra.append(cl)

        # Apply nspec if requested
        if nspec is not None:
            spectra = spectra[:nspec]

        return tuple(spectra) if len(spectra) > 1 else spectra[0]

    # Single alm case
    alms1 = jnp.asarray(alms1)

    # Infer lmax if not provided
    if lmax is None:
        if healpy_ordering:
            nalm = alms1.shape[0]
            lmax = int((-1 + np.sqrt(1 + 8 * (nalm - 1))) / 2)
        else:
            lmax = alms1.shape[0] - 1

    L = lmax + 1

    # Convert to s2fft format if needed
    if healpy_ordering:
        alms1 = flm_hp_to_2d_fast(alms1, L)

    if alms2 is None:
        # Auto-spectrum
        cl = _compute_cl(alms1, L, None)
    else:
        # Cross-spectrum
        alms2 = jnp.asarray(alms2)
        if healpy_ordering:
            alms2 = flm_hp_to_2d_fast(alms2, L)
        cl = _compute_cl(alms1, L, alms2)

    # Apply nspec if requested
    if nspec is not None:
        cl = cl[:nspec]

    return cl


@partial(
    jax.jit,
    static_argnames=[
        'lmax',
        'mmax',
        'iter',
        'alm',
        'nspec',
        'pol',
        'only_pol',
        'use_weights',
        'datapath',
        'gal_cut',
        'use_pixel_weights',
        'method',
    ],
)
@requires_s2fft
def anafast(
    map1: ArrayLike,
    map2: ArrayLike | None = None,
    nspec: int | None = None,
    lmax: int | None = None,
    mmax: int | None = None,
    iter: int = 3,
    alm: bool = False,
    pol: bool = True,
    use_weights: bool = False,
    datapath: str | None = None,
    gal_cut: float = 0,
    use_pixel_weights: bool = False,
    method: str = 'jax',
):
    """Compute the angular power spectrum from HEALPix map(s).

    Parameters
    ----------
    map1 : array-like
        First input map, shape (npix,). Must be in RING ordering.
    map2 : array-like, optional
        Second input map for cross-spectrum. If None, computes auto-spectrum of map1.
    nspec : int, optional
        Number of spectra to return. If None, returns all spectra.
        If provided, returns only the first nspec values. Default: None
    lmax : int, optional
        Maximum multipole l. Default: 3*nside - 1
    mmax : int, optional
        Maximum m. Default: lmax
    iter : int, optional
        Number of iterative refinement iterations. Default: 3
    alm : bool, optional
        If True, return both power spectrum and alm coefficients.
        If False, return only power spectrum. Default: False
    pol : bool, optional
        If True, assumes input maps are TQU (polarized).
        Not supported yet. Raises NotImplementedError if True. Default: True
    use_weights : bool, optional
        Enable ring weighting. Not supported. Raises NotImplementedError if True.
        Default: False
    datapath : str, optional
        Directory containing pixel weights. Not supported.
        Raises NotImplementedError if not None. Default: None
    gal_cut : float, optional
        Galactic cut in degrees. Not supported. Raises NotImplementedError if != 0.
        Default: 0
    use_pixel_weights : bool, optional
        Enable pixel-by-pixel weighting. Not supported.
        Raises NotImplementedError if True. Default: False
    method : str, optional
        Transform method ('jax', 'jax_healpy', 'jax_cuda'). Default: 'jax'
        JAX-specific parameter not present in healpy.

    Returns
    -------
    cl : array-like
        Angular power spectrum, shape (lmax+1,)
    alm1 : array-like, optional
        Alm coefficients of map1 in s2fft format (only if alm=True)
    alm2 : array-like, optional
        Alm coefficients of map2 in s2fft format (only if alm=True and map2 is provided)

    Notes
    -----
    The power spectrum is computed as:
        C_l = (1 / (2*l + 1)) * sum_m |a_lm|^2  (auto-spectrum)
        C_l = (1 / (2*l + 1)) * sum_m a_lm * conj(a'_lm)  (cross-spectrum)

    This function matches the healpy.sphtfunc.anafast API but only supports
    scalar (non-polarized) input maps. Several healpy parameters related to
    weighting and polarization are not yet implemented.

    Examples
    --------
    Compute auto-spectrum of a map:

    >>> import healpy as hp
    >>> import jax_healpy as jhp
    >>> nside = 32
    >>> map_data = hp.synfast(cl, nside)  # Generate test map
    >>> cl_estimated = jhp.anafast(map_data, pol=False)
    """
    # Validate unsupported parameters
    if pol:
        raise NotImplementedError('pol=True (polarized input) is not supported yet')
    if use_weights:
        raise NotImplementedError('use_weights is not supported')
    if datapath is not None:
        raise NotImplementedError('datapath is not supported')
    if gal_cut != 0:
        raise NotImplementedError('gal_cut is not supported')
    if use_pixel_weights:
        raise NotImplementedError('use_pixel_weights is not supported')

    map1 = jnp.asarray(map1)
    nside = npix2nside(map1.shape[-1])

    if lmax is None:
        lmax = 3 * nside - 1

    L = lmax + 1

    # Compute alm of first map
    alm1 = map2alm(map1, lmax=lmax, mmax=mmax, iter=iter, pol=False, healpy_ordering=False, method=method)

    if map2 is None:
        # Auto-spectrum
        cl = _compute_cl(alm1, L)
        if nspec is not None:
            cl = cl[:nspec]
        if alm:
            return cl, alm1
        return cl
    else:
        # Cross-spectrum
        map2 = jnp.asarray(map2)
        alm2 = map2alm(map2, lmax=lmax, mmax=mmax, iter=iter, pol=False, healpy_ordering=False, method=method)
        cl = _compute_cl(alm1, L, alm2)
        if nspec is not None:
            cl = cl[:nspec]
        if alm:
            return cl, alm1, alm2
        return cl


@partial(jax.jit, static_argnames=['fwhm', 'sigma', 'mmax', 'pol', 'inplace', 'verbose', 'healpy_ordering'])
def smoothalm(
    alms: ArrayLike,
    fwhm: float = 0.0,
    sigma: float | None = None,
    beam_window: ArrayLike | None = None,
    pol: bool = True,
    mmax: int | None = None,
    verbose: bool = True,
    inplace: bool = True,
    healpy_ordering: bool = True,
) -> ArrayLike:
    """Smooth spherical harmonic coefficients with a Gaussian beam.

    Applies smoothing directly to alm coefficients by multiplying with a beam window function.

    Parameters
    ----------
    alms : array-like
        Spherical harmonic coefficients to smooth
    fwhm : float, optional
        Full Width at Half Maximum of Gaussian beam in radians. Default: 0.0
    sigma : float, optional
        Gaussian standard deviation in radians (overrides fwhm if provided). Default: None
    beam_window : array-like, optional
        Custom beam window function (supersedes both fwhm and sigma). Default: None
    pol : bool, optional
        If True, expects TEB coefficients.
        Not supported yet. Raises NotImplementedError if True. Default: True
    mmax : int, optional
        Maximum m value for alm coefficients. Default: lmax (inferred from alm size)
    verbose : bool, optional
        Deprecated parameter (ignored). Default: True
    inplace : bool, optional
        If True, modifies input in-place. Ignored in JAX (arrays are immutable). Default: True
    healpy_ordering : bool, optional
        If True, input/output use healpy 1D format. If False, s2fft 2D format. Default: True

    Returns
    -------
    alms_smoothed : array-like
        Smoothed spherical harmonic coefficients

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> alm = jnp.array([...])  # Your alm coefficients
    >>> fwhm_rad = jnp.radians(5.0 / 60.0)  # 5 arcmin
    >>> alm_smooth = jhp.smoothalm(alm, fwhm=fwhm_rad, pol=False)
    """
    if pol:
        raise NotImplementedError('pol=True (polarized alm) is not supported yet')
    if inplace not in (None, True):
        warnings.warn('smoothalm ignores inplace parameter; JAX arrays are immutable', UserWarning)
    if verbose not in (None, True):
        warnings.warn('verbose parameter is ignored in JAX implementation', UserWarning)

    alms = jnp.asarray(alms)

    # Determine lmax from alm size
    if healpy_ordering:
        nalm = alms.shape[0]
        lmax = int((-1 + np.sqrt(1 + 8 * (nalm - 1))) / 2)
    else:
        lmax = alms.shape[0] - 1

    # Get or create beam window
    if beam_window is not None:
        bl = jnp.asarray(beam_window)
    else:
        bl = _compute_beam_window(lmax, fwhm=fwhm, sigma=sigma)

    # Apply smoothing using almxfl
    return almxfl(alms, bl, mmax=mmax, inplace=False, healpy_ordering=healpy_ordering)



@partial(
    jax.jit,
    static_argnames=[
        'fwhm',
        'sigma',
        'pol',
        'iter',
        'lmax',
        'mmax',
        'use_weights',
        'use_pixel_weights',
        'datapath',
        'verbose',
        'nest',
    ],
)
@requires_s2fft
def smoothing(
    map_in: ArrayLike,
    fwhm: float = 0.0,
    sigma: float | None = None,
    beam_window: ArrayLike | None = None,
    pol: bool = True,
    iter: int = 3,
    lmax: int | None = None,
    mmax: int | None = None,
    use_weights: bool = False,
    use_pixel_weights: bool = False,
    datapath: str | None = None,
    verbose: bool = True,
    nest: bool = False,
) -> ArrayLike:
    """Smooth a HEALPix map with a Gaussian beam.

    Applies Gaussian symmetric beam smoothing by transforming to harmonic space,
    applying beam filter, and transforming back to pixel space.

    Parameters
    ----------
    map_in : array-like
        Input HEALPix map, shape (npix,). Must be in RING ordering.
    fwhm : float, optional
        Full Width at Half Maximum in radians. Default: 0.0
    sigma : float, optional
        Gaussian sigma in radians (overrides fwhm if specified). Default: None
    beam_window : array-like, optional
        Custom beam window function (supersedes fwhm and sigma). Default: None
    pol : bool, optional
        If True, handles polarization (T,Q,U maps).
        Not supported yet. Raises NotImplementedError if True. Default: True
    iter : int, optional
        Number of iterations for map2alm. Default: 3
    lmax : int, optional
        Maximum l for spherical harmonic transform. Default: 3*nside-1
    mmax : int, optional
        Maximum m for spherical harmonic transform. Default: lmax
    use_weights : bool, optional
        Use ring weights. Not supported. Raises NotImplementedError if True. Default: False
    use_pixel_weights : bool, optional
        Use pixel-by-pixel weights. Not supported. Raises NotImplementedError if True. Default: False
    datapath : str, optional
        Path to weight files. Not supported. Raises NotImplementedError if not None. Default: None
    verbose : bool, optional
        Verbosity control. Ignored. Default: True
    nest : bool, optional
        Input map ordering. Not supported (only RING).
        Raises NotImplementedError if True. Default: False

    Returns
    -------
    map_out : array-like
        Smoothed HEALPix map in RING ordering, shape (npix,)

    Notes
    -----
    The smoothing does not remove monopole/dipole components.

    Examples
    --------
    >>> import jax_healpy as jhp
    >>> import jax.numpy as jnp
    >>> map_in = jnp.array([...])  # Your HEALPix map
    >>> fwhm_rad = jnp.radians(10.0)  # 10 degrees
    >>> map_smooth = jhp.smoothing(map_in, fwhm=fwhm_rad, pol=False)
    """
    if pol:
        raise NotImplementedError('pol=True (polarized maps) is not supported yet')
    if nest:
        raise NotImplementedError('nest=True is not supported; only RING ordering accepted')
    if use_weights:
        raise NotImplementedError('use_weights is not supported')
    if use_pixel_weights:
        raise NotImplementedError('use_pixel_weights is not supported')
    if datapath is not None:
        raise NotImplementedError('datapath is not supported')
    if verbose not in (None, True):
        warnings.warn('verbose parameter is ignored in JAX implementation', UserWarning)

    map_in = jnp.asarray(map_in)

    # Transform to alm
    alms = map2alm(map_in, lmax=lmax, mmax=mmax, iter=iter, pol=False, healpy_ordering=False)

    # Smooth alms
    alms_smooth = smoothalm(
        alms, fwhm=fwhm, sigma=sigma, beam_window=beam_window, pol=False, mmax=mmax, healpy_ordering=False
    )

    # Transform back to map
    map_out = alm2map(alms_smooth, nside=npix2nside(map_in.shape[0]), lmax=lmax, mmax=mmax, healpy_ordering=False)

    return map_out


@partial(jax.jit, static_argnames=['lmax', 'mmax', 'new', 'verbose', 'healpy_ordering'])
def synalm(
    prng_key: PRNGKeyArray,
    cls: ArrayLike | tuple[ArrayLike, ...],
    lmax: int | None = None,
    mmax: int | None = None,
    new: bool = False,
    verbose: bool = True,
    healpy_ordering: bool = True,
) -> ArrayLike | tuple[ArrayLike, ...]:
    """Generate random spherical harmonic coefficients from power spectrum.

    Creates alm coefficients drawn from Gaussian distributions with variances
    determined by the input power spectrum/spectra.

    Parameters
    ----------
    prng_key : PRNGKeyArray
        JAX random number generator key (required first argument)
    cls : array-like or tuple of arrays
        Power spectrum data. Can be:
        - Single 1D array for scalar (temperature-only) case
        - Tuple of 4 arrays (TT, EE, BB, TE) for polarization
        - Tuple of 6 arrays (TT, EE, BB, TE, EB, TB) for full correlations
    lmax : int, optional
        Maximum multipole moment. If None, uses max(len(cls)-1, 2*nside-1 if nside given)
    mmax : int, optional
        Maximum azimuthal order. Default: lmax
    new : bool, optional
        Ordering convention for input spectra. Default: False
        Note: Currently only default (False) ordering is fully supported.
    verbose : bool, optional
        Verbosity control. Ignored. Default: True
    healpy_ordering : bool, optional
        If True, output uses healpy 1D format. If False, s2fft 2D format. Default: True

    Returns
    -------
    alms : array-like or tuple of arrays
        Generated spherical harmonic coefficients.
        Single array for scalar case, tuple of 3 arrays (T, E, B) for polarization.

    Notes
    -----
    For scalar case:
    - m=0: a_l0 is real, drawn from N(0, C_l)
    - m>0: a_lm is complex, real and imag parts drawn from N(0, C_l/2)

    Polarization (TEB) case is not yet fully implemented.

    Examples
    --------
    Generate scalar alms:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> key = jax.random.PRNGKey(42)
    >>> ell = jnp.arange(65)
    >>> cl = 1.0 / (ell + 10)**2
    >>> alm = jhp.synalm(key, cl, lmax=64)
    """
    if new:
        warnings.warn('new parameter ordering convention not fully implemented', UserWarning)
    if verbose not in (None, True):
        warnings.warn('verbose parameter is ignored in JAX implementation', UserWarning)

    # Check if cls is a tuple (polarization case)
    if isinstance(cls, (tuple, list)):
        raise NotImplementedError(
            'Full TEB polarization support not yet implemented. Only scalar (single power spectrum) case supported.'
        )

    # Scalar case - use existing _generate_random_alm
    cls = jnp.asarray(cls)

    if lmax is None:
        lmax = cls.shape[0] - 1

    # Generate random alms
    alms = _generate_random_alm(cls, lmax, mmax, prng_key)

    # Convert to healpy ordering if requested
    if healpy_ordering:
        L = lmax + 1
        alms = flm_2d_to_hp_fast(alms, L)

    return alms


def pixwin(nside: int, pol: bool = False, lmax: int | None = None, datapath: str | None = None) -> ArrayLike:
    """Get pixel window function for a given nside.

    Returns the pixel window function that corrects for the smoothing effect
    of finite pixel size in HEALPix maps.

    Parameters
    ----------
    nside : int
        The nside value for which to retrieve the pixel window function
    pol : bool, optional
        If True, returns both temperature and polarization pixel windows.
        Default: False
    lmax : int, optional
        Maximum multipole moment of the power spectrum. Default: 3*nside-1
    datapath : str, optional
        Directory path for locating pixel window function files. Default: None

    Returns
    -------
    pixwin : array-like or tuple of arrays
        Temperature pixel window function, or (temperature, polarization) tuple if pol=True

    Raises
    ------
    NotImplementedError
        This function is not yet implemented in jax-healpy.

    Notes
    -----
    Pixel window functions are pre-computed and depend only on nside.
    Implementation requires bundling data files or downloading from healpy-data repository.
    """
    raise NotImplementedError(
        'pixwin is not yet implemented in jax-healpy. '
        'Pixel window data files need to be integrated. '
        'Consider using healpy.pixwin() and converting to JAX arrays as a workaround.'
    )


@partial(
    jax.jit,
    static_argnames=['nside', 'lmax', 'mmax', 'alm', 'pol', 'pixwin', 'fwhm', 'sigma', 'new', 'verbose', 'method'],
)
@requires_s2fft
def synfast(
    prng_key: PRNGKeyArray,
    cls: ArrayLike,
    nside: int,
    lmax: int | None = None,
    mmax: int | None = None,
    alm: bool = False,
    pol: bool = True,
    pixwin: bool = False,
    fwhm: float = 0.0,
    sigma: float | None = None,
    new: bool = False,
    verbose: bool = True,
    method: str = 'jax',
):
    """Generate a random HEALPix map from a power spectrum.

    Parameters
    ----------
    cls : array-like
        Input power spectrum C_l, shape (lmax+1,) or longer
    nside : int
        HEALPix nside parameter for output map
    lmax : int, optional
        Maximum multipole l. If None (or negative), matches healpy by using
        ``min(len(cls) - 1, 3 * nside - 1)``. Because jax-healpy relies on
        s2fft transforms that require ``lmax >= 2 * nside - 1``, a
        ``ValueError`` is raised if the requested or derived ``lmax`` falls
        below that threshold.
    mmax : int, optional
        Maximum m. Default: lmax
    alm : bool, optional
        If True, return both the map and the alm coefficients used to generate it.
        If False, return only the map. Default: False
    pol : bool, optional
        If True, assumes polarized output (TQU or TEB maps).
        Not supported yet. Raises NotImplementedError if True. Default: True
    pixwin : bool, optional
        Apply pixel window function. Not supported yet.
        Raises NotImplementedError if True. Default: False
    fwhm : float, optional
        Gaussian beam FWHM in radians for smoothing. Default: 0.0
    sigma : float, optional
        Gaussian beam sigma in radians. If specified, overrides fwhm.
    new : bool, optional
        If True, uses JAX PRNG with seed parameter.
        If False, uses numpy random state (not recommended for JAX).
        Default: False (to match healpy)
    verbose : bool, optional
        Verbosity flag. Accepted for API compatibility but ignored with a warning.
        Default: True
    method : str, optional
        Transform method ('jax', 'jax_healpy', 'jax_cuda'). Default: 'jax'
        JAX-specific parameter not present in healpy.
    seed : int, optional
        Random seed for reproducibility (only used if new=True). Default: 0
        JAX-specific parameter not present in healpy.

    Returns
    -------
    map : array-like
        Random HEALPix map in RING scheme, shape (npix,)
    alm : array-like, optional
        Spherical harmonic coefficients in s2fft format (only if alm=True)

    Notes
    -----
    This function generates random alm coefficients from the input power spectrum,
    then transforms them to a map using alm2map with optional smoothing.

    For each (l,m):
    - m=0: a_l0 is real, drawn from N(0, C_l)
    - m>0: a_lm is complex, real and imag parts drawn from N(0, C_l/2)

    This function matches the healpy.sphtfunc.synfast API but only supports
    scalar (non-polarized) output maps. Polarization and pixel window
    functionality are not yet implemented.

    Examples
    --------
    Generate a random map from a power spectrum:

    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> lmax = 64
    >>> ell = jnp.arange(lmax + 1)
    >>> cl = 1.0 / (ell + 10)**2  # Simple power-law spectrum
    >>> nside = 32
    >>> random_map = jhp.synfast(cl, nside, new=True, seed=42, pol=False)

    Generate a map and get the alm coefficients:

    >>> random_map, alm_coeffs = jhp.synfast(cl, nside, alm=True, new=True, seed=42, pol=False)
    """
    # Validate unsupported parameters
    if pol:
        raise NotImplementedError('pol=True (polarized output) is not supported yet')
    if pixwin:
        raise NotImplementedError('pixwin=True is not supported yet')
    if verbose not in (None, True):
        warnings.warn('verbose parameter is ignored in JAX implementation', UserWarning)
    if new is True:
        warnings.warn('new parameter is ignored in JAX implementation', UserWarning)

    cls = jnp.asarray(cls)
    cls_lmax = cls.shape[0] - 1

    if lmax is None or lmax < 0:
        healpy_default = min(cls_lmax, 3 * nside - 1)
        resolved_lmax = healpy_default
    else:
        resolved_lmax = int(lmax)

    min_supported_lmax = 2 * nside - 1
    if resolved_lmax < min_supported_lmax:
        raise ValueError(
            f'synfast requires lmax >= {min_supported_lmax} for nside={nside} when using the '
            f's2fft backend (got lmax={resolved_lmax}). Provide a longer C_l array or specify a '
            'larger lmax (you can zero-pad the spectrum) before calling synfast.'
        )

    alms = _generate_random_alm(cls, resolved_lmax, mmax, prng_key=prng_key)

    # Transform to map with optional smoothing
    map_synth = alm2map(
        alms,
        nside,
        lmax=resolved_lmax,
        mmax=mmax,
        pixwin=pixwin,
        fwhm=fwhm,
        sigma=sigma,
        healpy_ordering=False,
        method=method,
    )

    if alm:
        return map_synth, alms
    return map_synth


@partial(jax.jit, static_argnames=['spin', 'lmax', 'mmax', 'method', 'healpy_ordering'])
@requires_s2fft
def map2alm_spin(
    maps, spin: int, lmax: int | None = None, mmax: int | None = None, method: str = 'jax', healpy_ordering: bool = True
):
    """Compute spin-weighted spherical harmonic coefficients from HEALPix maps.

    Parameters
    ----------
    maps : list of array-like
        List of 2 input maps for spin != 0, or single map for spin=0.
        Each map has shape (npix,)
    spin : int
        Spin weight (0 for scalar, 2 for polarization, etc.)
    lmax : int, optional
        Maximum multipole l. Default: 3*nside - 1
    mmax : int, optional
        Maximum m. Default: lmax
    method : str, optional
        Transform method ('jax', 'jax_healpy', 'jax_cuda'). Default: 'jax'
    healpy_ordering : bool, optional
        If True, return alms in healpy format. If False, s2fft format. Default: True

    Returns
    -------
    alms : list of array-like or array-like
        For spin != 0: list of 2 alm arrays
        For spin = 0: single alm array

    Notes
    -----
    For polarization (spin=2), input maps should be [Q, U] and output will be [E_lm, B_lm].

    Examples
    --------
    Compute E and B modes from Q and U maps:

    >>> import healpy as hp
    >>> import jax_healpy as jhp
    >>> nside = 32
    >>> Q_map = hp.synfast(..., nside)
    >>> U_map = hp.synfast(..., nside)
    >>> E_lm, B_lm = jhp.map2alm_spin([Q_map, U_map], spin=2)
    """
    if spin != 0:
        # For spin != 0, expect list of 2 maps
        if not isinstance(maps, (list, tuple)) or len(maps) != 2:
            raise ValueError(f'For spin={spin}, maps must be a list/tuple of 2 arrays')
        maps = [jnp.asarray(m) for m in maps]
        nside = npix2nside(maps[0].shape[-1])
    else:
        maps = jnp.asarray(maps)
        nside = npix2nside(maps.shape[-1])

    if lmax is None:
        lmax = 3 * nside - 1
    target_L = lmax + 1
    if target_L < 2 * nside:
        raise NotImplementedError('map2alm_spin requires lmax >= 2*nside - 1 for s2fft transforms.')

    if mmax is not None and mmax != lmax:
        raise NotImplementedError('Specifying mmax != lmax is not implemented.')

    # Call core function (iter=0 for spin transforms, no iteration support yet)
    flm = _map2alm_core(
        maps=maps,
        lmax=target_L - 1,
        mmax=mmax,
        iter=0,
        method=method,
        spin=spin,
    )


    if healpy_ordering:
        if spin != 0:
            flm = [flm_2d_to_hp_fast(f, target_L) for f in flm]
        else:
            flm = flm_2d_to_hp_fast(flm, target_L)

    if pol:
          return flm, flm_E, flm_B
    return flm


@partial(jax.jit, static_argnames=['nside', 'spin', 'lmax', 'mmax', 'method', 'healpy_ordering'])
@requires_s2fft
def alm2map_spin(
    alms,
    nside: int,
    spin: int,
    lmax: int | None = None,
    mmax: int | None = None,
    method: str = 'jax',
    healpy_ordering: bool = True,
):
    """Compute HEALPix maps from spin-weighted spherical harmonic coefficients.

    Parameters
    ----------
    alms : list of array-like or array-like
        For spin != 0: list of 2 alm arrays
        For spin = 0: single alm array
    nside : int
        HEALPix nside parameter for output maps
    spin : int
        Spin weight (0 for scalar, 2 for polarization, etc.)
    lmax : int, optional
        Maximum multipole l. If None, inferred from alm size
    mmax : int, optional
        Maximum m. Default: lmax
    method : str, optional
        Transform method ('jax', 'jax_healpy', 'jax_cuda'). Default: 'jax'
    healpy_ordering : bool, optional
        If True, input alms are in healpy format. If False, s2fft format. Default: True

    Returns
    -------
    maps : list of array-like or array-like
        For spin != 0: list of 2 output maps
        For spin = 0: single output map

    Notes
    -----
    For polarization (spin=2), input should be [E_lm, B_lm] and output will be [Q, U].

    Examples
    --------
    Compute Q and U maps from E and B modes:

    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> nside = 32
    >>> E_lm = jnp.zeros((65, 129), dtype=complex)  # Example alms
    >>> B_lm = jnp.zeros((65, 129), dtype=complex)
    >>> Q_map, U_map = jhp.alm2map_spin([E_lm, B_lm], nside, spin=2, healpy_ordering=False)
    """
    if spin != 0:
        # For spin != 0, expect list of 2 alms
        if not isinstance(alms, (list, tuple)) or len(alms) != 2:
            raise ValueError(f'For spin={spin}, alms must be a list/tuple of 2 arrays')
        alms = [jnp.asarray(a) for a in alms]

        # Infer lmax from shape
        if lmax is None:
            if healpy_ordering:
                # For healpy format: nalm = mmax*(2*lmax+1-mmax)/2 + lmax+1
                # Simplest case: mmax=lmax => nalm = lmax*(lmax+2)/2 + 1
                # Solve: lmax = (-1 + sqrt(1 + 8*(nalm-1))) / 2
                nalm = alms[0].shape[0]
                lmax = int((-1 + jnp.sqrt(1 + 8 * (nalm - 1))) / 2)
            else:
                # For s2fft format: shape is (L, 2L-1)
                lmax = alms[0].shape[0] - 1
    else:
        alms = jnp.asarray(alms)
        if lmax is None:
            if healpy_ordering:
                nalm = alms.shape[0]
                lmax = int((-1 + jnp.sqrt(1 + 8 * (nalm - 1))) / 2)
            else:
                lmax = alms.shape[0] - 1

    # Call core function
    maps = _alm2map_core(
        alms=alms, nside=nside, lmax=lmax, mmax=mmax, method=method, spin=spin, healpy_ordering=healpy_ordering
    )

    return maps
