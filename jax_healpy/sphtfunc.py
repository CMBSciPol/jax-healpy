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

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import sqrtm
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
    'precompute_polarization_harmonic_transforms',
    'precompute_temperature_harmonic_transforms',
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
def precompute_temperature_harmonic_transforms(
    nside: int, lmax: int = None, sampling: str = 'healpix', pix2harm: bool = False
) -> list:
    """Pre-compute recursion coefficients for s2fft temperature transforms (spin=0).

    Only relevant when using the 'jax' method with s2fft. Pre-computing these
    coefficients can significantly speed up repeated calls to map2alm/alm2map.

    .. note::
        These coefficients are **not yet consumed** by :func:`map2alm` / :func:`alm2map`
        (those functions do not currently accept a ``precomps`` argument and recompute
        the coefficients internally). This helper is provided for direct use with s2fft
        and is reserved for a future ``precomps`` plumbing through the transforms.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter for the maps
    lmax : int, optional
        Maximum multipole moment. If None, uses 3*nside - 1
    sampling : str, optional
        Sampling scheme. Default: 'healpix'
    pix2harm : bool, optional
        If True, compute coefficients for forward transform (map2alm).
        If False, compute for inverse transform (alm2map). Default: False

    Returns
    -------
    list
        Pre-computed recursion coefficients for spin=0 transforms

    Examples
    --------
    >>> import jax_healpy as jhp
    >>> # Precompute for inverse transforms at nside=128
    >>> precomps = jhp.precompute_temperature_harmonic_transforms(128, lmax=383)
    >>> # Reserved for future use; alm2map/map2alm do not accept ``precomps`` yet.
    """
    if lmax is None:
        L = 3 * nside
    else:
        L = lmax + 1

    return generate_precomputes_jax(L, spin=0, sampling=sampling, nside=nside, forward=pix2harm)


@requires_s2fft
def precompute_polarization_harmonic_transforms(
    nside: int, lmax: int = None, sampling: str = 'healpix', pix2harm: bool = False
) -> tuple:
    """Pre-compute recursion coefficients for s2fft polarization transforms (spin=±2).

    Only relevant when using the 'jax' method with s2fft. Pre-computing these
    coefficients can significantly speed up repeated calls to polarization transforms.

    .. note::
        These coefficients are **not yet consumed** by the polarized :func:`map2alm` /
        :func:`alm2map` (those functions do not currently accept a ``precomps`` argument
        and recompute the coefficients internally). This helper is provided for direct use
        with s2fft and is reserved for a future ``precomps`` plumbing through the transforms.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter for the maps
    lmax : int, optional
        Maximum multipole moment. If None, uses 3*nside - 1
    sampling : str, optional
        Sampling scheme. Default: 'healpix'
    pix2harm : bool, optional
        If True, compute coefficients for forward transform (map2alm).
        If False, compute for inverse transform (alm2map). Default: False

    Returns
    -------
    tuple
        Tuple of (precomps_spin2, precomps_spinm2) containing pre-computed
        recursion coefficients for spin=+2 and spin=-2 transforms

    Examples
    --------
    >>> import jax_healpy as jhp
    >>> # Precompute for polarization transforms at nside=128
    >>> precomps_p2, precomps_m2 = jhp.precompute_polarization_harmonic_transforms(128, lmax=383)
    >>> # Reserved for future use; alm2map/map2alm do not accept ``precomps`` yet.
    """
    if lmax is None:
        L = 3 * nside
    else:
        L = lmax + 1

    precomps_plus2 = generate_precomputes_jax(L, spin=2, sampling=sampling, nside=nside, forward=pix2harm)
    precomps_minus2 = generate_precomputes_jax(L, spin=-2, sampling=sampling, nside=nside, forward=pix2harm)
    return precomps_plus2, precomps_minus2


def _compute_beam_window(lmax: int, fwhm: float = 0.0, sigma: float | None = None) -> ArrayLike:
    """Return Gaussian beam factors up to lmax."""
    if sigma is None:
        # fwhm=0 -> sigma=0 -> exp(0)=1 (identity beam), so no special-casing is needed
        sigma = fwhm / jnp.sqrt(8.0 * jnp.log(2.0))

    ell = jnp.arange(lmax + 1)
    # When sigma=0, exp returns 1.0 (no smoothing)
    return jnp.exp(-ell * (ell + 1) * sigma**2 / 2.0)


def _lmax_from_nalm(nalm: int) -> int:
    """Invert ``nalm = (lmax+1)*(lmax+2)/2`` for the healpy 1D alm layout (mmax == lmax)."""
    return int((-1 + np.sqrt(1 + 8 * (nalm - 1))) / 2)


def _resolve_lmax(nside: int, lmax: int | None) -> int:
    """Resolve ``lmax`` (default ``3*nside - 1``) and enforce the s2fft ``lmax >= 2*nside - 1`` floor."""
    target_lmax = 3 * nside - 1 if lmax is None else lmax
    if target_lmax < 2 * nside - 1:
        raise NotImplementedError('Transform requires lmax >= 2*nside - 1 for s2fft transforms.')
    return target_lmax


def get_valid_mask_alms(lmax: int, mmax_positive: int | None = None, mmax_negative: int | None = None) -> ArrayLike:
    """Boolean ``(L, M)`` mask selecting valid alm entries (``|m| <= ell``) over an m-range.

    Defaults span the full s2fft 2D m-axis (``m`` in ``[-lmax, lmax]``); pass
    ``mmax_positive=0`` for the negative-m half (the ``valid_left`` case).
    """
    L = lmax + 1
    if mmax_positive is None:
        mmax_positive = L  # exclusive upper bound -> m up to lmax
    if mmax_negative is None:
        mmax_negative = -L + 1  # inclusive lower bound -> m down to -lmax
    m_vals = jnp.arange(mmax_negative, mmax_positive)
    ell_vals = jnp.arange(L)
    ell_grid, m_grid = jnp.meshgrid(ell_vals, m_vals, indexing='ij')
    return jnp.abs(m_grid) <= ell_grid


def _compute_cl(alms: ArrayLike, L: int, alms2: ArrayLike | None = None) -> ArrayLike:
    """Compute C_l from one or two alm grids in s2fft layout."""
    ell_vals = jnp.arange(L)
    # valid_mask restricts the sum to |m| <= ell: a no-op for well-formed alms, and for malformed
    # ones it still returns the correct C_l (defined over |m| <= ell) rather than leaking power. We
    # deliberately do not raise inside this jitted reduction (a host callback would sync every call).
    valid_mask = get_valid_mask_alms(L - 1)

    alm_prod = jnp.abs(alms) ** 2 if alms2 is None else alms * jnp.conj(alms2)
    alm_prod_masked = alm_prod * valid_mask
    # For ell=0 only m=0 is valid and the divisor is 1, so the masked sum already gives C_0.
    cl = alm_prod_masked.sum(axis=1) / (2 * ell_vals + 1)
    # Cross-spectra should be real-valued
    if alms2 is not None:
        cl = jnp.real(cl)
    return cl


def _generate_random_alm(cl: ArrayLike, lmax: int, prng_key: PRNGKeyArray) -> ArrayLike:
    """Draw random alm with healpy reality convention (mmax == lmax)."""
    L = lmax + 1
    cl = jnp.asarray(cl)
    # Real working dtype follows the input spectrum (float64 only when x64 is enabled), so we
    # don't request an unavailable float64 and trigger JAX's silent-truncation warning.
    real_dtype = jnp.result_type(cl, jnp.float32)
    if cl.shape[0] < L:
        cl = jnp.pad(cl, (0, L - cl.shape[0]), constant_values=0)
    elif cl.shape[0] > L:
        cl = cl[:L]

    key_real, key_imag = jax.random.split(prng_key)
    rand_real = jax.random.normal(key_real, shape=(L, 2 * L - 1), dtype=real_dtype)
    rand_imag = jax.random.normal(key_imag, shape=(L, 2 * L - 1), dtype=real_dtype)

    # m_grid is reused below for the m==0 (real-variance) case. valid_mask here is load-bearing:
    # it zeroes the out-of-triangle entries so the random draw is a well-formed alm (|m| <= ell).
    _, m_grid = jnp.meshgrid(jnp.arange(L), jnp.arange(-L + 1, L), indexing='ij')
    valid_mask = get_valid_mask_alms(lmax)

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

    valid_left = get_valid_mask_alms(lmax, mmax_positive=0)
    left_half = conj_right_flipped * valid_left

    return jnp.concatenate([left_half, alms[:, L - 1 : L], right_half], axis=1)


def _getn(k: int) -> int:
    """Return n such that n*(n+1)/2 == k, or -1 if k is not triangular.

    Mirrors ``healpy.sphtfunc._sphtools._getn``; used to map a flat list of
    n(n+1)/2 cross-spectra to the number of underlying fields n.
    """
    n = int(round((np.sqrt(8 * k + 1) - 1) / 2))
    return n if n * (n + 1) // 2 == k else -1


def _as_spectra_list(cls):
    """Classify ``cls`` as scalar or multi-spectra input.

    Returns ``(is_multi, payload)``. ``payload`` is a 1D array for the scalar
    case, or a Python list of per-field spectra (1D arrays / ``None``) for the
    multi-spectra (polarization) case. Mirrors healpy's ``is_seq_of_seq`` rule:
    a sequence whose elements are themselves arrays is treated as multi-spectra.
    """
    if isinstance(cls, (list, tuple)):
        is_multi = any(c is not None and jnp.ndim(c) >= 1 for c in cls)
        if is_multi:
            return True, list(cls)
        return False, jnp.asarray(cls)
    arr = jnp.asarray(cls)
    if arr.ndim >= 2:
        return True, [arr[i] for i in range(arr.shape[0])]
    return False, arr


def _new_to_old_spectra_order(cls_new_order: list) -> list:
    """Reorder cls from new (by diagonal) to old (by row) order.

    Pure-Python reorder mirroring ``healpy.sphtfunc.new_to_old_spectra_order``.
    Example (n=3): ``TT, EE, BB, TE, EB, TB`` -> ``TT, TE, TB, EE, EB, BB``.
    """
    n = _getn(len(cls_new_order))
    if n < 0:
        raise ValueError('Input must be a list of n(n+1)/2 arrays')
    cls_old_order = []
    for i in range(n):
        for j in range(i, n):
            p = j - i
            q = i
            idx_new = p * (2 * n + 1 - p) // 2 + q
            cls_old_order.append(cls_new_order[idx_new])
    return cls_old_order


@requires_s2fft
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
        raise NotImplementedError('Specifying mmax != lmax is not implemented.')

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

        return jnp.real(f_complex)


@requires_s2fft
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
        raise NotImplementedError('Specifying mmax != lmax is not implemented.')

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


@jax.jit(
    static_argnames=[
        'nside',
        'lmax',
        'mmax',
        'pixwin',
        'fwhm',
        'sigma',
        'pol',
        'healpy_ordering',
        'method',
    ],
)
def alm2map(
    alms: ArrayLike,
    nside: int,
    lmax=None,
    mmax=None,
    pixwin=False,
    fwhm=0.0,
    sigma=None,
    pol: bool = True,
    healpy_ordering: bool = False,
    method: str = 'jax',
) -> ArrayLike:
    """Computes a Healpix map given the alm.

    The alm are given as a complex array. You can specify lmax
    and mmax, or they will be computed from array size (assuming
    lmax==mmax).

    Parameters
    ----------
    alms : complex, array or sequence of arrays
      A complex array of spherical harmonic coefficients (size
      ``mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1`` in healpy ordering), or,
      when ``pol=True``, a stack of ``n`` such arrays with ``n in {1, 2, 3}``
      (T / E,B / T,E,B — see ``pol``).
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
      If True, the leading axis selects polarization components and the output
      is the corresponding stack of maps:

      - ``1`` -> ``[alm_T]`` -> T map
      - ``2`` -> ``[alm_E, alm_B]`` -> ``(2, npix)`` Q, U maps (spin-2 only, no
        temperature). **This is a deliberate change of behavior vs healpy**,
        whose polarized ``alm2map`` only accepts the 3-component TEB input.
      - ``3`` -> ``[alm_T, alm_E, alm_B]`` -> ``(3, npix)`` T, Q, U maps

      Input shape is ``(n, nalm)`` for ``healpy_ordering=True`` or
      ``(n, L, 2L-1)`` for ``healpy_ordering=False``. A single (non-stacked) alm is
      always treated as a scalar field regardless of ``pol``. Gaussian smoothing
      (``fwhm``/``sigma``) is supported and applies the spin-2 beam correction to
      E/B; ``pixwin`` is not supported. Default: True (matches healpy).
    healpy ordering : bool, optional
      True if the input alms follow the healpy ordering. By default, the s2fft
      ordering is assumed.
    method : str, optional
      Transform backend ('jax', 'jax_healpy', 'jax_cuda'). Default: 'jax'.

    Returns
    -------
    maps : array
      A single Healpix map in RING scheme at ``nside`` (shape ``(npix,)``), or a
      stacked ``(n, npix)`` array of T/Q/U maps for polarized input.

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

    if pixwin:
        raise NotImplementedError('Pixel-window smoothing is not implemented; set pixwin=False.')

    # Polarized branch: input is shape (n, ...) with n in {1, 2, 3}.
    #   n == 1 -> T        -> T map                 (spin 0)
    #   n == 2 -> E, B     -> (Q, U) maps           (spin 2; not possible in healpy)
    #   n == 3 -> T, E, B  -> (T, Q, U) maps
    if pol and alms.ndim == expected_ndim + 1:
        n = alms.shape[0]
        if n not in (1, 2, 3):
            raise ValueError(f'pol=True requires alms with shape (n, ...), n in (1, 2, 3); got shape {alms.shape}.')

        target_L = 3 * nside if lmax is None else lmax + 1
        target_lmax = target_L - 1
        if mmax is not None and mmax != target_lmax:
            raise NotImplementedError('Specifying mmax != lmax is not implemented.')

        # Optional Gaussian smoothing with the spin-2 beam correction on E/B.
        if fwhm != 0 or sigma is not None:
            alms = smoothalm(alms, fwhm=fwhm, sigma=sigma, pol=True, mmax=mmax, healpy_ordering=healpy_ordering)

        if n == 2:
            # E, B -> Q, U (spin-2 only, no temperature)
            Q_map, U_map = _alm2map_core(
                alms=[alms[0], alms[1]],
                nside=nside,
                lmax=target_lmax,
                mmax=mmax,
                method=method,
                spin=2,
                healpy_ordering=healpy_ordering,
            )
            return jnp.stack([Q_map, U_map], axis=0)

        # leading alm is temperature (n == 1 -> T; n == 3 -> T, E, B)
        T_map = _alm2map_core(
            alms=alms[0],
            nside=nside,
            lmax=target_lmax,
            mmax=mmax,
            method=method,
            spin=0,
            healpy_ordering=healpy_ordering,
        )
        if n == 1:
            return T_map
        Q_map, U_map = _alm2map_core(
            alms=[alms[1], alms[2]],
            nside=nside,
            lmax=target_lmax,
            mmax=mmax,
            method=method,
            spin=2,
            healpy_ordering=healpy_ordering,
        )
        return jnp.stack([T_map, Q_map, U_map], axis=0)

    # Handle batched input
    if alms.ndim == expected_ndim + 1 + pol:
        return jax.vmap(alm2map, in_axes=(0,) + 9 * (None,))(
            alms,
            nside,
            lmax,
            mmax,
            pixwin,
            fwhm,
            sigma,
            pol,
            healpy_ordering,
            method,
        )

    if alms.ndim > expected_ndim:
        # only happens if pol=True
        raise NotImplementedError('TEB alms are not implemented.')

    if lmax is None:
        L = 3 * nside
    else:
        L = lmax + 1

    if mmax is not None:
        if lmax is None or mmax != lmax:
            raise NotImplementedError('Specifying mmax != lmax (or without lmax) is not implemented.')

    # Apply smoothing if requested (scalar spin-0 beam; mirrors the polarized branch above).
    if fwhm != 0 or sigma is not None:
        alms = smoothalm(alms, fwhm=fwhm, sigma=sigma, pol=False, mmax=mmax, healpy_ordering=healpy_ordering)

    # Call core function
    f = _alm2map_core(
        alms=alms, nside=nside, lmax=L - 1, mmax=mmax, method=method, spin=0, healpy_ordering=healpy_ordering
    )

    return f


@jax.jit(
    static_argnames=[
        'lmax',
        'mmax',
        'iter',
        'pol',
        'use_weights',
        'datapath',
        'gal_cut',
        'use_pixel_weights',
        'healpy_ordering',
        'method',
    ],
)
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
    healpy_ordering: bool = False,
    method: str = 'jax',
) -> ArrayLike:
    """Computes the alm of a Healpix map. The input maps must all be
    in ring ordering.

    For recommendations about how to set ``lmax`` and ``iter`` see the
    `Anafast documentation <https://healpix.sourceforge.io/html/fac_anafast.htm>`_.
    Pixels are weighted with the uniform value ``4*pi/n_pix`` before the transform.

    .. note::
        Weighting options (``use_weights``, ``use_pixel_weights``, ``datapath``) and
        ``gal_cut`` are not implemented and raise :class:`NotImplementedError` if set;
        only the uniform weighting is currently available.

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
      If True, the leading axis selects polarization components and the output is the
      corresponding stack of alms (accepts 1, 2, or 3 maps):

      - ``1`` -> ``[I]``        -> ``alm_T``                  (spin 0)
      - ``2`` -> ``[Q, U]``     -> ``(alm_E, alm_B)``         (spin-2 only, no temperature;
        a deliberate extension vs healpy, whose polarized ``map2alm`` requires I, Q, U)
      - ``3`` -> ``[I, Q, U]``  -> ``(alm_T, alm_E, alm_B)``

      A single 1-D map is always transformed as a scalar field regardless of ``pol``.
      If False, apply a spin-0 harmonic transform to each map (input can be any number of
      maps). Default: True.
    use_weights: bool, scalar, optional
      If True, use the ring weighting. Default: False.
    datapath : None or str, optional
      If given, the directory where to find the pixel weights.
      See in the docstring above details on how to set it up.
    gal_cut : float [degrees]
      pixels at latitude in [-gal_cut;+gal_cut] are not taken into account
    use_pixel_weights: bool, optional
      If True, use pixel by pixel weighting, healpy will automatically download the weights, if needed
    healpy_ordering : bool, optional
      By default, we follow the s2fft ordering for the alms. To use healpy
      ordering, set it to True.
    method : str, optional
      Transform backend ('jax', 'jax_healpy', 'jax_cuda'). Default: 'jax'.

    Returns
    -------
    alms : array
      A single alm array, or a stacked ``(n, ...)`` array of alm for polarized
      input (e.g. ``[almT, almE, almB]`` for 3 maps), matching healpy's array
      return -- not a tuple.
    """
    if use_weights:
        raise NotImplementedError('Specifying use_weights is not implemented.')
    if datapath is not None:
        raise NotImplementedError('Specifying datapath is not implemented.')
    if gal_cut != 0:
        raise NotImplementedError('Specifying gal_cut is not implemented.')
    if use_pixel_weights:
        raise NotImplementedError('Specifying use_pixel_weights is not implemented.')

    maps = jnp.asarray(maps)
    if maps.ndim == 0:
        raise ValueError('The input map must have at least one dimension.')
    if maps.ndim > 2:
        raise ValueError('The input map has too many dimensions.')

    # Polarized branch: input is shape (n, npix) with n in {1, 2, 3}.
    #   n == 1 -> I        -> alm_T                 (spin 0)
    #   n == 2 -> Q, U     -> (alm_E, alm_B)        (spin 2; not possible in healpy)
    #   n == 3 -> I, Q, U  -> (alm_T, alm_E, alm_B)
    if pol and maps.ndim == 2:
        n = maps.shape[0]
        if n not in (1, 2, 3):
            raise ValueError(
                f'pol=True requires 1 (I), 2 (Q, U), or 3 (I, Q, U) maps of shape (n, npix); got shape {maps.shape}.'
            )
        nside = npix2nside(maps.shape[-1])
        target_lmax = _resolve_lmax(nside, lmax)
        target_L = target_lmax + 1
        if mmax is not None and mmax != target_lmax:
            raise NotImplementedError('Specifying mmax != lmax is not implemented.')

        if n == 2:
            # Q, U -> E, B (spin-2 only, no temperature)
            alm_E, alm_B = _map2alm_core(
                maps=[maps[0], maps[1]], lmax=target_lmax, mmax=mmax, iter=iter, method=method, spin=2
            )
            out = [alm_E, alm_B]
        else:
            # leading map is temperature (n == 1 -> I; n == 3 -> I, Q, U)
            alm_T = _map2alm_core(maps=maps[0], lmax=target_lmax, mmax=mmax, iter=iter, method=method, spin=0)
            out = [alm_T]
            if n == 3:
                alm_E, alm_B = _map2alm_core(
                    maps=[maps[1], maps[2]], lmax=target_lmax, mmax=mmax, iter=iter, method=method, spin=2
                )
                out += [alm_E, alm_B]

        if healpy_ordering:
            out = [flm_2d_to_hp_fast(a, target_L) for a in out]

        # A single I map collapses to a bare temperature alm (scalar semantics);
        # otherwise return a stacked (n, ...) array, matching healpy (not a tuple).
        return out[0] if n == 1 else jnp.stack(out, axis=0)

    # Handle batched scalar input (pol=False, shape (n, npix))
    if maps.ndim > 1:
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
            healpy_ordering,
            method,
        )

    nside = npix2nside(maps.shape[-1])
    target_lmax = _resolve_lmax(nside, lmax)
    target_L = target_lmax + 1

    if mmax is not None and mmax != target_lmax:
        raise NotImplementedError('Specifying mmax != lmax is not implemented.')

    # Call core function
    flm = _map2alm_core(
        maps=maps,
        lmax=target_lmax,
        mmax=mmax,
        iter=iter,
        method=method,
        spin=0,
    )

    if healpy_ordering:
        flm = flm_2d_to_hp_fast(flm, target_L)

    return flm


@jax.jit(static_argnames=['mmax', 'healpy_ordering'])
@requires_s2fft
def almxfl(alm: ArrayLike, fl: ArrayLike, mmax: int | None = None, healpy_ordering: bool = False) -> ArrayLike:
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
    healpy_ordering : bool, optional
        If True, alm uses healpy 1D format. If False, uses s2fft 2D format. Default: False

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
        fl_2d = fl[..., None]
        return alm * fl_2d


@jax.jit(static_argnames=['lmax', 'pol'])
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
        If True, returns polarization beam components as an array of shape
        ``(lmax+1, 4)`` with columns [temperature, E (grad), B (curl), TE],
        following healpy (Challinor et al. 2000). Default: False

    Returns
    -------
    beam : array-like
        Beam window function. Shape ``(lmax+1,)`` for the temperature beam, or
        ``(lmax+1, 4)`` [T, E, B, TE] if ``pol=True``.
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
    g = _compute_beam_window(lmax, fwhm=fwhm, sigma=None)
    if not pol:
        return g

    # Polarization beam assuming a perfectly co-polarized beam (Challinor et al. 2000).
    # Columns: [temperature, E (grad), B (curl), TE], with factors exp([0, 2σ², 2σ², σ²]).
    sigma2 = (fwhm / jnp.sqrt(8.0 * jnp.log(2.0))) ** 2
    pol_factor = jnp.exp(jnp.stack([jnp.zeros_like(sigma2), 2.0 * sigma2, 2.0 * sigma2, sigma2]))
    return g[:, None] * pol_factor


@jax.jit(static_argnames=['lmax', 'mmax', 'lmax_out', 'nspec', 'healpy_ordering'])
@requires_s2fft
def alm2cl(
    alms1: ArrayLike | list[ArrayLike],
    alms2: ArrayLike | list[ArrayLike] | None = None,
    lmax: int | None = None,
    mmax: int | None = None,
    lmax_out: int | None = None,
    nspec: int | None = None,
    healpy_ordering: bool = False,
) -> ArrayLike:
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
        Maximum l of the returned spectra. If None, uses lmax from input.
        Otherwise each returned spectrum is truncated to lmax_out+1 multipoles.
    nspec : int, optional
        Number of leading entries to keep, matching healpy: ``cl[:nspec]``.
        For multiple alms this selects the first nspec spectra; for a single
        alm it slices the multipole axis. If None, returns all.
    healpy_ordering : bool, optional
        If True, input alms use healpy 1D format. If False, uses s2fft 2D format.
        Default: False

    Returns
    -------
    cl : array-like
        A single 1-D array of shape (lmax+1,) when there is only one spectrum
        (single alm, or a single retained entry). For multiple alms: a stacked
        ``(n(n+1)/2, lmax+1)`` array of spectra in diagonal order
        (11, 22, 33, 12, 23, 13, ...), matching healpy's ndarray return.

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

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> lmax = 16
    >>> cl_in = 1.0 / (jnp.arange(lmax + 1) + 1.0)
    >>> alm1 = jhp.synalm(jax.random.PRNGKey(0), cl_in, lmax=lmax)
    >>> cl = jhp.alm2cl(alm1)

    Compute cross-spectrum between two alms:

    >>> alm2 = jhp.synalm(jax.random.PRNGKey(1), cl_in, lmax=lmax)
    >>> cl_cross = jhp.alm2cl(alm1, alm2)

    Compute all spectra from multiple alms:

    >>> alm3 = jhp.synalm(jax.random.PRNGKey(2), cl_in, lmax=lmax)
    >>> cls = jhp.alm2cl([alm1, alm2, alm3])  # shape (6, lmax+1): cl11, cl22, cl33, cl12, cl23, cl13
    """
    # Handle multiple alms case
    if isinstance(alms1, (list, tuple)):
        alms1_list = [jnp.asarray(a) for a in alms1]
        n_alms = len(alms1_list)

        # Infer lmax from first alm if not provided
        if lmax is None:
            if healpy_ordering:
                lmax = _lmax_from_nalm(alms1_list[0].shape[0])
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

        # Truncate multipoles to lmax_out if requested (matches healpy)
        if lmax_out is not None:
            spectra = [s[: lmax_out + 1] for s in spectra]

        # Apply nspec if requested (selects the first nspec spectra)
        if nspec is not None:
            spectra = spectra[:nspec]

        return jnp.stack(spectra, axis=0) if len(spectra) > 1 else spectra[0]

    # Single alm case
    alms1 = jnp.asarray(alms1)

    # Infer lmax if not provided
    if lmax is None:
        if healpy_ordering:
            lmax = _lmax_from_nalm(alms1.shape[0])
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

    # Truncate multipoles to lmax_out if requested (matches healpy)
    if lmax_out is not None:
        cl = cl[: lmax_out + 1]

    # Apply nspec if requested (healpy slices the multipole axis for a single alm)
    if nspec is not None:
        cl = cl[:nspec]

    return cl


@jax.jit(
    static_argnames=[
        'lmax',
        'mmax',
        'iter',
        'alm',
        'nspec',
        'pol',
        'use_weights',
        'datapath',
        'gal_cut',
        'use_pixel_weights',
        'method',
    ],
)
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
) -> ArrayLike | tuple[ArrayLike, ...]:
    """Compute the angular power spectrum from HEALPix map(s).

    #TODO add option expanded_cross_cl which returns also ET BE and BT

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
        If True, input is a stack of maps. With ``(3, npix)`` I, Q, U input it
        returns 6 spectra ``(TT, EE, BB, TE, EB, TB)`` of shape ``(6, lmax+1)``.
        As a **deviation from healpy** (which requires I, Q, U), ``(2, npix)`` Q, U
        input is also accepted and returns the 3 spin-2 spectra
        ``(EE, BB, EB)`` of shape ``(3, lmax+1)``. Default: True
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
        Angular power spectrum. Shape (lmax+1,) for scalar input,
        (6, lmax+1) for pol=True with I, Q, U in order (TT, EE, BB, TE, EB, TB),
        or (3, lmax+1) for pol=True with Q, U only in order (EE, BB, EB).
    alm1 : array-like, optional
        Alm coefficients of map1 in s2fft format (only if alm=True).
        Shape (L, 2L-1) for scalar, (3, L, 2L-1) for pol=True (T, E, B stacked).
    alm2 : array-like, optional
        Alm coefficients of map2 in s2fft format (only if alm=True and map2 is provided)

    Notes
    -----
    The power spectrum is computed as:
        C_l = (1 / (2*l + 1)) * sum_m |a_lm|^2  (auto-spectrum)
        C_l = (1 / (2*l + 1)) * sum_m a_lm * conj(a'_lm)  (cross-spectrum)

    This function matches the healpy.sphtfunc.anafast API. Several healpy parameters
    related to map weighting (use_weights, datapath, gal_cut, use_pixel_weights) are
    not yet implemented.

    Examples
    --------
    Compute auto-spectrum of a map:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> nside = 8
    >>> lmax = 2 * nside - 1
    >>> cl_in = 1.0 / (jnp.arange(lmax + 1) + 1.0)
    >>> map_data = jhp.synfast(jax.random.PRNGKey(0), cl_in, nside, lmax=lmax, pol=False)
    >>> cl_estimated = jhp.anafast(map_data, lmax=lmax, pol=False)
    """
    # Validate unsupported parameters
    if use_weights:
        raise NotImplementedError('use_weights is not implemented')
    if datapath is not None:
        raise NotImplementedError('datapath is not implemented')
    if gal_cut != 0:
        raise NotImplementedError('gal_cut is not implemented')
    if use_pixel_weights:
        raise NotImplementedError('use_pixel_weights is not implemented')

    map1 = jnp.asarray(map1)

    # A 1-D map is always a scalar field; pol only applies to a (n, npix) stack. This
    # matches healpy, which returns the scalar auto-spectrum for a single map even with
    # pol=True (and mirrors the 1-D fall-through in map2alm).
    if pol and map1.ndim == 2:
        if map1.shape[0] not in (2, 3):
            raise ValueError(
                f'pol=True requires map1 with shape (3, npix) for I, Q, U or (2, npix) for Q, U; got {map1.shape}.'
            )
        nside = npix2nside(map1.shape[-1])
        if lmax is None:
            lmax = 3 * nside - 1
        L = lmax + 1
        ncomp = map1.shape[0]

        # alm1 is a stacked (ncomp, L, 2L-1) array: (T, E, B) for 3 maps or (E, B) for 2.
        alm1 = map2alm(map1, lmax=lmax, mmax=mmax, iter=iter, pol=True, healpy_ordering=False, method=method)

        if map2 is None:
            if ncomp == 3:
                # 6 spectra in healpy order: (TT, EE, BB, TE, EB, TB)
                alm_T1, alm_E1, alm_B1 = alm1
                spectra = (
                    _compute_cl(alm_T1, L),
                    _compute_cl(alm_E1, L),
                    _compute_cl(alm_B1, L),
                    _compute_cl(alm_T1, L, alm_E1),
                    _compute_cl(alm_E1, L, alm_B1),
                    _compute_cl(alm_T1, L, alm_B1),
                )
            else:
                # Q, U only -> 3 spectra (EE, BB, EB). Deviation from healpy (no temperature).
                alm_E1, alm_B1 = alm1
                spectra = (_compute_cl(alm_E1, L), _compute_cl(alm_B1, L), _compute_cl(alm_E1, L, alm_B1))
            cl = jnp.stack(spectra, axis=0)
            if nspec is not None:
                cl = cl[:nspec]
            if alm:
                return cl, alm1
            return cl

        map2 = jnp.asarray(map2)
        if map2.ndim != 2 or map2.shape[0] != ncomp:
            raise ValueError(
                f'pol=True cross-spectrum requires map2 with the same shape as map1 ({ncomp}, npix); got {map2.shape}.'
            )
        alm2 = map2alm(map2, lmax=lmax, mmax=mmax, iter=iter, pol=True, healpy_ordering=False, method=method)
        if ncomp == 3:
            alm_T1, alm_E1, alm_B1 = alm1
            alm_T2, alm_E2, alm_B2 = alm2
            spectra = (
                _compute_cl(alm_T1, L, alm_T2),
                _compute_cl(alm_E1, L, alm_E2),
                _compute_cl(alm_B1, L, alm_B2),
                _compute_cl(alm_T1, L, alm_E2),
                _compute_cl(alm_E1, L, alm_B2),
                _compute_cl(alm_T1, L, alm_B2),
            )
        else:
            alm_E1, alm_B1 = alm1
            alm_E2, alm_B2 = alm2
            spectra = (_compute_cl(alm_E1, L, alm_E2), _compute_cl(alm_B1, L, alm_B2), _compute_cl(alm_E1, L, alm_B2))
        cl = jnp.stack(spectra, axis=0)
        if nspec is not None:
            cl = cl[:nspec]
        if alm:
            return cl, alm1, alm2
        return cl

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


@jax.jit(static_argnames=['fwhm', 'sigma', 'mmax', 'pol', 'healpy_ordering'])
def smoothalm(
    alms: ArrayLike,
    fwhm: float = 0.0,
    sigma: float | None = None,
    beam_window: ArrayLike | None = None,
    pol: bool = True,
    mmax: int | None = None,
    healpy_ordering: bool = False,
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
        If True and ``alms`` is a stack of ``n`` components (n in {1, 2, 3}), the
        temperature component uses the spin-0 beam and E/B use the spin-2 beam
        (an extra ``exp(2σ²)`` factor), matching healpy. 1 -> [T], 2 -> [E, B],
        3 -> [T, E, B]. A single (non-stacked) alm is always smoothed as a scalar
        field regardless of ``pol`` (also matching healpy). Default: True
    mmax : int, optional
        Maximum m value for alm coefficients. Default: lmax (inferred from alm size)
    healpy_ordering : bool, optional
        If True, input/output use healpy 1D format. If False, s2fft 2D format. Default: False

    Returns
    -------
    alms_smoothed : array-like
        Smoothed spherical harmonic coefficients

    Examples
    --------
    >>> import jax
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> lmax = 16
    >>> cl_in = 1.0 / (jnp.arange(lmax + 1) + 1.0)
    >>> alm = jhp.synalm(jax.random.PRNGKey(0), cl_in, lmax=lmax)
    >>> fwhm_rad = float(np.radians(5.0 / 60.0))  # 5 arcmin (static, must be a float)
    >>> alm_smooth = jhp.smoothalm(alm, fwhm=fwhm_rad, pol=False)
    """
    alms = jnp.asarray(alms)

    def _lmax_of(component: ArrayLike) -> int:
        if healpy_ordering:
            return _lmax_from_nalm(component.shape[0])
        return component.shape[0] - 1

    # A single alm (no stacked component axis) is always smoothed as a scalar field,
    # matching healpy; pol only applies to a stacked (n, ...) input.
    expected_ndim = 1 if healpy_ordering else 2

    # Polarized branch: per-component beam (temperature spin 0, E/B spin 2).
    if pol and alms.ndim == expected_ndim + 1:
        n = alms.shape[0]
        # Per-component spin weights: 1 -> [T]; 2 -> [E, B]; 3 -> [T, E, B].
        if n == 1:
            spins = (0,)
        elif n == 2:
            spins = (2, 2)
        elif n == 3:
            spins = (0, 2, 2)
        else:
            raise ValueError('smoothalm pol supports 1 (T), 2 (E, B), or 3 (T, E, B) components')
        lmax = _lmax_of(alms[0])
        ell = jnp.arange(lmax + 1)
        if sigma is None:
            sigma_val = fwhm / np.sqrt(8.0 * np.log(2.0))
        else:
            sigma_val = sigma
        smoothed = []
        for i in range(n):
            if beam_window is not None:
                fact = jnp.asarray(beam_window)
            else:
                # healpy: fact = exp(-0.5 * (l(l+1) - s^2) * sigma^2), s = 2 for E/B.
                fact = jnp.exp(-0.5 * (ell * (ell + 1) - spins[i] ** 2) * sigma_val**2)
            smoothed.append(almxfl(alms[i], fact, mmax=mmax, healpy_ordering=healpy_ordering))
        return jnp.stack(smoothed, axis=0)

    # Scalar branch.
    lmax = _lmax_of(alms)
    if beam_window is not None:
        bl = jnp.asarray(beam_window)
    else:
        bl = _compute_beam_window(lmax, fwhm=fwhm, sigma=sigma)

    # Apply smoothing using almxfl
    return almxfl(alms, bl, mmax=mmax, healpy_ordering=healpy_ordering)


@jax.jit(
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
        'nest',
    ],
)
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
        If True and ``map_in`` is a stack of maps, treat them as polarized:
        1 -> [I], 2 -> [Q, U], 3 -> [I, Q, U]; E/B receive the spin-2 beam.
        A single map (1D input) is always smoothed as a scalar. Default: True
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
    >>> import jax
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> nside = 8
    >>> lmax = 2 * nside - 1
    >>> cl_in = 1.0 / (jnp.arange(lmax + 1) + 1.0)
    >>> map_in = jhp.synfast(jax.random.PRNGKey(0), cl_in, nside, lmax=lmax, pol=False)
    >>> fwhm_rad = float(np.radians(10.0))  # 10 degrees (static, must be a float)
    >>> map_smooth = jhp.smoothing(map_in, fwhm=fwhm_rad, pol=False)
    """
    if nest:
        raise NotImplementedError('nest=True is not implemented; only RING ordering accepted')
    if use_weights:
        raise NotImplementedError('use_weights is not implemented')
    if use_pixel_weights:
        raise NotImplementedError('use_pixel_weights is not implemented')
    if datapath is not None:
        raise NotImplementedError('datapath is not implemented')

    map_in = jnp.asarray(map_in)
    # A single map (1D) is always scalar; pol only applies to a stack of maps.
    pol_active = pol and map_in.ndim == 2

    if pol_active:
        nside = npix2nside(map_in.shape[-1])
        alms = jnp.asarray(map2alm(map_in, lmax=lmax, mmax=mmax, iter=iter, pol=True, healpy_ordering=False))
        if alms.ndim == 2:  # single component (1 map) -> add the component axis
            alms = alms[None, ...]
        alms_smooth = smoothalm(
            alms, fwhm=fwhm, sigma=sigma, beam_window=beam_window, pol=True, mmax=mmax, healpy_ordering=False
        )
        return alm2map(alms_smooth, nside=nside, lmax=lmax, mmax=mmax, pol=True, healpy_ordering=False)

    # Scalar branch (single map).
    alms = map2alm(map_in, lmax=lmax, mmax=mmax, iter=iter, pol=False, healpy_ordering=False)
    alms_smooth = smoothalm(
        alms, fwhm=fwhm, sigma=sigma, beam_window=beam_window, pol=False, mmax=mmax, healpy_ordering=False
    )
    return alm2map(alms_smooth, nside=npix2nside(map_in.shape[-1]), lmax=lmax, mmax=mmax, healpy_ordering=False)


@jax.jit(static_argnames=['lmax', 'mmax', 'new', 'healpy_ordering'])
@requires_s2fft
def synalm(
    prng_key: PRNGKeyArray,
    cls: ArrayLike | tuple[ArrayLike, ...],
    lmax: int | None = None,
    mmax: int | None = None,
    new: bool = False,
    healpy_ordering: bool = False,
) -> ArrayLike:
    """Generate random spherical harmonic coefficients from power spectrum.

    Creates alm coefficients drawn from Gaussian distributions with variances
    determined by the input power spectrum/spectra.

    Parameters
    ----------
    prng_key : PRNGKeyArray
        JAX random number generator key (required first argument)
    cls : array-like or sequence of arrays
        Power spectrum data. Can be:
        - Single 1D array for the scalar (temperature-only) case
        - Sequence of 4 arrays for polarization (ordering set by ``new``)
        - Sequence of n(n+1)/2 arrays for full field correlations
          (entries may be ``None`` to omit a cross-spectrum)
    lmax : int, optional
        Maximum multipole moment. If None, inferred from the longest spectrum.
    mmax : int, optional
        Maximum azimuthal order. Only ``mmax == lmax`` is supported; any other
        value raises NotImplementedError.
    new : bool, optional
        Ordering convention for a sequence of input spectra (matches healpy):
        - True: by diagonal, e.g. ``TT, EE, BB, TE, EB, TB`` (or ``TT, EE, BB, TE``)
        - False (default): by row, e.g. ``TT, TE, TB, EE, EB, BB`` (or ``TT, TE, EE, BB``)
        Ignored for a single input spectrum.
    healpy_ordering : bool, optional
        If True, output uses healpy 1D format. If False, s2fft 2D format. Default: False

    Returns
    -------
    alms : array-like
        Generated spherical harmonic coefficients. A single array for the scalar
        case, or a stacked ``(n, ...)`` array of T, E, B coefficients for the
        polarization case.

    Notes
    -----
    For the scalar case:
    - m=0: a_l0 is real, drawn from N(0, C_l)
    - m>0: a_lm is complex, real and imag parts drawn from N(0, C_l/2)

    For the polarization case the per-multipole field covariance matrix is built
    from the input (cross-)spectra and its matrix square root mixes independent
    unit-variance realizations, reproducing cross-spectra such as TE exactly.

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
    is_multi, payload = _as_spectra_list(cls)

    # Polarization / multi-spectra case: draw correlated alms whose per-multipole
    # field covariance matches the input (cross-)spectra, including cross-spectra
    # (e.g. CMB TE) that a per-field sqrt(C_l) scaling cannot capture.
    if is_multi:
        cls_list = [None if c is None else jnp.asarray(c) for c in payload]
        if lmax is None:
            lmax = max(c.shape[0] for c in cls_list if c is not None) - 1
        if mmax is not None and mmax != lmax:
            raise NotImplementedError('Specifying mmax != lmax is not implemented.')
        L = lmax + 1
        # Real working dtype follows the input spectra (float64 only when x64 is enabled).
        real_dtype = jnp.result_type(*[c for c in cls_list if c is not None], jnp.float32)

        def pad_cl(cl):
            if cl is None:
                return jnp.zeros(L, dtype=real_dtype)
            cl = jnp.asarray(cl)
            if cl.shape[0] < L:
                return jnp.pad(cl, (0, L - cl.shape[0]), constant_values=0)
            return cl[:L]

        # Normalize the spectra into healpy "old" (by-row) order with n(n+1)/2 entries.
        k = len(cls_list)
        if k == 4:
            # healpy: 4 cls always map to n=3 (T, E, B) with EB = TB = 0.
            if new:  # new input order: TT, EE, BB, TE
                row_order = _new_to_old_spectra_order([cls_list[0], cls_list[1], cls_list[2], cls_list[3], None, None])
            else:  # old input order: TT, TE, EE, BB
                row_order = [cls_list[0], cls_list[1], None, cls_list[2], None, cls_list[3]]
            n = 3
        else:
            n = _getn(k)
            if n < 0:
                raise ValueError('cls must contain 1, 4, or n(n+1)/2 spectra')
            row_order = _new_to_old_spectra_order(cls_list) if new else cls_list

        # Build the symmetric per-ell covariance matrix cov[l, i, j].
        cols = [pad_cl(c) for c in row_order]
        rows = [[None] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i, n):
                rows[i][j] = cols[idx]
                rows[j][i] = cols[idx]
                idx += 1
        cov = jnp.stack([jnp.stack(rows[i], axis=-1) for i in range(n)], axis=-2)  # (L, n, n)

        # Per-multipole matrix square root: principal sqrt of a symmetric PSD matrix is
        # symmetric PSD, so R @ R == cov (incl. the singular low-ell blocks). Mix n
        # independent unit-variance draws below.
        # TODO: in case of non-symmetric cls this needs to be done differently
        #       (dropping the imaginary part / assuming a symmetric sqrt no longer holds).
        sqrt_cov = jax.vmap(sqrtm)(cov)  # (L, n, n)
        keys = jax.random.split(prng_key, n)
        ones = jnp.ones(L, dtype=real_dtype)
        unit = jnp.stack([_generate_random_alm(ones, lmax, keys[i]) for i in range(n)], axis=0)  # (n, L, 2L-1)
        alms = jnp.einsum('lij,jlm->ilm', sqrt_cov, unit)  # (n, L, 2L-1)

        if healpy_ordering:
            alms = jnp.stack([flm_2d_to_hp_fast(alms[i], L) for i in range(n)], axis=0)
        return alms

    # Scalar case - use existing _generate_random_alm
    cls = jnp.asarray(payload)
    if lmax is None:
        lmax = cls.shape[0] - 1
    if mmax is not None and mmax != lmax:
        raise NotImplementedError('Specifying mmax != lmax is not implemented.')

    alms = _generate_random_alm(cls, lmax, prng_key)

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


@jax.jit(
    static_argnames=['nside', 'lmax', 'mmax', 'alm', 'pol', 'pixwin', 'fwhm', 'sigma', 'new', 'method'],
)
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
    method: str = 'jax',
) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
    """Generate a random HEALPix map from a power spectrum.

    #TODO add option to expect cl in 6 for polarized or 4 for EE BB EB or (10 maybe? or 9) for expanded cross

    Parameters
    ----------
    prng_key : PRNGKeyArray
        JAX random number generator key (required first argument). Build it from
        an integer seed with ``jax.random.PRNGKey(seed)``.
    cls : array-like or sequence of arrays
        Input power spectrum/spectra. A single 1D array for the scalar case, or a
        sequence of 4 or n(n+1)/2 spectra for polarization (ordering set by
        ``new``; see :func:`synalm`).
    nside : int
        HEALPix nside parameter for output map
    lmax : int, optional
        Maximum multipole l. If None (or negative), matches healpy by using
        ``min(len(cls) - 1, 3 * nside - 1)``. Because jax-healpy relies on
        s2fft transforms that require ``lmax >= 2 * nside - 1``, a
        ``ValueError`` is raised if the requested or derived ``lmax`` falls
        below that threshold.
    mmax : int, optional
        Maximum m. Only ``mmax == lmax`` is supported; any other value raises
        NotImplementedError.
    alm : bool, optional
        If True, return both the map and the alm coefficients used to generate it.
        If False, return only the map. Default: False
    pol : bool, optional
        If True and several spectra are given, assume they are TEB (auto- and
        cross-) spectra and return TQU maps stacked as ``(3, npix)``. With a
        single input spectrum ``pol`` has no effect (scalar output), matching
        healpy. Default: True
    pixwin : bool, optional
        Apply pixel window function. Not supported yet.
        Raises NotImplementedError if True. Default: False
    fwhm : float, optional
        Gaussian beam FWHM in radians for smoothing. Default: 0.0
        For polarized output the spin-2 beam correction is applied to E/B.
    sigma : float, optional
        Gaussian beam sigma in radians. If specified, overrides fwhm.
    new : bool, optional
        Ordering convention for a sequence of input spectra (matches healpy):
        True for the new by-diagonal order, False (default) for the old by-row
        order. Ignored for a single input spectrum. See :func:`synalm`.
    method : str, optional
        Transform method ('jax', 'jax_healpy', 'jax_cuda'). Default: 'jax'
        JAX-specific parameter not present in healpy.

    Returns
    -------
    map : array-like
        Random HEALPix map in RING scheme, shape ``(npix,)`` for scalar input or
        ``(3, npix)`` (T, Q, U) for polarized input.
    alm : array-like, optional
        Spherical harmonic coefficients in s2fft format (only if alm=True).

    Notes
    -----
    This function generates random alm coefficients from the input power
    spectrum/spectra (see :func:`synalm`), then transforms them to a map using
    :func:`alm2map`. For polarization the per-multipole field covariance is built
    from the (cross-)spectra and its matrix square root is used so that
    cross-spectra such as TE are reproduced exactly. Pixel window functionality
    is not yet implemented.

    The polarized version of synfast does not run on GPU yet: it uses ``sqrtm``,
    whose Schur decomposition is not yet implemented on GPU by JAX. The scalar
    (single-spectrum) path is unaffected.

    Examples
    --------
    Generate a random scalar map from a power spectrum:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> nside = 32
    >>> lmax = 3 * nside - 1
    >>> ell = jnp.arange(lmax + 1)
    >>> cl = 1.0 / (ell + 10)**2  # Simple power-law spectrum
    >>> key = jax.random.PRNGKey(42)
    >>> random_map = jhp.synfast(key, cl, nside, lmax=lmax, pol=False)

    Generate a map and get the alm coefficients:

    >>> random_map, alm_coeffs = jhp.synfast(key, cl, nside, lmax=lmax, alm=True, pol=False)
    """
    # Validate unsupported parameters
    if pixwin:
        raise NotImplementedError('pixwin=True is not implemented')

    is_multi, payload = _as_spectra_list(cls)
    # healpy: a single input spectrum means pol has no effect.
    pol_active = pol and is_multi

    # Resolve lmax following healpy: min(longest spectrum - 1, 3 * nside - 1).
    if is_multi:
        cls_lmax = max(jnp.asarray(c).shape[0] for c in payload if c is not None) - 1
    else:
        cls_lmax = jnp.asarray(payload).shape[0] - 1

    if lmax is None or lmax < 0:
        resolved_lmax = min(cls_lmax, 3 * nside - 1)
    else:
        resolved_lmax = int(lmax)

    if mmax is not None and mmax != resolved_lmax:
        raise NotImplementedError('Specifying mmax != lmax is not implemented.')

    min_supported_lmax = 2 * nside - 1
    if resolved_lmax < min_supported_lmax:
        raise ValueError(
            f'synfast requires lmax >= {min_supported_lmax} for nside={nside} when using the '
            f's2fft backend (got lmax={resolved_lmax}). Provide a longer C_l array or specify a '
            'larger lmax (you can zero-pad the spectrum) before calling synfast.'
        )

    # Like healpy: draw the alms (correlated TEB for several spectra, scalar otherwise)
    # then transform. pol_active drives whether E/B map to Q, U (spin-2) or to
    # independent spin-0 maps.
    alms = synalm(prng_key, cls, lmax=resolved_lmax, mmax=mmax, new=new, healpy_ordering=False)
    map_synth = alm2map(
        alms,
        nside,
        lmax=resolved_lmax,
        pixwin=pixwin,
        pol=pol_active,
        fwhm=fwhm,
        sigma=sigma,
        healpy_ordering=False,
        method=method,
    )

    if alm:
        return map_synth, alms
    return map_synth


@jax.jit(static_argnames=['spin', 'lmax', 'mmax', 'iter', 'method', 'healpy_ordering'])
def map2alm_spin(
    maps,
    spin: int,
    lmax: int | None = None,
    mmax: int | None = None,
    iter: int = 0,
    method: str = 'jax',
    healpy_ordering: bool = False,
) -> ArrayLike | list[ArrayLike]:
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
    iter : int, optional
        Number of iterative refinement iterations in the forward transform.
        Unlike healpy's ``map2alm_spin`` (which has no ``iter``), this is exposed
        here. Default: 0 (no iteration, matching healpy's behavior).
    method : str, optional
        Transform method ('jax', 'jax_healpy', 'jax_cuda'). Default: 'jax'
    healpy_ordering : bool, optional
        If True, return alms in healpy format. If False, s2fft format. Default: False

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

    >>> import jax
    >>> import jax_healpy as jhp
    >>> nside = 8
    >>> lmax = 2 * nside - 1
    >>> npix = 12 * nside**2
    >>> Q_map = jax.random.normal(jax.random.PRNGKey(0), (npix,))
    >>> U_map = jax.random.normal(jax.random.PRNGKey(1), (npix,))
    >>> E_lm, B_lm = jhp.map2alm_spin([Q_map, U_map], spin=2, lmax=lmax)
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

    lmax = _resolve_lmax(nside, lmax)
    target_L = lmax + 1

    if mmax is not None and mmax != lmax:
        raise NotImplementedError('Specifying mmax != lmax is not implemented.')

    # Call core function
    flm = _map2alm_core(
        maps=maps,
        lmax=target_L - 1,
        mmax=mmax,
        iter=iter,
        method=method,
        spin=spin,
    )

    if healpy_ordering:
        if spin != 0:
            flm = [flm_2d_to_hp_fast(f, target_L) for f in flm]
        else:
            flm = flm_2d_to_hp_fast(flm, target_L)

    return flm


@jax.jit(static_argnames=['nside', 'spin', 'lmax', 'mmax', 'method', 'healpy_ordering'])
def alm2map_spin(
    alms,
    nside: int,
    spin: int,
    lmax: int | None = None,
    mmax: int | None = None,
    method: str = 'jax',
    healpy_ordering: bool = False,
) -> ArrayLike | list[ArrayLike]:
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
        If True, input alms are in healpy format. If False, s2fft format. Default: False

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
                # healpy 1D layout (mmax == lmax): nalm = (lmax+1)*(lmax+2)/2
                lmax = _lmax_from_nalm(alms[0].shape[0])
            else:
                # For s2fft format: shape is (L, 2L-1)
                lmax = alms[0].shape[0] - 1
    else:
        alms = jnp.asarray(alms)
        if lmax is None:
            if healpy_ordering:
                lmax = _lmax_from_nalm(alms.shape[0])
            else:
                lmax = alms.shape[0] - 1

    # Call core function
    maps = _alm2map_core(
        alms=alms, nside=nside, lmax=lmax, mmax=mmax, method=method, spin=spin, healpy_ordering=healpy_ordering
    )

    return maps
