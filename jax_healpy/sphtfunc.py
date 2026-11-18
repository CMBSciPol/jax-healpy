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

import inspect
import logging
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Complex, Float, Inexact, Integer, PRNGKeyArray

# Two side effects of importing s2fft are undone:
# - s2fft checks the precision when it is imported, before the users had a chance to enable
#   64-bit precision after importing jax-healpy. Its logger is disabled during the import:
#   jax-healpy warns instead when the transforms are called (see warn_if_not_x64).
# - s2fft.recursions.price_mcewen ignores all the warnings of the process, with
#   warnings.filterwarnings('ignore'): catch_warnings restores the filters after the import.
#   Only the warnings that s2fft meant to hide are then ignored: in 32-bit precision, JAX
#   warns about each float64 or complex128 array that s2fft requests (when the transforms
#   are compiled, so they cannot be ignored only while the transforms are called).
_s2fft_logger = logging.getLogger('s2fft')
_s2fft_logger_disabled = _s2fft_logger.disabled
_s2fft_logger.disabled = True
try:
    with warnings.catch_warnings():
        from s2fft.recursions.price_mcewen import generate_precomputes_jax
        from s2fft.transforms import spherical
except ImportError:
    _S2FFT_IS_AVAILABLE = False
else:
    _S2FFT_IS_AVAILABLE = True
    warnings.filterwarnings('ignore', message='Explicitly requested dtype', category=UserWarning, module=r's2fft\.')
finally:
    _s2fft_logger.disabled = _s2fft_logger_disabled

from jax_healpy import _ducc0, npix2nside
from jax_healpy._precision import warn_if_not_x64
from jax_healpy.pixelfunc import UNSEEN, mask_bad

__all__ = [
    'alm2cl',
    'alm2map',
    'alm2map_spin',
    'almxfl',
    'anafast',
    'bad_pixel_mask',
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


_INSTALL_DUCC0 = 'install it with `pip install "jax-healpy[cpu]"`'
_INSTALL_S2FFT = 'install it with `pip install "jax-healpy[gpu]"`'


def _s2fft_is_available() -> bool:
    """Whether s2fft is installed."""
    return _S2FFT_IS_AVAILABLE


def requires_s2fft(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    """Make ``func`` raise an ``ImportError`` when it is called without s2fft."""

    @wraps(func)
    def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        if not _s2fft_is_available():
            raise ImportError(f"Missing optional library 's2fft': {_INSTALL_S2FFT}.")
        return func(*args, **kwargs)

    return wrapper


@requires_s2fft
def precompute_temperature_harmonic_transforms(
    nside: int, lmax: int | None = None, sampling: str = 'healpix', pix2harm: bool = False
) -> list:
    r"""Precompute the recursion coefficients of the scalar transforms of s2fft.

    These are the coefficients of the Price-McEwen recursion, from which s2fft computes the
    Wigner $d$-functions of its spin 0 transforms at `nside` and `lmax`. They are returned by
    `s2fft.recursions.price_mcewen.generate_precomputes_jax`.

    !!! note
        The transforms of jax-healpy compute their own coefficients and do not take these ones:
        they serve to call the transforms of s2fft directly, with their `precomps` argument.

    Parameters
    ----------
    nside : int
        The resolution of the maps.
    lmax : int, optional
        The maximum multipole $\ell_\mathrm{max}$. If None, `3 * nside - 1`.
    sampling : str, optional
        The sampling scheme of s2fft.
    pix2harm : bool, optional
        If True, the coefficients of the forward transform, from a map to its coefficients, and
        otherwise those of the inverse transform.

    Returns
    -------
    list
        The recursion coefficients, as returned by s2fft.

    Raises
    ------
    ImportError
        If s2fft is not installed.

    Examples
    --------
    >>> import jax_healpy as jhp
    >>> precomps = jhp.precompute_temperature_harmonic_transforms(16, pix2harm=True)
    """
    if lmax is None:
        L = 3 * nside
    else:
        L = lmax + 1

    return generate_precomputes_jax(L, spin=0, sampling=sampling, nside=nside, forward=pix2harm)


@requires_s2fft
def precompute_polarization_harmonic_transforms(
    nside: int, lmax: int | None = None, sampling: str = 'healpix', pix2harm: bool = False
) -> list:
    r"""Precompute the recursion coefficients of the spin 2 transforms of s2fft.

    These are the coefficients of the Price-McEwen recursion, from which s2fft computes the
    Wigner $d$-functions of its spin 2 transforms at `nside` and `lmax`. They are returned by
    `s2fft.recursions.price_mcewen.generate_precomputes_jax`. Those of spin $-2$ are not needed:
    the polarized transforms deduce the spin $-2$ coefficients of the maps from the spin 2 ones.

    !!! note
        The transforms of jax-healpy compute their own coefficients and do not take these ones:
        they serve to call the transforms of s2fft directly, with their `precomps` argument.

    Parameters
    ----------
    nside : int
        The resolution of the maps.
    lmax : int, optional
        The maximum multipole $\ell_\mathrm{max}$. If None, `3 * nside - 1`.
    sampling : str, optional
        The sampling scheme of s2fft.
    pix2harm : bool, optional
        If True, the coefficients of the forward transform, from the maps to their coefficients,
        and otherwise those of the inverse transform.

    Returns
    -------
    list
        The recursion coefficients of the spin 2 transforms, as returned by s2fft.

    Raises
    ------
    ImportError
        If s2fft is not installed.

    Examples
    --------
    >>> import jax_healpy as jhp
    >>> precomps = jhp.precompute_polarization_harmonic_transforms(16, pix2harm=True)
    """
    if lmax is None:
        L = 3 * nside
    else:
        L = lmax + 1

    return generate_precomputes_jax(L, spin=2, sampling=sampling, nside=nside, forward=pix2harm)


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


def _lmax_from_alm(alm: ArrayLike, healpy_ordering: bool) -> int:
    """Infer lmax from the shape of the alms (mmax == lmax), ignoring the leading axes.

    The alms have the shape ``(..., nalm)`` in the healpy layout and ``(..., L, 2L - 1)`` in
    the s2fft layout.
    """
    if not healpy_ordering:
        return alm.shape[-2] - 1
    nalm = alm.shape[-1]
    lmax = _lmax_from_nalm(nalm)
    if (lmax + 1) * (lmax + 2) // 2 != nalm:
        raise ValueError(f'Invalid number of alms in the healpy layout: {nalm}.')
    return lmax


def _resolve_lmax(nside: int, lmax: int | None) -> int:
    """Resolve ``lmax``, by default ``3*nside - 1``."""
    return 3 * nside - 1 if lmax is None else lmax


def _check_lmax_is_supported(method: str, nside: int, lmax: int) -> None:
    """Raise if the transforms of ``method`` cannot be computed at this ``lmax``.

    The transforms of s2fft require ``lmax >= 2*nside - 1``, those of ducc0 accept any ``lmax``,
    like those of healpy.
    """
    if method != 'ducc0' and lmax < 2 * nside - 1:
        raise NotImplementedError(
            f'The transforms of s2fft (method={method!r}) require lmax >= 2*nside - 1 = {2 * nside - 1} for '
            f"nside={nside}, got lmax={lmax}. Those of ducc0 (method='ducc0', on CPU) accept any lmax."
        )


_METHODS = ('ducc0', 'jax', 'jax_cuda')

_SLOW_ON_CPU = (
    'The spherical harmonic transforms are computed on CPU by s2fft, which is two to three orders of '
    f'magnitude slower there than ducc0: {_INSTALL_DUCC0}. To keep the transforms of s2fft without '
    "this warning, pass method='jax'."
)

_warned_slow_on_cpu = False


def _default_platform() -> str:
    """Return the platform of the device on which the computations run by default.

    It is the platform of the device set by ``jax.default_device``, as a context manager or as the
    ``jax_default_device`` option, and otherwise that of the default backend of JAX. The arrays
    that are committed to another device are not taken into account.
    """
    device = jax.config.jax_default_device
    if device is None:
        return jax.default_backend()
    if isinstance(device, str):
        return device
    return device.platform


def _default_method() -> str:
    """Return the transform backend used when none is requested.

    The transforms of ducc0 only run on CPU, where they are orders of magnitude faster than those
    of s2fft: they are the default there, if ducc0 is installed. The device is the default one of
    JAX, which ``jax.default_device`` can change. Those of s2fft are the default on
    the other devices. They are also used on CPU when ducc0 is missing, with a warning emitted once
    per session.

    An ``ImportError`` is raised if the library of the default backend is not installed.
    """
    global _warned_slow_on_cpu
    backend = _default_platform()
    if backend == 'cpu' and _ducc0.is_available():
        return 'ducc0'
    if not _s2fft_is_available():
        if not _ducc0.is_available():
            raise ImportError(
                'The spherical harmonic transforms require an optional library: ducc0 to compute them on CPU '
                '(`pip install "jax-healpy[cpu]"`) or s2fft to compute them on GPU (`pip install "jax-healpy[gpu]"`).'
            )
        raise ImportError(
            f"Missing optional library 's2fft', which computes the spherical harmonic transforms on {backend.upper()}: "
            f"{_INSTALL_S2FFT}. To compute them on the CPU with ducc0 instead, pass method='ducc0'."
        )
    if backend == 'cpu' and not _warned_slow_on_cpu:
        _warned_slow_on_cpu = True
        # stacklevel: _resolve_method, the wrapper of _with_default_method, then the caller
        warnings.warn(_SLOW_ON_CPU, UserWarning, stacklevel=4)
    return 'jax'


def _resolve_method(method: str | None) -> str:
    """Return the transform backend designated by ``method``, which is the default one for ``None``.

    A ``ValueError`` is raised if ``method`` is not a transform backend that jax-healpy can use, and
    an ``ImportError`` if the library of the backend is not installed, or if ducc0 is installed but
    cannot be imported.
    """
    if method is None:
        method = _default_method()
        if method == 'ducc0':
            _ducc0.check_import()
        return method
    if method == 'jax_healpy':
        # s2fft would accept it, but its healpy wrappers have no lowering rule (the compilation
        # fails) and they ignore the spin (polarization would be transformed as two scalar fields).
        raise ValueError(
            "method='jax_healpy' is not supported: the healpy wrappers of s2fft cannot be compiled by "
            "jax.jit and transform every field as a scalar, whatever its spin. Use method='ducc0' to run "
            "the transforms of a C++ library on CPU, or method='jax'."
        )
    if method not in _METHODS:
        raise ValueError(f'Invalid method {method!r}: the transform backends are {", ".join(map(repr, _METHODS))}.')
    if method == 'ducc0':
        if not _ducc0.is_available():
            raise ImportError(f"Missing optional library 'ducc0', required by method='ducc0': {_INSTALL_DUCC0}.")
        _ducc0.check_import()
    elif not _s2fft_is_available():
        raise ImportError(f"Missing optional library 's2fft', required by method={method!r}: {_INSTALL_S2FFT}.")
    return method


def _with_default_method(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    """Replace ``method=None`` by the default transform backend before ``func`` is called.

    This decorator is applied on top of ``jax.jit``, for which ``method`` is a static argument: the
    compiled functions are then cached per backend, instead of freezing the backend that was the
    default one when ``method=None`` was first traced.
    """
    signature = inspect.signature(func)

    def with_method(function: Callable) -> Callable:
        def call(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.arguments['method'] = _resolve_method(bound.arguments.get('method'))
            return function(*bound.args, **bound.kwargs)

        return call

    wrapper = wraps(func)(with_method(func))
    # The methods of the jitted function, such as lower or eval_shape, are not copied by wraps
    for name in ('lower', 'trace', 'eval_shape'):
        if hasattr(func, name):
            setattr(wrapper, name, with_method(getattr(func, name)))
    return wrapper


def _check_real_maps(maps: ArrayLike | list[ArrayLike]) -> None:
    """Raise a ``TypeError`` if a map to transform is complex.

    The transforms assume real maps: the imaginary part of a scalar map would be ignored, and the
    E and B coefficients of complex Q and U maps would be wrong, since they are deduced from the
    symmetry of the spin transforms of real fields.
    """
    for m in maps if isinstance(maps, (list, tuple)) else [maps]:
        if jnp.iscomplexobj(m):
            raise TypeError(f'The maps to transform must be real, not of type {jnp.result_type(m)}.')


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


def _spectra_real_dtype(*cls: ArrayLike) -> jnp.dtype:
    """Return the real dtype in which to draw alms from the spectra `cls`.

    It is the floating dtype of the spectra, at least float32. Non-floating spectra use
    JAX's default float dtype (float64 only when x64 is enabled, so that an unavailable
    float64 is never requested). The dtypes are read without their weak type: a weakly typed
    float64 spectrum, such as ``1 / (jnp.arange(n) + 10.0) ** 2``, would otherwise promote
    with float32 to float32 and produce complex64 alms.
    """
    dtype = jnp.result_type(*[jnp.asarray(cl).dtype for cl in cls])
    if not jnp.issubdtype(dtype, jnp.floating):
        return jnp.result_type(float)
    return jnp.promote_types(dtype, jnp.float32)


def _draw_unit_alms(prng_key: PRNGKeyArray, L: int, dtype: jnp.dtype) -> Array:
    """Draw unit-variance alms of a real field on the ``(L, L)`` grid of ``ell`` and ``m >= 0``.

    The real and imaginary parts of the m > 0 coefficients are drawn from N(0, 1/2), the real m = 0
    coefficients from N(0, 1), and the coefficients of m > ell are zero. Only the L(L+1)/2 valid
    coefficients are drawn, in the healpy order, then gathered into the grid.
    """
    nalm = L * (L + 1) // 2
    key_real, key_imag = jax.random.split(prng_key)
    real = jax.random.normal(key_real, shape=(nalm,), dtype=dtype)
    imag = jax.random.normal(key_imag, shape=(nalm,), dtype=dtype)
    ell = jnp.arange(L)[:, None]
    m = jnp.arange(L)[None, :]
    valid = m <= ell
    index = jnp.where(valid, m * (2 * L - 1 - m) // 2 + ell, 0)
    scale = jnp.where(m == 0, 1, np.sqrt(0.5)).astype(dtype)
    alms = (real[index] + 1j * jnp.where(m == 0, 0, imag[index])) * scale
    return jnp.where(valid, alms, 0)


def _alms_with_negative_m(alms: Array) -> Array:
    """Extend alms of shape ``(..., L, L)`` for ``m >= 0`` to the s2fft layout ``(..., L, 2L-1)``.

    The coefficients of negative m are those of a real field: a_{l,-m} = (-1)^m conj(a_{lm}).
    """
    L = alms.shape[-1]
    phase = (-1) ** jnp.arange(1, L)
    left_half = jnp.flip(jnp.conj(alms[..., 1:]) * phase, axis=-1)
    return jnp.concatenate([left_half, alms], axis=-1)


def _is_traced(x: ArrayLike) -> bool:
    """Whether ``x`` is traced by a JAX transformation, such as ``jit``, so that its value is unknown."""
    return isinstance(x, jax.core.Tracer)


def _generate_random_alm(cl: ArrayLike, lmax: int, prng_key: PRNGKeyArray) -> ArrayLike:
    """Draw random alm with healpy reality convention (mmax == lmax). A negative spectrum gives NaN alms."""
    L = lmax + 1
    cl = jnp.asarray(cl)
    real_dtype = _spectra_real_dtype(cl)
    if cl.shape[0] < L:
        cl = jnp.pad(cl, (0, L - cl.shape[0]), constant_values=0)
    elif cl.shape[0] > L:
        cl = cl[:L]

    # The square root is evaluated on positive values only, so that the gradient with respect
    # to the spectrum is finite where it vanishes (e.g. the monopole and dipole of the CMB).
    positive = cl > 0
    scale = jnp.where(positive, jnp.sqrt(jnp.where(positive, cl, 1)), jnp.where(cl == 0, 0, jnp.nan))
    alms = _draw_unit_alms(prng_key, L, real_dtype) * scale[:, None]
    return _alms_with_negative_m(alms)


def _spectra_covariance(cls_list: list, lmax: int | None, new: bool) -> Array:
    """Covariance matrices ``(L, n, n)`` of the fields at each multipole, from their (cross-)spectra.

    ``cls_list`` holds 4 or n(n+1)/2 spectra, or ``None`` for a zero cross-spectrum, in the order
    set by ``new`` (see [`synalm`][]).
    """
    if lmax is None:
        lmax = max(c.shape[0] for c in cls_list if c is not None) - 1
    L = lmax + 1
    real_dtype = _spectra_real_dtype(*[c for c in cls_list if c is not None])

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
    return cov


def _check_spectra(cls: ArrayLike | tuple[ArrayLike, ...], lmax: int | None, new: bool) -> None:
    """Raise a ``ValueError`` if the spectra are negative or if their covariance matrix is not
    positive semi-definite at some multipole. Traced spectra, whose values are unknown, are not
    checked: they give NaN alms.
    """
    is_multi, payload = _as_spectra_list(cls)
    spectra = payload if is_multi else [payload]
    if any(_is_traced(c) for c in spectra if c is not None):
        return
    if not is_multi:
        cl = np.asarray(payload)[: None if lmax is None else lmax + 1]
        if (cl < 0).any():
            raise ValueError(f'The power spectrum is negative at ell = {np.flatnonzero(cl < 0).tolist()}.')
        return
    factor = _cholesky_psd(_spectra_covariance([None if c is None else jnp.asarray(c) for c in payload], lmax, new))
    invalid = ~np.all(np.isfinite(np.asarray(factor)), axis=(-2, -1))
    if invalid.any():
        raise ValueError(
            'The covariance matrix of the spectra is not positive semi-definite at '
            f'ell = {np.flatnonzero(invalid).tolist()}: a spectrum is negative, or a correlation '
            'coefficient is greater than 1 in absolute value.'
        )


def _checks_spectra(
    get_lmax: Callable[[dict], int | None],
) -> Callable[[Callable[Param, ReturnType]], Callable[Param, ReturnType]]:
    """Check the spectra of a compiled function before calling it, where their values are known.

    ``get_lmax`` returns the maximum multipole of the spectra to check from the arguments of the call.
    """

    def decorator(function: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
        signature = inspect.signature(function)

        @wraps(function)
        def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            _check_spectra(arguments.arguments['cls'], get_lmax(arguments.arguments), arguments.arguments['new'])
            return function(*args, **kwargs)

        return wrapper

    return decorator


def _cholesky_psd(cov: ArrayLike) -> Array:
    """Lower Cholesky factors of a stack of covariance matrices of shape ``(..., n, n)``.

    The fields of zero variance are left out of the factorization: their rows and columns of
    the factor are zero. The other fields must have a positive definite covariance, up to a
    relative regularization of 100 times the machine precision, which handles perfectly
    correlated fields: otherwise, the factor is NaN. Unlike the matrix square root, the
    factorization is differentiable.
    """
    n = cov.shape[-1]
    variance = jnp.diagonal(cov, axis1=-2, axis2=-1)
    active = variance != 0
    active_pair = active[..., :, None] & active[..., None, :]
    eye = jnp.eye(n, dtype=cov.dtype)
    safe_cov = jnp.where(active_pair, cov, eye)
    jitter = 100 * jnp.finfo(cov.dtype).eps
    safe_cov = safe_cov + jitter * eye * jnp.diagonal(safe_cov, axis1=-2, axis2=-1)[..., None, :]
    return jnp.where(active_pair, jnp.linalg.cholesky(safe_cov), 0)


def _getn(k: int) -> int:
    """Return n such that n*(n+1)/2 == k, or -1 if k is not triangular.

    Mirrors ``healpy.sphtfunc._sphtools._getn``; used to map a flat list of
    n(n+1)/2 cross-spectra to the number of underlying fields n.
    """
    n = round((np.sqrt(8 * k + 1) - 1) / 2)
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


# Up to this lmax, the indices of the alms in the healpy ordering, which are below their number
# (lmax + 1)(lmax + 2) / 2, fit in 32-bit integers. Beyond, 64-bit integers are required, and so the
# 64-bit precision of JAX.
_INT32_ALMS_LMAX = 65534

# Up to this lmax, the flat indices m L + ell of the 2-D alms read by order, below L², fit in
# 32-bit integers.
_INT32_FLAT_INDICES_LMAX = 46339


def _alm_index_dtype(lmax: int) -> type:
    """Return the integer type of the indices of the alms of ``lmax`` in the healpy ordering.

    It is ``int32`` up to lmax = 65534 and ``int64`` beyond, which requires the 64-bit precision
    of JAX: a ``ValueError`` is raised without it.
    """
    if lmax <= _INT32_ALMS_LMAX:
        return jnp.int32
    if jnp.result_type(int) == jnp.int64:
        return jnp.int64
    raise ValueError(
        f'The alms of lmax={lmax} cannot be indexed with 32-bit integers, which are limited to '
        f"lmax={_INT32_ALMS_LMAX}: enable 64-bit precision with jax.config.update('jax_enable_x64', True)."
    )


def _order_offsets(lmax: int) -> np.ndarray:
    """Return, for each order m, the index of the coefficient (0, m) in the healpy ordering.

    The coefficient (ell, m) has the index ``offsets[m] + ell``. The offsets are below the number
    of alms, and computed with 64-bit integers.
    """
    m = np.arange(lmax + 1, dtype=np.int64)
    return m * (2 * lmax + 1 - m) // 2


def _alm_hp_to_2d(alms: Complex[Array, '... nalm'], lmax: int) -> Complex[Array, '... L M']:
    """Convert the alms of real fields from the healpy ordering (m >= 0) to the 2-D one (all m).

    Same conversion as ``flm_hp_to_2d_fast`` of s2fft, except that the coefficients are gathered
    instead of scattered: ``jax.linear_transpose`` can then transpose it.
    """
    L = lmax + 1
    dtype = _alm_index_dtype(lmax)
    ell = jnp.arange(L, dtype=dtype)[:, None]
    m = jnp.arange(-lmax, L, dtype=dtype)[None, :]
    abs_m = jnp.abs(m)
    valid = abs_m <= ell
    # index of (ell, |m|) in the healpy ordering
    index = jnp.where(valid, jnp.asarray(_order_offsets(lmax), dtype=dtype)[abs_m] + ell, 0)
    values = jnp.asarray(alms)[..., index]
    # a_{l,-m} = (-1)^m conj(a_{l,m})
    values = jnp.where(m < 0, jnp.where(abs_m % 2 == 0, 1, -1) * jnp.conj(values), values)
    return jnp.where(valid, values, 0)


# Up to this lmax, the indices of the alms in the healpy ordering are stored as constants in the
# compiled functions (2 MB at most), which costs nothing when they run. Beyond, they are computed
# each time the functions run: it is faster than reading constants that do not fit in the caches.
_STORED_INDICES_LMAX = 1023

# Up to this lmax, the orders of the alms can be computed from a square root with 32-bit integers:
# the largest integer of the computation, (2 lmax + 3)², does not exceed 2³¹ - 1.
_INT32_INDICES_LMAX = 23168


def _alm_hp_indices(
    lmax: int,
) -> tuple[Integer[Array | np.ndarray, ' nalm'], Integer[Array | np.ndarray, ' nalm']]:
    """Return the degree ell and the order m of each coefficient of the healpy ordering.

    The alms are stored order by order: (0, 0), (1, 0), ..., (lmax, 0), (1, 1), ..., (lmax, lmax).
    The indices are numpy arrays when they are stored in the compiled functions, and JAX arrays
    when they are computed. They are 32-bit integers up to lmax = 23168, and beyond if 64-bit
    precision is disabled, up to lmax = 65534 (see ``_alm_index_dtype``). Otherwise, they are
    64-bit integers.
    """
    L = lmax + 1
    nalm = L * (L + 1) // 2
    if lmax <= _STORED_INDICES_LMAX:
        m = np.repeat(np.arange(L, dtype=np.int32), np.arange(L, 0, -1))
        ell = np.arange(nalm, dtype=np.int32) - m * (2 * lmax + 1 - m) // 2
        return ell, m

    if lmax <= _INT32_INDICES_LMAX:
        dtype = jnp.int32
    elif jnp.result_type(int) == jnp.int64:
        dtype = jnp.int64
    else:
        # The square root below would overflow 32-bit integers: the orders are repeated instead,
        # which is about three times slower.
        dtype = _alm_index_dtype(lmax)
        m = jnp.repeat(jnp.arange(L, dtype=dtype), np.arange(L, 0, -1), total_repeat_length=nalm)
        ell = jnp.arange(nalm, dtype=dtype) - jnp.asarray(_order_offsets(lmax), dtype=dtype)[m]
        return ell, m

    # The first coefficient of the order m has the index m (b - m) / 2, with b = 2 lmax + 3: the
    # order of the coefficient i is the integer part of the smallest root of m² - b m + 2 i.
    index = jnp.arange(nalm, dtype=dtype)
    b = 2 * lmax + 3
    discriminant = (b * b - 8 * index).astype(jnp.result_type(float))
    m = ((b - jnp.sqrt(discriminant)) / 2).astype(dtype)
    # The rounding errors of the root can change its integer part by one: it is then corrected.
    m = m + (index >= (m + 1) * (b - m - 1) // 2) - (index < m * (b - m) // 2)
    ell = index - m * (2 * lmax + 1 - m) // 2
    return ell, m


def _alm_2d_to_hp(alms: Complex[Array, '... L M'], lmax: int) -> Complex[Array, '... nalm']:
    """Convert alms from the 2-D ordering (all m) to the healpy one (m >= 0).

    Same conversion as ``flm_2d_to_hp_fast`` of s2fft: the coefficients of m < 0 are dropped.
    """
    L = lmax + 1
    alms = jnp.asarray(alms)
    ell, m = _alm_hp_indices(lmax)
    # With the orders as rows, the coefficients are read in the order in which they are stored.
    by_order = jnp.swapaxes(alms[..., lmax:], -1, -2)
    keywords = {'mode': 'promise_in_bounds', 'indices_are_sorted': True, 'unique_indices': True}
    if lmax > _INT32_FLAT_INDICES_LMAX and ell.dtype == np.int32:
        # The flat indices would overflow 32-bit integers
        return by_order.at[..., m, ell].get(**keywords)
    by_order = by_order.reshape(*alms.shape[:-2], L * L)
    return by_order.at[..., m * L + ell].get(**keywords)


def _alm2map_ducc0(
    alms: ArrayLike, nside: int, lmax: int, spin: int, healpy_ordering: bool
) -> Float[Array, ' npix'] | list[Float[Array, ' npix']]:
    """alm2map with the transforms of ducc0, which exchange alms in the healpy ordering."""
    L = lmax + 1
    alms = [alms] if spin == 0 else list(alms)
    if not healpy_ordering:
        for alm in alms:
            if jnp.shape(alm)[-2:] != (L, 2 * L - 1):
                raise ValueError(
                    f'The alms have the shape {jnp.shape(alm)}, instead of {(L, 2 * L - 1)} for lmax={lmax}. '
                    'Pass healpy_ordering=True if they follow the healpy ordering.'
                )
        alms = [_alm_2d_to_hp(alm, lmax) for alm in alms]
    # Like the s2fft transforms, compute in the default precision of JAX.
    alms = jnp.stack(alms).astype(jnp.result_type(complex))
    nalm = L * (L + 1) // 2
    if alms.shape[-1] != nalm:
        raise ValueError(f'The alms have {alms.shape[-1]} coefficients, instead of {nalm} for lmax={lmax}.')

    maps = _ducc0.alm2map(alms, nside, lmax, spin)
    return maps[0] if spin == 0 else [maps[0], maps[1]]


def _map2alm_ducc0(
    maps: ArrayLike, lmax: int, iter: int, spin: int, bad_mask: ArrayLike | None, healpy_ordering: bool
) -> Complex[Array, '...'] | list[Complex[Array, '...']]:
    """map2alm with the transforms of ducc0, which exchange alms in the healpy ordering."""
    if spin == 0:
        maps = [_apply_bad_mask(maps, bad_mask)]
    else:
        masks = (None, None) if bad_mask is None else bad_mask
        maps = [_apply_bad_mask(m, mask) for m, mask in zip(maps, masks)]
    # Like the s2fft transforms, compute in the default precision of JAX.
    maps = jnp.stack(maps).astype(jnp.result_type(float))
    nside = npix2nside(maps.shape[-1])

    alms = _ducc0.map2alm(maps, nside, lmax, spin, iter)
    if not healpy_ordering:
        alms = _alm_hp_to_2d(alms, lmax)
    return alms[0] if spin == 0 else [alms[0], alms[1]]


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
        Transform backend ('ducc0', 'jax', 'jax_cuda'), resolved by the public functions
    spin : int
        Spin weight (0 for scalar, 2 for polarization, etc.)
    healpy_ordering : bool
        Whether input alms use healpy ordering

    Returns
    -------
    ArrayLike
        Output map(s)
    """
    warn_if_not_x64()
    if mmax is not None and mmax != lmax:
        raise NotImplementedError('Specifying mmax != lmax is not implemented.')
    _check_lmax_is_supported(method, nside, lmax)

    if method == 'ducc0':
        return _alm2map_ducc0(alms, nside, lmax, spin, healpy_ordering)

    L = lmax + 1

    # Convert from healpy ordering if needed
    if healpy_ordering:
        if spin != 0:
            # For spin transforms, convert list of 2 alms to s2fft format
            alms = [_alm_hp_to_2d(alm, lmax) for alm in alms]
        else:
            alms = _alm_hp_to_2d(alms, lmax)

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


def bad_pixel_mask(m: ArrayLike) -> Bool[Array, '...']:
    """Return the mask of the bad pixels of a map: the `UNSEEN` and the non-finite ones.

    A pixel is bad if it is equal to [`UNSEEN`][jax_healpy.UNSEEN], as tested by
    [`mask_bad`][jax_healpy.mask_bad], or if it is NaN or infinite. The mask is meant for the
    `bad_mask` argument of [`map2alm`][jax_healpy.map2alm] and
    [`map2alm_spin`][jax_healpy.map2alm_spin], which do not inspect the values of the pixels.

    Parameters
    ----------
    m : ArrayLike
        A map, of shape `(npix,)`, or a stack of maps, of shape `(n, npix)`.

    Returns
    -------
    jax.Array
        The mask, of the shape of `m`, True where the pixel is bad.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> jhp.bad_pixel_mask(jnp.array([0.0, jhp.UNSEEN, jnp.nan, jnp.inf]))
    Array([False,  True,  True,  True], dtype=bool)
    """
    m = jnp.asarray(m)
    return mask_bad(m) | ~jnp.isfinite(m)


def _apply_bad_mask(m: ArrayLike, bad_mask: ArrayLike | None) -> Inexact[Array, '...']:
    """Zero the pixels flagged by ``bad_mask``, returning a fresh array."""
    m = jnp.asarray(m)
    if bad_mask is None:
        return m
    return jnp.where(jnp.asarray(bad_mask), jnp.zeros((), dtype=m.dtype), m)


def _map2alm_core(
    maps: ArrayLike,
    lmax: int,
    mmax: int | None,
    iter: int,
    method: str,
    spin: int,
    bad_mask: ArrayLike | None = None,
    healpy_ordering: bool = False,
) -> Complex[Array, '...']:
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
        Transform backend ('ducc0', 'jax', 'jax_cuda'), resolved by the public functions
    spin : int
        Spin weight (0 for scalar, 2 for polarization, etc.)
    bad_mask : ArrayLike, optional
        Boolean mask matching the structure of ``maps``; flagged pixels are zeroed.
    healpy_ordering : bool
        Whether the output alms use healpy ordering, instead of the s2fft one

    Returns
    -------
    ArrayLike
        Output spherical harmonic coefficients
    """
    _check_real_maps(maps)
    warn_if_not_x64()
    if mmax is not None and mmax != lmax:
        raise NotImplementedError('Specifying mmax != lmax is not implemented.')
    npix = maps[0].shape[-1] if isinstance(maps, (list, tuple)) else jnp.shape(maps)[-1]
    _check_lmax_is_supported(method, npix2nside(npix), lmax)

    if method == 'ducc0':
        return _map2alm_ducc0(maps, lmax, iter, spin, bad_mask, healpy_ordering)

    # For spin transforms, use single-transform approach
    # Reference: healpy alm2map_spin/map2alm_spin convention
    if spin != 0:
        # Input: [Q_map, U_map] where Q and U are real-valued Stokes parameters
        # Output: [alm_E, alm_B] where E and B are complex-valued mode coefficients
        q_mask, u_mask = (None, None) if bad_mask is None else (bad_mask[0], bad_mask[1])
        q_map = _apply_bad_mask(maps[0], q_mask)
        u_map = _apply_bad_mask(maps[1], u_mask)

        nside = npix2nside(q_map.shape[-1])
        L = lmax + 1

        sampling = 'healpix'
        spmd = False

        # Generate precomputes for spin-s transform
        precomps = generate_precomputes_jax(L, spin, sampling, nside, False)

        # Single spin-s forward transform of the complex field Q + iU
        flm_pos = spherical.forward(
            q_map + 1j * u_map,
            L,
            spin=spin,
            nside=nside,
            sampling=sampling,
            method=method,
            reality=False,  # Complex input
            precomps=precomps,
            spmd=spmd,
            iter=iter,
        )

        # The spin -s coefficients of Q - iU are not transformed: since Q and U are real, they
        # follow from the symmetry  _{-s}a_{l,m} = (-1)^(s+m) conj(_{s}a_{l,-m}).
        # E and B are then extracted using the HEALPix convention,
        #     E = -(_{s}a + (-1)^s _{-s}a) / 2,    B = i (_{s}a - (-1)^s _{-s}a) / 2,
        # in which the sign (-1)^s cancels the one of the symmetry.
        m = jnp.arange(-L + 1, L)
        flm_mirror = jnp.where(m % 2 == 0, 1, -1) * jnp.conj(flm_pos[..., ::-1])
        alm_E = -0.5 * (flm_pos + flm_mirror)
        alm_B = 0.5j * (flm_pos - flm_mirror)

        if healpy_ordering:
            return [_alm_2d_to_hp(alm_E, lmax), _alm_2d_to_hp(alm_B, lmax)]
        return [alm_E, alm_B]
    else:
        # Scalar transform (spin=0)
        maps_complex = _apply_bad_mask(maps, bad_mask)
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

        if healpy_ordering:
            return _alm_2d_to_hp(flm_complex, lmax)
        return flm_complex


@_with_default_method
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
    method: str | None = None,
) -> ArrayLike:
    r"""Compute a map, or the T, Q, U maps, from their spherical harmonic coefficients.

    The map is synthesized in the RING scheme from the coefficients $a_{\ell m}$ up to
    $\ell_\mathrm{max}$, optionally smoothed by a Gaussian beam:

    $$m(\theta, \phi) = \sum_{\ell=0}^{\ell_\mathrm{max}} \sum_{m=-\ell}^{\ell}
    a_{\ell m} Y_{\ell m}(\theta, \phi).$$

    With `pol=True`, a stack of T, E, B coefficients gives the T, Q, U maps: the E and B
    coefficients are transformed by the spin 2 transform.

    !!! note "Difference from healpy"
        - The coefficients are in the layout of s2fft, unless `healpy_ordering=True` (see the
          [layout of the coefficients](#layout-of-the-coefficients)).
        - With `pol=True`, a stack of two sets of coefficients is read as E, B and gives the Q, U
          maps, without temperature. healpy requires the three sets T, E, B.
        - `pixwin=True` and `mmax != lmax` raise a `NotImplementedError`.
        - The arguments `inplace` and `verbose` do not exist.

    Parameters
    ----------
    alms : ArrayLike
        The coefficients: one set, of shape `(lmax + 1, 2 * lmax + 1)` in the layout of s2fft or
        `(nalm,)` in that of healpy, or a stack of sets along the first axis (see `pol`). A
        further leading axis is a batch, transformed set by set.
    nside : int
        The resolution of the map.
    lmax : int, optional
        The maximum multipole $\ell_\mathrm{max}$. If None, it is inferred from the shape of
        `alms`.
    mmax : int, optional
        The maximum order $m_\mathrm{max}$. Only `mmax = lmax` is implemented.
    pixwin : bool, optional
        Whether to multiply the coefficients by the pixel window function. Only False is
        implemented.
    fwhm : float, optional
        The full width at half maximum of the Gaussian beam that smooths the map, in radians. A
        Python or NumPy float, not a JAX array: it is a static argument, which cannot be
        differentiated.
    sigma : float, optional
        The standard deviation of the Gaussian beam, in radians, which overrides `fwhm`. A
        Python or NumPy float, like `fwhm`.
    pol : bool, optional
        If True, a stack of 1, 2 or 3 sets of coefficients is read as T; E, B; or T, E, B. If
        False, each set is transformed as a scalar field. A single set is always a scalar field.
    healpy_ordering : bool, optional
        If True, the coefficients are in the layout of healpy, and otherwise in that of s2fft.
    method : str, optional
        The transform backend, `'ducc0'`, `'jax'` or `'jax_cuda'` (see the
        [transform backends](#transform-backends)). If None, `'ducc0'` on CPU if ducc0 is
        installed, and `'jax'` otherwise.

    Returns
    -------
    jax.Array
        The map, of shape `(npix,)`, or the stack of maps along the first axis: Q, U for the
        E, B coefficients, and T, Q, U for the T, E, B ones.

    Raises
    ------
    NotImplementedError
        If `pixwin=True` or `mmax != lmax`, or if the transforms of s2fft are used with
        `lmax < 2 * nside - 1`.

    See Also
    --------
    - [`map2alm`][jax_healpy.map2alm]: the inverse transform.
    - [`alm2map_spin`][jax_healpy.alm2map_spin]: the transform of the coefficients of a spin-weighted field.

    Notes
    -----
    A map is recovered by [`map2alm`][jax_healpy.map2alm] followed by `alm2map` only if it is
    band-limited, for instance if it was synthesized from coefficients up to $\ell_\mathrm{max}$,
    and not if it has power at all the scales of the pixels, as white noise. The accuracy of the
    round trip then depends on `lmax`, `nside` and the number of iterations of `map2alm`: see
    section 5.1 of [Reinecke (2011)](https://arxiv.org/abs/1010.2084).

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> nside = 16
    >>> lmax = 3 * nside - 1
    >>> alms = jhp.synalm(jax.random.key(0), jnp.ones(lmax + 1))
    >>> jhp.alm2map(alms, nside).shape
    (3072,)

    The Q, U maps of E, B coefficients:

    >>> alms_eb = jnp.stack([alms, jnp.zeros_like(alms)])
    >>> jhp.alm2map(alms_eb, nside).shape
    (2, 3072)
    """
    alms = jnp.asarray(alms)
    if alms.ndim == 0:
        raise ValueError('Input alms must have at least one dimension.')
    expected_ndim = 1 if healpy_ordering else 2
    if alms.ndim > expected_ndim + 1 + pol:
        raise ValueError('Input alms have too many dimensions.')

    if pixwin:
        raise NotImplementedError('Pixel-window smoothing is not implemented; set pixwin=False.')

    if lmax is None:
        lmax = _lmax_from_alm(alms, healpy_ordering)

    # Polarized branch: input is shape (n, ...) with n in {1, 2, 3}.
    #   n == 1 -> T        -> T map                 (spin 0)
    #   n == 2 -> E, B     -> (Q, U) maps           (spin 2; not possible in healpy)
    #   n == 3 -> T, E, B  -> (T, Q, U) maps
    if pol and alms.ndim == expected_ndim + 1:
        n = alms.shape[0]
        if n not in (1, 2, 3):
            raise ValueError(f'pol=True requires alms with shape (n, ...), n in (1, 2, 3); got shape {alms.shape}.')

        target_lmax = lmax
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

    L = lmax + 1

    if mmax is not None and mmax != lmax:
        raise NotImplementedError('Specifying mmax != lmax is not implemented.')

    # Apply smoothing if requested (scalar spin-0 beam; mirrors the polarized branch above).
    if fwhm != 0 or sigma is not None:
        alms = smoothalm(alms, fwhm=fwhm, sigma=sigma, pol=False, mmax=mmax, healpy_ordering=healpy_ordering)

    # Call core function
    f = _alm2map_core(
        alms=alms, nside=nside, lmax=L - 1, mmax=mmax, method=method, spin=0, healpy_ordering=healpy_ordering
    )

    return f


@_with_default_method
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
    method: str | None = None,
    bad_mask: ArrayLike | None = None,
) -> ArrayLike:
    r"""Compute the spherical harmonic coefficients of a map, or of the T, Q, U maps.

    The coefficients $a_{\ell m}$ up to $\ell_\mathrm{max}$ are computed by a quadrature over
    the pixels of the map, in the RING scheme, with the uniform weight of the pixel area
    $4\pi / N_\mathrm{pix}$:

    $$a_{\ell m} = \frac{4\pi}{N_\mathrm{pix}} \sum_{p=0}^{N_\mathrm{pix}-1}
    m_p \, Y^*_{\ell m}(\theta_p, \phi_p),$$

    then refined by `iter` iterations. The
    [documentation of anafast](https://healpix.sourceforge.io/html/fac_anafast.htm) in HEALPix
    advises on the choice of `lmax` and `iter`. With `pol=True`, a stack of T, Q, U maps gives
    the T, E, B coefficients: the Q and U maps are transformed by the spin 2 transform.

    !!! note "Difference from healpy"
        - The coefficients are in the layout of s2fft, unless `healpy_ordering=True` (see the
          [layout of the coefficients](#layout-of-the-coefficients)).
        - The `UNSEEN` pixels are not replaced by zeros: the transform never inspects the
          values of the pixels, so that it is a linear function of the map. The bad pixels are
          passed as `bad_mask`, which [`bad_pixel_mask`][jax_healpy.bad_pixel_mask] computes.
        - With `pol=True`, a stack of two maps is read as Q, U and gives the E, B coefficients,
          without temperature. healpy requires the three maps T, Q, U.
        - The weights (`use_weights`, `use_pixel_weights`, `datapath`), `gal_cut` and
          `mmax != lmax` raise a `NotImplementedError`.
        - The argument `verbose` does not exist.

    Parameters
    ----------
    maps : ArrayLike
        The map, of shape `(npix,)` in the RING scheme, or a stack of maps, of shape
        `(n, npix)` (see `pol`).
    lmax : int, optional
        The maximum multipole $\ell_\mathrm{max}$. If None, `3 * nside - 1`.
    mmax : int, optional
        The maximum order $m_\mathrm{max}$. Only `mmax = lmax` is implemented.
    iter : int, optional
        The number of iterations that refine the coefficients.
    pol : bool, optional
        If True, a stack of 1, 2 or 3 maps is read as T; Q, U; or T, Q, U. If False, each map of
        the stack is transformed as a scalar field. A single map is always a scalar field.
    use_weights : bool, optional
        Whether to weight the rings. Only False is implemented.
    datapath : str, optional
        The directory of the pixel weights. Only None is implemented.
    gal_cut : float, optional
        The latitude in degrees: the pixels between `-gal_cut` and `gal_cut` are ignored. Only 0
        is implemented.
    use_pixel_weights : bool, optional
        Whether to weight the pixels. Only False is implemented.
    healpy_ordering : bool, optional
        If True, the coefficients are returned in the layout of healpy, and otherwise in that of
        s2fft.
    method : str, optional
        The transform backend, `'ducc0'`, `'jax'` or `'jax_cuda'` (see the
        [transform backends](#transform-backends)). If None, `'ducc0'` on CPU if ducc0 is
        installed, and `'jax'` otherwise.
    bad_mask : ArrayLike, optional
        The mask of the bad pixels, True where the pixel is bad, broadcastable to the shape of
        `maps`. These pixels are set to zero before the transform. If None, all the pixels are
        used.

    Returns
    -------
    jax.Array
        The coefficients, of shape `(lmax + 1, 2 * lmax + 1)` in the layout of s2fft or
        `(nalm,)` in that of healpy, or their stack along the first axis: E, B for the Q, U maps,
        T, E, B for the T, Q, U maps, and one set per map for `pol=False`.

    Raises
    ------
    NotImplementedError
        If a weighting, `gal_cut` or `mmax != lmax` is requested, or if the transforms of s2fft
        are used with `lmax < 2 * nside - 1`.
    TypeError
        If the maps are complex.
    ValueError
        If `pol=True` and the stack has more than 3 maps.

    See Also
    --------
    - [`alm2map`][jax_healpy.alm2map]: the inverse transform.
    - [`map2alm_spin`][jax_healpy.map2alm_spin]: the transform of the maps of a spin-weighted field.
    - [`anafast`][jax_healpy.anafast]: the power spectra of maps.

    Notes
    -----
    Since the bad pixels are an argument, and not a function of the values of the pixels, the
    transform is linear in `maps` for a given `bad_mask`. It can therefore be transposed with
    [`jax.linear_transpose`][], and used as a linear operator by an iterative solver.

    Examples
    --------
    >>> import jax
    >>> import jax_healpy as jhp
    >>> nside = 16
    >>> m = jax.random.normal(jax.random.key(0), (jhp.nside2npix(nside),))
    >>> m = m.at[0].set(jhp.UNSEEN)
    >>> alms = jhp.map2alm(m, bad_mask=jhp.bad_pixel_mask(m))
    >>> alms.shape
    (48, 95)
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

    if bad_mask is not None:
        bad_mask = jnp.broadcast_to(jnp.asarray(bad_mask), maps.shape)

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
        if mmax is not None and mmax != target_lmax:
            raise NotImplementedError('Specifying mmax != lmax is not implemented.')

        if n == 2:
            # Q, U -> E, B (spin-2 only, no temperature)
            alm_E, alm_B = _map2alm_core(
                maps=[maps[0], maps[1]],
                lmax=target_lmax,
                mmax=mmax,
                iter=iter,
                method=method,
                spin=2,
                bad_mask=None if bad_mask is None else [bad_mask[0], bad_mask[1]],
                healpy_ordering=healpy_ordering,
            )
            out = [alm_E, alm_B]
        else:
            # leading map is temperature (n == 1 -> I; n == 3 -> I, Q, U)
            alm_T = _map2alm_core(
                maps=maps[0],
                lmax=target_lmax,
                mmax=mmax,
                iter=iter,
                method=method,
                spin=0,
                bad_mask=None if bad_mask is None else bad_mask[0],
                healpy_ordering=healpy_ordering,
            )
            out = [alm_T]
            if n == 3:
                alm_E, alm_B = _map2alm_core(
                    maps=[maps[1], maps[2]],
                    lmax=target_lmax,
                    mmax=mmax,
                    iter=iter,
                    method=method,
                    spin=2,
                    bad_mask=None if bad_mask is None else [bad_mask[1], bad_mask[2]],
                    healpy_ordering=healpy_ordering,
                )
                out += [alm_E, alm_B]

        # A single I map collapses to a bare temperature alm (scalar semantics);
        # otherwise return a stacked (n, ...) array, matching healpy (not a tuple).
        return out[0] if n == 1 else jnp.stack(out, axis=0)

    # Handle batched scalar input (pol=False, shape (n, npix))
    if maps.ndim > 1:
        return jax.vmap(map2alm, in_axes=(0,) + 10 * (None,) + (None if bad_mask is None else 0,))(
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
            bad_mask,
        )

    nside = npix2nside(maps.shape[-1])
    target_lmax = _resolve_lmax(nside, lmax)

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
        bad_mask=bad_mask,
        healpy_ordering=healpy_ordering,
    )

    return flm


@jax.jit(static_argnames=['mmax', 'healpy_ordering'])
def almxfl(alm: ArrayLike, fl: ArrayLike, mmax: int | None = None, healpy_ordering: bool = False) -> ArrayLike:
    r"""Multiply spherical harmonic coefficients by a function of the multipole.

    $$a'_{\ell m} = f_\ell \, a_{\ell m},$$

    for instance to convolve a map with a beam, of window function
    [`gauss_beam`][jax_healpy.gauss_beam], or to filter it.

    !!! note "Difference from healpy"
        - The coefficients are in the layout of s2fft, unless `healpy_ordering=True` (see the
          [layout of the coefficients](#layout-of-the-coefficients)).
        - `fl` has `lmax + 1` elements, whereas healpy accepts fewer and sets the coefficients
          of the multipoles beyond them to zero.
        - `mmax` is ignored, and the argument `inplace` does not exist.

    Parameters
    ----------
    alm : ArrayLike
        The coefficients, of shape `(lmax + 1, 2 * lmax + 1)` in the layout of s2fft or
        `(nalm,)` in that of healpy.
    fl : ArrayLike
        The function $f_\ell$, of shape `(lmax + 1,)`.
    mmax : int, optional
        Ignored: the coefficients have `mmax = lmax`.
    healpy_ordering : bool, optional
        If True, the coefficients are in the layout of healpy, and otherwise in that of s2fft.

    Returns
    -------
    jax.Array
        The coefficients $a'_{\ell m}$, in the layout of `alm`.

    Examples
    --------
    Convolve a map with a Gaussian beam of 1 degree:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> import jax_healpy as jhp
    >>> lmax = 64
    >>> alms = jhp.synalm(jax.random.key(0), jnp.ones(lmax + 1))
    >>> bl = jhp.gauss_beam(np.radians(1.0), lmax=lmax)
    >>> jhp.almxfl(alms, bl).shape
    (65, 129)
    """
    alm = jnp.asarray(alm)
    fl = jnp.asarray(fl)

    # Determine lmax from fl
    lmax = fl.shape[0] - 1

    if healpy_ordering:
        # healpy 1D format: each coefficient is multiplied by the filter at its degree
        ell, _ = _alm_hp_indices(lmax)
        return alm * fl[ell]
    else:
        # s2fft 2D format: directly broadcast fl over m dimension
        fl_2d = fl[..., None]
        return alm * fl_2d


@jax.jit(static_argnames=['lmax', 'pol'])
def gauss_beam(fwhm: float, lmax: int = 512, pol: bool = False) -> ArrayLike:
    r"""Return the window function of a circular Gaussian beam.

    The window function of a Gaussian beam of standard deviation
    $\sigma = \mathrm{FWHM} / \sqrt{8 \ln 2}$ is

    $$B_\ell = \exp\left(-\frac{\ell (\ell + 1) \sigma^2}{2}\right):$$

    the beam multiplies the coefficients of the sky by $B_\ell$, and its power spectrum by
    $B_\ell^2$. With `pol=True`, the window functions of a perfectly co-polarized beam
    ([Challinor et al. 2000](https://arxiv.org/abs/astro-ph/0008228)) are added: $B_\ell e^{2\sigma^2}$
    for E and B, and $B_\ell e^{\sigma^2}$ for TE.

    Parameters
    ----------
    fwhm : float
        The full width at half maximum of the beam, in radians. It can be a JAX array, traced by
        `jax.jit` or differentiated by `jax.grad`.
    lmax : int, optional
        The maximum multipole $\ell_\mathrm{max}$.
    pol : bool, optional
        Whether to return the window functions of the polarization too.

    Returns
    -------
    jax.Array
        The window function, of shape `(lmax + 1,)`, or, if `pol=True`, the window functions of
        T, E, B and TE, of shape `(lmax + 1, 4)`.

    Examples
    --------
    >>> import numpy as np
    >>> import jax_healpy as jhp
    >>> jhp.gauss_beam(np.radians(1.0), lmax=2048).shape
    (2049,)
    >>> jhp.gauss_beam(np.radians(1.0), lmax=2048, pol=True).shape
    (2049, 4)
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
def alm2cl(
    alms1: ArrayLike | list[ArrayLike],
    alms2: ArrayLike | list[ArrayLike] | None = None,
    lmax: int | None = None,
    mmax: int | None = None,
    lmax_out: int | None = None,
    nspec: int | None = None,
    healpy_ordering: bool = False,
) -> ArrayLike:
    r"""Compute the auto- and cross-spectra of spherical harmonic coefficients.

    The cross-spectrum of the coefficients $a_{\ell m}$ and $b_{\ell m}$ is

    $$C_\ell = \frac{1}{2\ell + 1} \sum_{m=-\ell}^{\ell} \mathrm{Re}\left(a_{\ell m} b^*_{\ell m}\right),$$

    and the auto-spectrum that of $a_{\ell m}$ with itself. A list of $n$ sets of coefficients
    gives their $n(n+1)/2$ auto- and cross-spectra, ordered by diagonal as in healpy: for three
    sets, 11, 22, 33, 12, 23, 13.

    !!! note "Difference from healpy"
        - The coefficients are in the layout of s2fft, unless `healpy_ordering=True` (see the
          [layout of the coefficients](#layout-of-the-coefficients)).
        - Several sets of coefficients are passed as a list or a tuple: a stacked array is read
          as a single set, and raises an error. healpy accepts both.
        - With a list of sets, `alms2` is ignored: the cross-spectra of two lists are not
          implemented.

    Parameters
    ----------
    alms1 : ArrayLike or list[ArrayLike]
        A set of coefficients, of shape `(lmax + 1, 2 * lmax + 1)` in the layout of s2fft or
        `(nalm,)` in that of healpy, or a list of sets.
    alms2 : ArrayLike, optional
        A second set of coefficients, for the cross-spectrum with `alms1`, when `alms1` is a
        single set. If None, the auto-spectra of `alms1`.
    lmax : int, optional
        The maximum multipole $\ell_\mathrm{max}$ of the coefficients. If None, it is inferred
        from their shape.
    mmax : int, optional
        Ignored: the coefficients have `mmax = lmax`.
    lmax_out : int, optional
        The maximum multipole of the spectra. If None, `lmax`.
    nspec : int, optional
        The number of spectra to return, the first ones, for a list of sets of coefficients. For
        a single set, as in healpy, the number of multipoles of the spectrum. If None, all.
    healpy_ordering : bool, optional
        If True, the coefficients are in the layout of healpy, and otherwise in that of s2fft.

    Returns
    -------
    jax.Array
        The spectrum, of shape `(lmax_out + 1,)`, or the stack of the spectra, of shape
        `(n(n+1)/2, lmax_out + 1)`.

    See Also
    --------
    - [`anafast`][jax_healpy.anafast]: the power spectra of maps.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> lmax = 16
    >>> cl = 1 / (jnp.arange(lmax + 1) + 1)
    >>> alms1 = jhp.synalm(jax.random.key(1), cl)
    >>> alms2 = jhp.synalm(jax.random.key(2), cl)
    >>> jhp.alm2cl(alms1).shape
    (17,)
    >>> jhp.alm2cl(alms1, alms2).shape
    (17,)

    The 3 spectra 11, 22 and 12 of two sets of coefficients:

    >>> jhp.alm2cl([alms1, alms2]).shape
    (3, 17)
    """
    # Handle multiple alms case
    if isinstance(alms1, (list, tuple)):
        alms1_list = [jnp.asarray(a) for a in alms1]
        n_alms = len(alms1_list)

        # Infer lmax from first alm if not provided
        if lmax is None:
            lmax = _lmax_from_alm(alms1_list[0], healpy_ordering)

        L = lmax + 1

        # Convert to s2fft format if needed
        if healpy_ordering:
            alms1_list = [_alm_hp_to_2d(a, lmax) for a in alms1_list]

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
        lmax = _lmax_from_alm(alms1, healpy_ordering)

    L = lmax + 1

    # Convert to s2fft format if needed
    if healpy_ordering:
        alms1 = _alm_hp_to_2d(alms1, lmax)

    if alms2 is None:
        # Auto-spectrum
        cl = _compute_cl(alms1, L, None)
    else:
        # Cross-spectrum
        alms2 = jnp.asarray(alms2)
        if healpy_ordering:
            alms2 = _alm_hp_to_2d(alms2, lmax)
        cl = _compute_cl(alms1, L, alms2)

    # Truncate multipoles to lmax_out if requested (matches healpy)
    if lmax_out is not None:
        cl = cl[: lmax_out + 1]

    # Apply nspec if requested (healpy slices the multipole axis for a single alm)
    if nspec is not None:
        cl = cl[:nspec]

    return cl


@_with_default_method
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
    method: str | None = None,
) -> ArrayLike | tuple[ArrayLike, ...]:
    r"""Compute the power spectra of a map, or of the T, Q, U maps.

    The coefficients of the maps are computed by [`map2alm`][jax_healpy.map2alm], and their
    spectra as by [`alm2cl`][jax_healpy.alm2cl]:

    $$C_\ell = \frac{1}{2\ell + 1} \sum_{m=-\ell}^{\ell} \mathrm{Re}\left(a_{\ell m} b^*_{\ell m}\right),$$

    where $b_{\ell m}$ are the coefficients of `map2`, or those of `map1` for the auto-spectra.
    With `pol=True`, the T, Q, U maps give the six spectra TT, EE, BB, TE, EB, TB, where TE is
    the cross-spectrum of the T coefficients of `map1` and of the E ones of `map2`.

    !!! note "Difference from healpy"
        - With `pol=True`, a stack of two maps is read as Q, U and gives the three spectra
          EE, BB, EB. healpy requires the three maps T, Q, U.
        - With `alm=True`, the coefficients are returned in the layout of s2fft (see the
          [layout of the coefficients](#layout-of-the-coefficients)).
        - The weights (`use_weights`, `use_pixel_weights`, `datapath`), `gal_cut` and
          `mmax != lmax` raise a `NotImplementedError`.

    Parameters
    ----------
    map1 : ArrayLike
        The map, of shape `(npix,)` in the RING scheme, or, with `pol=True`, the stack of the
        T, Q, U maps or of the Q, U maps.
    map2 : ArrayLike, optional
        A second map, or stack of maps, of the shape of `map1`, for the cross-spectra. If None,
        the auto-spectra of `map1`.
    nspec : int, optional
        The number of spectra to return, the first ones, for a stack of maps. For a single map,
        as in healpy, the number of multipoles of the spectrum. If None, all.
    lmax : int, optional
        The maximum multipole $\ell_\mathrm{max}$. If None, `3 * nside - 1`.
    mmax : int, optional
        The maximum order $m_\mathrm{max}$. Only `mmax = lmax` is implemented.
    iter : int, optional
        The number of iterations of [`map2alm`][jax_healpy.map2alm].
    alm : bool, optional
        Whether to return the coefficients of the maps too.
    pol : bool, optional
        If True, a stack of maps is read as T, Q, U or as Q, U. A single map is always a scalar
        field.
    use_weights : bool, optional
        Whether to weight the rings. Only False is implemented.
    datapath : str, optional
        The directory of the pixel weights. Only None is implemented.
    gal_cut : float, optional
        The latitude in degrees: the pixels between `-gal_cut` and `gal_cut` are ignored. Only 0
        is implemented.
    use_pixel_weights : bool, optional
        Whether to weight the pixels. Only False is implemented.
    method : str, optional
        The transform backend, `'ducc0'`, `'jax'` or `'jax_cuda'` (see the
        [transform backends](#transform-backends)). If None, `'ducc0'` on CPU if ducc0 is
        installed, and `'jax'` otherwise.

    Returns
    -------
    cl : jax.Array
        The spectrum, of shape `(lmax + 1,)`, or the stack of the spectra: TT, EE, BB, TE, EB,
        TB, of shape `(6, lmax + 1)`, for T, Q, U maps, and EE, BB, EB, of shape
        `(3, lmax + 1)`, for Q, U maps.
    alm1 : jax.Array
        If `alm=True`, the coefficients of `map1`, in the layout of s2fft.
    alm2 : jax.Array
        If `alm=True` and `map2` is given, the coefficients of `map2`, in the layout of s2fft.

    Raises
    ------
    NotImplementedError
        If a weighting, `gal_cut` or `mmax != lmax` is requested, or if the transforms of s2fft
        are used with `lmax < 2 * nside - 1`.
    ValueError
        If `pol=True` and `map1` is a stack of other than 2 or 3 maps, or `map2` does not have
        the shape of `map1`.

    See Also
    --------
    - [`alm2cl`][jax_healpy.alm2cl]: the power spectra of spherical harmonic coefficients.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> nside = 16
    >>> cl = 1 / (jnp.arange(3 * nside) + 1)
    >>> m = jhp.synfast(jax.random.key(0), cl, nside)
    >>> jhp.anafast(m).shape
    (48,)

    The six spectra of T, Q, U maps:

    >>> tqu = jhp.synfast(jax.random.key(0), [cl, cl, cl, 0.5 * cl], nside, new=True)
    >>> jhp.anafast(tqu).shape
    (6, 48)
    """
    # TODO: add an option expanded_cross_cl that also returns the ET, BE and BT spectra
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
    r"""Smooth spherical harmonic coefficients by a Gaussian beam, or by a window function.

    The coefficients are multiplied by the window function $B_\ell$ of the beam (see
    [`gauss_beam`][jax_healpy.gauss_beam]), or by `beam_window`. With `pol=True`, the E and B
    coefficients of a stack are multiplied by the window function of the polarization,
    $B_\ell e^{2\sigma^2}$.

    !!! note "Difference from healpy"
        - The coefficients are in the layout of s2fft, unless `healpy_ordering=True` (see the
          [layout of the coefficients](#layout-of-the-coefficients)).
        - With `pol=True`, a stack of two sets of coefficients is read as E, B. healpy smooths
          the first one as temperature and the second one as polarization.
        - A new array is returned: the arguments `inplace` and `verbose` do not exist, and
          `mmax` is ignored.

    Parameters
    ----------
    alms : ArrayLike
        The coefficients: one set, of shape `(lmax + 1, 2 * lmax + 1)` in the layout of s2fft or
        `(nalm,)` in that of healpy, or a stack of sets along the first axis (see `pol`).
    fwhm : float, optional
        The full width at half maximum of the Gaussian beam, in radians. A Python or NumPy
        float, not a JAX array: it is a static argument, which cannot be differentiated.
    sigma : float, optional
        The standard deviation of the Gaussian beam, in radians, which overrides `fwhm`. A
        Python or NumPy float, like `fwhm`.
    beam_window : ArrayLike, optional
        The window function $B_\ell$, of shape `(lmax + 1,)`, which overrides `fwhm` and `sigma`
        and is applied to all the sets of coefficients.
    pol : bool, optional
        If True, a stack of 1, 2 or 3 sets of coefficients is read as T; E, B; or T, E, B. A
        single set is always a scalar field.
    mmax : int, optional
        Ignored: the coefficients have `mmax = lmax`.
    healpy_ordering : bool, optional
        If True, the coefficients are in the layout of healpy, and otherwise in that of s2fft.

    Returns
    -------
    jax.Array
        The smoothed coefficients, in the layout and of the shape of `alms`.

    Raises
    ------
    ValueError
        If `pol=True` and `alms` is a stack of more than 3 sets of coefficients.

    See Also
    --------
    - [`smoothing`][jax_healpy.smoothing]: the smoothing of maps.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> import jax_healpy as jhp
    >>> lmax = 16
    >>> alms = jhp.synalm(jax.random.key(0), jnp.ones(lmax + 1))
    >>> jhp.smoothalm(alms, fwhm=np.radians(10.0)).shape
    (17, 33)
    """
    alms = jnp.asarray(alms)

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
        lmax = _lmax_from_alm(alms[0], healpy_ordering)
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
    lmax = _lmax_from_alm(alms, healpy_ordering)
    if beam_window is not None:
        bl = jnp.asarray(beam_window)
    else:
        bl = _compute_beam_window(lmax, fwhm=fwhm, sigma=sigma)

    # Apply smoothing using almxfl
    return almxfl(alms, bl, mmax=mmax, healpy_ordering=healpy_ordering)


@_with_default_method
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
        'method',
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
    method: str | None = None,
) -> ArrayLike:
    r"""Smooth a map, or the T, Q, U maps, by a Gaussian beam, or by a window function.

    The map is transformed by [`map2alm`][jax_healpy.map2alm], its coefficients smoothed by
    [`smoothalm`][jax_healpy.smoothalm] and transformed back by
    [`alm2map`][jax_healpy.alm2map]. The bad pixels, `UNSEEN` or not finite, are set to zero
    before the transform and to [`UNSEEN`][jax_healpy.UNSEEN] in the smoothed map, map by map.
    The monopole and the dipole are not removed.

    !!! note "Difference from healpy"
        - The NaN and infinite pixels are bad pixels too. healpy transforms them, which makes
          the whole smoothed map NaN.
        - With `pol=True`, a stack of two maps is read as Q, U. healpy requires the three maps
          T, Q, U.
        - The weights (`use_weights`, `use_pixel_weights`, `datapath`), `nest=True` and
          `mmax != lmax` raise a `NotImplementedError`.
        - The argument `verbose` does not exist.

    Parameters
    ----------
    map_in : ArrayLike
        The map, of shape `(npix,)` in the RING scheme, or a stack of maps along the first axis
        (see `pol`).
    fwhm : float, optional
        The full width at half maximum of the Gaussian beam, in radians. A Python or NumPy
        float, not a JAX array: it is a static argument, which cannot be differentiated.
    sigma : float, optional
        The standard deviation of the Gaussian beam, in radians, which overrides `fwhm`. A
        Python or NumPy float, like `fwhm`.
    beam_window : ArrayLike, optional
        The window function $B_\ell$, of shape `(lmax + 1,)`, which overrides `fwhm` and `sigma`.
    pol : bool, optional
        If True, a stack of 1, 2 or 3 maps is read as T; Q, U; or T, Q, U, and the E and B
        coefficients are smoothed by the window function of the polarization. A single map is
        always a scalar field.
    iter : int, optional
        The number of iterations of [`map2alm`][jax_healpy.map2alm].
    lmax : int, optional
        The maximum multipole $\ell_\mathrm{max}$ of the transforms. If None, `3 * nside - 1`.
    mmax : int, optional
        The maximum order $m_\mathrm{max}$. Only `mmax = lmax` is implemented.
    use_weights : bool, optional
        Whether to weight the rings. Only False is implemented.
    use_pixel_weights : bool, optional
        Whether to weight the pixels. Only False is implemented.
    datapath : str, optional
        The directory of the pixel weights. Only None is implemented.
    nest : bool, optional
        Whether the map is in the NESTED scheme. Only False is implemented.
    method : str, optional
        The transform backend, `'ducc0'`, `'jax'` or `'jax_cuda'` (see the
        [transform backends](#transform-backends)). If None, `'ducc0'` on CPU if ducc0 is
        installed, and `'jax'` otherwise.

    Returns
    -------
    jax.Array
        The smoothed map, or stack of maps, in the RING scheme.

    Raises
    ------
    NotImplementedError
        If a weighting, `nest=True` or `mmax != lmax` is requested, or if the transforms of
        s2fft are used with `lmax < 2 * nside - 1`.

    See Also
    --------
    - [`smoothalm`][jax_healpy.smoothalm]: the smoothing of spherical harmonic coefficients.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> import jax_healpy as jhp
    >>> nside = 16
    >>> m = jax.random.normal(jax.random.key(0), (jhp.nside2npix(nside),))
    >>> jhp.smoothing(m, fwhm=np.radians(10.0)).shape
    (3072,)
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
    # Remember bad pixels (per map) to zero them before the transform and restore
    # them as UNSEEN in the output.
    bad = bad_pixel_mask(map_in)
    # A single map (1D) is always scalar; pol only applies to a stack of maps.
    pol_active = pol and map_in.ndim == 2

    if pol_active:
        nside = npix2nside(map_in.shape[-1])
        alms = jnp.asarray(
            map2alm(
                map_in, lmax=lmax, mmax=mmax, iter=iter, pol=True, healpy_ordering=False, bad_mask=bad, method=method
            )
        )
        if alms.ndim == 2:  # single component (1 map) -> add the component axis
            alms = alms[None, ...]
        alms_smooth = smoothalm(
            alms, fwhm=fwhm, sigma=sigma, beam_window=beam_window, pol=True, mmax=mmax, healpy_ordering=False
        )
        out = alm2map(alms_smooth, nside=nside, lmax=lmax, mmax=mmax, pol=True, healpy_ordering=False, method=method)
        # A single (1, npix) map collapses to a (npix,) output; align the mask shape.
        return jnp.where(bad.reshape(out.shape), UNSEEN, out)

    # Scalar branch (single map); bad already matches the output shape.
    alms = map2alm(
        map_in, lmax=lmax, mmax=mmax, iter=iter, pol=False, healpy_ordering=False, bad_mask=bad, method=method
    )
    alms_smooth = smoothalm(
        alms, fwhm=fwhm, sigma=sigma, beam_window=beam_window, pol=False, mmax=mmax, healpy_ordering=False
    )
    out = alm2map(
        alms_smooth, nside=npix2nside(map_in.shape[-1]), lmax=lmax, mmax=mmax, healpy_ordering=False, method=method
    )
    return jnp.where(bad, UNSEEN, out)


@_checks_spectra(lambda arguments: arguments['lmax'])
@jax.jit(static_argnames=['lmax', 'mmax', 'new', 'healpy_ordering'])
def synalm(
    prng_key: PRNGKeyArray,
    cls: ArrayLike | tuple[ArrayLike, ...],
    lmax: int | None = None,
    mmax: int | None = None,
    new: bool = False,
    healpy_ordering: bool = False,
) -> ArrayLike:
    r"""Draw Gaussian spherical harmonic coefficients of given power spectra.

    For a single spectrum $C_\ell$, the coefficients $a_{\ell 0}$ are real, drawn from
    $\mathcal{N}(0, C_\ell)$, and the real and imaginary parts of the $a_{\ell m}$ of $m > 0$ are
    drawn from $\mathcal{N}(0, C_\ell / 2)$. Those of $m < 0$ follow from the reality of the
    field: $a_{\ell, -m} = (-1)^m a_{\ell m}^*$.

    For several fields, such as T, E, B, the auto- and cross-spectra make the covariance matrix
    $\mathbf{C}_\ell$ of the fields at each multipole: its Cholesky factor $\mathbf{L}_\ell$,
    such that $\mathbf{C}_\ell = \mathbf{L}_\ell \mathbf{L}_\ell^\top$, correlates independent
    draws of unit variance, as in healpy. The cross-spectra, such as TE, are therefore
    reproduced on average.

    !!! note "Difference from healpy"
        - The random numbers are drawn from the JAX PRNG key `prng_key`, the first argument,
          instead of the global state of NumPy: the coefficients differ from those of healpy
          for the same seed, but follow the same statistics.
        - The coefficients are in the layout of s2fft, unless `healpy_ordering=True` (see the
          [layout of the coefficients](#layout-of-the-coefficients)).
        - `mmax != lmax` raises a `NotImplementedError`, and the argument `verbose` does not
          exist.

    Parameters
    ----------
    prng_key : PRNGKeyArray
        The JAX PRNG key, such as `jax.random.key(0)`.
    cls : ArrayLike or tuple[ArrayLike, ...]
        The spectrum, of shape `(lmax + 1,)`, or a sequence of the $n(n+1)/2$ auto- and
        cross-spectra of $n$ fields, in the order set by `new`. A sequence of 4 spectra is
        TT, EE, BB, TE (`new=True`) or TT, TE, EE, BB (`new=False`), the others being zero. A
        cross-spectrum can be None, for zero.
    lmax : int, optional
        The maximum multipole $\ell_\mathrm{max}$. If None, that of the longest spectrum.
    mmax : int, optional
        The maximum order $m_\mathrm{max}$. Only `mmax = lmax` is implemented.
    new : bool, optional
        If True, the spectra are ordered by diagonal, as TT, EE, BB, TE, EB, TB, and otherwise by
        row, as TT, TE, TB, EE, EB, BB. Ignored for a single spectrum.
    healpy_ordering : bool, optional
        If True, the coefficients are returned in the layout of healpy, and otherwise in that of
        s2fft.

    Returns
    -------
    jax.Array
        The coefficients, of shape `(lmax + 1, 2 * lmax + 1)` in the layout of s2fft or
        `(nalm,)` in that of healpy, or, for several fields, their stack along the first axis.

    Raises
    ------
    ValueError
        If a spectrum is negative, or if the covariance matrix of the fields is not positive
        semi-definite at a multipole, that is, if a cross-spectrum exceeds the geometric mean
        of the auto-spectra in absolute value. When the spectra are traced, for instance by
        `jax.jit`, the coefficients are NaN instead.
    NotImplementedError
        If `mmax != lmax`.

    See Also
    --------
    - [`synfast`][jax_healpy.synfast]: the maps drawn from power spectra.

    Notes
    -----
    The coefficients can be differentiated with respect to the spectra, including where the
    spectra vanish, as the monopole and the dipole of the CMB: there, the derivative is zero,
    whereas that of the square root of the spectrum is infinite, which does not affect the
    derivatives with respect to an amplitude or to the tensor-to-scalar ratio. For perfectly
    correlated fields, the derivative depends on the regularization of the Cholesky
    factorization and is not meaningful.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> cl = 1 / (jnp.arange(65) + 10) ** 2
    >>> jhp.synalm(jax.random.key(0), cl).shape
    (65, 129)

    The T, E, B coefficients of the spectra TT, EE, BB and TE:

    >>> jhp.synalm(jax.random.key(0), [cl, 0.1 * cl, 0.01 * cl, 0.2 * cl], new=True).shape
    (3, 65, 129)
    """
    warn_if_not_x64()
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
        real_dtype = _spectra_real_dtype(*[c for c in cls_list if c is not None])
        cov = _spectra_covariance(cls_list, lmax, new)  # (L, n, n)
        n = cov.shape[-1]

        # Per-multipole Cholesky factor, R @ R.T == cov, which mixes n independent unit-variance
        # draws, as in healpy: T is drawn first, then E is correlated with T, etc.
        factor = _cholesky_psd(cov)  # (L, n, n)
        keys = jax.random.split(prng_key, n)
        unit = jnp.stack([_draw_unit_alms(keys[i], L, real_dtype) for i in range(n)])  # (n, L, L), m >= 0
        alms = _alms_with_negative_m(jnp.einsum('lij,jlm->ilm', factor, unit))  # (n, L, 2L-1)

        if healpy_ordering:
            alms = _alm_2d_to_hp(alms, lmax)
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
        alms = _alm_2d_to_hp(alms, lmax)

    return alms


def pixwin(nside: int, pol: bool = False, lmax: int | None = None, datapath: str | None = None) -> ArrayLike:
    r"""Return the pixel window function: not implemented.

    The pixel window function $W_\ell$ of a resolution describes the smoothing of a map by its
    pixels: the pixelization multiplies the coefficients of the sky by $W_\ell$.

    !!! note "Difference from healpy"
        This function raises a `NotImplementedError`: healpy reads the window functions from
        data files, which jax-healpy does not ship. The window functions of healpy can be used
        instead, as in `jnp.asarray(healpy.pixwin(nside))`.

    Parameters
    ----------
    nside : int
        The resolution of the maps.
    pol : bool, optional
        Whether to return the window function of the polarization too.
    lmax : int, optional
        The maximum multipole $\ell_\mathrm{max}$. If None, `3 * nside - 1`.
    datapath : str, optional
        The directory of the data files of the window functions.

    Returns
    -------
    jax.Array or tuple[jax.Array, jax.Array]
        The window function of the temperature, or, if `pol=True`, those of the temperature and
        of the polarization.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError(
        'pixwin is not yet implemented in jax-healpy. '
        'Pixel window data files need to be integrated. '
        'Consider using healpy.pixwin() and converting to JAX arrays as a workaround.'
    )


@_checks_spectra(
    lambda arguments: (
        3 * arguments['nside'] - 1 if arguments['lmax'] is None or arguments['lmax'] < 0 else arguments['lmax']
    )
)
@_with_default_method
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
    method: str | None = None,
) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
    r"""Draw a Gaussian map, or the T, Q, U maps, of given power spectra.

    The coefficients are drawn by [`synalm`][jax_healpy.synalm] and transformed by
    [`alm2map`][jax_healpy.alm2map], after an optional Gaussian smoothing. With `pol=True`, the
    spectra of T, E, B give the T, Q, U maps.

    !!! note "Difference from healpy"
        - The random numbers are drawn from the JAX PRNG key `prng_key`, the first argument,
          instead of the global state of NumPy: the maps differ from those of healpy for the
          same seed, but follow the same statistics.
        - With `alm=True`, the coefficients are returned in the layout of s2fft (see the
          [layout of the coefficients](#layout-of-the-coefficients)).
        - `pixwin=True` and `mmax != lmax` raise a `NotImplementedError`, and the argument
          `verbose` does not exist.

    Parameters
    ----------
    prng_key : PRNGKeyArray
        The JAX PRNG key, such as `jax.random.key(0)`.
    cls : ArrayLike or tuple[ArrayLike, ...]
        The spectrum, or a sequence of the auto- and cross-spectra of T, E, B, in the order set
        by `new` (see [`synalm`][jax_healpy.synalm]).
    nside : int
        The resolution of the map.
    lmax : int, optional
        The maximum multipole $\ell_\mathrm{max}$. If None or negative, as in healpy, that of
        the longest spectrum, at most `3 * nside - 1`.
    mmax : int, optional
        The maximum order $m_\mathrm{max}$. Only `mmax = lmax` is implemented.
    alm : bool, optional
        Whether to return the coefficients too.
    pol : bool, optional
        If True, several spectra are those of T, E, B, which give the T, Q, U maps. If False,
        each field gives a map, transformed as a scalar field. A single spectrum always gives
        a single map.
    pixwin : bool, optional
        Whether to multiply the coefficients by the pixel window function. Only False is
        implemented.
    fwhm : float, optional
        The full width at half maximum of the Gaussian beam that smooths the map, in radians. A
        Python or NumPy float, not a JAX array: it is a static argument, which cannot be
        differentiated.
    sigma : float, optional
        The standard deviation of the Gaussian beam, in radians, which overrides `fwhm`. A
        Python or NumPy float, like `fwhm`.
    new : bool, optional
        If True, the spectra are ordered by diagonal, as TT, EE, BB, TE, EB, TB, and otherwise by
        row, as TT, TE, TB, EE, EB, BB. Ignored for a single spectrum.
    method : str, optional
        The transform backend, `'ducc0'`, `'jax'` or `'jax_cuda'` (see the
        [transform backends](#transform-backends)). If None, `'ducc0'` on CPU if ducc0 is
        installed, and `'jax'` otherwise.

    Returns
    -------
    map : jax.Array
        The map, of shape `(npix,)`, in the RING scheme, or the T, Q, U maps, of shape
        `(3, npix)`.
    alm : jax.Array
        If `alm=True`, the coefficients drawn, in the layout of s2fft.

    Raises
    ------
    ValueError
        If the spectra are not those of a Gaussian field (see [`synalm`][jax_healpy.synalm]).
    NotImplementedError
        If `pixwin=True` or `mmax != lmax`, or if the transforms of s2fft are used with
        `lmax < 2 * nside - 1`.

    See Also
    --------
    - [`synalm`][jax_healpy.synalm]: the coefficients drawn from power spectra.
    - [`anafast`][jax_healpy.anafast]: the power spectra of maps.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> nside = 16
    >>> cl = 1 / (jnp.arange(3 * nside) + 10) ** 2
    >>> jhp.synfast(jax.random.key(0), cl, nside).shape
    (3072,)

    The T, Q, U maps of the spectra TT, EE, BB and TE:

    >>> jhp.synfast(jax.random.key(0), [cl, 0.1 * cl, 0.01 * cl, 0.2 * cl], nside, new=True).shape
    (3, 3072)
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


@_with_default_method
@jax.jit(static_argnames=['spin', 'lmax', 'mmax', 'iter', 'method', 'healpy_ordering'])
def map2alm_spin(
    maps,
    spin: int,
    lmax: int | None = None,
    mmax: int | None = None,
    iter: int = 0,
    method: str | None = None,
    healpy_ordering: bool = False,
    bad_mask: ArrayLike | None = None,
) -> ArrayLike | list[ArrayLike]:
    r"""Compute the gradient and curl coefficients of the two maps of a spin-weighted field.

    The maps $Q$ and $U$ of a field of spin $s$, in the RING scheme, are the real and imaginary
    parts of the field $Q + iU$, whose spin $s$ coefficients ${}_s a_{\ell m}$ give the gradient
    (E) and curl (B) coefficients:

    $$a^E_{\ell m} = -\frac{{}_s a_{\ell m} + (-1)^s \, {}_{-s} a_{\ell m}}{2}, \qquad
    a^B_{\ell m} = i \, \frac{{}_s a_{\ell m} - (-1)^s \, {}_{-s} a_{\ell m}}{2},$$

    the coefficients of spin $-s$ being those of $Q - iU$. For $s = 2$, the Q, U maps of the
    polarization give its E, B coefficients. For $s = 0$, the function is
    [`map2alm`][jax_healpy.map2alm] of a single map.

    !!! note "Difference from healpy"
        - The coefficients are in the layout of s2fft, unless `healpy_ordering=True` (see the
          [layout of the coefficients](#layout-of-the-coefficients)).
        - The arguments `iter`, the number of iterations, and `bad_mask`, the bad pixels, are
          added. With `iter=0`, the default, the coefficients are those of healpy.
        - `spin=0` is accepted, with a single map.
        - `mmax != lmax` raises a `NotImplementedError`.

    Parameters
    ----------
    maps : list[ArrayLike]
        The two maps $Q$ and $U$, of shape `(npix,)`, or, for `spin=0`, a single map.
    spin : int
        The spin $s$ of the field.
    lmax : int, optional
        The maximum multipole $\ell_\mathrm{max}$. If None, `3 * nside - 1`.
    mmax : int, optional
        The maximum order $m_\mathrm{max}$. Only `mmax = lmax` is implemented.
    iter : int, optional
        The number of iterations that refine the coefficients, as in
        [`map2alm`][jax_healpy.map2alm].
    method : str, optional
        The transform backend, `'ducc0'`, `'jax'` or `'jax_cuda'` (see the
        [transform backends](#transform-backends)). If None, `'ducc0'` on CPU if ducc0 is
        installed, and `'jax'` otherwise.
    healpy_ordering : bool, optional
        If True, the coefficients are returned in the layout of healpy, and otherwise in that of
        s2fft.
    bad_mask : ArrayLike, optional
        The mask of the bad pixels, True where the pixel is bad, broadcastable to the shape of
        the stack of the maps. These pixels are set to zero before the transform. If None, all
        the pixels are used.

    Returns
    -------
    list[jax.Array] or jax.Array
        The E and B coefficients, of shape `(lmax + 1, 2 * lmax + 1)` in the layout of s2fft or
        `(nalm,)` in that of healpy, or, for `spin=0`, the coefficients of the map.

    Raises
    ------
    ValueError
        If `spin` is not zero and `maps` is not a list or a tuple of two maps.
    NotImplementedError
        If `mmax != lmax`, or if the transforms of s2fft are used with `lmax < 2 * nside - 1`.
    TypeError
        If the maps are complex.

    See Also
    --------
    - [`alm2map_spin`][jax_healpy.alm2map_spin]: the inverse transform.

    Examples
    --------
    >>> import jax
    >>> import jax_healpy as jhp
    >>> nside = 16
    >>> q, u = jax.random.normal(jax.random.key(0), (2, jhp.nside2npix(nside)))
    >>> alm_e, alm_b = jhp.map2alm_spin([q, u], spin=2)
    >>> alm_e.shape
    (48, 95)
    """
    if spin != 0:
        # For spin != 0, expect list of 2 maps
        if not isinstance(maps, (list, tuple)) or len(maps) != 2:
            raise ValueError(f'For spin={spin}, maps must be a list/tuple of 2 arrays')
        maps = [jnp.asarray(m) for m in maps]
        nside = npix2nside(maps[0].shape[-1])
        if bad_mask is not None:
            bad_mask = list(jnp.broadcast_to(jnp.asarray(bad_mask), (2, maps[0].shape[-1])))
    else:
        maps = jnp.asarray(maps)
        nside = npix2nside(maps.shape[-1])
        if bad_mask is not None:
            bad_mask = jnp.broadcast_to(jnp.asarray(bad_mask), maps.shape)

    lmax = _resolve_lmax(nside, lmax)

    if mmax is not None and mmax != lmax:
        raise NotImplementedError('Specifying mmax != lmax is not implemented.')

    # Call core function
    flm = _map2alm_core(
        maps=maps,
        lmax=lmax,
        mmax=mmax,
        iter=iter,
        method=method,
        spin=spin,
        bad_mask=bad_mask,
        healpy_ordering=healpy_ordering,
    )

    return flm


@_with_default_method
@jax.jit(static_argnames=['nside', 'spin', 'lmax', 'mmax', 'method', 'healpy_ordering'])
def alm2map_spin(
    alms,
    nside: int,
    spin: int,
    lmax: int | None = None,
    mmax: int | None = None,
    method: str | None = None,
    healpy_ordering: bool = False,
) -> ArrayLike | list[ArrayLike]:
    r"""Compute the two maps of a spin-weighted field from its gradient and curl coefficients.

    The gradient (E) and curl (B) coefficients of a field of spin $s$ give its maps $Q$ and $U$,
    in the RING scheme, the real and imaginary parts of the field $Q + iU$ of coefficients
    ${}_s a_{\ell m} = -(a^E_{\ell m} + i a^B_{\ell m})$. For $s = 2$, the E, B coefficients of
    the polarization give its Q, U maps. For $s = 0$, the function is
    [`alm2map`][jax_healpy.alm2map] of a single set of coefficients.

    !!! note "Difference from healpy"
        - The coefficients are in the layout of s2fft, unless `healpy_ordering=True` (see the
          [layout of the coefficients](#layout-of-the-coefficients)).
        - `lmax` is inferred from the shape of the coefficients if it is not given, whereas
          healpy requires it.
        - `spin=0` is accepted, with a single set of coefficients.
        - `mmax != lmax` raises a `NotImplementedError`.

    Parameters
    ----------
    alms : list[ArrayLike] or ArrayLike
        The E and B coefficients, of shape `(lmax + 1, 2 * lmax + 1)` in the layout of s2fft or
        `(nalm,)` in that of healpy, or, for `spin=0`, a single set of coefficients.
    nside : int
        The resolution of the maps.
    spin : int
        The spin $s$ of the field.
    lmax : int, optional
        The maximum multipole $\ell_\mathrm{max}$. If None, it is inferred from the shape of
        `alms`.
    mmax : int, optional
        The maximum order $m_\mathrm{max}$. Only `mmax = lmax` is implemented.
    method : str, optional
        The transform backend, `'ducc0'`, `'jax'` or `'jax_cuda'` (see the
        [transform backends](#transform-backends)). If None, `'ducc0'` on CPU if ducc0 is
        installed, and `'jax'` otherwise.
    healpy_ordering : bool, optional
        If True, the coefficients are in the layout of healpy, and otherwise in that of s2fft.

    Returns
    -------
    list[jax.Array] or jax.Array
        The maps $Q$ and $U$, of shape `(npix,)`, or, for `spin=0`, the map.

    Raises
    ------
    ValueError
        If `spin` is not zero and `alms` is not a list or a tuple of two sets of coefficients.
    NotImplementedError
        If `mmax != lmax`, or if the transforms of s2fft are used with `lmax < 2 * nside - 1`.

    See Also
    --------
    - [`map2alm_spin`][jax_healpy.map2alm_spin]: the inverse transform.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_healpy as jhp
    >>> nside = 16
    >>> lmax = 3 * nside - 1
    >>> alm_e = jhp.synalm(jax.random.key(0), jnp.ones(lmax + 1))
    >>> q, u = jhp.alm2map_spin([alm_e, jnp.zeros_like(alm_e)], nside, spin=2)
    >>> q.shape
    (3072,)
    """
    if spin != 0:
        # For spin != 0, expect list of 2 alms
        if not isinstance(alms, (list, tuple)) or len(alms) != 2:
            raise ValueError(f'For spin={spin}, alms must be a list/tuple of 2 arrays')
        alms = [jnp.asarray(a) for a in alms]

        if lmax is None:
            lmax = _lmax_from_alm(alms[0], healpy_ordering)
    else:
        alms = jnp.asarray(alms)
        if lmax is None:
            lmax = _lmax_from_alm(alms, healpy_ordering)

    # Call core function
    maps = _alm2map_core(
        alms=alms, nside=nside, lmax=lmax, mmax=mmax, method=method, spin=spin, healpy_ordering=healpy_ordering
    )

    return maps
