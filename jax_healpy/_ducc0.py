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

"""Spherical harmonic transforms of ducc0, usable from JAX.

ducc0 runs the transforms on the CPU, on all the physical cores. They are exposed as two JAX primitives,
``synthesis`` (alms -> maps) and ``adjoint_synthesis`` (maps -> alms), that can be compiled, batched
and differentiated like any other JAX operation:

- they are compiled as a call back to the host (``jax.pure_callback``);
- they are linear, so they are their own Jacobian-vector products;
- each one is the transpose of the other, which gives the reverse-mode derivatives;
- batches of small maps are handed to ducc0, which transforms several sets of alms or maps in one
  call, and those of large maps are transformed one after the other.

For the two primitives to be transposes of each other, they do not exchange the usual complex alms
of a real field, of which only m >= 0 is stored, but an equivalent real vector of size
``(lmax + 1)**2``, in which all the coefficients have the same weight:

    [Re a_l0 | sqrt(2) Re a_lm, m > 0 | sqrt(2) Im a_lm, m > 0]

with the (l, m) ordering of healpy in each block. ``alm_to_real`` and ``real_to_alm`` convert between
the two representations. ``alm2map`` and ``map2alm`` are the transforms on the complex alms.
"""

import importlib.util
import os
import sys
from collections.abc import Callable
from functools import cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir, xla
from jaxtyping import Array, Complex, Float

__all__ = [
    'adjoint_synthesis',
    'alm2map',
    'alm_to_real',
    'check_import',
    'is_available',
    'map2alm',
    'real_to_alm',
    'synthesis',
]

_SQRT2 = 2**0.5

# Batches of small maps are transformed in a single call to ducc0, which is 2 to 7 times faster
# than one call per map up to nside = 32. From the nside below, the maps of a batch are transformed
# one after the other instead: a transform already uses all the physical cores, and exchanging one large
# array with the host takes longer than exchanging several small ones (for 16 maps, it is 1.3 times
# faster at nside = 256, and makes no difference at nside = 128).
_SEQUENTIAL_BATCH_NSIDE = 256


@cache
def _physical_core_count() -> int | None:
    """The number of physical cores among the CPUs available to the process, if it can be found.

    It is read from the CPU topology of Linux, in which the hardware threads of a core (with
    simultaneous multithreading, or hyper-threading) share its list of CPUs.
    """
    if not sys.platform.startswith('linux'):
        return None
    cores = set()
    for cpu in os.sched_getaffinity(0):
        topology = f'/sys/devices/system/cpu/cpu{cpu}/topology'
        for name in ('core_cpus_list', 'thread_siblings_list'):
            try:
                with open(f'{topology}/{name}') as file:
                    cores.add(file.read().strip())
                break
            except OSError:
                continue
        else:
            return None
    return len(cores) or None


@cache
def _thread_count() -> int:
    """The number of threads of the transforms of ducc0.

    By default, one per physical core: the hardware threads of a core compete for its floating-point
    units and caches, and using them makes the transforms up to twice slower. If the environment
    variable ``DUCC0_NUM_THREADS`` or ``OMP_NUM_THREADS`` is set, it sizes the pool of threads of
    ducc0, which is then used whole (0).
    """
    if os.environ.get('DUCC0_NUM_THREADS') or os.environ.get('OMP_NUM_THREADS'):
        return 0
    return _physical_core_count() or 0


@cache
def is_available() -> bool:
    """Whether ducc0 is installed."""
    return importlib.util.find_spec('ducc0') is not None


def check_import() -> None:
    """Import ducc0, which is installed, and raise an ``ImportError`` if it cannot be imported.

    A broken installation then fails where a transform is called, with the error of the import,
    instead of in the call back to the host of the compiled function, where the error is buried
    in a ``JaxRuntimeError``.
    """
    try:
        import ducc0  # noqa: F401
    except ImportError as error:
        raise ImportError(
            f'ducc0 is installed but cannot be imported: {error}. Reinstall it, or report the problem; '
            "to compute the transforms with s2fft instead, pass method='jax'."
        ) from error


def alm2map(alms: Complex[Array, '... ncomp nalm'], nside: int, lmax: int, spin: int) -> Float[Array, '... ncomp npix']:
    """Transforms alms in the healpy ordering into maps.

    There is one component for spin 0, and two for the other spins: the gradient (E) and curl (B)
    alms, transformed into the two maps of the spin field (Q and U for spin 2).
    """
    return synthesis(alm_to_real(alms, lmax), nside=nside, lmax=lmax, spin=spin)


def map2alm(
    maps: Float[Array, '... ncomp npix'], nside: int, lmax: int, spin: int, iter: int
) -> Complex[Array, '... ncomp nalm']:
    """Transforms maps into alms in the healpy ordering, as healpy does.

    The pixels have the same weight, 4 pi / npix, and the alms are refined by ``iter`` Jacobi
    iterations: the residual between the maps and those synthesized from the alms is analysed in turn.
    """
    keywords = {'nside': nside, 'lmax': lmax, 'spin': spin}
    weight = 4 * np.pi / maps.shape[-1]
    x = weight * adjoint_synthesis(maps, **keywords)
    for _ in range(iter):
        x = x + weight * adjoint_synthesis(maps - synthesis(x, **keywords), **keywords)
    return real_to_alm(x, lmax)


def alm_to_real(alms: Complex[Array, '... nalm'], lmax: int) -> Float[Array, '... nreal']:
    """Converts alms in the healpy ordering (m >= 0) into their real representation."""
    positive_m = alms[..., lmax + 1 :] * _SQRT2
    return jnp.concatenate([alms[..., : lmax + 1].real, positive_m.real, positive_m.imag], axis=-1)


def real_to_alm(x: Float[Array, '... nreal'], lmax: int) -> Complex[Array, '... nalm']:
    """Converts the real representation of alms into alms in the healpy ordering (m >= 0)."""
    nalm = (lmax + 1) * (lmax + 2) // 2
    positive_m = (x[..., lmax + 1 : nalm] + 1j * x[..., nalm:]) / _SQRT2
    return jnp.concatenate([x[..., : lmax + 1].astype(positive_m.dtype), positive_m], axis=-1)


def synthesis(
    x: Float[Array, '... ncomp nreal'], *, nside: int, lmax: int, spin: int
) -> Float[Array, '... ncomp npix']:
    """Transforms alms, in their real representation, into maps."""
    return _synthesis_p.bind(x, nside=nside, lmax=lmax, spin=spin)


def adjoint_synthesis(
    maps: Float[Array, '... ncomp npix'], *, nside: int, lmax: int, spin: int
) -> Float[Array, '... ncomp nreal']:
    """Applies the transpose of ``synthesis`` to maps: sums of the pixels times the conjugate harmonics."""
    return _adjoint_synthesis_p.bind(maps, nside=nside, lmax=lmax, spin=spin)


# ---------------------------------------------------------------------------------------------------
# Host side: numpy arrays in, numpy arrays out
# ---------------------------------------------------------------------------------------------------


@cache
def _geometry(nside: int) -> dict[str, np.ndarray]:
    """Colatitude, number of pixels, first azimuth and first pixel index of the HEALPix rings."""
    import ducc0

    return ducc0.healpix.Healpix_Base(nside, 'RING').sht_info()


def _synthesis_host(x: np.ndarray, *, nside: int, lmax: int, spin: int) -> np.ndarray:
    import ducc0

    x = np.asarray(x)
    batch_shape, (ncomp, nreal) = x.shape[:-2], x.shape[-2:]
    npix = 12 * nside**2
    x = x.reshape(-1, ncomp, nreal)
    if x.shape[0] == 0:
        return np.zeros((*batch_shape, ncomp, npix), dtype=x.dtype)

    nalm = (lmax + 1) * (lmax + 2) // 2
    alms = np.empty((x.shape[0], ncomp, nalm), dtype=np.result_type(x.dtype, np.complex64))
    alms[..., : lmax + 1] = x[..., : lmax + 1]
    np.multiply(x[..., lmax + 1 : nalm], 1 / _SQRT2, out=alms.real[..., lmax + 1 :])
    np.multiply(x[..., nalm:], 1 / _SQRT2, out=alms.imag[..., lmax + 1 :])

    maps = ducc0.sht.synthesis(alm=alms, lmax=lmax, mmax=lmax, spin=spin, nthreads=_thread_count(), **_geometry(nside))
    return maps.reshape(*batch_shape, ncomp, npix)


def _adjoint_synthesis_host(maps: np.ndarray, *, nside: int, lmax: int, spin: int) -> np.ndarray:
    import ducc0

    maps = np.asarray(maps)
    batch_shape, (ncomp, npix) = maps.shape[:-2], maps.shape[-2:]
    nreal = (lmax + 1) ** 2
    maps = maps.reshape(-1, ncomp, npix)
    if maps.shape[0] == 0:
        return np.zeros((*batch_shape, ncomp, nreal), dtype=maps.dtype)

    alms = ducc0.sht.adjoint_synthesis(
        map=maps, lmax=lmax, mmax=lmax, spin=spin, nthreads=_thread_count(), **_geometry(nside)
    )

    nalm = alms.shape[-1]
    x = np.empty((maps.shape[0], ncomp, nreal), dtype=maps.dtype)
    x[..., : lmax + 1] = alms[..., : lmax + 1].real
    np.multiply(alms.real[..., lmax + 1 :], _SQRT2, out=x[..., lmax + 1 : nalm])
    np.multiply(alms.imag[..., lmax + 1 :], _SQRT2, out=x[..., nalm:])
    return x.reshape(*batch_shape, ncomp, nreal)


def _call_with_float32_pairs(host_function: Callable, x: np.ndarray, **params: int) -> np.ndarray:
    """Call a host function on a double precision array given, and returned, as pairs of float32."""
    x = np.asarray(x).view(np.float64)[..., 0]
    out = np.ascontiguousarray(host_function(x, **params))
    return out.view(np.float32).reshape(*out.shape, 2)


# ---------------------------------------------------------------------------------------------------
# JAX side: the two primitives
# ---------------------------------------------------------------------------------------------------


def _output_type(x: ShapedArray, name: str, in_size: int, out_size: int, spin: int) -> ShapedArray:
    ncomp = 1 if spin == 0 else 2
    if x.ndim < 2 or x.shape[-2:] != (ncomp, in_size):
        raise TypeError(f'{name} expects an operand of shape (..., {ncomp}, {in_size}) for spin {spin}, got {x.shape}.')
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError(f'{name} expects a real operand, got {x.dtype}.')
    return ShapedArray((*x.shape[:-1], out_size), x.dtype)


def _synthesis_abstract_eval(x: ShapedArray, *, nside: int, lmax: int, spin: int) -> ShapedArray:
    return _output_type(x, 'synthesis', (lmax + 1) ** 2, 12 * nside**2, spin)


def _adjoint_synthesis_abstract_eval(maps: ShapedArray, *, nside: int, lmax: int, spin: int) -> ShapedArray:
    return _output_type(maps, 'adjoint_synthesis', 12 * nside**2, (lmax + 1) ** 2, spin)


def _define_primitive(name: str, host_function: Callable, abstract_eval: Callable) -> Primitive:
    primitive = Primitive(name)
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(abstract_eval)

    def callback(x: Array, **params: int) -> Array:
        out = abstract_eval(ShapedArray(x.shape, x.dtype), **params)
        if x.dtype == np.float64:
            # Double precision arrays are exchanged with the host as pairs of single precision
            # numbers, with the same bits: JAX converts the arguments and the results of the
            # callback in the thread that runs it, where 64-bit precision may be disabled although
            # it is enabled in the thread that called the transform (it is the case when
            # jax.enable_x64 is used as a context manager). The arrays would be truncated to single
            # precision without notice, or rejected.
            result = jax.pure_callback(
                partial(_call_with_float32_pairs, host_function, **params),
                jax.ShapeDtypeStruct((*out.shape, 2), np.float32),
                lax.bitcast_convert_type(x, np.float32),
            )
            return lax.bitcast_convert_type(result, np.float64)
        return jax.pure_callback(partial(host_function, **params), jax.ShapeDtypeStruct(out.shape, out.dtype), x)

    mlir.register_lowering(primitive, mlir.lower_fun(callback, multiple_results=False))

    def batching_rule(batched_args: tuple[Array], batch_dims: tuple[int], **params: int) -> tuple[Array, int]:
        (x,), (batch_dim,) = batched_args, batch_dims
        x = jnp.moveaxis(x, batch_dim, 0)
        if params['nside'] >= _SEQUENTIAL_BATCH_NSIDE:
            return lax.map(lambda item: primitive.bind(item, **params), x), 0
        # The primitive accepts leading batch dimensions, that ducc0 transforms in a single call.
        return primitive.bind(x, **params), 0

    batching.primitive_batchers[primitive] = batching_rule
    return primitive


_synthesis_p = _define_primitive('ducc0_synthesis', _synthesis_host, _synthesis_abstract_eval)
_adjoint_synthesis_p = _define_primitive(
    'ducc0_adjoint_synthesis', _adjoint_synthesis_host, _adjoint_synthesis_abstract_eval
)

# Each primitive is linear, and the transpose of the other one.
ad.deflinear2(_synthesis_p, lambda cotangent, x, **params: [_adjoint_synthesis_p.bind(cotangent, **params)])
ad.deflinear2(_adjoint_synthesis_p, lambda cotangent, maps, **params: [_synthesis_p.bind(cotangent, **params)])
