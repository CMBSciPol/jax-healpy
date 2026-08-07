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

"""Differentiable HEALPix mask apodization (pure ``jax_healpy``).

A binary mask rings in harmonic space, so before computing masked-sky power spectra the mask is
tapered to zero over an edge band. This reimplements NaMaster's **C2** window
``f = (1 - cos(pi x)) / 2`` with ``x = sqrt((1 - cos d) / (1 - cos theta*))``, where ``d`` is the
great-circle distance to the nearest masked pixel (capped at the apodization scale ``theta*``).
The distance is obtained by a HEALPix **grassfire** (iterated ``min`` over true-neighbour
separations), so the whole pipeline is ``jnp`` + ``lax.scan`` and differentiable.

Two layers are exposed:

* :func:`apodize` -- the fixed C2 window (drop-in NaMaster equivalent, ``theta*`` static).
* :func:`grassfire_distance` (geometry, static iteration count from a *max* scale) +
  :func:`apodize_profile` (the taper, **differentiable in** ``theta_star_deg``). Splitting them
  lets the apodization scale enter as a traced parameter for the optimizable-apodization research
  layer without retracing the grassfire sweep count.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import jax_healpy as jhp

__all__ = ['apodize', 'grassfire_distance', 'apodize_profile']


def grassfire_distance(binary_mask, *, max_aposize_deg: float) -> Array:
    """Great-circle distance (radians) from each pixel to the nearest masked pixel.

    Computed by a HEALPix grassfire transform: seed the footprint at a large value and the masked
    region at 0, then iterate ``min`` over true-neighbour separations. The number of sweeps is
    **static**, derived from ``max_aposize_deg`` (an upper bound on the apodization scale you intend
    to use), so the taper scale ``theta*`` can later vary as a traced parameter without changing the
    iteration count. Masked pixels stay exactly 0.

    Parameters
    ----------
    binary_mask : array_like
        Binary (0/1) HEALPix mask in RING ordering, shape ``(npix,)``.
    max_aposize_deg : float
        Upper bound (degrees) on the apodization scale; sets the (static) number of grassfire
        sweeps ``ceil(2.5 * max_aposize_deg / resol) + 6``. Must be concrete under ``jit``.

    Returns
    -------
    jax.Array
        Distance field in radians, shape ``(npix,)``; 0 on masked pixels, growing into the footprint.

    Notes
    -----
    The neighbour-separation step materializes a ``(3, 8, npix)`` intermediate, so peak memory grows
    with ``npix`` (~10 GB at ``nside=2048`` in float64); apodize at a coarser ``nside`` if memory is
    tight.
    """
    binary = jnp.asarray(binary_mask)
    if binary.ndim != 1:
        raise ValueError(f'binary_mask must be 1-D (npix,), got shape {binary.shape}.')
    npix = binary.shape[0]
    nside = jhp.npix2nside(npix)

    ipix = jnp.arange(npix)
    neigh = jhp.get_all_neighbours(nside, ipix)
    neigh = neigh if neigh.shape[0] == 8 else neigh.T  # (8, npix), -1 = missing
    vecs = jhp.pix2vec(nside, ipix)
    vecs = vecs if vecs.shape[0] == 3 else vecs.T  # (3, npix)

    nb = jnp.clip(neigh, 0, npix - 1)
    sep = jnp.where(
        neigh >= 0,
        jnp.arccos(jnp.clip((vecs[:, None, :] * vecs[:, nb]).sum(0), -1.0, 1.0)),
        jnp.inf,
    )
    dist = jnp.where(binary <= 0, 0.0, 10.0)
    # number of grassfire sweeps to cover the max scale (static: nside known, max_aposize_deg concrete)
    resol = float(np.sqrt(4.0 * np.pi / npix))  # == healpy.nside2resol(nside)
    niter = int(np.ceil(2.5 * float(np.deg2rad(max_aposize_deg)) / resol)) + 6
    dist, _ = jax.lax.scan(lambda d, _: (jnp.minimum(d, (d[nb] + sep).min(0)), None), dist, None, length=niter)
    return dist


def apodize_profile(dist, theta_star_deg, *, apotype: str = 'C2') -> Array:
    """Apply the C2 taper to a grassfire distance field. **Differentiable in** ``theta_star_deg``.

    Parameters
    ----------
    dist : array_like
        Distance-to-nearest-masked-pixel field in radians (see :func:`grassfire_distance`); 0 on
        masked pixels.
    theta_star_deg : float or scalar array
        Apodization scale ``theta*`` in degrees. Enters only via ``clip(dist, 0, theta*)`` and the
        ``x`` normalization, so gradients flow through it (the grassfire sweep count does not depend
        on it).
    apotype : str, default='C2'
        Apodization window. Only ``'C2'`` is implemented.

    Returns
    -------
    jax.Array
        Apodized mask in ``[0, 1]``, same shape as ``dist``. Exactly 0 wherever ``dist == 0`` (the
        masked region), since ``x(0) = 0`` gives ``f = 0`` exactly.
    """
    if apotype != 'C2':
        raise NotImplementedError(f'apotype={apotype!r} not implemented; only "C2" is supported.')
    theta_star = jnp.deg2rad(theta_star_deg)
    d = jnp.clip(dist, 0.0, theta_star)
    u = (1 - jnp.cos(d)) / (1 - jnp.cos(theta_star))
    # Safe sqrt: ``u == 0`` on masked pixels (dist == 0), where ``d/dtheta* sqrt(u) = 0/0`` is NaN.
    # The double-``where`` keeps the value (x == 0 there) and the gradient finite (0 on that branch).
    safe_u = jnp.where(u > 0, u, 1.0)
    x = jnp.clip(jnp.where(u > 0, jnp.sqrt(safe_u), 0.0), 0.0, 1.0)
    return (1 - jnp.cos(jnp.pi * x)) / 2


def apodize(binary_mask, aposize_deg: float = 1.0, *, apotype: str = 'C2') -> Array:
    """Apodize a binary HEALPix mask with a NaMaster-style C2 window (pure JAX).

    Thin wrapper over :func:`grassfire_distance` + :func:`apodize_profile`, reproducing NaMaster's
    ``mask_apodization(..., apotype='C2')``. Differentiable w.r.t. the input mask values; for a
    differentiable apodization *scale*, call the two-layer API directly.

    Parameters
    ----------
    binary_mask : array_like
        Binary (0/1) HEALPix mask in RING ordering, shape ``(npix,)``.
    aposize_deg : float, default=1.0
        Apodization scale ``theta*`` in degrees. Treated as a static config (sets the number of
        grassfire iterations), so keep it concrete under ``jit``.
    apotype : str, default='C2'
        Apodization window. Only ``'C2'`` is implemented.

    Returns
    -------
    jax.Array
        Apodized mask in ``[0, 1]``, shape ``(npix,)``, ``float``. Exactly 0 wherever the input is 0
        (so ``apodized * map`` carries no outside-footprint pixels).
    """
    dist = grassfire_distance(binary_mask, max_aposize_deg=aposize_deg)
    return apodize_profile(dist, aposize_deg, apotype=apotype)
