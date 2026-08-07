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

"""Masked-sky angular power spectra: mode decoupling + E/B purification (pure ``jax_healpy``).

The public entry point is :func:`anafast_masked`:

* ``mask=None`` -> plain ``jhp.anafast`` (also serves as the *coupled* pseudo-``C_l`` when the
  caller has pre-multiplied the map by the apodized mask, e.g. for forward-model inference).
* ``mask`` given -> masked pseudo-``C_l``, **decoupled** into bandpowers (decoupling is implied by a
  non-None mask; there is no separate ``decouple`` flag). Spin-2 honours ``purify_e``/``purify_b``
  (Smith 2006), removing E->B leakage before estimation.

The MCM depends only on the mask, so it can be precomputed once with
:func:`jax_healpy.pseudo_cl.compute_mcm` and passed back via ``mcm=`` (e.g. frozen for inference).

NaMaster is **not** a runtime dependency -- it is only used as the test oracle.
"""

from __future__ import annotations

import jax.numpy as jnp

import jax_healpy as jhp

from ._mcm import MCM, _window_derivative_alms, compute_mcm
from ._utils import require_x64

__all__ = ['anafast_masked']


# --------------------------------------------------------------------------- #
# Decoupling (invert the binned MCM)                                          #
# --------------------------------------------------------------------------- #
def _decouple_spin0(cl_coupled, M0, B, S):
    return jnp.linalg.solve(B @ M0 @ S, B @ jnp.asarray(cl_coupled))


def _decouple_spin2(cl_EE, cl_BB, EEEE, EEBB, B, S):
    nb = B.shape[0]
    EEEE_b, EEBB_b = B @ EEEE @ S, B @ EEBB @ S
    Mbin = jnp.block([[EEEE_b, EEBB_b], [EEBB_b, EEEE_b]])
    rhs = jnp.concatenate([B @ jnp.asarray(cl_EE), B @ jnp.asarray(cl_BB)])
    dec = jnp.linalg.solve(Mbin, rhs)
    return dec[:nb], dec[nb:]


def _decouple_spin2_pure(cl_EE, cl_BB, EEEE, EEBB, PUREBB, B, S):
    """Decouple a *purified*-B spectrum: the pure-B<-EE leakage block vanishes analytically, so the
    binned system is block-triangular -- ``BB = (B PUREBB S)^-1 B C_tilde^BB`` decouples on its own
    (no E re-injection), and the standard EE row carries the residual EE<-BB term."""
    nb = B.shape[0]
    EEEE_b, EEBB_b, PUREBB_b = B @ EEEE @ S, B @ EEBB @ S, B @ PUREBB @ S
    Mbin = jnp.block([[EEEE_b, EEBB_b], [jnp.zeros_like(EEEE_b), PUREBB_b]])
    rhs = jnp.concatenate([B @ jnp.asarray(cl_EE), B @ jnp.asarray(cl_BB)])
    dec = jnp.linalg.solve(Mbin, rhs)
    return dec[:nb], dec[nb:]


# --------------------------------------------------------------------------- #
# E/B purification (Smith 2006, pure jax_healpy)                              #
# --------------------------------------------------------------------------- #
def _purify_eb(g1, g2, mask, nside, lmax, method, iter=0, task=(False, True)):
    """(E_alm, B_alm) purified a la Smith 2006. task=(purify_E, purify_B).

    ``iter`` is the (fixed, unrolled -> differentiable) refinement count for the spin field
    transforms; the iter=0 default reproduces the NaMaster-validated behaviour. Raising it
    suppresses the ~1% spin-2 transform floor that otherwise limits the residual E->B leakage.
    """
    g1, g2, mask = jnp.asarray(g1), jnp.asarray(g2), jnp.asarray(mask)
    ls = jnp.arange(lmax + 1)
    _, walm0, walm0b = _window_derivative_alms(mask, lmax, method)
    E, Bb = jhp.map2alm_spin([mask * g1, mask * g2], spin=2, lmax=lmax, iter=iter, healpy_ordering=True, method=method)
    alms = [E, Bb]
    walm1 = jnp.zeros_like(walm0)
    w0, w1 = jhp.alm2map_spin([walm0, walm1], nside, spin=1, lmax=lmax, healpy_ordering=True, method=method)
    p = jhp.map2alm_spin(
        [w0 * g1 + w1 * g2, w0 * g2 - w1 * g1], spin=1, lmax=lmax, iter=iter, healpy_ordering=True, method=method
    )
    f1 = jnp.zeros(lmax + 1).at[2:].set(2.0 / jnp.sqrt((ls[2:] + 2.0) * (ls[2:] - 1.0)))
    for i, do in enumerate(task):
        if do:
            alms[i] = alms[i] + jhp.almxfl(p[i], f1, healpy_ordering=True)
    w0, w1 = jhp.alm2map_spin([walm0b, walm1], nside, spin=2, lmax=lmax, healpy_ordering=True, method=method)
    p0 = jhp.map2alm(w0 * g1 + w1 * g2, lmax=lmax, pol=False, iter=iter, healpy_ordering=True, method=method)
    p1 = jhp.map2alm(w0 * g2 - w1 * g1, lmax=lmax, pol=False, iter=iter, healpy_ordering=True, method=method)
    p = [p0, p1]
    f3 = jnp.zeros(lmax + 1).at[2:].set(1.0 / jnp.sqrt((ls[2:] + 2.0) * (ls[2:] + 1.0) * ls[2:] * (ls[2:] - 1.0)))
    for i, do in enumerate(task):
        if do:
            alms[i] = alms[i] + jhp.almxfl(p[i], f3, healpy_ordering=True)
    return alms[0], alms[1]


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def anafast_masked(
    map1,
    map2=None,
    *,
    mask=None,
    lmax: int | None = None,
    method: str = 'jax',
    pol: bool = False,
    purify_e: bool = False,
    purify_b: bool = False,
    mcm: MCM | None = None,
    nlb: int = 16,
    iter: int = 0,
):
    """Angular power spectrum of a (optionally masked) HEALPix map. Returns ``(ell, cl)``.

    * ``mask=None`` -> plain ``jhp.anafast``. ``pol=False`` returns one ``C_l`` (shape
      ``(lmax+1,)``); ``pol=True`` takes ``map1=(2, npix)`` ``(Q, U)`` and returns ``(3, lmax+1)``
      = (EE, EB, BB). (Pre-multiply the map by the apodized mask and pass ``mask=None`` to get the
      *coupled* pseudo-``C_l`` used in forward-model inference.)
    * ``mask`` given -> masked pseudo-``C_l``, **decoupled** into bandpowers (``ell`` = effective
      bandpower multipoles). ``pol=False`` returns the decoupled scalar ``C_l`` (shape
      ``(n_bands,)``); ``pol=True`` returns ``(3, n_bands)`` with EE and BB **decoupled** and EB the
      **binned (coupled) pseudo** (its decoupling block is out of scope). ``purify_e`` / ``purify_b``
      apply the spin-2 purification before estimation.

    ``purify_*`` require a ``mask``. The MCM is built from ``mask`` unless a precomputed ``mcm`` is
    supplied. ``iter`` is the (fixed, unrolled -> differentiable) refinement count for the spin-2
    field transforms; the iter=0 default reproduces the NaMaster-validated behaviour, while a small
    ``iter>0`` lowers the ~1% spin-2 transform floor on the residual E->B leakage (study-anticipated
    for the EE>>BB dynamic range). Requires 64-bit precision (the spin-2 solve is ill-conditioned).
    """
    require_x64()
    map1 = jnp.asarray(map1)
    if pol and map2 is not None:
        raise NotImplementedError(
            'Spin-2 cross-spectra (pol=True with map2) are not supported; map2 is for scalar '
            'cross-spectra only. Pass a single (2, npix) map as map1.'
        )
    if mask is None:
        if purify_e or purify_b:
            raise ValueError('purify_e/purify_b require a mask (purification needs the mask).')
        if not pol:
            cl = jhp.anafast(map1, None if map2 is None else jnp.asarray(map2), lmax=lmax, pol=False, method=method)
            return jnp.arange(cl.shape[-1]) * 1.0, jnp.asarray(cl)
        g1, g2 = map1[0], map1[1]
        _lmax = lmax if lmax is not None else 3 * jhp.npix2nside(g1.shape[-1]) - 1
        E, Bb = jhp.map2alm_spin([g1, g2], spin=2, lmax=_lmax, iter=iter, healpy_ordering=True, method=method)
        ee = jhp.alm2cl(E, healpy_ordering=True)
        eb = jhp.alm2cl(E, Bb, healpy_ordering=True)
        bb = jhp.alm2cl(Bb, healpy_ordering=True)
        return jnp.arange(ee.shape[-1]) * 1.0, jnp.stack([ee, eb, bb], axis=0)

    mask = jnp.asarray(mask)
    nside = jhp.npix2nside(mask.shape[0])
    _lmax = lmax if lmax is not None else 3 * nside - 1
    if mcm is None:
        mcm = compute_mcm(mask, lmax=_lmax, nlb=nlb, pol=pol, purify_b=purify_b, method=method)

    if not pol:
        masked1 = mask * map1
        masked2 = None if map2 is None else mask * jnp.asarray(map2)
        ps = jhp.anafast(masked1, masked2, lmax=mcm.lmax, pol=False, method=method)
        dec = _decouple_spin0(ps, mcm.spin0, mcm.B, mcm.S)
        return mcm.ell_eff, jnp.asarray(dec)

    g1, g2 = map1[0], map1[1]
    if purify_e or purify_b:
        E, Bb = _purify_eb(g1, g2, mask, nside, mcm.lmax, method, iter=iter, task=(purify_e, purify_b))
    else:
        E, Bb = jhp.map2alm_spin(
            [mask * g1, mask * g2], spin=2, lmax=mcm.lmax, iter=iter, healpy_ordering=True, method=method
        )
    psEE = jhp.alm2cl(E, healpy_ordering=True)
    psEB = jhp.alm2cl(E, Bb, healpy_ordering=True)
    psBB = jhp.alm2cl(Bb, healpy_ordering=True)
    if purify_b:
        if mcm.pure_bb is None:
            raise ValueError(
                'purify_b decoupling needs a pure-B MCM; build it with compute_mcm(..., purify_b=True).'
            )
        dEE, dBB = _decouple_spin2_pure(psEE, psBB, mcm.eeee, mcm.eebb, mcm.pure_bb, mcm.B, mcm.S)
    else:
        dEE, dBB = _decouple_spin2(psEE, psBB, mcm.eeee, mcm.eebb, mcm.B, mcm.S)
    dEB = mcm.B @ psEB  # EB: binned (coupled) pseudo only; its decoupling block is out of scope
    return mcm.ell_eff, jnp.stack([jnp.asarray(dEE), jnp.asarray(dEB), jnp.asarray(dBB)], axis=0)
