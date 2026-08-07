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

"""Mode-coupling matrix (MCM) for masked-sky angular power spectra (MASTER, Hivon 2002).

A mask ``W`` couples multipoles: the measured ("pseudo") spectrum is ``C_l_tilde = sum_l' M_ll'
C_l'``. The coupling matrix ``M_ll'`` depends only on the mask power spectrum ``W_l = anafast(W)``.
Instead of explicit Wigner-3j sums, the MCM is evaluated in position space on Gauss-Legendre nodes
(``G(x) = sum_l (2l+1) W_l P_l(x)``, then a quadrature contraction with Legendre / Wigner-d
kernels), which is pure matmuls and recursions and therefore fully differentiable in ``W_l``.

For spin-2 the mask also mixes E and B; the coupling is a 2x2 block built from the
``d^l_{2,+/-2}`` Wigner-d kernels (``EEEE = (M++ + M--) / 2``, ``EEBB = (M++ - M--) / 2``).
"""

from __future__ import annotations

from functools import lru_cache, partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import jax_healpy as jhp

from ._utils import require_x64

__all__ = ['MCM', 'compute_mcm', 'bandpower_windows']


# --------------------------------------------------------------------------- #
# Legendre / Wigner-d kernels (upward ell-recursions on Gauss-Legendre nodes) #
# --------------------------------------------------------------------------- #
def _legendre_all(x, lmax):
    """P_l(x), l=0..lmax -> (lmax+1, N)."""

    def body(carry, ell):
        Plm1, Pl = carry
        return (Pl, ((2 * ell + 1) * x * Pl - ell * Plm1) / (ell + 1)), Pl

    P0 = jnp.ones_like(x)
    _, Pstk = jax.lax.scan(body, (P0, x), jnp.arange(1, lmax + 1))
    return jnp.concatenate([P0[None], Pstk], axis=0)


def _wigner_d2_all(x, lmax, mp):
    """d^l_{2,mp}(theta), x=cos(theta), l=0..lmax, mp=+/-2 -> (lmax+1, N)."""
    m = 2
    d2 = (1 + x) ** 2 / 4 if mp == 2 else (1 - x) ** 2 / 4

    def body(carry, ell):
        dlm1, dl = carry
        a = (2 * ell + 1) * (ell * (ell + 1) * x - m * mp)
        b = (ell + 1) * jnp.sqrt((ell**2 - m**2) * (ell**2 - mp**2))
        c = ell * jnp.sqrt(((ell + 1) ** 2 - m**2) * ((ell + 1) ** 2 - mp**2))
        return (dl, (a * dl - b * dlm1) / c), dl

    _, dstk = jax.lax.scan(body, (jnp.zeros_like(x), d2), jnp.arange(2, lmax + 1))
    return jnp.concatenate([jnp.zeros((2,) + x.shape), dstk], axis=0)


# --------------------------------------------------------------------------- #
# Window derivative harmonic coefficients (shared with E/B purification)      #
# --------------------------------------------------------------------------- #
def _window_derivative_alms(mask, lmax, method='jax'):
    """Harmonic coefficients of the window and its 1st/2nd covariant derivatives.

    Returns ``(w0_alm, w1_alm, w2_alm)`` (healpy-ordered), the spin-0 window and its spin-1 and
    spin-2 derivative windows -- the building blocks of both the Smith-2006 purification correction
    (:func:`jax_healpy.pseudo_cl._estimate._purify_eb`) and the pure-B mode-coupling matrix:

    * ``w0_alm`` = ``map2alm(mask)``                              (spin-0 window ``w``)
    * ``w1_alm`` = ``-sqrt(l(l+1)) w0_alm``                       (spin-1, first derivative)
    * ``w2_alm`` = ``sqrt((l-1)l(l+1)(l+2)) w0_alm = beta_{l,2} w0_alm``  (spin-2, second derivative)

    ``beta_{l,s} = sqrt((l+s)!/(l-s)!)``; ``w2_alm`` is built as ``w1_alm * -sqrt((l+2)(l-1))`` so the
    composition matches the historical :func:`_purify_eb` construction exactly (bit-for-bit).
    """
    ls = jnp.arange(lmax + 1)
    w0_alm = jhp.map2alm(jnp.asarray(mask), lmax=lmax, pol=False, iter=3, healpy_ordering=True, method=method)
    w1_alm = jhp.almxfl(w0_alm, -jnp.sqrt((ls + 1.0) * ls), healpy_ordering=True)
    beta2 = jnp.zeros(lmax + 1).at[2:].set(-jnp.sqrt((ls[2:] + 2.0) * (ls[2:] - 1.0)))
    w2_alm = jhp.almxfl(w1_alm, beta2, healpy_ordering=True)
    return w0_alm, w1_alm, w2_alm


@partial(jax.jit, static_argnums=(1,))
def _mcm_spin0(Wl, lmax):
    """Spin-0 mode-coupling matrix M[l1,l2]."""
    ells = jnp.arange(lmax + 1)
    x_np, w_np = np.polynomial.legendre.leggauss(3 * lmax + 5)
    x, wq = jnp.asarray(x_np), jnp.asarray(w_np)
    P = _legendre_all(x, lmax)
    G = jnp.sum((2 * ells + 1)[:, None] * Wl[:, None] * P, axis=0)
    return (2 * ells + 1)[None, :] / (8 * jnp.pi) * ((P * (wq * G)[None, :]) @ P.T)


@partial(jax.jit, static_argnums=(1,))
def _mcm_spin2(Wl, lmax):
    """Spin-2 MCM blocks (EE<-EE, EE<-BB) from the +/-2 Wigner-d kernels."""
    ells = jnp.arange(lmax + 1)
    x_np, w_np = np.polynomial.legendre.leggauss(3 * lmax + 5)
    x, wq = jnp.asarray(x_np), jnp.asarray(w_np)
    G = jnp.sum((2 * ells + 1)[:, None] * Wl[:, None] * _legendre_all(x, lmax), axis=0)
    d22, d2m2 = _wigner_d2_all(x, lmax, 2), _wigner_d2_all(x, lmax, -2)
    norm = (2 * ells + 1)[None, :] / (8 * jnp.pi)
    Mpp = norm * ((d22 * (wq * G)[None, :]) @ d22.T)
    Mmm = norm * ((d2m2 * (wq * G)[None, :]) @ d2m2.T)
    return 0.5 * (Mpp + Mmm), 0.5 * (Mpp - Mmm)


# --------------------------------------------------------------------------- #
# Pure-B mode-coupling matrix (Grain-Tristram-Stompor 2009 / NaMaster Alonso  #
# 2019). The purified-B pseudo must be decoupled with a *pure* MCM: feeding it #
# through the standard spin-2 block re-injects the bright E-power into BB.     #
#                                                                             #
# The pure-B coupling replaces the spin-2 Wigner-3j factor (l l' l''; 2 -2 0)  #
# by T = (l l' l''; 2 -2 0) + fac1*(l l' l''; 1 -2 1) + fac2*(l l' l''; 0 -2 2)#
# with fac1 = 2 sqrt(l''(l''+1)/((l+2)(l-1))) and                             #
#      fac2 = beta_{l'',2} / sqrt((l+2)(l+1)l(l-1)),  beta_{l,2}=sqrt((l-1)l(l+1)(l+2)).
# Then M^{pureB<-BB}_{ll'} = (2l'+1)/4pi sum_l'' (2l''+1) W_l'' T^2 [l+l'+l'' even].
# (The pure-B<-EE leakage block vanishes analytically -- the purity property -- so the
# purified BB decouples with this block alone.) Validated element-wise vs NaMaster's
# get_coupling_matrix() to ~1e-15. The Wigner-3j are window-independent constants, so
# the whole table is precomputed (cached) and the MCM is the differentiable contraction
# M = einsum(K, W_l), with gradients flowing only through the mask power spectrum W_l.
# --------------------------------------------------------------------------- #
def _drc3jj(il2, il3, im2, im3):
    """Wigner-3j ``(l1 il2 il3; m1 im2 im3)`` with ``m1=-im2-im3``, for ``l1`` in ``[l1min, l1max]``.

    Faithful translation of the SLATEC ``DRC3JJ`` bidirectional recursion (as used by NaMaster):
    numerically **stable at high l**, unlike the explicit Racah single-sum (which catastrophically
    cancels). Returns ``(l1min, thrcof)`` with ``thrcof[l1 - l1min] = the 3j``.
    """
    huge = np.sqrt(1.79e308 / 20.0)
    srhuge = np.sqrt(huge)
    tiny = 1.0 / huge
    srtiny = 1.0 / srhuge
    im1 = -im2 - im3
    l2, l3 = float(il2), float(il3)
    m1, m2, m3 = float(im1), float(im2), float(im3)
    sign2 = 1 if (abs(il2 + im2 - il3 + im3) % 2 == 0) else -1
    l1max = il2 + il3
    l1min = max(abs(il2 - il3), abs(im1))
    nfin = l1max - l1min + 1
    if nfin <= 0 or (il2 - abs(im2) < 0) or (il3 - abs(im3) < 0):
        return l1min, np.zeros(max(nfin, 0))
    thr = np.zeros(nfin)
    if l1max == l1min:
        thr[0] = sign2 / np.sqrt(l1min + l2 + l3 + 1)
        return l1min, thr

    # forward recursion
    l1 = float(l1min)
    newfac = 0.0
    c1 = 0.0
    sum1 = (l1 + l1 + 1) * tiny
    thr[0] = srtiny
    x = srtiny
    sumfor = sum1
    c1old = 0.0
    denom = 1.0
    lstep = 0
    converging = True
    while lstep < nfin - 1 and converging:
        lstep += 1
        l1 += 1
        oldfac = newfac
        a1 = (l1 + l2 + l3 + 1) * (l1 - l2 + l3) * (l1 + l2 - l3) * (-l1 + l2 + l3 + 1)
        a2 = (l1 + m1) * (l1 - m1)
        newfac = np.sqrt(a1 * a2)
        if l1 > 1:
            dv = -l2 * (l2 + 1) * m1 + l3 * (l3 + 1) * m1 + l1 * (l1 - 1) * (m3 - m2)
            denom = (l1 - 1) * newfac
            if lstep > 1:
                c1old = abs(c1)
            c1 = -(l1 + l1 - 1) * dv / denom
        else:
            c1 = -(l1 + l1 - 1) * l1 * (m3 - m2) / newfac
        if lstep <= 1:
            x = srtiny * c1
            thr[1] = x
            sum1 += tiny * (l1 + l1 + 1) * c1 * c1
        else:
            c2 = -l1 * oldfac / denom
            x = c1 * thr[lstep - 1] + c2 * thr[lstep - 2]
            thr[lstep] = x
            sumfor = sum1
            sum1 += (l1 + l1 + 1) * x * x
            if lstep < nfin - 1:
                if abs(x) >= srhuge:
                    for ii in range(lstep + 1):
                        if abs(thr[ii]) < srtiny:
                            thr[ii] = 0.0
                        thr[ii] /= srhuge
                    sum1 /= huge
                    sumfor /= huge
                    x /= srhuge
                if c1old <= abs(c1):
                    converging = False

    if nfin > 2:
        x1, x2, x3 = x, thr[lstep - 1], thr[lstep - 2]
        nstep2 = nfin - lstep - 1 + 3
        nfinp1 = nfin + 1
        l1 = float(l1max)
        thr[nfin - 1] = srtiny
        sum2 = tiny * (l1 + l1 + 1)
        l1 += 2
        y = srtiny
        sumbac = sum2
        lstep = 0
        while lstep < nstep2 - 1:  # backward recursion
            lstep += 1
            l1 -= 1
            oldfac = newfac
            a1s = (l1 + l2 + l3) * (l1 - l2 + l3 - 1) * (l1 + l2 - l3 - 1) * (-l1 + l2 + l3 + 2)
            a2s = (l1 + m1 - 1) * (l1 - m1 - 1)
            newfac = np.sqrt(a1s * a2s)
            dv = -l2 * (l2 + 1) * m1 + l3 * (l3 + 1) * m1 + l1 * (l1 - 1) * (m3 - m2)
            denom = l1 * newfac
            c1 = -(l1 + l1 - 1) * dv / denom
            if lstep <= 1:
                y = srtiny * c1
                thr[nfin - 2] = y
                sumbac = sum2
                sum2 += tiny * (l1 + l1 - 3) * c1 * c1
            else:
                c2 = -(l1 - 1) * oldfac / denom
                y = c1 * thr[nfin - lstep] + c2 * thr[nfinp1 - lstep]
                if lstep != nstep2 - 1:
                    thr[nfin - lstep - 1] = y
                    sumbac = sum2
                    sum2 += (l1 + l1 - 3) * y * y
                    if abs(y) >= srhuge:
                        for ii in range(lstep + 1):
                            index = nfin - ii - 1
                            if abs(thr[index]) < srtiny:
                                thr[index] = 0.0
                            thr[index] /= srhuge
                        sum2 /= huge
                        sumbac /= huge
        y3, y2, y1 = y, thr[nfin - lstep], thr[nfinp1 - lstep]
        ratio = (x1 * y1 + x2 * y2 + x3 * y3) / (x1 * x1 + x2 * x2 + x3 * x3)
        nlim = nfin - nstep2 + 1
        if abs(ratio) < 1:
            nlim += 1
            ratio = 1.0 / ratio
            thr[nlim - 1:nfin] *= ratio
            sumuni = ratio * ratio * sumbac + sumfor
        else:
            thr[:nlim] *= ratio
            sumuni = ratio * ratio * sumfor + sumbac
    else:
        sumuni = sum1

    cnorm = 1.0 / np.sqrt(sumuni)
    if np.copysign(1.0, thr[nfin - 1]) * sign2 <= 0:
        cnorm = -cnorm
    if abs(cnorm) < 1:
        thresh = tiny / abs(cnorm)
        thr = np.where(np.abs(thr) < thresh, 0.0, thr)
    return l1min, thr * cnorm


def _wigner3j_table(lmax, m1, m2):
    """``(l l' l''; m1 m2 m3)`` with ``m3=-m1-m2``, for ``l,l',l'' in [0,lmax]``.

    Stable (drc3jj recursion); a *window-independent constant*. Returns a dense
    ``(lmax+1, lmax+1, lmax+1)`` array, 0 outside the selection rules. Uses the cyclic identity
    ``(l l' l''; m1 m2 m3) = (l'' l l'; m3 m1 m2)`` so a single recursion over ``l''`` fills each row.
    """
    out = np.zeros((lmax + 1, lmax + 1, lmax + 1))
    for ell in range(lmax + 1):
        for ellp in range(lmax + 1):
            l1min, thr = _drc3jj(ell, ellp, m1, m2)  # (l'' l l'; m3 m1 m2) = (l l' l''; m1 m2 m3)
            hi = min(ell + ellp, lmax)
            if hi >= l1min:
                out[ell, ellp, l1min : hi + 1] = thr[: hi - l1min + 1]
    return out


@lru_cache(maxsize=4)
def _pureB_kernel(lmax: int) -> Array:
    """Constant kernel ``K[l,l',l'']`` with ``M^{pureB<-BB}_{ll'} = sum_l'' K[l,l',l''] W_l''``.

    Window-independent (depends only on ``lmax``); cached so an optimization loop that rebuilds the
    MCM every step pays the Wigner-3j cost only once.
    """
    w220 = _wigner3j_table(lmax, 2, -2)
    w121 = _wigner3j_table(lmax, 1, -2)
    w022 = _wigner3j_table(lmax, 0, -2)
    ell = np.arange(lmax + 1, dtype=float)
    A1 = np.zeros(lmax + 1)
    A1[2:] = 2.0 / np.sqrt((ell[2:] + 2.0) * (ell[2:] - 1.0))  # output-l prefactor of the spin-1 term
    A2 = np.zeros(lmax + 1)
    A2[2:] = 1.0 / np.sqrt((ell[2:] + 2.0) * (ell[2:] + 1.0) * ell[2:] * (ell[2:] - 1.0))  # spin-2 term
    b1 = np.sqrt(ell * (ell + 1.0))  # beta_{l'',1}
    b2 = np.zeros(lmax + 1)
    b2[2:] = np.sqrt((ell[2:] - 1.0) * ell[2:] * (ell[2:] + 1.0) * (ell[2:] + 2.0))  # beta_{l'',2}
    T = w220 + A1[:, None, None] * b1[None, None, :] * w121 + A2[:, None, None] * b2[None, None, :] * w022
    lsum = np.arange(lmax + 1)[:, None, None] + np.arange(lmax + 1)[None, :, None] + np.arange(lmax + 1)[None, None, :]
    even = (lsum % 2 == 0).astype(float)  # pureB<-BB is the even-parity block
    K = ((2 * ell + 1.0) / (4 * np.pi))[None, :, None] * (2 * ell + 1.0)[None, None, :] * T**2 * even
    return jnp.asarray(K)


def _mcm_spin2_pureB(Wl, lmax):
    """Pure-B (BB<-BB) mode-coupling matrix ``(lmax+1, lmax+1)``, differentiable in ``Wl``."""
    return jnp.einsum('lLk,k->lL', _pureB_kernel(lmax), jnp.asarray(Wl))


# --------------------------------------------------------------------------- #
# Linear bandpower binning (matches nmt.NmtBin.from_nside_linear)             #
# --------------------------------------------------------------------------- #
def _linear_bins(nside, nlb, lmax):
    """Binning operators B (n_bands, lmax+1) and S (lmax+1, n_bands), plus effective ells.

    Replicates ``nmt.NmtBin.from_nside_linear(nside, nlb)``: bins of ``nlb`` consecutive
    multipoles starting at l=2 (uniform weights 1/nlb), trailing partial bin dropped.
    """
    nbands = (lmax + 1 - 2) // nlb
    B = np.zeros((nbands, lmax + 1))
    S = np.zeros((lmax + 1, nbands))
    ell_eff = np.zeros(nbands)
    for q in range(nbands):
        ells = np.arange(2 + q * nlb, 2 + (q + 1) * nlb)
        B[q, ells] = 1.0 / nlb
        S[ells, q] = 1.0
        ell_eff[q] = ells.mean()
    return jnp.asarray(B), jnp.asarray(S), jnp.asarray(ell_eff)


# --------------------------------------------------------------------------- #
# MCM container + public builder                                              #
# --------------------------------------------------------------------------- #
class MCM(eqx.Module):
    """Precomputed mode-coupling data for a fixed (apodized) mask.

    Depends only on the mask, so build once and reuse (freeze for inference). Built by
    :func:`compute_mcm` and accepted by :func:`jax_healpy.pseudo_cl.anafast_masked` via ``mcm=``.
    """

    spin0: Array | None
    eeee: Array | None
    eebb: Array | None
    pure_bb: Array | None
    B: Array
    S: Array
    ell_eff: Array
    lmax: int = eqx.field(static=True)
    nlb: int = eqx.field(static=True)
    pol: bool = eqx.field(static=True)


def compute_mcm(
    mask, *, lmax: int | None = None, nlb: int = 16, pol: bool = False, purify_b: bool = False, method: str = 'jax'
) -> MCM:
    """Build the mode-coupling matrix + bandpower binning for an (apodized) ``mask``.

    Parameters
    ----------
    mask : array_like
        Apodized HEALPix mask (RING), shape ``(npix,)``.
    lmax : int, optional
        Max multipole; defaults to ``3*nside-1``.
    nlb : int, default=16
        Multipoles per bandpower (linear binning, matches NaMaster).
    pol : bool, default=False
        If True also build the spin-2 (EE<-EE, EE<-BB) blocks.
    purify_b : bool, default=False
        If True (and ``pol``), also build the **pure-B** (BB<-BB) coupling, needed to decouple a
        purified B-pseudo without re-injecting E-power. Matches NaMaster's purified coupling.
    method : str, default='jax'
        Backend passed to :func:`jax_healpy.anafast` when computing the mask power spectrum.

    Returns
    -------
    MCM
        The precomputed mode-coupling container (a differentiable ``eqx.Module``).
    """
    require_x64()
    mask = jnp.asarray(mask)
    nside = jhp.npix2nside(mask.shape[0])
    if lmax is None:
        lmax = 3 * nside - 1
    Wl = jhp.anafast(mask, lmax=lmax, pol=False, method=method)
    spin0 = _mcm_spin0(Wl, lmax)
    eeee, eebb = _mcm_spin2(Wl, lmax) if pol else (None, None)
    pure_bb = _mcm_spin2_pureB(Wl, lmax) if (pol and purify_b) else None
    B, S, ell_eff = _linear_bins(nside, nlb, lmax)
    return MCM(
        spin0=spin0, eeee=eeee, eebb=eebb, pure_bb=pure_bb, B=B, S=S, ell_eff=ell_eff, lmax=lmax, nlb=nlb, pol=pol
    )


def bandpower_windows(mcm: MCM) -> Array:
    """Scalar bandpower window functions ``W_{b,l}`` such that ``Chat_b = sum_l W_{b,l} C_l``.

    Decoupling is ``Chat_b = (B M S)^-1 B C_tilde`` with ``C_tilde = M C``, so the window mapping
    the *true* spectrum to the decoupled bandpowers is ``W = (B M S)^-1 B M`` (shape
    ``(n_bands, lmax+1)``). This is the analogue of NaMaster's ``get_bandpower_windows`` for the
    spin-0 block and is used to compare against it as an oracle.
    """
    if mcm.spin0 is None:
        raise NotImplementedError('bandpower_windows is only implemented for the spin-0 block.')
    return jnp.linalg.solve(mcm.B @ mcm.spin0 @ mcm.S, mcm.B @ mcm.spin0)
