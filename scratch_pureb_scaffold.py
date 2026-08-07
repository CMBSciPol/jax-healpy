"""Scaffold: validate the pure-B mode-coupling matrix formula (Alonso 2019 eq 57, 63-66).

Run in ffi12 (has sympy + pymaster), with the live repo on PYTHONPATH:
    env PYTHONPATH=/home/wassim/Projects/CMB/jax-healpy \
        micromamba run -n ffi12 python scratch_pureb_scaffold.py

Two independent checks (advisor's "split the two questions"):
  CHECK A: my sympy-3j STANDARD spin-2 MCM (EEEE/EEBB) == the code's GL `_mcm_spin2`.
           -> pins conventions/normalization/parity WITHOUT NaMaster.
  CHECK B: my sympy-3j PURE-B MCM == NaMaster get_coupling_matrix() pure-B blocks.
           -> tests the eq 63-66 T-replacement (the actual unknown).
"""

from __future__ import annotations

import functools

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

import healpy as hp  # noqa: E402
import pymaster as nmt  # noqa: E402
from sympy import N as symN  # noqa: E402
from sympy.physics.wigner import wigner_3j  # noqa: E402

import jax_healpy as jhp  # noqa: E402
from jax_healpy import pseudo_cl as pcl  # noqa: E402
from jax_healpy.pseudo_cl._mcm import _mcm_spin2  # noqa: E402

NSIDE = 8
LMAX = 3 * NSIDE - 1  # 23 == NaMaster's natural lmax for this nside (avoids purify lmax mismatch)
APOSIZE = 20.0


@functools.lru_cache(maxsize=None)
def w3j(l1, l2, l3, m1, m2, m3):
    """Exact Wigner-3j as float (0 if selection rules fail)."""
    if l3 < abs(l1 - l2) or l3 > l1 + l2:
        return 0.0
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
        return 0.0
    return float(wigner_3j(l1, l2, l3, m1, m2, m3))


def beta(l, s):
    """beta_{l,s} = sqrt((l+s)!/(l-s)!); 0 if l<s."""
    if l < s:
        return 0.0
    v = 1.0
    for k in range(l - s + 1, l + s + 1):
        v *= k
    return np.sqrt(v)


def standard_spin2_3j(Wl, lmax):
    """Standard spin-2 MCM via explicit 3j: returns (EEEE, EEBB).

    M+/-[l,l'] = (2l'+1)/(4pi) sum_l'' (2l''+1) W_l'' (l l' l''; 2 -2 0)^2 [1 +/- (-1)^L]/2
    EEEE = M+,  EEBB = M-.
    """
    EEEE = np.zeros((lmax + 1, lmax + 1))
    EEBB = np.zeros((lmax + 1, lmax + 1))
    for l in range(2, lmax + 1):
        for lp in range(2, lmax + 1):
            sp = sm = 0.0
            for lpp in range(abs(l - lp), min(l + lp, lmax) + 1):
                tj = w3j(l, lp, lpp, 2, -2, 0)
                if tj == 0.0:
                    continue
                base = (2 * lpp + 1) * Wl[lpp] * tj * tj
                par = (-1) ** (l + lp + lpp)
                sp += base * (1 + par) / 2
                sm += base * (1 - par) / 2
            norm = (2 * lp + 1) / (4 * np.pi)
            EEEE[l, lp] = norm * sp
            EEBB[l, lp] = norm * sm
    return EEEE, EEBB


def T_pureB(l, lp, lpp):
    """Replaced 3j factor for a purified-B output field.

    Uses the VALIDATED ``_purify_eb`` output coefficients (f1, f3 -- reciprocal-beta) times the
    mask-side derivative-window beta_{l'',s} (numerator) with the window sign sigma=[+,-,+]
    (w1 = -beta1 w0, w2 = +beta2 w0):

        T = (l l' l''; 2 -2 0)
            - f1(l) beta_{l'',1} (l l' l''; 1 -2 1)
            + f3(l) beta_{l'',2} (l l' l''; 0 -2 2)
    """
    # EXACT NaMaster wfac_ispure[1] = wss1 + fac_12*w12 + fac_02*w02 (both + ; signs are in the 3j)
    t = w3j(l, lp, lpp, 2, -2, 0)
    if l >= 2:
        fac_12 = 2.0 * np.sqrt(lpp * (lpp + 1.0) / ((l + 2.0) * (l - 1.0)))
        fac_02 = beta(lpp, 2) / np.sqrt((l + 2.0) * (l + 1.0) * l * (l - 1.0)) if l >= 2 else 0.0
        t += fac_12 * w3j(l, lp, lpp, 1, -2, 1)
        t += fac_02 * w3j(l, lp, lpp, 0, -2, 2)
    return t


def pureB_3j(Wl, lmax):
    """Pure-B MCM via explicit 3j: returns (pureB_from_EE, pureB_from_BB).

    Both 3j factors replaced by T (auto pure-B); parity splits E-leakage vs B-signal.
    """
    BE = np.zeros((lmax + 1, lmax + 1))  # pureB <- EE  (leakage; minus parity)
    BB = np.zeros((lmax + 1, lmax + 1))  # pureB <- BB  (signal; plus parity)
    for l in range(2, lmax + 1):
        for lp in range(2, lmax + 1):
            sp = sm = 0.0
            for lpp in range(abs(l - lp), min(l + lp, lmax) + 1):
                T = T_pureB(l, lp, lpp)
                if T == 0.0:
                    continue
                base = (2 * lpp + 1) * Wl[lpp] * T * T
                par = (-1) ** (l + lp + lpp)
                sp += base * (1 + par) / 2
                sm += base * (1 - par) / 2
            norm = (2 * lp + 1) / (4 * np.pi)
            BB[l, lp] = norm * sp
            BE[l, lp] = norm * sm
    return BE, BB


def main():
    # --- mask + W_l (use the code's apodize so everything is consistent) ---
    npix = hp.nside2npix(NSIDE)
    th, ph = hp.pix2ang(NSIDE, np.arange(npix))
    binary = np.ones(npix)
    binary[th > 1.5] = 0.0  # a polar cap footprint
    apo = np.asarray(pcl.apodize(binary, APOSIZE))
    Wl = np.asarray(jhp.anafast(apo, lmax=LMAX, pol=False, method='jax'))

    sl = slice(2, LMAX - 1)

    # ===== CHECK A: standard spin-2 MCM, my-3j vs code GL =====
    eeee_gl, eebb_gl = (np.asarray(x) for x in _mcm_spin2(jhp.anafast(apo, lmax=LMAX, method='jax'), LMAX))
    eeee_3j, eebb_3j = standard_spin2_3j(Wl, LMAX)

    def relmax(a, b):
        a, b = a[sl, sl], b[sl, sl]
        d = np.max(np.abs(a - b))
        s = np.max(np.abs(b))
        return d / s if s > 0 else d

    print('=== CHECK A: standard spin-2 MCM (my-3j vs code GL) ===')
    print(f'  EEEE relmax = {relmax(eeee_3j, eeee_gl):.3e}')
    print(f'  EEBB relmax = {relmax(eebb_3j, eebb_gl):.3e}')
    # also report a raw scale ratio in case of a constant normalization offset
    i, j = 10, 10
    print(f'  EEEE[10,10]: 3j={eeee_3j[i, j]:.6e}  gl={eeee_gl[i, j]:.6e}  ratio={eeee_3j[i, j] / eeee_gl[i, j]:.6f}')

    # ===== CHECK B: pure-B MCM, my-3j vs NaMaster =====
    BE_3j, BB_3j = pureB_3j(Wl, LMAX)

    rng = np.random.default_rng(0)
    f = nmt.NmtField(apo, [rng.standard_normal(npix), rng.standard_normal(npix)], spin=2, purify_b=True, lmax=LMAX)
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f, f, nmt.NmtBin.from_nside_linear(NSIDE, 1))
    M = np.asarray(w.get_coupling_matrix())  # (4(lmax+1), 4(lmax+1)), blocks EE,EB,BE,BB
    n = LMAX + 1
    # spin-2 x spin-2 coupling is 4x4-blocked: index = 4*l + {0:EE,1:EB,2:BE,3:BB}
    print(f'  NaMaster coupling matrix shape = {M.shape} (expect {4 * n})')

    def nmt_block(out_idx, in_idx):
        return M[out_idx::4, in_idx::4]  # (n, n): rows=output l, cols=input l'

    nmt_BB = nmt_block(3, 3)  # pureB <- BB
    nmt_BE = nmt_block(3, 0)  # pureB <- EE

    # --- LAYOUT DUMP: where do my even-L (T^2) and odd-L (T^2) values land in NaMaster's 4x4? ---
    print('=== LAYOUT: my even-L T^2 (=BB_3j) & odd-L T^2 (=BE_3j) vs NaMaster 4x4 blocks ===')
    for (a, ap) in [(10, 10), (10, 11), (10, 12)]:
        print(f'  (l={a}, l\'={ap}):  my even-L T^2={BB_3j[a, ap]:.4e}   my odd-L T^2={BE_3j[a, ap]:.4e}')
        for oi, on in enumerate(['EE', 'EB', 'BE', 'BB']):
            row = '  '.join(f'<-{inn}={M[4 * a + oi, 4 * ap + ij]:.3e}' for ij, inn in enumerate(['EE', 'EB', 'BE', 'BB']))
            print(f'    out {on}: {row}')

    sl_int = slice(4, LMAX - 4)  # interior, away from band edges / truncation
    print('=== CHECK B: pure-B MCM (my-3j vs NaMaster) ===')
    print(f'  pureB<-BB relmax (full {sl})      = {relmax(BB_3j, nmt_BB):.3e}')
    print(f'  pureB<-BB relmax (interior {sl_int}) = '
          f'{np.max(np.abs(BB_3j[sl_int, sl_int] - nmt_BB[sl_int, sl_int])) / np.max(np.abs(nmt_BB[sl_int, sl_int])):.3e}')
    print('  pureB<-BB diagonal (l: 3j / nmt = ratio):')
    for ll in range(4, LMAX - 1, 3):
        print(f'    l={ll:2d}: {BB_3j[ll, ll]:.4e} / {nmt_BB[ll, ll]:.4e} = {BB_3j[ll, ll] / nmt_BB[ll, ll]:.4f}')
    # locate the full-slice relmax
    blk_3j, blk_nmt = BB_3j[sl, sl], nmt_BB[sl, sl]
    idx = np.unravel_index(np.argmax(np.abs(blk_3j - blk_nmt)), blk_3j.shape)
    print(f'  pureB<-BB relmax at (l,l\')=({idx[0] + 2},{idx[1] + 2}): 3j={blk_3j[idx]:.3e} nmt={blk_nmt[idx]:.3e}')
    print(f'  pureB<-EE: max(|3j|)={np.max(np.abs(BE_3j[sl, sl])):.3e}  '
          f'max(|nmt|)={np.max(np.abs(nmt_BE[sl, sl])):.3e}  '
          f'(NaMaster ~0 = analytic purity; ours treated as 0 in decoupling)')


if __name__ == '__main__':
    main()
