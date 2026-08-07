"""NaMaster (pymaster) oracle tests for ``jax_healpy.pseudo_cl``.

Run **locally** in the conda/micromamba env that provides NaMaster (e.g. ``ffi12``):

    micromamba run -n ffi12 python -m pytest tests/pseudo_cl/test_oracle.py -vv

``pymaster`` is **hard-imported** (no ``importorskip``): a missing oracle must fail loudly, not
silently skip and hide a regression. NaMaster is not pip/uv-installable, so this module is
**excluded from the uv CI run** via ``--ignore=tests/pseudo_cl/test_oracle.py`` in
``.github/workflows/ci.yml`` (a deliberate, documented exclusion pending a conda-based CI job) --
it is *not* a per-test skip.
"""

from __future__ import annotations

import healpy as hp
import jax
import numpy as np
import pymaster as nmt

import jax_healpy as jhp
from jax_healpy import pseudo_cl as pcl


def test_apodize_wl_matches_namaster(apo: jax.Array, quadrant_mask: np.ndarray, lmax: int, nside: int, method: str) -> None:
    """Mask power spectrum W_l (which drives the MCM) matches NaMaster's C2 apodization."""
    apo_nmt = nmt.mask_apodization(quadrant_mask, 5.0, apotype='C2')
    wl = np.asarray(jhp.anafast(apo, lmax=lmax, pol=False, method=method))
    wl_nmt = hp.anafast(apo_nmt, lmax=lmax)
    sl = slice(2, 2 * nside)
    rel = np.max(np.abs(wl[sl] - wl_nmt[sl])) / np.max(np.abs(wl_nmt[sl]))
    assert rel < 5e-3  # study reports <= 2e-3


def test_decoupled_scalar_matches_namaster(apo: jax.Array, kappa: jax.Array, lmax: int, nlb: int, nside: int, method: str) -> None:
    """Scalar decoupling is an exact match to NaMaster (same MCM + binning)."""
    _, dec = pcl.anafast_masked(kappa, mask=apo, lmax=lmax, nlb=nlb, method=method)
    apo_np = np.asarray(apo)
    f0 = nmt.NmtField(apo_np, [np.asarray(kappa)], lmax=lmax)
    dec_nmt = np.asarray(nmt.compute_full_master(f0, f0, nmt.NmtBin.from_nside_linear(nside, nlb)))[0]
    rel = np.max(np.abs(np.asarray(dec)[1:-1] - dec_nmt[1:-1])) / np.max(np.abs(dec_nmt[1:-1]))
    assert rel < 1e-8  # exact MCM/decoupling


def test_decoupled_spin2_EE_matches_namaster(apo: jax.Array, pure_e_qu: jax.Array, lmax: int, nlb: int, nside: int, method: str) -> None:
    """Spin-2 EE decoupling matches NaMaster to the iter=0 transform floor (~1%)."""
    g1, g2 = np.asarray(pure_e_qu[0]), np.asarray(pure_e_qu[1])
    _, cl = pcl.anafast_masked(pure_e_qu, mask=apo, lmax=lmax, nlb=nlb, method=method, pol=True)
    bins = nmt.NmtBin.from_nside_linear(nside, nlb)
    fE = nmt.NmtField(np.asarray(apo), [g1, g2], spin=2, lmax=lmax)
    mEE = np.asarray(nmt.compute_full_master(fE, fE, bins))[0]
    rel = np.max(np.abs(np.asarray(cl)[0][1:-1] - mEE[1:-1])) / np.max(np.abs(mEE[1:-1]))
    assert rel < 5e-2  # spin-2 jax_healpy iter=0 floor (~1%)


def test_pure_bb_coupling_matrix_matches_namaster(apo: jax.Array, lmax: int, nside: int, method: str) -> None:
    """The pure-B (BB<-BB) mode-coupling matrix matches NaMaster's purified coupling element-wise.

    This is the strongest check of the pure-B MCM (Grain-Tristram-Stompor 2009 / Alonso 2019): the
    full unbinned coupling, not just a decoupled spectrum. The pure-B<-EE leakage block vanishes
    analytically in both, so only BB<-BB carries signal.
    """
    mcm = pcl.compute_mcm(apo, lmax=lmax, pol=True, purify_b=True, method=method)
    pure_bb = np.asarray(mcm.pure_bb)
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(0)
    f = nmt.NmtField(np.asarray(apo), [rng.standard_normal(npix), rng.standard_normal(npix)],
                     spin=2, purify_b=True, lmax=lmax)
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f, f, nmt.NmtBin.from_nside_linear(nside, 1))
    nmt_bb = np.asarray(w.get_coupling_matrix())[3::4, 3::4]  # pureB <- BB
    sl = slice(2, 2 * nside)
    rel = np.max(np.abs(pure_bb[sl, sl] - nmt_bb[sl, sl])) / np.max(np.abs(nmt_bb[sl, sl]))
    assert rel < 1e-6  # window-independent Wigner-3j -> machine-precision match


def test_purified_bb_leakage_floor_vs_namaster(apo: jax.Array, pure_e_qu: jax.Array, lmax: int, nlb: int, nside: int, method: str) -> None:
    """On a pure-E sky, our purified+decoupled BB matches NaMaster's, as a fraction of EE.

    BB here is a tiny E->B leakage residual. With the pure-B MCM the residual is ~1e-4 of EE
    (down from the ~1% of the standard-MCM bug); the bound is expressed as an *absolute* leakage
    level relative to the EE scale, set by the spin-2 iter=0 transform floor (not the MCM).
    """
    g1, g2 = np.asarray(pure_e_qu[0]), np.asarray(pure_e_qu[1])
    _, cl = pcl.anafast_masked(pure_e_qu, mask=apo, lmax=lmax, nlb=nlb, method=method, pol=True, purify_b=True)
    bins = nmt.NmtBin.from_nside_linear(nside, nlb)
    fE = nmt.NmtField(np.asarray(apo), [g1, g2], spin=2, purify_b=True, lmax=lmax)
    mcl = np.asarray(nmt.compute_full_master(fE, fE, bins))
    mEE, mBB = mcl[0], mcl[3]
    bb_ours = np.asarray(cl)[2]
    leak = np.max(np.abs(bb_ours[1:-1] - mBB[1:-1])) / np.max(np.abs(mEE[1:-1]))
    assert leak < 5e-3  # pure-B MCM: ~100x tighter than the old standard-MCM decoupling
