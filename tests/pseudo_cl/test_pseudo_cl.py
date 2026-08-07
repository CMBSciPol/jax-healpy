"""Pure-JAX tests for ``jax_healpy.pseudo_cl`` (no NaMaster oracle -- run in CI and locally).

NaMaster comparisons live in ``test_oracle.py`` (run only in the conda env that has pymaster).
Configuration constants and data fixtures (``apo``, ``kappa``, ``pure_e_qu``, ``nside``, ``lmax``,
``nlb``, ``method``) come from ``conftest.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_healpy as jhp
from jax_healpy import pseudo_cl as pcl


# --------------------------------------------------------------------------- apodize
def test_apodize_zero_outside_and_range(apo: jax.Array, quadrant_mask: np.ndarray) -> None:
    apo_np = np.asarray(apo)
    assert apo_np.min() >= 0.0 and apo_np.max() <= 1.0
    assert np.max(np.abs(apo_np[quadrant_mask <= 0])) == 0.0  # exactly zero outside the footprint


def test_apodize_jittable(quadrant_mask: np.ndarray) -> None:
    binary = jnp.asarray(quadrant_mask)
    out = jax.jit(lambda m: pcl.apodize(m, 5.0))(binary)
    assert out.shape == binary.shape


def test_apodize_equals_grassfire_profile_split(quadrant_mask: np.ndarray) -> None:
    """The one-shot ``apodize`` equals the explicit grassfire + profile two-layer call."""
    binary = jnp.asarray(quadrant_mask)
    one_shot = pcl.apodize(binary, 5.0)
    dist = pcl.grassfire_distance(binary, max_aposize_deg=5.0)
    split = pcl.apodize_profile(dist, 5.0)
    assert np.allclose(np.asarray(one_shot), np.asarray(split))


def test_apodize_profile_differentiable_in_theta(quadrant_mask: np.ndarray) -> None:
    """The taper scale ``theta*`` is a differentiable parameter (rung-1 of the optimizable ladder)."""
    binary = jnp.asarray(quadrant_mask)
    dist = pcl.grassfire_distance(binary, max_aposize_deg=10.0)  # static niter from the max bound

    def fsky_weight(theta_deg):
        return jnp.sum(pcl.apodize_profile(dist, theta_deg) ** 2)

    g = jax.grad(fsky_weight)(5.0)
    assert jnp.isfinite(g) and abs(float(g)) > 0.0


# --------------------------------------------------------------------------- mask=None
def test_mask_none_scalar_equals_anafast(kappa: jax.Array, lmax: int, method: str) -> None:
    _, cl = pcl.anafast_masked(kappa, lmax=lmax, method=method)
    assert np.allclose(np.asarray(cl), np.asarray(jhp.anafast(kappa, lmax=lmax, pol=False, method=method)))


def test_mask_none_pol_three_spectra(pure_e_qu: jax.Array, lmax: int, method: str) -> None:
    _, cl = pcl.anafast_masked(pure_e_qu, lmax=lmax, method=method, pol=True)
    assert cl.shape == (3, lmax + 1)  # EE, EB, BB


def test_purify_requires_mask(pure_e_qu: jax.Array, lmax: int) -> None:
    with pytest.raises(ValueError):
        pcl.anafast_masked(pure_e_qu, lmax=lmax, pol=True, purify_b=True)


def test_pol_cross_spectrum_not_supported(pure_e_qu: jax.Array, lmax: int) -> None:
    with pytest.raises(NotImplementedError):
        pcl.anafast_masked(pure_e_qu, pure_e_qu, lmax=lmax, pol=True)


# --------------------------------------------------------------------------- coupled pseudo
def test_coupled_pseudo_via_premask_scalar(apo: jax.Array, kappa: jax.Array, lmax: int, method: str) -> None:
    _, cl = pcl.anafast_masked(apo * kappa, lmax=lmax, method=method)  # premasked + mask=None
    manual = jhp.anafast(apo * kappa, lmax=lmax, pol=False, method=method)
    assert np.allclose(np.asarray(cl), np.asarray(manual))


# --------------------------------------------------------------------------- decoupling sanity (no oracle)
def test_mcm_precomputed_identical(apo: jax.Array, kappa: jax.Array, lmax: int, nlb: int, method: str) -> None:
    mcm = pcl.compute_mcm(apo, lmax=lmax, nlb=nlb, pol=False, method=method)
    _, a = pcl.anafast_masked(kappa, mask=apo, lmax=lmax, nlb=nlb, method=method)
    _, b = pcl.anafast_masked(kappa, mask=apo, lmax=lmax, nlb=nlb, method=method, mcm=mcm)
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_eb_is_binned_pseudo(apo: jax.Array, pure_e_qu: jax.Array, lmax: int, nlb: int, method: str) -> None:
    """The EB row is the binned (coupled) pseudo, not a decoupled/validated spectrum."""
    mcm = pcl.compute_mcm(apo, lmax=lmax, nlb=nlb, pol=True, method=method)
    E, Bb = jhp.map2alm_spin(
        [apo * pure_e_qu[0], apo * pure_e_qu[1]], spin=2, lmax=lmax, healpy_ordering=True, method=method
    )
    ps_eb = jhp.alm2cl(E, Bb, healpy_ordering=True)
    expected = np.asarray(mcm.B @ jnp.asarray(ps_eb))
    _, cl = pcl.anafast_masked(pure_e_qu, mask=apo, lmax=lmax, nlb=nlb, method=method, pol=True, mcm=mcm)
    assert np.allclose(np.asarray(cl)[1], expected)


def test_purify_b_reduces_leakage(apo: jax.Array, pure_e_qu: jax.Array, lmax: int, nlb: int, method: str) -> None:
    """Pure-E sky (true BB ~ 0): purification must shrink the spurious B-mode power."""
    _, raw = pcl.anafast_masked(pure_e_qu, mask=apo, lmax=lmax, nlb=nlb, method=method, pol=True)
    _, pur = pcl.anafast_masked(pure_e_qu, mask=apo, lmax=lmax, nlb=nlb, method=method, pol=True, purify_b=True)
    assert np.mean(np.abs(np.asarray(pur)[2])) < np.mean(np.abs(np.asarray(raw)[2]))


# --------------------------------------------------------------------------- pure-B MCM (purified-BB fix)
def test_compute_mcm_purify_b_block(apo: jax.Array, lmax: int, nlb: int, method: str) -> None:
    """``compute_mcm`` builds the pure-B (BB<-BB) coupling only when ``purify_b`` is requested."""
    std = pcl.compute_mcm(apo, lmax=lmax, nlb=nlb, pol=True, method=method)
    pur = pcl.compute_mcm(apo, lmax=lmax, nlb=nlb, pol=True, purify_b=True, method=method)
    assert std.pure_bb is None
    assert pur.pure_bb is not None and pur.pure_bb.shape == (lmax + 1, lmax + 1)
    assert bool(jnp.all(jnp.isfinite(pur.pure_bb)))


def test_purify_b_decoupled_differentiable_in_map(
    apo: jax.Array, pure_e_qu: jax.Array, lmax: int, nlb: int, method: str
) -> None:
    """Gradients flow (finite, no NaN) through the purified-B decoupling / pure-B MCM contraction."""

    def loss(qu):
        _, cl = pcl.anafast_masked(qu, mask=apo, lmax=lmax, nlb=nlb, method=method, pol=True, purify_b=True)
        return jnp.sum(cl[2])  # BB

    g = jax.grad(loss)(pure_e_qu)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_purify_b_requires_pure_mcm(apo: jax.Array, pure_e_qu: jax.Array, lmax: int, nlb: int, method: str) -> None:
    """A precomputed MCM built *without* ``purify_b`` cannot be reused for purified-B decoupling."""
    mcm = pcl.compute_mcm(apo, lmax=lmax, nlb=nlb, pol=True, method=method)  # no pure_bb
    with pytest.raises(ValueError):
        pcl.anafast_masked(
            pure_e_qu, mask=apo, lmax=lmax, nlb=nlb, method=method, pol=True, purify_b=True, mcm=mcm
        )


def test_bandpower_windows_shape_and_finite(apo: jax.Array, lmax: int, nlb: int, method: str) -> None:
    mcm = pcl.compute_mcm(apo, lmax=lmax, nlb=nlb, pol=False, method=method)
    bpw = pcl.bandpower_windows(mcm)
    assert bpw.shape == (mcm.B.shape[0], lmax + 1)
    assert bool(jnp.all(jnp.isfinite(bpw)))


# --------------------------------------------------------------------------- differentiable through the map
def test_decoupled_cl_differentiable_in_map(apo: jax.Array, kappa: jax.Array, lmax: int, nlb: int, method: str) -> None:
    def loss(m):
        _, dec = pcl.anafast_masked(m, mask=apo, lmax=lmax, nlb=nlb, method=method)
        return jnp.sum(dec)

    g = jax.grad(loss)(kappa)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.sum(jnp.abs(g))) > 0.0
