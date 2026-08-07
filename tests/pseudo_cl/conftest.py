"""Shared fixtures for the ``jax_healpy.pseudo_cl`` test suite.

Mirrors ``tests/sphtfunc/conftest.py``: the s2fft-backed spin transforms accumulate compiled XLA
state that nondeterministically corrupts results across a session, so caches are cleared before
every test. 64-bit precision is mandatory (the spin-2 decoupling solve returns all-NaN in float32).
"""

from __future__ import annotations

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_healpy as jhp
from jax_healpy import pseudo_cl as pcl

NS = 64
LMAX = 3 * NS - 1
NLB = 16
METHOD = 'jax'


@pytest.fixture(autouse=True)
def _isolate_jax_state():
    """Clear JAX caches and re-assert 64-bit precision before every test (see module docstring)."""
    jax.config.update('jax_enable_x64', True)
    jax.clear_caches()
    yield


@pytest.fixture(scope='module')
def nside() -> int:
    return NS


@pytest.fixture(scope='module')
def lmax() -> int:
    return LMAX


@pytest.fixture(scope='module')
def nlb() -> int:
    return NLB


@pytest.fixture(scope='module')
def method() -> str:
    return METHOD


@pytest.fixture(scope='module')
def quadrant_mask() -> np.ndarray:
    """Binary quadrant mask (pessimal 90 deg corners -- exact-decoupling oracle only)."""
    npix = hp.nside2npix(NS)
    m = np.ones(npix)
    th, ph = hp.pix2ang(NS, np.arange(npix))
    m[th < np.pi / 2] = 0.0
    m[(ph < 3 * np.pi / 2) & (ph > np.pi / 2)] = 0.0
    return m


@pytest.fixture(scope='module')
def apo(quadrant_mask: np.ndarray) -> jax.Array:
    """C2-apodized quadrant mask (aposize 5 deg)."""
    return pcl.apodize(jnp.asarray(quadrant_mask), 5.0)


@pytest.fixture(scope='module')
def kappa() -> jax.Array:
    """Random scalar field with a red power spectrum (seeded, via healpy synfast)."""
    ell = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1)
    cl[2:] = 1.0 / (ell[2:] + 10.0) ** 2.5
    np.random.seed(3)
    return jnp.asarray(hp.synfast(cl, NS, lmax=LMAX, pol=False))


@pytest.fixture(scope='module')
def pure_e_qu(kappa: jax.Array) -> jax.Array:
    """A pure-E spin-2 ``(Q, U)`` field (true BB == 0) built by E-filtering ``kappa``."""
    jax.clear_caches()
    ell = np.arange(LMAX + 1)
    fk = np.zeros(LMAX + 1)
    fk[2:] = np.sqrt((ell[2:] + 2) * (ell[2:] - 1) / (ell[2:] * (ell[2:] + 1)))
    E = jhp.almxfl(
        jhp.map2alm(kappa, lmax=LMAX, pol=False, healpy_ordering=True, method=METHOD),
        jnp.asarray(fk),
        healpy_ordering=True,
    )
    g1, g2 = jhp.alm2map([E, jnp.zeros_like(E)], NS, lmax=LMAX, pol=True, healpy_ordering=True, method=METHOD)
    return jnp.stack([g1, g2], axis=0)
