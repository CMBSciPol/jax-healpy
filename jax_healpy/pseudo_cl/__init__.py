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

"""Differentiable masked-sky angular power spectra (pseudo-Cl / MASTER) for ``jax_healpy``.

Apodize a mask, optionally purify the spin-2 B-mode (Smith 2006), build the mode-coupling matrix
and decouple the measured pseudo-spectrum into unbiased bandpowers -- all in pure, differentiable
JAX. Requires 64-bit precision (``jax.config.update('jax_enable_x64', True)``).
"""

from ._apodize import apodize, apodize_profile, grassfire_distance
from ._estimate import anafast_masked
from ._mcm import MCM, bandpower_windows, compute_mcm

__all__ = [
    'apodize',
    'grassfire_distance',
    'apodize_profile',
    'MCM',
    'compute_mcm',
    'bandpower_windows',
    'anafast_masked',
]
