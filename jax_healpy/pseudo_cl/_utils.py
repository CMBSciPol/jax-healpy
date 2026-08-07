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

from __future__ import annotations

import jax


def require_x64() -> None:
    """Raise if JAX 64-bit precision is disabled.

    The spin-2 mode-decoupling solve is ill-conditioned; in 32-bit precision it silently returns
    all-NaN rather than raising, so the masked-Cl entry points guard loudly. Enable 64-bit before
    use, e.g. ``jax.config.update('jax_enable_x64', True)`` or the ``JAX_ENABLE_X64=1`` env var.
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError(
            'jax_healpy.pseudo_cl requires 64-bit precision: the spin-2 decoupling solve is '
            'ill-conditioned and silently returns all-NaN in float32. Enable it before use with '
            "jax.config.update('jax_enable_x64', True) or set JAX_ENABLE_X64=1."
        )
