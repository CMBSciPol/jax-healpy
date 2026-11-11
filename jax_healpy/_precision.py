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

"""Warning about JAX's 32-bit precision, emitted when jax-healpy is used, not imported."""

import warnings

import jax

__all__ = ['warn_if_not_x64']

_MESSAGE = (
    'JAX is not using 64-bit precision. Results will diverge from healpy even at moderate '
    'nside and can overflow for nside > 8192. Enable it with '
    "jax.config.update('jax_enable_x64', True). See the README on numerical precision."
)

_warned = False


def warn_if_not_x64() -> None:
    """Warn, once per session, if JAX does not use 64-bit precision.

    The check runs when a function is called (or traced), not when jax-healpy is imported,
    so that enabling 64-bit precision after the import does not trigger the warning.
    """
    global _warned
    if _warned or jax.config.read('jax_enable_x64'):
        return
    _warned = True
    warnings.warn(_MESSAGE, UserWarning, stacklevel=3)
