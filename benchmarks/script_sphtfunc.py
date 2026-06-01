import timeit
from collections.abc import Callable
from pathlib import Path

import healpy as hp
import jax
import numpy as np

import jax_healpy as jhp

data_path = Path(jhp.__file__).parent.parent / 'tests/data'


def cla(data_path: Path) -> np.ndarray:
    return hp.read_cl(data_path / 'cl_wmap_band_iqumap_r9_7yr_W_v4_udgraded32_II_lmax64_rmmono_3iter.fits')


nside = 256
lmax = 2 * nside
fwhm_deg = 7.0


orig = hp.synfast(
    cla(data_path),
    nside,
    lmax=lmax,
    pixwin=False,
    fwhm=np.radians(fwhm_deg),
    new=False,
)
orig_d = jax.device_put(orig)


def map2alm_hp():
    hp.map2alm(orig, iter=0, lmax=2 * nside - 1, use_weights=False)


def map2alm_jhp():
    jhp.map2alm(orig_d).block_until_ready()


def time_it(func: Callable[[], None]) -> float:
    timer = timeit.Timer(func)
    number, _ = timer.autorange()
    execution_time = min(timer.repeat(number=number)) / number
    return execution_time


print(time_it(map2alm_hp))
# 0.0038599122400046326

print(time_it(map2alm_jhp))
# 0.437624677999338
