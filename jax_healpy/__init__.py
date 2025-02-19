from jax import config as _config

from .pixelfunc import (
    UNSEEN,
    ang2pix,
    ang2vec,
    isnpixok,
    isnsideok,
    maptype,
    nest2ring,
    npix2nside,
    npix2order,
    nside2npix,
    nside2order,
    nside2pixarea,
    nside2resol,
    order2npix,
    order2nside,
    pix2ang,
    pix2vec,
    pix2xyf,
    reorder,
    ring2nest,
    vec2ang,
    vec2pix,
    xyf2pix,
)
from .sphtfunc import alm2map, map2alm

__all__ = [
    'UNSEEN',
    'pix2ang',
    'ang2pix',
    'pix2xyf',
    'xyf2pix',
    'pix2vec',
    'vec2pix',
    'ang2vec',
    'vec2ang',
    # 'get_interp_weights',
    # 'get_interp_val',
    # 'get_all_neighbours',
    # 'max_pixrad',
    'nest2ring',
    'ring2nest',
    'reorder',
    # 'ud_grade',
    # 'UNSEEN',
    # 'mask_good',
    # 'mask_bad',
    # 'ma',
    # 'fit_dipole',
    # 'remove_dipole',
    # 'fit_monopole',
    # 'remove_monopole',
    'nside2npix',
    'npix2nside',
    'nside2order',
    'order2nside',
    'order2npix',
    'npix2order',
    'nside2resol',
    'nside2pixarea',
    'isnsideok',
    'isnpixok',
    # 'get_map_size',
    # 'get_min_valid_nside',
    # 'get_nside',
    'maptype',
    # 'ma_to_array',
    'alm2map',
    'map2alm',
]

_config.update('jax_enable_x64', True)
