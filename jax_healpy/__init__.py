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

from jax import config as _config

from ._query_disc import estimate_disc_pixel_count, estimate_disc_radius, query_disc
from .clustering._clustering import (
    combine_masks,
    from_cutout_to_fullmap,
    get_clusters,
    get_cutout_from_mask,
    normalize_by_first_occurrence,
    shuffle_labels,
)
from .clustering._kmeans import KMeans, kmeans_sample
from .pixelfunc import (
    UNSEEN,
    ang2pix,
    ang2vec,
    get_all_neighbours,
    get_interp_val,
    get_interp_weights,
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
    ud_grade,
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
    'get_interp_weights',
    'get_interp_val',
    'get_all_neighbours',
    # 'max_pixrad',
    'nest2ring',
    'ring2nest',
    'reorder',
    'ud_grade',
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
    'query_disc',
    'estimate_disc_pixel_count',
    'estimate_disc_radius',
    'alm2map',
    'map2alm',
    # Clustering
    'get_clusters',
    'get_cutout_from_mask',
    'from_cutout_to_fullmap',
    'kmeans_sample',
    'KMeans',
    'combine_masks',
    'normalize_by_first_occurrence',
    'shuffle_labels',
]

_config.update('jax_enable_x64', True)
