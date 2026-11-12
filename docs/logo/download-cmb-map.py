#!/usr/bin/env python
"""Download the Planck CMB map used as the texture of the jax-healpy logo.

The Planck PR3 NILC CMB map (IQU, nside 2048, about 1.6 GB) is downloaded from IRSA, and its
temperature component is written to ``COM_CMB_I-nilc_<nside>_R3.00_full.fits`` in this directory,
for each requested nside, which is the file read by ``make-logo.py``.

Usage::

    uv run python docs/logo/download-cmb-map.py [--nside 2048 1024 ...]
"""

import argparse
import urllib.request
from pathlib import Path

import healpy as hp

SCRIPT_DIR = Path(__file__).parent
PLANCK_URL = (
    'https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/'
    'COM_CMB_IQU-nilc_2048_R3.00_full.fits'
)


def download(url: str, path: Path) -> None:
    """Download a file, unless it is already present."""
    if path.exists():
        print(f'{path.name} already downloaded.')
        return
    print(f'Downloading {url}...')
    tmp_path = path.with_suffix(path.suffix + '.part')
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.rename(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Download the Planck CMB temperature map used by make-logo.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--nside', type=int, nargs='+', default=[2048], help='nside of the written maps')
    args = parser.parse_args()

    iqu_path = SCRIPT_DIR / PLANCK_URL.rsplit('/', 1)[-1]
    download(PLANCK_URL, iqu_path)
    map_2048 = hp.read_map(iqu_path, field=['I_STOKES'])

    for nside in args.nside:
        path = SCRIPT_DIR / f'COM_CMB_I-nilc_{nside}_R3.00_full.fits'
        print(f'Writing {path.name}...')
        hp.write_map(path, hp.ud_grade(map_2048, nside), overwrite=True)


if __name__ == '__main__':
    main()
