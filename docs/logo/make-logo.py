#!/usr/bin/env python
"""Generate the jax-healpy logo.

The text "jax-healpy" is rendered on a plate carree image, projected onto a low-resolution
HEALPix map to give it a pixelated look, and filled with the Planck CMB temperature map. The
result is displayed in Mollweide projection with a colormap inspired by the JAX logo colors.

The CMB map is downloaded by ``download-cmb-map.py``. With the default parameters, the logo
``jax-healpy-logo.png`` is reproduced up to a few pixels on the edges of the letters, which
depend on the versions of the rendering libraries. The version of the logo for the dark theme
is then derived from it by ``make-dark-logo.py``.

Usage::

    uv run python docs/logo/download-cmb-map.py
    uv run --with scipy python docs/logo/make-logo.py --no-show
    uv run python docs/logo/make-dark-logo.py
"""

import argparse
import io
from pathlib import Path

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_rgb
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, label, zoom

import jax_healpy as jhp

jax.config.update('jax_enable_x64', True)

SCRIPT_DIR = Path(__file__).parent

# Colors inspired by the JAX logo, from negative to positive temperatures.
JAX_COLORS = ['#2a56c6', '#5e97f6', '#ffffff', '#9c27b0', '#ea80fc']

# Temperature range of the colormap, in K.
CMB_RANGE = 300e-6

# Color used to detect the outside of the Mollweide ellipse, which is made transparent.
CHROMA = '#ff00ff'

# Color of the outlines of the letters and of the Mollweide ellipse.
OUTLINE_COLOR = [60, 60, 60, 255]


def create_text_mask_plate_carree(
    *,
    text: str,
    width: int,
    height: int,
    fontsize: int,
    stroke_width: int,
    height_scale: float,
) -> np.ndarray:
    """Return a binary mask with the text rendered on a plate carree image.

    Parameters
    ----------
    text : str
        The text to render.
    width : int
        Width of the image in pixels (longitude direction).
    height : int
        Height of the image in pixels (latitude direction).
    fontsize : int
        Font size in points.
    stroke_width : int
        Width of the text stroke in pixels.
    height_scale : float
        Scale factor for the text height (1.0 = normal, 1.2 = 20% taller).

    Returns
    -------
    np.ndarray
        Mask of shape (height, width), equal to 1 where the text is present.
    """
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)
    # The DejaVu Sans font is shipped with matplotlib, which makes the rendering reproducible.
    font_file = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans'))
    font = ImageFont.truetype(font_file, fontsize)

    # Center the text in the image
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    x = (width - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (height - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), text, fill=0, font=font, stroke_width=stroke_width, stroke_fill=0)
    mask = (np.array(img) < 128).astype(np.float32)

    if height_scale == 1.0:
        return mask

    # Stretch the text vertically, then crop or pad to the original height, centered
    mask = zoom(mask, (height_scale, 1.0), order=1)
    new_height = mask.shape[0]
    if new_height > height:
        start = (new_height - height) // 2
        mask = mask[start : start + height, :]
    elif new_height < height:
        pad_top = (height - new_height) // 2
        mask = np.pad(mask, ((pad_top, height - new_height - pad_top), (0, 0)))
    return mask


def plate_carree_to_healpix(
    plate_carree_map: np.ndarray,
    nside: int,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> np.ndarray:
    """Sample a plate carree image at the centers of the HEALPix pixels.

    Parameters
    ----------
    plate_carree_map : np.ndarray
        Image of shape (n_lat, n_lon) in plate carree projection.
    nside : int
        HEALPix nside of the output map.
    lon_range : tuple of float
        Longitude range of the image in degrees, in [-180, 180].
    lat_range : tuple of float
        Latitude range of the image in degrees.

    Returns
    -------
    np.ndarray
        HEALPix map in RING ordering.
    """
    theta, phi = jhp.pix2ang(nside, jnp.arange(jhp.nside2npix(nside)))
    lat = jnp.degrees(jnp.pi / 2 - theta)
    lon = jnp.degrees(phi)
    lon = jnp.where(lon > 180, lon - 360, lon)

    # The longitude is flipped, since it increases leftward in the Mollweide view.
    n_lat, n_lon = plate_carree_map.shape
    lon_norm = (lon_range[1] - lon) / (lon_range[1] - lon_range[0])
    lat_norm = (lat - lat_range[0]) / (lat_range[1] - lat_range[0])
    i_lon = jnp.clip((lon_norm * n_lon).astype(jnp.int32), 0, n_lon - 1)
    i_lat = jnp.clip(((1 - lat_norm) * n_lat).astype(jnp.int32), 0, n_lat - 1)
    return np.asarray(jnp.asarray(plate_carree_map)[i_lat, i_lon])


def create_text_mask_healpix(
    *,
    text: str,
    nside: int,
    nside_mask: int,
    fontsize: int,
    stroke_width: int,
    height_scale: float,
    lon_center: float,
    lat_center: float,
    lon_span: float,
    lat_span: float,
) -> np.ndarray:
    """Return the text mask as a HEALPix map, pixelated at `nside_mask`."""
    pc_width = 2048
    pc_height = int(pc_width * lat_span / lon_span)
    print(f'Creating the text mask ({pc_width}x{pc_height} pixels)...')
    text_mask_pc = create_text_mask_plate_carree(
        text=text,
        width=pc_width,
        height=pc_height,
        fontsize=fontsize,
        stroke_width=stroke_width,
        height_scale=height_scale,
    )

    lon_min, lon_max = lon_center - lon_span / 2, lon_center + lon_span / 2
    lat_min, lat_max = lat_center - lat_span / 2, lat_center + lat_span / 2
    print(f'Projecting the text mask onto HEALPix (nside_mask={nside_mask})...')
    text_mask_hp = plate_carree_to_healpix(
        text_mask_pc, nside=nside_mask, lon_range=(lon_min, lon_max), lat_range=(lat_min, lat_max)
    )

    # Only keep the text inside its region
    theta, phi = jhp.pix2ang(nside_mask, jnp.arange(jhp.nside2npix(nside_mask)))
    lat = np.asarray(jnp.degrees(jnp.pi / 2 - theta))
    lon = np.asarray(jnp.degrees(phi))
    lon = np.where(lon > 180, lon - 360, lon)
    in_region = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    mask = np.where(in_region, text_mask_hp, 0).astype(np.float32)

    # Upgrade the mask to nside, keeping the pixelated look
    print(f'Upgrading the mask from nside={nside_mask} to nside={nside}...')
    mask = np.asarray(jhp.ud_grade(jnp.asarray(mask), nside))
    return mask > 0.5


def get_colormap(name: str) -> Colormap:
    """Return the colormap, `jax` being the colormap inspired by the JAX logo."""
    if name != 'jax':
        return plt.get_cmap(name)
    nodes = np.linspace(0, 1, len(JAX_COLORS))
    rgb = [to_rgb(color) for color in JAX_COLORS]
    cmap_data = {
        channel: [(node, color[index], color[index]) for node, color in zip(nodes, rgb)]
        for index, channel in enumerate(['red', 'green', 'blue'])
    }
    return LinearSegmentedColormap('jax', cmap_data, N=256)


def render_logo(masked_cmb: np.ndarray, cmap: Colormap, dpi: int) -> np.ndarray:
    """Render the masked CMB map in Mollweide projection and return the RGBA image.

    The outside of the Mollweide ellipse is made transparent, and the letters and the ellipse
    are outlined.
    """
    hp.mollview(
        masked_cmb,
        cmap=cmap,
        min=-CMB_RANGE,
        max=CMB_RANGE,
        title='',
        cbar=False,
        xsize=2000,
        badcolor='white',
        bgcolor=CHROMA,
    )
    plt.gca().set_facecolor(CHROMA)
    buffer = io.BytesIO()
    plt.savefig(buffer, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor=CHROMA)
    plt.close()
    buffer.seek(0)
    data = np.array(Image.open(buffer).convert('RGBA'))

    # The white pixels are the UNSEEN pixels, around the letters, inside the ellipse.
    bg_tol = 8
    is_unseen = np.all(data[:, :, :3].astype(int) > 255 - bg_tol, axis=-1)

    # Flood-fill from the corners through the non-UNSEEN pixels to find the outside of the
    # ellipse: the UNSEEN pixels act as a barrier.
    labeled, _ = label(~is_unseen)
    corner_labels = {labeled[0, 0], labeled[0, -1], labeled[-1, 0], labeled[-1, -1]} - {0}
    outside = np.isin(labeled, list(corner_labels))
    data[outside, 3] = 0

    # Outline the letters
    is_opaque = data[:, :, 3] > 0
    is_unseen_opaque = is_unseen & is_opaque
    is_cmb = is_opaque & ~is_unseen_opaque
    data[binary_dilation(is_cmb, iterations=3) & is_unseen_opaque] = OUTLINE_COLOR

    # Outline the ellipse
    is_opaque = data[:, :, 3] > 0
    data[binary_dilation(is_opaque, iterations=3) & ~is_opaque] = OUTLINE_COLOR
    return data


def make_logo(
    *,
    nside: int = 2048,
    nside_mask: int = 32,
    fontsize: int = 365,
    stroke_width: int = 10,
    height_scale: float = 1.5,
    text: str = 'jax-healpy',
    cmb_file: str | Path | None = None,
    output_file: str | Path | None = None,
    mask_file: str | Path | None = None,
    lon_center: float = 0.0,
    lat_center: float = 0.0,
    lon_span: float = 360.0,
    lat_span: float = 120.0,
    cmap: str = 'jax',
    show: bool = True,
    dpi: int = 300,
) -> np.ndarray:
    """Generate the jax-healpy logo.

    Parameters
    ----------
    nside : int
        HEALPix nside of the CMB map.
    nside_mask : int
        HEALPix nside of the text mask (lower = more pixelated text).
    fontsize : int
        Font size of the text.
    stroke_width : int
        Stroke width of the text.
    height_scale : float
        Scale factor for the text height (1.0 = normal, 1.2 = 20% taller).
    text : str
        Text of the logo.
    cmb_file : str or Path, optional
        Path to the CMB FITS file. Default: the file written by ``download-cmb-map.py``.
    output_file : str or Path, optional
        Path to the logo. Default: ``jax-healpy-logo.png`` in this directory.
    mask_file : str or Path, optional
        If provided, path to an image of the text mask in Mollweide projection.
    lon_center : float
        Center longitude of the text in degrees.
    lat_center : float
        Center latitude of the text in degrees.
    lon_span : float
        Longitude span of the text in degrees.
    lat_span : float
        Latitude span of the text in degrees.
    cmap : str
        Colormap: `jax` for the colormap inspired by the JAX logo, or a matplotlib colormap.
    show : bool
        Whether to display the logo.
    dpi : int
        DPI of the output image.

    Returns
    -------
    np.ndarray
        The CMB map, UNSEEN outside the text.
    """
    if cmb_file is None:
        cmb_file = SCRIPT_DIR / f'COM_CMB_I-nilc_{nside}_R3.00_full.fits'
    if output_file is None:
        output_file = SCRIPT_DIR / 'jax-healpy-logo.png'

    mask = create_text_mask_healpix(
        text=text,
        nside=nside,
        nside_mask=nside_mask,
        fontsize=fontsize,
        stroke_width=stroke_width,
        height_scale=height_scale,
        lon_center=lon_center,
        lat_center=lat_center,
        lon_span=lon_span,
        lat_span=lat_span,
    )

    print(f'Loading the CMB map from {cmb_file}...')
    cmb = hp.read_map(cmb_file)
    if hp.get_nside(cmb) != nside:
        print(f'Resampling the CMB map from nside={hp.get_nside(cmb)} to nside={nside}...')
        cmb = np.asarray(jhp.ud_grade(jnp.asarray(cmb), nside))
    masked_cmb = np.where(mask, cmb, hp.UNSEEN)

    print(f'Saving the logo to {output_file}...')
    data = render_logo(masked_cmb, get_colormap(cmap), dpi)
    Image.fromarray(data).save(output_file)

    if mask_file is not None:
        print(f'Saving the mask to {mask_file}...')
        hp.mollview(mask.astype(np.float32), cmap='gray_r', title='', xsize=2000)
        plt.savefig(mask_file, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()

    if show:
        plt.imshow(data)
        plt.axis('off')
        plt.show()

    return masked_cmb


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate the jax-healpy logo', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--nside', type=int, default=2048, help='HEALPix nside of the CMB map')
    parser.add_argument(
        '--nside-mask', type=int, default=32, help='HEALPix nside of the text mask (lower = more pixelated)'
    )
    parser.add_argument('--fontsize', type=int, default=365, help='Font size in points')
    parser.add_argument('--stroke-width', type=int, default=10, help='Stroke width of the text')
    parser.add_argument('--height-scale', type=float, default=1.5, help='Scale factor for the text height')
    parser.add_argument('--text', default='jax-healpy', help='Text of the logo')
    parser.add_argument('--cmb-file', help='Path to the CMB FITS file (default: based on nside)')
    parser.add_argument('--output', help='Path to the logo (default: jax-healpy-logo.png in this directory)')
    parser.add_argument('--mask-output', help='Path to an image of the text mask')
    parser.add_argument('--lon-center', type=float, default=0.0, help='Center longitude of the text (degrees)')
    parser.add_argument('--lat-center', type=float, default=0.0, help='Center latitude of the text (degrees)')
    parser.add_argument('--lon-span', type=float, default=360.0, help='Longitude span of the text (degrees)')
    parser.add_argument('--lat-span', type=float, default=120.0, help='Latitude span of the text (degrees)')
    parser.add_argument('--cmap', default='jax', help='Colormap')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    parser.add_argument('--no-show', action='store_true', help='Do not display the logo')
    args = parser.parse_args()

    make_logo(
        nside=args.nside,
        nside_mask=args.nside_mask,
        fontsize=args.fontsize,
        stroke_width=args.stroke_width,
        height_scale=args.height_scale,
        text=args.text,
        cmb_file=args.cmb_file,
        output_file=args.output,
        mask_file=args.mask_output,
        lon_center=args.lon_center,
        lat_center=args.lat_center,
        lon_span=args.lon_span,
        lat_span=args.lat_span,
        cmap=args.cmap,
        show=not args.no_show,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
