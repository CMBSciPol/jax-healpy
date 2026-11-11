# Disc Queries

Queries of the pixels within a circular region on the sphere, for point source analysis,
aperture photometry or local statistics.

## Mathematical Background

`query_disc` returns the pixels whose centers satisfy

\[
\cos(\text{radius}) \leq \vec{v}_{\text{center}} \cdot \vec{v}_{\text{pixel}},
\]

where \(\vec{v}_{\text{center}}\) is the unit vector of the disc center and
\(\vec{v}_{\text{pixel}}\) that of the pixel center.

For inclusive queries, the pixels that are only partially within the disc are also returned: the
pixel boundaries are subdivided and a pixel is kept if any of the subdivision points falls within
the disc.

To be compatible with `jax.jit`, the output has a fixed size. The slots that are not filled by a
disc pixel contain the sentinel value `npix`, so that the output can be used directly to index a
map, as in `m.at[pixels].set(value)`.

## Accuracy

- Standard queries (`inclusive=False`) are exact for the pixel centers.
- Inclusive queries are more accurate at the disc boundary, at a computational cost which
  increases with `fact`.

## Functions

::: jax_healpy._query_disc.query_disc
::: jax_healpy._query_disc.estimate_disc_pixel_count
::: jax_healpy._query_disc.estimate_disc_radius
