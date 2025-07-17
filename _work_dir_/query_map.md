# `query_disc_internal` with RING Scheme

This document explains the implementation of the `query_disc_internal` function in `healpix_base.cc` when using the `RING` pixel ordering scheme.

## C++ Code

```cpp
template<typename I> template<typename I2>
  void T_Healpix_Base<I>::query_disc_internal
  (pointing ptg, double radius, int fact, rangeset<I2> &pixset) const
  {
  bool inclusive = (fact!=0);
  pixset.clear();
  ptg.normalize();

  if (scheme_==RING)
    {
    I fct=1;
    if (inclusive)
      {
      planck_assert (((I(1)<<order_max)/nside_)>=fact,
        "invalid oversampling factor");
      fct = fact;
      }
    T_Healpix_Base b2;
    double rsmall, rbig;
    if (fct>1)
      {
      b2.SetNside(fct*nside_,RING);
      rsmall = radius+b2.max_pixrad();
      rbig = radius+max_pixrad();
      }
    else
      rsmall = rbig = inclusive ? radius+max_pixrad() : radius;

    if (rsmall>=pi)
      { pixset.append(0,npix_); return; }

    rbig = min(pi,rbig);

    double cosrsmall = cos(rsmall);
    double cosrbig = cos(rbig);

    double z0 = cos(ptg.theta);
    double xa = 1./sqrt((1-z0)*(1+z0));

    I cpix=zphi2pix(z0,ptg.phi);

    double rlat1 = ptg.theta - rsmall;
    double zmax = cos(rlat1);
    I irmin = ring_above (zmax)+1;

    if ((rlat1<=0) && (irmin>1)) // north pole in the disk
      {
      I sp,rp; bool dummy;
      get_ring_info_small(irmin-1,sp,rp,dummy);
      pixset.append(0,sp+rp);
      }

    if ((fct>1) && (rlat1>0)) irmin=max(I(1),irmin-1);

    double rlat2 = ptg.theta + rsmall;
    double zmin = cos(rlat2);
    I irmax = ring_above (zmin);

    if ((fct>1) && (rlat2<pi)) irmax=min(4*nside_-1,irmax+1);

    for (I iz=irmin; iz<=irmax; ++iz)
      {
      double z=ring2z(iz);
      double x = (cosrbig-z*z0)*xa;
      double ysq = 1-z*z-x*x;
      double dphi=-1;
      bool fullcircle = false;
      if (ysq<=0) // no intersection, ring completely inside or outside
        {
        if (fct==1)
          dphi = 0;
        else
          {
          fullcircle = true;
          dphi = pi-1e-15;
          }
        }
      else
        dphi = atan2(sqrt(ysq),x);
      if (dphi>0)
        {
        I nr, ipix1;
        bool shifted;
        get_ring_info_small(iz,ipix1,nr,shifted);
        double shift = shifted ? 0.5 : 0.;

        I ipix2 = ipix1 + nr - 1; // highest pixel number in the ring

        I ip_lo = ifloor<I>(nr*inv_twopi*(ptg.phi-dphi) - shift)+1;
        I ip_hi = ifloor<I>(nr*inv_twopi*(ptg.phi+dphi) - shift);
        if (fullcircle)  // make sure we test the entire ring
          {
          if (ip_hi-ip_lo<nr-1)
            {
            if (ip_lo>0)
              --ip_lo;
            else
              ++ip_hi;
            }
          }

        if (fct>1)
          {
          while ((ip_lo<=ip_hi) && check_pixel_ring
                (*this,b2,ip_lo,nr,ipix1,fct,z0,ptg.phi,cosrsmall,cpix))
            ++ip_lo;
          while ((ip_hi>ip_lo) && check_pixel_ring
                (*this,b2,ip_hi,nr,ipix1,fct,z0,ptg.phi,cosrsmall,cpix))
            --ip_hi;
          }

        if (ip_lo<=ip_hi)
          {
          if (ip_hi>=nr)
            { ip_lo-=nr; ip_hi-=nr; }
          if (ip_lo<0)
            {
            pixset.append(ipix1,ipix1+ip_hi+1);
            pixset.append(ipix1+ip_lo+nr,ipix2+1);
            }
          else
            pixset.append(ipix1+ip_lo,ipix1+ip_hi+1);
          }
        }
      }
    if ((rlat2>=pi) && (irmax+1<4*nside_)) // south pole in the disk
      {
      I sp,rp; bool dummy;
      get_ring_info_small(irmax+1,sp,rp,dummy);
      pixset.append(sp,npix_);
      }
    }
  else // scheme_==NEST
    {
      // ... NEST implementation ...
    }
}
```

## Explanation

### High-Level Overview

The `query_disc_internal` function with the `RING` scheme is designed to find all pixels that fall within a specified circular disc on the sphere. The core idea is to iterate through the rings of pixels that could potentially overlap with the disc. For each of these rings, it calculates the range of pixels (in longitude, or `phi`) that are inside the disc.

The process can be summarized as follows:

1.  **Determine the range of rings to check:** Based on the disc's center and radius, it calculates the minimum and maximum latitude (and thus the minimum and maximum ring numbers, `irmin` and `irmax`) that the disc can possibly intersect.
2.  **Iterate through the rings:** It loops through each ring from `irmin` to `irmax`.
3.  **Calculate the intersection in longitude:** For each ring, it determines the range of longitudes (`phi`) that the disc covers.
4.  **Convert longitude to pixel indices:** This longitude range is then converted into a range of pixel indices within the current ring.
5.  **Add pixels to the result:** The identified pixel ranges are added to the `pixset`.

### Detailed Code Explanation

Here's a more detailed look at the key parts of the `query_disc_internal` function for the `RING` scheme:

*   **`inclusive` and `fact`:**
    *   The `inclusive` parameter, when true, means the function should also find pixels that only partially overlap with the disc.
    *   The `fact` parameter is an oversampling factor. If `fact > 1`, a higher-resolution Healpix grid (`b2`) is created. This is used to check for overlaps with more precision at the edges of the disc.

*   **Radius Adjustment (`rsmall`, `rbig`):**
    *   The disc's radius is adjusted to account for the size of the pixels.
    *   `rsmall` is the disc radius plus the maximum pixel radius of the (potentially oversampled) grid. This is the main radius used for the query.
    *   `rbig` is the disc radius plus the maximum pixel radius of the base grid. This is used for a quick check to see if an entire ring is inside the disc.

*   **Determining the Ring Range (`irmin`, `irmax`):**
    *   The function calculates the minimum and maximum `z` coordinates covered by the disc (`zmin`, `zmax`).
    *   `ring_above(z)` is then used to convert these `z` values into ring indices, giving the range of rings to check.
    *   There are special checks to handle cases where the north or south pole is inside the disc, in which case all pixels in the polar cap are added.

*   **Main Loop (`for (I iz=irmin; iz<=irmax; ++iz)`):**
    *   This loop iterates through each potentially overlapping ring.

*   **Calculating `dphi`:**
    *   For each ring, it calculates `dphi`, which is the half-width of the longitude range covered by the disc at that ring's latitude.
    *   This is done by solving for the intersection of the disc's circle and the ring's circle of constant latitude.
    *   If `ysq <= 0`, it means the ring is either completely inside or completely outside the disc.
maitre gims
*   **Finding Pixel Range (`ip_lo`, `ip_hi`):**
    *   The `dphi` value is used to calculate the starting (`ip_lo`) and ending (`ip_hi`) pixel indices within the current ring. This is based on the disc's center longitude (`ptg.phi`).

*   **Inclusive Check (`if (fct>1)`):**
    *   If `inclusive` mode is on, the `check_pixel_ring` function is called for the pixels at the edges of the calculated range (`ip_lo` and `ip_hi`).
    *   `check_pixel_ring` performs a more detailed check by examining the corners of the higher-resolution sub-pixels to see if any of them are inside the disc. This ensures that pixels that only partially overlap are included.

*   **Adding Pixels to `pixset`:**
    *   Finally, the calculated range of pixels (`ip_lo` to `ip_hi`) is added to the `pixset`.
    *   The code correctly handles the case where the pixel range wraps around the ring (e.g., from pixel `nr-1` to pixel `0`).
